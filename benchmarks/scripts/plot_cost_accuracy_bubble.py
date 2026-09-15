#!/usr/bin/env python3
"""Cost vs. accuracy bubble chart for one (harness, skill).

Three dimensions in one scatter:
  * x = cost per task (USD)
  * y = mean task score (0-100)
  * bubble AREA = total tokens processed (area-proportional, so a 2x-bigger
    bubble means 2x the tokens -- radius scales with sqrt(tokens))

Bubbles are colored by hosting basis (Bedrock metered vs self-hosted
hardware-derived), because those dollars are not comparable as raw numbers; the
two hues pass the dataviz colorblind validator in light and dark, and every
bubble is labelled with the model name so identity never rests on color alone.

Cost is sourced from gen_agent_report (_collect + _row_cost), so this chart and
the harness docs agree. One chart per harness (pass --harness).

Usage:
    uv run scripts/plot_cost_accuracy_bubble.py --harness pi --skill swe3
    uv run scripts/plot_cost_accuracy_bubble.py --harness claude-code --skill swe3 --dark

Output: docs/images/cost-accuracy-bubble-<code>-<skill>{,-dark}.png
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
from pathlib import Path
from typing import Any

from token_accounting import cache_partition_for_agent, compute_total_tokens_processed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
DEFAULT_DATA_DIR = _SCRIPTS_DIR.parent / "swe-benchmark-data"
DEFAULT_OUT_DIR = _REPO_ROOT / "docs" / "images"

_GEN_PATH = _SCRIPTS_DIR / "gen_agent_report.py"
_spec = importlib.util.spec_from_file_location("gen_agent_report", _GEN_PATH)
assert _spec is not None and _spec.loader is not None  # nosec B101 - import-by-path guard, not runtime validation
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

HARNESS_CODES = {"claude-code": "cc", "pi": "pi", "opencode": "oc", "kiro-cli": "kiro"}
HARNESS_LABELS = {"claude-code": "Claude Code", "pi": "pi"}

# Palette from the dataviz skill's validated reference instance. Bubbles are
# colored by hosting basis (the two accents are colorblind-checked in both modes);
# text wears ink tokens.
_THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e2",
        "metered": "#eb6834",  # Bedrock (warm accent)
        "hardware": "#3d7dca",  # self-hosted (validated cool accent)
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#333330",
        "metered": "#d95926",
        "hardware": "#4a90d9",
    },
}

# Bubble-area range (points^2): smallest and largest token counts map to these,
# area-proportional in between so the eye reads token magnitude honestly.
_AREA_MIN = 120.0
_AREA_MAX = 2600.0


def _human_tokens(value: float) -> str:
    """Compact token count (e.g. 82.7M)."""
    if value >= 1e9:
        return f"{value / 1e9:.1f}B"
    if value >= 1e6:
        return f"{value / 1e6:.0f}M"
    if value >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.0f}"


def _task_shape(data_dir: Path, harness: str, skill: str, repo: str) -> str | None:
    """Return a human 'N in : M out (~R:1)' string describing an average task.

    A "task" is one dataset problem. We summarize its scale as the median across
    all this (harness, skill) run's per-task token counts: the read-heavy input
    (prompt) side vs the output side. This tells the reader what "cost per task"
    is priced over.

    The input side is the prompt tokens PROCESSED once each, from
    ``compute_total_tokens_processed`` (with output=0 it returns just the prompt
    part). That collapses the cache into input on self-hosted partition runs
    (where cache_read/cache_write already live inside input_tokens) and keeps it
    additive on Bedrock -- so this ratio no longer ~2x double-counts the prompt
    on self-hosted runs (issue #136).
    """
    ins: list[int] = []
    outs: list[int] = []
    for model_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        summ = gen._read_json(
            model_dir / harness / skill / repo / gen.RUN_SUMMARY_FILENAME
        )
        if not summ:
            continue
        for task in summ.get("tasks", []):
            if task.get("failed"):
                continue
            i = task.get("input_tokens") or 0
            cr = task.get("cache_read_tokens") or 0
            cw = (
                task.get("cache_write_tokens") or task.get("cache_creation_tokens") or 0
            )
            o = task.get("output_tokens") or 0
            prompt_processed = compute_total_tokens_processed(
                i,
                0,
                cr,
                cw,
                context=f"plot_cost_accuracy_bubble:{summ.get('model_slug')}/{task.get('task')}",
                cache_partition=cache_partition_for_agent(summ.get("agent")),
            )
            if prompt_processed > 0 and o > 0:
                ins.append(prompt_processed)
                outs.append(o)
    if not ins:
        return None
    med_in = sorted(ins)[len(ins) // 2]
    med_out = sorted(outs)[len(outs) // 2]
    ratio = round(med_in / max(med_out, 1))
    return (
        f"{_human_tokens(med_in)} input : {_human_tokens(med_out)} output (~{ratio}:1)"
    )


def _collect_points(
    data_dir: Path, harness: str, skill: str, repo: str
) -> list[dict[str, Any]]:
    """Return per-model points (model, cost_per_task, score, tokens, bedrock).

    cost is per task (run total / scored tasks), from gen_agent_report so it
    matches the docs. A model with no derivable cost or no scored tasks is
    dropped with a logged note rather than placed at a fake origin.
    """
    pts: list[dict[str, Any]] = []
    for row in gen._collect(data_dir, harness, skill, repo):
        cost_str, basis = gen._row_cost(row)
        scored = row.get("num_scored") or 0
        total = row.get("total_tokens") or 0
        if cost_str == "--" or not scored or row.get("mean") is None or not total:
            logger.info(
                "skipping %s: no cost / no scored tasks / no tokens", row["model"]
            )
            continue
        pts.append(
            {
                "model": row["model"],
                "cost": float(cost_str.lstrip("$")) / scored,
                "score": float(row["mean"]),
                "tokens": total,
                "bedrock": basis.startswith("metered"),
            }
        )
    return pts


def _areas(tokens: list[int]) -> list[float]:
    """Map token counts to area-proportional bubble sizes (points^2).

    Linear in token count between _AREA_MIN and _AREA_MAX so bubble AREA (not
    radius) encodes magnitude -- the honest encoding for a quantity.
    """
    lo, hi = min(tokens), max(tokens)
    if hi == lo:
        return [(_AREA_MIN + _AREA_MAX) / 2 for _ in tokens]
    span = hi - lo
    return [_AREA_MIN + (v - lo) / span * (_AREA_MAX - _AREA_MIN) for v in tokens]


def _place_labels(
    fig: "plt.Figure",
    ax: "plt.Axes",
    points: list[dict[str, Any]],
    areas: list[float],
    t: dict[str, str],
) -> None:
    """Label each bubble, nudging overlaps apart and adding a leader arrow.

    Each label starts just outside its bubble (offset by the bubble radius). We
    then measure the labels' pixel bounding boxes and, for any pair that overlaps,
    push the higher one up until it clears -- iteratively, a few passes. A label
    that ends up moved from its natural spot gets a thin leader line back to its
    bubble so the association is unambiguous (the fix for colliding names like
    deepseek-v3.2 / nemotron-ultra-550b sitting at nearly the same point).
    """
    fig.canvas.draw()  # a renderer is needed so the transforms are valid
    anchors = [ax.transData.transform((p["cost"], p["score"])) for p in points]

    # Order by anchor y (top first) and assign labels; track occupied y-bands to
    # push overlaps upward. Boxes are in display pixels.
    line_h = 12.0  # approx label height in px at fontsize 7.2 + padding
    order = sorted(range(len(points)), key=lambda i: -anchors[i][1])
    placed_boxes: list[tuple[float, float, float, float]] = []
    for i in order:
        p = points[i]
        area = areas[i]
        ax_px, ay_px = anchors[i]
        radius = math.sqrt(area / math.pi)
        # natural label position: right of the bubble, roughly centered.
        lx = ax_px + radius + 4
        ly = ay_px
        # estimate width from character count (measured extents are overkill here).
        w = 6.6 * len(p["model"])
        # push up until this box clears all previously placed boxes.
        moved = False
        for _ in range(60):
            box = (lx, ly - line_h / 2, lx + w, ly + line_h / 2)
            clash = any(
                not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                for b in placed_boxes
            )
            if not clash:
                break
            ly += line_h
            moved = True
        placed_boxes.append((lx, ly - line_h / 2, lx + w, ly + line_h / 2))
        # convert the (possibly nudged) pixel position back to data coords.
        lx_data, ly_data = ax.transData.inverted().transform((lx, ly))
        arrow = (
            {"arrowstyle": "-", "color": t["muted"], "linewidth": 0.6, "shrinkA": 0}
            if moved
            else None
        )
        ax.annotate(
            p["model"],
            xy=(p["cost"], p["score"]),
            xytext=(lx_data, ly_data),
            textcoords="data",
            fontsize=7.2,
            color=t["ink"],
            va="center",
            zorder=6,
            arrowprops=arrow,
        )


def _plot(
    points: list[dict[str, Any]],
    *,
    harness: str,
    skill: str,
    mode: str,
    out_dir: Path,
    task_shape: str | None = None,
) -> Path:
    """Render the cost-vs-accuracy bubble chart (bubble area = tokens)."""
    t = _THEME[mode]
    label = HARNESS_LABELS.get(harness, harness)
    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9)
    ax.grid(True, color=t["grid"], linewidth=0.6)
    ax.set_axisbelow(True)

    areas = _areas([p["tokens"] for p in points])
    for p, area in zip(points, areas):
        ax.scatter(
            p["cost"],
            p["score"],
            s=area,
            color=t["metered"] if p["bedrock"] else t["hardware"],
            alpha=0.55,
            edgecolors=t["surface"],
            linewidths=1.0,
            zorder=3,
        )
    _place_labels(fig, ax, points, areas, t)

    ax.set_xlabel("Cost per task (USD)", fontsize=10, color=t["ink"])
    ax.set_ylabel("Mean task score (0-100)", fontsize=10, color=t["ink"])
    ax.set_title(
        f"{label} - {skill}: cost vs. accuracy (bubble area = tokens processed)",
        fontsize=13,
        color=t["ink"],
        loc="left",
    )

    # Hosting-basis color legend.
    color_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=9,
            color=t["metered"],
            label="metered (Bedrock)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=9,
            color=t["hardware"],
            label="hardware-derived (self-hosted)",
        ),
    ]
    leg1 = ax.legend(
        handles=color_handles,
        loc="lower right",
        fontsize=8,
        frameon=False,
        labelcolor=t["muted"],
        title="hosting",
        title_fontsize=8,
    )
    leg1.get_title().set_color(t["muted"])
    ax.add_artist(leg1)

    # Bubble-size reference legend (min / median / max tokens as sized dots).
    toks = sorted(p["tokens"] for p in points)
    ref = [toks[0], toks[len(toks) // 2], toks[-1]]
    ref_areas = _areas([toks[0], toks[len(toks) // 2], toks[-1], *toks])[:3]
    size_handles = [
        plt.scatter([], [], s=a, color=t["muted"], alpha=0.45, edgecolors=t["surface"])
        for a in ref_areas
    ]
    ax.legend(
        handles=size_handles,
        labels=[_human_tokens(v) for v in ref],
        loc="upper left",
        fontsize=8,
        frameon=False,
        labelcolor=t["muted"],
        title="tokens processed",
        title_fontsize=8,
        labelspacing=1.6,
        borderpad=1.0,
        handletextpad=1.4,
    )

    task_line = (
        f"A task = one real {skill} problem on this repo (5 tasks per run); the "
        f"median task processes ~{task_shape} tokens."
        if task_shape
        else ""
    )
    method_line = (
        "Bubble AREA is proportional to total tokens processed. Cost bases are not "
        "comparable as raw dollars: metered = real Bedrock bill; hardware-derived = "
        "blended $/token (throughput sweep, real instance) x tokens processed."
    )
    fig.text(0.01, 0.028, task_line, fontsize=6.8, color=t["muted"])
    fig.text(0.01, 0.006, method_line, fontsize=6.8, color=t["muted"])
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    code = HARNESS_CODES.get(harness, harness)
    suffix = "-dark" if mode == "dark" else ""
    out = out_dir / f"cost-accuracy-bubble-{code}-{skill}{suffix}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    logger.info("wrote %s (%d models)", out, len(points))
    return out


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cost vs. accuracy bubble chart (bubble area = tokens processed).",
        epilog="Example: uv run scripts/plot_cost_accuracy_bubble.py --harness pi --skill swe3",
    )
    parser.add_argument(
        "--harness", default="claude-code", help="Harness slug (default: claude-code)."
    )
    parser.add_argument(
        "--skill", default="swe3", help="SWE skill: 'swe3' (default) or 'swe2'."
    )
    parser.add_argument("--repo", default="mcp-gateway-registry", help="Dataset scope.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--dark", action="store_true", help="Render the dark-theme variant."
    )
    return parser.parse_args()


def main() -> None:
    """Collect per-model points and render the cost-vs-accuracy bubble chart."""
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    points = _collect_points(data_dir, args.harness, args.skill, args.repo)
    if not points:
        raise SystemExit(
            f"no costable models under "
            f"{data_dir}/*/{args.harness}/{args.skill}/{args.repo}"
        )
    _plot(
        points,
        harness=args.harness,
        skill=args.skill,
        mode="dark" if args.dark else "light",
        out_dir=args.out_dir.expanduser().resolve(),
        task_shape=_task_shape(data_dir, args.harness, args.skill, args.repo),
    )


if __name__ == "__main__":
    main()
