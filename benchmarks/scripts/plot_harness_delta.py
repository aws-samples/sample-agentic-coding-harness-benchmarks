#!/usr/bin/env python3
"""Dumbbell (connected-dot) small-multiples: which harness wins, per model, per metric.

The question this answers: does the HARNESS make a difference, and is one harness
generally better on cost / accuracy / tokens / latency across most models? A
grouped bar chart buries that in 24 bars per panel; a dumbbell shows it directly.

For each metric there is one panel. Each model is a row with two dots -- Claude
Code and pi -- joined by a line. The LINE is colored by which harness is BETTER
for that metric (accounting for direction: higher score is better, but lower
cost/tokens/latency is better), and the winning dot is drawn larger. So the eye
follows the connector: its direction is the winner, its length the magnitude of
the harness effect. Each panel title tallies "pi better on N of M" so the
prevalence -- the whole point -- is stated, not inferred.

Only models run under BOTH harnesses are shown (a comparison needs two dots).
Numbers come from gen_agent_report (_collect + _row_cost), matching the docs.

Usage:
    uv run scripts/plot_harness_delta.py --skill swe3
    uv run scripts/plot_harness_delta.py --skill swe3 --dark

Output: docs/images/harness-delta-<skill>{,-dark}.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
DEFAULT_DATA_DIR = _SCRIPTS_DIR.parent / "swe-benchmark-data"
DEFAULT_OUT_DIR = _REPO_ROOT / "docs" / "images"
# Machine-readable chart data lives apart from the rendered images.
DEFAULT_METRICS_DIR = _REPO_ROOT / "docs" / "metrics"

_GEN_PATH = _SCRIPTS_DIR / "gen_agent_report.py"
_spec = importlib.util.spec_from_file_location("gen_agent_report", _GEN_PATH)
assert _spec is not None and _spec.loader is not None  # nosec B101 - import-by-path guard, not runtime validation
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

HARNESSES = ("claude-code", "pi")
HARNESS_LABELS = {"claude-code": "Claude Code", "pi": "pi"}

# Chart font sizes (points). Sized up for legibility when the chart is embedded in
# slides and social posts. The two bottom notes (task shape, method) stay at
# FOOTNOTE_FONTSIZE so they read as fine print, not body text.
SUPTITLE_FONTSIZE = 16
PANEL_TITLE_FONTSIZE = 13
ROW_LABEL_FONTSIZE = 11
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 11
FOOTNOTE_FONTSIZE = 6.8

# Palette. Two validated categorical hues identify the two harness DOTS; the
# connecting line is colored by the winner so "who's better" reads at a glance.
_THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e2",
        "claude-code": "#3d7dca",
        "pi": "#eb6834",
        "tie": "#b9b8b4",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#333330",
        "claude-code": "#4a90d9",
        "pi": "#d95926",
        "tie": "#55544f",
    },
}

# For each metric: (label, x-axis formatter, higher_is_better).
_METRICS = [
    ("score", "Mean score (0-100)", lambda v, _p: f"{v:.0f}", True),
    ("cost", "Cost per task (USD)", lambda v, _p: f"${v:,.0f}", False),
    ("tokens", "Total tokens processed", None, False),  # tokens fmt set below
    ("minutes", "Wall-clock (min, 5 tasks)", lambda v, _p: f"{v:.0f}m", False),
]


def _human_tokens(value: float, _pos: int = 0) -> str:
    """Compact token count (e.g. 82.7M)."""
    if value >= 1e9:
        return f"{value / 1e9:.1f}B"
    if value >= 1e6:
        return f"{value / 1e6:.0f}M"
    if value >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.0f}"


def _task_shape(data_dir: Path, skill: str, repo: str) -> str | None:
    """Return 'N in : M out (~R:1)' for the median task, across both harnesses.

    Describes what a "task" is in token terms, so the reader knows what cost-per-
    task and tokens-processed are measured over. The input side is the read-heavy
    prompt (fresh input + cache read + cache write); the output side is generation.
    """
    ins: list[int] = []
    outs: list[int] = []
    for harness in HARNESSES:
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
                    task.get("cache_write_tokens")
                    or task.get("cache_creation_tokens")
                    or 0
                )
                o = task.get("output_tokens") or 0
                if (i + cr + cw) > 0 and o > 0:
                    ins.append(i + cr + cw)
                    outs.append(o)
    if not ins:
        return None
    med_in = sorted(ins)[len(ins) // 2]
    med_out = sorted(outs)[len(outs) // 2]
    ratio = round(med_in / max(med_out, 1))
    return (
        f"{_human_tokens(med_in)} input : {_human_tokens(med_out)} output (~{ratio}:1)"
    )


def _collect(data_dir: Path, skill: str, repo: str) -> dict[str, dict[str, Any]]:
    """Return {model: {harness: {score, cost, tokens, minutes}}} for models run
    under BOTH harnesses (a dumbbell needs two dots)."""
    out: dict[str, dict[str, Any]] = {}
    for harness in HARNESSES:
        for row in gen._collect(data_dir, harness, skill, repo):
            cost_str, _ = gen._row_cost(row)
            scored = row.get("num_scored") or 0
            if cost_str == "--" or not scored or row.get("mean") is None:
                continue
            out.setdefault(row["model"], {})[harness] = {
                "score": float(row["mean"]),
                "cost": float(cost_str.lstrip("$")) / scored,
                "tokens": row.get("total_tokens") or 0,
                "minutes": (row.get("latency_seconds") or 0) / 60.0,
            }
    return {m: d for m, d in out.items() if all(h in d for h in HARNESSES)}


def _winner(cc: float, pi: float, higher_is_better: bool) -> str:
    """Return which harness is better for one metric ('claude-code'|'pi'|'tie')."""
    if abs(cc - pi) < 1e-9 or (cc and abs(cc - pi) / max(abs(cc), abs(pi)) < 0.02):
        return "tie"  # within 2% -> effectively a wash
    better_pi = pi > cc if higher_is_better else pi < cc
    return "pi" if better_pi else "claude-code"


def _write_data_json(
    per: dict[str, dict[str, Any]], *, skill: str, repo: str, out_dir: Path
) -> Path:
    """Emit the machine-readable data behind the harness-delta chart.

    One JSON file with, for every model run under BOTH harnesses, the four
    metrics per harness (score, cost/task, tokens, minutes), the per-metric
    winner (same 2%-tie rule as the chart), and per-model / per-metric win
    tallies. This is the single source the author reads when writing the
    hand-authored "Reading the chart" commentary -- so the prose is grounded
    in the same numbers the chart draws, without hardcoding them in prose.
    """
    metric_dirs = {"score": True, "cost": False, "tokens": False, "minutes": False}
    models: dict[str, Any] = {}
    tally = {k: {"pi": 0, "claude-code": 0, "tie": 0} for k in metric_dirs}
    for model in sorted(per):
        cc, pi = per[model]["claude-code"], per[model]["pi"]
        metrics: dict[str, Any] = {}
        for key, higher_is_better in metric_dirs.items():
            win = _winner(cc[key], pi[key], higher_is_better)
            tally[key][win] += 1
            metrics[key] = {
                "claude_code": cc[key],
                "pi": pi[key],
                "winner": win,
                "higher_is_better": higher_is_better,
            }
        models[model] = metrics

    payload = {
        "note": (
            "Data behind docs/images/harness-delta-<skill>.png. Emitted by "
            "plot_harness_delta.py into docs/metrics/. The author-maintained "
            "'Reading the chart' block in the swe comparison doc is written from "
            "THIS file; regenerate the chart, then update that prose to match."
        ),
        "skill": skill,
        "repo": repo,
        "tie_rule": "within 2% counts as a tie",
        "metrics": {
            "score": "mean task score 0-100 (higher is better)",
            "cost": "USD per task (lower is better; cost bases differ by hosting)",
            "tokens": "total tokens processed over the run (lower is better)",
            "minutes": "wall-clock minutes for the 5-task run (lower is better)",
        },
        "n_models": len(models),
        "win_tally": tally,
        "models": models,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"harness-delta-{skill}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s (%d models)", out_path, len(models))
    return out_path


def _panel(
    ax: "plt.Axes",
    models: list[str],
    per: dict[str, dict[str, Any]],
    key: str,
    *,
    label: str,
    higher_is_better: bool,
    xfmt: Any,
    t: dict[str, str],
) -> None:
    """Draw one metric's dumbbell panel and title it with the pi-win tally."""
    y = np.arange(len(models))
    pi_wins = 0
    counted = 0
    for i, m in enumerate(models):
        cc_v = per[m]["claude-code"][key]
        pi_v = per[m]["pi"][key]
        win = _winner(cc_v, pi_v, higher_is_better)
        if win != "tie":
            counted += 1
            pi_wins += win == "pi"
        line_color = t[win] if win != "tie" else t["tie"]
        ax.plot(
            [cc_v, pi_v],
            [i, i],
            color=line_color,
            linewidth=2.2,
            zorder=2,
            solid_capstyle="round",
        )
        # winning dot larger; both dots wear their harness color.
        cc_big = win == "claude-code"
        ax.scatter(
            cc_v,
            i,
            s=90 if cc_big else 46,
            color=t["claude-code"],
            edgecolors=t["surface"],
            linewidths=0.8,
            zorder=3,
        )
        ax.scatter(
            pi_v,
            i,
            s=90 if win == "pi" else 46,
            color=t["pi"],
            edgecolors=t["surface"],
            linewidths=0.8,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=ROW_LABEL_FONTSIZE, color=t["ink"])
    ax.invert_yaxis()
    tally = f"pi better on {pi_wins} of {counted}" if counted else "all ties"
    ax.set_title(
        f"{label}   ({tally})",
        fontsize=PANEL_TITLE_FONTSIZE,
        color=t["ink"],
        loc="left",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(xfmt))
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=TICK_FONTSIZE, length=0)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    lo = min(min(per[m]["claude-code"][key], per[m]["pi"][key]) for m in models)
    hi = max(max(per[m]["claude-code"][key], per[m]["pi"][key]) for m in models)
    pad = (hi - lo) * 0.08 or 1
    ax.set_xlim(lo - pad, hi + pad)


def _plot(
    per: dict[str, dict[str, Any]],
    *,
    skill: str,
    mode: str,
    out_dir: Path,
    task_shape: str | None = None,
) -> Path:
    """Render the 2x2 dumbbell small-multiples."""
    t = _THEME[mode]
    # Order by pi-vs-cc score gap is tempting, but a stable read is best: order by
    # best score across harnesses (best at top), shared down every panel.
    models = sorted(per, key=lambda m: -max(per[m][h]["score"] for h in HARNESSES))
    height = max(6.0, 0.36 * len(models) + 2.2)
    fig, axes = plt.subplots(2, 2, figsize=(15, height), facecolor=t["surface"])
    ax_list = axes.flat

    fmts = {
        "score": _METRICS[0][2],
        "cost": _METRICS[1][2],
        "tokens": _human_tokens,
        "minutes": _METRICS[3][2],
    }
    for ax, (key, label, _fmt, hib) in zip(ax_list, _METRICS):
        ax.set_facecolor(t["surface"])
        _panel(
            ax, models, per, key, label=label, higher_is_better=hib, xfmt=fmts[key], t=t
        )

    # Legend: the two harness dots + what a bold connector means.
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            color=t["claude-code"],
            label="Claude Code",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=8, color=t["pi"], label="pi"
        ),
        plt.Line2D(
            [],
            [],
            color=t["muted"],
            linewidth=2.2,
            label="line + larger dot = better harness for that metric",
        ),
    ]
    fig.suptitle(
        f"Does the harness matter? Claude Code vs pi on {skill}, per model "
        f"({len(models)} run under both)",
        fontsize=SUPTITLE_FONTSIZE,
        color=t["ink"],
        x=0.02,
        y=0.985,
        ha="left",
    )
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        ncol=3,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        labelcolor=t["muted"],
        bbox_to_anchor=(0.5, 0.945),
    )
    task_line = (
        f"A task = one real {skill} problem on this repo (5 tasks per run); the "
        f"median task processes ~{task_shape} tokens. "
        if task_shape
        else ""
    )
    method_line = (
        "Each row: the same model under both harnesses; the connector points to the "
        "better harness for that metric (higher score / lower cost, tokens, latency; "
        "<2% gap counts as a tie). Comparing one model's two harnesses is fair even "
        "for cost -- its hosting basis is identical under both."
    )
    fig.text(0.01, 0.028, task_line, fontsize=FOOTNOTE_FONTSIZE, color=t["muted"])
    fig.text(0.01, 0.006, method_line, fontsize=FOOTNOTE_FONTSIZE, color=t["muted"])
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))

    suffix = "-dark" if mode == "dark" else ""
    out = out_dir / f"harness-delta-{skill}{suffix}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    logger.info("wrote %s (%d models)", out, len(models))
    return out


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Dumbbell small-multiples: which harness wins per model, per metric.",
        epilog="Example: uv run scripts/plot_harness_delta.py --skill swe3",
    )
    parser.add_argument(
        "--skill", default="swe3", help="SWE skill: 'swe3' (default) or 'swe2'."
    )
    parser.add_argument("--repo", default="mcp-gateway-registry", help="Dataset scope.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help="Where to write the machine-readable chart data JSON.",
    )
    parser.add_argument(
        "--dark", action="store_true", help="Render the dark-theme variant."
    )
    return parser.parse_args()


def main() -> None:
    """Collect both harnesses' per-model metrics and render the dumbbell facet."""
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    per = _collect(data_dir, args.skill, args.repo)
    if not per:
        raise SystemExit(f"no models run under BOTH harnesses for skill={args.skill}")
    # Emit the machine-readable data first (light theme run only, to avoid a
    # duplicate write on the --dark pass -- the numbers are theme-independent).
    if not args.dark:
        _write_data_json(
            per,
            skill=args.skill,
            repo=args.repo,
            out_dir=args.metrics_dir.expanduser().resolve(),
        )
    _plot(
        per,
        skill=args.skill,
        mode="dark" if args.dark else "light",
        out_dir=args.out_dir.expanduser().resolve(),
        task_shape=_task_shape(data_dir, args.skill, args.repo),
    )


if __name__ == "__main__":
    main()
