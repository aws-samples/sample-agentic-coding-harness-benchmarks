#!/usr/bin/env python3
"""Plot cost vs quality per complexity tier, to pick a model per class of work.

The whole-dataset cost/quality chart answers "which model is worth its price
overall". That is the wrong question when your backlog is not uniformly hard: a
model that is poor value on trivial work can be the only sane choice on hard
work, and the single-mean view hides it.

This draws one line per complexity tier through the models in ascending cost, so
each line is the cost/quality path you walk by upgrading the model *for that class
of task*. A flat segment means the upgrade bought nothing; a long horizontal jump
means it cost a great deal to buy it.

Tiers wear the same single-hue ordinal ramp as the complexity breakdown chart
(low < medium < high is an ordering, not three unrelated categories), and each
model gets its own marker shape, so model identity never rests on colour.

Usage:
    uv run scripts/plot_tier_frontier.py --scope mcp-gateway-registry-v2 \
        --models claude-haiku-4-5 claude-sonnet-5 claude-opus-5 --both
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never a display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
_BENCHMARKS_DIR = _SCRIPTS_DIR.parent
_REPO_ROOT = _BENCHMARKS_DIR.parent
DEFAULT_DATA_DIR = _BENCHMARKS_DIR / "swe-benchmark-data"
DEFAULT_OUT_DIR = _REPO_ROOT / "docs" / "images"
RUN_SUMMARY_FILENAME = "run-summary.json"

HARNESS_CODES = {"claude-code": "cc", "pi": "pi", "omp": "omp", "kiro-cli": "kiro"}
HARNESS_LABELS = {
    "claude-code": "Claude Code",
    "pi": "Pi",
    "omp": "omp",
    "kiro-cli": "Kiro CLI",
}

TIERS = ("trivial", "low", "medium", "high")
# Marker per model, assigned in the order the models are given (ascending cost),
# so identity is carried by shape as well as position.
MARKERS = ("o", "s", "^", "D", "v")
# Model-name prefix dropped from point labels; "claude-" on every label is noise.
LABEL_PREFIX_TO_DROP = "claude-"

# Same validated ordinal ramp as plot_complexity_breakdown.py -- the two charts
# describe the same tiers and must not disagree about which blue means "high".
# 4 steps since the trivial tier was added; re-validated in both modes.
_THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e2",
        "tiers": ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b"),
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#333330",
        "tiers": ("#cde2fb", "#9ec5f4", "#5598e7", "#1c5cab"),
    },
}


def _tier_means(summary: dict) -> dict[str, tuple[float, float]]:
    """Return tier -> (mean cost, mean score) for one model's run.

    Args:
        summary: A parsed run-summary.json.

    Returns:
        Tier name -> (mean cost per task, mean task score). Tiers with no scored
        task are omitted.
    """
    out: dict[str, tuple[float, float]] = {}
    for tier in TIERS:
        rows = [
            t
            for t in summary.get("tasks", [])
            if t.get("complexity") == tier
            and t.get("task_score") is not None
            and t.get("total_cost_usd") is not None
        ]
        if rows:
            out[tier] = (
                mean(t["total_cost_usd"] for t in rows),
                mean(t["task_score"] for t in rows),
            )
    return out


def _load(data_dir: Path, models: list[str], harness: str, skill: str, scope: str):
    """Load every model's summary and reduce it to per-tier means.

    Args:
        data_dir: The swe-benchmark-data root.
        models: Model slugs to include.
        harness: Harness slug.
        skill: Skill folder.
        scope: Dataset scope folder.

    Returns:
        List of (model slug, {tier: (cost, score)}), in ascending overall cost.

    Raises:
        SystemExit: If fewer than two models have a summary -- a one-point line
            states nothing, which is the whole reason this chart exists.
    """
    loaded = []
    for model in models:
        path = data_dir / model / harness / skill / scope / RUN_SUMMARY_FILENAME
        if not path.is_file():
            logger.warning("skipping %s: no summary at %s", model, path)
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        tiers = _tier_means(summary)
        if tiers:
            loaded.append((model, tiers))
    if len(loaded) < 2:
        raise SystemExit(
            f"need at least 2 models with summaries under {scope}; found {len(loaded)}"
        )
    # Ascending cost: the lines then read left-to-right as "upgrade the model".
    loaded.sort(key=lambda mt: mean(c for c, _ in mt[1].values()))
    return loaded


def _plot(loaded, *, mode: str, harness: str, skill: str, scope: str, out_dir: Path):
    """Render the per-tier cost/quality paths and save the PNG.

    Args:
        loaded: Output of ``_load``.
        mode: "light" or "dark".
        harness: Harness slug, for the title and filename.
        skill: Skill name, for the title and filename.
        scope: Dataset scope, for the subtitle and filename.
        out_dir: Where to write the PNG.

    Returns:
        The written path.
    """
    theme = _THEME[mode]
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=150)
    fig.patch.set_facecolor(theme["surface"])

    for tier, color in zip(TIERS, theme["tiers"]):
        pts = [(t[tier][0], t[tier][1], m) for m, t in loaded if tier in t]
        if len(pts) < 2:
            continue
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=color,
            linewidth=2,
            zorder=2,
            label=tier,
        )
        for (x, y, model), marker in zip(pts, MARKERS):
            ax.plot(
                x,
                y,
                marker=marker,
                markersize=10,
                color=color,
                markeredgecolor=theme["surface"],
                markeredgewidth=2,  # surface ring keeps crossing marks separable
                zorder=3,
            )
        # Label the tier once, at its most expensive end, rather than every point.
        ax.annotate(
            tier,
            xy=(pts[-1][0], pts[-1][1]),
            xytext=(10, -3),
            textcoords="offset points",
            fontsize=10,
            color=theme["ink"],
            va="center",
            zorder=4,
        )

    ax.set_xlabel("Mean cost per task ($)", fontsize=11, color=theme["muted"])
    ax.set_ylabel("Mean task score (0-100)", fontsize=11, color=theme["muted"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"])
    ax.set_facecolor(theme["surface"])
    ax.set_xlim(left=0)

    tier_handles = [
        Line2D([], [], color=c, linewidth=3, label=t)
        for t, c in zip(TIERS, theme["tiers"])
    ]
    model_handles = [
        Line2D(
            [],
            [],
            color=theme["muted"],
            marker=mk,
            linestyle="none",
            markersize=9,
            label=m.removeprefix(LABEL_PREFIX_TO_DROP),
        )
        for (m, _), mk in zip(loaded, MARKERS)
    ]
    legend = ax.legend(
        handles=tier_handles + model_handles,
        loc="lower right",
        frameon=False,
        fontsize=10,
        ncol=2,
    )
    for text in legend.get_texts():
        text.set_color(theme["ink"])

    fig.suptitle(
        f"What a model upgrade buys, per complexity tier -- "
        f"{HARNESS_LABELS.get(harness, harness)} harness, /{skill}",
        fontsize=13.5,
        color=theme["ink"],
        y=0.97,
    )
    fig.text(
        0.5,
        0.915,
        f"{scope}: each line walks one tier's tasks from the cheapest model to the "
        "costliest. Flat = the upgrade bought little; wide = it cost a lot.",
        ha="center",
        va="top",
        fontsize=9.5,
        color=theme["muted"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    code = HARNESS_CODES.get(harness, harness)
    suffix = "-dark" if mode == "dark" else ""
    out = out_dir / f"tier-frontier-{code}-{skill}-{scope}{suffix}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Plot cost vs quality per complexity tier, across models."
    )
    p.add_argument("--models", nargs="+", required=True, help="Model slugs to include")
    p.add_argument("--harness", default="pi")
    p.add_argument("--skill", default="swe3")
    p.add_argument("--scope", default="mcp-gateway-registry-v2")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--dark", action="store_true")
    p.add_argument("--both", action="store_true", help="Render light and dark")
    return p.parse_args()


def main() -> None:
    """Load every model's per-tier means and render the requested variant(s)."""
    args = _parse_args()
    loaded = _load(args.data_dir, args.models, args.harness, args.skill, args.scope)
    modes = ("light", "dark") if args.both else (("dark",) if args.dark else ("light",))
    for mode in modes:
        _plot(
            loaded,
            mode=mode,
            harness=args.harness,
            skill=args.skill,
            scope=args.scope,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
