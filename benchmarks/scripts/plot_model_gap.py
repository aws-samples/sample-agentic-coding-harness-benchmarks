#!/usr/bin/env python3
"""Plot the per-task score gap between two models, coloured by complexity tier.

The tier charts answer "does a model upgrade pay off more on harder work?" by
comparing tier *means*. That question has a tidy answer and a misleading one: the
means differ by only a few points, while the per-task gaps behind them range from
-12 to +16. Averaging inside a tier hides that entirely.

This plots one bar per task -- the score of the upgraded model minus the
baseline -- sorted by size and coloured by tier. If complexity predicted the
payoff, the colours would band. Whether they do is the point of the chart, and
the caption states the measured share of variance that tier actually explains, so
the reader does not have to eyeball it.

Usage:
    uv run scripts/plot_model_gap.py --baseline claude-sonnet-5 \
        --upgrade claude-opus-5 --scope mcp-gateway-registry-v2 --both
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
from matplotlib.patches import Patch  # noqa: E402

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
HARNESS_LABELS = {"claude-code": "Claude Code", "pi": "Pi", "omp": "omp"}

TIERS = ("trivial", "low", "medium", "high")
LABEL_PREFIX_TO_DROP = "claude-"

# The same validated ordinal ramp the other v2 charts use, so a tier keeps one
# colour across every chart in the set.
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


def _load(data_dir: Path, model: str, harness: str, skill: str, scope: str) -> dict:
    """Load one model's committed run summary.

    Args:
        data_dir: The swe-benchmark-data root.
        model: Model slug.
        harness: Harness slug.
        skill: Skill folder.
        scope: Dataset scope folder.

    Returns:
        Task id -> task row, for tasks that carry a score.

    Raises:
        SystemExit: If the summary does not exist.
    """
    path = data_dir / model / harness / skill / scope / RUN_SUMMARY_FILENAME
    if not path.is_file():
        raise SystemExit(f"no run summary at {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {
        t["task"]: t
        for t in summary.get("tasks", [])
        if t.get("task_score") is not None
    }


def _gaps(base: dict, upgrade: dict) -> list[tuple[float, str, str]]:
    """Return (gap, tier, task) for every task both models scored, largest first.

    Args:
        base: Baseline model's rows, keyed by task.
        upgrade: Upgraded model's rows, keyed by task.

    Returns:
        Sorted list of per-task gaps.

    Raises:
        SystemExit: If the two models share no scored task.
    """
    shared = [t for t in upgrade if t in base]
    if not shared:
        raise SystemExit("the two models share no scored task")
    rows = [
        (
            upgrade[t]["task_score"] - base[t]["task_score"],
            upgrade[t].get("complexity") or base[t].get("complexity") or "",
            t,
        )
        for t in shared
    ]
    rows.sort(reverse=True)
    return rows


def _variance_explained(rows: list[tuple[float, str, str]]) -> float | None:
    """Return the share of gap variance attributable to the tier, 0-1.

    A one-way between-groups decomposition: how much of the spread in per-task
    gaps is captured by which tier the task is in, versus differences between
    tasks inside the same tier.

    Args:
        rows: Output of ``_gaps``.

    Returns:
        Between-group sum of squares over the total, or None if the total is zero
        (every gap identical) or fewer than two tiers are present.
    """
    by: dict[str, list[float]] = {}
    for gap, tier, _ in rows:
        by.setdefault(tier, []).append(gap)
    if len(by) < 2:
        return None
    grand = mean(g for g, _, _ in rows)
    between = sum(len(v) * (mean(v) - grand) ** 2 for v in by.values())
    within = sum(sum((x - mean(v)) ** 2 for x in v) for v in by.values())
    total = between + within
    return between / total if total else None


def _plot(
    rows: list[tuple[float, str, str]],
    *,
    baseline: str,
    upgrade: str,
    mode: str,
    harness: str,
    skill: str,
    scope: str,
    out_dir: Path,
) -> Path:
    """Render the per-task gap bars and save the PNG.

    Args:
        rows: Output of ``_gaps``.
        baseline: Baseline model slug (the cheaper model).
        upgrade: Upgraded model slug.
        mode: "light" or "dark".
        harness: Harness slug, for the title and filename.
        skill: Skill name, for the title and filename.
        scope: Dataset scope, for the subtitle and filename.
        out_dir: Where to write the PNG.

    Returns:
        The written path.
    """
    theme = _THEME[mode]
    colour = dict(zip(TIERS, theme["tiers"]))
    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=150)
    fig.patch.set_facecolor(theme["surface"])

    pos = list(range(len(rows)))
    ax.barh(
        pos,
        [r[0] for r in rows],
        color=[colour.get(r[1], theme["muted"]) for r in rows],
        height=0.72,  # leaves a surface gap between adjacent bars
        zorder=3,
    )
    ax.set_yticks(pos)
    ax.set_yticklabels([r[2] for r in rows], fontsize=8.5, color=theme["ink"])
    ax.invert_yaxis()

    # Value at each bar end, on the outside, so a negative bar's label does not
    # sit on top of the zero line.
    span = max(abs(r[0]) for r in rows) or 1
    for p, (gap, _, _) in zip(pos, rows):
        off = 0.35 if gap >= 0 else -0.35
        ax.text(
            gap + off,
            p,
            f"{gap:+.1f}",
            va="center",
            ha="left" if gap >= 0 else "right",
            fontsize=8.5,
            color=theme["ink"],
            zorder=4,
        )
    ax.set_xlim(-span * 1.25, span * 1.25)
    # Zero is the meaningful reference here: left of it the upgrade lost.
    ax.axvline(0, color=theme["ink"], linewidth=1.2, zorder=2)

    base_label = baseline.removeprefix(LABEL_PREFIX_TO_DROP)
    up_label = upgrade.removeprefix(LABEL_PREFIX_TO_DROP)
    ax.set_xlabel(
        f"Task score: {up_label} minus {base_label}  "
        f"(left of zero = {base_label} scored higher)",
        fontsize=10.5,
        color=theme["muted"],
    )
    ax.grid(True, axis="x", color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"])
    ax.set_facecolor(theme["surface"])

    present = [t for t in TIERS if any(r[1] == t for r in rows)]
    legend = ax.legend(
        handles=[Patch(facecolor=colour[t], label=t) for t in present],
        loc="lower right",
        frameon=False,
        fontsize=10,
        title="Task complexity",
    )
    for text in legend.get_texts():
        text.set_color(theme["ink"])
    if legend.get_title():
        legend.get_title().set_color(theme["muted"])
        legend.get_title().set_fontsize(9)

    fig.suptitle(
        f"What the {up_label} upgrade buys, task by task -- "
        f"{HARNESS_LABELS.get(harness, harness)} harness, /{skill}",
        fontsize=13.5,
        color=theme["ink"],
        y=0.975,
    )
    share = _variance_explained(rows)
    note = (
        f"{scope}: {len(rows)} tasks, mean gap "
        f"{mean(r[0] for r in rows):+.2f}. Bars are sorted by size, not grouped by "
        "tier -- the colours land where they land."
    )
    if share is not None:
        note += (
            f" Complexity explains {share * 100:.0f}% of the variance in these gaps."
        )
    fig.text(
        0.5, 0.935, note, ha="center", va="top", fontsize=9.5, color=theme["muted"]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    code = HARNESS_CODES.get(harness, harness)
    suffix = "-dark" if mode == "dark" else ""
    out = (
        out_dir
        / f"model-gap-{base_label}-vs-{up_label}-{code}-{skill}-{scope}{suffix}.png"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    fig.savefig(out, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Plot the per-task score gap between two models by tier."
    )
    p.add_argument("--baseline", required=True, help="Cheaper model slug")
    p.add_argument("--upgrade", required=True, help="More expensive model slug")
    p.add_argument("--harness", default="pi")
    p.add_argument("--skill", default="swe3")
    p.add_argument("--scope", default="mcp-gateway-registry-v2")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--dark", action="store_true")
    p.add_argument("--both", action="store_true", help="Render light and dark")
    return p.parse_args()


def main() -> None:
    """Load both models, compute the per-task gaps, and render."""
    args = _parse_args()
    base = _load(args.data_dir, args.baseline, args.harness, args.skill, args.scope)
    up = _load(args.data_dir, args.upgrade, args.harness, args.skill, args.scope)
    rows = _gaps(base, up)
    modes = ("light", "dark") if args.both else (("dark",) if args.dark else ("light",))
    for mode in modes:
        _plot(
            rows,
            baseline=args.baseline,
            upgrade=args.upgrade,
            mode=mode,
            harness=args.harness,
            skill=args.skill,
            scope=args.scope,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
