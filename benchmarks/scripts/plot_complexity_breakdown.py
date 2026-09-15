#!/usr/bin/env python3
"""Render the complexity-tier view of a single model's run on one dataset.

The cost/quality charts plot one point per model and say nothing about *which*
tasks a model handled. That is fine for a five-task dataset whose tasks are all
medium/high, but the v2 dataset is deliberately balanced across low / medium /
high complexity, and the question it exists to answer is where a model starts to
break down. This renders that, in two panels:

* **Score by task, grouped by complexity** -- every task as its own bar, tiers
  banded together with their mean called out. Answers "which tasks did it get
  right, and does difficulty predict the score?"
* **Artifact profile by complexity** -- the five judged artifacts (issue spec ->
  LLD -> review -> testing -> implementation) as a line per tier. Answers "*where*
  in the pipeline does difficulty bite?", which the per-task view cannot show.

Complexity is ORDINAL (low < medium < high), so the tiers wear a single-hue
ordinal ramp rather than three categorical hues -- the ramp itself encodes the
ordering. Both ramps are validated with the dataviz skill's checker (light and
dark, ``--ordinal``).

Scores are read verbatim from the committed ``run-summary.json``; nothing is
re-scored here.

Usage:
    uv run scripts/plot_complexity_breakdown.py --model claude-haiku-4-5 \
        --scope mcp-gateway-registry-v2
    uv run scripts/plot_complexity_breakdown.py --model claude-haiku-4-5 \
        --scope mcp-gateway-registry-v2 --dark
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

# Short per-harness code used (with the skill) to suffix chart filenames, matching
# the other plot scripts so a chart is self-identifying and never overwrites
# another harness's.
HARNESS_CODES = {
    "claude-code": "cc",
    "pi": "pi",
    "omp": "omp",
    "opencode": "oc",
    "kiro-cli": "kiro",
}
HARNESS_LABELS = {
    "claude-code": "Claude Code",
    "pi": "Pi",
    "omp": "omp",
    "opencode": "opencode",
    "kiro-cli": "Kiro CLI",
}

# Complexity tiers, in order. The order is the encoding -- do not sort these.
# "trivial" was added after the first three-model run; a tier with no tasks is
# dropped at render time, so this list stays valid for older datasets.
TIERS = ("trivial", "low", "medium", "high")

# The five judged artifacts, in the order the skill produces them, so the profile
# panel reads left-to-right as the task actually progressed: specify, design,
# review, plan the tests, then build it.
ARTIFACTS = ("github_issue", "lld", "review", "testing", "implementation")
ARTIFACT_LABELS = ("Issue spec", "LLD", "Review", "Testing", "Implementation")

# Ordinal ramp (dataviz reference palette, blue). Validated with
# `validate_palette.js --ordinal` in both modes: monotone lightness, adjacent
# delta-L >= 0.06, single hue, and the step nearest the surface clearing 2:1
# (light end 2.06:1 on light, 2.63:1 on dark). Re-validated at 4 steps when the
# trivial tier was added -- the 3-step light ramp's top step had to darken from
# #b7d3f6 (1.50:1, FAIL) to #86b6ef. Marks carry the tier; text wears ink tokens
# only.
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


def _load_summary(data_dir: Path, model: str, harness: str, skill: str, scope: str):
    """Load a committed run-summary.json.

    Args:
        data_dir: The swe-benchmark-data root.
        model: Model slug (folder name).
        harness: Harness slug (folder name).
        skill: Skill folder name.
        scope: Dataset scope folder name.

    Returns:
        The parsed summary dict.

    Raises:
        SystemExit: If the summary does not exist.
    """
    path = data_dir / model / harness / skill / scope / RUN_SUMMARY_FILENAME
    if not path.is_file():
        raise SystemExit(f"no run summary at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _tier_rows(summary: dict) -> dict[str, list[dict]]:
    """Group a summary's scored task rows by complexity tier, best score first.

    Args:
        summary: A parsed run-summary.json.

    Returns:
        Tier name -> its task rows, each sorted by descending score.

    Raises:
        SystemExit: If no task carries a complexity label.
    """
    grouped: dict[str, list[dict]] = {t: [] for t in TIERS}
    for row in summary.get("tasks", []):
        tier = row.get("complexity")
        if tier in grouped and row.get("task_score") is not None:
            grouped[tier].append(row)
    if not any(grouped.values()):
        raise SystemExit("no scored tasks carry a complexity label")
    for rows in grouped.values():
        rows.sort(key=lambda r: r["task_score"], reverse=True)
    return {t: rows for t, rows in grouped.items() if rows}


def _artifact_profile(rows: list[dict]) -> list[float | None]:
    """Return the mean score per artifact across ``rows``.

    Args:
        rows: Task rows from one tier.

    Returns:
        One mean per entry in ARTIFACTS; None where no task carries that artifact.
    """
    profile: list[float | None] = []
    for artifact in ARTIFACTS:
        totals = [
            (r.get("eval_scores") or {}).get(artifact, {}).get("total")
            for r in rows
            if (r.get("eval_scores") or {}).get(artifact, {}).get("total") is not None
        ]
        profile.append(mean(totals) if totals else None)
    return profile


def _plot_tasks(ax, grouped: dict[str, list[dict]], theme: dict) -> None:
    """Draw the per-task score bars, banded by tier.

    Bars are horizontal so the task slugs read as text rather than rotated
    labels, and each bar is value-labelled at its end -- with 15 bars and no
    other way to read a value, the end label replaces gridline-counting.

    Args:
        ax: The axes to draw on.
        grouped: Tier -> sorted task rows.
        theme: The resolved theme dict.
    """
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    tier_spans: list[tuple[str, int, int, float]] = []

    row_index = 0
    for tier, color in zip(TIERS, theme["tiers"]):
        rows = grouped.get(tier) or []
        if not rows:
            continue
        start = row_index
        for row in rows:
            labels.append(row["task"])
            values.append(row["task_score"])
            colors.append(color)
            row_index += 1
        tier_spans.append(
            (tier, start, row_index - 1, mean(r["task_score"] for r in rows))
        )

    # Top-to-bottom reading order: invert so index 0 sits at the top.
    positions = list(range(len(values)))
    ax.barh(
        positions,
        values,
        color=colors,
        height=0.72,  # leaves a surface gap between adjacent bars
        zorder=3,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.5, color=theme["ink"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Task score (0-100)", fontsize=10, color=theme["muted"])

    for pos, value in zip(positions, values):
        ax.text(
            value + 1.5,
            pos,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=theme["ink"],
            zorder=4,
        )

    # Tier bands: a right-edge bracket with the tier name and its mean, so the
    # grouping is stated in text and not carried by the color ramp alone.
    for tier, start, end, tier_mean in tier_spans:
        ax.annotate(
            f"{tier}  mean {tier_mean:.1f}",
            xy=(99, (start + end) / 2),
            fontsize=9,
            color=theme["muted"],
            ha="right",
            va="center",
            rotation=90,
        )
        if end + 1 < len(values):
            ax.axhline(end + 0.5, color=theme["grid"], linewidth=1.0, zorder=1)

    ax.grid(True, axis="x", color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"])
    ax.set_facecolor(theme["surface"])
    ax.set_title(
        "Score by task, grouped by complexity",
        fontsize=11,
        color=theme["ink"],
        pad=10,
        loc="left",
    )


def _plot_profile(ax, grouped: dict[str, list[dict]], theme: dict) -> None:
    """Draw the artifact-stage profile, one line per complexity tier.

    Args:
        ax: The axes to draw on.
        grouped: Tier -> task rows.
        theme: The resolved theme dict.
    """
    xs = list(range(len(ARTIFACTS)))
    for tier, color in zip(TIERS, theme["tiers"]):
        rows = grouped.get(tier) or []
        if not rows:
            continue
        profile = _artifact_profile(rows)
        pts = [(x, y) for x, y in zip(xs, profile) if y is not None]
        if not pts:
            continue
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=color,
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor=theme["surface"],
            markeredgewidth=2,  # surface ring keeps overlapping marks separable
            label=tier,
            zorder=3,
        )
        # Direct-label the line end -- three labels, not a number per point.
        ax.annotate(
            f"{tier} {pts[-1][1]:.0f}",
            xy=pts[-1],
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9,
            color=theme["ink"],
            va="center",
            zorder=4,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(ARTIFACT_LABELS, fontsize=9, color=theme["ink"])
    ax.set_xlim(-0.3, len(ARTIFACTS) - 0.3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean artifact score (0-100)", fontsize=10, color=theme["muted"])
    ax.grid(True, axis="y", color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"])
    ax.set_facecolor(theme["surface"])
    ax.set_title(
        "Where difficulty bites: mean score per artifact",
        fontsize=11,
        color=theme["ink"],
        pad=10,
        loc="left",
    )


def _plot(
    summary: dict,
    *,
    mode: str,
    harness: str,
    skill: str,
    scope: str,
    out_dir: Path,
) -> Path:
    """Render both panels and save the PNG.

    Args:
        summary: The parsed run-summary.json.
        mode: "light" or "dark".
        harness: Harness slug, for the title and filename.
        skill: Skill name, for the title and filename.
        scope: Dataset scope, named in the subtitle.
        out_dir: Where to write the PNG.

    Returns:
        The written path.
    """
    theme = _THEME[mode]
    grouped = _tier_rows(summary)
    fig, (ax_tasks, ax_profile) = plt.subplots(
        1, 2, figsize=(15, 8), dpi=150, gridspec_kw={"width_ratios": [1.25, 1]}
    )
    fig.patch.set_facecolor(theme["surface"])

    _plot_tasks(ax_tasks, grouped, theme)
    _plot_profile(ax_profile, grouped, theme)

    # Legend for the tier ramp: >= 2 series, so identity is never color-alone.
    handles = [
        Line2D([], [], color=c, linewidth=6, label=t)
        for t, c in zip(TIERS, theme["tiers"])
        if grouped.get(t)
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.015),
        title="Task complexity",
    )
    for text in legend.get_texts():
        text.set_color(theme["ink"])
    if legend.get_title():
        legend.get_title().set_color(theme["muted"])
        legend.get_title().set_fontsize(9)

    model = summary.get("model_slug") or summary.get("model") or "model"
    overall = summary.get("mean_task_score_excl_failed")
    fig.suptitle(
        f"{model} by task complexity -- "
        f"{HARNESS_LABELS.get(harness, harness)} harness, /{skill}",
        fontsize=14,
        color=theme["ink"],
        y=0.985,
    )
    refs = summary.get("refs") or []
    ref_phrase = (
        f"{len(refs)} release tags" if len(refs) > 1 else (refs[0] if refs else "n/a")
    )
    fig.text(
        0.5,
        0.945,
        f"{scope}: {summary.get('num_scored')} of {summary.get('num_tasks')} tasks "
        f"scored across {ref_phrase}"
        + (f"; overall mean {overall}" if overall is not None else ""),
        ha="center",
        va="top",
        fontsize=9.5,
        color=theme["muted"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    code = HARNESS_CODES.get(harness, harness)
    suffix = "-dark" if mode == "dark" else ""
    # The model is part of the filename: this chart is per-model, so leaving it
    # out makes a second model silently overwrite the first one's chart.
    out = out_dir / f"complexity-{model}-{code}-{skill}-{scope}{suffix}.png"
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(out, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Plot per-task scores and artifact profile by complexity tier."
    )
    p.add_argument("--model", required=True, help="Model slug (folder name)")
    p.add_argument("--harness", default="pi", help="Harness slug (default: pi)")
    p.add_argument("--skill", default="swe3", help="Skill folder (default: swe3)")
    p.add_argument(
        "--scope",
        default="mcp-gateway-registry-v2",
        help="Dataset scope folder (default: mcp-gateway-registry-v2)",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--dark", action="store_true", help="Render the dark variant")
    p.add_argument(
        "--both", action="store_true", help="Render light and dark in one go"
    )
    return p.parse_args()


def main() -> None:
    """Load the summary and render the requested variant(s)."""
    args = _parse_args()
    summary = _load_summary(
        args.data_dir, args.model, args.harness, args.skill, args.scope
    )
    modes = ("light", "dark") if args.both else (("dark",) if args.dark else ("light",))
    for mode in modes:
        _plot(
            summary,
            mode=mode,
            harness=args.harness,
            skill=args.skill,
            scope=args.scope,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
