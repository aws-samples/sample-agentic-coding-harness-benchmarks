#!/usr/bin/env python3
"""Render radar (spider) charts of per-dimension quality scores from eval data.

Two views, one point per model, for the models whose committed
``run-summary.json`` carries the per-artifact ``eval_scores`` breakdown (the
judge's four criteria for each of the six artifacts):

* **By criterion** -- Completeness, Correctness, Specificity, Risk-awareness,
  each averaged across all of a model's artifacts and shown as a percentage of
  the 25-point-per-criterion maximum. Answers "where is this model strong/weak
  in *how* it works a task?"
* **By artifact** -- github-issue, LLD, review, testing, implementation, each
  the mean artifact total (0-100). Answers "which deliverable is this model
  best at producing?"

Only a subset of models currently embed ``eval_scores`` (the runs produced on
this node). The chart notes that the same breakdown for the remaining models is
coming as their eval data is backfilled. Scores are read verbatim from the
committed summaries -- no re-scoring here.

Usage:
    uv run scripts/plot_quality_radar.py
    uv run scripts/plot_quality_radar.py --dark
    uv run scripts/plot_quality_radar.py --repo mcp-gateway-registry --out-dir ../docs/images
"""

from __future__ import annotations

import argparse
import json
import logging
from math import pi
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never a display
import matplotlib.pyplot as plt  # noqa: E402

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

# Short per-harness code used (with the skill) to suffix chart filenames so each
# agent+skill's charts are self-identifying and never overwrite another's (e.g.
# quality-radar-cc-swe2.png, quality-radar-pi-swe3.png). An unknown harness falls
# back to its own slug.
HARNESS_CODES = {"claude-code": "cc", "pi": "pi", "opencode": "oc", "kiro-cli": "kiro"}
# Human-readable harness names for the chart title (the code is for filenames).
HARNESS_LABELS = {
    "claude-code": "Claude Code",
    "pi": "pi",
    "opencode": "opencode",
    "kiro-cli": "kiro-cli",
}


def _harness_code(harness: str) -> str:
    """Return the short filename code for a harness slug (cc, pi, ...)."""
    return HARNESS_CODES.get(harness, harness)


# The judge's four criteria (each scored 0-25 per artifact) and the six
# artifacts a /swe2 run produces. Order is fixed so every chart reads the same.
CRITERIA = ("completeness", "correctness", "specificity", "risk_awareness")
CRITERION_LABELS = ("Completeness", "Correctness", "Specificity", "Risk-awareness")
CRITERION_MAX = 25.0
ARTIFACTS = ("github_issue", "lld", "review", "testing", "implementation")
ARTIFACT_LABELS = ("GitHub issue", "LLD", "Review", "Testing", "Implementation")

# Categorical palette from the dataviz skill's validated reference instance
# (slots 1-4, blue/orange/aqua/yellow), fixed order, validated in both modes with
# scripts/validate_palette.js (all CVD + normal-vision checks PASS; the light
# contrast WARN is covered by the legend + direct labels this chart always draws).
# Text and grid wear neutral ink tokens; series color carries identity.
_THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#d8d7d3",
        "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3a37",
        "series": ("#3987e5", "#d95926", "#199e70", "#c98500"),
    },
}


def _read_json(path: Path) -> dict | None:
    """Return the parsed JSON object at ``path``, or None if absent/invalid."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _model_dimensions(
    summary: dict,
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Return (by_criterion_pct, by_artifact_total) means from a run summary.

    ``by_criterion_pct`` averages each criterion over every artifact of every
    task and scales it to 0-100 (percent of the 25-point max). ``by_artifact_total``
    averages each artifact's 0-100 total over tasks. Returns None when the summary
    carries no per-artifact ``eval_scores`` (so the caller can skip the model).
    """
    crit_sums: dict[str, list[float]] = {c: [] for c in CRITERIA}
    art_totals: dict[str, list[float]] = {a: [] for a in ARTIFACTS}
    saw_scores = False
    for task in summary.get("tasks", []):
        eval_scores = task.get("eval_scores") or {}
        for artifact, scores in eval_scores.items():
            saw_scores = True
            if artifact in art_totals and isinstance(scores.get("total"), (int, float)):
                art_totals[artifact].append(float(scores["total"]))
            for criterion in CRITERIA:
                value = scores.get(criterion)
                if isinstance(value, (int, float)):
                    crit_sums[criterion].append(float(value))
    if not saw_scores:
        return None

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    by_criterion = {
        c: round(_mean(crit_sums[c]) / CRITERION_MAX * 100.0, 1) for c in CRITERIA
    }
    by_artifact = {
        a: round(_mean(art_totals[a]), 1) for a in ARTIFACTS if art_totals[a]
    }
    return by_criterion, by_artifact


def _collect(
    data_dir: Path, repo: str, harness: str, skill: str, top_n: int | None = None
) -> tuple[list[tuple[str, dict, dict]], int]:
    """Return ([(model, by_criterion, by_artifact)], total_eligible).

    Reads ``<data-dir>/<model>/<harness>/<skill>/<repo>/run-summary.json`` so the
    radar plots one (agent, skill) at a time. When more models are eligible than
    ``top_n`` (a readable/validated-palette cap), only the ``top_n`` highest by
    mean task score are returned; ``total_eligible`` reports how many qualified so
    the caption can say "top N of M". A too-dense radar (7 overlapping polygons)
    is unreadable and exceeds the validated palette, so capping is both a
    legibility and an accessibility decision.
    """
    scored: list[tuple[float, str, dict, dict]] = []
    for model_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        summary = _read_json(model_dir / harness / skill / repo / RUN_SUMMARY_FILENAME)
        if summary is None:
            continue
        # Skip a run with no scored tasks (e.g. a 0/5 harness collapse): its
        # failed tasks still carry zero-valued eval_scores, but plotting a
        # collapsed all-zero polygon just adds a phantom legend entry. This
        # matches the cost-quality chart, which excludes the same runs.
        mean = summary.get("mean_task_score_excl_failed")
        if not isinstance(mean, (int, float)):
            logger.info("  excluding %s: no scored tasks", model_dir.name)
            continue
        dims = _model_dimensions(summary)
        if dims is None:
            continue
        scored.append((float(mean), model_dir.name, dims[0], dims[1]))
    total = len(scored)
    scored.sort(key=lambda r: r[0], reverse=True)
    if top_n is not None and total > top_n:
        logger.info("  %d models eligible; plotting the top %d by score", total, top_n)
        scored = scored[:top_n]
    return [(name, byc, bya) for _mean, name, byc, bya in scored], total


def _plot_one(
    ax,
    labels: tuple[str, ...],
    series: list[tuple[str, list[float]]],
    theme: dict,
    title: str,
) -> None:
    """Draw one radar panel: closed polygon per model over the given axes."""
    n = len(labels)
    # Angles for each axis, closing the loop back to the first.
    angles = [i / n * 2 * pi for i in range(n)] + [0.0]
    ax.set_theta_offset(pi / 2)  # first axis at top
    ax.set_theta_direction(-1)  # clockwise

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color=theme["ink"])
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(
        ["20", "40", "60", "80", "100"], fontsize=8, color=theme["muted"]
    )
    ax.tick_params(colors=theme["muted"])
    ax.grid(True, color=theme["grid"], linewidth=0.8)
    ax.spines["polar"].set_color(theme["grid"])
    ax.set_facecolor(theme["surface"])
    ax.set_title(title, fontsize=12, color=theme["ink"], pad=18)

    for (model, values), color in zip(series, theme["series"]):
        closed = values + values[:1]
        ax.plot(angles, closed, color=color, linewidth=2, label=model, zorder=3)
        ax.fill(angles, closed, color=color, alpha=0.12, zorder=2)


def _plot(
    models: list[tuple[str, dict, dict]],
    *,
    mode: str,
    repo: str,
    harness: str,
    skill: str,
    n_total: int,
    out_dir: Path,
) -> Path:
    """Render both radar panels side by side and save to ``out_dir``."""
    theme = _THEME[mode]
    fig, (ax_c, ax_a) = plt.subplots(
        1, 2, figsize=(14, 7), dpi=150, subplot_kw={"projection": "polar"}
    )
    fig.patch.set_facecolor(theme["surface"])

    crit_series = [(m, [by_c[c] for c in CRITERIA]) for m, by_c, _ in models]
    art_series = [(m, [by_a.get(a, 0.0) for a in ARTIFACTS]) for m, _, by_a in models]
    _plot_one(
        ax_c, CRITERION_LABELS, crit_series, theme, "By rubric criterion (% of max)"
    )
    _plot_one(ax_a, ARTIFACT_LABELS, art_series, theme, "By artifact (score 0-100)")

    # One shared legend below both panels -- identity is never color-alone.
    # The repo (dataset provenance) lives in the legend title to keep it out
    # of the chart title, which leads with the harness + skill.
    handles, labels = ax_c.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(models),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        title=repo,
    )
    for text in legend.get_texts():
        text.set_color(theme["ink"])
    if legend.get_title():
        legend.get_title().set_color(theme["muted"])
        legend.get_title().set_fontsize(9)

    harness_label = HARNESS_LABELS.get(harness, harness)
    fig.suptitle(
        f"Quality by dimension -- {harness_label} harness, /{skill}",
        fontsize=14,
        color=theme["ink"],
        y=1.02,
        x=0.5,
        ha="center",
    )
    n_shown = len(models)
    if n_shown < n_total:
        # Capped for legibility / validated palette: show the highest scorers.
        note = (
            f"Judge-scored dimensions for the top {n_shown} of {n_total} models "
            "(by mean task score) that carry the per-artifact eval breakdown; the "
            "rest are in the results table. More than a few overlapping polygons is "
            "unreadable, so the radar shows the leaders."
        )
    else:
        note = (
            f"Judge-scored dimensions for the {n_shown} of {n_total} models whose "
            "runs carry the per-artifact eval breakdown; the same view for the "
            "remaining models is coming as their eval data is backfilled."
        )
    fig.text(
        0.5,
        -0.06,
        note,
        ha="center",
        va="top",
        fontsize=8,
        color=theme["muted"],
        wrap=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    # Every harness gets a short-code suffix (cc, pi, ...) so each agent's chart
    # is self-identifying and never overwrites another's.
    mode_suffix = "-dark" if mode == "dark" else ""
    out = out_dir / f"quality-radar-{_harness_code(harness)}-{skill}{mode_suffix}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=theme["surface"])
    plt.close(fig)
    logger.info("wrote %s (%d models)", out, n_shown)
    return out


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Render per-dimension quality radar charts from eval_scores."
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--repo", default="mcp-gateway-registry")
    p.add_argument(
        "--harness",
        default="claude-code",
        help="Coding-agent folder to read (default: claude-code). Artifacts live "
        "at <model>/<harness>/<skill>/<repo>/.",
    )
    p.add_argument(
        "--skill",
        default="swe3",
        help="SWE skill folder to read: 'swe3' (default) or 'swe2'.",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--dark", action="store_true", help="Render the dark-theme variant")
    p.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Cap the radar to the N highest-scoring models (default: the "
        "validated palette size). More series than that is unreadable and exceeds "
        "the colorblind-safe palette; the full set stays in the results table.",
    )
    return p.parse_args()


def main() -> None:
    """Collect eval dimensions and render the radar chart(s)."""
    args = _parse_args()
    # Default cap is the validated palette size: more series is both unreadable
    # and beyond the colorblind-safe colors we have validated.
    top_n = args.top_n if args.top_n is not None else len(_THEME["light"]["series"])
    models, n_total = _collect(
        args.data_dir, args.repo, args.harness, args.skill, top_n=top_n
    )
    if len(models) < 1:
        raise SystemExit(
            f"no models with eval_scores under "
            f"{args.data_dir}/*/{args.harness}/{args.skill}/{args.repo}"
        )
    if len(models) > len(_THEME["light"]["series"]):
        raise SystemExit(
            f"{len(models)} models but only {len(_THEME['light']['series'])} "
            "validated series colors; lower --top-n or add validated hues."
        )
    mode = "dark" if args.dark else "light"
    logger.info("models on radar: %s", ", ".join(m for m, _, _ in models))
    _plot(
        models,
        mode=mode,
        repo=args.repo,
        harness=args.harness,
        skill=args.skill,
        n_total=n_total,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
