#!/usr/bin/env python3
"""Render a cost-vs-quality scatter (with a Pareto frontier) from run artifacts.

Reads the scored benchmark runs under ``swe-benchmark-data/`` and plots one point
per model: mean cost per task on the x-axis, mean task score (the same 0-100
scores shown in the README leaderboard) on the y-axis. Non-dominated models --
those where no other model is both cheaper and higher-scoring -- are connected by
a highlighted frontier line, so the cost/quality trade-off is read at a glance.

Each model's numbers come from its committed ``run-summary.json`` when present
(the reproducible, machine-readable per-run record written by
``summarize_run.py``), falling back to aggregating the per-task ``metrics.json``
(``total_cost_usd``) and ``eval.json`` (``task_score``) when it is not. Using the
summary means the chart plots every model in the repo -- including runs produced
on a different node whose gitignored per-task files are not present locally. A
task that scored 0 (a model failure -- missing artifacts) is an unresolved
anomaly, not a quality reading, so it is EXCLUDED from both the score and cost
means and noted on the chart, pending investigation.

Cost is HARDWARE-DERIVED, not token-priced: when a model has a throughput sweep
(``self-hosted/vllm/benchmark-output/throughput/<model>/performance-summary.json``)
its cost per task is the cheapest blended $/token there (instance $/hr / measured
tokens/sec) times this run's actual input+output tokens per task, averaged over
the non-failed tasks. Only when no performance summary exists does it fall back
to run-summary's token-priced ``total_cost_usd`` estimate.

Usage:
    uv run scripts/plot_cost_quality.py
    uv run scripts/plot_cost_quality.py --repo mcp-gateway-registry --dark
    uv run scripts/plot_cost_quality.py --data-dir ../swe-benchmark-data --out chart.png
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from token_accounting import cache_partition_for_agent, compute_total_tokens_processed

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
DEFAULT_IMAGES_DIR = _REPO_ROOT / "docs" / "images"
# Machine-readable frontier data lives apart from the rendered images.
DEFAULT_METRICS_DIR = _REPO_ROOT / "docs" / "metrics"
METRICS_FILENAME = "metrics.json"

# Short per-harness code used (with the skill) to suffix chart filenames so each
# agent+skill's chart is self-identifying and never overwrites another's
# (cost-quality-cc-swe2.png, cost-quality-pi-swe3.png). An unknown harness falls
# back to its own slug.
HARNESS_CODES = {
    "claude-code": "cc",
    "pi": "pi",
    "omp": "omp",
    "opencode": "oc",
    "kiro-cli": "kiro",
}
# Human-readable harness names for the chart title (the code is for filenames).
HARNESS_LABELS = {
    "claude-code": "Claude Code",
    "pi": "pi",
    "omp": "omp",
    "opencode": "opencode",
    "kiro-cli": "kiro-cli",
}

# Chart font sizes (points). Sized up for legibility when the chart is embedded
# in slides and social posts. The two footnotes stay at FOOTNOTE_FONTSIZE so the
# pricing-basis and excluded-task notes read as fine print, not body text.
TITLE_FONTSIZE = 19
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
POINT_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 14
FOOTNOTE_FONTSIZE = 9


def _default_output(harness: str, skill: str, dark: bool) -> Path:
    """Committed docs/images path for a (harness, skill) cost-quality chart.

    Keyed by both harness and skill (e.g. cost-quality-cc-swe3.png), because
    swe2 and swe3 differ materially in tokens/accuracy and get separate charts.
    Defaults here so the chart the README embeds stays in sync when re-run.
    (swe-benchmark-data is gitignored; docs/images is tracked.)
    """
    code = HARNESS_CODES.get(harness, harness)
    suffix = "-dark" if dark else ""
    return DEFAULT_IMAGES_DIR / f"cost-quality-{code}-{skill}{suffix}.png"


EVAL_FILENAME = "eval.json"
# The committed, machine-readable per-run summary (written by summarize_run.py).
# Preferred source: it carries the same excluded-failure means as the leaderboard
# and, unlike the gitignored per-task metrics.json/eval.json, is present for every
# model in the repo -- including runs produced on a different node. This is what
# makes the chart reproducible from committed data alone.
RUN_SUMMARY_FILENAME = "run-summary.json"
# Hardware-derived per-token cost lives in the throughput sweep's summary, one
# per model. Cost per task = (this model's cheapest blended $/token) x (this
# run's actual input+output tokens for the task) -- so cost reflects BOTH the
# measured serving economics AND the real token load of the quality run, rather
# than the token-priced estimate that run-summary.total_cost_usd carries for
# self-hosted models. See self-hosted/vllm/cost-per-task-methodology.md.
PERF_SUMMARY_DIR = (
    _REPO_ROOT / "self-hosted" / "vllm" / "benchmark-output" / "throughput"
)
PERF_SUMMARY_FILENAME = "performance-summary.json"


def _blended_cost_per_token(
    model: str, arms: dict[str, str] | None = None
) -> float | None:
    """Return the cheapest blended $/token for a model from its perf summary.

    The blended lens charges every processed token (prompt + generation) the
    same measured GPU slice; the cheapest concurrency level is the model's best
    sustainable per-token cost on its benchmarked instance. Returns None when no
    performance summary exists for the model (e.g. not swept for throughput).

    Args:
        model: The model slug, which by default also names its throughput arm.
        arms: Optional ``{model: arm-directory}`` overrides. A model swept on
            more than one instance has one summary per arm, and the arm chosen
            decides the hardware basis of that point. The bare slug is the
            CANONICAL arm; an alternative basis is a suffixed sibling (e.g.
            ``gemma-4-31b-g6e``) and must be asked for by name. Note the
            canonical arms are no longer one shared instance -- most are p5en,
            but kimi-k3 is p6-b300, minimax-m3 is p5e and minicpm5-2b is
            g6e.4xlarge -- so a cost axis built from them already mixes hardware
            bases and is directional across models (see _DEFAULT_COST_BASIS_NOTE).
    """
    arm = (arms or {}).get(model, model)
    summary = _read_json(PERF_SUMMARY_DIR / arm / PERF_SUMMARY_FILENAME)
    if summary is None:
        return None
    rates = [
        r["blended_cost_per_token_usd"]
        for r in summary.get("levels", [])
        if isinstance(r.get("blended_cost_per_token_usd"), (int, float))
    ]
    return min(rates) if rates else None


# Palette (from the dataviz skill's validated reference instance). Text always
# wears ink tokens; a coloured mark beside a label carries identity, never the
# label itself.
# Categorical hues are the first three slots of the reference palette, in both
# modes: blue for the metered-bill points, orange for the hardware-derived ones,
# aqua for the frontier line and its fill. Those three are the set that clears
# the ALL-PAIRS colourblind and normal-vision floors a scatter needs -- a fourth
# slot would put yellow beside orange and fail. Validated with the palette
# checker; light aqua sits at 2.74:1 against the surface, under the 3:1 bar, so
# it carries the required relief: every point is directly labelled and the
# frontier is named in the legend, never colour alone.
_THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e2",
        # Leader lines are wayfinding, not data. Drawn in "muted" they read as
        # dark elbows competing with the marks, so they get their own token a
        # step above the grid: visible when traced, invisible when not.
        "leader": "#c9c7c0",
        "dot": "#33322f",
        "accent": "#1baf7a",
        "bedrock": "#2a78d6",
        "self_hosted": "#eb6834",
        "label_bg": "#ffffff",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#333330",
        "leader": "#4a4a46",
        "dot": "#d7d6cf",
        "accent": "#199e70",
        "bedrock": "#3987e5",
        "self_hosted": "#d95926",
        "label_bg": "#26262410",
    },
}


@dataclass
class ModelPoint:
    """One model's aggregate for the scatter.

    Means are over the tasks the model actually completed with a non-zero
    score. Zero-score tasks (a genuine model failure -- missing artifacts) are
    an unresolved anomaly, not a quality measurement, so they are excluded from
    both the score and cost means and surfaced separately (``excluded``) pending
    investigation.
    """

    model: str
    mean_cost: float
    mean_score: float
    n_tasks: int
    n_scored: int
    excluded: list[str]
    hosting: str = (
        "self-hosted"  # "Bedrock" (metered) or "self-hosted" (hardware-derived)
    )
    # The coding agent that produced the run. Empty on a single-harness chart
    # (its title already names the harness); set by the combined chart, which
    # reports it in the frontier JSON and folds it into ``label``.
    harness: str = ""
    # Optional ready-made chart label. ``model`` stays the identity used for
    # lookups and for every emitted JSON; only the drawn text changes. The
    # combined chart uses it to name both the model and the harness that won.
    label: str = ""


def _read_json(path: Path) -> dict | None:
    """Return the parsed JSON object at ``path``, or None if absent/invalid."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _task_score(eval_data: dict | None) -> float | None:
    """Extract ``task_score`` from an eval.json object, or None when missing."""
    if not eval_data:
        return None
    score = eval_data.get("task_score")
    return float(score) if isinstance(score, (int, float)) else None


def _point_from_summary(
    model_repo_dir: Path, model: str, arms: dict[str, str] | None = None
) -> ModelPoint | None:
    """Build a ModelPoint from the committed run-summary.json, if present.

    run-summary.json already carries the leaderboard-convention means (failed
    0-score tasks excluded) and is committed for every model, so it is the
    preferred, fully reproducible source. Returns None when the file is absent
    or lacks a usable mean score, so the caller can fall back to per-task files.

    Args:
        model_repo_dir: ``<data-dir>/<model>/<harness>/<repo>`` directory.
        model: The model-slug (passed in, not derived from the path, since the
            harness level now sits between the model and repo directories).

    Returns:
        The model's aggregate, or None if no usable summary exists.
    """
    summary = _read_json(model_repo_dir / RUN_SUMMARY_FILENAME)
    if summary is None:
        return None
    score = summary.get("mean_task_score_excl_failed")
    if not isinstance(score, (int, float)):
        return None
    excluded = summary.get("failed_tasks") or []

    # Cost: prefer the hardware-derived blended figure (per-token rate from the
    # throughput sweep x this run's actual per-task tokens, averaged over the
    # non-failed tasks). Fall back to run-summary's token-priced estimate only
    # when no performance summary exists for the model.
    cost = _blended_mean_cost(summary, model, arms)
    if cost is None:
        est = summary.get("mean_cost_usd_excl_failed")
        cost = float(est) if isinstance(est, (int, float)) else 0.0

    hosting = "Bedrock" if summary.get("provider") == "bedrock" else "self-hosted"
    return ModelPoint(
        model=model,
        mean_cost=cost,
        mean_score=float(score),
        n_tasks=int(summary.get("num_tasks") or 0),
        n_scored=int(summary.get("num_scored") or 0),
        excluded=list(excluded),
        hosting=hosting,
    )


def _blended_mean_cost(
    summary: dict, model: str, arms: dict[str, str] | None = None
) -> float | None:
    """Mean blended cost per task from perf-summary per-token rate x run tokens.

    Uses the model's cheapest blended $/token (hardware-derived, from the
    throughput sweep) and this run's actual input+output tokens per task,
    averaged over the tasks that were NOT failed -- matching the score mean's
    exclusion convention. Returns None when the model has no performance summary
    (so the caller falls back to the token-priced estimate).
    """
    per_token = _blended_cost_per_token(model, arms)
    if per_token is None:
        return None
    failed = set(summary.get("failed_tasks") or [])
    costs: list[float] = []
    model_slug = summary.get("model_slug") or model
    for task in summary.get("tasks", []):
        if task.get("failed") or task.get("task") in failed:
            continue
        # Price the tokens the server ACTUALLY processed once each. The blended
        # rate was measured over every server-side token counted once, so the
        # count must match. compute_total_tokens_processed detects whether the
        # cache fields are a PARTITION of input_tokens (self-hosted vLLM: cache is
        # already inside input, so total = input + output) or ADDITIVE (Bedrock:
        # total = input + output + cache_read + cache_write). Adding the cache
        # unconditionally, as this used to, ~2x double-counted self-hosted runs
        # (issue #136).
        tokens = compute_total_tokens_processed(
            task.get("input_tokens") or 0,
            task.get("output_tokens") or 0,
            task.get("cache_read_tokens") or 0,
            task.get("cache_write_tokens") or task.get("cache_creation_tokens") or 0,
            context=f"plot_cost_quality:{model_slug}/{task.get('task')}",
            cache_partition=cache_partition_for_agent(summary.get("agent")),
        )
        if tokens > 0:
            costs.append(tokens * per_token)
    return sum(costs) / len(costs) if costs else None


def _aggregate_model(
    model_repo_dir: Path, model: str, arms: dict[str, str] | None = None
) -> ModelPoint | None:
    """Aggregate one model's cost and score under a repo directory.

    Prefers the committed ``run-summary.json`` (present for every model and
    reproducible from git). Falls back to aggregating the per-task
    ``metrics.json`` / ``eval.json`` when no summary exists (e.g. a fresh run
    not yet summarized). Tasks that scored 0 -- a genuine model failure
    (missing/empty artifacts) rather than a quality measurement -- are
    **excluded** from both means and returned in ``excluded`` for a visible
    note, pending investigation.

    Args:
        model_repo_dir: ``<data-dir>/<model>/<harness>/<repo>`` directory.
        model: The model-slug (passed in, not derived from the path).

    Returns:
        The model's aggregate, or None if it has neither a summary nor tasks.
    """
    from_summary = _point_from_summary(model_repo_dir, model, arms)
    if from_summary is not None:
        return from_summary

    costs: list[float] = []
    scores: list[float] = []
    excluded: list[str] = []
    n_tasks = 0
    for task_dir in sorted(p for p in model_repo_dir.iterdir() if p.is_dir()):
        metrics = _read_json(task_dir / METRICS_FILENAME)
        if metrics is None:
            continue
        n_tasks += 1
        score = _task_score(_read_json(task_dir / EVAL_FILENAME))
        # A 0 (or unscored) task is a model failure, not a quality signal:
        # exclude it from both means and note it separately.
        if not score:
            excluded.append(task_dir.name)
            continue
        cost = metrics.get("total_cost_usd")
        costs.append(float(cost) if isinstance(cost, (int, float)) else 0.0)
        scores.append(score)
    if n_tasks == 0:
        return None
    return ModelPoint(
        model=model,
        mean_cost=sum(costs) / len(costs) if costs else 0.0,
        mean_score=sum(scores) / len(scores) if scores else 0.0,
        n_tasks=n_tasks,
        n_scored=len(scores),
        excluded=excluded,
    )


def _collect_points(
    data_dir: Path,
    repo: str,
    harness: str,
    skill: str,
    models: list[str] | None = None,
    arms: dict[str, str] | None = None,
) -> list[ModelPoint]:
    """Collect one ModelPoint per model that has ``harness`` runs for ``repo``.

    Artifacts live at ``<data-dir>/<model>/<harness>/<repo>/``; this plots the
    results from one coding agent (harness) at a time so a model's Claude Code
    and pi runs are never blended on the same chart.

    Args:
        data_dir: The ``swe-benchmark-data`` root.
        repo: The dataset repo subfolder to aggregate (e.g. mcp-gateway-registry).
        harness: The coding-agent folder to read (e.g. ``claude-code`` or ``pi``).
        skill: The skill folder to read (e.g. ``swe3``).
        models: Restrict the chart to these model slugs. None plots every model
            with runs. A named slug that has no runs is an error rather than a
            silent omission: a frontier missing a model the caller asked for
            would be read as that model being dominated.
        arms: Optional ``{model: throughput-arm}`` overrides deciding which
            sweep prices each model -- see ``_blended_cost_per_token``.

    Returns:
        Model aggregates sorted by descending mean score.

    Raises:
        SystemExit: If no model has scorable runs for the repo under this
            harness, or if a slug named in ``models`` produced no point.
    """
    wanted = set(models or ())
    points: list[ModelPoint] = []
    for model_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if wanted and model_dir.name not in wanted:
            continue
        repo_dir = model_dir / harness / skill / repo
        if not repo_dir.is_dir():
            continue
        point = _aggregate_model(repo_dir, model_dir.name, arms)
        if point is None:
            continue
        # A model that never produced a scored task (e.g. one that could not be
        # served at a usable context window on this node) is "not viable", not a
        # $0 / 0% data point -- excluding it keeps it off the frontier. Log the
        # skip so the omission is explicit, never silent.
        if point.n_scored == 0:
            logger.warning(
                "  excluding %s: no scored tasks (not a viable run to plot)",
                point.model,
            )
            continue
        points.append(point)
    if not points:
        raise SystemExit(
            f"no scorable runs found under {data_dir} for repo '{repo}' with "
            f"harness '{harness}'. Run the benchmark and judge first."
        )
    if wanted:
        missing = sorted(wanted - {p.model for p in points})
        if missing:
            raise SystemExit(
                f"--models named {missing} but they have no scorable "
                f"{harness}/{skill}/{repo} runs. Plotting the rest would show a "
                f"frontier that silently omits them; fix the slug or drop it."
            )
    return sorted(points, key=lambda p: p.mean_score, reverse=True)


def _pareto_frontier(points: list[ModelPoint]) -> list[ModelPoint]:
    """Return the non-dominated points: cheapest-and-best trade-off curve.

    A point dominates another when it is both no more expensive and no
    lower-scoring, and strictly better on at least one axis. The frontier is the
    set of points nothing dominates, ordered by ascending cost for drawing.

    Args:
        points: All model aggregates.

    Returns:
        The frontier points, ordered by ascending mean cost.
    """
    frontier: list[ModelPoint] = []
    for candidate in points:
        dominated = any(
            other is not candidate
            and other.mean_cost <= candidate.mean_cost
            and other.mean_score >= candidate.mean_score
            and (
                other.mean_cost < candidate.mean_cost
                or other.mean_score > candidate.mean_score
            )
            for other in points
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda p: p.mean_cost)


def _point_dict(p: ModelPoint) -> dict:
    """Serialize one model point for the frontier JSON."""
    entry = {
        "model": p.model,
        "mean_score": round(p.mean_score, 2),
        "mean_cost_per_task": round(p.mean_cost, 4),
        "hosting": p.hosting,
        "n_scored": p.n_scored,
        "n_tasks": p.n_tasks,
        "completed": f"{p.n_scored}/{p.n_tasks}",
        "excluded_tasks": p.excluded,
    }
    # Only the combined chart sets a harness; omitting the key elsewhere keeps
    # the existing per-harness JSONs byte-identical.
    if p.harness:
        entry["harness"] = p.harness
    return entry


class ScopeMismatchError(RuntimeError):
    """An output file was built from a different (harness, skill, repo)."""


def _guard_scope_change(
    out_path: Path,
    *,
    harness: str,
    skill: str,
    repo: str,
    force: bool = False,
) -> None:
    """Refuse to overwrite a frontier JSON that was built from another scope.

    ``--repo`` defaults to the v1 dataset while the headline results are v2, so
    running a documented command without the flag silently rebuilds a 19-model
    v2 chart from whatever v1 runs happen to exist, and the wrong file is
    written before anyone notices. The scope is already recorded in the payload,
    so compare it with what is on disk and stop rather than clobber. Both
    directions matter: rebuilding the v1 combined chart at v2 scope is the same
    bug in reverse.

    Args:
        out_path: The JSON about to be written.
        harness: Harness slug for the run being written.
        skill: Skill folder for the run being written.
        repo: Dataset scope for the run being written.
        force: Overwrite despite a mismatch. For a deliberate re-scope.

    Raises:
        ScopeMismatchError: The file exists, records a different scope, and
            ``force`` is not set.
    """
    if force or not out_path.exists():
        return
    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return  # Unreadable or not ours: writing a fresh one is the repair.

    def _harness_of(doc: dict) -> str | None:
        """Read the harness from either shape: single-harness or combined."""
        one = doc.get("harness")
        if isinstance(one, str):
            return one
        many = doc.get("harnesses")
        return "+".join(many) if isinstance(many, list) and many else None

    was = (_harness_of(existing), existing.get("skill"), existing.get("repo"))
    now = (harness, skill, repo)
    if None in was or was == now:
        return
    raise ScopeMismatchError(
        f"{out_path.name} was built from harness={was[0]} skill={was[1]} "
        f"repo={was[2]}, but this run is harness={now[0]} skill={now[1]} "
        f"repo={now[2]}. Writing would replace it with a different dataset's "
        f"numbers. Re-run with --repo {was[2]} to regenerate what is there, or "
        f"pass --force if the re-scope is deliberate."
    )


def _write_frontier_json(
    points: list[ModelPoint],
    *,
    harness: str,
    skill: str,
    repo: str,
    out_dir: Path,
    stem: str | None = None,
    models_filter: list[str] | None = None,
    throughput_arms: dict[str, str] | None = None,
    force: bool = False,
) -> Path:
    """Emit the Pareto frontier (score vs cost/task) as machine-readable JSON.

    Reuses the SAME ``_pareto_frontier`` that draws the cost-quality chart, so
    the file and the chart never diverge. Emits three frontiers: the combined
    set (labelled as a cross-hosting view, non-authoritative on raw dollars) and
    one per hosting basis (Bedrock-only, self-hosted-only) -- the honest
    like-for-like comparisons, since a metered API bill and a hardware-derived
    figure are not comparable as raw dollars (see cost-per-task-methodology.md).

    Args:
        stem: Output filename stem. Defaults to the fleet-wide
            ``pareto-frontier-<code>-<skill>``. A filtered run MUST pass its own
            stem: a subset frontier written to the fleet-wide path would read as
            the whole fleet, and every model left out would look dominated.
        models_filter: The ``--models`` restriction, recorded in the payload so
            the file states which models it covers instead of implying all.
    """
    bedrock = [p for p in points if p.hosting == "Bedrock"]
    selfh = [p for p in points if p.hosting != "Bedrock"]
    payload = {
        "note": (
            "Pareto frontier (mean score vs mean cost/task) behind "
            f"docs/images/cost-quality-*-{skill}.png. Emitted by plot_cost_quality.py. "
            "A model is on a frontier when nothing scores at least as high for at "
            "most the cost. Use the per-hosting frontiers for cost claims; the "
            "combined frontier mixes a metered Bedrock bill with a hardware-derived "
            "self-hosted figure and is directional only (see "
            "cost-per-task-methodology.md)."
        ),
        "harness": harness,
        "skill": skill,
        "repo": repo,
        # Absent = every model with runs. Present = this file covers ONLY these,
        # so a model's absence here says nothing about whether it is dominated.
        "models_filter": sorted(models_filter) if models_filter else None,
        # Which throughput sweep priced each model, where it was not the
        # same-named one. This IS the hardware basis of those points, so it
        # belongs in the record rather than only in the command that made it.
        "throughput_arm_overrides": dict(sorted(throughput_arms.items()))
        if throughput_arms
        else None,
        "frontier_rule": "non-dominated on (max score, min cost/task)",
        "combined_frontier_cross_hosting_directional": [
            _point_dict(p) for p in _pareto_frontier(points)
        ],
        "bedrock_frontier": [_point_dict(p) for p in _pareto_frontier(bedrock)],
        "self_hosted_frontier": [_point_dict(p) for p in _pareto_frontier(selfh)],
        "all_models": [
            _point_dict(p) for p in sorted(points, key=lambda p: -p.mean_score)
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    name = stem or f"pareto-frontier-{HARNESS_CODES.get(harness, harness)}-{skill}"
    out_path = out_dir / f"{name}.json"
    _guard_scope_change(out_path, harness=harness, skill=skill, repo=repo, force=force)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", out_path)
    return out_path


def _point_name(point: ModelPoint) -> str:
    """Name a point: its caller-supplied label, else the bare model slug.

    A single-harness chart names its harness in the title, so the model alone
    reads best there. The combined chart mixes harnesses and supplies a label
    naming both.
    """
    return point.label or point.model


def _label(point: ModelPoint) -> str:
    """Build a point label; mark models whose mean excludes a failed task."""
    name = _point_name(point)
    if point.excluded:
        return f"{name}*"
    return name


def _spread(ys: list[float], step: float) -> list[float]:
    """Push a sorted-ascending list apart to >= ``step`` spacing, keeping center.

    A single bottom-up pass raises each value to clear the one below it, which
    drifts the whole group upward; subtracting the net mean shift re-centers it
    on the original cluster. Inputs must be sorted ascending.
    """
    out = list(ys)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + step)
    drift = sum(out) / len(out) - sum(ys) / len(ys)
    return [y - drift for y in out]


def _label_sides(
    ax, fig, points: list[ModelPoint], offsets: dict[int, float], label_chars: int
) -> dict[int, str]:
    """Choose which side of its dot each label sits on.

    ``_label_offsets`` only keeps labels from overlapping EACH OTHER; a label
    can still be drawn straight across another model's marker, which reads as
    if it belonged to that dot. Any label whose text would run over another
    point is flipped to the left of its own dot instead.

    Args:
        ax: The axes (already drawn, so transforms are valid).
        fig: The figure (for the pixel <-> point conversion).
        points: All model aggregates.
        offsets: The vertical offsets from ``_label_offsets``, in points.
        label_chars: Typical label length, used to estimate text width.

    Returns:
        ``{id(point): "left" | "right"}``.
    """
    to_px = ax.transData.transform
    px = {id(p): to_px((p.mean_cost, p.mean_score)) for p in points}
    text_w = 12 + POINT_LABEL_FONTSIZE * 0.6 * label_chars
    half_line = POINT_LABEL_FONTSIZE * 1.35 * fig.dpi / 72.0 * 0.5
    # Flipping left is only an option while the text still fits inside the axes;
    # past that it would run out over the y-axis instead.
    left_edge_px = ax.transAxes.transform((0.0, 0.0))[0]
    sides: dict[int, str] = {}
    for point in points:
        x_px, y_px = px[id(point)]
        label_y = y_px + offsets[id(point)] * fig.dpi / 72.0
        collides = any(
            other is not point
            and x_px < px[id(other)][0] <= x_px + text_w
            and abs(px[id(other)][1] - label_y) < half_line
            for other in points
        )
        room_on_left = x_px - text_w > left_edge_px
        sides[id(point)] = "left" if collides and room_on_left else "right"
    return sides


def _column_beside(
    group: list[ModelPoint],
    points: list[ModelPoint],
    px: dict[int, tuple[float, float]],
    widths: dict[int, float],
    line_px: float,
    y0_px: float,
    y1_px: float,
    to_data: Callable[[tuple[float, float]], tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Place a bunched group's labels in a clear column just right of the group.

    The column is anchored a little to the right of the group's rightmost dot,
    left-aligned, with each label stacked near its own dot's height (one line
    apart, slid inside the axes). Returns the data-coordinate anchor for every
    member, or an empty dict when the band to the right is not clear of other
    dots -- in which case the caller leaves the group where it was.
    """
    member_ids = {id(p) for p in group}
    anchor_px = max(px[id(p)][0] for p in group) + 16
    band_width = max(widths[id(p)] for p in group)
    top_px = max(px[id(p)][1] for p in group) + line_px
    bottom_px = min(px[id(p)][1] for p in group) - line_px
    for other in points:
        if id(other) in member_ids:
            continue
        ox, oy = px[id(other)]
        if anchor_px - 8 <= ox <= anchor_px + band_width and bottom_px <= oy <= top_px:
            return {}  # a dot sits in the target band; do not route over it

    ordered = sorted(group, key=lambda p: px[id(p)][1])
    ys = _spread([px[id(p)][1] for p in ordered], line_px * 1.4)
    shift = 0.0
    if min(ys) - line_px / 2 < y0_px:
        shift = y0_px - (min(ys) - line_px / 2)
    elif max(ys) + line_px / 2 > y1_px:
        shift = y1_px - (max(ys) + line_px / 2)
    anchor_x_data = to_data((anchor_px, 0.0))[0]
    return {
        id(point): (anchor_x_data, to_data((0.0, y_px + shift))[1])
        for point, y_px in zip(ordered, ys)
    }


def _reroute_crowded_clusters(
    ax,
    fig,
    points: list[ModelPoint],
    label_weight: str,
) -> dict[int, tuple[float, float]]:
    """Move the pile of labels jammed at the left axis into open space beside it.

    ``_label_offsets`` keeps labels from overlapping, but when several of the
    cheapest models pile into the left margin at nearly the same x, all it can do
    is stack their labels straight down the dot column against the y-axis -- a
    crowded ladder of near-vertical leader lines, with no room to escape sideways
    (there is no plot left of the axis). This finds that leftmost pile and, when
    the band to its right is clear, relocates the whole group's labels to a single
    anchor column in that band, each near its own dot's height, so their leaders
    run out horizontally instead of stacking.

    Only the leftmost cluster is touched, and only when it is genuinely jammed at
    the edge: three or more dots inside the left twentieth of the axis, packed
    closer than a label-height apart in x. Every other cluster has open plot above
    it and is left to the ordinary vertical spread, so well-spread charts and the
    denser mid-chart groups are untouched.

    Returns ``{id(point): (anchor_cost, label_score)}`` in DATA coordinates for
    each relocated label; points absent from the dict keep their offset placement.
    """
    to_px = ax.transData.transform
    to_data = ax.transData.inverted().transform
    px = {id(p): to_px((p.mean_cost, p.mean_score)) for p in points}
    widths = _text_widths_px(ax, fig, points, label_weight)
    line_px = POINT_LABEL_FONTSIZE * 1.35 * fig.dpi / 72.0
    x0_px = ax.transAxes.transform((0.0, 0.0))[0]
    x1_px = ax.transAxes.transform((1.0, 0.0))[0]
    y0_px = ax.transAxes.transform((0.0, 0.0))[1]
    y1_px = ax.transAxes.transform((0.0, 1.0))[1]

    # Walk from the leftmost dot, adding neighbours while the x-gap stays under a
    # label-height (dots that close together are practically stacked). Stop at the
    # first real gap: that ends the left pile.
    order = sorted(points, key=lambda p: px[id(p)][0])
    pile: list[ModelPoint] = []
    for point in order:
        if pile and px[id(point)][0] - px[id(pile[-1])][0] > line_px * 1.2:
            break
        pile.append(point)

    # Reroute only a real pile jammed at the axis: three or more dots whose
    # leftmost sits inside the left twentieth of the plot.
    jammed_at_edge = (
        px[id(pile[0])][0] < x0_px + 0.05 * (x1_px - x0_px) if pile else False
    )
    if len(pile) < 3 or not jammed_at_edge:
        return {}
    return _column_beside(pile, points, px, widths, line_px, y0_px, y1_px, to_data)


def _text_widths_px(ax, fig, points: list[ModelPoint], weight: str) -> dict[int, float]:
    """Return each label's real rendered width in pixels, keyed by ``id(point)``.

    Measured rather than estimated from a character count: label lengths here vary
    by more than 2x (``glm-5.3`` against ``nemotron-ultra-550b*``), and a single
    average width both over-clusters the short labels and, worse, under-detects
    collisions between the long ones. Each probe artist is removed immediately, so
    nothing is added to the figure.
    """
    renderer = fig.canvas.get_renderer()
    widths: dict[int, float] = {}
    for point in points:
        probe = ax.text(
            0, 0, _label(point), fontsize=POINT_LABEL_FONTSIZE, fontweight=weight
        )
        widths[id(point)] = probe.get_window_extent(renderer=renderer).width
        probe.remove()
    return widths


def _label_offsets(
    ax,
    fig,
    points: list[ModelPoint],
    label_chars: int = 22,
    label_weight: str = "normal",
    centre_moved: bool = True,
) -> dict[int, float]:
    """Return each label's vertical offset (in points) to avoid overlaps.

    Labels sit to the right of their dot at the dot's y-level. Two labels collide
    when their text boxes would overlap in BOTH axes -- the left one's text runs
    far enough right to reach the other's, and they sit within about a line of
    each other in y. Colliding points are grouped into clusters and spread apart
    vertically, centered on the cluster; every isolated label keeps a 0 offset
    (stays pinned to its dot, no leader line). Offsets are returned in display
    points, keyed by ``id(point)``, so the caller can pass them straight to
    ``annotate`` and decide a leader line is needed exactly when the offset is
    non-zero.

    Clustering is iterated to a fixed point, and both reasons are load-bearing:

    1. The first pass groups on DOT positions, but spreading moves labels, so a
       label pushed away from its own cluster can land on the row of a label it
       did not originally collide with.
    2. A label that moved gets CENTRED over its dot by the caller (so its leader
       line is vertical), which widens it leftward by half its text -- into space
       the right-of-dot geometry said was free. This is how ``minimax-m2.5`` came
       to sit on ``qwen3-coder-480b*``: the two dots are far enough apart that
       neither's right-side box reached the other, but once both were centred
       their boxes met.

    So each round re-tests every pair using the box each label will ACTUALLY be
    drawn in given the current offsets, merges whatever now overlaps, and spreads
    again until a round changes nothing.

    Args:
        ax: The axes (already drawn, so transforms are valid).
        fig: The figure (for DPI when converting pixels <-> points).
        points: All model aggregates.
        label_chars: Fallback label length in characters, used only if a label's
            width cannot be measured.
        label_weight: Font weight the labels will be drawn at, so the measured
            widths match what actually gets rendered.
        centre_moved: Whether the caller centres a displaced label over its dot.
            Must match the caller's ``vertical_leaders and leader_lines``, or the
            boxes reasoned about here are not the boxes drawn.

    Returns:
        ``{id(point): dy_in_points}`` -- 0.0 for labels that did not move.
    """
    to_px = ax.transData.transform
    line_px = POINT_LABEL_FONTSIZE * 1.35 * fig.dpi / 72.0  # one label's height in px
    fallback_w = POINT_LABEL_FONTSIZE * 0.6 * label_chars
    widths = _text_widths_px(ax, fig, points, label_weight)
    px = {id(p): to_px((p.mean_cost, p.mean_score)) for p in points}
    # Vertical clearance one label needs from another. Kept above a bare line
    # height so descenders and the leader-line elbow have room.
    y_touch_px = line_px * 1.6
    # Spread spacing between labels in a cluster. Kept just above y_touch so a
    # displaced label clears its neighbour without being flung far from its dot:
    # a bigger step only lengthens the leader lines without buying legibility.
    step_px = line_px * 1.8

    x0_px, x1_px = (
        ax.transAxes.transform((0.0, 0.0))[0],
        ax.transAxes.transform((1.0, 0.0))[0],
    )
    y0_px, y1_px = (
        ax.transAxes.transform((0.0, 0.0))[1],
        ax.transAxes.transform((0.0, 1.0))[1],
    )

    def x_span(point: ModelPoint, moved: bool) -> tuple[float, float]:
        """The horizontal pixel extent this label will occupy as drawn.

        Two placements, matching the caller exactly: a label that has not moved
        sits ~12px right of its dot, and one that HAS moved is centred over the
        dot instead (unless centring would push it outside the axes, in which case
        the caller leaves it right-of-dot). Widths are per-label and measured, so a
        short slug is not clustered with a distant point and a long one is not
        missed.
        """
        x_px = px[id(point)][0]
        width = widths.get(id(point), fallback_w)
        half = width / 2
        if centre_moved and moved and x_px - half > x0_px and x_px + half < x1_px:
            return (x_px - half, x_px + half)
        return (x_px + 12, x_px + 12 + width)

    def x_overlaps(a: ModelPoint, b: ModelPoint, offsets_px: dict[int, float]) -> bool:
        """True if the two labels would share horizontal space as drawn."""
        a0, a1 = x_span(a, abs(offsets_px.get(id(a), 0.0)) > 1e-6)
        b0, b1 = x_span(b, abs(offsets_px.get(id(b), 0.0)) > 1e-6)
        return a0 < b1 and b0 < a1

    parent = {id(p): id(p) for p in points}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> bool:
        """Merge two clusters; True if they were not already one."""
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    def spread_all() -> dict[int, float]:
        """Spread every multi-member cluster, returning offsets in pixels."""
        clusters: dict[int, list[ModelPoint]] = {}
        for point in points:
            clusters.setdefault(find(id(point)), []).append(point)
        out = {id(p): 0.0 for p in points}
        for members in clusters.values():
            if len(members) < 2:
                continue  # isolated label: no move, no line
            members.sort(key=lambda p: px[id(p)][1])  # by pixel-y, ascending
            # A big cluster spread at the full step can be taller than the plot,
            # which pushes its end labels off the axes entirely -- into the title
            # or the footnotes, where they read as stray text and not as labels.
            # So measure the run that _spread actually produced (it keeps the
            # original spacing where it already exceeds the step, so its extent
            # can be larger than (n-1)*step) and, if it does not fit, replace it
            # with an evenly spaced run sized to the band. Tighter spacing beats a
            # label leaving the plot.
            band = y1_px - y0_px
            spread_px = _spread([px[id(p)][1] for p in members], step_px)
            if (max(spread_px) - min(spread_px)) + line_px > band:
                step = max(line_px, (band - line_px) / (len(members) - 1))
                mid = (y0_px + y1_px) / 2
                first = mid - (len(members) - 1) * step / 2
                spread_px = [first + i * step for i in range(len(members))]
            # Slide the run inside the band. Both edges are checked, and after the
            # rebuild above the run is guaranteed to fit, so one shift suffices.
            shift = 0.0
            if min(spread_px) - line_px / 2 < y0_px:
                shift = y0_px - (min(spread_px) - line_px / 2)
            elif max(spread_px) + line_px / 2 > y1_px:
                shift = y1_px - (max(spread_px) + line_px / 2)
            for point, new_y in zip(members, spread_px):
                out[id(point)] = new_y + shift - px[id(point)][1]
        return out

    # Seed the clusters from the dot positions, then iterate: spread, re-test at
    # the resulting positions, merge whatever now collides, spread again. Bounded
    # so a pathological layout cannot loop forever -- each round can only merge
    # clusters, so it converges in at most len(points) rounds anyway.
    unmoved: dict[int, float] = {id(p): 0.0 for p in points}
    for a, b in itertools.combinations(points, 2):
        if (
            x_overlaps(a, b, unmoved)
            and abs(px[id(a)][1] - px[id(b)][1]) < line_px * 2.8
        ):
            union(id(a), id(b))

    offsets_px = spread_all()
    for _ in range(len(points)):
        merged = False
        for a, b in itertools.combinations(points, 2):
            if not x_overlaps(a, b, offsets_px):
                continue
            ay = px[id(a)][1] + offsets_px[id(a)]
            by = px[id(b)][1] + offsets_px[id(b)]
            if abs(ay - by) < y_touch_px and union(id(a), id(b)):
                merged = True
        if not merged:
            break
        offsets_px = spread_all()

    # display-y grows downward in some backends; transData is bottom-up, so a
    # higher pixel value = higher on screen. Convert the deltas to points.
    return {key: dy * 72.0 / fig.dpi for key, dy in offsets_px.items()}


def _parse_arms(specs: list[str] | None) -> dict[str, str]:
    """Parse ``MODEL=ARM`` overrides into a mapping, validating each arm exists.

    A typo'd arm would fall through to "no performance summary", which silently
    reprices that model with the token-priced fallback instead of the hardware
    basis every other point uses -- a wrong dot rather than a missing one. So a
    nonexistent arm is a hard error.

    Raises:
        SystemExit: On a malformed spec or an arm with no performance summary.
    """
    arms: dict[str, str] = {}
    for spec in specs or ():
        model, sep, arm = spec.partition("=")
        if not sep or not model.strip() or not arm.strip():
            raise SystemExit(f"--throughput-arm expects MODEL=ARM, got {spec!r}")
        model, arm = model.strip(), arm.strip()
        if not (PERF_SUMMARY_DIR / arm / PERF_SUMMARY_FILENAME).is_file():
            raise SystemExit(
                f"--throughput-arm {model}={arm}: no {PERF_SUMMARY_FILENAME} under "
                f"{PERF_SUMMARY_DIR / arm}. Run the throughput sweep for that arm "
                f"first; falling back would price it on a different basis."
            )
        arms[model] = arm
    return arms


def _escape_dollars(text: str) -> str:
    """Escape ``$`` so matplotlib renders a price, not a MathText formula.

    A note naming two rates ("p5en $27.72/hr, g6e $4.533/hr") contains a PAIR of
    dollar signs, which matplotlib reads as a MathText region: it italicizes the
    span and drops both signs, so the rates the note exists to state vanish.
    Escaping every unescaped ``$`` prints the money.
    """
    return re.sub(r"(?<!\\)\$", r"\\$", text)


# Each self-hosted point is priced from its OWN throughput sweep (the canonical
# arm, the bare model slug), on whichever instance that model was actually
# served on. That was one shared basis while every sweep ran on p5en, but it no
# longer is: kimi-k3 was swept on p6-b300.48xlarge, minimax-m3 on p5e.48xlarge
# and minicpm5-2b on g6e.4xlarge. Even among the p5en arms the hourly rate
# varies from $3.465 to $27.72 with TP proration. So the note must NOT promise a
# single fleet-wide rate. A chart that deliberately prices on a non-canonical
# arm (--throughput-arm) should still pass its own note via --cost-basis-note.
_DEFAULT_COST_BASIS_NOTE = (
    "Self-hosted cost basis: each point is priced from that model's own "
    "throughput sweep, at the rate of the instance it was served on (3-year EC2 "
    "Instance Savings Plan, prorated by TP for a partial-box run) -- see "
    "self-hosted/vllm/pricing.json. Most models were swept on p5en.48xlarge, but "
    "not all, and the prorated rate differs between them, so self-hosted dollars "
    "are directional across rows rather than one common basis."
)


def _plot(
    points: list[ModelPoint],
    frontier: list[ModelPoint],
    *,
    mode: str,
    title: str,
    cost_label: str,
    output: Path,
    frontier_label: str = "Cost/quality frontier",
    cost_basis_note: str = _DEFAULT_COST_BASIS_NOTE,
    leader_lines: bool = True,
    label_weight: str = "normal",
    marker_for: Callable[[ModelPoint], str] | None = None,
    color_for: Callable[[ModelPoint], str] | None = None,
    accent_color: str | None = None,
    extra_legend: list | None = None,
    label_backing: bool = False,
    log_x: bool = False,
    avoid_markers: bool = True,
    vertical_leaders: bool = True,
) -> None:
    """Render the scatter with its frontier and save to ``output``.

    Args:
        points: All model aggregates.
        frontier: The non-dominated subset (ascending cost).
        mode: "light" or "dark" theme.
        title: Chart title.
        cost_label: X-axis label (cost provenance is caller's responsibility).
        output: Destination image path.
        frontier_label: Legend text for the frontier line.
        cost_basis_note: Fine-print note naming the cost basis.
        leader_lines: Draw a thin line from a displaced label back to its dot.
            Off for charts whose labels are self-identifying enough not to need
            them.
        label_weight: Font weight for the point labels. Regular by default --
            bold at label length reads as emphasis on every point at once,
            which is emphasis on nothing.
        marker_for: Optional per-point marker chooser; defaults to a circle for
            every point. The combined chart uses it to encode the harness.
        color_for: Optional per-point colour chooser. Without it a point is the
            warm accent when it sits on the frontier and a recessive neutral
            otherwise -- i.e. colour encodes rank. Supplying it moves colour
            onto the entity (the harness), leaving the frontier to be read from
            the line that connects its points.
        accent_color: Override the accent -- the frontier line AND the tint
            under it. They are one colour by design: the fill is the line at
            low alpha, which is what makes the shaded region read as belonging
            to the frontier rather than as a second, unexplained object.
        extra_legend: Optional extra legend handles, e.g. the marker key that
            says which shape is which harness.
        label_backing: Draw a surface-coloured plate behind each label. Off by
            default: the plate reads as a UI chip around every model name, which
            is chrome the chart does not need, and it hides label collisions
            instead of exposing them. Turn it on only where labels must sit over
            a dense frontier fill and would otherwise be unreadable.
        log_x: Put cost on a log scale. Cost spans nearly two orders of
            magnitude, so a linear axis crushes the cheapest models into the
            left margin, and their labels cannot sit beside their own dots.
        avoid_markers: Flip a label to the left of its dot when drawing it to
            the right would run the text across another model's marker (only
            while the text still fits inside the axes).
        vertical_leaders: Centre a displaced label over its own dot so the
            leader line runs vertically instead of diagonally -- a tick up to
            the label rather than a wire across the plot. Falls back to side
            placement when a centred label would overhang the axes.
    """
    theme = dict(_THEME[mode])
    if accent_color:
        theme["accent"] = accent_color
    fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    if log_x:
        ax.set_xscale("log")

    # Frontier: a recessive accent line under the marks, filled to the baseline.
    if len(frontier) >= 2:
        fx = [p.mean_cost for p in frontier]
        fy = [p.mean_score for p in frontier]
        ax.plot(
            fx,
            fy,
            color=theme["accent"],
            linewidth=2,
            linestyle="--",
            zorder=2,
            label=frontier_label,
        )
        # Gradient fill under frontier: strongest near the line, fading to
        # transparent at the bottom. Uses imshow with a vertical alpha gradient
        # clipped to the frontier polygon.
        import numpy as np
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MplPath
        from matplotlib.colors import to_rgba

        y_bottom = min(p.mean_score for p in points) - 5
        # Build polygon: frontier line top, then straight down to bottom
        poly_x = fx + [fx[-1], fx[0]]
        poly_y = fy + [y_bottom, y_bottom]
        poly_verts = list(zip(poly_x, poly_y))
        if log_x:
            # imshow maps its extent linearly, so a log axis needs a plain fill.
            ax.fill_between(
                fx, fy, y_bottom, color=theme["accent"], alpha=0.08, zorder=1
            )
        poly_path = MplPath(poly_verts + [poly_verts[0]], closed=True)
        patch = PathPatch(poly_path, facecolor="none", edgecolor="none")
        ax.add_patch(patch)

        # Render gradient image clipped to the polygon
        x_min, x_max = min(fx), max(fx)
        y_min, y_max = y_bottom, max(fy)
        gradient = np.linspace(1, 0, 256).reshape(256, 1)
        accent_rgba = to_rgba(theme["accent"])
        ax.imshow(
            gradient,
            extent=[x_min, x_max, y_min, y_max],
            origin="upper",
            aspect="auto",
            cmap=None,
            vmin=0,
            vmax=1,
            alpha=0.12,
            zorder=1,
            interpolation="bicubic",
        )
        # Apply color by using a custom colormap from accent to transparent
        from matplotlib.colors import LinearSegmentedColormap

        accent_cmap = LinearSegmentedColormap.from_list(
            "accent_fade",
            [(*accent_rgba[:3], 0.15), (*accent_rgba[:3], 0.0)],
        )
        # Clear the plain imshow and redo with the colormap
        ax.images[-1].remove()
        im = ax.imshow(
            gradient,
            extent=[x_min, x_max, y_min, y_max],
            origin="upper",
            aspect="auto",
            cmap=accent_cmap,
            vmin=0,
            vmax=1,
            zorder=1,
            interpolation="bicubic",
        )
        im.set_clip_path(patch)
        if log_x:
            im.remove()
            patch.remove()

    # Dots now; labels later (after the limits are final) so the declutter pass
    # can measure real text height. Frontier points are already accent from the
    # frontier line; the rest are a recessive dark neutral.
    frontier_ids = {id(p) for p in frontier}
    for point in points:
        on_frontier = id(point) in frontier_ids
        ax.scatter(
            point.mean_cost,
            point.mean_score,
            s=140,
            marker=marker_for(point) if marker_for else "o",
            color=(
                color_for(point)
                if color_for
                else (theme["accent"] if on_frontier else theme["dot"])
            ),
            edgecolors=theme["surface"],
            linewidths=2,
            zorder=3,
        )

    ax.set_xlabel(
        cost_label, fontsize=AXIS_LABEL_FONTSIZE, color=theme["ink"], labelpad=10
    )
    ax.set_ylabel(
        "Mean task score (0-100)",
        fontsize=AXIS_LABEL_FONTSIZE,
        color=theme["ink"],
        labelpad=10,
    )
    ax.set_title(title, fontsize=TITLE_FONTSIZE, color=theme["ink"], pad=16, loc="left")

    ax.grid(True, color=theme["grid"], linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=TICK_FONTSIZE)
    if log_x:
        # A log axis defaults to decade ticks (10^0, 10^1), which is useless on
        # a chart whose whole point is the dollar figure. Label the 1-2-5 steps
        # in plain dollars instead.
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"${v:g}" if v >= 1 else f"${v:.2f}")
        )
        ax.xaxis.set_minor_formatter(NullFormatter())

    # Headroom so labels near the axis edges do not clip.
    xs = [p.mean_cost for p in points]
    ys = [p.mean_score for p in points]
    xpad = max((max(xs) - min(xs)) * 0.12, 1.0)
    ypad = max((max(ys) - min(ys)) * 0.12, 3.0)
    if log_x:
        ax.set_xlim(min(xs) / 1.5, max(xs) * 2.6)
    else:
        ax.set_xlim(max(0.0, min(xs) - xpad), max(xs) + xpad * 2.2)
    ax.set_ylim(max(0.0, min(ys) - ypad), min(100.0, max(ys) + ypad))

    # Labels last, after the limits are final. Only labels that actually collide
    # (close in BOTH x and y) are spread apart in y, and only those get a leader
    # line back to the dot -- an isolated point keeps the plain right-of-dot
    # offset with no line. A draw() fixes the data<->pixel scale so a label's
    # rendered size can be expressed in data units.
    fig.canvas.draw()
    # Size the collision band to the longest label actually drawn, so the wider
    # labels of a combined chart are spread rather than left overlapping.
    longest = max((len(_label(p)) for p in points), default=22)
    label_chars = max(22, longest)
    dy_by_point = _label_offsets(
        ax,
        fig,
        points,
        label_chars=label_chars,
        label_weight=label_weight,
        # Must mirror the `centred` condition below, or the placer avoids
        # collisions between boxes that are not the ones drawn.
        centre_moved=vertical_leaders and leader_lines,
    )
    sides = (
        _label_sides(ax, fig, points, dy_by_point, label_chars)
        if avoid_markers
        else {id(p): "right" for p in points}
    )
    # A pile-up of dots at one x (the cheap models in the left margin) cannot be
    # fixed by vertical spread alone -- the labels just stack down the dot column.
    # Route such a group's labels into the clear band beside it instead; a no-op
    # when nothing is that crowded.
    reroute = _reroute_crowded_clusters(ax, fig, points, label_weight)
    # A centred label spans half its width each side of the dot, so it can only
    # be centred while both halves stay inside the axes.
    x0_px, x1_px = (
        ax.transAxes.transform((0.0, 0.0))[0],
        ax.transAxes.transform((1.0, 0.0))[0],
    )
    half_w_px = (POINT_LABEL_FONTSIZE * 0.6 * label_chars) / 2
    for point in points:
        if id(point) in reroute:
            anchor_x, label_y = reroute[id(point)]
            ax.annotate(
                _label(point),
                (point.mean_cost, point.mean_score),
                textcoords="data",
                xytext=(anchor_x, label_y),
                fontsize=POINT_LABEL_FONTSIZE,
                fontweight=label_weight,
                color=theme["ink"],
                ha="left",
                va="center",
                zorder=4,
                bbox=(
                    {
                        "boxstyle": "round,pad=0.3",
                        "facecolor": theme["surface"],
                        "edgecolor": "none",
                        "alpha": 0.85,
                    }
                    if label_backing
                    else None
                ),
                arrowprops={
                    "arrowstyle": "-",
                    "color": theme["leader"],
                    "linewidth": 0.8,
                    "shrinkA": 2,
                    "shrinkB": 3,
                    # Horizontal into the label (angleA=0), vertical off the dot
                    # (angleB=90): an L-shaped callout reaching out to the column.
                    "connectionstyle": "angle,angleA=0,angleB=90,rad=0",
                },
            )
            continue
        dy_pts = dy_by_point[id(point)]
        moved = abs(dy_pts) > 1e-6
        on_left = sides[id(point)] == "left"
        dot_x_px = ax.transData.transform((point.mean_cost, point.mean_score))[0]
        centred = (
            vertical_leaders
            and moved
            and leader_lines
            and dot_x_px - half_w_px > x0_px
            and dot_x_px + half_w_px < x1_px
        )
        ax.annotate(
            _label(point),
            (point.mean_cost, point.mean_score),
            textcoords="offset points",
            xytext=(0 if centred else (-12 if on_left else 12), dy_pts),
            fontsize=POINT_LABEL_FONTSIZE,
            fontweight=label_weight,
            color=theme["ink"],
            ha="center" if centred else ("right" if on_left else "left"),
            va="center",
            zorder=4,
            bbox=(
                {
                    "boxstyle": "round,pad=0.3",
                    "facecolor": theme["surface"],
                    "edgecolor": "none",
                    "alpha": 0.85,
                }
                if label_backing
                else None
            ),
            arrowprops=(
                {
                    "arrowstyle": "-",
                    "color": theme["leader"],
                    "linewidth": 0.8,
                    "shrinkA": 2,
                    "shrinkB": 3,
                    # Right-angle elbow so the segment meeting the label is
                    # horizontal (a clean callout tick into the text) rather than
                    # a slanted diagonal. angleA is the text (xytext) end -> 0 =
                    # horizontal; angleB is the dot (xy) end -> 90 = vertical.
                    "connectionstyle": "angle,angleA=0,angleB=90,rad=0",
                }
                if moved and leader_lines
                else None
            ),
        )

    handles, _ = ax.get_legend_handles_labels()
    handles.extend(extra_legend or [])
    if handles:
        legend = ax.legend(
            handles=handles, loc="lower right", frameon=False, fontsize=LEGEND_FONTSIZE
        )
        for text in legend.get_texts():
            text.set_color(theme["muted"])

    # Pricing-basis note, shown prominently so no one misreads the dollars. For
    # self-hosted/mixed charts this states the g6e/p5en GPU rate basis; for a
    # kiro-cli chart (all points priced in Kiro credits) the caller passes the
    # credit-basis note instead. See _cost_basis_note / cost-per-task-methodology.md.
    fig.text(
        0.5,
        -0.02,
        _escape_dollars(cost_basis_note),
        ha="center",
        va="top",
        fontsize=FOOTNOTE_FONTSIZE,
        color=theme["muted"],
        wrap=True,
    )

    # Note any excluded failed tasks so the chart is self-explaining: a 0-score
    # (missing-artifact) task is a model failure, not a quality reading, so it is
    # left out of the means, pending investigation.
    excl_notes = [
        f"{_point_name(p)}: {', '.join(p.excluded)}" for p in points if p.excluded
    ]
    if excl_notes:
        note = (
            "* Mean excludes a failed task (0 score / missing artifacts), pending "
            "investigation -- " + "; ".join(excl_notes)
        )
        fig.text(
            0.5,
            -0.055,
            note,
            ha="center",
            va="top",
            fontsize=FOOTNOTE_FONTSIZE,
            color=theme["muted"],
            wrap=True,
        )

    fig.tight_layout()

    # A compact roll-call of the frontier models in the right margin, cheapest
    # first, so "which models win" reads at a glance without tracing the line
    # back to each dot. Placed after tight_layout in axes coordinates just past
    # the right spine; bbox_inches="tight" below grows the saved canvas to
    # include it (and the surface facecolor fills the new strip).
    if frontier:
        ordered = sorted(frontier, key=lambda p: p.mean_cost)
        # A fixed-width table of the frontier models, cheapest first, with a
        # delta column showing what each step up the frontier buys: the quality
        # gained and the extra cost per task over the row below it. Left-anchored
        # inside the plot above the lower-right legend, so it adds nothing to the
        # canvas width (a right margin would shrink the plot).
        header = f"{'':<15}{'Quality':>7}{'$/task':>8}  Δ (qual / cost)"
        table_lines = [header]
        prev = None
        for p in ordered:
            q = f"{p.mean_score:.1f}"
            c = f"${p.mean_cost:.2f}"
            if prev is None:
                delta = f"{'--':>6}"
            else:
                # Delta from the displayed (rounded) values, so a reader who
                # subtracts the two columns gets exactly the number shown here.
                dq = f"{round(p.mean_score, 1) - round(prev.mean_score, 1):+.1f}"
                dc = f"+${round(p.mean_cost, 2) - round(prev.mean_cost, 2):.2f}"
                delta = f"{dq:>6} / {dc}"
            table_lines.append(f"{_point_name(p):<15}{q:>7}{c:>8}  {delta}")
            prev = p
        ax.text(
            0.60,
            0.47,
            "Quality and cost per task",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=LEGEND_FONTSIZE,
            fontweight="bold",
            color=theme["accent"],
        )
        ax.text(
            0.60,
            0.20,
            _escape_dollars("\n".join(table_lines)),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=LEGEND_FONTSIZE - 2,
            family="monospace",
            color=theme["ink"],
            linespacing=1.7,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "wrote %s (%d models, %d on frontier)", output, len(points), len(frontier)
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot a cost-vs-quality scatter with a Pareto frontier from "
        "benchmark run artifacts.",
        epilog="Example:\n"
        "  uv run scripts/plot_cost_quality.py --repo mcp-gateway-registry\n"
        "  uv run scripts/plot_cost_quality.py --dark --out chart-dark.png",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"swe-benchmark-data root (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--repo",
        default="mcp-gateway-registry",
        help="Dataset repo subfolder to aggregate (default: mcp-gateway-registry)",
    )
    parser.add_argument(
        "--harness",
        default="claude-code",
        help="Coding-agent folder to read: 'claude-code' (default) or 'pi'. "
        "Artifacts live at <model>/<harness>/<skill>/<repo>/.",
    )
    parser.add_argument(
        "--skill",
        default="swe3",
        help="SWE skill folder to read: 'swe3' (default) or 'swe2'. swe2 and swe3 "
        "get separate charts (they differ in tokens/accuracy).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Restrict the chart to these model slugs (default: every model with "
        "runs). A named slug with no scorable runs is an error, not a silent skip.",
    )
    parser.add_argument(
        "--throughput-arm",
        action="append",
        default=None,
        metavar="MODEL=ARM",
        help="Price MODEL from throughput arm ARM instead of the same-named "
        "sweep, e.g. qwen3.8-27b=qwen3.8-27b-g6e. Repeatable. The canonical arm "
        "(bare slug) is the p5en sweep; use this to chart the g6e basis instead.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: docs/images/cost-quality-<code>.png, "
        "where <code> is the harness code, e.g. cc or pi; -dark suffix in dark mode)",
    )
    parser.add_argument(
        "--dark", action="store_true", help="Render the dark-mode theme"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite outputs even when the existing Pareto-frontier JSON was "
            "built from a different --repo / --harness / --skill. Without this, a "
            "scope mismatch is an error rather than a silent replacement."
        ),
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help="Where to write the Pareto-frontier JSON (default: docs/metrics/).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Override the chart title",
    )
    parser.add_argument(
        "--cost-basis-note",
        default=None,
        help="Override the fine-print cost-basis note. Use it when --models "
        "narrows the chart to one instance family, so the note names only the "
        "rate that actually priced the points shown.",
    )
    parser.add_argument(
        "--cost-label",
        # Basis-neutral: the chart mixes hardware-derived costs (self-hosted:
        # instance $/hr / measured tokens/sec) with real metered Bedrock bills
        # (Anthropic models). Naming one basis in the axis label misrepresents
        # the other, so the axis states only the quantity; provenance lives in
        # the caption/footnotes (see the README leaderboard notes).
        default=None,
        help="X-axis label; make cost provenance explicit. Defaults to a "
        "basis-appropriate label per harness (kiro-cli => Kiro credits).",
    )
    return parser.parse_args()


def main() -> None:
    """Aggregate the artifacts and render the cost-quality chart."""
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"data dir not found: {data_dir}")

    mode = "dark" if args.dark else "light"
    # The default paths are keyed by (harness, skill) only, so a --models run
    # would overwrite the committed fleet-wide chart and frontier JSON with a
    # subset that still looks fleet-wide. Make the caller name the artifact.
    if args.models and not args.out:
        raise SystemExit(
            "--models requires --out: the default path is the fleet-wide chart "
            "for this harness/skill, and writing a subset there would present a "
            "partial frontier as the complete one."
        )
    output = args.out or _default_output(args.harness, args.skill, args.dark)
    # Title leads with the harness and skill (what the chart is OF); the repo and
    # its dataset provenance move into the frontier legend to declutter the title.
    harness_label = HARNESS_LABELS.get(args.harness, args.harness)
    title = args.title or f"Cost vs. quality -- {harness_label} harness, /{args.skill}"
    frontier_label = f"Cost/quality frontier ({args.repo})"

    # kiro-cli prices every point in Kiro credits (not GPU-seconds or a metered
    # Bedrock bill), so give it a credit-basis axis label and footnote instead of
    # the default self-hosted/Anthropic wording. See cost-per-task-methodology.md.
    is_kiro = args.harness == "kiro-cli"
    # Avoid two "$" in the kiro label: matplotlib treats a paired $...$ as a
    # MathText region (would italicize the text and drop the dollar signs), so
    # spell the credit rate as "USD" instead.
    cost_label = args.cost_label or (
        "Mean cost per task (USD) -- kiro-cli, Kiro credits at 0.04 USD/credit (see notes)"
        if is_kiro
        else "Mean cost per task ($) -- self-hosted hardware-derived; "
        "Anthropic metered (see notes)"
    )
    cost_basis_note = args.cost_basis_note or (
        "Cost basis: kiro-cli is priced in Kiro credits at $0.04/credit "
        "(configurable) -- see docs/cost-per-task-methodology.md."
        if is_kiro
        else _DEFAULT_COST_BASIS_NOTE
    )

    arms = _parse_arms(args.throughput_arm)
    points = _collect_points(
        data_dir, args.repo, args.harness, args.skill, args.models, arms
    )
    for point in points:
        logger.info(
            "  %-32s score=%.2f cost=$%.2f (%d/%d scored)",
            point.model,
            point.mean_score,
            point.mean_cost,
            point.n_scored,
            point.n_tasks,
        )
    frontier = _pareto_frontier(points)
    metrics_dir = args.metrics_dir.expanduser().resolve()
    stem = f"pareto-frontier-{output.stem}" if args.models else None
    # Check the scope on BOTH themes, before either output is touched: the dark
    # run writes no JSON, so without this it would clobber the dark image using
    # the very scope the light run just refused.
    _guard_scope_change(
        metrics_dir
        / f"{stem or f'pareto-frontier-{HARNESS_CODES.get(args.harness, args.harness)}-{args.skill}'}.json",
        harness=args.harness,
        skill=args.skill,
        repo=args.repo,
        force=args.force,
    )
    # Emit the machine-readable frontier once (light run), theme-independent.
    if not args.dark:
        _write_frontier_json(
            points,
            harness=args.harness,
            skill=args.skill,
            repo=args.repo,
            out_dir=metrics_dir,
            # A filtered run names its own artifacts (enforced above), so derive
            # the JSON stem from the image and never clobber the fleet-wide file.
            stem=stem,
            models_filter=args.models,
            force=args.force,
        )
    # Colour carries the cost BASIS, which is the chart's main reading hazard: a
    # metered Bedrock bill and a hardware-derived self-hosted figure are dollars
    # measured differently (see cost-per-task-methodology.md). The legend names
    # both so provenance is never inferred from position on the axis.
    palette = _THEME[mode]
    # Only when the chart actually mixes both bases. ModelPoint.hosting is a
    # binary read of provider == "bedrock", so on a single-basis chart it says
    # nothing true: a kiro-cli run is priced in Kiro credits, neither a metered
    # Bedrock bill nor hardware-derived, and colouring it "self-hosted" against
    # an empty Bedrock swatch states the opposite of the cost-basis note below.
    hostings = {p.hosting for p in points}
    hosting_legend = (
        []
        if len(hostings) < 2
        else [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=11,
                markerfacecolor=palette["bedrock"],
                markeredgecolor=palette["surface"],
                markeredgewidth=2,
                label="Bedrock -- metered bill",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=11,
                markerfacecolor=palette["self_hosted"],
                markeredgecolor=palette["surface"],
                markeredgewidth=2,
                label="Self-hosted -- hardware-derived",
            ),
        ]
    )
    color_for = (
        (lambda p: palette["bedrock" if p.hosting == "Bedrock" else "self_hosted"])
        if hosting_legend
        else None
    )
    _plot(
        points,
        frontier,
        mode=mode,
        title=title,
        cost_label=cost_label,
        output=output,
        frontier_label=frontier_label,
        cost_basis_note=cost_basis_note,
        color_for=color_for,
        extra_legend=hosting_legend or None,
    )


if __name__ == "__main__":
    try:
        main()
    except ScopeMismatchError as exc:
        # A wrong --repo is a mistake to correct, not a stack trace to read.
        logger.error("%s", exc)
        raise SystemExit(2) from None
