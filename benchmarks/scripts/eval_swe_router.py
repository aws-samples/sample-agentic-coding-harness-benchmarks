#!/usr/bin/env python3
"""Score the swe-router skill against the runs it would have replaced.

WHY THIS EXISTS
---------------
The router recommends a model per task. Whether that is worth doing is an
empirical question, and this repository already holds the answer: all 16
measured models ran all 21 tasks of ``mcp-gateway-registry-v2``, so for any
model the router picks we can look up what that model ACTUALLY scored and cost
on that task rather than estimating it.

So this replays the router over every task in a dataset, joins its pick to the
recorded run, and compares the result against a fixed-model baseline (running
one model -- by default the top scorer -- on everything). The output is the
per-task table plus the two totals that matter: how much cheaper routing was,
and how much quality it gave up to get there.

THE CIRCULARITY, AND THE HONEST VERSION
---------------------------------------
The router reads ``models.json``, whose per-tier means are computed FROM these
same 21 tasks. Replaying it over them is therefore in-sample: the router is
partly being asked to predict data it has already seen, which flatters it.

``--holdout`` removes that. It rebuilds every model's tier mean and cost mean
with the routed task EXCLUDED, writes that into a temporary models.json, and
routes from it -- so each pick is made without knowing the task it is about to
be scored on. Leave-one-out is the honest number; the default in-sample run is
the upper bound. Both are emitted, and the report says which it is.

WHAT THE FLOOR IS
-----------------
The skill derives a quality floor per task from the consequence of the change
being wrong, which is a judgment a script cannot make. So the floor here is an
explicit policy input: one value for every task (``--floor``), or a per-task
mapping (``--floors-file``). ``--floor-sweep`` runs several and reports each,
which is the useful form -- a single floor is one point on a curve.

Run from the ``benchmarks/`` directory:

    uv run scripts/eval_swe_router.py
    uv run scripts/eval_swe_router.py --holdout --floor-sweep 55,65,70,75
    uv run scripts/eval_swe_router.py --available claude-opus-5,claude-sonnet-5
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path
from collections.abc import Callable
from functools import partial
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent.parent
BENCHMARKS_DIR = _SCRIPTS_DIR.parent

from token_accounting import (  # noqa: E402
    cache_partition_for_agent,
    compute_total_tokens_processed,
)

# The router skill ships beside the repo's other skills; route.py is imported
# rather than shelled out to so the selection under test is the real one.
_SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "swe-router"
sys.path.insert(0, str(_SKILL_DIR))

from route import RouteError, route  # noqa: E402
from swe_router_rubric import (  # noqa: E402
    BUCKET_TO_ROUTE_TIER,
    load_policy,
    relative_bucket,
    relative_edges,
)

# Hardware-derived $/token for self-hosted models, same source and same
# "cheapest sustainable concurrency level" rule plot_cost_quality.py uses, so a
# cost here is on the published frontier's basis rather than a second opinion.
PERF_SUMMARY_DIR = (
    REPO_ROOT / "self-hosted" / "vllm" / "benchmark-output" / "throughput"
)
PERF_SUMMARY_FILENAME = "performance-summary.json"
RUN_SUMMARY_FILENAME = "run-summary.json"
DATA_DIR = BENCHMARKS_DIR / "swe-benchmark-data"

DEFAULT_DATASET = "dataset/mcp-gateway-registry-v2.yaml"
DEFAULT_HARNESS = "omp"
DEFAULT_SKILL = "swe3"
DEFAULT_BASELINE = "claude-opus-5"
# Production service: it ships, a defect reaches someone. The dataset's tasks are
# real closed issues from a deployed gateway, so this is the floor its own
# consequences imply. Overridable, and --floor-sweep is the better question.
DEFAULT_FLOOR = 70.0
TIERS = ("trivial", "low", "medium", "high")
# Two models within this many points at a tier are indistinguishable on 5-6
# tasks run once each. Mirrors the skill's tie band; kept as a constant so the
# report can state the band it applied.
DEFAULT_TIE_BAND = 3.0
# --column-model regression routes on one predicted score per model and task.
# route.py still selects by tier key, so the predictions travel under this one.
PREDICTED_TIER = "medium"
# Below this many scored tasks a model's line is not fitted; its mean stands in.
MIN_POINTS_FOR_FIT = 3


def _read_json(path: Path) -> dict[str, Any] | None:
    """Return parsed JSON at ``path``, or None when missing or unparseable.

    Args:
        path: File to read.

    Returns:
        The parsed object, or None.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _blended_cost_per_token(model: str) -> float | None:
    """Cheapest measured blended $/token for a self-hosted model.

    Args:
        model: Model slug, which also names its throughput arm.

    Returns:
        The rate, or None when the model was never swept (i.e. it is metered).
    """
    summary = _read_json(PERF_SUMMARY_DIR / model / PERF_SUMMARY_FILENAME)
    if summary is None:
        return None
    rates = [
        level["blended_cost_per_token_usd"]
        for level in summary.get("levels", [])
        if isinstance(level.get("blended_cost_per_token_usd"), (int, float))
    ]
    return min(rates) if rates else None


def _task_cost(
    task: dict[str, Any],
    per_token: float | None,
    agent: str | None = None,
) -> float:
    """Cost of one recorded task run, on the model's own cost basis.

    A self-hosted model is priced hardware-derived (its measured $/token times
    the tokens the server actually processed); a metered model uses the bill the
    provider returned. Mixing the two on one axis is directional, which is why
    the report labels each model's basis.

    Args:
        task: One entry from a run-summary's ``tasks`` list.
        per_token: The model's blended $/token, or None when metered.
        agent: The agent that produced the run. An agent with disjoint token
            counts (codex) declares its cache shape instead of having it
            detected from the data (issue #183).

    Returns:
        Cost in USD.
    """
    if per_token is None:
        cost = task.get("total_cost_usd")
        return float(cost) if isinstance(cost, (int, float)) else 0.0
    tokens = compute_total_tokens_processed(
        task.get("input_tokens") or 0,
        task.get("output_tokens") or 0,
        task.get("cache_read_tokens") or 0,
        task.get("cache_write_tokens") or task.get("cache_creation_tokens") or 0,
        context=f"eval_swe_router:{task.get('task')}",
        cache_partition=cache_partition_for_agent(agent),
    )
    return tokens * per_token


def _load_results(
    harness: str,
    skill: str,
    scope: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load every model's per-task result for one harness/skill/scope.

    Args:
        harness: Harness folder (e.g. ``omp``).
        skill: Skill folder (e.g. ``swe3``).
        scope: Dataset scope folder (e.g. ``mcp-gateway-registry-v2``).

    Returns:
        ``{model_slug: {task_id: {score, cost, failed, complexity, hosting}}}``.
        A task the model failed keeps its entry with ``failed`` True and a None
        score, so the router can never be credited with a run that did not
        produce artifacts.
    """
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for model_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        model = model_dir.name
        summary_path = model_dir / harness / skill / scope / RUN_SUMMARY_FILENAME
        summary = _read_json(summary_path)
        if summary is None:
            continue
        per_token = _blended_cost_per_token(model)
        failed_ids = set(summary.get("failed_tasks") or [])
        per_task: dict[str, dict[str, Any]] = {}
        for task in summary.get("tasks", []):
            task_id = task.get("task")
            score = task.get("task_score")
            failed = bool(task.get("failed")) or task_id in failed_ids or not score
            per_task[task_id] = {
                "score": None if failed else float(score),
                "cost_usd": _task_cost(task, per_token, summary.get("agent")),
                "failed": failed,
                "complexity": task.get("complexity"),
                "cost_basis": "hardware-derived" if per_token else "metered",
            }
        if per_task:
            results[model] = per_task
    return results


def _tier_stats(
    results: dict[str, dict[str, dict[str, Any]]],
    exclude_task: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Recompute every model's tier means from raw per-task results.

    Mirrors the convention behind ``models.json``: a failed task is excluded
    from both the score mean and the cost mean (it is a model failure, not a
    quality measurement) and is instead reported as a completion shortfall.

    Args:
        results: Output of ``_load_results``.
        exclude_task: A task id to leave out entirely, for leave-one-out
            routing. None keeps every task (the in-sample case).

    Returns:
        ``{model: {score, cost_per_task_usd, score_by_complexity,
        completion_by_complexity, tasks_completed, tasks_total}}``.
    """
    stats: dict[str, dict[str, Any]] = {}
    for model, tasks in results.items():
        scores: list[float] = []
        costs: list[float] = []
        by_tier: dict[str, list[float]] = {}
        completed: dict[str, list[int]] = {}
        for task_id, record in tasks.items():
            if task_id == exclude_task:
                continue
            tier = record["complexity"]
            done = completed.setdefault(tier, [0, 0])
            done[1] += 1
            if record["failed"]:
                continue
            done[0] += 1
            scores.append(record["score"])
            costs.append(record["cost_usd"])
            by_tier.setdefault(tier, []).append(record["score"])
        if not scores:
            continue
        stats[model] = {
            "score": round(statistics.fmean(scores), 2),
            "cost_per_task_usd": round(statistics.fmean(costs), 4),
            "score_by_complexity": {
                tier: round(statistics.fmean(vals), 2) for tier, vals in by_tier.items()
            },
            "completion_by_complexity": {
                tier: f"{done[0]}/{done[1]}" for tier, done in completed.items()
            },
            "tasks_completed": len(scores),
            "tasks_total": sum(d[1] for d in completed.values()),
        }
    return stats


def _models_json(
    stats: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build a models.json payload route.py can consume.

    Args:
        stats: Output of ``_tier_stats``.
        provenance: The provenance block to carry through.

    Returns:
        A schema-1.0 models.json mapping.
    """
    return {
        "schema_version": "1.0",
        "generated_by": "eval_swe_router.py (leave-one-out)",
        "provenance": provenance,
        "models": [
            {
                "model": model,
                "score": s["score"],
                "cost_per_task_usd": s["cost_per_task_usd"],
                "hosting": "self-hosted"
                if _blended_cost_per_token(model)
                else "Bedrock",
                "tasks_completed": s["tasks_completed"],
                "tasks_total": s["tasks_total"],
                "excluded_tasks": [],
                "on_combined_frontier": False,
                "on_hosting_frontier": False,
                "score_by_complexity": s["score_by_complexity"],
                "completion_by_complexity": s["completion_by_complexity"],
            }
            for model, s in sorted(stats.items(), key=lambda kv: -kv[1]["score"])
        ],
    }


def _route_task(
    tier: str,
    floor: float,
    available: list[str],
    models_path: Path,
    allowed_file: Path | None,
    no_allow_list: bool,
    tie_band: float,
    min_completion: float = 0.0,
) -> dict[str, Any]:
    """Ask the router for one task's model.

    Args:
        tier: The task's complexity tier.
        floor: The quality floor policy for this task.
        available: Model slugs the developer could select.
        models_path: models.json to route from (in-sample or leave-one-out).
        allowed_file: Explicit allow-list path, or None to let route.py find one.
        no_allow_list: Ignore organisational policy entirely.
        tie_band: Points below which two models count as tied.
        min_completion: Skip a model that finished less than this share of the
            tasks in the column it is read from.

    Returns:
        The route.py result dict.

    Raises:
        RouteError: On unusable inputs.
    """
    return route(
        tier=tier,
        floor=floor,
        available=available,
        models_path=models_path,
        aliases_path=_SKILL_DIR / "model-aliases.json",
        allowed_file=allowed_file,
        no_allow_list=no_allow_list,
        tie_band=tie_band,
        min_completion=min_completion,
    )


def _regrouped(
    results: dict[str, dict[str, dict[str, Any]]],
    columns: dict[str, str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """A copy of the results with each task filed under the given column."""
    return {
        model: {
            task_id: {
                **record,
                "complexity": columns.get(task_id, record["complexity"]),
            }
            for task_id, record in per_task.items()
        }
        for model, per_task in results.items()
    }


def _fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares line through (difficulty sum, score) points.

    Falls back to a flat line at the mean when there are too few points or no
    spread in difficulty, where a slope would be noise.

    Args:
        points: ``(x, y)`` pairs.

    Returns:
        ``(intercept, slope)``.
    """
    ys = [y for _, y in points]
    if len(points) < MIN_POINTS_FOR_FIT:
        return (statistics.fmean(ys) if ys else 0.0), 0.0
    xs = [x for x, _ in points]
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0:
        return y_mean, 0.0
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / sxx
    return y_mean - slope * x_mean, slope


def _predicted_stats(
    results: dict[str, dict[str, dict[str, Any]]],
    sums: dict[str, int],
    task_id: str,
    holdout: bool,
) -> dict[str, dict[str, Any]]:
    """Each model's stats for routing one task, its score predicted from difficulty.

    For every model, fits score against the rubric's difficulty sum over its
    scored runs on the other tasks (all tasks without ``holdout``) and predicts
    its score at this task's sum. Every task informs every prediction, so no
    model's estimate rests on a thin column. Failed runs are left out of the
    fit, as they are left out of the column means; the completion count still
    reports them.

    Args:
        results: Output of ``_load_results``.
        sums: Each task's consolidated difficulty sum from the rubric pass.
        task_id: The task being routed.
        holdout: Fit without the routed task.

    Returns:
        ``_tier_stats``-shaped stats with the prediction under ``PREDICTED_TIER``.
    """
    exclude = task_id if holdout else None
    stats = _tier_stats(results, exclude_task=exclude)
    for model, s in stats.items():
        points = [
            (float(sums[tid]), rec["score"])
            for tid, rec in results[model].items()
            if tid != exclude and tid in sums and not rec["failed"]
        ]
        intercept, slope = _fit_line(points)
        predicted = min(100.0, max(0.0, intercept + slope * sums[task_id]))
        s["score_by_complexity"] = {PREDICTED_TIER: round(predicted, 2)}
        s["completion_by_complexity"] = {
            PREDICTED_TIER: f"{s['tasks_completed']}/{s['tasks_total']}"
        }
    return stats


def _evaluate(
    tasks: list[dict[str, Any]],
    results: dict[str, dict[str, dict[str, Any]]],
    baseline: str,
    floors: dict[str, float],
    available: list[str],
    allowed_file: Path | None,
    no_allow_list: bool,
    tie_band: float,
    holdout: bool,
    provenance: dict[str, Any],
    judged_tiers: dict[str, str] | None = None,
    column_for: Callable[[str], dict[str, str]] | None = None,
    stats_for: Callable[[str], dict[str, dict[str, Any]]] | None = None,
    fixed_stats: dict[str, dict[str, Any]] | None = None,
    min_completion: float = 0.0,
) -> dict[str, Any]:
    """Replay the router over every task and compare it to the baseline.

    Args:
        tasks: Dataset tasks, each with ``id`` and ``complexity``.
        results: Output of ``_load_results``.
        baseline: Model slug run on every task for comparison.
        floors: Per-task quality floor.
        available: Model slugs the developer could select.
        allowed_file: Explicit allow-list path, or None.
        no_allow_list: Ignore organisational policy entirely.
        tie_band: Points below which two models count as tied.
        holdout: Route each task from means that exclude that task.
        provenance: Provenance block for the synthesized models.json.
        judged_tiers: Per-task tier from a judged run, overriding the dataset's
            own ``complexity`` label. None keeps the dataset label.
        column_for: Given the task being routed, the column of EVERY task, used
            both to route it and to rebuild the columns it is routed against.
            Lets the grouping itself depend on the held-out task, as relative
            rubric buckets do. Overrides ``judged_tiers`` and the dataset label.
        stats_for: Given the task being routed, every model's stats with one
            predicted score under ``PREDICTED_TIER``. Overrides ``column_for``.
        fixed_stats: Every model's stats from ANOTHER dataset, routed against
            for every task. ``results`` then only supplies each routed task's
            outcome. ``column_for`` still names the routed task's column.
        min_completion: Passed to route.py: skip a model whose completion in
            the column falls below this share.

    Returns:
        A result mapping with ``rows`` (one per task) and ``totals``.

    Raises:
        SystemExit: If the baseline model has no recorded runs.
    """
    judged_tiers = judged_tiers or {}
    if baseline not in results:
        raise SystemExit(
            f"baseline model {baseline!r} has no runs for this harness/skill/scope; "
            f"available: {', '.join(sorted(results))}"
        )
    rows: list[dict[str, Any]] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="router-eval-"))
    in_sample_path = tmp_dir / "models-in-sample.json"
    in_sample_path.write_text(
        json.dumps(_models_json(fixed_stats or _tier_stats(results), provenance)),
        encoding="utf-8",
    )
    for task in tasks:
        task_id = task["id"]
        # A judged run supplies BOTH halves of step 1, so the tier comes from the
        # judgment rather than the dataset label -- otherwise the eval would hand
        # the router a perfect classifier it would not have in real use.
        tier = judged_tiers.get(task_id, task["complexity"])
        floor = floors.get(task_id, DEFAULT_FLOOR)
        grouped = results
        if column_for is not None:
            columns = column_for(task_id)
            tier = columns[task_id]
            grouped = _regrouped(results, columns)
        if fixed_stats is not None:
            models_path = in_sample_path
        elif stats_for is not None:
            tier = PREDICTED_TIER
            models_path = tmp_dir / f"models-{task_id}.json"
            models_path.write_text(
                json.dumps(_models_json(stats_for(task_id), provenance)),
                encoding="utf-8",
            )
        elif holdout or column_for is not None:
            models_path = tmp_dir / f"models-{task_id}.json"
            stats = _tier_stats(grouped, exclude_task=task_id if holdout else None)
            models_path.write_text(
                json.dumps(_models_json(stats, provenance)), encoding="utf-8"
            )
        else:
            models_path = in_sample_path
        routed = _route_task(
            tier=tier,
            floor=floor,
            available=available,
            models_path=models_path,
            allowed_file=allowed_file,
            no_allow_list=no_allow_list,
            tie_band=tie_band,
            min_completion=min_completion,
        )
        row = _row(task_id, tier, floor, routed, results, baseline)
        row["dataset_complexity"] = task["complexity"]
        row["tier_matches_dataset"] = tier == task["complexity"]
        rows.append(row)
    return {"rows": rows, "totals": _totals(rows, baseline)}


def _row(
    task_id: str,
    tier: str,
    floor: float,
    routed: dict[str, Any],
    results: dict[str, dict[str, dict[str, Any]]],
    baseline: str,
) -> dict[str, Any]:
    """Join one routing decision to the runs it picked and replaced.

    Args:
        task_id: The task.
        tier: Its complexity tier.
        floor: The floor policy applied.
        routed: The route.py result for this task.
        results: Output of ``_load_results``.
        baseline: The comparison model slug.

    Returns:
        One report row.
    """
    base = results[baseline][task_id]
    pick = routed.get("recommended")
    row: dict[str, Any] = {
        "task": task_id,
        "complexity": tier,
        "floor": floor,
        "baseline_model": baseline,
        "baseline_score": base["score"],
        "baseline_cost_usd": round(base["cost_usd"], 4),
        "baseline_failed": base["failed"],
        "router_status": routed["status"],
    }
    if pick is None:
        # No candidate cleared the floor: the skill's instruction is to stay put,
        # so the honest comparison is the baseline run, at baseline cost.
        row.update(
            {
                "recommended_model": None,
                "recommended_reason": routed.get("reason"),
                "predicted_score": None,
                "actual_score": base["score"],
                "actual_cost_usd": round(base["cost_usd"], 4),
                "actual_failed": base["failed"],
                "switched": False,
                "score_delta": 0.0,
                "cost_delta_usd": 0.0,
                "cost_saving_pct": 0.0,
                "met_floor": bool(base["score"] and base["score"] >= floor),
            }
        )
        return row
    model = pick["model"]
    actual = results.get(model, {}).get(task_id)
    if actual is None:
        raise SystemExit(
            f"router picked {model!r} for {task_id!r} but no run is recorded for it"
        )
    score_delta = (
        None
        if actual["score"] is None or base["score"] is None
        else round(actual["score"] - base["score"], 2)
    )
    cost_delta = round(actual["cost_usd"] - base["cost_usd"], 4)
    row.update(
        {
            "recommended_model": model,
            # What the router BELIEVED it was buying (the tier mean it selected
            # on) beside what the model actually scored on this one task. The
            # gap between the two is the router's per-task prediction error.
            "predicted_score": pick["score"],
            "actual_score": actual["score"],
            "actual_cost_usd": round(actual["cost_usd"], 4),
            "actual_failed": actual["failed"],
            "switched": model != baseline,
            "score_delta": score_delta,
            "cost_delta_usd": cost_delta,
            "cost_saving_pct": round(-cost_delta / base["cost_usd"] * 100, 1)
            if base["cost_usd"]
            else 0.0,
            "met_floor": bool(actual["score"] and actual["score"] >= floor),
            "cost_basis": actual["cost_basis"],
        }
    )
    return row


def _totals(rows: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    """Aggregate the per-task rows into the headline comparison.

    Scores are meaned over tasks where BOTH arms produced a scored run, so the
    two means describe the same set of tasks. Costs are summed over every task,
    because a failed run still cost money.

    Args:
        rows: Per-task rows from ``_row``.
        baseline: The comparison model slug.

    Returns:
        The totals mapping.
    """
    base_cost = sum(r["baseline_cost_usd"] for r in rows)
    routed_cost = sum(r["actual_cost_usd"] for r in rows)
    paired = [
        r
        for r in rows
        if r["baseline_score"] is not None and r["actual_score"] is not None
    ]
    base_mean = statistics.fmean(r["baseline_score"] for r in paired) if paired else 0.0
    routed_mean = statistics.fmean(r["actual_score"] for r in paired) if paired else 0.0
    return {
        "tasks": len(rows),
        "tasks_switched": sum(1 for r in rows if r["switched"]),
        "tasks_scored_both_arms": len(paired),
        "baseline_model": baseline,
        "baseline_total_cost_usd": round(base_cost, 2),
        "routed_total_cost_usd": round(routed_cost, 2),
        "cost_saving_usd": round(base_cost - routed_cost, 2),
        "cost_saving_pct": round((base_cost - routed_cost) / base_cost * 100, 1)
        if base_cost
        else 0.0,
        "baseline_mean_score": round(base_mean, 2),
        "routed_mean_score": round(routed_mean, 2),
        "mean_score_delta": round(routed_mean - base_mean, 2),
        # The same comparison over EVERY task, a failed run counted as 0. The
        # paired means above leave out a task the routed model failed, which
        # flatters a router that picks a model that then fails outright.
        "baseline_mean_score_failures_as_zero": round(
            statistics.fmean(r["baseline_score"] or 0.0 for r in rows), 2
        )
        if rows
        else 0.0,
        "routed_mean_score_failures_as_zero": round(
            statistics.fmean(r["actual_score"] or 0.0 for r in rows), 2
        )
        if rows
        else 0.0,
        # The router's own failure mode: it picked a model to clear a floor and
        # the model then landed under it. Counted for both arms so the baseline
        # is held to the same test.
        "mean_score_delta_failures_as_zero": round(
            statistics.fmean(
                (r["actual_score"] or 0.0) - (r["baseline_score"] or 0.0) for r in rows
            ),
            2,
        )
        if rows
        else 0.0,
        "tasks_below_floor_routed": sum(1 for r in rows if not r["met_floor"]),
        "tasks_below_floor_baseline": sum(
            1
            for r in rows
            if not (r["baseline_score"] and r["baseline_score"] >= r["floor"])
        ),
        "tasks_failed_routed": sum(1 for r in rows if r["actual_failed"]),
        "tasks_failed_baseline": sum(1 for r in rows if r["baseline_failed"]),
        "models_used": sorted(
            {r["recommended_model"] for r in rows if r["recommended_model"]}
        ),
        # How the work actually split across models. A row where nothing cleared
        # the floor is counted separately from one where the baseline was picked
        # on merit: both run the same model, but only the second is a choice.
        "model_counts": dict(
            sorted(
                Counter(
                    r["recommended_model"] or f"(no pick -- stayed on {baseline})"
                    for r in rows
                ).items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
        ),
        # Mean score expressed against the baseline, since a delta in points is
        # hard to size without knowing the baseline it moved from.
        "quality_delta_pct": round((routed_mean - base_mean) / base_mean * 100, 1)
        if base_mean
        else 0.0,
        # The aggregate saving above is total-over-total, which is what actually
        # lands on a bill. This is the mean of the per-task percentages, which
        # weights a $4 task the same as a $32 one and so reads much higher --
        # kept only so the two are never confused for each other.
        "mean_per_task_saving_pct": round(
            statistics.fmean(r["cost_saving_pct"] for r in rows), 1
        )
        if rows
        else 0.0,
    }


def _fmt(value: Any, spec: str = "") -> str:
    """Format a value for a markdown cell, rendering None as an em dash.

    Args:
        value: The value.
        spec: A format spec applied to non-None values.

    Returns:
        The cell text.
    """
    if value is None:
        return "--"
    return format(value, spec) if spec else str(value)


def _markdown(report: dict[str, Any]) -> str:
    """Render the report as a markdown document.

    Args:
        report: The full report mapping.

    Returns:
        The markdown source.
    """
    cfg = report["config"]
    lines: list[str] = []
    if len(report["runs"]) > 1:
        lines += [
            "## Summary across floors",
            "",
            "One row per quality floor. A higher floor buys quality with money: "
            "it forces the router onto stronger models, so the saving shrinks "
            "and the score climbs back toward the baseline.",
            "",
            "| Floor | Router cost | Saving | Router score | "
            f"{cfg['baseline']} score | Δ score | Under floor | Models used |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for run in report["runs"]:
            t = run["totals"]
            lines.append(
                f"| {run['floor']:.0f} | ${t['routed_total_cost_usd']:,.2f} "
                f"| {t['cost_saving_pct']:.1f}% | {t['routed_mean_score']:.2f} "
                f"| {t['baseline_mean_score']:.2f} | {t['mean_score_delta']:+.2f} "
                f"| {t['tasks_below_floor_routed']}/{t['tasks']} "
                f"| {', '.join(t['models_used'])} |"
            )
        lines.append("")
    for run in report["runs"]:
        totals = run["totals"]
        floor_label = (
            f"Floor {run['floor']:.0f}"
            if run["floor"] is not None
            else "Judged floors and tiers"
        )
        lines += [
            f"## {floor_label}",
            "",
            f"| Task | Tier | Floor | Router pick | Predicted | Actual | "
            f"{cfg['baseline']} | Δ score | Cost | Baseline cost | Saving |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in run["rows"]:
            pick = row["recommended_model"] or f"_stay on {cfg['baseline']}_"
            flag = " ⚠" if not row["met_floor"] else ""
            lines.append(
                f"| {row['task']} | {row['complexity']} | {_fmt(row['floor'], '.0f')} "
                f"| {pick} | {_fmt(row['predicted_score'], '.2f')} "
                f"| {_fmt(row['actual_score'], '.1f')}{flag} "
                f"| {_fmt(row['baseline_score'], '.1f')} "
                f"| {_fmt(row['score_delta'], '+.1f')} "
                f"| ${_fmt(row['actual_cost_usd'], '.2f')} "
                f"| ${_fmt(row['baseline_cost_usd'], '.2f')} "
                f"| {_fmt(row['cost_saving_pct'], '+.0f')}% |"
            )
        lines += [
            "",
            f"**Totals over {totals['tasks']} tasks** "
            f"({totals['tasks_switched']} switched away from {cfg['baseline']})",
            "",
            "| | Router | Baseline | Difference |",
            "|---|---:|---:|---:|",
            f"| Total cost | ${totals['routed_total_cost_usd']:,.2f} "
            f"| ${totals['baseline_total_cost_usd']:,.2f} "
            f"| **-${totals['cost_saving_usd']:,.2f} "
            f"({totals['cost_saving_pct']:.1f}%)** |",
            f"| Mean score ({totals['tasks_scored_both_arms']} tasks scored "
            f"in both arms) | {totals['routed_mean_score']:.2f} "
            f"| {totals['baseline_mean_score']:.2f} "
            f"| **{totals['mean_score_delta']:+.2f}** |",
            f"| Tasks under floor | {totals['tasks_below_floor_routed']} "
            f"| {totals['tasks_below_floor_baseline']} "
            f"| {totals['tasks_below_floor_routed'] - totals['tasks_below_floor_baseline']:+d} |",
            f"| Tasks failed outright | {totals['tasks_failed_routed']} "
            f"| {totals['tasks_failed_baseline']} "
            f"| {totals['tasks_failed_routed'] - totals['tasks_failed_baseline']:+d} |",
            "",
            f"Models the router used: {', '.join(totals['models_used']) or 'none'}.",
            "",
        ]
    return "\n".join(lines)


def _headline(totals: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Build the headline: what the router chose, and what it bought.

    Stated in the order a reader needs it -- how many models the routing
    actually used (the thing that separates routing from picking one cheap model
    and stopping), then the two numbers that decide whether it was worth doing.

    Args:
        totals: The run's totals block.
        cfg: The report config, for the baseline's name.

    Returns:
        Markdown lines.
    """
    counts = totals["model_counts"]
    picked = ", ".join(
        f"**{model}** {n}x" for model, n in counts.items() if not model.startswith("(")
    )
    stayed = next(
        (n for model, n in counts.items() if model.startswith("(")),
        0,
    )
    stayed_note = (
        f" On {stayed} further task(s) nothing cleared the floor, so the skill's "
        f"answer was to stay on `{cfg['baseline']}`."
        if stayed
        else ""
    )
    return [
        f"> Across the {totals['tasks']} tasks the router selected "
        f"**{len(totals['models_used'])} different models**: {picked}."
        f"{stayed_note}",
        ">",
        f"> Against running `{cfg['baseline']}` on everything, that cost "
        f"**{totals['cost_saving_pct']:.1f}% less** "
        f"(${totals['routed_total_cost_usd']:,.2f} against "
        f"${totals['baseline_total_cost_usd']:,.2f}) for a quality change of "
        f"**{totals['quality_delta_pct']:+.1f}%** "
        f"({totals['routed_mean_score']:.2f} against "
        f"{totals['baseline_mean_score']:.2f} mean task score, "
        f"{totals['mean_score_delta']:+.2f} points).",
        ">",
        f"> The saving is total-over-total, which is what lands on a bill. The "
        f"mean of the per-task percentages is "
        f"{totals['mean_per_task_saving_pct']:.1f}%, higher because it weights a "
        f"cheap task the same as an expensive one.",
        "",
    ]


def _document(report: dict[str, Any]) -> str:
    """Wrap the per-floor tables in a header that states the method.

    Args:
        report: The full report mapping.

    Returns:
        The complete markdown document.
    """
    cfg = report["config"]
    prov = report["provenance"]
    sampling = (
        f"Zero-shot: every task routes from columns built on `{cfg['columns_from']}` "
        "alone, a different repository, so no pick saw any task of this one."
        if cfg.get("columns_from")
        else "Leave-one-out: each task routes from tier means recomputed with that "
        "task excluded, so no pick knows the run it is scored against."
        if cfg["holdout"]
        else "In-sample: the router read tier means computed from these same "
        "tasks, so this is an upper bound. Re-run with --holdout for the "
        "leave-one-out number."
    )
    header = ["# Does the swe-router pay for itself?", ""]
    if len(report["runs"]) == 1:
        header += _headline(report["runs"][0]["totals"], cfg)
    header += [
        f"Replays the `swe-router` skill over all {cfg['tasks']} tasks of "
        f"`{cfg['scope']}`, then looks up what the model it picked ACTUALLY "
        f"scored and cost on that task, against running "
        f"`{cfg['baseline']}` on everything.",
        "",
        f"- **Sampling.** {sampling}",
        f"- **Floor.** {cfg['floor_note']}",
        f"- **Tier.** {cfg['tier_note']}",
        f"- **Candidates.** {len(cfg['available'])} model(s) the developer could "
        f"select: {', '.join(cfg['available'])}. "
        + (
            "The organisational allow-list was ignored (`--no-allow-list`)."
            if cfg["no_allow_list"]
            else f"Filtered further by the allow-list at `{cfg['allow_list']}`."
        ),
        "- **Cost basis.** Metered provider bills for Bedrock models; "
        "hardware-derived ($/token from the throughput sweep x tokens the "
        "server processed) for self-hosted ones. Mixing the two on one axis is "
        "directional -- see `docs/cost-per-task-methodology.md`.",
        f"- **Scoring.** `task_score` from the repo-grounded "
        f"`{prov.get('judge', {}).get('model', 'LLM')}` judge. One run per task, "
        f"so a per-task gap under ~3 points is noise.",
        f"- **Runs.** {prov.get('harness')} harness, /{prov.get('skill')}, "
        f"measured {prov.get('measured_on')}.",
        "",
        "A ⚠ marks a task where the model the router picked landed below the "
        "floor it was chosen to clear. That is the router getting it wrong, and "
        "the totals count it.",
        "",
    ]
    return "\n".join(header) + "\n" + _markdown(report)


def _load_tasks(dataset_path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read task ids and complexity tiers from a dataset YAML.

    Args:
        dataset_path: Path to the dataset file.

    Returns:
        The task list and the dataset's output scope.

    Raises:
        SystemExit: If the dataset is missing, unparseable, or has a task with
            no complexity tier.
    """
    try:
        data = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"cannot read dataset {dataset_path}: {exc}") from exc
    tasks = []
    for task in data.get("tasks", []):
        tier = task.get("complexity")
        if tier not in TIERS:
            raise SystemExit(
                f"task {task.get('id')!r} has complexity {tier!r}, "
                f"expected one of {', '.join(TIERS)}"
            )
        tasks.append({"id": task["id"], "complexity": tier})
    if not tasks:
        raise SystemExit(f"dataset {dataset_path} has no tasks")
    return tasks, data.get("output_scope") or data.get("name")


def _apply_rubric_inputs(
    path: Path,
    tasks: list[dict[str, Any]],
    bucketing: str,
    holdout: bool,
) -> tuple[dict[str, float], Callable[[str], dict[str, str]], dict[str, Any]]:
    """Route on a rubric pass instead of the skill's floor and tier.

    Each task's required quality becomes its floor and its difficulty bucket its
    column, and the columns are rebuilt to match: every model's existing score
    on a task is filed under that task's bucket rather than its dataset tier.
    No run is repeated; only the grouping changes.

    ``absolute`` bucketing uses the bucket the rubric pass recorded.
    ``relative`` cuts the tasks' difficulty sums into quarters -- with
    ``holdout``, quarters of the OTHER tasks, so the routed task never shapes
    the boundaries it is placed by. route.py still knows four tier names, so
    the buckets travel as those keys (``BUCKET_TO_ROUTE_TIER``).

    Args:
        path: A rubric-inputs JSON from ``run-swe-router-headless.py --mode rubric``.
        tasks: Dataset tasks.
        bucketing: ``absolute`` or ``relative``.
        holdout: Leave the routed task out of the quarters (relative only).

    Returns:
        Per-task floors, the per-task column function for ``_evaluate``, and the
        rubric pass's metadata.

    Raises:
        SystemExit: If the file is not a rubric pass or misses a task.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("mode") != "rubric":
        raise SystemExit(f"{path} is not a rubric pass (mode={data.get('mode')!r})")
    scored = data.get("tasks") or {}
    missing = {t["id"] for t in tasks} - set(scored)
    if missing:
        raise SystemExit(
            f"--rubric-inputs is missing {len(missing)} task(s): {', '.join(sorted(missing))}"
        )
    floors = {tid: float(v["required_quality"]) for tid, v in scored.items()}
    sums = {tid: int(v["difficulty_sum"]) for tid, v in scored.items()}
    absolute = {tid: BUCKET_TO_ROUTE_TIER[v["bucket"]] for tid, v in scored.items()}

    def column_for(routed: str) -> dict[str, str]:
        if bucketing == "absolute":
            return absolute
        reference = [s for tid, s in sums.items() if not (holdout and tid == routed)]
        edges = relative_edges(reference)
        return {
            tid: BUCKET_TO_ROUTE_TIER[relative_bucket(s, edges)]
            for tid, s in sums.items()
        }

    meta = dict(data.get("judged_by") or {})
    meta.update(mode="rubric", bucketing=bucketing)
    return floors, column_for, meta


def _columns_from_other_dataset(
    dataset_path: Path,
    rubric_inputs: Path,
    harness: str,
    skill: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Every model's bucket columns, built from a different dataset.

    This is the zero-shot case: the router has only another repository's
    benchmark to go on. Each of that dataset's tasks is filed under the bucket
    its own rubric pass gave it, and the columns are its models' means there.
    Buckets are absolute, so the routed tasks are placed by the same fixed
    ranges.

    Args:
        dataset_path: The dataset whose results build the columns.
        rubric_inputs: That dataset's rubric pass.
        harness: Harness folder.
        skill: Skill folder.

    Returns:
        ``_tier_stats`` output for that dataset, and its scope.

    Raises:
        SystemExit: If the dataset has no results or the rubric pass misses a task.
    """
    tasks, scope = _load_tasks(dataset_path)
    results = _load_results(harness, skill, scope)
    if not results:
        raise SystemExit(
            f"no results for --columns-from under {harness}/{skill}/{scope}"
        )
    _, column_for, _ = _apply_rubric_inputs(rubric_inputs, tasks, "absolute", False)
    known = {t["id"] for t in tasks}
    columns = {tid: c for tid, c in column_for("").items() if tid in known}
    return _tier_stats(_regrouped(results, columns)), scope


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Score the swe-router skill against the runs it would replace.",
        epilog=(
            "Examples:\n"
            "  uv run scripts/eval_swe_router.py\n"
            "  uv run scripts/eval_swe_router.py --holdout --floor-sweep 55,65,70,75\n"
            "  uv run scripts/eval_swe_router.py --no-allow-list --baseline claude-opus-5\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset YAML path.")
    parser.add_argument("--harness", default=DEFAULT_HARNESS, help="Harness folder.")
    parser.add_argument("--skill", default=DEFAULT_SKILL, help="Skill folder.")
    parser.add_argument(
        "--scope", default=None, help="Scope folder (default: the dataset's)."
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help="Model run on every task for comparison. Default: %(default)s.",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=DEFAULT_FLOOR,
        help="Quality floor applied to every task. Default: %(default)s.",
    )
    parser.add_argument(
        "--floor-sweep",
        default=None,
        help="Comma-separated floors to run instead of --floor (e.g. 55,65,70,75).",
    )
    parser.add_argument(
        "--judged-inputs",
        type=Path,
        default=None,
        help="JSON from a real run of the skill's step 1: {tasks: {id: {floor, "
        "tier}}}. Supplies BOTH the floor and the tier per task, replacing "
        "--floor/--floor-sweep and the dataset's complexity label.",
    )
    parser.add_argument(
        "--rubric-inputs",
        type=Path,
        default=None,
        help="JSON from run-swe-router-headless.py --mode rubric. Each task's "
        "required quality is its floor and its difficulty bucket its column; "
        "every model's existing scores are regrouped by bucket. Replaces "
        "--judged-inputs, --floor and the dataset's complexity label.",
    )
    parser.add_argument(
        "--bucketing",
        choices=("absolute", "relative"),
        default=None,
        help="Difficulty buckets for --rubric-inputs: absolute ranges, or "
        "quarters of the tasks' own difficulty sums. Default: the routing "
        "policy's difficulty.bucketing.",
    )
    parser.add_argument(
        "--column-model",
        choices=("buckets", "regression"),
        default="buckets",
        help="With --rubric-inputs: compare the required quality against bucket "
        "column means, or against each model's score predicted from the task's "
        "difficulty sum by a line fitted over the other tasks. Default: %(default)s.",
    )
    parser.add_argument(
        "--columns-from",
        type=Path,
        default=None,
        help="Zero-shot replay: build the bucket columns from THIS other dataset's "
        "results instead of --dataset's, and route --dataset's tasks against them. "
        "Needs --rubric-inputs and --columns-rubric-inputs; absolute buckets only.",
    )
    parser.add_argument(
        "--columns-rubric-inputs",
        type=Path,
        default=None,
        help="The rubric pass for --columns-from's dataset.",
    )
    parser.add_argument(
        "--floors-file",
        type=Path,
        default=None,
        help="JSON mapping of task id to floor, overriding --floor per task.",
    )
    parser.add_argument(
        "--available",
        default=None,
        help="Comma-separated models the developer can select. Default: every "
        "model with recorded runs.",
    )
    parser.add_argument(
        "--allowed-file", type=Path, default=None, help="Override the allow-list path."
    )
    parser.add_argument(
        "--no-allow-list",
        action="store_true",
        help="Ignore the organisational allow-list entirely.",
    )
    parser.add_argument(
        "--tie-band",
        type=float,
        default=DEFAULT_TIE_BAND,
        help="Points below which two models count as tied. Default: %(default)s.",
    )
    parser.add_argument(
        "--min-completion",
        type=float,
        default=0.0,
        help="Skip a model that finished less than this share of the tasks in "
        "the column it is read from, as the shipped policy's "
        "min_completion_in_bucket does (1.0). Default: %(default)s, the old rule.",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Route each task from tier means that EXCLUDE that task "
        "(leave-one-out). Removes the in-sample advantage.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "docs" / "metrics" / "swe-router-eval.json",
        help="Where to write the JSON report. Default: %(default)s.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=REPO_ROOT / "docs" / "swe-router-evaluation.md",
        help="Where to write the markdown report. Default: %(default)s.",
    )
    return parser.parse_args()


def main() -> None:
    """Replay the router over a dataset and write the JSON and markdown reports."""
    args = _parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = BENCHMARKS_DIR / dataset_path
    tasks, dataset_scope = _load_tasks(dataset_path)
    scope = args.scope or dataset_scope

    results = _load_results(args.harness, args.skill, scope)
    if not results:
        raise SystemExit(
            f"no run-summary.json found under {DATA_DIR}/*/{args.harness}/"
            f"{args.skill}/{scope}/"
        )
    logger.info(
        "loaded %d model(s) x %d task(s) from %s/%s/%s",
        len(results),
        len(tasks),
        args.harness,
        args.skill,
        scope,
    )

    available = (
        [m.strip() for m in args.available.split(",") if m.strip()]
        if args.available
        else sorted(results)
    )
    judged_tiers: dict[str, str] = {}
    judged_meta: dict[str, Any] = {}
    file_floors: dict[str, float] = {}
    if args.rubric_inputs and args.judged_inputs:
        raise SystemExit("use --rubric-inputs or --judged-inputs, not both")
    column_for = None
    stats_for = None
    if args.column_model == "regression" and not args.rubric_inputs:
        raise SystemExit("--column-model regression needs --rubric-inputs")
    if args.rubric_inputs:
        bucketing = args.bucketing or load_policy().difficulty.bucketing
        file_floors, column_for, judged_meta = _apply_rubric_inputs(
            args.rubric_inputs, tasks, bucketing, args.holdout
        )
        if args.column_model == "regression":
            scored = json.loads(args.rubric_inputs.read_text(encoding="utf-8"))["tasks"]
            sums = {tid: int(v["difficulty_sum"]) for tid, v in scored.items()}

            stats_for = partial(_predicted_stats, results, sums, holdout=args.holdout)
            judged_meta["column_model"] = "regression"
    fixed_stats = None
    if args.columns_from:
        if not (args.rubric_inputs and args.columns_rubric_inputs):
            raise SystemExit(
                "--columns-from needs --rubric-inputs and --columns-rubric-inputs"
            )
        if bucketing != "absolute" or args.column_model != "buckets" or args.holdout:
            raise SystemExit(
                "--columns-from supports absolute buckets only, without --holdout "
                "(the routed tasks are never in the columns)"
            )
        columns_path = args.columns_from
        if not columns_path.is_absolute():
            columns_path = BENCHMARKS_DIR / columns_path
        fixed_stats, columns_scope = _columns_from_other_dataset(
            columns_path, args.columns_rubric_inputs, args.harness, args.skill
        )
        judged_meta["columns_from"] = columns_scope
    if args.judged_inputs:
        judged = json.loads(args.judged_inputs.read_text(encoding="utf-8"))
        judged_meta = judged.get("judged_by") or {}
        for task_id, entry in (judged.get("tasks") or {}).items():
            file_floors[task_id] = float(entry["floor"])
            judged_tiers[task_id] = str(entry["tier"])
        known = {t["id"] for t in tasks}
        missing = known - set(file_floors)
        if missing:
            raise SystemExit(
                f"--judged-inputs is missing {len(missing)} task(s) the dataset "
                f"has: {', '.join(sorted(missing))}"
            )
    if args.floors_file:
        file_floors = {
            str(k): float(v)
            for k, v in json.loads(args.floors_file.read_text(encoding="utf-8")).items()
        }
    sweep = (
        [float(f) for f in args.floor_sweep.split(",") if f.strip()]
        if args.floor_sweep
        else [args.floor]
    )
    provenance = (_read_json(_SKILL_DIR / "models.json") or {}).get("provenance") or {}

    runs: list[dict[str, Any]] = []
    for floor in sweep:
        floors = {t["id"]: file_floors.get(t["id"], floor) for t in tasks}
        try:
            evaluated = _evaluate(
                tasks=tasks,
                results=results,
                baseline=args.baseline,
                floors=floors,
                available=available,
                allowed_file=args.allowed_file,
                no_allow_list=args.no_allow_list,
                tie_band=args.tie_band,
                holdout=args.holdout,
                provenance=provenance,
                judged_tiers=judged_tiers,
                column_for=column_for,
                stats_for=stats_for,
                fixed_stats=fixed_stats,
                min_completion=args.min_completion,
            )
        except RouteError as exc:
            raise SystemExit(f"routing failed at floor {floor}: {exc}") from exc
        totals = evaluated["totals"]
        logger.info(
            "floor %.0f: %s -> $%.2f vs $%.2f baseline (%.1f%% saved), "
            "mean score %.2f vs %.2f (%+.2f; %+.2f with failures as 0), "
            "%d task(s) under floor",
            floor,
            "zero-shot"
            if fixed_stats
            else "leave-one-out"
            if args.holdout
            else "in-sample",
            totals["routed_total_cost_usd"],
            totals["baseline_total_cost_usd"],
            totals["cost_saving_pct"],
            totals["routed_mean_score"],
            totals["baseline_mean_score"],
            totals["mean_score_delta"],
            totals["mean_score_delta_failures_as_zero"],
            totals["tasks_below_floor_routed"],
        )
        runs.append({"floor": None if file_floors else floor, **evaluated})

    report = {
        "config": {
            "dataset": str(dataset_path.relative_to(REPO_ROOT)),
            "scope": scope,
            "harness": args.harness,
            "skill": args.skill,
            "baseline": args.baseline,
            "tasks": len(tasks),
            "available": available,
            "allow_list": str(args.allowed_file) if args.allowed_file else "auto",
            "no_allow_list": args.no_allow_list,
            "tie_band": args.tie_band,
            "holdout": args.holdout,
            "columns_from": judged_meta.get("columns_from"),
            "min_completion": args.min_completion,
            "judged_by": judged_meta,
            "floor_note": (
                "Required quality per task, from the rubric pass by "
                f"{judged_meta.get('harness')} + {judged_meta.get('model')}: blast "
                "radius and verification, priced by routing-policy.json."
                if judged_meta.get("mode") == "rubric"
                else "Judged per task by "
                f"{judged_meta.get('harness')} + {judged_meta.get('model')} "
                "running the skill's step 1 against the cloned repo -- the real "
                "judgment the skill asks for, not a policy constant. See "
                "`swe-router-judged-inputs.md`."
                if judged_meta
                else "The skill derives a quality floor from the consequence of "
                "the change being wrong, which a script cannot judge. Here it is "
                "a policy input, applied uniformly per run: "
                + (
                    "per-task floors from --floors-file"
                    if file_floors
                    else ", ".join(f"{f:.0f}" for f in sweep)
                )
                + "."
            ),
            "tier_note": (
                f"Difficulty bucket per task from the rubric pass "
                f"({judged_meta.get('bucketing')} bucketing); every model's "
                "existing scores regrouped into bucket columns (easy, moderate, "
                "hard, very hard shown as trivial, low, medium, high). The "
                "dataset's own label is kept beside each row."
                if judged_meta.get("mode") == "rubric"
                else "Classified per task by the same judged run, NOT read from the "
                "dataset. Each row carries the dataset's own `complexity` label "
                "beside it so disagreement is visible."
                if judged_meta
                else "Taken from each task's `complexity` field in the dataset, "
                "which hands the router a perfect classifier it would not have "
                "in real use."
            ),
            "floors_label": (
                f"per-task floors judged by {judged_meta.get('harness')} + "
                f"{judged_meta.get('model')} running the skill's step 1"
                if judged_meta
                else "per-task floors from --floors-file"
                if file_floors
                else ", ".join(f"{f:.0f}" for f in sweep)
            ),
        },
        "provenance": provenance,
        "runs": runs,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_document(report), encoding="utf-8")
    logger.info("wrote %s", args.out_json)
    logger.info("wrote %s", args.out_md)


if __name__ == "__main__":
    main()
