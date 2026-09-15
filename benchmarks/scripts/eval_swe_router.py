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
    )


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
        json.dumps(_models_json(_tier_stats(results), provenance)), encoding="utf-8"
    )
    for task in tasks:
        task_id = task["id"]
        # A judged run supplies BOTH halves of step 1, so the tier comes from the
        # judgment rather than the dataset label -- otherwise the eval would hand
        # the router a perfect classifier it would not have in real use.
        tier = judged_tiers.get(task_id, task["complexity"])
        floor = floors.get(task_id, DEFAULT_FLOOR)
        if holdout:
            models_path = tmp_dir / f"models-{task_id}.json"
            models_path.write_text(
                json.dumps(
                    _models_json(_tier_stats(results, exclude_task=task_id), provenance)
                ),
                encoding="utf-8",
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
        # The router's own failure mode: it picked a model to clear a floor and
        # the model then landed under it. Counted for both arms so the baseline
        # is held to the same test.
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
        "Leave-one-out: each task routes from tier means recomputed with that "
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
            )
        except RouteError as exc:
            raise SystemExit(f"routing failed at floor {floor}: {exc}") from exc
        totals = evaluated["totals"]
        logger.info(
            "floor %.0f: %s -> $%.2f vs $%.2f baseline (%.1f%% saved), "
            "mean score %.2f vs %.2f (%+.2f), %d task(s) under floor",
            floor,
            "leave-one-out" if args.holdout else "in-sample",
            totals["routed_total_cost_usd"],
            totals["baseline_total_cost_usd"],
            totals["cost_saving_pct"],
            totals["routed_mean_score"],
            totals["baseline_mean_score"],
            totals["mean_score_delta"],
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
            "judged_by": judged_meta,
            "floor_note": (
                "Judged per task by "
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
                "Classified per task by the same judged run, NOT read from the "
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
