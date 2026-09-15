#!/usr/bin/env python3
"""Repair omp token counts that the ``agent_end``-scoped extractor undercounted.

WHY THIS EXISTS (issue #157)
----------------------------
``_pi_result_from_events`` used to sum per-message usage over
``agent_end.messages`` -- the settled conversation carried on the final
``agent_end`` event. omp emits ``agent_start`` more than once (context
compaction, and the todo reminder that nudges the agent to keep going), and
every extra ``agent_start`` RESETS that message list. So ``agent_end`` reports
only the messages since the last restart and drops every token before it.

Measured across the 231 saved ``omp-stream.jsonl`` files:

  * 200 single-``agent_start`` streams -- the whole-stream sum equals the
    ``agent_end`` sum, per message, to the token. Those runs were never wrong.
  * 30 multi-``agent_start`` streams -- ``agent_end.messages`` is an exact
    SUFFIX of the stream, never a different value.
  * 25 runs lost tokens, by 14x to 704x on output.

The bug also corrupts ``token_accounting.compute_total_tokens_processed``, which
decides whether the cache fields are ADDITIVE or a PARTITION of input by testing
``cache_read + cache_write ~= input_tokens``. A truncated input fails that test,
so Prometheus cache is added on top and DOUBLE COUNTED -- which is why one model
came out too expensive rather than too cheap.

``run-swe-headless.py`` now sums the stream, so new runs are correct. This script
repairs the runs already on disk.

TWO REPAIR MODES
----------------
``exact``    The run has a stream that covers it completely: re-sum
             ``message_end`` usage across the whole file. Retried runs share one
             stream (the log is opened in append mode), so the file is split on
             ``agent_end`` boundaries and summed per invocation, matching how the
             harness aggregates.

``imputed``  The stream is gone. Detect breakage from ``metrics.json`` alone, then
             estimate from ``num_turns`` -- which this bug leaves INTACT, because
             turns are counted from whole-stream ``turn_start`` events. The
             estimate is the complexity cohort's median tokens-per-turn times this
             run's real turn count. Validated against the 25 ground-truth runs:
             mean error -1.8%, worst -3.3%, against -15.5%/-37.5% for dropping the
             run and -9.4%/-28.6% for dropping it within its complexity cohort.

WHAT IS WRITTEN
---------------
Canonical fields are updated IN PLACE so every downstream consumer reads the
repaired number without knowing this script exists. A ``token_accounting_repair``
block records the method, the detector that fired, and the original values, so
nothing is destroyed and any row can be audited. Both ``metrics.json`` and the
model's ``run-summary.json`` (per-task rows plus the cost mean) are updated.

Cache fields are overwritten from the stream ONLY when the stream reports them
(Bedrock meters cache per message). Against vLLM the stream reports zero and the
real reuse comes from the Prometheus block, so those are preserved.

Usage:
    uv run scripts/repair_omp_tokens.py --dry-run
    uv run scripts/repair_omp_tokens.py --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import date
from pathlib import Path
from typing import Any

from token_accounting import compute_total_tokens_processed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _SCRIPTS_DIR.parent / "swe-benchmark-data"
STREAM_FILENAME = "omp-stream.jsonl"
METRICS_FILENAME = "metrics.json"
RUN_SUMMARY_FILENAME = "run-summary.json"
REPAIR_KEY = "token_accounting_repair"
USAGE_FIELDS = ("input", "output", "cacheRead", "cacheWrite")

# Detector thresholds, calibrated on the 231 stream-verified runs.
#
# output-per-turn below 100 catches 19 of the 23 breakages that carry a signature,
# with ZERO false alarms across 208 verified-clean runs. The harness's own floor of
# 20 caught 12 and a floor of 50 caught 18, so widening costs nothing.
OUTPUT_PER_TURN_FLOOR = 100.0
# cache/input above 1.3 applies only to a self-hosted run carrying Prometheus cache,
# where cache is a PARTITION of input and the healthy ratio sits at 1.00-1.03. A
# truncated input pushes it to 1.6-1.9. It separates perfectly on the verified runs
# and catches breakage that leaves output-per-turn looking healthy.
CACHE_RATIO_CEILING = 1.3
# Below this many turns a per-turn rate means nothing: a 2-turn run is a crash, not
# this bug, and the harness already excludes such runs from its means.
MIN_TURNS_FOR_RATE = 10


def _num(value: Any) -> float:
    """Coerce a possibly-absent metrics field to a number (absent means zero)."""
    return 0 if value is None else value


def _read_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object, returning None when it is missing or unparseable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object back with the repo's two-space, trailing-newline style."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_stream(stream: Path) -> dict[str, Any] | None:
    """Re-sum a run's true usage from its saved omp event stream.

    Sums ``message_end`` usage across the whole file instead of reading the final
    ``agent_end``. Only assistant messages carry a ``usage`` object (``toolResult``,
    ``user`` and ``custom`` message_end events have none), so no role filter is
    needed; ``turn_end`` mirrors ``message_end`` and is skipped or every message
    would count twice.

    Also returns the turn and invocation counts, which the caller uses to confirm
    the stream actually covers the run it sits beside -- one stream on disk is a
    leftover from an earlier attempt and describes different work.

    Args:
        stream: Path to the run's ``omp-stream.jsonl``.

    Returns:
        Summed usage plus ``cost``, ``turns`` and ``invocations``, or None when the
        file carries no usage at all.
    """
    totals: dict[str, Any] = dict.fromkeys(USAGE_FIELDS, 0)
    totals["cost"] = 0.0
    turns = invocations = 0
    seen = False
    try:
        handle = stream.open(encoding="utf-8")
    except OSError:
        return None
    with handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "turn_start":
                turns += 1
                continue
            if kind == "agent_end":
                invocations += 1
                continue
            if kind != "message_end":
                continue
            usage = (event.get("message") or {}).get("usage")
            if not isinstance(usage, dict):
                continue
            seen = True
            for field in USAGE_FIELDS:
                totals[field] += usage.get(field) or 0
            cost = usage.get("cost")
            if isinstance(cost, dict):
                cost = cost.get("total")
            if isinstance(cost, (int, float)):
                totals["cost"] += cost
    if not seen:
        return None
    totals["turns"] = turns
    totals["invocations"] = max(invocations, 1)
    return totals


def stream_covers_run(stream_totals: dict[str, Any], metrics: dict[str, Any]) -> bool:
    """Return True when the stream describes the same run ``metrics.json`` records.

    The stream log is opened in append mode, so a retried run's invocations share
    one file -- but a run whose earlier attempt was streamed on a different day can
    leave a stale file that covers only part of the work. Repairing from it would
    REPLACE good totals with smaller ones. Turn count and invocation count are both
    untouched by the token bug, so they are a reliable identity check.

    Args:
        stream_totals: The result of :func:`read_stream`.
        metrics: The run's parsed ``metrics.json``.

    Returns:
        True when turns and invocations agree and the stream can be trusted.
    """
    return stream_totals["turns"] == (metrics.get("num_turns") or 0) and stream_totals[
        "invocations"
    ] == (metrics.get("agent_invocations") or 1)


def detect_broken(metrics: dict[str, Any]) -> str | None:
    """Decide whether a stream-less run lost tokens, from ``metrics.json`` alone.

    Two signals, both calibrated against the stream-verified runs: an implausibly
    low output-per-turn, and -- for a self-hosted run carrying Prometheus cache --
    a cache/input ratio that has drifted off the partition signature because
    ``input_tokens`` was truncated.

    Args:
        metrics: The run's parsed ``metrics.json``.

    Returns:
        The name of the detector that fired, or None when the run looks sound.
    """
    turns = _num(metrics.get("num_turns"))
    output = _num(metrics.get("output_tokens"))
    input_tokens = _num(metrics.get("input_tokens"))
    cache = _num(metrics.get("cache_read_tokens")) + _num(
        metrics.get("cache_creation_tokens")
    )
    if input_tokens > 0 and cache > 0 and cache / input_tokens > CACHE_RATIO_CEILING:
        return "cache_input_ratio"
    if turns >= MIN_TURNS_FOR_RATE and output / turns < OUTPUT_PER_TURN_FLOOR:
        return "output_per_turn"
    return None


def impute_from_turns(
    target: dict[str, Any], cohort: list[dict[str, Any]]
) -> dict[str, int]:
    """Estimate a broken run's token counts from its (uncorrupted) turn count.

    ``num_turns`` comes from whole-stream ``turn_start`` events, so this bug never
    touched it -- which is exactly why turns and latency stayed correct while the
    token columns collapsed. Tokens track turns closely, so the cohort's median
    per-turn rate times this run's real turns recovers the count to within a few
    percent.

    Prefers peers of the same complexity, since a trivial task and a high one burn
    very different amounts per turn, and widens to the whole model when that cohort
    has no healthy member left.

    Args:
        target: The broken run's record (needs ``turns`` and ``complexity``).
        cohort: The model's healthy runs, used as the rate reference.

    Returns:
        Estimated ``input_tokens`` and ``output_tokens``.
    """
    peers = [
        r for r in cohort if r["complexity"] == target["complexity"] and r["turns"]
    ]
    if not peers:
        peers = [r for r in cohort if r["turns"]]
    if not peers:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": round(
            statistics.median(r["input"] / r["turns"] for r in peers) * target["turns"]
        ),
        "output_tokens": round(
            statistics.median(r["output"] / r["turns"] for r in peers) * target["turns"]
        ),
    }


def collect_runs(model_dir: Path) -> list[dict[str, Any]]:
    """Read every task under one model's omp run directory.

    Args:
        model_dir: The ``<model>/omp/<skill>/<repo>`` directory.

    Returns:
        One record per task carrying its metrics, its stream totals when the
        stream both exists and covers the run, and the repair verdict.
    """
    runs: list[dict[str, Any]] = []
    for task_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        metrics = _read_json(task_dir / METRICS_FILENAME)
        if not metrics:
            continue
        stream_totals = None
        stream = task_dir / STREAM_FILENAME
        if stream.exists():
            totals = read_stream(stream)
            if totals and stream_covers_run(totals, metrics):
                stream_totals = totals
            elif totals:
                logger.warning(
                    "stale stream ignored for %s/%s: stream has %s turns / %s "
                    "invocations, metrics.json records %s / %s",
                    model_dir.parts[-4],
                    task_dir.name,
                    totals["turns"],
                    totals["invocations"],
                    metrics.get("num_turns"),
                    metrics.get("agent_invocations"),
                )
        runs.append(
            {
                "dir": task_dir,
                "task": task_dir.name,
                "metrics": metrics,
                "stream": stream_totals,
                "complexity": metrics.get("complexity"),
                "turns": _num(metrics.get("num_turns")),
                "input": _num(metrics.get("input_tokens")),
                "output": _num(metrics.get("output_tokens")),
            }
        )
    return runs


def plan_repairs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decide what each run's corrected token counts should be.

    Runs with a trustworthy stream are recomputed exactly. Runs without one are
    tested by :func:`detect_broken` and, when broken, estimated from turns using
    the model's healthy runs as the rate reference. A run that is already correct
    gets no entry.

    Args:
        runs: The model's task records from :func:`collect_runs`.

    Returns:
        One repair plan per run that needs changing.
    """
    plans: list[dict[str, Any]] = []
    for run in runs:
        if run["stream"] is None:
            continue
        totals = run["stream"]
        if totals["input"] == run["input"] and totals["output"] == run["output"]:
            continue
        fields = {
            "input_tokens": totals["input"],
            "output_tokens": totals["output"],
        }
        # Bedrock meters cache per message, and that count is truncated by the same
        # bug. Against vLLM the stream reports zero and the real reuse comes from
        # the Prometheus block -- overwriting there would destroy good data.
        if totals["cacheRead"]:
            fields["cache_read_tokens"] = totals["cacheRead"]
        if totals["cacheWrite"]:
            fields["cache_creation_tokens"] = totals["cacheWrite"]
        if totals["cost"]:
            fields["total_cost_usd"] = totals["cost"]
        plans.append(
            {
                "run": run,
                "method": "exact_from_stream",
                "detector": None,
                "fields": fields,
            }
        )
    # Healthy peers for imputation: a run is a usable rate reference when it has a
    # trustworthy stream (so its counts are known good) or no detector fires on it.
    healthy = [
        r
        for r in runs
        if r["turns"]
        and (r["stream"] is not None or detect_broken(r["metrics"]) is None)
    ]
    for run in runs:
        if run["stream"] is not None:
            continue
        detector = detect_broken(run["metrics"])
        if detector is None:
            continue
        cohort = [
            {
                "complexity": r["complexity"],
                "turns": r["turns"],
                "input": (r["stream"]["input"] if r["stream"] else r["input"]),
                "output": (r["stream"]["output"] if r["stream"] else r["output"]),
            }
            for r in healthy
            if r is not run
        ]
        plans.append(
            {
                "run": run,
                "method": "imputed_from_turns",
                "detector": detector,
                "fields": impute_from_turns(run, cohort),
            }
        )
    return plans


def apply_plan(plan: dict[str, Any], today: str) -> dict[str, Any]:
    """Rewrite one run's ``metrics.json`` with the corrected counts.

    Canonical fields are replaced in place; the pre-repair values move into a
    ``token_accounting_repair`` block alongside the method and detector, so the
    change is auditable and nothing is lost. ``total_tokens`` is recomputed through
    ``compute_total_tokens_processed`` because correcting ``input_tokens`` can flip
    that module's partition test -- which is a second way this bug distorted cost.

    Args:
        plan: One entry from :func:`plan_repairs`.
        today: ISO date recorded as ``repaired_at``.

    Returns:
        The updated metrics dict (also written to disk by the caller).
    """
    metrics = plan["run"]["metrics"]
    original = {
        key: metrics.get(key)
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
            "total_cost_usd",
        )
    }
    metrics.update(plan["fields"])
    metrics[REPAIR_KEY] = {
        "method": plan["method"],
        "detector": plan["detector"],
        "issue": 157,
        "repaired_at": today,
        # Which canonical fields this repair actually produced. The run-summary
        # update keys off this so it never overwrites a field the repair did not
        # compute -- notably the Prometheus cache counts, which live only in the
        # summary rows for a self-hosted run.
        "fields_replaced": sorted(plan["fields"]),
        "original": original,
    }
    # The harness's own suspicion note described the pre-repair numbers.
    if metrics.get("token_accounting_warning"):
        metrics["token_accounting_warning"] = None
    return metrics


def update_run_summary(model_dir: Path, runs: list[dict[str, Any]], today: str) -> bool:
    """Rewrite the model's ``run-summary.json`` from the repaired per-task metrics.

    The cost/quality chart prefers ``run-summary.json`` over the per-task files, so
    leaving it stale would leave every chart stale. Per-task rows are refreshed
    from ``metrics.json`` and the cost mean is recomputed over the same non-failed
    tasks the summary already excludes.

    Args:
        model_dir: The model's omp run directory.
        runs: The model's task records, with metrics already repaired.
        today: ISO date recorded as ``repaired_at``.

    Returns:
        True when the summary was rewritten.
    """
    path = model_dir / RUN_SUMMARY_FILENAME
    summary = _read_json(path)
    if not summary:
        return False
    by_task = {r["task"]: r["metrics"] for r in runs}
    failed = set(summary.get("failed_tasks") or [])
    costs: list[float] = []
    touched = False
    for row in summary.get("tasks") or []:
        metrics = by_task.get(row.get("task"))
        if not metrics or REPAIR_KEY not in metrics:
            if row.get("task") not in failed and row.get("total_cost_usd") is not None:
                costs.append(row["total_cost_usd"])
            continue
        touched = True
        row["input_tokens"] = metrics.get("input_tokens")
        row["output_tokens"] = metrics.get("output_tokens")
        # Cache stays where it is unless the repair actually produced a new value.
        # summarize_run.py fills these rows from the server-side Prometheus block,
        # which metrics.json does NOT carry for a vLLM run -- a self-hosted row can
        # read cache_read 6,600,528 while its metrics.json reads 0. Copying
        # metrics.json across unconditionally wipes the only record of that reuse,
        # and the wipe is invisible in the chart because zero cache still yields a
        # plausible (smaller) total.
        replaced = metrics[REPAIR_KEY].get("fields_replaced") or []
        if "cache_read_tokens" in replaced:
            row["cache_read_tokens"] = metrics.get("cache_read_tokens")
        if "cache_creation_tokens" in replaced:
            row["cache_write_tokens"] = metrics.get("cache_creation_tokens")
        # Recompute from the ROW, not from metrics.json: the row is the one that
        # carries both the repaired counts and the Prometheus cache.
        row["total_tokens"] = compute_total_tokens_processed(
            int(_num(row.get("input_tokens"))),
            int(_num(row.get("output_tokens"))),
            int(_num(row.get("cache_read_tokens"))),
            int(_num(row.get("cache_write_tokens"))),
        )
        if "total_cost_usd" in replaced:
            row["total_cost_usd"] = metrics["total_cost_usd"]
        row[REPAIR_KEY] = metrics[REPAIR_KEY]["method"]
        if row.get("task") not in failed and row.get("total_cost_usd") is not None:
            costs.append(row["total_cost_usd"])
    if not touched:
        return False
    if costs:
        summary["mean_cost_usd_excl_failed"] = round(statistics.mean(costs), 4)
    summary[REPAIR_KEY] = {"issue": 157, "repaired_at": today}
    _write_json(path, summary)
    return True


def _parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="swe-benchmark-data root",
    )
    parser.add_argument("--agent", default="omp", help="agent directory to repair")
    parser.add_argument("--skill", default="swe3", help="skill directory to repair")
    parser.add_argument(
        "--repo", default="mcp-gateway-registry-v2", help="scope directory"
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the repairs (default is a dry run)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report only (the default)"
    )
    return parser.parse_args()


def main() -> int:
    """Repair every omp run under the requested scope.

    Returns:
        0 on success.
    """
    args = _parse_args()
    today = date.today().isoformat()
    model_dirs = sorted(
        p
        for p in args.data_dir.glob(f"*/{args.agent}/{args.skill}/{args.repo}")
        if p.is_dir()
    )
    if not model_dirs:
        logger.error("no model directories under %s", args.data_dir)
        return 1

    grand_exact = grand_imputed = grand_clean = 0
    for model_dir in model_dirs:
        model = model_dir.parts[-4]
        runs = collect_runs(model_dir)
        plans = plan_repairs(runs)
        exact = [p for p in plans if p["method"] == "exact_from_stream"]
        imputed = [p for p in plans if p["method"] == "imputed_from_turns"]
        grand_exact += len(exact)
        grand_imputed += len(imputed)
        grand_clean += len(runs) - len(plans)
        if not plans:
            logger.info("%-20s %2d runs, all correct", model, len(runs))
            continue
        logger.info(
            "%-20s %2d runs: %d exact, %d imputed, %d already correct",
            model,
            len(runs),
            len(exact),
            len(imputed),
            len(runs) - len(plans),
        )
        for plan in plans:
            run = plan["run"]
            logger.info(
                "    %-14s %-48s out %9s -> %9s  %s",
                plan["method"].split("_")[0],
                run["task"],
                f"{int(run['output']):,}",
                f"{int(plan['fields']['output_tokens']):,}",
                plan["detector"] or "",
            )
            if args.apply:
                metrics = apply_plan(plan, today)
                _write_json(run["dir"] / METRICS_FILENAME, metrics)
        if args.apply:
            update_run_summary(model_dir, runs, today)

    logger.info(
        "%s: %d exact, %d imputed, %d already correct",
        "APPLIED" if args.apply else "DRY RUN (nothing written)",
        grand_exact,
        grand_imputed,
        grand_clean,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
