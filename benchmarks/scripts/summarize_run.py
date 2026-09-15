#!/usr/bin/env python3
"""Summarize one model+dataset benchmark run into run-summary.json and .md.

Reads a ``{model-slug}/{harness}/{scope}/`` folder under ``swe-benchmark-data``
(one subfolder per task, each with ``metrics.json`` and, when scored, ``eval.json``)
and writes two sibling files:

  * ``run-summary.json`` -- machine-readable, for later charting / aggregation.
  * ``run-summary.md``   -- human-readable, rendered from the same data.

A task that scored 0 is treated as a model failure (missing/empty artifacts) and
is EXCLUDED from the headline mean (score and cost), matching the leaderboard
convention; it is still listed with its 0 so the failure stays visible.

Usage:
    uv run scripts/summarize_run.py --folder ../swe-benchmark-data/gemma-4-31b/claude-code/mcp-gateway-registry
    uv run scripts/summarize_run.py --folder <dir> --run-date 2026-07-24
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from token_accounting import cache_partition_for_agent, compute_total_tokens_processed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Everything a full /swe2 run emits: the four design artifacts plus the
# implementation artifact (patch.diff + implementation.md). The produced count
# in the run summary is out of all six.
ARTIFACT_FILENAMES = (
    "github-issue.md",
    "lld.md",
    "review.md",
    "testing.md",
    "patch.diff",
    "implementation.md",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    """Return the parsed JSON object at ``path``, or None if absent/invalid."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _task_row(task_dir: Path) -> dict[str, Any] | None:
    """Build one task's summary row from its metrics.json and eval.json.

    Args:
        task_dir: A single task's artifact directory.

    Returns:
        A row dict, or None if the folder has no metrics.json (not a task).
    """
    metrics = _read_json(task_dir / "metrics.json")
    if metrics is None:
        return None
    # The normalized cross-agent block ("metrics"); fall back to the legacy
    # "metrics_that_matter" alias for older metrics.json files.
    mm = metrics.get("metrics") or metrics.get("metrics_that_matter", {}) or {}
    eval_data = _read_json(task_dir / "eval.json")
    score = None
    if eval_data and isinstance(eval_data.get("task_score"), (int, float)):
        score = float(eval_data["task_score"])
    produced = sum(1 for f in ARTIFACT_FILENAMES if (task_dir / f).exists())
    # KV-cache footprint (peak/mean) sampled while the run was in flight. Lives
    # under the vllm_prometheus gauges; absent on the Bedrock path (no /metrics).
    kv = (
        (metrics.get("vllm_prometheus", {}) or {})
        .get("gauges_sampled", {})
        .get("gauges", {})
        .get("vllm:kv_cache_usage_perc")
    )
    kv_usage = (
        {"peak": kv.get("peak"), "mean": kv.get("mean")}
        if isinstance(kv, dict)
        else None
    )
    return {
        "task": task_dir.name,
        "complexity": metrics.get("complexity"),
        # The tag this task cloned. Recorded per task because a dataset may pin a
        # different ref per task (v2-style, each task at the release before its
        # fix), so a single run-level ref would not describe the run.
        "ref": metrics.get("ref"),
        "artifacts_produced": produced,
        "artifacts_expected": len(ARTIFACT_FILENAMES),
        "num_turns": mm.get("num_turns"),
        "input_tokens": mm.get("input_tokens"),
        "output_tokens": mm.get("output_tokens"),
        # total_tokens = total tokens processed once each, ALWAYS recomputed here
        # from the per-field counts via compute_total_tokens_processed (issue
        # #136). We deliberately do NOT trust mm["total_tokens"]: upstream used to
        # write it as input+output+cache_read+cache_write unconditionally, which
        # ~2x double-counted self-hosted partition runs (where cache_read/write
        # already live inside input_tokens). Recomputing here keeps the derived
        # total consistent regardless of what the metrics.json carried. The agent
        # named in the file decides whether the cache shape is declared (disjoint
        # counts, e.g. codex) or detected from the data (issue #183).
        "total_tokens": compute_total_tokens_processed(
            mm.get("input_tokens") or 0,
            mm.get("output_tokens") or 0,
            mm.get("cache_read_tokens") or 0,
            mm.get("cache_write_tokens") or mm.get("cache_creation_tokens") or 0,
            context=f"summarize_run:{task_dir.name}",
            cache_partition=cache_partition_for_agent(metrics.get("agent")),
        ),
        "latency_seconds": mm.get("latency_seconds"),
        # Cost from the normalized block (which now carries it); fall back to the
        # top-level field for older metrics.json.
        "total_cost_usd": mm.get("total_cost_usd", metrics.get("total_cost_usd")),
        # Why the run ended (success / error_max_turns / ...), so a topped-up or
        # failed task is diagnosable from the committed rollup alone.
        "result_subtype": metrics.get("result_subtype"),
        "task_score": score,
        # Derived serving-efficiency signals, folded up from metrics.json so the
        # committed rollup carries them even though the per-task metrics.json is
        # gitignored (and the raw vllm_prometheus dump is not). These are what let
        # runs be compared on cache/KV efficiency, not just tokens (see the
        # cost-per-task methodology). Any may be None: cache/KV come from the vLLM
        # Prometheus surface, so they are absent on the Bedrock path.
        "cache_read_tokens": mm.get("cache_read_tokens"),
        "cache_write_tokens": mm.get("cache_write_tokens"),
        "prefix_cache_hit_rate": mm.get("prefix_cache_hit_rate"),
        "generation_tokens_per_sec": mm.get("generation_tokens_per_sec"),
        "kv_cache_usage": kv_usage,
        # Top-up provenance: >1 invocation means the artifact set was completed by
        # a focused top-up pass, not a single clean run (see the harness's
        # completion loop). Defaults keep a normal single-shot run at 1 / [].
        "agent_invocations": metrics.get("agent_invocations", 1),
        "topped_up_artifacts": metrics.get("topped_up_artifacts", []),
        # Embed the judge's per-artifact criterion breakdown (from eval.json) so
        # run-summary.json is self-contained -- the committed rollup carries the
        # scores + judge notes even though the per-task eval.json is gitignored.
        # None when the task was not scored (a failure or no judge run).
        "eval_scores": (eval_data or {}).get("scores"),
        "is_error": metrics.get("is_error"),
        "failed": not score,  # 0 or missing score == model failure
    }


def _summarize(folder: Path, run_date: str | None) -> dict[str, Any]:
    """Aggregate every task folder under ``folder`` into a summary dict.

    Args:
        folder: The ``{model-slug}/{scope}/`` run directory.
        run_date: Optional ISO date to stamp; omitted when None.

    Returns:
        The structured summary (also written as run-summary.json).

    Raises:
        SystemExit: If no task folders with metrics.json are found.
    """
    rows = [
        row
        for task_dir in sorted(p for p in folder.iterdir() if p.is_dir())
        if (row := _task_row(task_dir)) is not None
    ]
    if not rows:
        raise SystemExit(f"no task folders with metrics.json under {folder}")

    # Identity/serving from the first task's metrics (uniform across a run).
    first = _read_json(folder / rows[0]["task"] / "metrics.json") or {}
    # run-summary.json is committed to git, so drop the judge block's local-only
    # temp path (repo_root, e.g. /tmp/swe-judge-repos/...); it is machine-specific
    # noise, not provenance worth committing.
    judge = dict((first.get("evaluation") or {}).get("judge") or {})
    judge.pop("repo_root", None)
    refs = sorted({r["ref"] for r in rows if r.get("ref")})
    scored = [r for r in rows if not r["failed"]]
    failed = [r for r in rows if r["failed"]]
    mean_score = (
        round(sum(r["task_score"] for r in scored) / len(scored), 2) if scored else None
    )
    costs = [r["total_cost_usd"] for r in scored if r["total_cost_usd"] is not None]
    mean_cost = round(sum(costs) / len(costs), 2) if costs else None

    # Layout: <model-slug>/<harness>/<skill>/<repo>. folder is the <repo> (scope)
    # dir. Prefer the identity fields the metrics.json records (model_slug, agent,
    # skill); fall back to path position for older files. Path fallbacks assume the
    # new 4-level depth (skill=parent, harness=parent.parent, model=parent^3).
    summary: dict[str, Any] = {
        "model": first.get("model"),
        "model_slug": first.get("model_slug") or folder.parent.parent.parent.name,
        "agent": first.get("agent") or folder.parent.parent.name,
        "skill": first.get("skill") or folder.parent.name,
        "scope": folder.name,
        "provider": first.get("provider"),
        # One ref when every task cloned the same tag (the common case), else
        # None -- a multi-ref dataset has no single run-level ref, and reporting
        # the first task's would be wrong. "refs" always lists what was cloned.
        "ref": refs[0] if len(refs) == 1 else None,
        "refs": refs,
        "serving": first.get("serving"),
        "judge": judge or None,
        "num_tasks": len(rows),
        "num_scored": len(scored),
        "num_failed": len(failed),
        "failed_tasks": [r["task"] for r in failed],
        "mean_task_score_excl_failed": mean_score,
        "mean_cost_usd_excl_failed": mean_cost,
        "tasks": sorted(rows, key=lambda r: r["task_score"] or -1, reverse=True),
    }
    if run_date:
        summary["run_date"] = run_date
    return summary


def _refs_phrase(summary: dict[str, Any]) -> str:
    """Describe the ref(s) a run cloned, for the summary header line.

    Args:
        summary: The summary dict, carrying "ref" and "refs".

    Returns:
        ``ref 1.24.4`` for a single-ref run, or ``8 refs: ...`` when the dataset
        pins a different tag per task.
    """
    refs = summary.get("refs") or ([summary["ref"]] if summary.get("ref") else [])
    if not refs:
        return "ref unknown"
    if len(refs) == 1:
        return f"ref {refs[0]}"
    return f"{len(refs)} refs: {', '.join(refs)}"


def _render_markdown(summary: dict[str, Any]) -> str:
    """Render the human-readable run-summary.md from the summary dict."""
    s = summary
    serving = s.get("serving") or {}
    serving_line = (
        ", ".join(f"{k}={v}" for k, v in serving.items() if v is not None) or "n/a"
    )
    headline = (
        f"{s['num_scored']} of {s['num_tasks']} tasks scored"
        + (
            f"; {s['num_failed']} failed ({', '.join(s['failed_tasks'])}), "
            "excluded from the mean"
            if s["num_failed"]
            else "; no failures"
        )
        + "."
    )
    lines = [
        f"# Benchmark run summary: {s['model']} on {s['scope']}",
        "",
        f"- Model: {s['model']}",
        f"- Agent (harness): {s.get('agent')}",
        f"- Skill: {s.get('skill')}",
        f"- Provider: {s['provider']}",
        f"- Dataset scope: {s['scope']} ({s['num_tasks']} tasks, {_refs_phrase(s)})",
        f"- Serving: {serving_line}",
    ]
    if s.get("run_date"):
        lines.append(f"- Run date: {s['run_date']}")
    lines += [
        "",
        headline,
        "",
        "## Results",
        "",
        "| Task | Artifacts | Turns | Prefix-cache | Cost (est $) | Judge score |",
        "|---|---|---|---|---|---|",
    ]
    for r in s["tasks"]:
        score = "0.0 (model failure)" if r["failed"] else r["task_score"]
        cost = f"{r['total_cost_usd']:.2f}" if r["total_cost_usd"] is not None else "--"
        hit = r.get("prefix_cache_hit_rate")
        cache = f"{hit * 100:.1f}%" if isinstance(hit, (int, float)) else "--"
        # Flag a task whose artifact set was completed by a focused top-up pass
        # (more than one agent invocation) rather than a single clean run.
        topup = (
            f" (topped up: {', '.join(r['topped_up_artifacts'])})"
            if r.get("agent_invocations", 1) > 1 and r.get("topped_up_artifacts")
            else " (top-up attempted)"
            if r.get("agent_invocations", 1) > 1
            else ""
        )
        lines.append(
            f"| {r['task']}{topup} | {r['artifacts_produced']}/{r['artifacts_expected']} "
            f"| {r['num_turns']} | {cache} | {cost} | {score} |"
        )
    lines += [
        "",
        f"Mean over the {s['num_scored']} completed tasks: "
        f"{s['mean_task_score_excl_failed']} "
        f"(mean cost ${s['mean_cost_usd_excl_failed']}). A 0-score task is a model "
        "failure (missing artifacts) and is excluded from the means, pending "
        "investigation. Cost is a token-based estimate for self-hosted models.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Write run-summary.json and run-summary.md for a benchmark run.",
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="The {model-slug}/{scope}/ run directory under swe-benchmark-data.",
    )
    parser.add_argument(
        "--run-date",
        default=None,
        help="Optional ISO date to stamp in the summary (e.g. 2026-07-24).",
    )
    return parser.parse_args()


def main() -> None:
    """Summarize a run folder into run-summary.json and run-summary.md."""
    args = _parse_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"not a directory: {folder}")
    summary = _summarize(folder, args.run_date)

    json_path = folder / "run-summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    md_path = folder / "run-summary.md"
    md_path.write_text(_render_markdown(summary), encoding="utf-8")
    logger.info(
        "wrote %s and %s (%d scored, %d failed, mean %.2f)",
        json_path,
        md_path,
        summary["num_scored"],
        summary["num_failed"],
        summary["mean_task_score_excl_failed"] or 0.0,
    )


if __name__ == "__main__":
    main()
