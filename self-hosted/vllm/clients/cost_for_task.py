#!/usr/bin/env python3
"""Cost an ARBITRARY task from a throughput sweep's per-token cost.

The performance summary (see build_performance_summary.py) already carries, at
each concurrency level, the hardware-derived cost per token for this model on
this instance -- a property of the model + hardware + load, NOT of any one task.
So any task run anywhere on the same served model can be costed from its input
and output token counts alone:

    cost = cost_per_input_token * input_tokens + cost_per_output_token * output_tokens

This lets you cost a real /benchmark run, a production trace, or a hypothetical
workload -- anything you have token counts for -- without re-running the sweep.
Both lenses the summary defines are reported:

    * Lens A -- blended (measured): input and output cost the same per token.
    * Lens B -- lab-style split: input priced at ``w`` x output (input cheaper).

Usage:
    uv run python -m clients.cost_for_task \\
        --summary benchmark-output/throughput/qwen3.6-35b/performance-summary.json \\
        --input-tokens 1510243 --output-tokens 30359

    # Cost at one specific concurrency (default: report every level + the cheapest)
    uv run python -m clients.cost_for_task --summary ... \\
        --input-tokens 200000 --output-tokens 8000 --concurrency 20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _task_cost(
    level: dict[str, Any], in_tokens: int, out_tokens: int
) -> dict[str, Any]:
    """Cost one task at one concurrency level under both lenses.

    Args:
        level: A single level dict from the summary's ``levels`` list.
        in_tokens: Input (prompt) tokens for the task.
        out_tokens: Output (generation) tokens for the task.

    Returns:
        The level's concurrency plus blended and split task costs (USD), each
        None when that level had no measured throughput to derive a rate from.
    """
    blended_per = level.get("blended_cost_per_token_usd")
    split_in_per = level.get("split_cost_per_input_token_usd")
    split_out_per = level.get("split_cost_per_output_token_usd")

    blended = (
        blended_per * (in_tokens + out_tokens) if blended_per is not None else None
    )
    split = (
        split_in_per * in_tokens + split_out_per * out_tokens
        if split_in_per is not None and split_out_per is not None
        else None
    )
    return {
        "concurrency": level["concurrency"],
        "output_tokens_per_second": level.get("output_tokens_per_second"),
        "task_cost_blended_usd": round(blended, 4) if blended is not None else None,
        "task_cost_split_usd": round(split, 4) if split is not None else None,
    }


def _cost_across_levels(
    summary: dict[str, Any], in_tokens: int, out_tokens: int, concurrency: int | None
) -> list[dict[str, Any]]:
    """Cost the task at every level (or one level if ``concurrency`` is given)."""
    levels = summary.get("levels", [])
    if concurrency is not None:
        levels = [level for level in levels if level["concurrency"] == concurrency]
        if not levels:
            have = ", ".join(str(level["concurrency"]) for level in summary["levels"])
            raise SystemExit(f"concurrency {concurrency} not in summary (have: {have})")
    return [_task_cost(level, in_tokens, out_tokens) for level in levels]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cost an arbitrary task from a throughput sweep's per-token cost.",
        epilog=(
            "Example:\n"
            "  uv run python -m clients.cost_for_task \\\n"
            "    --summary benchmark-output/throughput/qwen3.6-35b/"
            "performance-summary.json \\\n"
            "    --input-tokens 1510243 --output-tokens 30359\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--summary", required=True, type=Path, help="performance-summary.json path"
    )
    parser.add_argument(
        "--input-tokens", type=int, required=True, help="Task INPUT (prompt) tokens"
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        required=True,
        help="Task OUTPUT (generation) tokens",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Cost at only this concurrency level (default: every level)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the result as JSON instead of a table"
    )
    return parser.parse_args()


def main() -> None:
    """Cost the given task across the summary's concurrency levels."""
    args = _parse_args()
    if not args.summary.is_file():
        raise SystemExit(f"summary not found: {args.summary}")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = _cost_across_levels(
        summary, args.input_tokens, args.output_tokens, args.concurrency
    )

    cheapest = min(
        (r for r in rows if r["task_cost_blended_usd"] is not None),
        key=lambda r: r["task_cost_blended_usd"],
        default=None,
    )
    result = {
        "model": summary.get("model"),
        "instance_type": summary.get("instance_type"),
        "dollars_per_hour": summary.get("dollars_per_hour"),
        "split_input_weight": summary.get("split_input_weight"),
        "task_input_tokens": args.input_tokens,
        "task_output_tokens": args.output_tokens,
        "cheapest_blended_usd": cheapest["task_cost_blended_usd"] if cheapest else None,
        "cheapest_at_concurrency": cheapest["concurrency"] if cheapest else None,
        "levels": rows,
    }

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    print(
        f"{result['model']} on {result['instance_type']} "
        f"(${result['dollars_per_hour']}/hr, split w={result['split_input_weight']})"
    )
    print(
        f"task = {args.input_tokens:,} input : {args.output_tokens:,} output tokens\n"
    )
    print(f"{'concurrency':>11} {'out tok/s':>10} {'$ blended':>11} {'$ split':>11}")
    print("-" * 46)
    for r in rows:
        blended = r["task_cost_blended_usd"]
        split = r["task_cost_split_usd"]
        print(
            f"{r['concurrency']:>11} {str(r['output_tokens_per_second']):>10} "
            f"{('$' + format(blended, '.4f')) if blended is not None else 'n/a':>11} "
            f"{('$' + format(split, '.4f')) if split is not None else 'n/a':>11}"
        )
    if cheapest:
        print(
            f"\ncheapest: ${cheapest['task_cost_blended_usd']:.4f} blended "
            f"at concurrency {cheapest['concurrency']}"
        )


if __name__ == "__main__":
    main()
