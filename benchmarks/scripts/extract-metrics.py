#!/usr/bin/env python3
"""Extract metrics from Claude Code session JSONL files.

Parses session transcripts to compute per-run metrics:
- Input/output/thinking tokens
- Cache read/write tokens
- Cost (computed from wall-clock × instance cost for self-hosted)
- Tool call count
- Error count
- Prompt/generation throughput (tokens/sec)

Usage:
    python3 extract-metrics.py <session-jsonl> <metrics-json-to-update>
    python3 extract-metrics.py --session-id <id> <metrics-json-to-update>
    python3 extract-metrics.py --latest <metrics-json-to-update>

The script updates an existing metrics.json with token/cache/throughput fields.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime


def find_session_jsonl(session_id=None, latest=False):
    """Find session JSONL file."""
    base = Path.home() / ".claude" / "projects"

    if session_id:
        for jsonl in base.rglob(f"{session_id}.jsonl"):
            return jsonl

    if latest:
        all_jsonls = list(base.rglob("*.jsonl"))
        if all_jsonls:
            return max(all_jsonls, key=lambda p: p.stat().st_mtime)

    return None


def extract_from_jsonl(jsonl_path):
    """Parse a session JSONL and extract metrics."""
    metrics = {
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": 0,
        "errors": 0,
        "api_calls": 0,
        "first_timestamp": None,
        "last_timestamp": None,
    }

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Track timestamps
            ts = entry.get("timestamp") or entry.get("ts")
            if ts:
                if metrics["first_timestamp"] is None:
                    metrics["first_timestamp"] = ts
                metrics["last_timestamp"] = ts

            # Extract usage from API responses
            usage = entry.get("usage") or {}
            if not usage:
                # Check nested in message
                msg = entry.get("message") or entry.get("response") or {}
                usage = msg.get("usage") or {}

            if usage:
                metrics["api_calls"] += 1
                metrics["input_tokens"] += usage.get("input_tokens", 0)
                metrics["output_tokens"] += usage.get("output_tokens", 0)
                metrics["cache_read_tokens"] += usage.get("cache_read_input_tokens", 0) or usage.get("cache_read", 0)
                metrics["cache_write_tokens"] += usage.get("cache_creation_input_tokens", 0) or usage.get("cache_write", 0)

            # Count tool uses
            entry_type = entry.get("type") or ""
            if entry_type == "tool_use" or entry.get("tool_use"):
                metrics["tool_calls"] += 1

            # Count content blocks with tool_use type
            content = entry.get("content") or []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        metrics["tool_calls"] += 1

            # Count errors
            if entry.get("error") or entry_type == "error":
                metrics["errors"] += 1

    # Compute wall clock from timestamps
    if metrics["first_timestamp"] and metrics["last_timestamp"]:
        try:
            t1 = datetime.fromisoformat(metrics["first_timestamp"].replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(metrics["last_timestamp"].replace("Z", "+00:00"))
            metrics["wall_clock_seconds"] = (t2 - t1).total_seconds()
        except (ValueError, TypeError):
            pass

    # Compute throughput
    wall = metrics.get("wall_clock_seconds", 0)
    if wall and wall > 0:
        metrics["generation_tokens_per_sec"] = round(metrics["output_tokens"] / wall, 1)
        metrics["prompt_tokens_per_sec"] = round(metrics["input_tokens"] / wall, 1)

    # Clean up internal fields
    del metrics["first_timestamp"]
    del metrics["last_timestamp"]

    return metrics


def update_metrics_json(metrics_path, extracted):
    """Update an existing metrics.json with extracted data."""
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            existing = json.load(f)
    else:
        existing = {}

    # Update with extracted values (don't overwrite non-null existing values)
    for key, value in extracted.items():
        if value is not None and value != 0:
            existing[key] = value

    # Compute task cost if we have wall_clock and instance cost
    wall = existing.get("wall_clock_seconds", 0)
    cost_per_hr = existing.get("instance_cost_per_hr", 0)
    if wall and cost_per_hr:
        existing["task_cost_usd"] = round((wall / 3600) * cost_per_hr, 4)

    # Cache hit rate
    cache_read = existing.get("cache_read_tokens", 0)
    total_input = existing.get("input_tokens", 0)
    if total_input > 0:
        existing["cache_hit_rate"] = round(cache_read / total_input, 3)

    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)

    return existing


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    jsonl_path = None
    metrics_path = None

    args = sys.argv[1:]

    if args[0] == "--latest":
        jsonl_path = find_session_jsonl(latest=True)
        metrics_path = args[1] if len(args) > 1 else None
    elif args[0] == "--session-id":
        jsonl_path = find_session_jsonl(session_id=args[1])
        metrics_path = args[2] if len(args) > 2 else None
    else:
        jsonl_path = Path(args[0])
        metrics_path = args[1] if len(args) > 1 else None

    if not jsonl_path or not jsonl_path.exists():
        print(f"Error: Could not find session JSONL: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting from: {jsonl_path}")
    extracted = extract_from_jsonl(jsonl_path)

    print(f"  API calls: {extracted['api_calls']}")
    print(f"  Input tokens: {extracted['input_tokens']:,}")
    print(f"  Output tokens: {extracted['output_tokens']:,}")
    print(f"  Cache read: {extracted['cache_read_tokens']:,}")
    print(f"  Cache write: {extracted['cache_write_tokens']:,}")
    print(f"  Tool calls: {extracted['tool_calls']}")
    print(f"  Errors: {extracted['errors']}")

    if metrics_path:
        updated = update_metrics_json(metrics_path, extracted)
        print(f"\nUpdated: {metrics_path}")
        if "task_cost_usd" in updated:
            print(f"  Task cost: ${updated['task_cost_usd']:.4f}")
        if "cache_hit_rate" in updated:
            print(f"  Cache hit rate: {updated['cache_hit_rate']*100:.1f}%")
    else:
        print(json.dumps(extracted, indent=2))


if __name__ == "__main__":
    main()
