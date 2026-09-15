#!/usr/bin/env python3
"""Replay a conversation JSONL against a vLLM endpoint for performance benchmarking.

Reads a replay.jsonl file (produced by extract-metrics.py or the session parser),
sends each call's messages to the vLLM Anthropic Messages API, and records
per-call metrics: input tokens, output tokens, latency, prompt tok/s, generation tok/s.

Usage:
    python3 replay-benchmark.py <replay.jsonl> <endpoint> [lines]

    replay.jsonl  — path to the replay file (each line = one API call with messages array)
    endpoint      — vLLM base URL (e.g. http://127.0.0.1:8000)
    lines         — number of lines to replay (default: 1, 0 = full file)

Example:
    python3 replay-benchmark.py replay.jsonl http://127.0.0.1:8000 0
    python3 replay-benchmark.py replay.jsonl http://127.0.0.1:8000 5

Output:
    Prints per-call metrics and a summary table to stdout.
    Saves results to replay-results.json in the same directory as the input file.
"""

import argparse
import json
import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def send_request(endpoint, messages, model="kimi-k2.7-code", max_tokens=16000):
    """Send a single request to the vLLM Anthropic Messages API."""
    url = f"{endpoint}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "local",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    start_ms = time.time() * 1000
    response = requests.post(url, headers=headers, json=payload, timeout=600)
    end_ms = time.time() * 1000
    latency_ms = end_ms - start_ms

    if response.status_code != 200:
        return {
            "error": True,
            "status_code": response.status_code,
            "body": response.text[:500],
            "latency_ms": latency_ms,
        }

    data = response.json()
    usage = data.get("usage", {})

    return {
        "error": False,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "latency_ms": round(latency_ms, 1),
        "model": data.get("model", ""),
        "stop_reason": data.get("stop_reason", ""),
    }


def detect_model(endpoint):
    """Auto-detect the served model name from the endpoint."""
    try:
        resp = requests.get(f"{endpoint}/v1/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            real = [m["id"] for m in models if "claude" not in m["id"].lower()]
            if real:
                return real[0]
            if models:
                return models[0]["id"]
    except Exception:  # nosec B110 - best-effort detection, fallback below
        return "kimi-k2.7-code"
    return "kimi-k2.7-code"


def main():
    parser = argparse.ArgumentParser(
        description="Replay a conversation JSONL against a vLLM endpoint for benchmarking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("jsonl", help="Path to the replay.jsonl file")
    parser.add_argument("endpoint", help="vLLM base URL (e.g. http://127.0.0.1:8000)")
    parser.add_argument(
        "lines",
        nargs="?",
        type=int,
        default=1,
        help="Number of lines to replay (default: 1, 0 = full file)",
    )
    parser.add_argument(
        "--model", default=None, help="Model name to use (auto-detected if not set)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=16000, help="Max output tokens per call"
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.jsonl):
        print(f"Error: file not found: {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    # Load replay lines
    with open(args.jsonl) as f:
        all_lines = [line.strip() for line in f if line.strip()]

    total_available = len(all_lines)
    num_lines = total_available if args.lines == 0 else min(args.lines, total_available)

    # Detect model
    model = args.model or detect_model(args.endpoint)

    concurrency = args.concurrency

    print("Replay benchmark")
    print(f"  File: {args.jsonl}")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Model: {model}")
    print(f"  Lines to replay: {num_lines} / {total_available}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Max output tokens: {args.max_tokens}")
    print(f"{'=' * 90}")
    print(
        f"{'Call':>4} | {'Input tok':>10} | {'Output tok':>10} | {'Latency ms':>10} | {'Prompt t/s':>10} | {'Gen t/s':>10} | {'Status'}"
    )
    print(f"{'-' * 90}")

    results = [None] * num_lines
    total_start = time.time()

    def run_call(i):
        call_data = json.loads(all_lines[i])
        messages = call_data.get("messages", [])
        result = send_request(
            args.endpoint, messages, model=model, max_tokens=args.max_tokens
        )
        result["call_number"] = i
        return i, result

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(run_call, i): i for i in range(num_lines)}

        for future in as_completed(futures):
            i, result = future.result()
            results[i] = result

            if result["error"]:
                print(
                    f"{i:4d} | {'ERROR':>10} | {'-':>10} | {result['latency_ms']:>10.0f} | {'-':>10} | {'-':>10} | {result.get('status_code', '?')}: {result.get('body', '')[:40]}"
                )
                continue

            input_tok = result["input_tokens"]
            output_tok = result["output_tokens"]
            latency = result["latency_ms"]

            prompt_tps = round(input_tok / (latency / 1000), 1) if latency > 0 else 0
            gen_tps = round(output_tok / (latency / 1000), 1) if latency > 0 else 0

            result["prompt_tokens_per_sec"] = prompt_tps
            result["generation_tokens_per_sec"] = gen_tps

            print(
                f"{i:4d} | {input_tok:>10,} | {output_tok:>10,} | {latency:>10,.0f} | {prompt_tps:>10,.1f} | {gen_tps:>10,.1f} | {result['stop_reason']}"
            )

    total_elapsed = time.time() - total_start

    # Summary
    successful = [r for r in results if not r.get("error")]
    if successful:
        total_input = sum(r["input_tokens"] for r in successful)
        total_output = sum(r["output_tokens"] for r in successful)
        total_latency = sum(r["latency_ms"] for r in successful)
        avg_prompt_tps = (
            round(total_input / (total_latency / 1000), 1) if total_latency > 0 else 0
        )
        avg_gen_tps = (
            round(total_output / (total_latency / 1000), 1) if total_latency > 0 else 0
        )

        print(f"\n{'=' * 80}")
        print(f"Summary ({len(successful)} successful / {num_lines} total calls)")
        print(f"  Total input tokens:    {total_input:>12,}")
        print(f"  Total output tokens:   {total_output:>12,}")
        print(
            f"  Total latency:         {total_latency:>12,.0f} ms ({total_latency / 1000:.1f}s)"
        )
        print(f"  Wall clock:            {total_elapsed:>12,.1f}s")
        print(f"  Avg prompt tok/s:      {avg_prompt_tps:>12,.1f}")
        print(f"  Avg generation tok/s:  {avg_gen_tps:>12,.1f}")
        print(f"  Errors:                {num_lines - len(successful):>12}")

        summary = {
            "model": model,
            "endpoint": args.endpoint,
            "replay_file": args.jsonl,
            "concurrency": concurrency,
            "calls_total": num_lines,
            "calls_successful": len(successful),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_latency_ms": round(total_latency, 1),
            "wall_clock_seconds": round(total_elapsed, 1),
            "avg_prompt_tokens_per_sec": avg_prompt_tps,
            "avg_generation_tokens_per_sec": avg_gen_tps,
            "per_call_results": results,
        }

        # Save results
        output_path = os.path.join(os.path.dirname(args.jsonl), "replay-results.json")
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
