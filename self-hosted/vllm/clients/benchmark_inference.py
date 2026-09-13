#!/usr/bin/env python3
"""Benchmark a vLLM endpoint and write analysis-ready CSV files.

The client uses streaming chat completions for exact request-level timing and
scrapes vLLM's Prometheus endpoint before and after the measured run. It writes:

* one row per request with TTFT, TTLT, TPOT, token counts, and status;
* one row per vLLM Prometheus sample with before/after values and deltas; and
* one run summary with throughput and latency percentiles.

Usage:
    uv run clients/benchmark_inference.py \
      --model qwen3.6-35b --requests 20 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "qwen3-coder-30b"
DEFAULT_PROMPT = "Write a Python function that returns the nth Fibonacci number."


@dataclass(frozen=True)
class PromptCase:
    """A named prompt used by one or more benchmark requests."""

    prompt_id: str
    prompt: str


@dataclass
class RequestResult:
    """Client-observed metrics for one streaming request."""

    run_id: str
    request_id: int
    prompt_id: str
    prompt_sha256: str
    prompt_characters: int
    model: str
    started_at_utc: str
    status: str
    error: str
    client_queue_ms: float
    ttft_ms: float | None
    ttlt_ms: float | None
    e2e_ms: float
    generation_ms: float | None
    tpot_ms: float | None
    output_tokens_per_s: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    chunks: int
    output_characters: int
    finish_reason: str
    server_request_id: str
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class MetricSample:
    """One parsed Prometheus sample."""

    metric: str
    metric_type: str
    labels_json: str
    value: float

    @property
    def key(self) -> tuple[str, str]:
        """Stable key for matching before and after snapshots."""
        return self.metric, self.labels_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL (env: BASE_URL)",
    )
    parser.add_argument(
        "--metrics-url",
        default=None,
        help="Prometheus metrics URL (default: derive /metrics from --base-url)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", DEFAULT_MODEL),
        help="served model name (env: MODEL)",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=DEFAULT_PROMPT)
    prompt_group.add_argument(
        "--prompts-file",
        type=Path,
        help=(
            "JSONL or line-delimited prompt file. JSON objects use "
            '{"id":"case-id","prompt":"..."}'
        ),
    )
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-output"),
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="optional label included in output filenames and summary",
    )
    parser.add_argument(
        "--no-server-metrics",
        action="store_true",
        help="skip before/after /metrics snapshots",
    )
    return parser.parse_args()


def derive_metrics_url(base_url: str) -> str:
    """Derive vLLM's /metrics URL from an OpenAI-compatible /v1 URL."""
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/metrics", "", ""))


def load_prompts(prompt: str, prompts_file: Path | None) -> list[PromptCase]:
    """Load a single prompt or line-delimited prompt cases."""
    if prompts_file is None:
        return [PromptCase(prompt_id="prompt-1", prompt=prompt)]

    cases: list[PromptCase] = []
    for line_number, raw_line in enumerate(
        prompts_file.read_text().splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            value = line

        if isinstance(value, str):
            cases.append(PromptCase(prompt_id=f"line-{line_number}", prompt=value))
            continue
        if isinstance(value, dict) and isinstance(value.get("prompt"), str):
            cases.append(
                PromptCase(
                    prompt_id=str(value.get("id", f"line-{line_number}")),
                    prompt=value["prompt"],
                )
            )
            continue
        raise ValueError(
            f"{prompts_file}:{line_number}: expected a string or object with prompt"
        )

    if not cases:
        raise ValueError(f"{prompts_file}: no prompts found")
    return cases


def parse_prometheus_metrics(text: str) -> dict[tuple[str, str], MetricSample]:
    """Parse vLLM samples from Prometheus text exposition format."""
    samples: dict[tuple[str, str], MetricSample] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if not sample.name.startswith("vllm:") or sample.name.endswith("_created"):
                continue
            labels_json = json.dumps(
                sample.labels, sort_keys=True, separators=(",", ":")
            )
            metric = MetricSample(
                metric=sample.name,
                metric_type=family.type,
                labels_json=labels_json,
                value=float(sample.value),
            )
            samples[metric.key] = metric
    return samples


async def fetch_metrics(
    client: httpx.AsyncClient, metrics_url: str
) -> dict[tuple[str, str], MetricSample]:
    """Fetch and parse one vLLM Prometheus snapshot."""
    response = await client.get(metrics_url)
    response.raise_for_status()
    return parse_prometheus_metrics(response.text)


def _delta_has_output(delta: dict[str, Any]) -> bool:
    """Return whether a streaming delta carries model output."""
    for key in ("content", "reasoning", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return True
    return bool(delta.get("tool_calls"))


def _output_characters(delta: dict[str, Any]) -> int:
    total = 0
    for key in ("content", "reasoning", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str):
            total += len(value)
    if delta.get("tool_calls"):
        total += len(json.dumps(delta["tool_calls"], separators=(",", ":")))
    return total


async def measure_request(
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    run_id: str,
    request_id: int,
    prompt_case: PromptCase,
    model: str,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> RequestResult:
    """Send one streaming request and measure client-observed latency."""
    queued_at = time.perf_counter()
    async with semaphore:
        started = time.perf_counter()
        started_at_utc = datetime.now(timezone.utc).isoformat()
        first_output_at: float | None = None
        last_output_at: float | None = None
        chunks = 0
        output_characters = 0
        prompt_tokens: int | None = None
        output_tokens: int | None = None
        total_tokens: int | None = None
        finish_reason = ""
        server_request_id = ""
        status = "ok"
        error = ""

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_case.prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        try:
            async with client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    event = json.loads(data)
                    server_request_id = str(event.get("id", server_request_id))
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        prompt_tokens = _optional_int(usage.get("prompt_tokens"))
                        output_tokens = _optional_int(usage.get("completion_tokens"))
                        total_tokens = _optional_int(usage.get("total_tokens"))

                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta") or {}
                    if not isinstance(delta, dict) or not _delta_has_output(delta):
                        continue
                    now = time.perf_counter()
                    if first_output_at is None:
                        first_output_at = now
                    last_output_at = now
                    chunks += 1
                    output_characters += _output_characters(delta)
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            status = "error"
            error = str(exc)

        ended = time.perf_counter()
        ttft_ms = _milliseconds(first_output_at - started) if first_output_at else None
        ttlt_ms = _milliseconds(last_output_at - started) if last_output_at else None
        e2e_ms = _milliseconds(ended - started)
        generation_s = (
            last_output_at - first_output_at
            if first_output_at is not None and last_output_at is not None
            else None
        )
        generation_ms = (
            _milliseconds(generation_s) if generation_s is not None else None
        )

        tpot_ms: float | None = None
        output_tokens_per_s: float | None = None
        if generation_s and output_tokens and output_tokens > 1:
            tpot_ms = _milliseconds(generation_s / (output_tokens - 1))
            output_tokens_per_s = (output_tokens - 1) / generation_s

        return RequestResult(
            run_id=run_id,
            request_id=request_id,
            prompt_id=prompt_case.prompt_id,
            prompt_sha256=hashlib.sha256(prompt_case.prompt.encode()).hexdigest(),
            prompt_characters=len(prompt_case.prompt),
            model=model,
            started_at_utc=started_at_utc,
            status=status,
            error=error,
            client_queue_ms=_milliseconds(started - queued_at),
            ttft_ms=ttft_ms,
            ttlt_ms=ttlt_ms,
            e2e_ms=e2e_ms,
            generation_ms=generation_ms,
            tpot_ms=tpot_ms,
            output_tokens_per_s=output_tokens_per_s,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            chunks=chunks,
            output_characters=output_characters,
            finish_reason=finish_reason,
            server_request_id=server_request_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile."""
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_summary(
    *,
    run_id: str,
    run_name: str,
    model: str,
    base_url: str,
    requests: int,
    concurrency: int,
    warmup: int,
    benchmark_duration_s: float,
    results: list[RequestResult],
    before_metrics: dict[tuple[str, str], MetricSample],
    after_metrics: dict[tuple[str, str], MetricSample],
) -> dict[str, Any]:
    """Build one summary row for the benchmark run."""
    successful = [result for result in results if result.status == "ok"]
    summary: dict[str, Any] = {
        "run_id": run_id,
        "run_name": run_name,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "base_url": base_url,
        "requests": requests,
        "concurrency": concurrency,
        "warmup_requests": warmup,
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "benchmark_duration_s": benchmark_duration_s,
        "completed_requests_per_s": (
            len(successful) / benchmark_duration_s if benchmark_duration_s else None
        ),
        "prompt_tokens": _sum_optional(result.prompt_tokens for result in successful),
        "output_tokens": _sum_optional(result.output_tokens for result in successful),
        "total_tokens": _sum_optional(result.total_tokens for result in successful),
    }
    output_tokens = summary["output_tokens"]
    total_tokens = summary["total_tokens"]
    summary["aggregate_output_tokens_per_s"] = (
        output_tokens / benchmark_duration_s
        if output_tokens is not None and benchmark_duration_s
        else None
    )
    summary["aggregate_total_tokens_per_s"] = (
        total_tokens / benchmark_duration_s
        if total_tokens is not None and benchmark_duration_s
        else None
    )

    for field_name in ("ttft_ms", "ttlt_ms", "e2e_ms", "tpot_ms"):
        values = [
            float(value)
            for result in successful
            if (value := getattr(result, field_name)) is not None
        ]
        summary[f"{field_name}_mean"] = sum(values) / len(values) if values else None
        for label, quantile in (
            ("p50", 0.50),
            ("p90", 0.90),
            ("p95", 0.95),
            ("p99", 0.99),
        ):
            summary[f"{field_name}_{label}"] = percentile(values, quantile)

    for output_name, metric_root in (
        ("server_ttft_ms_mean", "vllm:time_to_first_token_seconds"),
        ("server_tpot_ms_mean", "vllm:request_time_per_output_token_seconds"),
        ("server_e2e_ms_mean", "vllm:e2e_request_latency_seconds"),
        ("server_queue_ms_mean", "vllm:request_queue_time_seconds"),
        ("server_prefill_ms_mean", "vllm:request_prefill_time_seconds"),
        ("server_decode_ms_mean", "vllm:request_decode_time_seconds"),
    ):
        value = _histogram_mean_delta(before_metrics, after_metrics, metric_root)
        summary[output_name] = value * 1000 if value is not None else None

    for output_name, metric_name in (
        ("server_prompt_tokens_delta", "vllm:prompt_tokens_total"),
        ("server_generation_tokens_delta", "vllm:generation_tokens_total"),
        ("server_success_requests_delta", "vllm:request_success_total"),
        ("server_preemptions_delta", "vllm:num_preemptions_total"),
    ):
        summary[output_name] = _metric_delta(before_metrics, after_metrics, metric_name)
    return summary


def write_request_csv(path: Path, results: list[RequestResult]) -> None:
    """Write one row per measured request."""
    field_names = [field.name for field in fields(RequestResult)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def write_server_metrics_csv(
    path: Path,
    run_id: str,
    before_metrics: dict[tuple[str, str], MetricSample],
    after_metrics: dict[tuple[str, str], MetricSample],
) -> None:
    """Write long-form before/after server metric samples."""
    field_names = [
        "run_id",
        "metric",
        "metric_type",
        "labels_json",
        "before_value",
        "after_value",
        "delta",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for key in sorted(before_metrics.keys() | after_metrics.keys()):
            before = before_metrics.get(key)
            after = after_metrics.get(key)
            before_value = before.value if before else None
            after_value = after.value if after else None
            delta = (
                after_value - before_value
                if before_value is not None and after_value is not None
                else None
            )
            sample = after or before
            assert sample is not None  # nosec B101 - at least one snapshot is always present here
            writer.writerow(
                {
                    "run_id": run_id,
                    "metric": sample.metric,
                    "metric_type": sample.metric_type,
                    "labels_json": sample.labels_json,
                    "before_value": before_value,
                    "after_value": after_value,
                    "delta": delta,
                }
            )


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    """Write the single-row benchmark summary."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


async def run_benchmark(
    args: argparse.Namespace,
) -> tuple[list[RequestResult], dict[str, Any], tuple[Path, Path, Path]]:
    """Run warmup, measured requests, metric snapshots, and CSV output."""
    prompts = load_prompts(args.prompt, args.prompts_file)
    metrics_url = args.metrics_url or derive_metrics_url(args.base_url)
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    run_id = _build_run_id(args.model, args.run_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.output_dir / f"{run_id}-requests.csv"
    metrics_path = args.output_dir / f"{run_id}-server-metrics.csv"
    summary_path = args.output_dir / f"{run_id}-summary.csv"

    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(
        max_connections=max(args.concurrency + 2, 10),
        max_keepalive_connections=max(args.concurrency, 5),
    )
    headers = {"Authorization": f"Bearer {os.environ.get('API_KEY', 'not-needed')}"}

    before_metrics: dict[tuple[str, str], MetricSample] = {}
    after_metrics: dict[tuple[str, str], MetricSample] = {}
    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, headers=headers
    ) as client:
        warmup_semaphore = asyncio.Semaphore(1)
        for index in range(args.warmup):
            result = await measure_request(
                client=client,
                semaphore=warmup_semaphore,
                endpoint=endpoint,
                run_id=run_id,
                request_id=-(index + 1),
                prompt_case=prompts[index % len(prompts)],
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed + index,
            )
            if result.status != "ok":
                raise RuntimeError(f"warmup request failed: {result.error}")

        if not args.no_server_metrics:
            try:
                before_metrics = await fetch_metrics(client, metrics_url)
            except httpx.HTTPError as exc:
                logger.warning(
                    "Could not fetch before metrics from %s: %s", metrics_url, exc
                )

        semaphore = asyncio.Semaphore(args.concurrency)
        benchmark_started = time.perf_counter()
        tasks = [
            measure_request(
                client=client,
                semaphore=semaphore,
                endpoint=endpoint,
                run_id=run_id,
                request_id=index + 1,
                prompt_case=prompts[index % len(prompts)],
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed + args.warmup + index,
            )
            for index in range(args.requests)
        ]
        results = await asyncio.gather(*tasks)
        benchmark_duration_s = time.perf_counter() - benchmark_started

        if not args.no_server_metrics:
            try:
                after_metrics = await fetch_metrics(client, metrics_url)
            except httpx.HTTPError as exc:
                logger.warning(
                    "Could not fetch after metrics from %s: %s", metrics_url, exc
                )

    summary = build_summary(
        run_id=run_id,
        run_name=args.run_name,
        model=args.model,
        base_url=args.base_url,
        requests=args.requests,
        concurrency=args.concurrency,
        warmup=args.warmup,
        benchmark_duration_s=benchmark_duration_s,
        results=results,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
    )
    write_request_csv(request_path, results)
    write_server_metrics_csv(metrics_path, run_id, before_metrics, after_metrics)
    write_summary_csv(summary_path, summary)
    return results, summary, (request_path, metrics_path, summary_path)


def _build_run_id(model: str, run_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = run_name or model
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", label).strip("-").lower()
    return f"{timestamp}-{slug or 'vllm'}"


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _milliseconds(seconds: float) -> float:
    return seconds * 1000


def _sum_optional(values: Any) -> int | None:
    present = [int(value) for value in values if value is not None]
    return sum(present) if present else None


def _metric_delta(
    before_metrics: dict[tuple[str, str], MetricSample],
    after_metrics: dict[tuple[str, str], MetricSample],
    metric_name: str,
) -> float | None:
    before = sum(
        sample.value
        for sample in before_metrics.values()
        if sample.metric == metric_name
    )
    after = sum(
        sample.value
        for sample in after_metrics.values()
        if sample.metric == metric_name
    )
    if not before_metrics or not after_metrics:
        return None
    return after - before


def _histogram_mean_delta(
    before_metrics: dict[tuple[str, str], MetricSample],
    after_metrics: dict[tuple[str, str], MetricSample],
    metric_root: str,
) -> float | None:
    total = _metric_delta(before_metrics, after_metrics, f"{metric_root}_sum")
    count = _metric_delta(before_metrics, after_metrics, f"{metric_root}_count")
    if total is None or count is None or count <= 0:
        return None
    return total / count


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("requests", "concurrency", "max_tokens"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero")
    if args.warmup < 0:
        raise ValueError("--warmup must be zero or greater")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    try:
        _validate_args(args)
        results, summary, paths = asyncio.run(run_benchmark(args))
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    logger.info(
        "completed %s/%s requests in %.2fs",
        summary["successful_requests"],
        len(results),
        summary["benchmark_duration_s"],
    )
    logger.info(
        "TTFT p50/p95: %s / %s ms | TTLT p50/p95: %s / %s ms",
        _format_number(summary["ttft_ms_p50"]),
        _format_number(summary["ttft_ms_p95"]),
        _format_number(summary["ttlt_ms_p50"]),
        _format_number(summary["ttlt_ms_p95"]),
    )
    for path in paths:
        logger.info("wrote %s", path)


def _format_number(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"


if __name__ == "__main__":
    main()
