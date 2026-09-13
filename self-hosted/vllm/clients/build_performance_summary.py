#!/usr/bin/env python3
"""Build performance-summary.json for a throughput sweep of one model+instance.

Reads the shared DuckDB written by the concurrency sweep (one named collector
session per level, ``{model}_c{N}``) and derives, per concurrency level, the
server-measured throughput, latency, and saturation -- then a realistic cost per
token and per task from the instance's hourly price:

    cost_per_output_token = (dollars_per_hour / 3600) / output_tokens_per_second
    cost_per_task         = cost_per_output_token * output_tokens_per_task

Throughput comes from vLLM's own counters (generation/prompt tokens), NOT from
whether an agentic session finished -- the sweep deliberately cuts sessions off
at the window, so the counter delta over the session's window IS the sustained
throughput at that concurrency. The output is analysis-ready for a simple HTML
dashboard (see build_performance_dashboard.py).

Usage:
    uv run python -m clients.build_performance_summary \\
        --model gemma-4-31b --db benchmark-output/throughput/gemma-4-31b/throughput-metrics.duckdb \\
        --instance-type g6e.12xlarge --dollars-per-hour 10.49 \\
        --output-tokens-per-task 24000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import duckdb

try:  # works whether run as `python -m clients.X` or as a script in clients/
    from clients import pricing
except ImportError:  # pragma: no cover - direct-script fallback
    import pricing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Counter metrics -> throughput via (last - first) / window.
GEN_TOKENS = "vllm:generation_tokens_total"
PROMPT_TOKENS = "vllm:prompt_tokens_total"
# Gauge metrics -> saturation via peak/mean while under load.
KV_USAGE = "vllm:kv_cache_usage_perc"
REQ_RUNNING = "vllm:num_requests_running"
REQ_WAITING = "vllm:num_requests_waiting"
# Histogram metrics -> latency over the window's requests. We report PERCENTILES
# (p50/p90) from the _bucket series, not just the _sum/_count mean: the mean is
# distorted by outliers and by how many requests happen to complete in the
# window (at low concurrency, few completions let a handful of cold-cache
# prefills dominate, making the mean fall as concurrency rises -- an artifact,
# not a real improvement). Percentiles from cumulative buckets are robust.
TTFT = "vllm:time_to_first_token_seconds"
TPOT = "vllm:request_time_per_output_token_seconds"
# TTFT decomposes into scheduler QUEUE wait + actual PREFILL. For an input-heavy
# agentic workload TTFT is dominated by queue wait once the server saturates, so
# splitting them out is the honest UX story (see cost-per-task-methodology.md).
QUEUE_TIME = "vllm:request_queue_time_seconds"
PREFILL_TIME = "vllm:request_prefill_time_seconds"

# Percentiles to report from each latency histogram.
_PERCENTILES = (50, 90, 99)


def _counter_rate(con: duckdb.DuckDBPyConnection, session: str, metric: str) -> dict:
    """Return {delta, seconds, per_second} for a counter over a session window.

    Sums across label sets per scrape (server-wide counters may be split by
    labels), then takes the last-minus-first over the session's elapsed time.
    """
    row = con.execute(
        """
        WITH per_scrape AS (
            SELECT scraped_at, sum(value) AS v
            FROM vllm_metric_samples
            WHERE session_name = ? AND metric = ?
            GROUP BY scraped_at
        )
        SELECT
            max(v) - min(v) AS delta,
            date_diff('second', min(scraped_at), max(scraped_at)) AS seconds
        FROM per_scrape
        """,
        [session, metric],
    ).fetchone()
    delta = float(row[0]) if row and row[0] is not None else 0.0
    seconds = float(row[1]) if row and row[1] else 0.0
    return {
        "delta_tokens": round(delta),
        "window_seconds": round(seconds, 1),
        "tokens_per_second": round(delta / seconds, 2) if seconds > 0 else None,
    }


def _gauge_stats(con: duckdb.DuckDBPyConnection, session: str, metric: str) -> dict:
    """Return {peak, mean} for a gauge over a session window (max across labels)."""
    row = con.execute(
        """
        WITH per_scrape AS (
            SELECT scraped_at, max(value) AS v
            FROM vllm_metric_samples
            WHERE session_name = ? AND metric = ?
            GROUP BY scraped_at
        )
        SELECT max(v), avg(v) FROM per_scrape
        """,
        [session, metric],
    ).fetchone()
    peak = float(row[0]) if row and row[0] is not None else None
    mean = float(row[1]) if row and row[1] is not None else None
    return {
        "peak": round(peak, 4) if peak is not None else None,
        "mean": round(mean, 4) if mean is not None else None,
    }


def _hist_mean_ms(
    con: duckdb.DuckDBPyConnection, session: str, base: str
) -> float | None:
    """Mean latency (ms) over a window from a histogram's _sum/_count deltas."""

    def _delta(suffix: str) -> float:
        row = con.execute(
            """
            WITH per_scrape AS (
                SELECT scraped_at, sum(value) AS v
                FROM vllm_metric_samples
                WHERE session_name = ? AND metric = ?
                GROUP BY scraped_at
            )
            SELECT max(v) - min(v) FROM per_scrape
            """,
            [session, base + suffix],
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    sum_delta = _delta("_sum")
    count_delta = _delta("_count")
    if count_delta <= 0:
        return None
    return round((sum_delta / count_delta) * 1000, 1)


def _hist_percentiles_ms(
    con: duckdb.DuckDBPyConnection, session: str, base: str
) -> dict[str, float | None]:
    """Percentiles (ms) over a window from a Prometheus histogram's buckets.

    Reads the cumulative ``_bucket`` series (one per ``le`` upper-edge), takes the
    last-minus-first delta per edge to get this window's per-bucket counts, then
    walks the cumulative distribution to find each target percentile's bucket
    edge. Robust to outliers and to how many requests completed, unlike the mean.

    The reported value is the bucket's upper edge (Prometheus histograms bound,
    not interpolate), so it is an upper estimate within that bucket. ``+Inf`` maps
    to ``None`` -- if a percentile lands in the overflow bucket, more than that
    fraction of requests exceeded the largest finite edge (report as ">edge").

    Args:
        con: Open DuckDB connection.
        session: Collector session name (one concurrency level).
        base: Histogram metric base (without the ``_bucket`` suffix).

    Returns:
        ``{"p50": ms, "p90": ms, "p99": ms}``; a value is None when that
        percentile falls in the ``+Inf`` overflow bucket or there is no data.
    """
    rows = con.execute(
        """
        SELECT labels, scraped_at, value
        FROM vllm_metric_samples
        WHERE session_name = ? AND metric = ?
        ORDER BY scraped_at
        """,
        [session, base + "_bucket"],
    ).fetchall()

    # Per-edge window delta = last cumulative count - first cumulative count.
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    for labels, _scraped_at, value in rows:
        edge = json.loads(labels).get("le")
        if edge is None:
            continue
        val = float(value)
        first.setdefault(edge, val)
        last[edge] = val

    def _edge_key(edge: str) -> float:
        return float("inf") if edge == "+Inf" else float(edge)

    edges = sorted(first, key=_edge_key)
    cumulative = [(edge, last[edge] - first[edge]) for edge in edges]
    total = cumulative[-1][1] if cumulative else 0.0
    out: dict[str, float | None] = {f"p{p}": None for p in _PERCENTILES}
    if total <= 0:
        return out

    for p in _PERCENTILES:
        target = total * p / 100.0
        for edge, cum in cumulative:
            if cum >= target:
                out[f"p{p}"] = None if edge == "+Inf" else round(float(edge) * 1000, 1)
                break
    return out


def _levels(con: duckdb.DuckDBPyConnection, model: str) -> list[tuple[int, str]]:
    """Return [(concurrency, session_name)] for this model's sweep, sorted by N."""
    rows = con.execute(
        "SELECT DISTINCT session_name FROM collector_sessions WHERE session_name LIKE ?",
        [f"{model}_c%"],
    ).fetchall()
    out = []
    for (name,) in rows:
        suffix = name.rsplit("_c", 1)[-1]
        if suffix.isdigit():
            out.append((int(suffix), name))
    return sorted(out)


def _level_costs(
    dps: float, prompt_tps: float | None, gen_tps: float | None, w: float
) -> dict[str, Any]:
    """Two cost lenses for one level from the machine's $/second.

    On a fixed-cost machine the $/hr is not itemized per token, so cost per token
    depends on how the GPU-second is attributed:

    * Lens A -- **blended / measured** (no assumption): every processed token
      costs the same GPU slice, so ``blended = $/s / (prompt_tps + gen_tps)``.
      Input and output cost the same per token.
    * Lens B -- **lab-style split** (one convention ``w``): an input token counts
      as ``w`` of an output token when dividing GPU time, giving the familiar
      input-cheaper-than-output shape. ``cost_out = $/s / (gen_tps + w*prompt_tps)``
      and ``cost_in = w * cost_out``. ``w`` is a chosen convention, not measured.

    Args:
        dps: Dollars per second (instance $/hr / 3600).
        prompt_tps: Prompt (input) tokens/s served at this level.
        gen_tps: Generation (output) tokens/s at this level.
        w: Lens-B input weight (input token = w x output token).

    Returns:
        A dict of both lenses' per-token costs (USD), or Nones when no throughput.
    """
    p = prompt_tps or 0.0
    g = gen_tps or 0.0
    total = p + g
    blended = dps / total if total > 0 else None
    denom_b = g + w * p
    cost_out_b = dps / denom_b if denom_b > 0 else None
    cost_in_b = w * cost_out_b if cost_out_b is not None else None
    return {
        # Lens A: blended, measured -- input == output.
        "blended_cost_per_token_usd": blended,
        # Lens B: lab-style split with input weight w.
        "split_input_weight": w,
        "split_cost_per_input_token_usd": cost_in_b,
        "split_cost_per_output_token_usd": cost_out_b,
    }


def _build(
    db: Path,
    model: str,
    instance_type: str,
    dph: float,
    out_tokens_per_task: int | None,
    input_tokens_per_task: int | None,
    input_weight: float,
) -> dict[str, Any]:
    """Assemble the performance summary from the DuckDB sweep sessions."""
    con = duckdb.connect(str(db), read_only=True)
    try:
        levels = _levels(con, model)
        if not levels:
            raise SystemExit(f"no sweep sessions '{model}_c*' found in {db}")
        dps = dph / 3600.0
        n_in = input_tokens_per_task
        m_out = out_tokens_per_task
        rows: list[dict[str, Any]] = []
        for concurrency, session in levels:
            gen = _counter_rate(con, session, GEN_TOKENS)
            prompt = _counter_rate(con, session, PROMPT_TOKENS)
            out_tps = gen["tokens_per_second"]
            prompt_tps = prompt["tokens_per_second"]
            costs = _level_costs(dps, prompt_tps, out_tps, input_weight)

            # cost-per-task under each lens, using the measured N:M token counts.
            def _task_cost(per_in: float | None, per_out: float | None) -> float | None:
                if per_out is None or m_out is None:
                    return None
                total = per_out * m_out
                if per_in is not None and n_in is not None:
                    total += per_in * n_in
                return round(total, 4)

            blended = costs["blended_cost_per_token_usd"]
            row = {
                "concurrency": concurrency,
                "session_name": session,
                "output_tokens_per_second": out_tps,
                "prompt_tokens_per_second": prompt_tps,
                "window_seconds": gen["window_seconds"],
                "kv_cache_usage": _gauge_stats(con, session, KV_USAGE),
                "requests_running": _gauge_stats(con, session, REQ_RUNNING),
                "requests_waiting": _gauge_stats(con, session, REQ_WAITING),
                # TTFT/TPOT: percentiles from buckets (robust) plus the mean for
                # continuity. TTFT is also decomposed into queue wait vs prefill.
                "ttft_ms_mean": _hist_mean_ms(con, session, TTFT),
                "ttft_ms_pctl": _hist_percentiles_ms(con, session, TTFT),
                "tpot_ms_mean": _hist_mean_ms(con, session, TPOT),
                "tpot_ms_pctl": _hist_percentiles_ms(con, session, TPOT),
                "queue_ms_mean": _hist_mean_ms(con, session, QUEUE_TIME),
                "queue_ms_pctl": _hist_percentiles_ms(con, session, QUEUE_TIME),
                "prefill_ms_mean": _hist_mean_ms(con, session, PREFILL_TIME),
                **costs,
                # Convenience per-1M figures for the dashboard axes.
                "blended_cost_per_1m_tokens_usd": round(blended * 1e6, 2)
                if blended
                else None,
                "split_cost_per_1m_output_tokens_usd": round(
                    costs["split_cost_per_output_token_usd"] * 1e6, 2
                )
                if costs["split_cost_per_output_token_usd"]
                else None,
                "split_cost_per_1m_input_tokens_usd": round(
                    costs["split_cost_per_input_token_usd"] * 1e6, 2
                )
                if costs["split_cost_per_input_token_usd"]
                else None,
                # cost per blended task (N input + M output) under both lenses.
                "task_cost_blended_usd": _task_cost(blended, blended),
                "task_cost_split_usd": _task_cost(
                    costs["split_cost_per_input_token_usd"],
                    costs["split_cost_per_output_token_usd"],
                ),
            }
            rows.append(row)

        # Peak sustained throughput across levels (the saturation ceiling).
        best = max(
            (r for r in rows if r["output_tokens_per_second"]),
            key=lambda r: r["output_tokens_per_second"],
            default=None,
        )
        return {
            "model": model,
            "instance_type": instance_type,
            "dollars_per_hour": dph,
            # The "blended task" definition (N:M input:output), from real /swe runs.
            "task_input_tokens": n_in,
            "task_output_tokens": m_out,
            "task_input_output_ratio": round(n_in / m_out, 1)
            if n_in and m_out
            else None,
            "split_input_weight": input_weight,
            "peak_output_tokens_per_second": best["output_tokens_per_second"]
            if best
            else None,
            "peak_at_concurrency": best["concurrency"] if best else None,
            "min_blended_cost_per_1m_tokens_usd": min(
                (
                    r["blended_cost_per_1m_tokens_usd"]
                    for r in rows
                    if r["blended_cost_per_1m_tokens_usd"]
                ),
                default=None,
            ),
            "min_task_cost_blended_usd": min(
                (
                    r["task_cost_blended_usd"]
                    for r in rows
                    if r["task_cost_blended_usd"]
                ),
                default=None,
            ),
            "min_task_cost_split_usd": min(
                (r["task_cost_split_usd"] for r in rows if r["task_cost_split_usd"]),
                default=None,
            ),
            "levels": rows,
        }
    finally:
        con.close()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Build performance-summary.json from a throughput-sweep DuckDB."
    )
    p.add_argument("--model", required=True, help="Served model name (session prefix)")
    p.add_argument("--db", required=True, type=Path, help="Sweep DuckDB path")
    p.add_argument(
        "--instance-type",
        default="unknown",
        help="EC2 instance type; also used to look up the rate in pricing.json "
        "when --dollars-per-hour is omitted",
    )
    p.add_argument(
        "--dollars-per-hour",
        type=float,
        default=None,
        help="Instance on-demand $/hr. Omit to resolve from pricing.json by "
        "--instance-type (and --tp for a partial-box run).",
    )
    p.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Tensor-parallel size (GPUs used). With a partial box, the "
        "pricing.json rate is scaled by tp/gpus_per_instance.",
    )
    p.add_argument(
        "--output-tokens-per-task",
        type=int,
        default=None,
        help="Mean OUTPUT tokens per agentic task (the M in the N:M blended task).",
    )
    p.add_argument(
        "--input-tokens-per-task",
        type=int,
        default=None,
        help="Mean INPUT tokens per agentic task (the N in the N:M blended task).",
    )
    p.add_argument(
        "--input-weight",
        type=float,
        default=0.25,
        help="Lens B: an input token counts as this fraction of an output token "
        "when splitting GPU cost (lab convention; default 0.25).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: <db dir>/performance-summary.json)",
    )
    return p.parse_args()


def main() -> None:
    """Build and write the performance summary."""
    args = _parse_args()
    db = args.db.expanduser().resolve()
    if not db.is_file():
        raise SystemExit(f"DuckDB not found: {db}")
    # Resolve the hourly rate from pricing.json (single source of truth) unless
    # the caller passed an explicit --dollars-per-hour override.
    dph = args.dollars_per_hour
    if dph is None:
        dph = pricing.resolve(args.instance_type, tp=args.tp)
        logger.info(
            "resolved $%.4f/hr for %s (tp=%s) from pricing.json",
            dph,
            args.instance_type,
            args.tp,
        )
    summary = _build(
        db,
        args.model,
        args.instance_type,
        dph,
        args.output_tokens_per_task,
        args.input_tokens_per_task,
        args.input_weight,
    )
    out = args.out or (db.parent / "performance-summary.json")
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info(
        "wrote %s: %d levels, peak %s tok/s @ c=%s | blended $%.2f/1M | "
        "task $%.2f blended / $%.2f split (N:M=%s:%s, w=%s)",
        out,
        len(summary["levels"]),
        summary["peak_output_tokens_per_second"],
        summary["peak_at_concurrency"],
        summary["min_blended_cost_per_1m_tokens_usd"] or 0.0,
        summary["min_task_cost_blended_usd"] or 0.0,
        summary["min_task_cost_split_usd"] or 0.0,
        summary["task_input_tokens"],
        summary["task_output_tokens"],
        summary["split_input_weight"],
    )


if __name__ == "__main__":
    main()
