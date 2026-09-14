"""Build a self-contained HTML dashboard from the vLLM metrics DuckDB.

Reads the analytics views written by ``collect_metrics.py`` and renders a single
standalone HTML file with the metric time series embedded inline as JSON. The
output has no runtime dependencies: no web server, no CDN, no network. Open it
with a double-click, or re-run this script to refresh the snapshot.

The dashboard groups the ``vllm:*`` metrics into the same categories the
collector captures: token throughput, scheduler/concurrency state, KV-cache
utilization, request latency (TTFT, TPOT, queue, end-to-end), and request
outcomes by finish reason.

The served model name (read from the metric labels) is embedded in the output
filename automatically, so dashboards from different models never overwrite one
another: ``--output benchmark-output/dashboard.html`` against a ``qwen3.6-35b``
run writes ``benchmark-output/dashboard-qwen3-6-35b.html``.

Usage:
    uv run python -m clients.build_dashboard
    uv run python -m clients.build_dashboard --db benchmark-output/vllm-metrics.duckdb \
        --output benchmark-output/dashboard.html
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_DB = "benchmark-output/vllm-metrics.duckdb"
DEFAULT_OUTPUT = "benchmark-output/dashboard.html"

# Cumulative counters we render as per-interval rates (delta / delta_seconds).
_RATE_METRICS = {
    "vllm:generation_tokens_total": "Generation tokens/s",
    "vllm:prompt_tokens_total": "Prompt tokens/s",
}

# Point-in-time gauges rendered directly as a time series.
_GAUGE_METRICS = {
    "vllm:num_requests_running": "Running",
    "vllm:num_requests_waiting": "Waiting",
}

# Histogram base names rendered as per-interval mean latency (delta sum / delta count).
_LATENCY_METRICS = {
    "vllm:time_to_first_token_seconds": "Time to first token",  # nosec B105 - chart label, not a secret
    "vllm:inter_token_latency_seconds": "Inter-token latency (TPOT)",  # nosec B105 - chart label, not a secret
    "vllm:request_queue_time_seconds": "Queue time",
    "vllm:e2e_request_latency_seconds": "End-to-end latency",
}


def _connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open the metrics database read-only.

    Args:
        db_path: Path to the DuckDB file written by the collector.

    Returns:
        A read-only DuckDB connection.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"Metrics database not found at '{db_path}'. Start the collector first "
            "(./scripts/vllm-metrics.sh start) or pass --db."
        )
    return duckdb.connect(str(db_path), read_only=True)


def _fetch_sessions(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Return collector session metadata, oldest first."""
    rows = con.execute(
        """
        SELECT
            cs.session_name,
            cs.started_at,
            cs.stopped_at,
            cs.interval_seconds,
            count(sc.scrape_id) AS scrapes,
            count(sc.scrape_id) FILTER (WHERE sc.status <> 'ok') AS failed
        FROM collector_sessions cs
        LEFT JOIN metric_scrapes sc USING (session_id)
        GROUP BY 1, 2, 3, 4
        ORDER BY cs.started_at
        """
    ).fetchall()
    return [
        {
            "session_name": r[0],
            "started_at": r[1].isoformat() if r[1] else None,
            "stopped_at": r[2].isoformat() if r[2] else None,
            "interval_seconds": r[3],
            "scrapes": r[4],
            "failed": r[5],
        }
        for r in rows
    ]


def _fetch_gauge_series(
    con: duckdb.DuckDBPyConnection, metric: str
) -> list[dict[str, Any]]:
    """Return an ``[{t, value}]`` time series for a single gauge metric."""
    rows = con.execute(
        """
        SELECT scraped_at, value
        FROM vllm_metric_samples
        WHERE metric = ?
        ORDER BY scraped_at
        """,
        [metric],
    ).fetchall()
    return [{"t": r[0].isoformat(), "value": r[1]} for r in rows]


def _fetch_rate_series(
    con: duckdb.DuckDBPyConnection, metric: str
) -> list[dict[str, Any]]:
    """Return a per-interval rate series for a cumulative counter.

    Uses the delta between consecutive scrapes divided by the elapsed seconds.
    Counter resets (a negative delta, e.g. after a server restart) are dropped.
    """
    rows = con.execute(
        """
        WITH ordered AS (
            SELECT
                session_id,
                scraped_at,
                value,
                lag(value) OVER w AS prev_value,
                lag(scraped_at) OVER w AS prev_at
            FROM vllm_metric_samples
            WHERE metric = ?
            WINDOW w AS (PARTITION BY session_id ORDER BY scraped_at)
        )
        SELECT
            scraped_at,
            (value - prev_value)
                / nullif(epoch(scraped_at) - epoch(prev_at), 0) AS rate
        FROM ordered
        WHERE prev_value IS NOT NULL AND value >= prev_value
        ORDER BY scraped_at
        """,
        [metric],
    ).fetchall()
    return [{"t": r[0].isoformat(), "value": r[1] or 0.0} for r in rows]


def _fetch_latency_series(
    con: duckdb.DuckDBPyConnection, base: str
) -> list[dict[str, Any]]:
    """Return per-interval mean latency (seconds) for a histogram metric.

    Mean over the interval is ``delta(sum) / delta(count)`` between scrapes,
    which reflects only the requests that completed during that interval.
    """
    rows = con.execute(
        """
        WITH s AS (
            SELECT session_id, scraped_at, value,
                   lag(value) OVER w AS prev
            FROM vllm_metric_samples WHERE metric = ?
            WINDOW w AS (PARTITION BY session_id ORDER BY scraped_at)
        ),
        c AS (
            SELECT session_id, scraped_at, value,
                   lag(value) OVER w AS prev
            FROM vllm_metric_samples WHERE metric = ?
            WINDOW w AS (PARTITION BY session_id ORDER BY scraped_at)
        )
        SELECT s.scraped_at,
               (s.value - s.prev) / nullif(c.value - c.prev, 0) AS mean_latency
        FROM s JOIN c USING (session_id, scraped_at)
        WHERE s.prev IS NOT NULL AND (c.value - c.prev) > 0
        ORDER BY s.scraped_at
        """,
        [f"{base}_sum", f"{base}_count"],
    ).fetchall()
    return [{"t": r[0].isoformat(), "value": r[1]} for r in rows if r[1] is not None]


def _fetch_finish_reasons(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Return the latest cumulative request count per finish reason."""
    rows = con.execute(
        """
        SELECT json_extract_string(labels, '$.finished_reason') AS reason,
               max(value) AS count
        FROM metric_samples
        WHERE metric = 'vllm:request_success_total'
        GROUP BY 1
        HAVING max(value) > 0
        ORDER BY 2 DESC
        """
    ).fetchall()
    return [{"reason": r[0] or "unknown", "count": r[1]} for r in rows]


def _cumulative_mean(con: duckdb.DuckDBPyConnection, base: str) -> float | None:
    """Return the run-wide mean latency (seconds) from the latest sum/count."""
    row = con.execute(
        """
        SELECT
            max(value) FILTER (WHERE metric = ? ) AS s,
            max(value) FILTER (WHERE metric = ? ) AS c
        FROM metric_samples
        WHERE metric IN (?, ?)
        """,
        [f"{base}_sum", f"{base}_count", f"{base}_sum", f"{base}_count"],
    ).fetchone()
    if not row or not row[1]:
        return None
    return row[0] / row[1]


def _counter_total(con: duckdb.DuckDBPyConnection, metric: str) -> float:
    """Return the latest cumulative value of a counter metric."""
    row = con.execute(
        "SELECT max(value) FROM metric_samples WHERE metric = ?", [metric]
    ).fetchone()
    return (row[0] if row and row[0] is not None else 0.0) or 0.0


def _gauge_peak(con: duckdb.DuckDBPyConnection, metric: str) -> float:
    """Return the peak observed value of a gauge metric."""
    row = con.execute(
        "SELECT max(value) FROM metric_samples WHERE metric = ?", [metric]
    ).fetchone()
    return (row[0] if row and row[0] is not None else 0.0) or 0.0


def _observed_duration_seconds(con: duckdb.DuckDBPyConnection) -> float:
    """Return the total observed collection span in seconds (first to last scrape)."""
    row = con.execute(
        "SELECT epoch(max(scraped_at)) - epoch(min(scraped_at)) FROM metric_scrapes"
    ).fetchone()
    return (row[0] if row and row[0] is not None else 0.0) or 0.0


def _format_mtok(tokens: float) -> str:
    """Format a raw token count as millions of tokens (Mtok).

    Args:
        tokens: Raw token count.

    Returns:
        The count in millions with three decimals, e.g. ``0.348`` for 347,990.
    """
    return f"{tokens / 1_000_000:,.3f}"


def _build_kpis(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Assemble the headline stat-tile numbers.

    Leads with the run-level totals (duration, tokens, throughput), then the
    latency means, then the capacity and cache indicators.
    """
    gen = _counter_total(con, "vllm:generation_tokens_total")
    prompt = _counter_total(con, "vllm:prompt_tokens_total")
    total_tokens = gen + prompt
    duration = _observed_duration_seconds(con)
    gen_throughput = gen / duration if duration else 0.0
    prompt_throughput = prompt / duration if duration else 0.0
    requests = sum(r["count"] for r in _fetch_finish_reasons(con))
    ttft = _cumulative_mean(con, "vllm:time_to_first_token_seconds")
    e2e = _cumulative_mean(con, "vllm:e2e_request_latency_seconds")
    tpot = _cumulative_mean(con, "vllm:inter_token_latency_seconds")
    queries = _counter_total(con, "vllm:prefix_cache_queries_total")
    hits = _counter_total(con, "vllm:prefix_cache_hits_total")
    hit_rate = (hits / queries * 100.0) if queries else 0.0
    peak_conc = _gauge_peak(con, "vllm:num_requests_running")
    kv_peak = _gauge_peak(con, "vllm:kv_cache_usage_perc") * 100.0

    return [
        {"label": "Total duration", "value": f"{duration:,.0f}", "unit": "s"},
        {
            "label": "Total tokens",
            "value": _format_mtok(total_tokens),
            "unit": "Mtok",
            "detail": f"{_format_mtok(gen)} generation · {_format_mtok(prompt)} prompt Mtok",
        },
        {
            "label": "Prompt throughput",
            "value": f"{prompt_throughput:,.1f}",
            "unit": "tokens/s",
        },
        {
            "label": "Generation throughput",
            "value": f"{gen_throughput:,.1f}",
            "unit": "tokens/s",
        },
        {"label": "Requests", "value": f"{requests:,.0f}", "unit": "completed"},
        {
            "label": "Mean TTFT",
            "value": f"{ttft * 1000:,.0f}" if ttft is not None else "-",
            "unit": "ms",
        },
        {
            "label": "Mean TPOT",
            "value": f"{tpot * 1000:,.1f}" if tpot is not None else "-",
            "unit": "ms/token",
        },
        {
            "label": "Mean E2E latency",
            "value": f"{e2e:,.2f}" if e2e is not None else "-",
            "unit": "s",
        },
        {"label": "Peak concurrency", "value": f"{peak_conc:,.0f}", "unit": "requests"},
        {"label": "Prefix cache hit rate", "value": f"{hit_rate:,.1f}", "unit": "%"},
        {"label": "Peak KV-cache use", "value": f"{kv_peak:,.2f}", "unit": "%"},
    ]


def _fetch_model_name(con: duckdb.DuckDBPyConnection) -> str | None:
    """Return the served model name recorded in the metric labels, if any."""
    row = con.execute(
        "SELECT model_name FROM vllm_metric_samples WHERE model_name IS NOT NULL LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def _model_slug(model: str) -> str:
    """Turn a served model name into a filesystem-safe filename fragment.

    Model ids often carry a org prefix and mixed case (e.g.
    ``meta-llama/Llama-3-8B-Instruct``); this lowercases, replaces any run of
    non-alphanumeric characters with a single hyphen, and trims stray hyphens so
    the result is safe to embed in a filename.

    Args:
        model: The served model name from the metric labels.

    Returns:
        A slug such as ``meta-llama-llama-3-8b-instruct``; empty if the input
        has no alphanumeric characters.
    """
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def _stamp_model(output_path: Path, model: str | None) -> Path:
    """Insert the model slug before the output file's suffix.

    ``dashboard.html`` with model ``qwen3.6-35b`` becomes
    ``dashboard-qwen3-6-35b.html`` (the slug lowercases and hyphenates). Returns
    the path unchanged when the model is unknown or slugifies to nothing, or when
    the slug is already present in the stem (so re-running is idempotent).

    Args:
        output_path: The requested output path.
        model: The served model name, or None if the DB recorded none.

    Returns:
        The output path with the model slug embedded in the filename.
    """
    slug = _model_slug(model) if model else ""
    if not slug or slug in output_path.stem:
        return output_path
    return output_path.with_name(f"{output_path.stem}-{slug}{output_path.suffix}")


def _build_payload(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Collect every dashboard data structure into one JSON-serialisable dict."""
    sessions = _fetch_sessions(con)
    total_scrapes = sum(s["scrapes"] for s in sessions)
    total_failed = sum(s["failed"] for s in sessions)

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "sessions": sessions,
        "summary": {
            "sessions": len(sessions),
            "scrapes": total_scrapes,
            "failed_scrapes": total_failed,
            "model": _fetch_model_name(con),
        },
        "kpis": _build_kpis(con),
        "throughput": [
            {"name": label, "points": _fetch_rate_series(con, metric)}
            for metric, label in _RATE_METRICS.items()
        ],
        "concurrency": [
            {"name": label, "points": _fetch_gauge_series(con, metric)}
            for metric, label in _GAUGE_METRICS.items()
        ],
        "kv_cache": [
            {
                "name": "KV-cache utilization %",
                "points": [
                    {"t": p["t"], "value": p["value"] * 100.0}
                    for p in _fetch_gauge_series(con, "vllm:kv_cache_usage_perc")
                ],
            }
        ],
        "latency": [
            {"name": label, "points": _fetch_latency_series(con, base)}
            for base, label in _LATENCY_METRICS.items()
        ],
        "finish_reasons": _fetch_finish_reasons(con),
    }


def _render_html(payload: dict[str, Any]) -> str:
    """Inline the payload into the standalone HTML template."""
    data_json = json.dumps(payload, separators=(",", ":"))
    return _HTML_TEMPLATE.replace("/*__DATA__*/", data_json)


def build_dashboard(db_path: Path, output_path: Path) -> Path:
    """Read the metrics DB and write the standalone HTML dashboard.

    Args:
        db_path: Path to the collector's DuckDB file.
        output_path: Destination HTML file.

    Returns:
        The path the dashboard was written to.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    logger.info("Reading metrics from %s", db_path)
    con = _connect(db_path)
    try:
        payload = _build_payload(con)
    finally:
        con.close()

    logger.info(
        "Loaded %d session(s), %d scrapes across %d metric groups",
        payload["summary"]["sessions"],
        payload["summary"]["scrapes"],
        4,
    )
    # Embed the served model in the filename so dashboards from different models
    # never overwrite one another.
    output_path = _stamp_model(output_path, payload["summary"]["model"])
    html = _render_html(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Wrote dashboard to %s (%d bytes)", output_path, len(html))
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML dashboard from the vLLM metrics DuckDB.",
        epilog=(
            "Examples:\n"
            "  uv run python -m clients.build_dashboard\n"
            "  uv run python -m clients.build_dashboard --db benchmark-output/vllm-metrics.duckdb "
            "--output benchmark-output/dashboard.html\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"path to the metrics DuckDB (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"output HTML path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and build the dashboard."""
    args = _parse_args(argv)
    build_dashboard(Path(args.db), Path(args.output))


# The template carries a design-system-validated categorical palette (blue,
# orange, aqua, yellow) plus sequential blue, in both light and dark steps. The
# data object is injected at the /*__DATA__*/ marker.
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vLLM metrics dashboard</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface-1: #fcfcfb;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
    --axis: #52514e;
    --accent: #6d3fd4;
    --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
    --seq: #256abf;
  }
  html[data-theme="dark"] {
    color-scheme: dark;
    /* Admin-console dark: deep cool page, lifted cards with a visible ring, a
       violet brand accent for chrome. Series steps validated against the card
       surface #1e2128 (dataviz dark gate: all pass, contrast >= 3:1). */
    --page: #16181d; --surface-1: #1e2128;
    --text-primary: #e8eaef; --text-secondary: #9aa1af; --muted: #6f7686;
    --grid: #2a2e37; --baseline: #3a3f4b; --border: rgba(255,255,255,0.08);
    --axis: #d8d2f5;
    --accent: #a78bfa;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
    --seq: #3987e5;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 15px;
    -webkit-font-smoothing: antialiased;
  }
  header {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 16px 24px; border-bottom: 1px solid var(--border);
    background: var(--surface-1); position: sticky; top: 0; z-index: 5;
  }
  header h1 {
    font-size: 17px; margin: 0; font-weight: 650; display: inline-flex; align-items: center; gap: 9px;
  }
  header h1::before {
    content: ""; width: 9px; height: 20px; border-radius: 3px;
    background: var(--accent); display: inline-block;
  }
  header .meta { color: var(--text-secondary); font-size: 13px; }
  header .spacer { flex: 1; }
  button.toggle {
    font: inherit; font-size: 13px; color: var(--text-secondary); background: transparent;
    border: 1px solid var(--border); border-radius: 8px; padding: 6px 13px; cursor: pointer;
    transition: color .12s, border-color .12s;
  }
  button.toggle:hover { color: var(--text-primary); border-color: var(--accent); }
  main { padding: 20px 24px; max-width: 1180px; margin: 0 auto; }
  .kpis {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 28px;
  }
  .tile {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px 16px; position: relative; overflow: hidden;
  }
  .tile::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--accent);
  }
  .tile .label { color: var(--text-secondary); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
  .tile .value {
    font-size: 30px; font-weight: 650; margin-top: 8px; line-height: 1.1;
    font-variant-numeric: tabular-nums;
  }
  .tile .unit { color: var(--muted); font-size: 12px; margin-left: 4px; font-weight: 400; }
  .tile .detail { color: var(--text-secondary); font-size: 11px; margin-top: 6px; font-variant-numeric: tabular-nums; }
  .panel {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 18px 8px; margin-bottom: 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.20);
  }
  .panel h2 { font-size: 14px; margin: 0 0 2px; font-weight: 600; }
  .panel .sub { color: var(--text-secondary); font-size: 12px; margin: 0 0 8px; }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 4px 0 8px; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; color: var(--text-secondary); font-size: 12px; }
  .legend i { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  .chart { position: relative; }
  .chart-reset {
    position: absolute; top: 0; right: 0; z-index: 2;
    background: var(--surface-1); color: var(--text-secondary);
    border: 1px solid var(--border); border-radius: 8px; padding: 3px 9px;
    font: inherit; font-size: 11px; cursor: pointer;
  }
  .chart-reset:hover { color: var(--text-primary); border-color: var(--accent); }
  .zoom-hit { cursor: crosshair; }
  .zoom-band { fill: var(--accent); fill-opacity: 0.14; stroke: var(--accent); stroke-opacity: 0.4; stroke-width: 1; }
  text.axis { fill: var(--axis); }
  .gridline { stroke: var(--grid); stroke-width: 1; }
  .baseline { stroke: var(--baseline); stroke-width: 1; }
  .tooltip {
    position: fixed; pointer-events: none; background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px;
    font-size: 12px; color: var(--text-primary); box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    opacity: 0; transition: opacity .08s; z-index: 10; max-width: 260px;
  }
  .tooltip .tt-row { display: flex; align-items: center; gap: 6px; }
  .tooltip .tt-row i { width: 9px; height: 9px; border-radius: 2px; }
  .tooltip .tt-t { color: var(--text-secondary); margin-bottom: 4px; }
  .empty { color: var(--muted); font-size: 13px; padding: 20px 0; text-align: center; }
  table.dataview { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  table.dataview th, table.dataview td {
    text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); font-size: 12px;
  }
  table.dataview th { color: var(--text-secondary); font-weight: 600; }
  .hidden { display: none; }
</style>
</head>
<body>
<header>
  <h1>vLLM metrics dashboard</h1>
  <div class="meta" id="header-meta"></div>
  <div class="spacer"></div>
  <button class="toggle" id="table-toggle">Table view</button>
  <button class="toggle" id="theme-toggle">Light</button>
</header>
<main id="root"></main>
<div class="tooltip" id="tooltip"></div>
<script id="payload" type="application/json">/*__DATA__*/</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];
const tooltip = document.getElementById("tooltip");
let tableMode = false;

function fmt(v, digits) {
  if (v === null || v === undefined) return "-";
  const d = digits === undefined ? (Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 1 ? 2 : 3) : digits;
  return v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
}
function fmtTime(iso) {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function el(tag, attrs, children) {
  const ns = ["svg","g","path","line","rect","circle","text","polyline","clipPath","defs"].includes(tag);
  const node = ns ? document.createElementNS("http://www.w3.org/2000/svg", tag)
                  : document.createElement(tag);
  for (const k in (attrs || {})) {
    if (k === "text") node.textContent = attrs[k];
    else node.setAttribute(k, attrs[k]);
  }
  (children || []).forEach(c => node.appendChild(c));
  return node;
}

// A monotonically increasing id so each chart's clip-path has a unique target.
let _clipSeq = 0;

// A multi-series line chart with gridlines, axes, crosshair + tooltip, and
// drag-to-zoom on the time (x) axis. Identity comes from the panel legend
// (multi-series) or title (single series), so no direct end-labels are drawn --
// they collide when series share an endpoint.
//
// Zoom: click-drag horizontally to select a time window; the x-domain narrows to
// it and the y-axis rescales to the points now visible (so a spike no longer
// flattens the baseline). Double-click or the "Reset zoom" button restores the
// full range. Each chart owns its own zoom state via this closure, so small
// multiples zoom independently.
function lineChart(series, opts) {
  opts = opts || {};
  // Small multiples render in a narrow column; a compact viewBox keeps the
  // font-to-display ratio close to 1 so axis text stays legible when scaled down.
  const compact = !!opts.compact;
  const W = compact ? 400 : 560, H = compact ? 224 : 240;
  const m = compact ? { t: 12, r: 16, b: 28, l: 46 } : { t: 16, r: 20, b: 34, l: 58 };
  const af = compact ? 13 : 12;  // axis font size, in viewBox user units
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const pts = series.flatMap(s => s.points);
  if (!pts.length) return el("div", { class: "empty", text: "No samples in this range." });

  const times = pts.map(p => new Date(p.t).getTime());
  const fullMin = Math.min(...times), fullMax = Math.max(...times);
  const clipId = "clip" + (_clipSeq++);
  // Which categorical hue each series draws. colorOffset lets a series keep its
  // identity color when a multi-series measure is split into single-series small
  // multiples (e.g. throughput -> Generation on series-1, Prompt on series-2).
  const colorOf = si => SERIES[(si + (opts.colorOffset || 0)) % SERIES.length];

  // Zoom domain over the x (time) axis; null means the full extent.
  let zoom = null;
  // A pixel drag narrower than this is treated as a click, not a zoom selection.
  const DRAG_PX = 6;

  const container = el("div", { class: "chart" });
  const reset = el("button", { class: "chart-reset hidden", type: "button", text: "Reset zoom" });
  container.appendChild(reset);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  container.appendChild(svg);

  reset.addEventListener("click", () => { zoom = null; draw(); });

  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const xMin = zoom ? zoom.lo : fullMin, xMax = zoom ? zoom.hi : fullMax;
    reset.classList.toggle("hidden", !zoom);

    // Points inside the current window drive the y-scale, so zooming past a spike
    // rescales to the detail underneath rather than keeping the spike's headroom.
    const visible = pts.filter(p => { const t = new Date(p.t).getTime(); return t >= xMin && t <= xMax; });
    let yMax = Math.max(...(visible.length ? visible : pts).map(p => p.value), opts.minMax || 0);
    if (yMax <= 0) yMax = 1;
    yMax *= 1.08;
    const xScale = t => m.l + (xMax === xMin ? pw / 2 : (t - xMin) / (xMax - xMin) * pw);
    const yScale = v => m.t + ph - (v / yMax) * ph;

    // Clip the data marks to the plot rect so lines zoomed out of range don't spill.
    const defs = el("defs", {});
    const clip = el("clipPath", { id: clipId });
    clip.appendChild(el("rect", { x: m.l, y: m.t, width: pw, height: ph }));
    defs.appendChild(clip); svg.appendChild(defs);

    // Y gridlines + ticks (fewer on compact charts to avoid crowding)
    const ticks = compact ? 3 : 4;
    for (let i = 0; i <= ticks; i++) {
      const v = yMax * i / ticks, y = yScale(v);
      svg.appendChild(el("line", { class: i === 0 ? "baseline" : "gridline", x1: m.l, y1: y, x2: m.l + pw, y2: y }));
      svg.appendChild(el("text", { class: "axis", "font-size": af, x: m.l - 8, y: y + af / 3, "text-anchor": "end", text: fmt(v) + (opts.yUnit || "") }));
    }
    // X ticks: first/mid/last on wide charts, first/last only when compact
    const xTicks = compact ? [xMin, xMax] : [xMin, (xMin + xMax) / 2, xMax];
    xTicks.forEach((t, i) => {
      const x = xScale(t);
      const anchor = i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle";
      svg.appendChild(el("text", { class: "axis", "font-size": af, x, y: H - m.b / 2.6, "text-anchor": anchor, text: fmtTime(new Date(t).toISOString()) }));
    });

    // Lines (+ optional area fill for a single series), clipped to the plot rect
    const marks = el("g", { "clip-path": `url(#${clipId})` });
    series.forEach((s, si) => {
      if (!s.points.length) return;
      const color = colorOf(si);
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${xScale(new Date(p.t).getTime())},${yScale(p.value)}`).join(" ");
      if (opts.fill && series.length === 1) {
        const area = d + `L${xScale(new Date(s.points[s.points.length-1].t).getTime())},${yScale(0)} L${xScale(new Date(s.points[0].t).getTime())},${yScale(0)} Z`;
        marks.appendChild(el("path", { d: area, fill: color, "fill-opacity": 0.12, stroke: "none" }));
      }
      marks.appendChild(el("path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    });
    svg.appendChild(marks);

    // Crosshair + hover dots, and the drag-selection band (hidden until dragging)
    const band = el("rect", { class: "zoom-band", y: m.t, height: ph, width: 0, opacity: 0 });
    svg.appendChild(band);
    const crosshair = el("line", { class: "gridline", y1: m.t, y2: m.t + ph, opacity: 0 });
    svg.appendChild(crosshair);
    const dots = series.map((s, si) => {
      const c = el("circle", { r: 4, fill: colorOf(si), stroke: "var(--surface-1)", "stroke-width": 2, opacity: 0 });
      svg.appendChild(c); return c;
    });
    const hit = el("rect", { class: "zoom-hit", x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" });
    svg.appendChild(hit);

    // Map a client mouse x to a viewBox px clamped to the plot, and to a time.
    const clientToPx = ev => {
      const box = svg.getBoundingClientRect();
      return Math.max(m.l, Math.min(m.l + pw, (ev.clientX - box.left) / box.width * W));
    };
    const pxToTime = px => xMin + (px - m.l) / pw * (xMax - xMin);

    let dragStart = null;  // viewBox px where a drag began
    const hideHover = () => {
      crosshair.setAttribute("opacity", 0); dots.forEach(d => d.setAttribute("opacity", 0)); tooltip.style.opacity = 0;
    };

    hit.addEventListener("mousedown", ev => {
      dragStart = clientToPx(ev);
      band.setAttribute("x", dragStart); band.setAttribute("width", 0); band.setAttribute("opacity", 1);
      hideHover();
      ev.preventDefault();
    });
    hit.addEventListener("mousemove", ev => {
      const px = clientToPx(ev);
      if (dragStart !== null) {
        // Drawing a zoom selection: draw the band, suppress the hover readout.
        band.setAttribute("x", Math.min(dragStart, px));
        band.setAttribute("width", Math.abs(px - dragStart));
        return;
      }
      const t = pxToTime(px);
      crosshair.setAttribute("x1", xScale(t)); crosshair.setAttribute("x2", xScale(t)); crosshair.setAttribute("opacity", 1);
      let rows = "", refT = null;
      series.forEach((s, si) => {
        if (!s.points.length) { dots[si].setAttribute("opacity", 0); return; }
        let best = s.points[0], bd = Infinity;
        s.points.forEach(p => { const dd = Math.abs(new Date(p.t).getTime() - t); if (dd < bd) { bd = dd; best = p; } });
        dots[si].setAttribute("cx", xScale(new Date(best.t).getTime()));
        dots[si].setAttribute("cy", yScale(best.value)); dots[si].setAttribute("opacity", 1);
        refT = best.t;
        rows += `<div class="tt-row"><i style="background:${colorOf(si)}"></i>${s.name}: <b>${fmt(best.value)}${opts.yUnit || ""}</b></div>`;
      });
      tooltip.innerHTML = `<div class="tt-t">${refT ? fmtTime(refT) : ""}</div>${rows}`;
      tooltip.style.opacity = 1;
      tooltip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 220) + "px";
      tooltip.style.top = (ev.clientY + 14) + "px";
    });
    const finishDrag = ev => {
      if (dragStart === null) return;
      const px = clientToPx(ev);
      const lo = Math.min(dragStart, px), hi = Math.max(dragStart, px);
      dragStart = null;
      band.setAttribute("opacity", 0);
      if (hi - lo < DRAG_PX) return;  // treated as a click, not a zoom
      const tLo = pxToTime(lo), tHi = pxToTime(hi);
      zoom = { lo: tLo, hi: tHi };
      draw();
    };
    hit.addEventListener("mouseup", finishDrag);
    hit.addEventListener("mouseleave", ev => {
      if (dragStart !== null) finishDrag(ev);
      hideHover();
    });
    hit.addEventListener("dblclick", () => { zoom = null; draw(); });
  }

  draw();
  return container;
}

// Horizontal bar chart (categorical), one bar per finish reason.
function barChart(rows) {
  if (!rows.length) return el("div", { class: "empty", text: "No completed requests recorded." });
  const W = 560, barH = 30, gap = 10, m = { t: 8, r: 60, b: 8, l: 96 };
  const H = m.t + m.b + rows.length * (barH + gap);
  const max = Math.max(...rows.map(r => r.count));
  const pw = W - m.l - m.r;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  rows.forEach((r, i) => {
    const y = m.t + i * (barH + gap);
    const w = max ? Math.max(2, r.count / max * pw) : 2;
    const color = SERIES[i % SERIES.length];
    svg.appendChild(el("text", { class: "axis", x: m.l - 10, y: y + barH / 2 + 4, "text-anchor": "end", text: r.reason }));
    const bar = el("rect", { x: m.l, y, width: w, height: barH, rx: 4, fill: color });
    bar.addEventListener("mousemove", ev => {
      tooltip.innerHTML = `<div class="tt-row"><i style="background:${color}"></i>${r.reason}: <b>${fmt(r.count, 0)}</b></div>`;
      tooltip.style.opacity = 1;
      tooltip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 220) + "px";
      tooltip.style.top = (ev.clientY + 14) + "px";
    });
    bar.addEventListener("mouseleave", () => tooltip.style.opacity = 0);
    svg.appendChild(bar);
    svg.appendChild(el("text", { class: "axis", x: m.l + w + 8, y: y + barH / 2 + 4, text: fmt(r.count, 0) }));
  });
  return svg;
}

function legend(series) {
  const box = el("div", { class: "legend" });
  series.forEach((s, i) => {
    const span = el("span", {});
    span.appendChild(el("i", { style: `background:${SERIES[i % SERIES.length]}` }));
    span.appendChild(document.createTextNode(s.name));
    box.appendChild(span);
  });
  return box;
}

function panel(title, sub, bodyNodes, series) {
  const p = el("div", { class: "panel" });
  p.appendChild(el("h2", { text: title }));
  if (sub) p.appendChild(el("p", { class: "sub", text: sub }));
  if (series && series.length > 1) p.appendChild(legend(series));
  bodyNodes.forEach(n => p.appendChild(n));
  return p;
}

function dataTable(series, valueLabel) {
  const table = el("table", { class: "dataview" });
  const head = el("tr", {});
  head.appendChild(el("th", { text: "Time" }));
  series.forEach(s => head.appendChild(el("th", { text: s.name })));
  table.appendChild(head);
  const times = [...new Set(series.flatMap(s => s.points.map(p => p.t)))].sort();
  times.forEach(t => {
    const tr = el("tr", {});
    tr.appendChild(el("td", { text: fmtTime(t) }));
    series.forEach(s => {
      const pt = s.points.find(p => p.t === t);
      tr.appendChild(el("td", { text: pt ? fmt(pt.value) : "-" }));
    });
    table.appendChild(tr);
  });
  return table;
}

function render() {
  const root = document.getElementById("root");
  root.innerHTML = "";
  document.getElementById("header-meta").textContent =
    `${DATA.summary.sessions} session(s) - ${DATA.summary.scrapes} scrapes` +
    (DATA.summary.failed_scrapes ? ` - ${DATA.summary.failed_scrapes} failed` : "") +
    (DATA.summary.model ? ` - model ${DATA.summary.model}` : "") +
    ` - generated ${fmtTime(DATA.generated_at)}`;

  // KPI row
  const kpis = el("div", { class: "kpis" });
  DATA.kpis.forEach(k => {
    const tile = el("div", { class: "tile" });
    tile.appendChild(el("div", { class: "label", text: k.label }));
    const v = el("div", { class: "value", text: k.value });
    v.appendChild(el("span", { class: "unit", text: " " + k.unit }));
    tile.appendChild(v);
    if (k.detail) tile.appendChild(el("div", { class: "detail", text: k.detail }));
    kpis.appendChild(tile);
  });
  root.appendChild(kpis);

  const makeChartOrTable = (series, opts) =>
    tableMode ? dataTable(series) : lineChart(series, opts);

  // Throughput: prompt and generation rates differ by ~100x, so a shared axis
  // hides the smaller series. Split into small multiples (each an honest single
  // axis) rather than a dual-axis chart. Each keeps its original identity color.
  const throughputGrid = el("div", { class: "grid-2" });
  DATA.throughput.forEach((s, si) => {
    const body = tableMode ? dataTable([s]) : lineChart([s], { yUnit: "", colorOffset: si });
    const sub = s.points.length ? "Per-interval rate (tokens/s)" : "No samples in range";
    throughputGrid.appendChild(panel(s.name, sub, [body], null));
  });
  root.appendChild(panel(
    "Token throughput",
    "Per-interval rate from cumulative token counters, split into small multiples (independent scales)",
    [throughputGrid], null));

  // Concurrency
  root.appendChild(panel(
    "Scheduler concurrency", "In-flight and queued requests at each scrape",
    [makeChartOrTable(DATA.concurrency, {})], DATA.concurrency));

  // KV cache
  root.appendChild(panel(
    "KV-cache utilization", "Fraction of the paged KV cache in use (%)",
    [makeChartOrTable(DATA.kv_cache, { yUnit: "%", fill: true })], DATA.kv_cache));

  // Latency small multiples (each its own y-axis -> one axis per chart)
  const latencyGrid = el("div", { class: "grid-2" });
  DATA.latency.forEach(s => {
    const ms = { name: s.name, points: s.points.map(p => ({ t: p.t, value: p.value * 1000 })) };
    const body = tableMode ? dataTable([ms]) : lineChart([ms], { yUnit: "ms" });
    const sub = ms.points.length ? "Mean per interval (ms)" : "No completions in range";
    latencyGrid.appendChild(panel(s.name, sub, [body], null));
  });
  root.appendChild(panel("Request latency", "Client-relevant latency, split into small multiples (independent scales)", [latencyGrid], null));

  // Finish reasons
  root.appendChild(panel(
    "Request outcomes", "Completed requests by finish reason (cumulative)",
    [tableMode
      ? (() => { const s = { name: "count", points: DATA.finish_reasons.map(r => ({ t: r.reason, value: r.count })) };
                 const t = el("table", { class: "dataview" });
                 const h = el("tr", {}); h.appendChild(el("th", { text: "Reason" })); h.appendChild(el("th", { text: "Count" })); t.appendChild(h);
                 DATA.finish_reasons.forEach(r => { const tr = el("tr", {}); tr.appendChild(el("td", { text: r.reason })); tr.appendChild(el("td", { text: fmt(r.count, 0) })); t.appendChild(tr); });
                 return t; })()
      : barChart(DATA.finish_reasons)], null));
}

document.getElementById("theme-toggle").addEventListener("click", () => {
  const html = document.documentElement;
  const dark = html.getAttribute("data-theme") === "dark";
  html.setAttribute("data-theme", dark ? "light" : "dark");
  document.getElementById("theme-toggle").textContent = dark ? "Dark" : "Light";
});
document.getElementById("table-toggle").addEventListener("click", () => {
  tableMode = !tableMode;
  document.getElementById("table-toggle").textContent = tableMode ? "Chart view" : "Table view";
  render();
});
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
