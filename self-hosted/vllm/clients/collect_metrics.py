#!/usr/bin/env python3
"""Continuously collect vLLM Prometheus metrics into DuckDB.

The collector is independent of benchmark traffic. It polls vLLM's /metrics
endpoint, stores every vllm:* sample, and records failed scrapes so monitoring
gaps remain visible.

Usage:
    uv run python -m clients.collect_metrics --interval 1
    uv run python -m clients.collect_metrics --duration 60 --session-name smoke
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import httpx

from clients.benchmark_inference import (
    MetricSample,
    derive_metrics_url,
    parse_prometheus_metrics,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_DATABASE = Path("benchmark-output/vllm-metrics.duckdb")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS collector_sessions (
    session_id VARCHAR PRIMARY KEY,
    session_name VARCHAR NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    stopped_at TIMESTAMPTZ,
    metrics_url VARCHAR NOT NULL,
    interval_seconds DOUBLE NOT NULL,
    collector_pid INTEGER NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS metric_scrape_id_seq START 1;

CREATE TABLE IF NOT EXISTS metric_scrapes (
    scrape_id BIGINT PRIMARY KEY DEFAULT nextval('metric_scrape_id_seq'),
    session_id VARCHAR NOT NULL,
    sequence_number BIGINT NOT NULL,
    scraped_at TIMESTAMPTZ NOT NULL,
    scrape_duration_ms DOUBLE NOT NULL,
    status VARCHAR NOT NULL,
    error VARCHAR NOT NULL,
    sample_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS metric_samples (
    scrape_id BIGINT NOT NULL,
    metric VARCHAR NOT NULL,
    metric_type VARCHAR NOT NULL,
    labels JSON NOT NULL,
    value DOUBLE NOT NULL
);

CREATE OR REPLACE VIEW vllm_metric_samples AS
SELECT
    q.scrape_id,
    q.session_id,
    s.session_name,
    q.sequence_number,
    q.scraped_at,
    q.scrape_duration_ms,
    m.metric,
    m.metric_type,
    m.labels,
    json_extract_string(m.labels, '$.model_name') AS model_name,
    json_extract_string(m.labels, '$.engine') AS engine,
    m.value
FROM metric_samples AS m
JOIN metric_scrapes AS q USING (scrape_id)
JOIN collector_sessions AS s USING (session_id);

CREATE OR REPLACE VIEW vllm_metric_deltas AS
SELECT
    *,
    value - lag(value) OVER (
        PARTITION BY session_id, metric, CAST(labels AS VARCHAR)
        ORDER BY sequence_number
    ) AS delta
FROM vllm_metric_samples;
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible vLLM base URL (env: BASE_URL)",
    )
    parser.add_argument(
        "--metrics-url",
        default=os.environ.get("METRICS_URL"),
        help="Prometheus endpoint (default: derive /metrics from --base-url)",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(os.environ.get("METRICS_DB", DEFAULT_DATABASE)),
        help="DuckDB output path (env: METRICS_DB)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.environ.get("METRICS_INTERVAL", "1")),
        help="seconds between scrape starts (env: METRICS_INTERVAL; default: 1)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="stop after this many seconds; 0 runs until interrupted",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10,
        help="HTTP timeout per scrape in seconds",
    )
    parser.add_argument(
        "--session-name",
        default="",
        help="optional label for this collector process",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero")
    if args.duration < 0:
        raise ValueError("--duration must be zero or greater")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")


def initialize_database(
    database: Path,
    *,
    session_id: str,
    session_name: str,
    metrics_url: str,
    interval: float,
) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    try:
        connection.execute(SCHEMA_SQL)
        connection.execute(
            """
            INSERT INTO collector_sessions
                (session_id, session_name, started_at, metrics_url,
                 interval_seconds, collector_pid)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                session_id,
                session_name,
                datetime.now(timezone.utc),
                metrics_url,
                interval,
                os.getpid(),
            ],
        )
    finally:
        connection.close()


def store_scrape(
    database: Path,
    *,
    session_id: str,
    sequence_number: int,
    scraped_at: datetime,
    scrape_duration_ms: float,
    status: str,
    error: str,
    samples: list[MetricSample],
) -> int:
    """Atomically store one scrape and return its database ID."""
    connection = duckdb.connect(str(database))
    try:
        connection.begin()
        scrape_id = connection.execute(
            """
            INSERT INTO metric_scrapes
                (session_id, sequence_number, scraped_at, scrape_duration_ms,
                 status, error, sample_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING scrape_id
            """,
            [
                session_id,
                sequence_number,
                scraped_at,
                scrape_duration_ms,
                status,
                error,
                len(samples),
            ],
        ).fetchone()
        assert scrape_id is not None  # nosec B101 - INSERT ... RETURNING always yields a row
        scrape_id_value = int(scrape_id[0])
        if samples:
            connection.execute(
                """
                INSERT INTO metric_samples
                    (scrape_id, metric, metric_type, labels, value)
                SELECT
                    ?,
                    unnest(?),
                    unnest(?),
                    unnest(?)::JSON,
                    unnest(?)
                """,
                [
                    scrape_id_value,
                    [sample.metric for sample in samples],
                    [sample.metric_type for sample in samples],
                    [sample.labels_json for sample in samples],
                    [sample.value for sample in samples],
                ],
            )
        connection.commit()
        return scrape_id_value
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def finish_session(database: Path, session_id: str) -> None:
    connection = duckdb.connect(str(database))
    try:
        connection.execute(
            "UPDATE collector_sessions SET stopped_at = ? WHERE session_id = ?",
            [datetime.now(timezone.utc), session_id],
        )
    finally:
        connection.close()


def collect(args: argparse.Namespace, stop_event: threading.Event) -> None:
    metrics_url = args.metrics_url or derive_metrics_url(args.base_url)
    session_id = str(uuid.uuid4())
    session_name = args.session_name or f"collector-{session_id[:8]}"
    initialize_database(
        args.database,
        session_id=session_id,
        session_name=session_name,
        metrics_url=metrics_url,
        interval=args.interval,
    )

    logger.info("collecting %s every %.2fs", metrics_url, args.interval)
    logger.info("DuckDB: %s", args.database)
    logger.info("session: %s (%s)", session_name, session_id)

    started = time.monotonic()
    deadline = started + args.duration if args.duration else None
    next_scrape = started
    sequence_number = 0
    with httpx.Client(timeout=args.timeout) as client:
        try:
            while not stop_event.is_set():
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    break
                wake_at = (
                    min(next_scrape, deadline) if deadline is not None else next_scrape
                )
                if now < wake_at and stop_event.wait(wake_at - now):
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break

                sequence_number += 1
                scrape_started = time.monotonic()
                scraped_at = datetime.now(timezone.utc)
                status = "ok"
                error = ""
                samples: list[MetricSample] = []
                try:
                    response = client.get(metrics_url)
                    response.raise_for_status()
                    samples = list(parse_prometheus_metrics(response.text).values())
                except (httpx.HTTPError, ValueError) as exc:
                    status = "error"
                    error = str(exc)
                    logger.warning("scrape %s failed: %s", sequence_number, exc)

                duration_ms = (time.monotonic() - scrape_started) * 1000
                try:
                    store_scrape(
                        args.database,
                        session_id=session_id,
                        sequence_number=sequence_number,
                        scraped_at=scraped_at,
                        scrape_duration_ms=duration_ms,
                        status=status,
                        error=error,
                        samples=samples,
                    )
                except duckdb.Error as exc:
                    logger.error(
                        "scrape %s could not be stored; database may be locked: %s",
                        sequence_number,
                        exc,
                    )
                else:
                    logger.info(
                        "scrape %s: %s samples in %.1fms",
                        sequence_number,
                        len(samples),
                        duration_ms,
                    )
                next_scrape += args.interval
                if next_scrape < time.monotonic():
                    next_scrape = time.monotonic()
        finally:
            try:
                finish_session(args.database, session_id)
            except duckdb.Error as exc:
                logger.warning("could not mark collector session stopped: %s", exc)
            logger.info("collector stopped after %s scrapes", sequence_number)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    stop_event = threading.Event()

    def request_stop(signum: int, _frame: object) -> None:
        logger.info("received signal %s; stopping", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        collect(args, stop_event)
    except (OSError, duckdb.Error) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
