"""Tests for the DuckDB-to-HTML dashboard generator."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from clients import build_dashboard
from clients.benchmark_inference import MetricSample
from clients.collect_metrics import initialize_database, store_scrape

_LABELS = '{"model_name":"test-model","engine":"0"}'


def _sample(metric: str, value: float, labels: str = _LABELS) -> MetricSample:
    """Build one parsed Prometheus sample for the fixture."""
    return MetricSample(
        metric=metric, metric_type="gauge", labels_json=labels, value=value
    )


def _make_db(path: Path) -> None:
    """Create a metrics database via the collector's own schema and writers.

    Reusing ``initialize_database`` / ``store_scrape`` guarantees the fixture
    exercises the real tables and analytics views rather than a hand-rolled copy
    that could drift from the collector.
    """
    initialize_database(
        path,
        session_id="s1",
        session_name="collector-s1",
        metrics_url="http://x/metrics",
        interval=5.0,
    )
    base = datetime(2026, 7, 22, 4, 0, 0, tzinfo=timezone.utc)
    # Two scrapes 5s apart; a cumulative counter and a histogram sum/count pair.
    for i in range(2):
        samples = [
            _sample(
                "vllm:generation_tokens_total", 1000.0 + i * 500.0
            ),  # +500/5s -> 100/s
            _sample("vllm:num_requests_running", float(i)),
            _sample("vllm:time_to_first_token_seconds_sum", 0.2 + i * 0.4),
            _sample("vllm:time_to_first_token_seconds_count", float(i * 2)),
        ]
        if i == 1:
            samples.append(
                _sample(
                    "vllm:request_success_total",
                    7.0,
                    '{"finished_reason":"stop","model_name":"test-model"}',
                )
            )
        store_scrape(
            path,
            session_id="s1",
            sequence_number=i,
            scraped_at=base + timedelta(seconds=5 * i),
            scrape_duration_ms=1.0,
            status="ok",
            error="",
            samples=samples,
        )


class BuildDashboardTest(unittest.TestCase):
    def test_rate_series_computes_per_second(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "metrics.duckdb"
            _make_db(database)
            con = build_dashboard._connect(database)
            try:
                series = build_dashboard._fetch_rate_series(
                    con, "vllm:generation_tokens_total"
                )
            finally:
                con.close()
        # 500 tokens over 5 seconds = 100 tokens/s, one delta point.
        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(series[0]["value"], 100.0)

    def test_latency_mean_uses_interval_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "metrics.duckdb"
            _make_db(database)
            con = build_dashboard._connect(database)
            try:
                series = build_dashboard._fetch_latency_series(
                    con, "vllm:time_to_first_token_seconds"
                )
            finally:
                con.close()
        # delta_sum 0.4 / delta_count 2 = 0.2s mean over the interval.
        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(series[0]["value"], 0.2)

    def test_finish_reasons_filters_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "metrics.duckdb"
            _make_db(database)
            con = build_dashboard._connect(database)
            try:
                reasons = build_dashboard._fetch_finish_reasons(con)
            finally:
                con.close()
        self.assertEqual(reasons, [{"reason": "stop", "count": 7.0}])

    def test_model_name_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "metrics.duckdb"
            _make_db(database)
            con = build_dashboard._connect(database)
            try:
                model = build_dashboard._fetch_model_name(con)
            finally:
                con.close()
        self.assertEqual(model, "test-model")

    def test_format_mtok_uses_millions(self) -> None:
        self.assertEqual(build_dashboard._format_mtok(347_990.0), "0.348")
        self.assertEqual(build_dashboard._format_mtok(2_500_000.0), "2.500")

    def test_model_slug_lowercases_and_hyphenates(self) -> None:
        self.assertEqual(build_dashboard._model_slug("qwen3.6-35b"), "qwen3-6-35b")
        self.assertEqual(
            build_dashboard._model_slug("meta-llama/Llama-3-8B-Instruct"),
            "meta-llama-llama-3-8b-instruct",
        )
        self.assertEqual(build_dashboard._model_slug("!!!"), "")

    def test_stamp_model_inserts_slug_before_suffix(self) -> None:
        stamped = build_dashboard._stamp_model(
            Path("out/dashboard.html"), "qwen3.6-35b"
        )
        self.assertEqual(stamped, Path("out/dashboard-qwen3-6-35b.html"))

    def test_stamp_model_is_idempotent_and_safe(self) -> None:
        base = Path("out/dashboard.html")
        # Already stamped -> unchanged (no double suffix).
        stamped = build_dashboard._stamp_model(base, "qwen3.6-35b")
        self.assertEqual(build_dashboard._stamp_model(stamped, "qwen3.6-35b"), stamped)
        # Unknown or empty model -> unchanged.
        self.assertEqual(build_dashboard._stamp_model(base, None), base)
        self.assertEqual(build_dashboard._stamp_model(base, "!!!"), base)

    def test_missing_db_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                FileNotFoundError, "Metrics database not found"
            ):
                build_dashboard._connect(Path(temp_dir) / "nope.duckdb")

    def test_build_writes_self_contained_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "metrics.duckdb"
            _make_db(database)
            out = Path(temp_dir) / "dash.html"
            written = build_dashboard.build_dashboard(database, out)
            # The served model is stamped into the filename.
            self.assertEqual(written.name, "dash-test-model.html")
            html = written.read_text(encoding="utf-8")
        # Self-contained: no external script/style/CDN references.
        self.assertNotIn("<script src=", html)
        self.assertNotIn("http://", html.split('type="application/json"')[0])
        # The embedded payload is valid JSON with the expected groups.
        match = re.search(
            r'<script id="payload" type="application/json">(.*?)</script>', html, re.S
        )
        assert match is not None
        payload = json.loads(match.group(1))
        self.assertEqual(payload["summary"]["model"], "test-model")
        self.assertLessEqual(
            {"throughput", "concurrency", "kv_cache", "latency", "finish_reasons"},
            set(payload),
        )

    def test_line_charts_support_zoom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "metrics.duckdb"
            _make_db(database)
            out = Path(temp_dir) / "dash.html"
            html = build_dashboard.build_dashboard(database, out).read_text(
                encoding="utf-8"
            )
        # Drag-to-zoom wiring: a reset control, a selection band, and the
        # mousedown handler that starts a drag are all present in the template.
        self.assertIn("chart-reset", html)
        self.assertIn("zoom-band", html)
        self.assertIn('"mousedown"', html)


if __name__ == "__main__":
    unittest.main()
