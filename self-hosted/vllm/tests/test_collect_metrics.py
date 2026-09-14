import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from clients.benchmark_inference import MetricSample
from clients.collect_metrics import finish_session, initialize_database, store_scrape


class CollectMetricsTest(unittest.TestCase):
    def test_stores_samples_and_exposes_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "metrics.duckdb"
            initialize_database(
                database,
                session_id="session",
                session_name="test",
                metrics_url="http://localhost:8000/metrics",
                interval=5,
            )
            first = MetricSample(
                metric="vllm:prompt_tokens_total",
                metric_type="counter",
                labels_json='{"engine":"0","model_name":"model"}',
                value=10,
            )
            second = MetricSample(
                metric="vllm:prompt_tokens_total",
                metric_type="counter",
                labels_json='{"engine":"0","model_name":"model"}',
                value=25,
            )
            for sequence, sample in enumerate((first, second), start=1):
                store_scrape(
                    database,
                    session_id="session",
                    sequence_number=sequence,
                    scraped_at=datetime.now(timezone.utc),
                    scrape_duration_ms=2.5,
                    status="ok",
                    error="",
                    samples=[sample],
                )
            finish_session(database, "session")

            connection = duckdb.connect(str(database), read_only=True)
            try:
                rows = connection.execute(
                    """
                    SELECT sequence_number, model_name, value, delta
                    FROM vllm_metric_deltas
                    ORDER BY sequence_number
                    """
                ).fetchall()
                stopped_at = connection.execute(
                    "SELECT stopped_at FROM collector_sessions"
                ).fetchone()
            finally:
                connection.close()

            self.assertEqual(rows[0], (1, "model", 10.0, None))
            self.assertEqual(rows[1], (2, "model", 25.0, 15.0))
            self.assertIsNotNone(stopped_at)

    def test_records_failed_scrape_without_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "metrics.duckdb"
            initialize_database(
                database,
                session_id="session",
                session_name="test",
                metrics_url="http://localhost:8000/metrics",
                interval=5,
            )
            store_scrape(
                database,
                session_id="session",
                sequence_number=1,
                scraped_at=datetime.now(timezone.utc),
                scrape_duration_ms=10,
                status="error",
                error="connection refused",
                samples=[],
            )

            connection = duckdb.connect(str(database), read_only=True)
            try:
                scrape = connection.execute(
                    "SELECT status, error, sample_count FROM metric_scrapes"
                ).fetchone()
                sample_count = connection.execute(
                    "SELECT count(*) FROM metric_samples"
                ).fetchone()
            finally:
                connection.close()

            self.assertEqual(scrape, ("error", "connection refused", 0))
            self.assertEqual(sample_count, (0,))


if __name__ == "__main__":
    unittest.main()
