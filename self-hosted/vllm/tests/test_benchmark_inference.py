import unittest

from clients.benchmark_inference import (
    MetricSample,
    RequestResult,
    _delta_has_output,
    build_summary,
    derive_metrics_url,
    parse_prometheus_metrics,
    percentile,
)


def _result(request_id: int, ttft_ms: float, ttlt_ms: float) -> RequestResult:
    return RequestResult(
        run_id="run",
        request_id=request_id,
        prompt_id="prompt",
        prompt_sha256="abc",
        prompt_characters=10,
        model="model",
        started_at_utc="2026-01-01T00:00:00+00:00",
        status="ok",
        error="",
        client_queue_ms=0.0,
        ttft_ms=ttft_ms,
        ttlt_ms=ttlt_ms,
        e2e_ms=ttlt_ms + 1,
        generation_ms=ttlt_ms - ttft_ms,
        tpot_ms=10.0,
        output_tokens_per_s=100.0,
        prompt_tokens=10,
        output_tokens=20,
        total_tokens=30,
        chunks=20,
        output_characters=80,
        finish_reason="stop",
        server_request_id=f"request-{request_id}",
        max_tokens=20,
        temperature=0.0,
    )


def _metric(name: str, value: float) -> MetricSample:
    return MetricSample(
        metric=name,
        metric_type="histogram",
        labels_json='{"model_name":"model"}',
        value=value,
    )


class BenchmarkInferenceTest(unittest.TestCase):
    def test_derive_metrics_url(self) -> None:
        self.assertEqual(
            derive_metrics_url("http://localhost:8000/v1"),
            "http://localhost:8000/metrics",
        )
        self.assertEqual(
            derive_metrics_url("https://example.test/prefix/v1/"),
            "https://example.test/prefix/metrics",
        )

    def test_parse_prometheus_metrics_filters_non_vllm_and_created(self) -> None:
        text = """
# HELP vllm:time_to_first_token_seconds TTFT
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{model_name="model"} 1.5
vllm:time_to_first_token_seconds_count{model_name="model"} 2
vllm:time_to_first_token_seconds_created{model_name="model"} 123
# HELP process_cpu_seconds CPU
# TYPE process_cpu_seconds counter
process_cpu_seconds_total 5
"""
        metrics = parse_prometheus_metrics(text)
        names = {sample.metric for sample in metrics.values()}
        self.assertEqual(
            names,
            {
                "vllm:time_to_first_token_seconds_sum",
                "vllm:time_to_first_token_seconds_count",
            },
        )

    def test_percentile_uses_linear_interpolation(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile(values, 0.5), 2.5)
        self.assertAlmostEqual(percentile(values, 0.95), 3.85)
        self.assertIsNone(percentile([], 0.5))

    def test_delta_output_detection(self) -> None:
        self.assertFalse(_delta_has_output({"role": "assistant"}))
        self.assertTrue(_delta_has_output({"content": "token"}))
        self.assertTrue(_delta_has_output({"reasoning_content": "thought"}))
        self.assertTrue(_delta_has_output({"tool_calls": [{"index": 0}]}))

    def test_summary_combines_client_and_server_metrics(self) -> None:
        before_samples = [
            _metric("vllm:time_to_first_token_seconds_sum", 1.0),
            _metric("vllm:time_to_first_token_seconds_count", 2.0),
            _metric("vllm:prompt_tokens_total", 100.0),
        ]
        after_samples = [
            _metric("vllm:time_to_first_token_seconds_sum", 1.4),
            _metric("vllm:time_to_first_token_seconds_count", 4.0),
            _metric("vllm:prompt_tokens_total", 120.0),
        ]
        before = {sample.key: sample for sample in before_samples}
        after = {sample.key: sample for sample in after_samples}

        summary = build_summary(
            run_id="run",
            run_name="test",
            model="model",
            base_url="http://localhost:8000/v1",
            requests=2,
            concurrency=2,
            warmup=1,
            benchmark_duration_s=2.0,
            results=[_result(1, 100.0, 300.0), _result(2, 200.0, 500.0)],
            before_metrics=before,
            after_metrics=after,
        )

        self.assertEqual(summary["successful_requests"], 2)
        self.assertEqual(summary["ttft_ms_p50"], 150.0)
        self.assertAlmostEqual(summary["server_ttft_ms_mean"], 200.0)
        self.assertEqual(summary["server_prompt_tokens_delta"], 20.0)
        self.assertEqual(summary["aggregate_output_tokens_per_s"], 20.0)


if __name__ == "__main__":
    unittest.main()
