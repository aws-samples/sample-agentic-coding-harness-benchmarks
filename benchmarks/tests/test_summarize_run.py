"""Tests for the run summarizer (run-summary.json / .md)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_SUMMARIZE_PATH = _SCRIPTS_DIR / "summarize_run.py"
_spec = importlib.util.spec_from_file_location("summarize_run", _SUMMARIZE_PATH)
assert _spec is not None and _spec.loader is not None
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)

_ARTIFACTS = ("github-issue.md", "lld.md", "review.md", "testing.md")


def _write_task(
    scope_dir: Path,
    task: str,
    *,
    score: float | None,
    n_artifacts: int = 4,
    cost: float = 5.0,
    turns: int = 20,
    agent_invocations: int = 1,
    topped_up_artifacts: list[str] | None = None,
    ref: str = "1.2.3",
) -> None:
    """Create a task folder with metrics.json, artifacts, and optional eval.json."""
    d = scope_dir / task
    d.mkdir(parents=True)
    for name in _ARTIFACTS[:n_artifacts]:
        (d / name).write_text(f"# {name}\nbody\n", encoding="utf-8")
    metrics: dict[str, Any] = {
        "task": task,
        "ref": ref,
        "model": "test-model",
        "model_slug": "test-model",
        "agent": "claude",
        "skill": "swe3",
        "provider": "endpoint",
        "complexity": "medium",
        "serving": {
            "instance_type": "g6e.12xlarge",
            "tensor_parallel_size": 4,
            "precision": "BF16",
            "context_window": 200000,
        },
        "total_cost_usd": cost,
        "is_error": False,
        "agent_invocations": agent_invocations,
        "topped_up_artifacts": topped_up_artifacts or [],
        "metrics_that_matter": {
            "num_turns": turns,
            "input_tokens": 1000,
            "output_tokens": 200,
            "latency_seconds": 100.0,
            "cache_read_tokens": 900,
            "cache_write_tokens": 100,
            "prefix_cache_hit_rate": 0.9,
            "generation_tokens_per_sec": 2.0,
        },
        "vllm_prometheus": {
            "gauges_sampled": {
                "gauges": {"vllm:kv_cache_usage_perc": {"peak": 0.08, "mean": 0.05}}
            }
        },
    }
    (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if score is not None:
        (d / "eval.json").write_text(
            json.dumps({"task": task, "model": "test-model", "task_score": score}),
            encoding="utf-8",
        )


class SummarizeRunTest(unittest.TestCase):
    def test_clean_run_mean_over_all_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp) / "test-model" / "claude-code" / "some-repo"
            _write_task(scope, "task-a", score=60.0, cost=4.0)
            _write_task(scope, "task-b", score=50.0, cost=6.0)
            s = summarize._summarize(scope, run_date="2026-07-24")
            self.assertEqual(s["num_scored"], 2)
            self.assertEqual(s["num_failed"], 0)
            self.assertEqual(s["mean_task_score_excl_failed"], 55.0)
            self.assertEqual(s["mean_cost_usd_excl_failed"], 5.0)
            self.assertEqual(s["serving"]["precision"], "BF16")
            self.assertEqual(s["model_slug"], "test-model")
            # agent + skill come from the metrics.json identity fields.
            self.assertEqual(s["agent"], "claude")
            self.assertEqual(s["skill"], "swe3")

    def test_efficiency_signals_folded_into_task_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp) / "test-model" / "pi" / "some-repo"
            _write_task(scope, "task-a", score=60.0)
            row = summarize._summarize(scope, run_date=None)["tasks"][0]
            # Derived cache/KV signals are carried up from metrics.json.
            self.assertEqual(row["prefix_cache_hit_rate"], 0.9)
            self.assertEqual(row["cache_read_tokens"], 900)
            self.assertEqual(row["cache_write_tokens"], 100)
            self.assertEqual(row["generation_tokens_per_sec"], 2.0)
            self.assertEqual(row["kv_cache_usage"], {"peak": 0.08, "mean": 0.05})
            # A single-shot run defaults to one invocation, no top-ups.
            self.assertEqual(row["agent_invocations"], 1)
            self.assertEqual(row["topped_up_artifacts"], [])
            # total_tokens is partition-aware (issue #136): here cache_read(900) +
            # cache_write(100) == input_tokens(1000), so the cache is a PARTITION
            # of input (self-hosted vLLM style) and is already counted inside
            # input_tokens. total_tokens must therefore be input + output only,
            # NOT input + output + cache (which would ~2x double-count).
            self.assertEqual(row["total_tokens"], 1000 + 200)

    def test_reads_normalized_metrics_block_and_result_subtype(self) -> None:
        # New-format metrics.json carries a "metrics" block (total_cost_usd) and a
        # top-level result_subtype; summarize must read them. total_tokens is
        # RECOMPUTED (issue #136), not trusted from the block: the stored value
        # below is deliberately wrong (99999) to prove summarize ignores it. Here
        # cache_read(2000)+cache_write(50)=2050 vs input(5) is additive (not a
        # partition), so the recomputed total is 5 + 100 + 2000 + 50 = 2155.
        import json

        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp) / "m" / "claude-code" / "r"
            d = scope / "task-a"
            d.mkdir(parents=True)
            for name in ("github-issue.md", "lld.md", "review.md", "testing.md"):
                (d / name).write_text("x", encoding="utf-8")
            (d / "eval.json").write_text(
                json.dumps({"task_score": 70.0, "scores": {}}), encoding="utf-8"
            )
            (d / "metrics.json").write_text(
                json.dumps(
                    {
                        "task": "task-a",
                        "complexity": "medium",
                        "result_subtype": "error_max_turns",
                        "metrics": {
                            "input_tokens": 5,
                            "output_tokens": 100,
                            "cache_read_tokens": 2000,
                            "cache_write_tokens": 50,
                            "total_tokens": 99999,
                            "total_cost_usd": 1.23,
                            "num_turns": 40,
                            "latency_seconds": 12.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            row = summarize._summarize(scope, run_date=None)["tasks"][0]
            self.assertEqual(row["total_tokens"], 2155)
            self.assertEqual(row["total_cost_usd"], 1.23)
            self.assertEqual(row["result_subtype"], "error_max_turns")
            self.assertEqual(row["input_tokens"], 5)

    def test_topup_provenance_surfaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp) / "test-model" / "pi" / "some-repo"
            _write_task(
                scope,
                "task-a",
                score=55.0,
                agent_invocations=2,
                topped_up_artifacts=["patch.diff", "implementation.md"],
            )
            row = summarize._summarize(scope, run_date=None)["tasks"][0]
            self.assertEqual(row["agent_invocations"], 2)
            self.assertEqual(
                row["topped_up_artifacts"], ["patch.diff", "implementation.md"]
            )

    def test_failed_task_excluded_from_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp) / "test-model" / "claude-code" / "some-repo"
            _write_task(scope, "good", score=60.0, cost=4.0)
            # A 0-score task (missing artifact): excluded from the mean, still listed.
            _write_task(scope, "bad", score=0.0, n_artifacts=3, cost=9.0)
            s = summarize._summarize(scope, run_date=None)
            self.assertEqual(s["num_scored"], 1)
            self.assertEqual(s["failed_tasks"], ["bad"])
            self.assertEqual(s["mean_task_score_excl_failed"], 60.0)
            # Failed task's cost is excluded too.
            self.assertEqual(s["mean_cost_usd_excl_failed"], 4.0)

    def test_missing_eval_counts_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp) / "test-model" / "claude-code" / "some-repo"
            _write_task(scope, "unscored", score=None)
            s = summarize._summarize(scope, run_date=None)
            self.assertEqual(s["num_failed"], 1)
            self.assertIsNone(s["mean_task_score_excl_failed"])

    def test_markdown_flags_failure_and_serving(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp) / "test-model" / "claude-code" / "some-repo"
            _write_task(scope, "good", score=60.0)
            _write_task(scope, "bad", score=0.0, n_artifacts=3)
            md = summarize._render_markdown(summarize._summarize(scope, run_date=None))
            self.assertIn("model failure", md)
            self.assertIn("precision=BF16", md)
            self.assertIn("1 failed (bad)", md)

    def test_no_tasks_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp) / "empty" / "repo"
            scope.mkdir(parents=True)
            with self.assertRaises(SystemExit):
                summarize._summarize(scope, run_date=None)


class RefsTest(unittest.TestCase):
    """A dataset may pin a different ref per task; the summary must say so."""

    def test_single_ref_run_reports_that_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp) / "m" / "claude-code" / "swe3" / "repo"
            _write_task(scope, "task-a", score=60.0)
            _write_task(scope, "task-b", score=50.0)
            s = summarize._summarize(scope, run_date=None)
            self.assertEqual(s["ref"], "1.2.3")
            self.assertEqual(s["refs"], ["1.2.3"])
            self.assertIn("ref 1.2.3", summarize._render_markdown(s))

    def test_multi_ref_run_has_no_single_ref(self) -> None:
        # Reporting the first task's ref would describe only 1 of N clones.
        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp) / "m" / "pi" / "swe3" / "repo-v2"
            _write_task(scope, "task-a", score=60.0, ref="1.23.0")
            _write_task(scope, "task-b", score=50.0, ref="1.27.1")
            s = summarize._summarize(scope, run_date=None)
            self.assertIsNone(s["ref"])
            self.assertEqual(s["refs"], ["1.23.0", "1.27.1"])
            self.assertIn("2 refs: 1.23.0, 1.27.1", summarize._render_markdown(s))

    def test_each_task_row_carries_its_own_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scope = Path(tmp) / "m" / "pi" / "swe3" / "repo-v2"
            _write_task(scope, "task-a", score=60.0, ref="1.23.0")
            _write_task(scope, "task-b", score=50.0, ref="1.27.1")
            rows = {
                r["task"]: r["ref"]
                for r in summarize._summarize(scope, run_date=None)["tasks"]
            }
            self.assertEqual(rows, {"task-a": "1.23.0", "task-b": "1.27.1"})


if __name__ == "__main__":
    unittest.main()
