"""Tests for the complexity-tier breakdown chart's data shaping."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location(
    "plot_complexity_breakdown", _SCRIPTS_DIR / "plot_complexity_breakdown.py"
)
plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot)


def _task(name: str, cx: str, score: float | None, **artifacts: int) -> dict:
    """Build a run-summary task row."""
    return {
        "task": name,
        "complexity": cx,
        "task_score": score,
        "eval_scores": {k: {"total": v} for k, v in artifacts.items()} or None,
    }


class TierRowsTest(unittest.TestCase):
    def test_groups_by_tier_in_fixed_order(self) -> None:
        # low < medium < high is the encoding; the ramp depends on this order,
        # so it must never fall out of the data's ordering.
        summary = {
            "tasks": [
                _task("c", "high", 40.0),
                _task("a", "low", 70.0),
                _task("b", "medium", 55.0),
            ]
        }
        self.assertEqual(list(plot._tier_rows(summary)), ["low", "medium", "high"])

    def test_sorts_each_tier_by_descending_score(self) -> None:
        summary = {
            "tasks": [
                _task("lo", "low", 42.8),
                _task("hi", "low", 72.0),
                _task("mid", "low", 64.4),
            ]
        }
        got = [r["task"] for r in plot._tier_rows(summary)["low"]]
        self.assertEqual(got, ["hi", "mid", "lo"])

    def test_drops_empty_tiers(self) -> None:
        # A dataset need not populate every tier; an empty band must not be drawn.
        summary = {"tasks": [_task("a", "low", 70.0)]}
        self.assertEqual(list(plot._tier_rows(summary)), ["low"])

    def test_excludes_unscored_tasks(self) -> None:
        # A failed task has no score; averaging it as 0 would understate the tier.
        summary = {"tasks": [_task("a", "low", 70.0), _task("b", "low", None)]}
        self.assertEqual([r["task"] for r in plot._tier_rows(summary)["low"]], ["a"])

    def test_raises_when_no_task_has_complexity(self) -> None:
        summary = {"tasks": [{"task": "a", "task_score": 50.0}]}
        with self.assertRaisesRegex(SystemExit, "complexity"):
            plot._tier_rows(summary)


class ArtifactProfileTest(unittest.TestCase):
    def test_means_follow_the_declared_artifact_order(self) -> None:
        # The panel reads left-to-right as the task progressed, so the order is
        # the skill's, not the dict's.
        rows = [
            _task(
                "a",
                "high",
                50.0,
                implementation=10,
                github_issue=80,
                lld=60,
                review=50,
                testing=40,
            )
        ]
        self.assertEqual(plot._artifact_profile(rows), [80, 60, 50, 40, 10])

    def test_averages_across_tasks(self) -> None:
        rows = [
            _task("a", "low", 60.0, github_issue=80),
            _task("b", "low", 40.0, github_issue=60),
        ]
        self.assertEqual(plot._artifact_profile(rows)[0], 70)

    def test_missing_artifact_is_none_not_zero(self) -> None:
        # None leaves a gap in the line; 0 would draw a cliff that did not happen.
        rows = [_task("a", "low", 60.0, github_issue=80)]
        self.assertEqual(plot._artifact_profile(rows)[1:], [None] * 4)


class RenderTest(unittest.TestCase):
    def test_writes_a_png_named_for_harness_skill_and_scope(self) -> None:
        summary = {
            "model_slug": "test-model",
            "num_tasks": 2,
            "num_scored": 2,
            "refs": ["1.0.0", "2.0.0"],
            "mean_task_score_excl_failed": 55.0,
            "tasks": [
                _task("a", "low", 70.0, github_issue=80, implementation=60),
                _task("b", "high", 40.0, github_issue=70, implementation=20),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = plot._plot(
                summary,
                mode="light",
                harness="pi",
                skill="swe3",
                scope="repo-v2",
                out_dir=Path(tmp),
            )
            self.assertEqual(out.name, "complexity-test-model-pi-swe3-repo-v2.png")
            self.assertGreater(out.stat().st_size, 0)

    def test_dark_variant_gets_its_own_filename(self) -> None:
        summary = {
            "model_slug": "m",
            "tasks": [_task("a", "low", 70.0, github_issue=80)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = plot._plot(
                summary,
                mode="dark",
                harness="pi",
                skill="swe3",
                scope="repo-v2",
                out_dir=Path(tmp),
            )
            self.assertEqual(out.name, "complexity-m-pi-swe3-repo-v2-dark.png")


class LoadSummaryTest(unittest.TestCase):
    def test_reads_the_scoped_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "m" / "pi" / "swe3" / "repo-v2"
            folder.mkdir(parents=True)
            (folder / "run-summary.json").write_text(
                json.dumps({"model_slug": "m"}), encoding="utf-8"
            )
            got = plot._load_summary(root, "m", "pi", "swe3", "repo-v2")
            self.assertEqual(got["model_slug"], "m")

    def test_missing_summary_is_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SystemExit, "no run summary"):
                plot._load_summary(Path(tmp), "m", "pi", "swe3", "nope")


if __name__ == "__main__":
    unittest.main()
