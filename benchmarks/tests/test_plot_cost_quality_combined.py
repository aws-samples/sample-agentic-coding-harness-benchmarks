"""Tests for per-model harness selection on the combined cost/quality chart.

The combined chart plots one point per model, so it has to decide which
harness's run represents that model. Dominance settles the clear cases; where
neither harness dominates, the lower cost/point wins. Both paths are load-
bearing for what the chart claims, so both are pinned here.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_SPEC = importlib.util.spec_from_file_location(
    "plot_cost_quality_combined", _SCRIPTS_DIR / "plot_cost_quality_combined.py"
)
combined = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(combined)

cq = combined.cq


def _point(model: str, harness: str, score: float, cost: float) -> cq.ModelPoint:
    """Build a minimal ModelPoint; only model/harness/score/cost matter here."""
    return cq.ModelPoint(
        model=model,
        mean_cost=cost,
        mean_score=score,
        n_tasks=5,
        n_scored=5,
        excluded=[],
        hosting="Bedrock",
        harness=harness,
    )


class SelectBestHarnessTest(unittest.TestCase):
    def test_keeps_the_dominating_harness(self) -> None:
        # pi is both higher-scoring and cheaper, so it wins outright.
        points = [
            _point("claude-opus-5", "claude-code", 70.76, 24.05),
            _point("claude-opus-5", "pi", 75.72, 8.28),
        ]

        winners, records = combined._select_best_harness(points)

        self.assertEqual([w.harness for w in winners], ["pi"])
        self.assertEqual(records[0]["decided_by"], "dominance")

    def test_breaks_a_non_dominated_tie_on_cost_per_point(self) -> None:
        # Claude Code scores 1.5 higher but costs 6.5x; cost/point picks pi.
        points = [
            _point("claude-sonnet-5", "claude-code", 68.04, 24.64),
            _point("claude-sonnet-5", "pi", 66.52, 3.81),
        ]

        winners, records = combined._select_best_harness(points)

        self.assertEqual([w.harness for w in winners], ["pi"])
        self.assertEqual(records[0]["decided_by"], "cost_per_point")

    def test_tie_break_can_pick_the_lower_scoring_harness(self) -> None:
        # Guards against a score-only shortcut: here the cheaper run is also the
        # better value, and it happens to be Claude Code.
        points = [
            _point("kimi-k2.7-code", "claude-code", 55.44, 6.2563),
            _point("kimi-k2.7-code", "pi", 60.68, 11.0351),
        ]

        winners, _ = combined._select_best_harness(points)

        self.assertEqual([w.harness for w in winners], ["claude-code"])

    def test_single_harness_model_passes_through(self) -> None:
        points = [_point("grok-4.6", "pi", 56.28, 13.34)]

        winners, records = combined._select_best_harness(points)

        self.assertEqual([w.harness for w in winners], ["pi"])
        self.assertIn("single-harness", records[0]["verdict"])
        self.assertEqual(records[0]["runners_up"], [])

    def test_records_the_runner_up_so_nothing_is_hidden(self) -> None:
        points = [
            _point("claude-opus-5", "claude-code", 70.76, 24.05),
            _point("claude-opus-5", "pi", 75.72, 8.28),
        ]

        _, records = combined._select_best_harness(points)

        runners_up = records[0]["runners_up"]
        self.assertEqual(len(runners_up), 1)
        self.assertEqual(runners_up[0]["harness"], "claude-code")

    def test_one_point_per_model_across_many_models(self) -> None:
        points = [
            _point("a", "claude-code", 50.0, 2.0),
            _point("a", "pi", 60.0, 1.0),
            _point("b", "claude-code", 40.0, 1.0),
            _point("b", "pi", 30.0, 3.0),
            _point("c", "pi", 20.0, 0.5),
        ]

        winners, _ = combined._select_best_harness(points)

        self.assertEqual(len(winners), 3)
        self.assertEqual(len({w.model for w in winners}), 3)
        # Sorted highest score first, matching the per-harness charts.
        self.assertEqual([w.model for w in winners], ["a", "b", "c"])


class CostPerPointTest(unittest.TestCase):
    def test_divides_cost_by_score(self) -> None:
        self.assertAlmostEqual(
            combined._cost_per_point(_point("m", "pi", 50.0, 10.0)), 0.2
        )

    def test_unscoreable_point_sorts_last(self) -> None:
        self.assertEqual(
            combined._cost_per_point(_point("m", "pi", 0.0, 10.0)), float("inf")
        )


class DominatesTest(unittest.TestCase):
    def test_equal_points_do_not_dominate_each_other(self) -> None:
        a = _point("m", "pi", 50.0, 10.0)
        b = _point("m", "claude-code", 50.0, 10.0)

        self.assertFalse(combined._dominates(a, b))
        self.assertFalse(combined._dominates(b, a))

    def test_better_on_one_axis_and_equal_on_the_other_dominates(self) -> None:
        better = _point("m", "pi", 50.0, 9.0)
        worse = _point("m", "claude-code", 50.0, 10.0)

        self.assertTrue(combined._dominates(better, worse))


if __name__ == "__main__":
    unittest.main()
