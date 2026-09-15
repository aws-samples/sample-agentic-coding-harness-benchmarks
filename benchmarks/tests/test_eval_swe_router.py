"""Tests for the swe-router evaluation harness.

The joins this script makes are the whole point of it -- a mistake in the cost
basis or the failed-task convention would show up as a plausible number rather
than an error -- so these cover the arithmetic, the exclusion rules, and the
leave-one-out rebuild rather than the report's prose.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import eval_swe_router as ev  # noqa: E402


def _results() -> dict[str, dict[str, dict]]:
    """Two models over three tasks; the cheap one failed the hard task."""
    return {
        "strong": {
            "a": {
                "score": 90.0,
                "cost_usd": 10.0,
                "failed": False,
                "complexity": "low",
                "cost_basis": "metered",
            },
            "b": {
                "score": 80.0,
                "cost_usd": 20.0,
                "failed": False,
                "complexity": "high",
                "cost_basis": "metered",
            },
            "c": {
                "score": 70.0,
                "cost_usd": 30.0,
                "failed": False,
                "complexity": "high",
                "cost_basis": "metered",
            },
        },
        "cheap": {
            "a": {
                "score": 75.0,
                "cost_usd": 1.0,
                "failed": False,
                "complexity": "low",
                "cost_basis": "hardware-derived",
            },
            "b": {
                "score": 60.0,
                "cost_usd": 2.0,
                "failed": False,
                "complexity": "high",
                "cost_basis": "hardware-derived",
            },
            "c": {
                "score": None,
                "cost_usd": 3.0,
                "failed": True,
                "complexity": "high",
                "cost_basis": "hardware-derived",
            },
        },
    }


class TaskCostTest(unittest.TestCase):
    def test_metered_uses_the_provider_bill(self) -> None:
        task = {"total_cost_usd": 4.25, "input_tokens": 10, "output_tokens": 10}
        self.assertEqual(ev._task_cost(task, None), 4.25)

    def test_metered_missing_bill_is_zero_not_an_error(self) -> None:
        self.assertEqual(ev._task_cost({"task": "x"}, None), 0.0)

    def test_hardware_derived_prices_processed_tokens(self) -> None:
        # Cache fields are additive here (their sum is nowhere near input), so
        # every field counts once: 100 + 50 + 20 + 10 = 180 tokens.
        task = {
            "task": "x",
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 20,
            "cache_write_tokens": 10,
        }
        self.assertAlmostEqual(ev._task_cost(task, 0.5), 90.0)

    def test_hardware_derived_does_not_double_count_a_partitioned_cache(self) -> None:
        # cache_read + cache_write == input_tokens, so input already contains
        # them: the total is input + output, not input + output + cache.
        task = {
            "task": "x",
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 90,
            "cache_write_tokens": 10,
        }
        self.assertAlmostEqual(ev._task_cost(task, 1.0), 150.0)


class TierStatsTest(unittest.TestCase):
    def test_failed_task_is_excluded_from_both_means(self) -> None:
        stats = ev._tier_stats(_results())
        # cheap scored only a (75) and b (60); c failed and is left out.
        self.assertEqual(stats["cheap"]["score"], 67.5)
        self.assertEqual(stats["cheap"]["cost_per_task_usd"], 1.5)

    def test_failed_task_still_counts_against_completion(self) -> None:
        stats = ev._tier_stats(_results())
        self.assertEqual(stats["cheap"]["completion_by_complexity"]["high"], "1/2")
        self.assertEqual(stats["strong"]["completion_by_complexity"]["high"], "2/2")

    def test_tier_means_are_per_tier(self) -> None:
        stats = ev._tier_stats(_results())
        self.assertEqual(stats["strong"]["score_by_complexity"]["low"], 90.0)
        self.assertEqual(stats["strong"]["score_by_complexity"]["high"], 75.0)

    def test_holdout_removes_the_task_entirely(self) -> None:
        stats = ev._tier_stats(_results(), exclude_task="b")
        # strong's high tier now holds only c (70), not the mean of b and c.
        self.assertEqual(stats["strong"]["score_by_complexity"]["high"], 70.0)
        self.assertEqual(stats["strong"]["tasks_total"], 2)


class RowTest(unittest.TestCase):
    def _routed(self, model: str, score: float) -> dict:
        return {
            "status": "ok",
            "recommended": {"model": model, "score": score, "cost_per_task_usd": 1.5},
        }

    def test_switch_records_both_arms_and_the_saving(self) -> None:
        row = ev._row(
            "a", "low", 70.0, self._routed("cheap", 67.5), _results(), "strong"
        )
        self.assertTrue(row["switched"])
        self.assertEqual(row["actual_score"], 75.0)
        self.assertEqual(row["baseline_score"], 90.0)
        self.assertEqual(row["score_delta"], -15.0)
        self.assertEqual(row["cost_delta_usd"], -9.0)
        self.assertEqual(row["cost_saving_pct"], 90.0)

    def test_predicted_and_actual_are_kept_apart(self) -> None:
        # The tier mean the router selected on is not the task's own score; the
        # gap between them is the router's prediction error and must survive.
        row = ev._row(
            "a", "low", 70.0, self._routed("cheap", 67.5), _results(), "strong"
        )
        self.assertEqual(row["predicted_score"], 67.5)
        self.assertEqual(row["actual_score"], 75.0)

    def test_pick_below_the_floor_is_flagged(self) -> None:
        row = ev._row(
            "b", "high", 70.0, self._routed("cheap", 60.0), _results(), "strong"
        )
        self.assertFalse(row["met_floor"])

    def test_picking_the_baseline_is_not_a_switch(self) -> None:
        row = ev._row(
            "a", "low", 70.0, self._routed("strong", 90.0), _results(), "strong"
        )
        self.assertFalse(row["switched"])
        self.assertEqual(row["cost_delta_usd"], 0.0)

    def test_no_recommendation_falls_back_to_the_baseline_run(self) -> None:
        routed = {"status": "nothing_clears_floor", "recommended": None, "reason": "r"}
        row = ev._row("a", "low", 95.0, routed, _results(), "strong")
        self.assertIsNone(row["recommended_model"])
        self.assertFalse(row["switched"])
        self.assertEqual(row["actual_cost_usd"], 10.0)
        self.assertEqual(row["cost_saving_pct"], 0.0)

    def test_a_failed_pick_leaves_the_score_delta_undefined(self) -> None:
        row = ev._row(
            "c", "high", 70.0, self._routed("cheap", 60.0), _results(), "strong"
        )
        self.assertTrue(row["actual_failed"])
        self.assertIsNone(row["score_delta"])
        # It still cost money, so the cost is real.
        self.assertEqual(row["actual_cost_usd"], 3.0)

    def test_picking_a_model_with_no_run_is_an_error_not_a_gap(self) -> None:
        with self.assertRaises(SystemExit):
            ev._row(
                "a", "low", 70.0, self._routed("absent", 80.0), _results(), "strong"
            )


class TotalsTest(unittest.TestCase):
    def test_costs_sum_over_every_task_including_failures(self) -> None:
        rows = [
            ev._row(
                "a",
                "low",
                70.0,
                {
                    "status": "ok",
                    "recommended": {
                        "model": "cheap",
                        "score": 1,
                        "cost_per_task_usd": 1,
                    },
                },
                _results(),
                "strong",
            ),
            ev._row(
                "c",
                "high",
                70.0,
                {
                    "status": "ok",
                    "recommended": {
                        "model": "cheap",
                        "score": 1,
                        "cost_per_task_usd": 1,
                    },
                },
                _results(),
                "strong",
            ),
        ]
        totals = ev._totals(rows, "strong")
        self.assertEqual(totals["routed_total_cost_usd"], 4.0)
        self.assertEqual(totals["baseline_total_cost_usd"], 40.0)
        self.assertEqual(totals["cost_saving_usd"], 36.0)
        self.assertEqual(totals["cost_saving_pct"], 90.0)

    def test_score_means_cover_only_tasks_scored_in_both_arms(self) -> None:
        rows = [
            ev._row(
                "a",
                "low",
                70.0,
                {
                    "status": "ok",
                    "recommended": {
                        "model": "cheap",
                        "score": 1,
                        "cost_per_task_usd": 1,
                    },
                },
                _results(),
                "strong",
            ),
            ev._row(
                "c",
                "high",
                70.0,
                {
                    "status": "ok",
                    "recommended": {
                        "model": "cheap",
                        "score": 1,
                        "cost_per_task_usd": 1,
                    },
                },
                _results(),
                "strong",
            ),
        ]
        totals = ev._totals(rows, "strong")
        # Task c has no routed score, so neither arm's mean may include it.
        self.assertEqual(totals["tasks_scored_both_arms"], 1)
        self.assertEqual(totals["routed_mean_score"], 75.0)
        self.assertEqual(totals["baseline_mean_score"], 90.0)
        self.assertEqual(totals["mean_score_delta"], -15.0)

    def test_both_arms_are_held_to_the_same_floor(self) -> None:
        rows = [
            ev._row(
                "b",
                "high",
                85.0,
                {
                    "status": "ok",
                    "recommended": {
                        "model": "cheap",
                        "score": 1,
                        "cost_per_task_usd": 1,
                    },
                },
                _results(),
                "strong",
            ),
        ]
        totals = ev._totals(rows, "strong")
        self.assertEqual(totals["tasks_below_floor_routed"], 1)
        self.assertEqual(totals["tasks_below_floor_baseline"], 1)


class CommittedDataTest(unittest.TestCase):
    """The join must reproduce models.json, or the report is measuring something else.

    ``models.json`` was generated from these same run summaries by a different
    code path. If this script's loader, cost basis or exclusion rule ever drifts
    from that one, the two stop agreeing -- and a per-task lookup against a
    ranking built on other numbers is silently meaningless. Cheap to assert,
    impossible to notice otherwise.
    """

    def setUp(self) -> None:
        self.published = ev._read_json(ev._SKILL_DIR / "models.json")
        if self.published is None:
            self.skipTest("swe-router skill is not installed here")
        self.results = ev._load_results("omp", "swe3", "mcp-gateway-registry-v2")
        if not self.results:
            self.skipTest("no committed omp/swe3 run summaries")

    def test_overall_scores_and_tier_means_match_models_json(self) -> None:
        stats = ev._tier_stats(self.results)
        for model in self.published["models"]:
            slug = model["model"]
            with self.subTest(model=slug):
                self.assertIn(slug, stats)
                self.assertAlmostEqual(stats[slug]["score"], model["score"], places=2)
                for tier, mean in model["score_by_complexity"].items():
                    self.assertAlmostEqual(
                        stats[slug]["score_by_complexity"][tier], mean, places=2
                    )

    def test_cost_per_task_matches_models_json(self) -> None:
        stats = ev._tier_stats(self.results)
        for model in self.published["models"]:
            slug = model["model"]
            with self.subTest(model=slug):
                # models.json rounds to 4dp from a slightly different pipeline;
                # a cent of drift is rounding, a dollar is a bug.
                self.assertAlmostEqual(
                    stats[slug]["cost_per_task_usd"],
                    model["cost_per_task_usd"],
                    delta=0.01,
                )


if __name__ == "__main__":
    unittest.main()
