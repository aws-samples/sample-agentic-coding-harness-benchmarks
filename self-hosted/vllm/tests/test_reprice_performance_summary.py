"""Tests for repricing a committed performance-summary.json at a new hourly rate.

The property under test is that repricing is *exact*, not approximate: because
every cost field is linear in $/hr, scaling by ``new/old`` must reproduce what a
rebuild from the same DuckDB would produce. So the tests check the algebra
against a recomputed-from-throughput expectation rather than against a golden
number, and check that nothing measured moves.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from clients import pricing
from clients.reprice_performance_summary import (
    RepriceError,
    _infer_tp,
    reprice,
    reprice_file,
)

_OLD_FULL = 41.1424


def _level(concurrency: int, out_tps: float, rate: float) -> dict:
    """One sweep level priced at ``rate`` $/hr, exactly as the builder derives it."""
    per_token = (rate / 3600.0) / out_tps
    return {
        "concurrency": concurrency,
        "output_tokens_per_second": out_tps,
        "ttft_ms_mean": 900.0 + concurrency,
        "queue_ms_mean": 12.0 * concurrency,
        "kv_cache_pct_mean": 4.0 * concurrency,
        "blended_cost_per_token_usd": per_token,
        "split_cost_per_input_token_usd": per_token * 0.1,
        "split_cost_per_output_token_usd": per_token * 1.9,
        "blended_cost_per_1m_tokens_usd": round(per_token * 1e6, 4),
        "split_cost_per_1m_input_tokens_usd": round(per_token * 0.1 * 1e6, 4),
        "split_cost_per_1m_output_tokens_usd": round(per_token * 1.9 * 1e6, 4),
        "task_cost_blended_usd": round(per_token * 100_000, 4),
        "task_cost_split_usd": round(per_token * 1.4 * 100_000, 4),
        "split_input_weight": 0.1,
    }


def _summary(rate: float = _OLD_FULL, instance: str = "p5en.48xlarge") -> dict:
    """A minimal but realistically shaped summary priced at ``rate``."""
    levels = [_level(1, 40.0, rate), _level(10, 300.0, rate), _level(20, 250.0, rate)]
    cheapest = min(lvl["blended_cost_per_token_usd"] for lvl in levels)
    return {
        "model": "test-model",
        "instance_type": instance,
        "dollars_per_hour": rate,
        "output_tokens_per_task": 100_000,
        "peak_output_tokens_per_second": 300.0,
        "min_blended_cost_per_1m_tokens_usd": round(cheapest * 1e6, 2),
        "min_task_cost_blended_usd": round(cheapest * 100_000, 4),
        "min_task_cost_split_usd": round(cheapest * 1.4 * 100_000, 4),
        "levels": levels,
    }


class TestInferTp(unittest.TestCase):
    """TP must be recoverable from a recorded partial-box rate, or rejected."""

    def test_full_box(self) -> None:
        self.assertEqual(_infer_tp(_OLD_FULL, 41.1424, 8), 8)

    def test_half_box(self) -> None:
        self.assertEqual(_infer_tp(_OLD_FULL, 20.5712, 8), 4)

    def test_single_gpu(self) -> None:
        self.assertEqual(_infer_tp(_OLD_FULL, 5.1428, 8), 1)

    def test_fractional_share_is_rejected_not_rounded(self) -> None:
        """A rate that is not a whole-GPU share means the basis is wrong."""
        with self.assertRaises(RepriceError) as ctx:
            _infer_tp(_OLD_FULL, 7.0, 8)
        self.assertIn("whole-GPU share", str(ctx.exception))

    def test_rate_above_full_box_is_rejected(self) -> None:
        with self.assertRaises(RepriceError):
            _infer_tp(_OLD_FULL, 99.0, 8)

    def test_nonpositive_old_rate_is_rejected(self) -> None:
        with self.assertRaises(RepriceError):
            _infer_tp(0.0, 5.0, 8)


class TestReprice(unittest.TestCase):
    """Repricing scales every cost exactly and leaves measurements alone."""

    def setUp(self) -> None:
        self.old = _summary()
        self.new_rate = 27.72
        self.factor = self.new_rate / _OLD_FULL
        self.new = reprice(self.old, self.new_rate, self.factor)

    def test_rate_recorded(self) -> None:
        self.assertAlmostEqual(self.new["dollars_per_hour"], 27.72, places=4)

    def test_input_not_mutated(self) -> None:
        """The caller's dict (and its levels) must survive untouched."""
        self.assertAlmostEqual(self.old["dollars_per_hour"], _OLD_FULL, places=4)
        self.assertAlmostEqual(
            self.old["levels"][1]["blended_cost_per_token_usd"],
            (_OLD_FULL / 3600.0) / 300.0,
        )

    def test_costs_match_a_rebuild_at_the_new_rate(self) -> None:
        """The real assertion: scaling == recomputing from throughput.

        This is what makes repricing legitimate rather than a fudge -- the result
        is identical to what build_performance_summary would emit from the same
        DuckDB with --dollars-per-hour set to the new rate.
        """
        for level in self.new["levels"]:
            expected = (self.new_rate / 3600.0) / level["output_tokens_per_second"]
            self.assertAlmostEqual(
                level["blended_cost_per_token_usd"], expected, places=12
            )
            self.assertAlmostEqual(
                level["blended_cost_per_1m_tokens_usd"],
                round(expected * 1e6, 4),
                places=3,
            )

    def test_measurements_are_untouched(self) -> None:
        for old_level, new_level in zip(self.old["levels"], self.new["levels"]):
            for key in (
                "concurrency",
                "output_tokens_per_second",
                "ttft_ms_mean",
                "queue_ms_mean",
                "kv_cache_pct_mean",
            ):
                self.assertEqual(old_level[key], new_level[key], msg=key)
        self.assertEqual(self.new["peak_output_tokens_per_second"], 300.0)
        self.assertEqual(self.new["output_tokens_per_task"], 100_000)

    def test_split_input_weight_is_not_scaled(self) -> None:
        """The weight is a ratio, not a price; scaling it would corrupt the split."""
        for level in self.new["levels"]:
            self.assertEqual(level["split_input_weight"], 0.1)

    def test_min_fields_agree_exactly_with_the_level_they_came_from(self) -> None:
        """The headline must equal a level's own figure, bit for bit.

        Scaling the stored (already-rounded) min instead of recomputing it from
        the scaled levels leaves the headline a cent off the level it claims to
        report -- the exact bug this test caught.
        """
        cheapest = min(
            self.new["levels"], key=lambda lvl: lvl["blended_cost_per_token_usd"]
        )
        self.assertEqual(
            self.new["min_blended_cost_per_1m_tokens_usd"],
            cheapest["blended_cost_per_1m_tokens_usd"],
        )
        self.assertEqual(
            self.new["min_task_cost_blended_usd"],
            min(lvl["task_cost_blended_usd"] for lvl in self.new["levels"]),
        )
        self.assertEqual(
            self.new["min_task_cost_split_usd"],
            min(lvl["task_cost_split_usd"] for lvl in self.new["levels"]),
        )

    def test_cheapest_level_is_preserved(self) -> None:
        """A positive scale factor cannot reorder levels, so the knee cannot move."""
        self.assertEqual(
            min(
                self.new["levels"],
                key=lambda level: level["blended_cost_per_token_usd"],
            )["concurrency"],
            10,
        )

    def test_provenance_recorded(self) -> None:
        meta = self.new["_repriced"]
        self.assertAlmostEqual(meta["from_dollars_per_hour"], _OLD_FULL, places=4)
        self.assertAlmostEqual(meta["to_dollars_per_hour"], 27.72, places=4)
        self.assertAlmostEqual(meta["factor"], self.factor, places=5)

    def test_round_trip_returns_to_the_original(self) -> None:
        """Scaling down then back up must recover the original costs.

        Exactness matters: if repricing lost information, a fleet repriced twice
        would drift away from its measurements.
        """
        back = reprice(self.new, _OLD_FULL, _OLD_FULL / self.new_rate)
        for orig, rt in zip(self.old["levels"], back["levels"]):
            self.assertAlmostEqual(
                orig["blended_cost_per_token_usd"],
                rt["blended_cost_per_token_usd"],
                places=12,
            )

    def test_unknown_cost_like_field_is_left_alone(self) -> None:
        """A field the tool does not know about must not be silently scaled."""
        summary = _summary()
        summary["levels"][0]["some_future_cost_usd"] = 1.0
        out = reprice(summary, 27.72, 27.72 / _OLD_FULL)
        self.assertEqual(out["levels"][0]["some_future_cost_usd"], 1.0)

    def test_empty_levels_is_tolerated(self) -> None:
        out = reprice({"dollars_per_hour": _OLD_FULL}, 27.72, 27.72 / _OLD_FULL)
        self.assertEqual(out["levels"], [])


class TestRepriceFile(unittest.TestCase):
    """End-to-end: a file on disk is repriced against the live pricing.json."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "performance-summary.json"
        self.addCleanup(self._tmp.cleanup)

    def _write(self, summary: dict) -> None:
        self.path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def test_full_box_repriced_to_pricing_json(self) -> None:
        self._write(_summary())
        changed = reprice_file(self.path, _OLD_FULL, "p5en.48xlarge", None, False)
        self.assertTrue(changed)
        out = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            out["dollars_per_hour"], pricing.resolve("p5en.48xlarge", 8), places=4
        )

    def test_half_box_keeps_its_proration(self) -> None:
        """A TP=4 arm must land on half the new full rate, not the whole box."""
        self._write(_summary(rate=_OLD_FULL / 2))
        reprice_file(self.path, _OLD_FULL, "p5en.48xlarge", None, False)
        out = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            out["dollars_per_hour"], pricing.resolve("p5en.48xlarge", 4), places=4
        )

    def test_dry_run_writes_nothing(self) -> None:
        self._write(_summary())
        before = self.path.read_text(encoding="utf-8")
        changed = reprice_file(self.path, _OLD_FULL, "p5en.48xlarge", None, True)
        self.assertTrue(changed)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_instance_filter_skips_other_instances(self) -> None:
        """A g6e arm must not be touched by a p5en reprice."""
        self._write(_summary(rate=4.533, instance="g6e.12xlarge"))
        before = self.path.read_text(encoding="utf-8")
        changed = reprice_file(self.path, _OLD_FULL, "p5en.48xlarge", None, False)
        self.assertFalse(changed)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_already_at_target_rate_is_idempotent(self) -> None:
        """Re-running the tool must not scale an already-repriced file again."""
        self._write(_summary())
        reprice_file(self.path, _OLD_FULL, "p5en.48xlarge", None, False)
        first = json.loads(self.path.read_text(encoding="utf-8"))
        changed = reprice_file(self.path, _OLD_FULL, "p5en.48xlarge", None, False)
        self.assertFalse(changed)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), first)

    def test_tp_override_wins_over_inference(self) -> None:
        self._write(_summary())
        reprice_file(self.path, _OLD_FULL, "p5en.48xlarge", 4, False)
        out = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            out["dollars_per_hour"], pricing.resolve("p5en.48xlarge", 4), places=4
        )

    def test_missing_rate_is_skipped_not_crashed(self) -> None:
        self._write({"instance_type": "p5en.48xlarge", "levels": []})
        self.assertFalse(
            reprice_file(self.path, _OLD_FULL, "p5en.48xlarge", None, False)
        )


class TestPricingHasNoDiscount(unittest.TestCase):
    """The placeholder-discount concept must stay gone, not merely be unused."""

    def test_no_entry_carries_a_discount_key(self) -> None:
        for name, entry in pricing._load()["instances"].items():
            self.assertNotIn("discount", entry, msg=name)

    def test_a_leftover_discount_key_fails_closed(self) -> None:
        """Ignoring it would let an assumption silently reprice the whole repo."""
        with self.assertRaises(pricing.PricingError) as ctx:
            pricing._effective_full({"dollars_per_hour": 10.0, "discount": 0.35})
        self.assertIn("no longer", str(ctx.exception))

    def test_resolve_returns_the_published_rate_verbatim(self) -> None:
        self.assertAlmostEqual(pricing.resolve("p5en.48xlarge", 8), 27.72, places=4)
        self.assertAlmostEqual(pricing.resolve("p5en.48xlarge", 4), 13.86, places=4)
        self.assertAlmostEqual(pricing.resolve("g6e.12xlarge", 4), 4.533, places=4)


if __name__ == "__main__":
    unittest.main()
