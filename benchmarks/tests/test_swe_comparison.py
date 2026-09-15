"""Tests for the cross-harness /swe comparison: the cost-vs-accuracy bubble
chart's point collector and the doc generator's table output.

Rendering (matplotlib) is not unit-tested; the data logic is what must be right
so the chart, the tables, and the per-harness docs all agree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import unittest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bub = _load("plot_cost_accuracy_bubble")
swe = _load("gen_swe_comparison")


class BubblePointsTest(unittest.TestCase):
    def test_drops_uncostable_and_carries_cost_score_tokens(self) -> None:
        collected = [
            {
                "model": "ok",
                "total_tokens": 2000,
                "provider": "endpoint",
                "mean": 55.0,
                "num_scored": 5,
                "num_tasks": 5,
            },
            {
                "model": "nocost",
                "total_tokens": 1000,
                "provider": "endpoint",
                "mean": 40.0,
                "num_scored": 5,
                "num_tasks": 5,
            },
            {
                "model": "notokens",
                "total_tokens": 0,
                "provider": "endpoint",
                "mean": 40.0,
                "num_scored": 5,
                "num_tasks": 5,
            },
        ]
        cost_map = {
            "ok": ("$10.00", "hardware-derived (g6e.12xlarge)"),
            "nocost": ("--", "hardware-derived"),
            "notokens": ("$5.00", "hardware-derived (g6e.12xlarge)"),
        }
        with mock.patch.object(bub.gen, "_collect", return_value=collected):
            with mock.patch.object(
                bub.gen, "_row_cost", side_effect=lambda r: cost_map[r["model"]]
            ):
                pts = bub._collect_points(Path("/x"), "pi", "swe3", "repo")
        # only "ok" survives (nocost has no cost, notokens has no tokens).
        self.assertEqual([p["model"] for p in pts], ["ok"])
        self.assertEqual(pts[0]["cost"], 2.0)  # $10 / 5 scored = $2/task
        self.assertEqual(pts[0]["score"], 55.0)
        self.assertEqual(pts[0]["tokens"], 2000)
        self.assertFalse(pts[0]["bedrock"])

    def test_areas_are_proportional_to_tokens(self) -> None:
        # A 2x-larger token count yields a 2x-larger AREA (linear in tokens).
        areas = bub._areas([1000, 2000, 3000])
        self.assertLess(areas[0], areas[1])
        self.assertLess(areas[1], areas[2])
        # midpoint token count -> midpoint area (linear map).
        self.assertAlmostEqual(areas[1], (areas[0] + areas[2]) / 2, places=6)


class ComparisonDocTest(unittest.TestCase):
    def test_table_has_cost_per_task_and_point(self) -> None:
        rows = [
            {
                "model": "m1",
                "mean": 60.0,
                "completed": "5/5",
                "total_tokens": 2_000_000,
                "cost": 10.0,
                "cost_str": "$10.00",
                "basis": "hardware-derived (p5en.48xlarge)",
                "cost_per_task": 2.0,
                "cost_per_point": 10.0 / 60.0,
                "minutes": 30.0,
                "bedrock": False,
            }
        ]
        out = "\n".join(swe._table(rows, "pi"))
        self.assertIn("### pi", out)
        self.assertIn("self-hosted", out)
        self.assertIn("$2.00", out)  # cost/task
        self.assertIn("2.0M", out)  # tokens humanized
        self.assertIn("30m", out)  # wall-clock

    def test_zero_scored_row_renders_without_error(self) -> None:
        rows = [
            {
                "model": "failer",
                "mean": None,
                "completed": "0/5",
                "total_tokens": 100,
                "cost": None,
                "cost_str": "--",
                "basis": "hardware-derived",
                "cost_per_task": None,
                "cost_per_point": None,
                "minutes": 0.0,
                "bedrock": False,
            }
        ]
        out = "\n".join(swe._table(rows, "pi"))
        self.assertIn("-- (0 scored)", out)


if __name__ == "__main__":
    unittest.main()
