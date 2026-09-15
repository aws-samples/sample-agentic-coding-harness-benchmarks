"""Tests for the quality-radar top-N capping.

The radar can only show a few legible, colorblind-safe series. When more models
carry eval_scores than the palette allows, it must plot the highest scorers and
report the true total so the caption can say "top N of M".
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_SPEC = importlib.util.spec_from_file_location(
    "plot_quality_radar", _SCRIPTS_DIR / "plot_quality_radar.py"
)
radar = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(radar)


def _write_model(
    root: Path, model: str, harness: str, repo: str, mean: float, skill: str = "swe3"
) -> None:
    d = root / model / harness / skill / repo
    d.mkdir(parents=True)
    # eval_scores must be present for the model to be radar-eligible.
    scores = {
        "github_issue": {
            "total": mean,
            "completeness": 20,
            "correctness": 20,
            "specificity": 20,
            "risk_awareness": 20,
        }
    }
    (d / "run-summary.json").write_text(
        json.dumps(
            {
                "model_slug": model,
                "mean_task_score_excl_failed": mean,
                "tasks": [{"task": "t", "eval_scores": scores}],
            }
        ),
        encoding="utf-8",
    )


class RadarTopNTest(unittest.TestCase):
    def test_caps_to_top_n_by_score_and_reports_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, mean in [("a", 90), ("b", 80), ("c", 70), ("d", 60), ("e", 50)]:
                _write_model(root, name, "claude-code", "repo", mean)
            models, total = radar._collect(root, "repo", "claude-code", "swe3", top_n=3)
            self.assertEqual(total, 5)  # all 5 eligible
            self.assertEqual([m[0] for m in models], ["a", "b", "c"])  # top 3 by score

    def test_no_cap_when_under_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, mean in [("a", 90), ("b", 80)]:
                _write_model(root, name, "claude-code", "repo", mean)
            models, total = radar._collect(root, "repo", "claude-code", "swe3", top_n=4)
            self.assertEqual(total, 2)
            self.assertEqual(len(models), 2)


if __name__ == "__main__":
    unittest.main()
