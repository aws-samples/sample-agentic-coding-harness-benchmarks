"""Tests for the swe-router rubric: policy, mapping, parsing and consolidation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import eval_swe_router as ev  # noqa: E402
import swe_router_rubric as r  # noqa: E402


def _scores(**overrides: int) -> dict[str, int]:
    """All six dimensions at 0, with overrides."""
    base = {d: 0 for d in r.DIMENSIONS}
    base.update(overrides)
    return base


def _judgment(confidence: str = "high", **overrides: int) -> dict:
    """A parsed judgment as extract_judgment returns it."""
    return {
        "scores": _scores(**overrides),
        "evidence": {},
        "confidence": confidence,
        "summary": "s",
    }


def _reply(scores: dict[str, int], confidence: str = "high") -> str:
    """An agent reply carrying one fenced rubric block."""
    body = {
        "scores": {d: {"score": v, "evidence": f"e-{d}"} for d, v in scores.items()},
        "confidence": confidence,
        "summary": "needs a passthrough",
    }
    return f"I read the repo.\n```json\n{json.dumps(body)}\n```"


def _write_policy(data: dict) -> Path:
    """Write a policy dict to a temp file."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


class PolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = r.load_policy()
        self.raw = json.loads(r.DEFAULT_POLICY.read_text(encoding="utf-8"))

    def test_shipped_policy_loads(self) -> None:
        self.assertEqual(self.policy.required_quality.by_blast_radius[1], 70)

    def test_every_reachable_bar_has_a_meaning(self) -> None:
        rq = self.policy.required_quality
        bars = {
            q + bump
            for q in rq.by_blast_radius.values()
            for bump in (0, rq.add_when_verification_is_2)
        }
        self.assertEqual({str(int(b)) for b in bars} - set(rq.meaning), set())

    def test_missing_blast_radius_level_is_rejected(self) -> None:
        self.raw["required_quality"]["by_blast_radius"].pop("2")
        with self.assertRaisesRegex(r.RubricError, "by_blast_radius"):
            r.load_policy(_write_policy(self.raw))

    def test_gap_between_buckets_is_rejected(self) -> None:
        self.raw["difficulty"]["buckets"]["moderate"] = [3, 3]
        with self.assertRaisesRegex(r.RubricError, "must start at"):
            r.load_policy(_write_policy(self.raw))

    def test_buckets_must_cover_the_full_sum(self) -> None:
        self.raw["difficulty"]["buckets"]["very_hard"] = [6, 7]
        with self.assertRaisesRegex(r.RubricError, "cover"):
            r.load_policy(_write_policy(self.raw))

    def test_missing_policy_file_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(r.RubricError, "no routing policy"):
            r.load_policy(Path("/nonexistent/routing-policy.json"))


class MappingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = r.load_policy()

    def test_blast_radius_sets_the_bar(self) -> None:
        self.assertEqual(r.required_quality(_scores(blast_radius=2), self.policy), 75)

    def test_unverifiable_work_raises_the_bar(self) -> None:
        scores = _scores(blast_radius=1, verification=2)
        self.assertEqual(r.required_quality(scores, self.policy), 75)

    def test_difficulty_dimensions_never_move_the_bar(self) -> None:
        hard = _scores(hidden_contract=2, silent_invariant=2, ambiguity=2)
        self.assertEqual(r.required_quality(hard, self.policy), 60)

    def test_bar_dimensions_never_move_the_bucket(self) -> None:
        scores = _scores(blast_radius=2, verification=2)
        self.assertEqual(r.bucket(scores, self.policy), "easy")

    def test_bucket_edges(self) -> None:
        cases = {1: "easy", 2: "moderate", 5: "hard", 6: "very_hard"}
        got = {}
        for total in cases:
            dims = dict.fromkeys(self.policy.difficulty.dimensions, 0)
            for d in list(dims)[: total // 2]:
                dims[d] = 2
            if total % 2:
                dims[list(dims)[total // 2]] = 1
            got[total] = r.bucket(_scores(**dims), self.policy)
        self.assertEqual(got, cases)


class ExtractTest(unittest.TestCase):
    def test_reads_a_fenced_block(self) -> None:
        out = r.extract_judgment(_reply(_scores(blast_radius=2)))
        self.assertEqual(
            (out["scores"]["blast_radius"], out["evidence"]["ambiguity"]),
            (2, "e-ambiguity"),
        )

    def test_takes_the_last_valid_block(self) -> None:
        text = _reply(_scores(ambiguity=1)) + "\n" + _reply(_scores(ambiguity=2))
        self.assertEqual(r.extract_judgment(text)["scores"]["ambiguity"], 2)

    def test_reads_a_bare_object_without_a_fence(self) -> None:
        body = _reply(_scores(verification=1)).split("```json\n")[1].split("\n```")[0]
        self.assertEqual(
            r.extract_judgment("Answer: " + body)["scores"]["verification"], 1
        )

    def test_rejects_a_score_off_the_scale(self) -> None:
        with self.assertRaisesRegex(r.RubricError, "0, 1 or 2"):
            r.extract_judgment(_reply(_scores(ambiguity=3)))

    def test_rejects_a_boolean_score(self) -> None:
        with self.assertRaisesRegex(r.RubricError, "0, 1 or 2"):
            r.extract_judgment(_reply({**_scores(), "ambiguity": True}))

    def test_rejects_a_missing_dimension(self) -> None:
        scores = _scores()
        scores.pop("verification")
        with self.assertRaisesRegex(r.RubricError, "exactly"):
            r.extract_judgment(_reply(scores))

    def test_rejects_an_unknown_confidence(self) -> None:
        with self.assertRaisesRegex(r.RubricError, "confidence"):
            r.extract_judgment(_reply(_scores(), confidence="certain"))


class ConsolidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = r.load_policy()

    def test_lower_median_keeps_scores_on_the_scale(self) -> None:
        runs = [_judgment(ambiguity=1), _judgment(ambiguity=2)]
        out = r.consolidate(runs, self.policy)
        self.assertEqual(out["scores"]["ambiguity"], 1)

    def test_disagreeing_repeats_are_flagged(self) -> None:
        runs = [_judgment(blast_radius=1), _judgment(blast_radius=2)]
        out = r.consolidate(runs, self.policy)
        self.assertEqual(
            (out["quality_unanimous"], out["quality_seen"]), (False, [70.0, 75.0])
        )

    def test_quality_and_bucket_come_from_the_consolidated_scores(self) -> None:
        runs = [_judgment(blast_radius=2, hidden_contract=2, ambiguity=2)] * 3
        out = r.consolidate(runs, self.policy)
        self.assertEqual((out["required_quality"], out["bucket"]), (75, "hard"))

    def test_no_judgments_is_an_error(self) -> None:
        with self.assertRaisesRegex(r.RubricError, "no successful"):
            r.consolidate([], self.policy)


class PromptTest(unittest.TestCase):
    def test_prompt_carries_rubric_task_and_every_dimension(self) -> None:
        prompt = r.build_prompt("t-1", "do it", Path("/tmp/clone"), "RUBRIC-TEXT")  # nosec B108
        self.assertTrue(
            all(
                s in prompt for s in ("RUBRIC-TEXT", "t-1", "/tmp/clone", *r.DIMENSIONS)
            )  # nosec B108
        )

    def test_prompt_never_names_a_price_or_model(self) -> None:
        # The model describes the task; the policy prices it.
        prompt = r.build_prompt("t", "x", Path("/c"), r.DEFAULT_RUBRIC.read_text())
        self.assertFalse(any(w in prompt for w in ("required_quality", "$", "claude-")))


class ReplayRubricInputsTest(unittest.TestCase):
    def _inputs(self, tasks: dict, mode: str = "rubric") -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({"mode": mode, "judged_by": {"model": "m"}, "tasks": tasks}, tmp)
        tmp.close()
        return Path(tmp.name)

    def _five(self) -> dict:
        sums = {"a": 2, "b": 4, "c": 5, "d": 6, "e": 8}
        return {
            t: {"required_quality": 70, "bucket": "hard", "difficulty_sum": s}
            for t, s in sums.items()
        }

    def test_absolute_uses_the_recorded_bucket_and_sets_floors(self) -> None:
        path = self._inputs(
            {"a": {"required_quality": 75, "bucket": "very_hard", "difficulty_sum": 7}}
        )
        floors, column_for, meta = ev._apply_rubric_inputs(
            path, [{"id": "a"}], "absolute", holdout=True
        )
        self.assertEqual(
            (floors["a"], column_for("a")["a"], meta["bucketing"]),
            (75.0, "high", "absolute"),
        )

    def test_relative_quarters_leave_out_the_routed_task(self) -> None:
        tasks = [{"id": t} for t in "abcde"]
        _, column_for, _ = ev._apply_rubric_inputs(
            self._inputs(self._five()), tasks, "relative", holdout=True
        )
        # Routing e: quarters of a-d (2, 4, 5, 6), so e's 8 is very hard and a's 2 easy.
        cols = column_for("e")
        self.assertEqual((cols["a"], cols["e"]), ("trivial", "high"))

    def test_relative_columns_differ_by_which_task_is_held_out(self) -> None:
        tasks = [{"id": t} for t in "abcde"]
        _, column_for, _ = ev._apply_rubric_inputs(
            self._inputs(self._five()), tasks, "relative", holdout=True
        )
        self.assertNotEqual(column_for("a"), column_for("e"))

    def test_regrouping_does_not_touch_the_original_results(self) -> None:
        results = {"m1": {"a": {"complexity": "trivial", "score": 80}}}
        grouped = ev._regrouped(results, {"a": "high"})
        self.assertEqual(
            (grouped["m1"]["a"]["complexity"], results["m1"]["a"]["complexity"]),
            ("high", "trivial"),
        )

    def test_rejects_a_floor_mode_file(self) -> None:
        with self.assertRaisesRegex(SystemExit, "not a rubric pass"):
            ev._apply_rubric_inputs(
                self._inputs({}, mode="floor"), [], "absolute", False
            )

    def test_rejects_a_file_missing_a_task(self) -> None:
        with self.assertRaisesRegex(SystemExit, "missing 1 task"):
            ev._apply_rubric_inputs(self._inputs({}), [{"id": "a"}], "absolute", False)


class ZeroShotColumnsTest(unittest.TestCase):
    def _columns(self, rubric_tasks: dict) -> dict:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({"mode": "rubric", "tasks": rubric_tasks}, tmp)
        tmp.close()
        rec = lambda s: {  # noqa: E731
            "score": s,
            "cost_usd": 1.0,
            "failed": False,
            "complexity": "low",
        }
        results = {"m1": {"a": rec(90.0), "b": rec(60.0)}}
        tasks = [{"id": "a", "complexity": "low"}, {"id": "b", "complexity": "low"}]
        with (
            mock.patch.object(ev, "_load_tasks", return_value=(tasks, "other")),
            mock.patch.object(ev, "_load_results", return_value=results),
        ):
            stats, _ = ev._columns_from_other_dataset(
                Path("other.yaml"), Path(tmp.name), "omp", "swe3"
            )
        return stats["m1"]["score_by_complexity"]

    def test_columns_are_the_other_datasets_bucket_means(self) -> None:
        cols = self._columns(
            {
                "a": {"required_quality": 60, "bucket": "easy", "difficulty_sum": 0},
                "b": {
                    "required_quality": 70,
                    "bucket": "very_hard",
                    "difficulty_sum": 7,
                },
            }
        )
        self.assertEqual(cols, {"trivial": 90.0, "high": 60.0})

    def test_rubric_entries_outside_that_dataset_are_ignored(self) -> None:
        cols = self._columns(
            {
                "a": {"required_quality": 60, "bucket": "easy", "difficulty_sum": 0},
                "b": {"required_quality": 70, "bucket": "easy", "difficulty_sum": 1},
                "x": {"required_quality": 70, "bucket": "hard", "difficulty_sum": 4},
            }
        )
        self.assertEqual(cols, {"trivial": 75.0})


class RegressionColumnTest(unittest.TestCase):
    def _results(self) -> dict:
        # Model "steep" loses 10 points per difficulty step; "flat" does not.
        sums = {"a": 0, "b": 2, "c": 4, "d": 6, "e": 8}
        rec = lambda s, failed=False: {  # noqa: E731
            "score": None if failed else s,
            "cost_usd": 1.0,
            "failed": failed,
            "complexity": "low",
        }
        results = {
            "steep": {t: rec(90 - 5 * x) for t, x in sums.items()},
            "flat": {t: rec(80.0) for t in sums},
        }
        return results, sums

    def test_fit_recovers_a_clean_line(self) -> None:
        a, b = ev._fit_line([(0, 90), (2, 80), (4, 70)])
        self.assertEqual((round(a, 6), round(b, 6)), (90, -5))

    def test_too_few_points_fall_back_to_the_mean(self) -> None:
        self.assertEqual(ev._fit_line([(1, 60), (3, 80)]), (70, 0.0))

    def test_no_spread_in_difficulty_falls_back_to_the_mean(self) -> None:
        self.assertEqual(ev._fit_line([(4, 60), (4, 70), (4, 80)]), (70, 0.0))

    def test_prediction_falls_with_difficulty_for_a_degrading_model(self) -> None:
        results, sums = self._results()
        easy = ev._predicted_stats(results, sums, "a", holdout=True)
        hard = ev._predicted_stats(results, sums, "d", holdout=True)
        key = ev.PREDICTED_TIER
        self.assertEqual(
            (
                easy["steep"]["score_by_complexity"][key],
                hard["steep"]["score_by_complexity"][key],
            ),
            (90.0, 60.0),
        )

    def test_holdout_leaves_the_routed_task_out_of_the_fit(self) -> None:
        results, sums = self._results()
        results["steep"]["d"]["score"] = 100.0  # an outlier on the routed task
        pred = ev._predicted_stats(results, sums, "d", holdout=True)
        self.assertEqual(pred["steep"]["score_by_complexity"][ev.PREDICTED_TIER], 60.0)

    def test_failed_runs_are_left_out_of_the_fit(self) -> None:
        # Routing a and dropping failed b leaves c, d, e: still a clean line.
        results, sums = self._results()
        results["steep"]["b"] = {
            "score": None,
            "cost_usd": 1.0,
            "failed": True,
            "complexity": "low",
        }
        pred = ev._predicted_stats(results, sums, "a", holdout=True)
        self.assertEqual(pred["steep"]["score_by_complexity"][ev.PREDICTED_TIER], 90.0)


class RelativeBucketTest(unittest.TestCase):
    def test_needs_four_tasks(self) -> None:
        with self.assertRaisesRegex(r.RubricError, "at least 4"):
            r.relative_edges([1, 2, 3])

    def test_a_tie_falls_to_the_lower_bucket(self) -> None:
        edges = r.relative_edges([2, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8])
        self.assertEqual(r.relative_bucket(5, edges), "moderate")

    def test_shipped_policy_is_absolute(self) -> None:
        # Relative quarters sorted the 21 v2 tasks better but split them into
        # columns of 3-12 tasks, too few for steady per-model means; absolute
        # buckets routed better on that data (docs/swe-router-rubric-evaluation.md).
        self.assertEqual(r.load_policy().difficulty.bucketing, "absolute")


if __name__ == "__main__":
    unittest.main()
