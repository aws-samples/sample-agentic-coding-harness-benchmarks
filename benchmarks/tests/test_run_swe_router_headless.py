"""Tests for the headless swe-router judgment driver.

The parsing and consolidation are where a silent wrong answer could enter: a
mis-read floor or a mis-consolidated repeat becomes a plausible number that the
downstream eval then routes on. The agent invocation itself is not exercised
here (it costs money and needs Bedrock); these cover everything around it.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))


def _load_driver():
    """Import run-swe-router-headless.py by path (its filename carries a dash)."""
    path = _SCRIPTS_DIR / "run-swe-router-headless.py"
    spec = importlib.util.spec_from_file_location("router_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = _load_driver()


class ExtractJudgmentTest(unittest.TestCase):
    def test_reads_a_fenced_block(self) -> None:
        text = 'Some reasoning.\n\n```json\n{"floor": 70, "tier": "low"}\n```'
        got = rr._extract_judgment(text)
        self.assertEqual(got["floor"], 70.0)
        self.assertEqual(got["tier"], "low")

    def test_last_fenced_block_wins(self) -> None:
        # A model that shows its working may emit a draft block first.
        text = (
            '```json\n{"floor": 55, "tier": "trivial"}\n```\n'
            'On reflection:\n```json\n{"floor": 75, "tier": "high"}\n```'
        )
        self.assertEqual(rr._extract_judgment(text)["floor"], 75.0)

    def test_bare_object_without_a_fence_is_still_read(self) -> None:
        text = 'Here is my answer: {"floor": 65, "tier": "medium"}'
        got = rr._extract_judgment(text)
        self.assertEqual(got["floor"], 65.0)
        self.assertEqual(got["tier"], "medium")

    def test_extra_fields_survive(self) -> None:
        text = '```json\n{"floor": 80, "tier": "high", "adjustment": 5}\n```'
        self.assertEqual(rr._extract_judgment(text)["adjustment"], 5)

    def test_an_invented_tier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rr._extract_judgment('```json\n{"floor": 70, "tier": "enormous"}\n```')

    def test_a_floor_off_the_skills_scale_is_rejected(self) -> None:
        # A model answering out of 10 rather than 100 must fail loudly, not be
        # recorded as an absurdly low quality bar.
        with self.assertRaises(ValueError):
            rr._extract_judgment('```json\n{"floor": 7, "tier": "low"}\n```')
        with self.assertRaises(ValueError):
            rr._extract_judgment('```json\n{"floor": 95, "tier": "low"}\n```')

    def test_no_json_at_all_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            rr._extract_judgment("I think the floor should be about seventy.")


class ConsolidateTest(unittest.TestCase):
    def _j(self, floor: float, tier: str) -> dict:
        return {"floor": floor, "tier": tier, "base_floor": floor, "adjustment": 0}

    def test_unanimous_repeats_report_no_spread(self) -> None:
        got = rr._consolidate([self._j(70, "low")] * 3)
        self.assertEqual(got["floor"], 70)
        self.assertEqual(got["tier"], "low")
        self.assertTrue(got["floor_unanimous"])
        self.assertTrue(got["tier_unanimous"])
        self.assertEqual(got["floor_spread"], 0)

    def test_floor_is_the_median_not_the_mean(self) -> None:
        # A mean would invent 71.67, a value the skill's table cannot produce.
        got = rr._consolidate(
            [self._j(70, "low"), self._j(70, "low"), self._j(75, "low")]
        )
        self.assertEqual(got["floor"], 70)
        self.assertFalse(got["floor_unanimous"])
        self.assertEqual(got["floor_spread"], 5)
        self.assertEqual(got["floors_seen"], [70, 70, 75])

    def test_tier_is_the_mode(self) -> None:
        got = rr._consolidate(
            [self._j(70, "medium"), self._j(70, "low"), self._j(70, "medium")]
        )
        self.assertEqual(got["tier"], "medium")
        self.assertFalse(got["tier_unanimous"])
        self.assertEqual(got["tiers_seen"], {"low": 1, "medium": 2})

    def test_a_single_judgment_consolidates_to_itself(self) -> None:
        got = rr._consolidate([self._j(65, "trivial")])
        self.assertEqual((got["floor"], got["tier"]), (65, "trivial"))
        self.assertEqual(got["attempts"], 1)

    def test_no_judgments_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            rr._consolidate([])


class OmpFinalTextTest(unittest.TestCase):
    def _msg(self, role: str, text: str) -> dict:
        return {
            "type": "message_end",
            "message": {"role": role, "content": [{"type": "text", "text": text}]},
        }

    def test_takes_the_last_assistant_message(self) -> None:
        events = [
            self._msg("assistant", "thinking out loud"),
            self._msg("user", "not this"),
            self._msg("assistant", "final answer"),
        ]
        self.assertEqual(rr._omp_final_text(events), "final answer")

    def test_plain_string_content_is_handled(self) -> None:
        events = [
            {"type": "message_end", "message": {"role": "assistant", "content": "hi"}}
        ]
        self.assertEqual(rr._omp_final_text(events), "hi")

    def test_empty_assistant_messages_are_skipped(self) -> None:
        events = [self._msg("assistant", "real"), self._msg("assistant", "   ")]
        self.assertEqual(rr._omp_final_text(events), "real")

    def test_a_stream_with_no_assistant_message_yields_empty(self) -> None:
        self.assertEqual(rr._omp_final_text([{"type": "turn_start"}]), "")


class PromptTest(unittest.TestCase):
    def test_prompt_carries_the_skill_and_forbids_selection(self) -> None:
        class _Task:
            id = "some-task"
            problem_statement = "Do the thing."

        prompt = rr._build_prompt(_Task(), Path("/tmp/clone"))
        self.assertIn("SWE Router", prompt)
        self.assertIn("Do NOT run route.py", prompt)
        self.assertIn("/tmp/clone", prompt)
        self.assertIn("Do the thing.", prompt)
        # The output contract must name every tier the parser accepts.
        for tier in rr.VALID_TIERS:
            self.assertIn(tier, prompt)


class CloneIsolationTest(unittest.TestCase):
    """Repeats of one task must never share a clone directory.

    The harness names its clone parent after the task and wipes it before
    cloning. That is safe for one run per task; with repeats it would let two
    attempts delete each other's checkout mid-run. This asserts the driver hands
    the harness a per-attempt parent so the collision cannot happen.
    """

    def test_each_attempt_gets_its_own_clone_parent(self) -> None:
        seen: list[str] = []

        class _FakeHarness:
            @staticmethod
            def _clone_repo(task, ref, clone_dir, log_prefix=""):
                seen.append(clone_dir)
                raise RuntimeError("stop here; the clone dir is what is under test")

        class _Task:
            id = "same-task"
            problem_statement = "x"
            repo = "https://example.com/r"

        class _Dataset:
            @staticmethod
            def resolved_ref(task):
                return "main"

        class _Config:
            clone_dir = "/tmp/router-test"  # nosec B108 - test fixture path

        real = rr.HARNESS
        rr.HARNESS = _FakeHarness
        try:
            for attempt in (1, 2, 3):
                # A clone failure is deliberately NOT caught by _judge_task -- it
                # is an environment fault, not a judgment that failed -- so the
                # fake's error propagates and the test expects it.
                with self.assertRaises(RuntimeError):
                    rr._judge_task(_Config(), _Dataset(), _Task(), attempt, "l")
        finally:
            rr.HARNESS = real
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(set(seen)), 3, f"clone dirs collided: {seen}")


if __name__ == "__main__":
    unittest.main()
