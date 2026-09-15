"""Tests for the vended swe-router selection logic."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_VEND = _REPO_ROOT / "vend" / "swe-router"

_spec = importlib.util.spec_from_file_location("route", _VEND / "route.py")
route = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(route)

MODELS = _VEND / "models.json"
ALIASES = _VEND / "model-aliases.json"


def _route(**kw):
    """Call route() with the vended data and sensible defaults."""
    args = dict(
        models_path=MODELS,
        aliases_path=ALIASES,
        allowed_file=None,
        no_allow_list=True,
        tie_band=route.DEFAULT_TIE_BAND,
        available=None,
    )
    args.update(kw)
    return route.route(**args)


class NormalizeTest(unittest.TestCase):
    def test_strips_routing_and_packaging_from_names(self) -> None:
        # Every one of these is claude-sonnet-5 wearing a different hat.
        for name in (
            "claude-sonnet-5",
            "Claude Sonnet 5",
            "us.anthropic.claude-sonnet-5",
            "anthropic/claude-sonnet-5",
            "us.anthropic.claude-sonnet-5[1m]",
            "CLAUDE_SONNET_5",
        ):
            self.assertEqual(route.normalize(name), "claudesonnet5", name)

    def test_keeps_distinct_models_distinct(self) -> None:
        self.assertNotEqual(
            route.normalize("claude-opus-4-5"), route.normalize("claude-opus-4-8")
        )


class SelectionTest(unittest.TestCase):
    def test_picks_the_cheapest_model_clearing_the_floor(self) -> None:
        r = _route(tier="high", floor=70)
        self.assertEqual(r["status"], "ok")
        cleared = r["cleared_floor"]
        self.assertEqual(
            r["recommended"]["cost_per_task_usd"],
            min(c["cost_per_task_usd"] for c in cleared),
        )

    def test_the_pick_is_never_dominated(self) -> None:
        # Cheaper and at least as good would mean the wrong model was chosen.
        for tier in route.TIERS:
            for floor in (55, 60, 65, 70, 75):
                r = _route(tier=tier, floor=floor)
                p = r.get("recommended")
                if not p:
                    continue
                for other in r["cleared_floor"]:
                    if other["model"] == p["model"]:
                        continue
                    self.assertFalse(
                        other["cost_per_task_usd"] < p["cost_per_task_usd"]
                        and other["score"] >= p["score"],
                        f"{tier}/{floor}: {other['model']} dominates {p['model']}",
                    )

    def test_reads_the_tier_not_the_overall_mean(self) -> None:
        # qwen3.8-27b averages 78.48 overall and 71.45 on high. A floor of 75
        # must reject it for hard work.
        r = _route(tier="high", floor=75)
        self.assertNotEqual(r["recommended"]["model"], "qwen3.8-27b")
        r = _route(tier="low", floor=75)
        self.assertEqual(r["recommended"]["model"], "qwen3.8-27b")

    def test_flags_a_margin_inside_the_noise(self) -> None:
        # 71.45 against a floor of 70 is not a reliable pass.
        r = _route(tier="high", floor=70)
        p = r["recommended"]
        self.assertEqual(p["model"], "qwen3.8-27b")
        self.assertFalse(p["margin_is_meaningful"])
        self.assertLess(p["margin_over_floor"], route.DEFAULT_TIE_BAND)

    def test_flags_a_model_that_did_not_finish_every_task(self) -> None:
        r = _route(tier="high", floor=70)
        self.assertFalse(r["recommended"]["finished_every_task"])
        self.assertEqual(r["recommended"]["completion"], "4/5")

    def test_nothing_clears_an_impossible_floor(self) -> None:
        r = _route(tier="high", floor=95)
        self.assertEqual(r["status"], "nothing_clears_floor")
        self.assertIsNone(r["recommended"])
        self.assertIn("short by", r["reason"])

    def test_availability_filters_before_ranking(self) -> None:
        # A Bedrock-only developer must not be sent to a self-hosted model.
        r = _route(
            tier="high", floor=70, available=["claude-opus-5", "claude-sonnet-5"]
        )
        self.assertEqual(r["recommended"]["model"], "claude-sonnet-5")

    def test_unmeasured_available_models_are_named_not_scored(self) -> None:
        r = _route(tier="low", floor=70, available=["gpt-9", "llama-7"])
        self.assertEqual(r["status"], "no_candidates")
        self.assertIn("gpt-9", r["excluded"]["available_but_not_measured"])
        self.assertIn("none of which this benchmark", r["reason"])

    def test_aliases_resolve_from_any_spelling(self) -> None:
        r = _route(
            tier="low",
            floor=60,
            available=["us.anthropic.claude-sonnet-5[1m]", "Claude Haiku 4.5"],
        )
        self.assertEqual(
            sorted(r["candidates_considered"]),
            ["claude-haiku-4-5", "claude-sonnet-5"],
        )

    def test_bad_tier_is_rejected(self) -> None:
        with self.assertRaisesRegex(route.RouteError, "--tier must be one of"):
            _route(tier="enormous", floor=70)


class AllowListTest(unittest.TestCase):
    def _write(self, body: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmp.write(body)
        tmp.close()
        return Path(tmp.name)

    def test_allow_list_is_a_hard_constraint(self) -> None:
        p = self._write("claude-haiku-4-5  # cheap\n")
        r = _route(tier="low", floor=50, allowed_file=p, no_allow_list=False)
        self.assertEqual(r["candidates_considered"], ["claude-haiku-4-5"])

    def test_comments_and_blank_lines_are_ignored(self) -> None:
        # The whole reason for a plain list: a commented-out model is a
        # comment, and cannot be mistaken for policy.
        p = self._write(
            "# do not enable yet:\n"
            "#   claude-opus-5\n"
            "\n"
            "claude-haiku-4-5  # approved for docs\n"
        )
        r = _route(tier="low", floor=50, allowed_file=p, no_allow_list=False)
        self.assertEqual(r["candidates_considered"], ["claude-haiku-4-5"])

    def test_allowed_but_unmeasured_models_are_named(self) -> None:
        p = self._write("some-model-we-never-ran\n")
        r = _route(tier="low", floor=50, allowed_file=p, no_allow_list=False)
        self.assertEqual(r["status"], "no_candidates")
        self.assertIn(
            "some-model-we-never-ran", r["excluded"]["allowed_but_not_measured"]
        )
        self.assertIn("measured none of them", r["reason"])

    def test_a_missing_explicit_allow_list_is_an_error(self) -> None:
        # Ignoring it would apply no policy while the caller believed one was.
        with self.assertRaisesRegex(route.RouteError, "does not exist"):
            _route(
                tier="low",
                floor=50,
                allowed_file=Path("/nope/allowed-models.txt"),
                no_allow_list=False,
            )

    def test_an_allow_list_with_only_comments_is_an_error(self) -> None:
        # Permitting nothing is almost never what someone meant to write.
        p = self._write("# every model is commented out\n#claude-opus-5\n")
        with self.assertRaisesRegex(route.RouteError, "lists no models"):
            _route(tier="low", floor=50, allowed_file=p, no_allow_list=False)

    def test_the_shipped_lists_commented_suggestions_are_not_policy(self) -> None:
        # The file suggests adding claude-sonnet-5 and claude-haiku-4-5 as
        # commented lines. A comment is a comment.
        path = _VEND / "allowed-models.txt"
        names = route.parse_allow_list(path)
        self.assertNotIn("claude-sonnet-5", names)
        self.assertNotIn("claude-haiku-4-5", names)
        self.assertIn(
            "claude-sonnet-5",
            path.read_text(encoding="utf-8"),
            "the file should still suggest sonnet as an addition",
        )

    def test_the_shipped_allow_list_parses(self) -> None:
        names = route.parse_allow_list(_VEND / "allowed-models.txt")
        self.assertEqual(len(names), 6)
        self.assertIn("claude-opus-5", names)

    def test_the_shipped_allow_list_is_the_frontier(self) -> None:
        # It is the frontier by construction, so a regenerated frontier must
        # not leave the list quietly stale.
        payload = json.loads((_VEND / "models.json").read_text(encoding="utf-8"))
        frontier = {m["model"] for m in payload["models"] if m["on_combined_frontier"]}
        names = set(route.parse_allow_list(_VEND / "allowed-models.txt"))
        self.assertEqual(names, frontier)

    def test_the_shipped_allow_list_says_what_it_excludes(self) -> None:
        # Four of the five are self-hosted, so the list as written leaves a
        # hosted-API developer with one option. Shipping that silently would be
        # a trap.
        text = (_VEND / "allowed-models.txt").read_text(encoding="utf-8")
        self.assertIn("claude-sonnet-5", text)
        self.assertIn("self-hosted", text)

    def test_the_shipped_allow_list_forces_opus_for_bedrock_users(self) -> None:
        # Recorded because it is the argument for editing the shipped list:
        # four of its five models are self-hosted.
        r = _route(
            tier="high",
            floor=70,
            available=["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
            allowed_file=_VEND / "allowed-models.txt",
            no_allow_list=False,
        )
        self.assertEqual(r["recommended"]["model"], "claude-opus-5")


class SkillTriggerTest(unittest.TestCase):
    """The description is the only thing that decides whether the skill fires."""

    def setUp(self) -> None:
        self.text = (_VEND / "SKILL.md").read_text(encoding="utf-8")
        self.desc = re.search(r'^description: "(.*)"$', self.text, re.M).group(1)

    def test_description_names_when_to_run_and_when_not_to(self) -> None:
        # A trigger with no negative half fires on typos and gets switched off.
        self.assertIn("BEFORE starting a substantial coding task", self.desc)
        self.assertIn("Do NOT run it", self.desc)

    def test_description_fits_the_frontmatter_budget(self) -> None:
        # Descriptions are read in bulk to decide which skill fires; a long one
        # crowds out the others.
        self.assertLess(len(self.desc), 1024, "description is getting long")

    def test_skill_has_an_early_bail_out(self) -> None:
        # Firing broadly is only safe if the first thing it does is check
        # whether it should have.
        self.assertIn("Do not run when", self.text)
        self.assertLess(
            self.text.index("Do not run when"),
            self.text.index("Three steps"),
            "the bail-out must come before the procedure",
        )

    def test_skill_states_it_runs_once_per_task(self) -> None:
        self.assertIn("Once per task", self.text)


class CliTest(unittest.TestCase):
    def test_runs_as_a_script_on_stdlib_alone(self) -> None:
        # The skill shells out to it wherever it is installed, with no venv.
        out = subprocess.run(
            [
                sys.executable,
                str(_VEND / "route.py"),
                "--tier",
                "low",
                "--floor",
                "60",
                "--no-allow-list",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        payload = json.loads(out.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertIn("recommended", payload)

    def test_exits_nonzero_when_nothing_is_recommended(self) -> None:
        out = subprocess.run(
            [
                sys.executable,
                str(_VEND / "route.py"),
                "--tier",
                "high",
                "--floor",
                "99",
                "--no-allow-list",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(out.returncode, 1)
        self.assertEqual(json.loads(out.stdout)["status"], "nothing_clears_floor")

    def test_route_py_imports_nothing_outside_the_standard_library(self) -> None:
        src = (_VEND / "route.py").read_text(encoding="utf-8")
        imports = set(re.findall(r"^(?:from|import)\s+([a-zA-Z_][\w.]*)", src, re.M))
        allowed = {"argparse", "json", "re", "sys", "pathlib", "typing", "__future__"}
        self.assertEqual(imports - allowed, set())


import re  # noqa: E402  (used by the import-check test above)

if __name__ == "__main__":
    unittest.main()
