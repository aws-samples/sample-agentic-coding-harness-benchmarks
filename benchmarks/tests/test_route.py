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
        # claude-haiku-5-5 averages 72.69 overall but 79.16 on low and 60.00
        # on high. A floor of 75 must take it for low work and reject it for
        # hard work, as it must qwen3.8-27b (78.48 overall, 71.45 on high).
        r = _route(tier="high", floor=75)
        self.assertNotIn(r["recommended"]["model"], {"claude-haiku-5-5", "qwen3.8-27b"})
        r = _route(tier="low", floor=75)
        self.assertEqual(r["recommended"]["model"], "claude-haiku-5-5")

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
        # The file suggests the self-hosted kimi-k3, gemma-4-31b and
        # qwen3.6-35b as commented lines. A comment is a comment.
        path = _VEND / "allowed-models.txt"
        names = route.parse_allow_list(path)
        self.assertNotIn("kimi-k3", names)
        self.assertNotIn("qwen3.6-35b", names)
        self.assertIn(
            "#   kimi-k3",
            path.read_text(encoding="utf-8"),
            "the file should still suggest kimi-k3 as an addition",
        )

    def test_the_shipped_allow_list_parses(self) -> None:
        names = route.parse_allow_list(_VEND / "allowed-models.txt")
        self.assertEqual(len(names), 5)
        self.assertIn("claude-opus-5", names)

    def test_the_shipped_allow_list_is_the_frontier(self) -> None:
        # It is the frontier by construction, so a regenerated frontier must
        # not leave the list quietly stale.
        payload = json.loads((_VEND / "models.json").read_text(encoding="utf-8"))
        frontier = {m["model"] for m in payload["models"] if m["on_combined_frontier"]}
        names = set(route.parse_allow_list(_VEND / "allowed-models.txt"))
        self.assertEqual(names, frontier)

    def test_the_shipped_allow_list_says_what_it_excludes(self) -> None:
        # qwen3.8-27b is self-hosted, so a hosted-API developer who keeps it
        # would be sent to a model they cannot select. The file must say so.
        text = (_VEND / "allowed-models.txt").read_text(encoding="utf-8")
        self.assertIn("qwen3.8-27b, is self-hosted", text)

    def test_the_shipped_allow_list_forces_opus_for_bedrock_users(self) -> None:
        # Recorded because it is the argument for editing the shipped list:
        # an available model that is not on it (claude-sonnet-5,
        # claude-haiku-4-5) is never recommended, however well it scores.
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

    def test_skill_names_no_measured_model(self) -> None:
        # Model names and their scores belong in models.json, which the build
        # regenerates. A name in SKILL.md goes stale the next time it does.
        models = json.loads(MODELS.read_text(encoding="utf-8"))["models"]
        named = [m["model"] for m in models if m["model"] in self.text]
        self.assertEqual(named, [])

    def test_skill_runs_route_py_with_the_rubric(self) -> None:
        self.assertIn("python3 route.py --confidence", self.text)
        self.assertIn("rubric.md", self.text)


def _rubric(**overrides: int) -> dict[str, int]:
    """Six rubric scores, all 1 unless overridden."""
    scores = {d: 1 for d in route.DIMENSIONS}
    scores.update(overrides)
    return scores


def _models_file(models: list[dict]) -> Path:
    """A schema-2 models.json holding just these models."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({"schema_version": "2.0", "models": models}, tmp)
    tmp.close()
    return Path(tmp.name)


def _model(name: str, cost: float, scores: dict, completion: dict) -> dict:
    return {
        "model": name,
        "score": 80.0,
        "cost_per_task_usd": cost,
        "score_by_difficulty": scores,
        "completion_by_difficulty": completion,
    }


class RubricParseTest(unittest.TestCase):
    def test_accepts_plain_nested_and_wrapped_scores(self) -> None:
        plain = _rubric()
        nested = {d: {"score": v, "evidence": "x"} for d, v in plain.items()}
        for text in (plain, nested, {"scores": nested}):
            self.assertEqual(route.parse_rubric(json.dumps(text)), plain)

    def test_rejects_a_missing_dimension(self) -> None:
        scores = _rubric()
        del scores["ambiguity"]
        with self.assertRaisesRegex(route.RouteError, "must score exactly"):
            route.parse_rubric(json.dumps(scores))

    def test_rejects_a_score_outside_0_to_2(self) -> None:
        for bad in (3, -1, True, "1"):
            with self.assertRaisesRegex(route.RouteError, "must be 0, 1 or 2"):
                route.parse_rubric(json.dumps(_rubric(blast_radius=bad)))


class RubricPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = route.load_policy(route.DEFAULT_POLICY)

    def test_shipped_policy_loads(self) -> None:
        self.assertEqual(
            list(self.policy["difficulty"]["buckets"]), list(route.BUCKETS)
        )

    def test_matches_the_benchmark_implementation_for_every_score(self) -> None:
        # The build buckets benchmark tasks with swe_router_rubric.py; the skill
        # buckets the developer's task with route.py. They must never disagree.
        sys.path.insert(0, str(_REPO_ROOT / "benchmarks" / "scripts"))
        import itertools

        import swe_router_rubric as rubric

        typed = rubric.load_policy(route.DEFAULT_POLICY)
        for combo in itertools.product(route.SCORES, repeat=len(route.DIMENSIONS)):
            scores = dict(zip(route.DIMENSIONS, combo))
            self.assertEqual(
                (
                    route.bucket_for(scores, self.policy),
                    route.required_quality(scores, self.policy),
                ),
                (rubric.bucket(scores, typed), rubric.required_quality(scores, typed)),
                scores,
            )

    def test_blast_radius_and_verification_set_the_required_quality(self) -> None:
        self.assertEqual(
            route.required_quality(_rubric(blast_radius=0), self.policy), 60
        )
        self.assertEqual(
            route.required_quality(
                _rubric(blast_radius=2, verification=2), self.policy
            ),
            80,
        )

    def test_a_malformed_policy_fails_with_a_message(self) -> None:
        policy = json.loads(route.DEFAULT_POLICY.read_text(encoding="utf-8"))
        policy["difficulty"]["buckets"]["hard"] = "4-5"
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(policy, tmp)
        tmp.close()
        with self.assertRaisesRegex(route.RouteError, "bucket hard must be"):
            route.load_policy(Path(tmp.name))

    def test_a_missing_policy_is_an_error(self) -> None:
        with self.assertRaisesRegex(route.RouteError, "no routing policy"):
            route.load_policy(Path("/nonexistent/routing-policy.json"))


class RubricRoutingTest(unittest.TestCase):
    def _go(self, models: list[dict], **kw) -> dict:
        args = dict(
            rubric=_rubric(), confidence="high", models_path=_models_file(models)
        )
        args.update(kw)
        return _route(tie_band=None, **args)

    def test_picks_the_cheapest_model_clearing_the_bucket_column(self) -> None:
        # All ones: difficulty sum 4 (hard), blast radius 1 (70).
        models = [
            _model("small", 1.0, {"hard": 72.0}, {"hard": "10/10"}),
            _model("large", 9.0, {"hard": 85.0}, {"hard": "10/10"}),
        ]
        r = self._go(models)
        self.assertEqual(
            (r["bucket"], r["required_quality"], r["recommended"]["model"]),
            ("hard", 70.0, "small"),
        )

    def test_skips_a_model_that_failed_a_task_in_the_bucket(self) -> None:
        models = [
            _model("small", 1.0, {"hard": 90.0}, {"hard": "9/10"}),
            _model("large", 9.0, {"hard": 85.0}, {"hard": "10/10"}),
        ]
        r = self._go(models)
        self.assertEqual(
            (r["recommended"]["model"], r["skipped_for_failures"][0]["model"]),
            ("large", "small"),
        )

    def test_reads_the_next_harder_bucket_when_one_is_missing(self) -> None:
        models = [_model("only", 1.0, {"hard": 75.0}, {"hard": "5/5"})]
        r = self._go(models, rubric=_rubric(hidden_contract=0, ambiguity=0))
        self.assertEqual(
            (r["bucket"], r["scored_from_harder_bucket"]),
            ("moderate", {"only": "hard"}),
        )

    def test_low_confidence_stays_on_the_current_model(self) -> None:
        r = self._go([], confidence="low")
        self.assertEqual((r["status"], r["recommended"]), ("unsure", None))

    def test_schema_1_models_cannot_serve_the_rubric(self) -> None:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(
            {"schema_version": "1.1", "models": [{"model": "m", "score": 1}]}, tmp
        )
        tmp.close()
        with self.assertRaisesRegex(route.RouteError, "no score_by_difficulty"):
            self._go([], models_path=Path(tmp.name))

    def test_the_shipped_data_pick_is_never_dominated(self) -> None:
        for combo in ((0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 1, 1), (2, 2, 2, 2)):
            for radius in (0, 1, 2):
                dims = dict(zip(route.DIMENSIONS[:4], combo), blast_radius=radius)
                r = _route(tie_band=None, rubric=_rubric(**dims), confidence="high")
                pick = r["recommended"]
                if not pick:
                    continue
                for other in r["cleared_floor"]:
                    self.assertFalse(
                        other["score"] > pick["score"]
                        and other["cost_per_task_usd"] < pick["cost_per_task_usd"]
                    )


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

    def _cli(self, *argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(_VEND / "route.py"), "--no-allow-list", *argv],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_rubric_runs_as_a_script(self) -> None:
        out = self._cli("--confidence", "high", "--rubric", json.dumps(_rubric()))
        self.assertEqual(
            (out.returncode, json.loads(out.stdout)["bucket"]), (0, "hard")
        )

    def test_needs_the_rubric_or_the_legacy_pair(self) -> None:
        self.assertEqual(self._cli("--tier", "low").returncode, 2)

    def test_rejects_the_rubric_with_a_tier(self) -> None:
        out = self._cli("--rubric", json.dumps(_rubric()), "--tier", "low")
        self.assertEqual(out.returncode, 2)

    def test_an_old_python_gets_a_message_not_a_traceback(self) -> None:
        code = (
            "import runpy, sys; sys.version_info = (3, 7, 0); "
            f"sys.argv = ['route.py', '--tier', 'low', '--floor', '60']; "
            f"runpy.run_path({str(_VEND / 'route.py')!r}, run_name='__main__')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        self.assertEqual(
            (out.returncode, json.loads(out.stdout)["error"]),
            (2, "route.py needs Python 3.8 or later"),
        )

    def test_installer_refuses_without_python3_and_writes_nothing(self) -> None:
        # A stub python3 that fails stands in for a missing or too-old one. The
        # check runs before any download, so this needs no network.
        tmp = Path(tempfile.mkdtemp())
        (tmp / "bin").mkdir()
        stub = tmp / "bin" / "python3"
        stub.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        stub.chmod(0o755)
        env = {"PATH": f"{tmp / 'bin'}:/usr/bin:/bin", "HOME": str(tmp)}
        out = subprocess.run(
            ["bash", str(_VEND / "install.sh"), "--dir", str(tmp / "skills")],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(
            (
                out.returncode,
                "Nothing was written" in out.stderr,
                (tmp / "skills").exists(),
            ),
            (1, True, False),
        )

    def test_route_py_imports_nothing_outside_the_standard_library(self) -> None:
        src = (_VEND / "route.py").read_text(encoding="utf-8")
        imports = set(re.findall(r"^(?:from|import)\s+([a-zA-Z_][\w.]*)", src, re.M))
        allowed = {"argparse", "json", "re", "sys", "pathlib", "typing", "__future__"}
        self.assertEqual(imports - allowed, set())


import re  # noqa: E402  (used by the import-check test above)

if __name__ == "__main__":
    unittest.main()
