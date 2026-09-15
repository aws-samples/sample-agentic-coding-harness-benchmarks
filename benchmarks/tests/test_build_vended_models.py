"""Tests for the vended models.json generator."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location(
    "build_vended_models", _SCRIPTS_DIR / "build_vended_models.py"
)
bvm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bvm)

_VEND_DIR = _REPO_ROOT / "vend" / "swe-router"


def _frontier(**overrides) -> dict:
    """Build a minimal frontier JSON shaped like the real one."""
    data = {
        "harness": "omp",
        "skill": "swe3",
        "repo": "mcp-gateway-registry-v2",
        "all_models": [
            {
                "model": "big",
                "mean_score": 80.0,
                "mean_cost_per_task": 10.0,
                "hosting": "Bedrock",
                "n_scored": 21,
                "n_tasks": 21,
                "excluded_tasks": [],
            },
            {
                "model": "small",
                "mean_score": 60.0,
                "mean_cost_per_task": 1.0,
                "hosting": "self-hosted",
                "n_scored": 20,
                "n_tasks": 21,
                "excluded_tasks": ["a-task"],
            },
        ],
        "combined_frontier_cross_hosting_directional": [{"model": "small"}],
        "bedrock_frontier": [{"model": "big"}],
        "self_hosted_frontier": [{"model": "small"}],
    }
    data.update(overrides)
    return data


def _write(data: dict) -> Path:
    """Write a frontier dict to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


class BuildTest(unittest.TestCase):
    def test_includes_every_model_not_only_the_frontier(self) -> None:
        # The skill filters to what the user's assistant offers before ranking,
        # and that set may contain no frontier model at all. Shipping only the
        # frontier would leave it mute for those users.
        out = bvm.build(_write(_frontier()), _REPO_ROOT)
        self.assertEqual({m["model"] for m in out["models"]}, {"big", "small"})

    def test_frontier_membership_is_recorded(self) -> None:
        out = bvm.build(_write(_frontier()), _REPO_ROOT)
        by = {m["model"]: m for m in out["models"]}
        self.assertTrue(by["small"]["on_combined_frontier"])
        self.assertFalse(by["big"]["on_combined_frontier"])
        # big is on the Bedrock frontier even though it is off the combined one.
        self.assertTrue(by["big"]["on_hosting_frontier"])

    def test_frontier_flags_are_documented_as_context_not_a_key(self) -> None:
        # They come from overall means and can disagree with the per-tier
        # ranking, so the payload has to say what they are for.
        out = bvm.build(_write(_frontier()), _REPO_ROOT)
        note = out["measurement_basis"]["frontier_flags"]
        self.assertIn("not a selection key", note)
        self.assertIn("score_by_complexity", note)

    def test_a_frontier_model_can_still_lose_at_a_tier(self) -> None:
        # The reason the flags are not a selection key, pinned against the real
        # data: qwen3.8-27b is on the combined frontier and trails
        # claude-sonnet-5, which is not, on high-complexity work.
        payload = bvm.build(bvm.DEFAULT_SOURCE, _REPO_ROOT)
        by = {m["model"]: m for m in payload["models"]}
        q, s = by["qwen3.8-27b"], by["claude-sonnet-5"]
        self.assertTrue(q["on_combined_frontier"])
        self.assertFalse(s["on_combined_frontier"])
        self.assertLess(
            q["score_by_complexity"]["high"], s["score_by_complexity"]["high"]
        )

    def test_models_are_ordered_by_score_descending(self) -> None:
        out = bvm.build(_write(_frontier()), _REPO_ROOT)
        scores = [m["score"] for m in out["models"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_incomplete_runs_stay_visible(self) -> None:
        # A mean over fewer tasks should not be averaged into silence.
        out = bvm.build(_write(_frontier()), _REPO_ROOT)
        small = next(m for m in out["models"] if m["model"] == "small")
        self.assertEqual((small["tasks_completed"], small["tasks_total"]), (20, 21))
        self.assertEqual(small["excluded_tasks"], ["a-task"])

    def test_provenance_carries_the_measurement_context(self) -> None:
        # A vended file has no reader who knows this repo, so the context
        # travels with the data.
        out = bvm.build(_write(_frontier()), _REPO_ROOT)
        p = out["provenance"]
        self.assertEqual(p["harness"], "omp")
        self.assertEqual(p["skill"], "swe3")
        self.assertEqual(p["dataset"], "mcp-gateway-registry-v2")
        self.assertTrue(p["measured_on"])
        self.assertIn("model", p["judge"])

    def test_cost_basis_is_explained_for_both_hostings(self) -> None:
        out = bvm.build(_write(_frontier()), _REPO_ROOT)
        basis = out["measurement_basis"]["cost_basis"]
        self.assertIn("Bedrock", basis)
        self.assertIn("self-hosted", basis)

    def test_missing_source_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(SystemExit, "no frontier JSON"):
            bvm.build(Path("/nope/absent.json"), _REPO_ROOT)

    def test_empty_source_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(SystemExit, "no all_models"):
            bvm.build(_write(_frontier(all_models=[])), _REPO_ROOT)


class CommittedArtifactTest(unittest.TestCase):
    """The vended files are committed, so they are checked like any other input."""

    def test_committed_models_json_matches_the_generator(self) -> None:
        # Guards the staleness trap: opus once moved $7.63 -> $11.95 with every
        # score unchanged, so a drifted copy looks perfectly plausible.
        payload = bvm.build(bvm.DEFAULT_SOURCE, _REPO_ROOT)
        expected = json.dumps(payload, indent=2) + "\n"
        self.assertEqual(
            bvm.DEFAULT_OUT.read_text(encoding="utf-8"),
            expected,
            "vend/swe-router/models.json is stale; run build_vended_models.py",
        )

    def test_score_by_complexity_is_present_for_every_model(self) -> None:
        # The skill compares a floor against the tier, not the overall mean.
        # A model without tier scores silently falls back to the mean, which is
        # what recommends qwen3.8-27b (74.74 overall, 57.2 on high) for hard work.
        payload = bvm.build(bvm.DEFAULT_SOURCE, _REPO_ROOT)
        for m in payload["models"]:
            self.assertTrue(
                m["score_by_complexity"], f"{m['model']} has no per-tier scores"
            )

    def test_completion_by_complexity_is_present_for_every_model(self) -> None:
        # A failure is excluded from the mean rather than averaged in, so the
        # completion counter is the only place it shows.
        payload = bvm.build(bvm.DEFAULT_SOURCE, _REPO_ROOT)
        for m in payload["models"]:
            self.assertTrue(
                m["completion_by_complexity"], f"{m['model']} has no completion counts"
            )

    def test_failed_tasks_are_excluded_from_tier_means(self) -> None:
        # devstral-2-123b failed 2 of 6 medium tasks. Averaging those zeros in
        # would put its tier means below the overall score they sit beside.
        payload = bvm.build(bvm.DEFAULT_SOURCE, _REPO_ROOT)
        d = next(m for m in payload["models"] if m["model"] == "devstral-2-123b")
        self.assertEqual(d["completion_by_complexity"]["medium"], "4/6")
        self.assertGreater(d["score_by_complexity"]["medium"], 30.0)

    def test_tier_scores_bracket_the_overall_mean(self) -> None:
        # A sanity check on the join: per-tier means must straddle the overall
        # mean, or the two came from different runs.
        payload = bvm.build(bvm.DEFAULT_SOURCE, _REPO_ROOT)
        for m in payload["models"]:
            tiers = list(m["score_by_complexity"].values())
            if len(tiers) < 2:
                continue
            self.assertLessEqual(min(tiers), m["score"] + 0.01, m["model"])
            self.assertGreaterEqual(max(tiers), m["score"] - 0.01, m["model"])

    def test_source_commit_tracks_the_frontier_not_head(self) -> None:
        # Stamping HEAD would make the file differ after every unrelated commit
        # and turn --check into noise. It must identify the data's version.
        payload = bvm.build(bvm.DEFAULT_SOURCE, _REPO_ROOT)
        stamped = payload["provenance"]["source_commit"]
        head = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        last_touch = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO_ROOT),
                "log",
                "-1",
                "--format=%h",
                "--",
                str(bvm.DEFAULT_SOURCE.relative_to(_REPO_ROOT)),
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        self.assertEqual(stamped, last_touch)
        if head != last_touch:
            self.assertNotEqual(stamped, head)

    def test_vend_dir_holds_exactly_the_portable_files(self) -> None:
        # The portability contract: the skill must work in a directory holding
        # only these. An extra file is a dependency someone will start relying on.
        # Importing route.py in tests leaves a __pycache__; it is a build
        # artifact of running the checks, not something anyone installs.
        # install.sh is the installer, not part of the skill: it fetches the
        # five installed files and never copies itself, so it cannot become a
        # runtime dependency. test_installer_installs_exactly_the_skill_files
        # holds that line.
        self.assertEqual(
            sorted(p.name for p in _VEND_DIR.iterdir() if p.is_file()),
            [
                "README.md",
                "SKILL.md",
                "allowed-models.txt",
                "install.sh",
                "model-aliases.json",
                "models.json",
                "route.py",
            ],
        )

    def test_installer_installs_exactly_the_skill_files(self) -> None:
        # The installer's file list is the install contract, and it is a shell
        # string rather than anything importable -- so read it back and check it
        # against the directory. A file added to vend/ but not to SKILL_FILES
        # would be documented as part of the skill and never actually installed.
        text = (_VEND_DIR / "install.sh").read_text(encoding="utf-8")
        declared = next(
            line.split('"')[1]
            for line in text.splitlines()
            if line.startswith("SKILL_FILES=")
        )
        self.assertEqual(
            sorted(declared.split()),
            [
                "SKILL.md",
                "allowed-models.txt",
                "model-aliases.json",
                "models.json",
                "route.py",
            ],
        )
        self.assertNotIn("install.sh", declared)

    def test_readme_tier_tables_match_the_data(self) -> None:
        # The README prints the four tier tables so a reader can see the scan.
        # Hand-copied numbers drift; a regenerated models.json must not leave
        # the documented tables quietly wrong.
        import re

        payload = json.loads(bvm.DEFAULT_OUT.read_text())
        by = {m["model"]: m for m in payload["models"]}
        text = (_VEND_DIR / "README.md").read_text(encoding="utf-8")
        checked = 0
        for tier in ("trivial", "low", "medium", "high"):
            block = text.split(f"**`{tier}`** —", 1)
            if len(block) < 2:
                continue
            table = block[1].split("\n\n", 2)[1]
            rows = re.findall(
                r"^\| \$([\d.]+) \| `([^`]+)` \| ([\d.]+) \|", table, re.M
            )
            self.assertTrue(rows, f"no rows parsed for {tier}")
            costs = [float(c) for c, _, _ in rows]
            self.assertEqual(costs, sorted(costs), f"{tier} table not cheapest-first")
            for cost, model, score in rows:
                self.assertAlmostEqual(
                    float(score), by[model]["score_by_complexity"][tier], places=1
                )
                self.assertAlmostEqual(
                    float(cost), by[model]["cost_per_task_usd"], places=2
                )
                checked += 1
        self.assertGreaterEqual(checked, 60, "expected four full tier tables")

    def test_every_model_has_an_alias_entry(self) -> None:
        # A model with no aliases can never be matched against an assistant's
        # list, so it is invisible to the skill that ships beside it.
        models = {m["model"] for m in json.loads(bvm.DEFAULT_OUT.read_text())["models"]}
        aliases = json.loads((_VEND_DIR / "model-aliases.json").read_text())["aliases"]
        self.assertEqual(models - set(aliases), set())

    def test_every_alias_entry_names_a_real_model(self) -> None:
        models = {m["model"] for m in json.loads(bvm.DEFAULT_OUT.read_text())["models"]}
        aliases = json.loads((_VEND_DIR / "model-aliases.json").read_text())["aliases"]
        self.assertEqual(set(aliases) - models, set())

    def test_aliases_are_unambiguous_across_models(self) -> None:
        # Two models sharing a normalized alias would make matching a coin flip.
        aliases = json.loads((_VEND_DIR / "model-aliases.json").read_text())["aliases"]
        # Duplicates inside one model are harmless -- "claude-opus-5" and
        # "Claude Opus 5" normalize to the same key and both mean that model.
        # A key shared by two DIFFERENT models makes matching a coin flip.
        seen: dict[str, str] = {}
        for model, names in aliases.items():
            for name in names:
                key = name.lower().replace("-", "").replace("_", "").replace(" ", "")
                self.assertEqual(
                    seen.setdefault(key, model),
                    model,
                    f"alias {name!r} is claimed by {seen[key]} and {model}",
                )

    def test_skill_does_not_reference_the_benchmark_repo_internals(self) -> None:
        # Portability: the vended skill must not tell a consumer to look at a
        # path that only exists inside this repository.
        text = (_VEND_DIR / "SKILL.md").read_text(encoding="utf-8")
        for path in ("benchmarks/", "docs/metrics/", "run-swe-headless"):
            self.assertNotIn(path, text, f"SKILL.md references {path}")


if __name__ == "__main__":
    unittest.main()
