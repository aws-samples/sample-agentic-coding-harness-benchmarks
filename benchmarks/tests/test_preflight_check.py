"""Tests for the end-to-end benchmark pre-flight helper."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))


def _load_preflight():
    """Import preflight_check.py by path (module name has no dashes, but be explicit)."""
    path = _SCRIPTS_DIR / "preflight_check.py"
    spec = importlib.util.spec_from_file_location("preflight_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pf = _load_preflight()

_DATASET = """\
schema_version: "1.0"
name: t
title: T
description: d
default_ref: main
metrics: [input_tokens]
complexity_levels: [low]
tasks:
  - id: task-one
    repo: https://github.com/example/my-repo
    complexity: low
    tags: [x]
    problem_statement: do the thing
  - id: task-two
    repo: https://github.com/example/my-repo
    complexity: low
    tags: [x]
    problem_statement: do the other thing
"""


class TargetDirsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = _SCRIPTS_DIR.parent / "dataset" / "_preflight_test.yaml"
        self.ds.write_text(_DATASET, encoding="utf-8")

    def tearDown(self) -> None:
        self.ds.unlink(missing_ok=True)

    def test_one_dir_per_task_with_slug(self) -> None:
        dirs = pf._target_dirs(str(self.ds), "us.anthropic.claude-opus-4-8")
        self.assertEqual(len(dirs), 2)
        # Layout is <model>/<harness>/<skill>/<repo>/<task>; default agent claude ->
        # claude-code, default skill swe3; Bedrock prefix stripped for the slug.
        self.assertTrue(
            str(dirs[0]).endswith("claude-opus-4-8/claude-code/swe3/my-repo/task-one")
        )
        self.assertTrue(
            str(dirs[1]).endswith("claude-opus-4-8/claude-code/swe3/my-repo/task-two")
        )

    def test_plain_model_slug_unchanged(self) -> None:
        dirs = pf._target_dirs(str(self.ds), "qwen3-coder-30b")
        self.assertTrue(
            str(dirs[0]).endswith("qwen3-coder-30b/claude-code/swe3/my-repo/task-one")
        )

    def test_pi_agent_uses_pi_harness_level(self) -> None:
        dirs = pf._target_dirs(str(self.ds), "qwen3-coder-30b", agent="pi")
        self.assertTrue(
            str(dirs[0]).endswith("qwen3-coder-30b/pi/swe3/my-repo/task-one")
        )

    def test_skill_is_its_own_path_level(self) -> None:
        # swe2 and swe3 are sibling levels under the harness; neither is a suffix.
        swe3 = pf._target_dirs(str(self.ds), "qwen3-coder-30b", skill="swe3")
        swe2 = pf._target_dirs(str(self.ds), "qwen3-coder-30b", skill="swe2")
        self.assertTrue(
            str(swe3[0]).endswith("qwen3-coder-30b/claude-code/swe3/my-repo/task-one")
        )
        self.assertTrue(
            str(swe2[0]).endswith("qwen3-coder-30b/claude-code/swe2/my-repo/task-one")
        )

    def test_output_scope_replaces_the_repo_level(self) -> None:
        # A second dataset over the same repo must clear its OWN folder, never
        # the first dataset's committed results.
        scoped = _SCRIPTS_DIR.parent / "dataset" / "_preflight_test_v2.yaml"
        scoped.write_text(
            _DATASET.replace(
                "default_ref: main\n", "default_ref: main\noutput_scope: my-repo-v2\n"
            ),
            encoding="utf-8",
        )
        try:
            dirs = pf._target_dirs(str(scoped), "qwen3-coder-30b", agent="pi")
            self.assertTrue(
                str(dirs[0]).endswith("qwen3-coder-30b/pi/swe3/my-repo-v2/task-one")
            )
        finally:
            scoped.unlink(missing_ok=True)


class ExistingTest(unittest.TestCase):
    def test_only_folders_with_artifacts_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            has = root / "with-artifact"
            has.mkdir()
            (has / "lld.md").write_text("x", encoding="utf-8")
            empty = root / "empty"
            empty.mkdir()
            missing = root / "does-not-exist"
            found = pf._existing([has, empty, missing])
            self.assertEqual(found, [has])

    def test_failed_attempt_without_artifacts_counts(self) -> None:
        # A failed attempt can leave only metrics.json and the agent's stream
        # file; the next run would append to that stream (issue #196).
        with tempfile.TemporaryDirectory() as tmp:
            leftover = Path(tmp) / "failed-task"
            leftover.mkdir()
            (leftover / "metrics.json").write_text("{}", encoding="utf-8")
            (leftover / "strands-stream.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertEqual(pf._existing([leftover]), [leftover])

    def test_folder_outside_the_root_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "swe-benchmark-data"
            escape = (
                root / "model" / "omp" / "swe3" / "scope" / ".." / ".." / ".." / ".."
            )
            with self.assertRaises(pf.DatasetError):
                pf._assert_inside_root(root, [escape / ".." / "scripts"])

    def test_folder_inside_the_root_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "swe-benchmark-data"
            pf._assert_inside_root(
                root, [root / "model" / "omp" / "swe3" / "s" / "task"]
            )

    def test_clear_removes_a_failed_attempt_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            leftover = Path(tmp) / "failed-task"
            leftover.mkdir()
            (leftover / "metrics.json").write_text("{}", encoding="utf-8")
            pf._run_clear([leftover])
            self.assertFalse(leftover.exists())


class RefIsShaTest(unittest.TestCase):
    def test_hex_refs_are_shas(self) -> None:
        self.assertTrue(pf._ref_is_sha("d67f0fc"))
        self.assertTrue(pf._ref_is_sha("d67f0fcba86dda58d67f0fcba86dda58d67f0fcb"))

    def test_tags_and_branches_are_not_shas(self) -> None:
        # A version tag and a branch name must be checked as refs, not skipped.
        self.assertFalse(pf._ref_is_sha("1.27.1"))
        self.assertFalse(pf._ref_is_sha("v1.0.20"))
        self.assertFalse(pf._ref_is_sha("main"))


class DiagnoseGitErrorTest(unittest.TestCase):
    def test_missing_credential_is_explained(self) -> None:
        hint = pf._diagnose_git_error("fatal: could not read Username for 'https://x'")
        self.assertIsNotNone(hint)
        self.assertIn("credential.helper", hint)

    def test_untrusted_ca_is_explained(self) -> None:
        hint = pf._diagnose_git_error(
            "fatal: unable to access: SSL certificate problem"
        )
        self.assertIsNotNone(hint)
        self.assertIn("NODE_EXTRA_CA_CERTS", hint)

    def test_unrecognized_error_returns_none(self) -> None:
        self.assertIsNone(pf._diagnose_git_error("fatal: something entirely new"))


class CheckRepoRefTest(unittest.TestCase):
    def test_reachable_repo_returns_none(self) -> None:
        with mock.patch.object(pf.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="ref\n", stderr="")
            self.assertIsNone(pf._check_repo_ref("https://host/org/repo", "1.2.3"))

    def test_named_ref_is_passed_to_ls_remote(self) -> None:
        # A tag must be checked explicitly, or a reachable repo with a missing tag
        # would pass the pre-flight and fail at clone time.
        with mock.patch.object(pf.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="ref\n", stderr="")
            pf._check_repo_ref("https://host/org/repo", "1.2.3")
        args = run.call_args[0][0]
        self.assertIn("refs/tags/1.2.3", args)
        self.assertIn("refs/heads/1.2.3", args)

    def test_sha_ref_checks_the_repo_only(self) -> None:
        # ls-remote cannot list a bare commit, so only reachability is checked.
        with mock.patch.object(pf.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="ref\n", stderr="")
            pf._check_repo_ref("https://host/org/repo", "d67f0fc")
        args = run.call_args[0][0]
        self.assertFalse(any(a.startswith("refs/") for a in args))

    def test_prompting_is_disabled(self) -> None:
        # An unattended pre-flight must fail rather than wait on a password prompt.
        with mock.patch.object(pf.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="ref\n", stderr="")
            pf._check_repo_ref("https://host/org/repo", "main")
        self.assertEqual(run.call_args[1]["env"]["GIT_TERMINAL_PROMPT"], "0")

    def test_missing_named_ref_is_reported(self) -> None:
        error = subprocess.CalledProcessError(2, ["git"], output="", stderr="")
        with mock.patch.object(pf.subprocess, "run", side_effect=error):
            problem = pf._check_repo_ref("https://host/org/repo", "9.9.9")
        self.assertIn("no branch or tag named '9.9.9'", problem)

    def test_auth_failure_is_reported(self) -> None:
        error = subprocess.CalledProcessError(
            128,
            ["git"],
            output="",
            stderr="fatal: Authentication failed for 'https://x'",
        )
        with mock.patch.object(pf.subprocess, "run", side_effect=error):
            problem = pf._check_repo_ref("https://host/org/repo", "main")
        self.assertIn("rejected", problem)

    def test_timeout_is_reported(self) -> None:
        with mock.patch.object(
            pf.subprocess, "run", side_effect=subprocess.TimeoutExpired(["git"], 60)
        ):
            problem = pf._check_repo_ref("https://host/org/repo", "main")
        self.assertIn("timed out", problem)

    def test_missing_git_is_reported(self) -> None:
        with mock.patch.object(pf.subprocess, "run", side_effect=FileNotFoundError):
            problem = pf._check_repo_ref("https://host/org/repo", "main")
        self.assertIn("git is not installed", problem)


class RunCheckReposTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = _SCRIPTS_DIR.parent / "dataset" / "_preflight_repos_test.yaml"
        self.ds.write_text(_DATASET, encoding="utf-8")

    def tearDown(self) -> None:
        self.ds.unlink(missing_ok=True)

    def test_duplicate_repo_ref_checked_once(self) -> None:
        # Both tasks share one repo and ref, so twenty tasks over two repos must
        # cost two network calls, not twenty.
        with mock.patch.object(pf, "_check_repo_ref", return_value=None) as check:
            code = pf._run_check_repos(str(self.ds), None)
        self.assertEqual(code, 0)
        self.assertEqual(check.call_count, 1)

    def test_unreachable_repo_exits_nonzero(self) -> None:
        with mock.patch.object(pf, "_check_repo_ref", return_value="nope"):
            code = pf._run_check_repos(str(self.ds), None)
        self.assertEqual(code, 1)

    def test_task_filter_is_applied(self) -> None:
        with mock.patch.object(pf, "_check_repo_ref", return_value=None) as check:
            pf._run_check_repos(str(self.ds), ["task-one"])
        self.assertEqual(check.call_count, 1)

    def test_unknown_task_id_raises(self) -> None:
        with self.assertRaises(pf.DatasetError):
            pf._run_check_repos(str(self.ds), ["no-such-task"])


if __name__ == "__main__":
    unittest.main()
