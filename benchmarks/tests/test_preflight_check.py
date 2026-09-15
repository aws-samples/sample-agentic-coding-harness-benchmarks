"""Tests for the end-to-end benchmark pre-flight helper."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
