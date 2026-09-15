"""Tests for the SWE benchmark dataset loader."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# The benchmark scripts are not a package; add the scripts dir to the path so
# dataset_loader (underscore name, importable) can be imported by module name.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from dataset_loader import DatasetError, load_dataset  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SHIPPED_DATASET = _REPO_ROOT / "benchmarks" / "dataset" / "mcp-gateway-registry.yaml"

_MINIMAL = """\
schema_version: "1.0"
name: tiny
title: Tiny dataset
description: A minimal valid dataset.
default_ref: main
metrics: [input_tokens, output_tokens, num_turns]
complexity_levels: [low, medium, high]
tasks:
  - id: only-task
    repo: https://github.com/example/repo
    complexity: low
    tags: [demo]
    problem_statement: |
      Do the thing.
"""


def _write(text: str) -> Path:
    """Write dataset text to a temp file and return its path."""
    temp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    temp.write(text)
    temp.close()
    return Path(temp.name)


class LoadDatasetTest(unittest.TestCase):
    def test_shipped_dataset_loads(self) -> None:
        dataset = load_dataset(_SHIPPED_DATASET)
        self.assertEqual(dataset.name, "mcp-gateway-registry-swe")
        self.assertEqual(len(dataset.tasks), 5)
        self.assertIn("num_turns", dataset.metrics)

    def test_ref_defaults_to_dataset_default(self) -> None:
        dataset = load_dataset(_write(_MINIMAL))
        self.assertEqual(dataset.task_by_id("only-task").ref, "main")

    def test_task_ref_overrides_default(self) -> None:
        text = _MINIMAL.replace(
            "    complexity: low", '    ref: "1.2.3"\n    complexity: low'
        )
        dataset = load_dataset(_write(text))
        self.assertEqual(dataset.task_by_id("only-task").ref, "1.2.3")

    def test_missing_file_raises(self) -> None:
        with self.assertRaisesRegex(DatasetError, "not found"):
            load_dataset("/nonexistent/dataset.yaml")

    def test_unsupported_schema_version_raises(self) -> None:
        text = _MINIMAL.replace('schema_version: "1.0"', 'schema_version: "9.9"')
        with self.assertRaisesRegex(DatasetError, "unsupported schema_version"):
            load_dataset(_write(text))

    def test_bad_complexity_raises(self) -> None:
        text = _MINIMAL.replace("    complexity: low", "    complexity: extreme")
        with self.assertRaisesRegex(DatasetError, "complexity 'extreme'"):
            load_dataset(_write(text))

    def test_missing_problem_source_raises(self) -> None:
        text = _MINIMAL.replace("    problem_statement: |\n      Do the thing.\n", "")
        with self.assertRaisesRegex(DatasetError, "at least one of"):
            load_dataset(_write(text))

    def test_issue_url_alone_is_valid(self) -> None:
        text = _MINIMAL.replace(
            "    problem_statement: |\n      Do the thing.\n",
            "    problem_issue_url: https://github.com/example/repo/issues/1\n",
        )
        dataset = load_dataset(_write(text))
        task = dataset.task_by_id("only-task")
        self.assertIsNone(task.problem_statement)
        self.assertTrue(task.problem_issue_url)

    def test_duplicate_task_id_raises(self) -> None:
        text = (
            _MINIMAL
            + """\
  - id: only-task
    repo: https://github.com/example/repo
    complexity: high
    tags: [dupe]
    problem_statement: duplicate id
"""
        )
        with self.assertRaisesRegex(DatasetError, "duplicate task id"):
            load_dataset(_write(text))

    def test_ground_truth_is_optional_and_parsed(self) -> None:
        dataset = load_dataset(_SHIPPED_DATASET)
        faiss = dataset.task_by_id("remove-faiss")
        self.assertIsNotNone(faiss.ground_truth)
        self.assertTrue(faiss.ground_truth.expectations)
        # Minimal dataset omits ground_truth entirely.
        minimal = load_dataset(_write(_MINIMAL))
        self.assertIsNone(minimal.task_by_id("only-task").ground_truth)


class OutputScopeTest(unittest.TestCase):
    """output_scope names the results folder when the repo name is not enough."""

    def test_defaults_to_the_repo_name(self) -> None:
        dataset = load_dataset(_write(_MINIMAL))
        self.assertIsNone(dataset.output_scope)
        self.assertEqual(dataset.scope_for("repo"), "repo")

    def test_overrides_the_repo_name_when_set(self) -> None:
        text = _MINIMAL.replace(
            "default_ref: main\n", "default_ref: main\noutput_scope: repo-v2\n"
        )
        dataset = load_dataset(_write(text))
        self.assertEqual(dataset.scope_for("repo"), "repo-v2")

    def test_shipped_datasets_keep_the_repo_name(self) -> None:
        # v1 must not move: its results are committed and feed the charts.
        v1 = load_dataset(_SHIPPED_DATASET)
        self.assertEqual(v1.scope_for("mcp-gateway-registry"), "mcp-gateway-registry")

    def test_v2_gets_its_own_scope(self) -> None:
        v2 = load_dataset(_SHIPPED_DATASET.with_name("mcp-gateway-registry-v2.yaml"))
        self.assertEqual(
            v2.scope_for("mcp-gateway-registry"), "mcp-gateway-registry-v2"
        )

    def test_rejects_a_path_instead_of_a_folder_name(self) -> None:
        text = _MINIMAL.replace(
            "default_ref: main\n", "default_ref: main\noutput_scope: a/b\n"
        )
        with self.assertRaisesRegex(DatasetError, "single folder name"):
            load_dataset(_write(text))


class V2DatasetTest(unittest.TestCase):
    """The v2 dataset's distinguishing properties, asserted rather than assumed."""

    def setUp(self) -> None:
        self.dataset = load_dataset(
            _SHIPPED_DATASET.with_name("mcp-gateway-registry-v2.yaml")
        )

    def test_every_tier_is_populated(self) -> None:
        # The tier counts are deliberately NOT asserted exactly: tasks get
        # re-tiered when a run shows a label was wrong (build-docker-images was
        # moved low -> medium after three models scored it worst-in-tier), and a
        # hard-coded 5/5/5 would turn a considered correction into a test break.
        # What must hold is that all four tiers exist and each has enough tasks
        # to mean something.
        counts: dict[str, int] = {}
        for task in self.dataset.tasks:
            counts[task.complexity] = counts.get(task.complexity, 0) + 1
        self.assertEqual(set(counts), {"trivial", "low", "medium", "high"})
        for tier, n in counts.items():
            self.assertGreaterEqual(n, 4, f"tier '{tier}' has only {n} tasks")

    def test_every_task_pins_its_own_ref(self) -> None:
        # The point of v2: each task clones the release before its fix, so the
        # defect is present. A task falling back to default_ref is a mistake.
        for task in self.dataset.tasks:
            self.assertIsNotNone(task.ref, f"task '{task.id}' has no explicit ref")
        refs = {self.dataset.resolved_ref(t) for t in self.dataset.tasks}
        self.assertGreater(len(refs), 1)

    def test_every_task_records_ground_truth(self) -> None:
        for task in self.dataset.tasks:
            self.assertIsNotNone(
                task.ground_truth, f"task '{task.id}' has no ground_truth"
            )


if __name__ == "__main__":
    unittest.main()
