"""Tests for the agentic codex exec artifact judge."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import codex_judge  # noqa: E402
from judge_common import JudgeError  # noqa: E402


def _valid_result(task: str = "task-a", model: str = "candidate-a") -> dict[str, Any]:
    artifact = {
        "completeness": 10,
        "correctness": 10,
        "specificity": 10,
        "risk_awareness": 10,
        "total": 40,
        "notes": "Grounded but incomplete.",
    }
    return {
        "task": task,
        "model": model,
        "scores": {
            "github_issue": dict(artifact),
            "lld": dict(artifact),
            "review": dict(artifact),
            "testing": dict(artifact),
            "implementation": dict(artifact),
        },
        "task_score": 40.0,
        "verdict": "Useful, with material gaps.",
    }


def _artifact_folder(root: Path, *, with_metrics: bool = True) -> Path:
    # Layout is <model>/<repo>/<task>: leaf is the task, grandparent the model.
    folder = root / "candidate-a" / "repo-a" / "task-a"
    folder.mkdir(parents=True)
    for filename in ("github-issue.md", "lld.md", "review.md", "testing.md"):
        (folder / filename).write_text(
            f"# {filename}\n\nArtifact body.\n", encoding="utf-8"
        )
    if with_metrics:
        (folder / "metrics.json").write_text(
            json.dumps(
                {
                    "task": "task-a",
                    "model": "candidate-a",
                    "repo": "https://example.invalid/owner/repo",
                    "ref": "v1.2.3",
                    "input_tokens": 99,
                }
            ),
            encoding="utf-8",
        )
    return folder


class ResolveRepoRefTest(unittest.TestCase):
    def test_missing_metrics_fails_loudly(self) -> None:
        with self.assertRaisesRegex(JudgeError, "metrics.json is required"):
            codex_judge._resolve_repo_ref(None)

    def test_missing_repo_fails(self) -> None:
        with self.assertRaisesRegex(JudgeError, "missing a non-empty 'repo'"):
            codex_judge._resolve_repo_ref({"ref": "v1"})

    def test_missing_ref_fails(self) -> None:
        with self.assertRaisesRegex(JudgeError, "missing a non-empty 'ref'"):
            codex_judge._resolve_repo_ref({"repo": "https://example.invalid/r"})

    def test_returns_stripped_pair(self) -> None:
        repo, ref = codex_judge._resolve_repo_ref(
            {"repo": " https://example.invalid/r ", "ref": " main "}
        )
        self.assertEqual(repo, "https://example.invalid/r")
        self.assertEqual(ref, "main")


class CloneDirTest(unittest.TestCase):
    def test_is_deterministic_and_ref_sensitive(self) -> None:
        root = Path("/tmp/clones")
        a = codex_judge._clone_dir("https://x/owner/repo.git", "v1", root)
        b = codex_judge._clone_dir("https://x/owner/repo.git", "v1", root)
        c = codex_judge._clone_dir("https://x/owner/repo.git", "v2", root)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue(a.name.startswith("repo-"))


class CloneRepoAtRefTest(unittest.TestCase):
    def test_reuses_existing_checkout_without_cloning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = codex_judge._clone_dir("https://x/owner/repo", "v1", root)
            (target / ".git").mkdir(parents=True)

            def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
                return mock.Mock(returncode=0, stdout="deadbeef\n", stderr="")

            with mock.patch.object(codex_judge.subprocess, "run", fake_run) as _:
                with mock.patch.object(codex_judge, "_run_git") as run_git:
                    result = codex_judge.clone_repo_at_ref(
                        "https://x/owner/repo", "v1", clone_root=root
                    )
            self.assertEqual(result, target)
            run_git.assert_not_called()

    def test_clones_and_checks_out_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with mock.patch.object(codex_judge, "_run_git") as run_git:
                result = codex_judge.clone_repo_at_ref(
                    "https://x/owner/repo", "v1", clone_root=root
                )
            self.assertEqual(run_git.call_count, 2)
            clone_args = run_git.call_args_list[0].args[0]
            checkout_args = run_git.call_args_list[1].args[0]
            self.assertEqual(clone_args[0], "clone")
            self.assertIn("checkout", checkout_args)
            self.assertIn("v1", checkout_args)
            self.assertEqual(
                result, codex_judge._clone_dir("https://x/owner/repo", "v1", root)
            )

    def test_empty_repo_fails(self) -> None:
        with self.assertRaisesRegex(JudgeError, "repo URL is empty"):
            codex_judge.clone_repo_at_ref("", "v1")

    def test_empty_ref_fails(self) -> None:
        with self.assertRaisesRegex(JudgeError, "ref is empty"):
            codex_judge.clone_repo_at_ref("https://x/r", "")


class EvaluateWithCodexTest(unittest.TestCase):
    def test_clones_and_runs_codex_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = _artifact_folder(Path(temp_dir))
            fake_clone = Path(temp_dir) / "clone"
            fake_clone.mkdir()

            captured: dict[str, Any] = {}

            def fake_clone_fn(repo, ref, **kwargs):  # noqa: ANN001, ANN003
                captured["repo"] = repo
                captured["ref"] = ref
                return fake_clone

            run_metrics = {
                "token_usage": {"input_tokens": 100, "output_tokens": 20},
                "duration_ms": 12345,
            }

            def fake_run_codex(prompt, *, working_root, **kwargs):  # noqa: ANN001, ANN003
                captured["working_root"] = working_root
                return json.dumps(_valid_result()), dict(run_metrics)

            with mock.patch.object(codex_judge, "clone_repo_at_ref", fake_clone_fn):
                with mock.patch.object(codex_judge, "_run_codex", fake_run_codex):
                    result = codex_judge.evaluate_artifact_folder_with_codex(
                        folder, reasoning_effort="high"
                    )

            eval_data = json.loads((folder / "eval.json").read_text(encoding="utf-8"))
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(captured["repo"], "https://example.invalid/owner/repo")
        self.assertEqual(captured["ref"], "v1.2.3")
        self.assertEqual(captured["working_root"], fake_clone)
        self.assertEqual(result["judge"]["provider"], "codex-exec")
        self.assertTrue(result["judge"]["repo_grounded"])
        self.assertEqual(result["judge"]["repo_ref"], "v1.2.3")
        # The local working checkout path is deliberately NOT recorded (eval.json /
        # metrics.json are committed; a /tmp path is machine-specific noise).
        self.assertNotIn("repo_root", result["judge"])
        self.assertEqual(result["judge"]["reasoning_effort"], "high")
        self.assertEqual(result["judge"]["duration_ms"], 12345)
        self.assertEqual(result["judge"]["token_usage"]["input_tokens"], 100)
        self.assertEqual(eval_data, result)
        self.assertEqual(metrics["evaluation"], result)
        self.assertEqual(metrics["input_tokens"], 99)

    def test_missing_artifact_scores_zero_without_running_codex(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = _artifact_folder(Path(temp_dir))
            # Simulate a model failure: one required artifact was never written.
            (folder / "github-issue.md").unlink()
            with mock.patch.object(codex_judge, "_run_codex") as run_codex:
                with mock.patch.object(codex_judge, "clone_repo_at_ref") as clone_fn:
                    result = codex_judge.evaluate_artifact_folder_with_codex(folder)
            run_codex.assert_not_called()
            clone_fn.assert_not_called()
            self.assertEqual(result["task_score"], 0.0)
            self.assertIn("MODEL FAILURE", result["verdict"])
            self.assertIn("github-issue.md", result["verdict"])
            self.assertEqual(
                result["judge"]["scored_zero_missing_artifacts"], ["github-issue.md"]
            )
            self.assertFalse(result["judge"]["repo_grounded"])
            # eval.json written and mirrored into metrics.json.
            eval_data = json.loads((folder / "eval.json").read_text(encoding="utf-8"))
            self.assertEqual(eval_data["task_score"], 0.0)
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["evaluation"]["task_score"], 0.0)

    def test_missing_metrics_fails_before_running_codex(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = _artifact_folder(Path(temp_dir), with_metrics=False)
            with mock.patch.object(codex_judge, "_run_codex") as run_codex:
                with self.assertRaisesRegex(JudgeError, "metrics.json is required"):
                    codex_judge.evaluate_artifact_folder_with_codex(folder)
            run_codex.assert_not_called()

    def test_explicit_repo_skips_clone(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = _artifact_folder(Path(temp_dir), with_metrics=False)
            local_repo = Path(temp_dir) / "local"
            local_repo.mkdir()

            def fake_run_codex(prompt, *, working_root, **kwargs):  # noqa: ANN001, ANN003
                return json.dumps(_valid_result()), {}

            with mock.patch.object(codex_judge, "clone_repo_at_ref") as clone_fn:
                with mock.patch.object(codex_judge, "_run_codex", fake_run_codex):
                    result = codex_judge.evaluate_artifact_folder_with_codex(
                        folder, repo=local_repo
                    )
            clone_fn.assert_not_called()
            # Local working path not recorded (committed files stay path-free).
            self.assertNotIn("repo_root", result["judge"])
            self.assertNotIn("repo_ref", result["judge"])

    def test_invalid_arithmetic_does_not_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = _artifact_folder(Path(temp_dir))
            original_metrics = (folder / "metrics.json").read_text(encoding="utf-8")
            invalid = _valid_result()
            invalid["scores"]["lld"]["total"] = 41
            fake_clone = Path(temp_dir) / "clone"
            fake_clone.mkdir()

            with mock.patch.object(
                codex_judge, "clone_repo_at_ref", return_value=fake_clone
            ):
                with mock.patch.object(
                    codex_judge, "_run_codex", return_value=(json.dumps(invalid), {})
                ):
                    with self.assertRaisesRegex(JudgeError, "invalid evaluation"):
                        codex_judge.evaluate_artifact_folder_with_codex(folder)

            self.assertFalse((folder / "eval.json").exists())
            self.assertEqual(
                (folder / "metrics.json").read_text(encoding="utf-8"),
                original_metrics,
            )


def _extra_folder(root: Path, task: str, model: str) -> Path:
    """Create a second artifact folder with its own metrics.json under root."""
    folder = root / model / "repo-a" / task
    folder.mkdir(parents=True)
    for filename in ("github-issue.md", "lld.md", "review.md", "testing.md"):
        (folder / filename).write_text(f"# {filename}\n\nBody.\n", encoding="utf-8")
    (folder / "metrics.json").write_text(
        json.dumps(
            {
                "task": task,
                "model": model,
                "repo": "https://example.invalid/owner/repo",
                "ref": "v1.2.3",
            }
        ),
        encoding="utf-8",
    )
    return folder


class DiscoverArtifactFoldersTest(unittest.TestCase):
    def test_finds_every_folder_with_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            a = _artifact_folder(root)
            b = _extra_folder(root, "task-b", "candidate-b")
            found = codex_judge._discover_artifact_folders(root)
            self.assertEqual(found, sorted([a.resolve(), b.resolve()]))

    def test_ignores_folders_without_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _artifact_folder(root, with_metrics=False)
            found = codex_judge._discover_artifact_folders(root)
            self.assertEqual(found, [])

    def test_single_folder_passed_directly_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = _artifact_folder(Path(temp_dir))
            found = codex_judge._discover_artifact_folders(folder)
            self.assertEqual(found, [folder.resolve()])

    def test_missing_root_fails(self) -> None:
        with self.assertRaisesRegex(JudgeError, "not a directory"):
            codex_judge._discover_artifact_folders("/no/such/dir/here")


class EvaluateTreeWithCodexTest(unittest.TestCase):
    def test_judges_every_discovered_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _artifact_folder(root)
            _extra_folder(root, "task-b", "candidate-b")

            judged: list[Path] = []

            def fake_eval(folder, **kwargs):  # noqa: ANN001, ANN003
                judged.append(Path(folder))
                return _valid_result()

            with mock.patch.object(
                codex_judge, "evaluate_artifact_folder_with_codex", fake_eval
            ):
                results = codex_judge.evaluate_tree_with_codex(root)

            self.assertEqual(len(results), 2)
            self.assertEqual(len(judged), 2)

    def test_one_bad_folder_does_not_abort_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            good = _artifact_folder(root)
            bad = _extra_folder(root, "task-b", "candidate-b")

            def fake_eval(folder, **kwargs):  # noqa: ANN001, ANN003
                if Path(folder) == bad.resolve() or Path(folder) == bad:
                    raise JudgeError("boom")
                return _valid_result()

            with mock.patch.object(
                codex_judge, "evaluate_artifact_folder_with_codex", fake_eval
            ):
                results = codex_judge.evaluate_tree_with_codex(root)

            self.assertEqual(list(results), [str(good.resolve())])

    def test_no_folders_found_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(JudgeError, "no artifact folders found"):
                codex_judge.evaluate_tree_with_codex(temp_dir)

    def test_no_overwrite_skips_folders_with_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            done = _artifact_folder(root)
            (done / "eval.json").write_text("{}", encoding="utf-8")
            pending = _extra_folder(root, "task-b", "candidate-b")

            judged: list[Path] = []

            def fake_eval(folder, **kwargs):  # noqa: ANN001, ANN003
                judged.append(Path(folder))
                return _valid_result()

            with mock.patch.object(
                codex_judge, "evaluate_artifact_folder_with_codex", fake_eval
            ):
                results = codex_judge.evaluate_tree_with_codex(root, overwrite=False)

            self.assertEqual(judged, [pending.resolve()])
            self.assertEqual(list(results), [str(pending.resolve())])


class BuildCodexCmdTest(unittest.TestCase):
    def test_reads_prompt_from_stdin_and_includes_working_root(self) -> None:
        cmd = codex_judge._build_codex_cmd(
            codex_bin="codex",
            working_root=Path("/repo"),
            output_file=Path("/tmp/out.txt"),
            model="gpt-x",
            reasoning_effort="high",
            sandbox="read-only",
            output_schema_file=None,
        )
        self.assertEqual(cmd[-1], "-")
        self.assertIn("--json", cmd)
        self.assertIn("--cd", cmd)
        self.assertIn("/repo", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("gpt-x", cmd)
        self.assertIn("model_reasoning_effort=high", cmd)


class ParseCodexEventsTest(unittest.TestCase):
    def test_extracts_token_usage_from_turn_completed(self) -> None:
        stdout = "\n".join(
            [
                '{"type":"thread.started","thread_id":"abc"}',
                '{"type":"turn.started"}',
                '{"type":"item.completed","item":{"type":"agent_message"}}',
                '{"type":"turn.completed","usage":{"input_tokens":8031,'
                '"cached_input_tokens":10,"output_tokens":5}}',
            ]
        )
        metrics = codex_judge._parse_codex_events(stdout)
        self.assertEqual(metrics["token_usage"]["input_tokens"], 8031)
        self.assertEqual(metrics["token_usage"]["output_tokens"], 5)

    def test_accepts_rollout_style_token_count(self) -> None:
        stdout = (
            '{"payload":{"type":"token_count","info":{'
            '"total_token_usage":{"total_tokens":1050},'
            '"model_context_window":258400}}}'
        )
        metrics = codex_judge._parse_codex_events(stdout)
        self.assertEqual(metrics["token_usage"]["total_tokens"], 1050)
        self.assertEqual(metrics["context_window"], 258400)

    def test_last_usage_wins(self) -> None:
        stdout = "\n".join(
            [
                '{"type":"turn.completed","usage":{"output_tokens":100}}',
                '{"type":"turn.completed","usage":{"output_tokens":200}}',
            ]
        )
        metrics = codex_judge._parse_codex_events(stdout)
        self.assertEqual(metrics["token_usage"]["output_tokens"], 200)

    def test_malformed_lines_are_ignored(self) -> None:
        stdout = "not json\n\n{broken\n{}\n"
        self.assertEqual(codex_judge._parse_codex_events(stdout), {})


if __name__ == "__main__":
    unittest.main()
