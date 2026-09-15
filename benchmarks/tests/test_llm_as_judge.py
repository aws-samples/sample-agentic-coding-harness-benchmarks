"""Tests for the one-shot Bedrock artifact judge."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from llm_as_judge import (  # noqa: E402
    JudgeError,
    evaluate_artifact_folder,
    render_judge_prompt,
)


class _FakeResponse:
    def __init__(self, result: dict[str, Any]) -> None:
        self._payload = {
            "id": "response-1",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(result)}],
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, result: dict[str, Any]) -> None:
        self.response = _FakeResponse(result)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.response


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
    folder = root / "task-a" / "candidate-a"
    folder.mkdir(parents=True)
    for filename in ("github-issue.md", "lld.md", "review.md", "testing.md"):
        (folder / filename).write_text(
            f"# {filename}\n\nArtifact with $variables and </submission> text.\n",
            encoding="utf-8",
        )
    if with_metrics:
        (folder / "metrics.json").write_text(
            json.dumps(
                {
                    "task": "task-a",
                    "model": "candidate-a",
                    "repo": "https://example.invalid/repo",
                    "ref": "abc123",
                    "input_tokens": 99,
                }
            ),
            encoding="utf-8",
        )
    return folder


class RenderJudgePromptTest(unittest.TestCase):
    def test_renders_all_json_escaped_artifact_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = _artifact_folder(Path(temp_dir))
            prompt, task, candidate, _ = render_judge_prompt(folder)

        self.assertEqual(task, "task-a")
        self.assertEqual(candidate, "candidate-a")
        self.assertIn('"task_id": "task-a"', prompt)
        self.assertIn("github-issue.md", prompt)
        self.assertIn("$variables", prompt)
        self.assertNotIn("$GITHUB_ISSUE_JSON", prompt)

    def test_missing_artifact_fails_before_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = _artifact_folder(Path(temp_dir))
            (folder / "testing.md").unlink()
            with self.assertRaisesRegex(JudgeError, "missing testing.md"):
                render_judge_prompt(folder)


class EvaluateArtifactFolderTest(unittest.TestCase):
    def test_one_request_writes_eval_and_merges_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = _artifact_folder(Path(temp_dir))
            session = _FakeSession(_valid_result())
            result = evaluate_artifact_folder(
                folder,
                "judge-model",
                base_url="https://bedrock.example/openai/v1",
                api_key="test-token",
                reasoning_effort="medium",
                session=session,
            )
            eval_data = json.loads((folder / "eval.json").read_text(encoding="utf-8"))
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(len(session.calls), 1)
        self.assertEqual(
            session.calls[0]["url"], "https://bedrock.example/openai/v1/responses"
        )
        self.assertEqual(session.calls[0]["json"]["model"], "judge-model")
        self.assertEqual(session.calls[0]["json"]["reasoning"], {"effort": "medium"})
        self.assertFalse(session.calls[0]["json"]["store"])
        self.assertEqual(
            session.calls[0]["json"]["text"]["format"]["type"], "json_schema"
        )
        self.assertEqual(result["judge"]["model"], "judge-model")
        self.assertEqual(eval_data, result)
        self.assertEqual(metrics["evaluation"], result)
        self.assertEqual(metrics["input_tokens"], 99)

    def test_invalid_arithmetic_does_not_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = _artifact_folder(Path(temp_dir))
            original_metrics = (folder / "metrics.json").read_text(encoding="utf-8")
            invalid = _valid_result()
            invalid["scores"]["lld"]["total"] = 41
            session = _FakeSession(invalid)

            with self.assertRaisesRegex(JudgeError, "invalid evaluation"):
                evaluate_artifact_folder(
                    folder,
                    "judge-model",
                    base_url="https://bedrock.example/openai/v1",
                    api_key="test-token",
                    session=session,
                )

            self.assertFalse((folder / "eval.json").exists())
            self.assertEqual(
                (folder / "metrics.json").read_text(encoding="utf-8"),
                original_metrics,
            )


if __name__ == "__main__":
    unittest.main()
