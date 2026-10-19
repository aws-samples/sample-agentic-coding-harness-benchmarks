"""Tests for the Strands runner's SDK-free helpers and argument checks.

The SDK is imported lazily inside the runner, so these run without the
optional ``strands`` dependency group installed.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import strands_agent_runner as runner  # noqa: E402


class UsageDictTest(unittest.TestCase):
    def test_missing_keys_default_to_zero(self) -> None:
        usage = runner._usage_dict({"inputTokens": 5})
        self.assertEqual(
            usage,
            {
                "inputTokens": 5,
                "outputTokens": 0,
                "totalTokens": 0,
                "cacheReadInputTokens": 0,
                "cacheWriteInputTokens": 0,
            },
        )

    def test_none_usage_is_all_zero(self) -> None:
        self.assertEqual(sum(runner._usage_dict(None).values()), 0)


class PreviewTest(unittest.TestCase):
    def test_long_value_is_truncated_to_one_line(self) -> None:
        text = runner._preview("a\n" * 500, limit=10)
        self.assertEqual(text, "a a a a a ...")


class ParseArgsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.skill_dir = Path(self._tmp.name)
        (self.skill_dir / "SKILL.md").write_text("---\nname: x\n---\n")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _args(self, *extra: str) -> list[str]:
        return [
            "--model",
            "m",
            "--skill-dir",
            str(self.skill_dir),
            "--prompt",
            "p",
            *extra,
        ]

    def test_bedrock_with_region_parses(self) -> None:
        args = runner._parse_args(
            self._args("--provider", "bedrock", "--aws-region", "us-east-1")
        )
        self.assertEqual(args.aws_region, "us-east-1")

    def test_endpoint_without_url_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            runner._parse_args(self._args("--provider", "endpoint"))

    def test_missing_skill_md_is_rejected(self) -> None:
        (self.skill_dir / "SKILL.md").unlink()
        with self.assertRaises(SystemExit):
            runner._parse_args(
                self._args("--provider", "endpoint", "--endpoint", "http://x")
            )


class _FakeOpenAIModel:
    """Stands in for strands.models.openai.OpenAIModel without the SDK."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def format_request(self, *args: object, **kwargs: object) -> dict:
        return {"model": "m", "tools": []}


class BuildEndpointModelTest(unittest.TestCase):
    """The endpoint branch, with the SDK module stubbed so no install is needed."""

    def _build(self, context_window: int) -> object:
        openai_module = mock.MagicMock()
        openai_module.OpenAIModel = _FakeOpenAIModel
        args = argparse.Namespace(
            provider="endpoint",
            endpoint="http://127.0.0.1:8000/",
            model="m",
            max_tokens=100,
            context_window=context_window,
        )
        with mock.patch.dict(sys.modules, {"strands.models.openai": openai_module}):
            return runner._build_model(args)

    def test_known_window_leaves_room_for_the_output(self) -> None:
        # vLLM counts max_tokens (100 here) against the window.
        model = self._build(context_window=131072)
        self.assertEqual(model.kwargs["context_window_limit"], 131072 - 100)

    def test_unknown_window_leaves_sdk_default(self) -> None:
        model = self._build(context_window=0)
        self.assertNotIn("context_window_limit", model.kwargs)

    def test_request_without_tools_omits_tools_key(self) -> None:
        model = self._build(context_window=0)
        self.assertNotIn("tools", model.format_request([]))


class RaiseRecursionLimitTest(unittest.TestCase):
    """Strands recurses per tool turn, so the runner raises the limit (#208)."""

    def setUp(self) -> None:
        self._original = sys.getrecursionlimit()

    def tearDown(self) -> None:
        sys.setrecursionlimit(self._original)

    def test_low_limit_is_raised(self) -> None:
        sys.setrecursionlimit(1000)
        runner._raise_recursion_limit(5000)
        self.assertEqual(sys.getrecursionlimit(), 5000)

    def test_default_limit_covers_the_turn_budget(self) -> None:
        needed = runner.MAX_TOOL_TURNS * runner.FRAMES_PER_TOOL_TURN
        self.assertGreater(runner.RECURSION_LIMIT, needed)

    def test_higher_limit_is_never_lowered(self) -> None:
        sys.setrecursionlimit(20000)
        runner._raise_recursion_limit(5000)
        self.assertEqual(sys.getrecursionlimit(), 20000)


class HarnessModeTest(unittest.TestCase):
    """--harness builds the agent with create_harness() (agent=strands-harness)."""

    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            provider="endpoint",
            endpoint="http://127.0.0.1:8000",
            model="m",
            max_tokens=100,
            context_window=0,
            harness=True,
        )

    def _build(self) -> dict:
        harness_module = mock.MagicMock()
        tools_module = mock.MagicMock()
        tools_module.make_shell = lambda **kw: ("SHELL", kw["name"])
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "strands_harness": harness_module,
                    "strands.vended_tools": tools_module,
                },
            ),
            mock.patch.object(runner, "_build_model", return_value="MODEL"),
        ):
            runner._build_agent(self._args(), [runner._EventStreamHooks()])
        return harness_module.create_harness.call_args.kwargs

    def test_uses_our_model_instance(self) -> None:
        self.assertEqual(self._build()["model"], "MODEL")

    def test_only_the_coding_tools_are_enabled(self) -> None:
        self.assertEqual(
            self._build()["builtin_tools"], ["shell", "read", "write", "edit"]
        )

    def test_bash_is_added_as_a_shell_alias(self) -> None:
        self.assertEqual(self._build()["tools"], [("SHELL", "bash")])

    def test_memory_and_session_are_off(self) -> None:
        kwargs = self._build()
        self.assertEqual((kwargs["memory"], kwargs["session"]), (False, False))

    def test_streamed_text_stays_off_stdout(self) -> None:
        self.assertIsNone(self._build()["callback_handler"])

    def test_harness_instructions_map_the_skill_tool_names(self) -> None:
        for name in ("Read", "Edit", "Write", "Bash", "Grep", "Glob", "Task"):
            with self.subTest(tool=name):
                self.assertIn(name, runner.HARNESS_INSTRUCTIONS)

    def test_tool_map_is_in_the_instructions(self) -> None:
        self.assertIn(runner.HARNESS_TOOL_MAP, runner.HARNESS_INSTRUCTIONS)

    def test_tool_map_names_bash_as_an_alias(self) -> None:
        self.assertIn("bash is an alias of shell", runner.HARNESS_TOOL_MAP)

    def test_harness_flag_parses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "SKILL.md").write_text("x", encoding="utf-8")
            args = runner._parse_args(
                [
                    "--provider",
                    "endpoint",
                    "--endpoint",
                    "http://x",
                    "--model",
                    "m",
                    "--skill-dir",
                    tmp,
                    "--prompt",
                    "p",
                    "--harness",
                ]
            )
        self.assertTrue(args.harness)


class CapToolResultSwitchTest(unittest.TestCase):
    def test_hooks_leave_results_alone_when_the_cap_is_off(self) -> None:
        event = mock.MagicMock()
        event.tool_use = {"name": "read", "input": {}}
        event.result = {
            "status": "success",
            "toolUseId": "t",
            "content": [{"text": "x" * 200_000}],
        }
        event.duration = 0.1
        original = event.result
        with mock.patch.object(runner, "_emit"):
            runner._EventStreamHooks(cap_results=False)._on_tool_call(event)
        self.assertIs(event.result, original)


class LoopGuardTest(unittest.TestCase):
    """The loop guard cancels, then stops, a repeated tool call."""

    def _call(
        self, guard: runner._LoopGuard, command: str = "git diff"
    ) -> mock.MagicMock:
        event = mock.MagicMock()
        event.tool_use = {"name": "bash", "input": {"command": command}}
        event.cancel_tool = False
        with mock.patch.object(runner, "_emit"):
            guard._before_tool_call(event)
        return event

    def test_calls_up_to_the_warning_count_run(self) -> None:
        guard = runner._LoopGuard(warn=3, stop=5)
        events = [self._call(guard) for _ in range(3)]
        self.assertEqual([e.cancel_tool for e in events], [False, False, False])

    def test_a_call_past_the_warning_count_is_cancelled(self) -> None:
        guard = runner._LoopGuard(warn=3, stop=5)
        for _ in range(3):
            self._call(guard)
        self.assertIn("Loop guard", self._call(guard).cancel_tool)

    def test_different_inputs_are_counted_separately(self) -> None:
        guard = runner._LoopGuard(warn=1, stop=5)
        self._call(guard, "git diff")
        self.assertFalse(self._call(guard, "git status").cancel_tool)

    def test_the_stop_count_ends_the_turn(self) -> None:
        guard = runner._LoopGuard(warn=3, stop=5)
        for _ in range(5):
            self._call(guard)
        after = mock.MagicMock()
        after.end_turn = False
        guard._after_tools(after)
        self.assertIn("Stopped by the loop guard", after.end_turn)

    def test_no_stop_leaves_the_turn_open(self) -> None:
        guard = runner._LoopGuard(warn=3, stop=5)
        self._call(guard)
        after = mock.MagicMock()
        after.end_turn = False
        guard._after_tools(after)
        self.assertFalse(after.end_turn)

    def test_thresholds_clear_every_healthy_task_measured(self) -> None:
        # No healthy minicpm5-2b task repeated an identical call over 16 times.
        self.assertGreater(runner.LOOP_WARN_REPEATS, 16)


class SystemPromptToolMapTest(unittest.TestCase):
    def test_every_tool_the_skill_names_is_mapped(self) -> None:
        # swe3/SKILL.md names these Claude Code tools (issue #202).
        for name in ("Read", "Edit", "Write", "Bash", "Grep", "Glob", "Task"):
            with self.subTest(tool=name):
                self.assertIn(f"{name}", runner.SYSTEM_PROMPT)

    def test_map_names_only_the_two_real_tools(self) -> None:
        self.assertIn("exactly two tools: file_editor and shell", runner.SYSTEM_PROMPT)


class PromptWithSkillTest(unittest.TestCase):
    def test_skill_text_comes_before_the_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            (skill_dir / "SKILL.md").write_text("SKILL BODY", encoding="utf-8")
            prompt = runner._prompt_with_skill(skill_dir, "TASK")
        self.assertEqual(
            prompt,
            "SKILL BODY\n\n---\n\n"
            "Follow the skill instructions above to complete the following task.\n\n"
            "TASK",
        )


class DropEmptyToolsTest(unittest.TestCase):
    def test_empty_list_is_removed(self) -> None:
        self.assertEqual(
            runner._drop_empty_tools({"tools": [], "model": "m"}), {"model": "m"}
        )

    def test_non_empty_list_is_kept(self) -> None:
        request = {"tools": [{"type": "function"}]}
        self.assertEqual(runner._drop_empty_tools(request), request)


class CapToolResultTest(unittest.TestCase):
    def _result(self, *blocks: dict) -> dict:
        return {"toolUseId": "t1", "status": "success", "content": list(blocks)}

    def test_small_result_is_unchanged(self) -> None:
        result = self._result({"text": "abc"})
        self.assertEqual(runner._cap_tool_result(result, limit=10), (result, 0))

    def test_large_text_is_cut_to_the_limit(self) -> None:
        capped, total = runner._cap_tool_result(
            self._result({"text": "x" * 50}), limit=10
        )
        self.assertTrue(
            capped["content"][0]["text"].startswith("x" * 10 + "\n\n[Output truncated")
        )
        self.assertEqual(total, 50)

    def test_large_json_block_is_counted_and_cut(self) -> None:
        result = self._result({"json": {"output": "y" * 50, "exit_code": 0}})
        capped, total = runner._cap_tool_result(result, limit=10)
        self.assertGreater(total, 50)
        self.assertEqual(capped["status"], "success")
        self.assertEqual(capped["toolUseId"], "t1")


if __name__ == "__main__":
    unittest.main()
