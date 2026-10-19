#!/usr/bin/env python3
"""Run one benchmark task with a Strands Agents coding agent.

Strands Agents is a Python SDK with no CLI of its own, so this script plays the
part that ``pi -p`` or ``codex exec`` play for the other harnesses: the headless
runner (``run-swe-headless.py``) starts it as a subprocess, it drives one task to
completion, and it reports what happened as JSON lines on stdout.

By default the agent is a hand-built ``Agent`` with the stock Strands coding
tools (``shell`` and ``file_editor``). With ``--harness`` (agent=strands-harness)
it comes from the Strands team's ``create_harness()`` instead, with its own
``shell``/``read``/``write``/``edit`` tools and context manager. The SWE skill's ``SKILL.md`` goes ahead of the task prompt, worded as
``_build_omp_cmd`` words it for omp, so the rules sit in the pinned first message
and survive summarization (issue #201). The working directory is the cloned task
repo; the harness sets it when it starts this process.

stdout carries only JSON lines, one object per line:

- ``{"type": "model_call", ...}`` after every model response, with the running
  token usage so far. Strands rolls usage up after this hook fires, so the
  figure trails by one call; it exists so a killed run still leaves a trail.
- ``{"type": "tool_call", ...}`` after every tool call. ``truncated_from_chars``
  is the original size of a result cut to ``MAX_TOOL_RESULT_CHARS``, else 0.
- ``{"type": "loop_guard", ...}`` when the loop guard cancels a repeated call
  (``action`` ``warn``) or ends the attempt (``action`` ``stop``).
- ``{"type": "result", ...}`` once, at the end, with the final usage totals.

Logs go to stderr. The API key for an OpenAI-compatible endpoint is read from
the ``STRANDS_API_KEY`` environment variable, never from argv.

Example:
    uv run --group strands python scripts/strands_agent_runner.py \\
        --provider bedrock --model us.anthropic.claude-haiku-4-5-20251001-v1:0 \\
        --aws-region us-west-2 --skill-dir ../.claude/skills/swe3 \\
        --prompt "Use the swe3 skill to complete this task. ..."
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

PROVIDER_BEDROCK = "bedrock"
PROVIDER_ENDPOINT = "endpoint"
API_KEY_ENV = "STRANDS_API_KEY"
DEFAULT_API_KEY = "local"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_COMPACT_FRACTION = 0.9
# Characters of tool input and final message echoed into the event stream. Enough
# to follow a run in the log without copying whole files into it.
PREVIEW_CHARS = 300
FINAL_MESSAGE_CHARS = 4000
# Largest tool result the model sees, about 25K tokens. Strands' file_editor
# returns whole files up to 1 MB and its shell tool does not truncate, so one
# read of a large file can overflow the window before summarization can help
# (issue #199). Claude Code's Read tool stops at about the same size.
MAX_TOOL_RESULT_CHARS = 100_000
TRUNCATION_NOTE = (
    "\n\n[Output truncated: the tool returned {total:,} characters and only the "
    "first {kept:,} are shown. Read a smaller part instead: file_editor view with "
    "view_range, or grep, head or sed in the shell.]"
)
EXIT_OK = 0
EXIT_AGENT_ERROR = 1
# Strands' event loop calls itself after every round of tool calls, adding 3
# frames per turn, so Python's default limit of 1,000 frames ends a task after
# about 320 turns (issue #208). Size the limit for MAX_TOOL_TURNS, plus headroom
# for the frames below the loop; a task that runs past it still stops with a
# RecursionError. Our longest finished task took 222 turns; one looping
# attempt reached 316, just under the old ceiling.
MAX_TOOL_TURNS = 1000
FRAMES_PER_TOOL_TURN = 3
RECURSION_HEADROOM = 500
RECURSION_LIMIT = MAX_TOOL_TURNS * FRAMES_PER_TOOL_TURN + RECURSION_HEADROOM
# Loop guard: a small model can repeat one tool call hundreds of times (674
# identical `git diff --stat` calls in one task). Across 42 task attempts on
# minicpm5-2b, no healthy task repeated an identical call more than 16 times;
# the three loops reached 29, 542 and 674. From the warning count on, the
# repeated call is cancelled and the model is told to move on; at the stop count
# the attempt ends cleanly, so the harness scores the artifacts already written.
LOOP_WARN_REPEATS = 20
LOOP_STOP_REPEATS = 40
LOOP_WARN_MESSAGE = (
    "Loop guard: you have run this exact tool call {n} times, and the result "
    "will not change. Do not run it again. Use what you already have: finish "
    "the remaining artifacts, or try a different approach."
)
LOOP_STOP_MESSAGE = (
    "Stopped by the loop guard: the same tool call was repeated {n} times."
)

SYSTEM_PROMPT = (
    "You are a software engineering agent running non-interactively: no human "
    "will answer questions, so make reasonable decisions and keep going until "
    "the task is complete. Your working directory is the cloned repository for "
    "the task. The file_editor tool needs absolute paths. Each shell call starts "
    "a fresh shell, so pass absolute paths or cd within the same command.\n\n"
    # The swe3 skill was written for Claude Code and names its tools; this agent
    # has only shell and file_editor, and vLLM drops a call to any other name
    # without an error, which ends the attempt (issue #202).
    "You have exactly two tools: file_editor and shell. The skill instructions "
    "name tools from another agent. Map them like this:\n"
    "- Read: file_editor with command view (use view_range for part of a file)\n"
    "- Edit: file_editor with command str_replace or insert\n"
    "- Write: file_editor with command create for a new file; create refuses to "
    "overwrite, so change an existing file with str_replace\n"
    "- Bash, Grep, Glob: shell (for example grep -rn, find, ls, git)\n"
    "- Task: does not exist here; do all the work yourself\n"
    "Never call a tool by any other name."
)

# --harness builds the agent with the Strands team's create_harness() in place
# of the hand-built Agent above. Benchmark-safe settings: only the coding tools
# (no web_fetch or web_search, so the agent cannot look up the upstream fix; no
# subagent, since swe3 is single-agent; no programmatic_tool_caller), and no
# memory or session, so nothing carries between runs or tasks. Its context
# manager (which offloads large tool results), prompt caching, skills and the
# todos and environment plugins stay as shipped.
HARNESS_BUILTIN_TOOLS = ["shell", "read", "write", "edit"]
HARNESS_BUILTIN_PLUGINS = ["todos", "environment"]
# The tool map covers the Claude Code names the swe3 skill uses (issue #202);
# the harness tools share them in lower case, and vLLM drops a call whose name
# does not match exactly. In the system prompt alone it sits after the
# harness's own long prompt, and a small model still called `bash` on every
# attempt, so --harness also puts it at the top of the first message. Even
# there it slipped once in 44 turns, so `bash` is also a real tool: an alias of
# the harness shell (HARNESS_BASH_ALIAS), so the slip runs instead of ending
# the attempt.
HARNESS_TOOL_MAP = (
    "TOOLS: your tools are shell, read, write, edit and todo_write, all lower "
    "case. The skill below names tools from another agent. Map them like this:\n"
    "- Read: read\n"
    "- Edit: edit\n"
    "- Write: write\n"
    "- Bash, Grep, Glob: shell (for example grep -rn, find, ls, git); bash "
    "is an alias of shell\n"
    "- Task: does not exist here; do all the work yourself\n"
    "Never call a tool by any other name."
)
HARNESS_BASH_ALIAS = "bash"
# Appended after the harness's own system prompt.
HARNESS_INSTRUCTIONS = (
    "You are running non-interactively: no human will answer questions or "
    "approve steps, so make reasonable decisions and keep going until the task "
    "is complete. Your working directory is the cloned repository for the task.\n\n"
    + HARNESS_TOOL_MAP
)


def _emit(event: dict[str, Any]) -> None:
    """Write one event to stdout as a single JSON line and flush it."""
    sys.stdout.write(json.dumps(event, default=str) + "\n")
    sys.stdout.flush()


def _usage_dict(usage: Any) -> dict[str, int]:
    """Copy a Strands ``Usage`` mapping into a plain dict with every key present."""
    usage = usage or {}
    return {
        "inputTokens": int(usage.get("inputTokens", 0) or 0),
        "outputTokens": int(usage.get("outputTokens", 0) or 0),
        "totalTokens": int(usage.get("totalTokens", 0) or 0),
        "cacheReadInputTokens": int(usage.get("cacheReadInputTokens", 0) or 0),
        "cacheWriteInputTokens": int(usage.get("cacheWriteInputTokens", 0) or 0),
    }


def _preview(value: Any, limit: int = PREVIEW_CHARS) -> str:
    """Return a one-line, length-capped string form of a value for the log."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def _prompt_with_skill(skill_dir: Path, prompt: str) -> str:
    """Put the skill's SKILL.md ahead of the task prompt.

    Uses the wording ``_build_omp_cmd`` in run-swe-headless.py uses for omp, so
    both harnesses hand the model the same text. With the skill in the first
    message, the model never has to fetch it, and ``pin_first=1`` keeps it out
    of every summary (issue #201).
    """
    skill_md = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    return (
        f"{skill_md}\n\n"
        "---\n\n"
        "Follow the skill instructions above to complete the following task.\n\n"
        f"{prompt}"
    )


def _drop_empty_tools(request: dict[str, Any]) -> dict[str, Any]:
    """Remove an empty ``tools`` list from an OpenAI chat-completions request.

    Strands' ``OpenAIModel`` always sends ``tools``, even as ``[]`` on a call
    with no tools such as a conversation summary. vLLM rejects an empty array
    with HTTP 400, matching the OpenAI API, so the summary fails and the agent
    cannot recover from a context overflow. Remove this once
    strands-agents/harness-sdk#4854 is fixed (issue #198).
    """
    if not request.get("tools"):
        request.pop("tools", None)
    return request


def _result_text(content: list[dict[str, Any]]) -> str:
    """Join the text and JSON blocks of a tool result into one string."""
    parts = []
    for block in content:
        if "text" in block:
            parts.append(str(block["text"]))
        elif "json" in block:
            parts.append(json.dumps(block["json"], default=str))
    return "\n".join(parts)


def _cap_tool_result(
    result: dict[str, Any],
    limit: int = MAX_TOOL_RESULT_CHARS,
) -> tuple[dict[str, Any], int]:
    """Truncate a tool result whose text is longer than ``limit`` characters.

    Args:
        result: A Strands ``ToolResult`` (``content``, ``status``, ``toolUseId``).
        limit: The most characters of output to keep.

    Returns:
        The result to give the model, and the original character count when the
        result was truncated (0 when it was left unchanged).
    """
    text = _result_text(result.get("content") or [])
    if len(text) <= limit:
        return result, 0
    note = TRUNCATION_NOTE.format(total=len(text), kept=limit)
    capped = {**result, "content": [{"text": text[:limit] + note}]}
    return capped, len(text)


def _build_model(args: argparse.Namespace) -> Any:
    """Build the Strands model for the chosen provider.

    Amazon Bedrock uses ``BedrockModel`` with automatic prompt caching (Strands
    only adds cache points for models that support them). An OpenAI-compatible
    endpoint (vLLM, the LiteLLM proxy) uses ``OpenAIModel``.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A configured Strands model instance.
    """
    if args.provider == PROVIDER_BEDROCK:
        from strands.models import CacheConfig
        from strands.models.bedrock import BedrockModel

        bedrock_config: dict[str, Any] = {
            "model_id": args.model,
            "max_tokens": args.max_tokens,
            "cache_config": CacheConfig(strategy="auto"),
        }
        if args.context_window > 0:
            bedrock_config["context_window_limit"] = args.context_window
        return BedrockModel(region_name=args.aws_region, **bedrock_config)

    from strands.models.openai import OpenAIModel

    class _NoEmptyToolsOpenAIModel(OpenAIModel):  # type: ignore[misc, valid-type]
        def format_request(self, *a: Any, **kw: Any) -> dict[str, Any]:
            return _drop_empty_tools(super().format_request(*a, **kw))

    client_args = {
        "base_url": args.endpoint.rstrip("/") + "/v1",
        "api_key": os.environ.get(API_KEY_ENV, DEFAULT_API_KEY),
    }
    openai_config: dict[str, Any] = {
        "model_id": args.model,
        "params": {"max_tokens": args.max_tokens},
    }
    # Without this Strands assumes a 200K window, so proactive compaction on a
    # smaller served window (e.g. 128K) would trigger only after an overflow.
    # vLLM counts the requested max_tokens against the window, so the prompt
    # only has the rest; compacting at a fraction of the full window still
    # overflowed 160 times in one run (issue #197).
    if args.context_window > args.max_tokens:
        openai_config["context_window_limit"] = args.context_window - args.max_tokens
    return _NoEmptyToolsOpenAIModel(client_args=client_args, **openai_config)


def _build_conversation_manager(args: argparse.Namespace) -> Any:
    """Build a summarizing conversation manager that keeps the task prompt.

    ``pin_first=1`` keeps the first user message (the task) out of every
    summary. When the context window is known, compression runs proactively at
    ``compact_fraction`` of it, the Strands counterpart of the other harnesses'
    auto-compaction threshold; otherwise it runs when the model reports an
    overflow.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A ``SummarizingConversationManager``.
    """
    from strands.agent.conversation_manager import SummarizingConversationManager

    proactive: dict[str, float] | None = None
    if args.context_window > 0:
        proactive = {"compression_threshold": args.compact_fraction}
    return SummarizingConversationManager(pin_first=1, proactive_compression=proactive)


def _call_fingerprint(tool_use: dict[str, Any]) -> str:
    """Identify a tool call by its tool name and exact input."""
    return json.dumps(
        [tool_use.get("name"), tool_use.get("input")], sort_keys=True, default=str
    )


class _LoopGuard:
    """Strands hook provider that stops an agent repeating one tool call.

    Counts every call by tool name and exact input, over the whole attempt,
    since a loop can cycle through a few commands rather than repeat one back to
    back. Past ``warn`` repeats the call is cancelled with a message to the
    model; at ``stop`` repeats the attempt ends after the current tool batch.
    """

    def __init__(
        self, warn: int = LOOP_WARN_REPEATS, stop: int = LOOP_STOP_REPEATS
    ) -> None:
        self.warn = warn
        self.stop = stop
        self.counts: dict[str, int] = {}
        self.stopped = False

    def register_hooks(self, registry: Any, **_: Any) -> None:
        """Register the before-tool and after-tools callbacks."""
        from strands.hooks import AfterToolsEvent, BeforeToolCallEvent

        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)
        registry.add_callback(AfterToolsEvent, self._after_tools)

    def _before_tool_call(self, event: Any) -> None:
        tool_use = event.tool_use or {}
        key = _call_fingerprint(tool_use)
        n = self.counts.get(key, 0) + 1
        self.counts[key] = n
        if n <= self.warn:
            return
        event.cancel_tool = LOOP_WARN_MESSAGE.format(n=n)
        action = "warn"
        if n >= self.stop:
            self.stopped = True
            action = "stop"
        _emit(
            {
                "type": "loop_guard",
                "action": action,
                "repeats": n,
                "name": tool_use.get("name"),
                "input": _preview(tool_use.get("input", {})),
            }
        )

    def _after_tools(self, event: Any) -> None:
        if self.stopped:
            event.end_turn = LOOP_STOP_MESSAGE.format(n=self.stop)


class _EventStreamHooks:
    """Strands hook provider that mirrors model and tool activity to stdout."""

    def __init__(self, cap_results: bool = True) -> None:
        self.model_calls = 0
        # The hand-built agent needs our cap on tool output (issue #199); the
        # harness offloads large results itself, so --harness turns the cap off.
        self.cap_results = cap_results

    def register_hooks(self, registry: Any, **_: Any) -> None:
        """Register the model and tool callbacks with the agent's hook registry."""
        from strands.hooks import AfterModelCallEvent, AfterToolCallEvent

        registry.add_callback(AfterModelCallEvent, self._on_model_call)
        registry.add_callback(AfterToolCallEvent, self._on_tool_call)

    def _on_model_call(self, event: Any) -> None:
        self.model_calls += 1
        stop = event.stop_response
        _emit(
            {
                "type": "model_call",
                "index": self.model_calls,
                "stop_reason": getattr(stop, "stop_reason", None),
                "error": str(event.exception) if event.exception else None,
                "usage": _usage_dict(event.agent.event_loop_metrics.accumulated_usage),
            }
        )

    def _on_tool_call(self, event: Any) -> None:
        tool_use = event.tool_use or {}
        result = event.result or {}
        truncated_from = 0
        if result and self.cap_results:
            result, truncated_from = _cap_tool_result(result)
            if truncated_from:
                event.result = result
        _emit(
            {
                "type": "tool_call",
                "name": tool_use.get("name"),
                "input": _preview(tool_use.get("input", {})),
                "status": result.get("status"),
                "duration_s": event.duration,
                "truncated_from_chars": truncated_from,
            }
        )


def _build_harness_agent(args: argparse.Namespace, hooks: list[Any]) -> Any:
    """Build the agent with create_harness() and the benchmark-safe settings.

    The model instance is ours (``_build_model``), so the empty-``tools`` fix
    and the context-window limit still apply; the harness uses a pre-built
    model as given.
    """
    from strands.vended_tools import make_shell
    from strands_harness import create_harness

    # The same shell the harness builds, under the name small models reach for.
    bash_alias = make_shell(
        name=HARNESS_BASH_ALIAS,
        description="Alias of the shell tool: runs a shell command.",
    )
    return create_harness(
        model=_build_model(args),
        instructions=HARNESS_INSTRUCTIONS,
        tools=[bash_alias],
        builtin_tools=HARNESS_BUILTIN_TOOLS,
        builtin_plugins=HARNESS_BUILTIN_PLUGINS,
        session=False,
        memory=False,
        hooks=hooks,
        callback_handler=None,
    )


def _build_agent(args: argparse.Namespace, hooks: list[Any]) -> Any:
    """Assemble the Strands agent: model, tools and hooks."""
    if args.harness:
        return _build_harness_agent(args, hooks)
    from strands import Agent
    from strands.vended_tools import file_editor, shell

    return Agent(
        model=_build_model(args),
        system_prompt=SYSTEM_PROMPT,
        tools=[shell, file_editor],
        conversation_manager=_build_conversation_manager(args),
        hooks=hooks,
        # The default handler prints streamed text to stdout, which would
        # corrupt the JSON-lines stream the harness parses.
        callback_handler=None,
    )


def _run(args: argparse.Namespace) -> int:
    """Run the task and emit the final result event.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The process exit code.
    """
    prompt = _prompt_with_skill(args.skill_dir, args.prompt)
    if args.harness:
        prompt = f"{HARNESS_TOOL_MAP}\n\n---\n\n{prompt}"
    stream = _EventStreamHooks(cap_results=not args.harness)
    guard = _LoopGuard()
    agent = _build_agent(args, [stream, guard])
    logger.info(
        "Strands agent starting: build=%s provider=%s model=%s cwd=%s skill=%s",
        "create_harness" if args.harness else "Agent",
        args.provider,
        args.model,
        os.getcwd(),
        args.skill_dir,
    )
    try:
        result = agent(prompt)
    except Exception as exc:
        logger.exception("Strands agent failed")
        _emit(
            {
                "type": "result",
                "is_error": True,
                "error": f"{type(exc).__name__}: {exc}",
                "usage": _usage_dict(agent.event_loop_metrics.accumulated_usage),
                "cycle_count": agent.event_loop_metrics.cycle_count,
                "model_calls": stream.model_calls,
                "loop_guard_stopped": guard.stopped,
            }
        )
        return EXIT_AGENT_ERROR
    _emit(
        {
            "type": "result",
            "is_error": False,
            "stop_reason": result.stop_reason,
            "final_message": str(result)[:FINAL_MESSAGE_CHARS],
            "usage": _usage_dict(result.metrics.accumulated_usage),
            "cycle_count": result.metrics.cycle_count,
            "model_calls": stream.model_calls,
            "loop_guard_stopped": guard.stopped,
        }
    )
    return EXIT_OK


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run one benchmark task with a Strands Agents coding agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Amazon Bedrock\n"
            "  strands_agent_runner.py --provider bedrock --model "
            "us.anthropic.claude-haiku-4-5-20251001-v1:0 \\\n"
            "      --aws-region us-west-2 --skill-dir .claude/skills/swe3 "
            '--prompt "..."\n\n'
            "  # Self-hosted vLLM (key, if any, in STRANDS_API_KEY)\n"
            "  strands_agent_runner.py --provider endpoint --model qwen3-coder-30b \\\n"
            "      --endpoint http://127.0.0.1:8000 --context-window 262144 \\\n"
            '      --skill-dir .claude/skills/swe3 --prompt "..."\n'
        ),
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=[PROVIDER_BEDROCK, PROVIDER_ENDPOINT],
        help="Where model requests go.",
    )
    parser.add_argument("--model", required=True, help="Model id or served name.")
    parser.add_argument(
        "--aws-region",
        default=os.environ.get("AWS_REGION"),
        help="Amazon Bedrock region (provider=bedrock). Default: $AWS_REGION.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="OpenAI-compatible base URL without /v1 (provider=endpoint).",
    )
    parser.add_argument(
        "--skill-dir",
        required=True,
        type=Path,
        help="Directory holding the skill's SKILL.md.",
    )
    parser.add_argument("--prompt", required=True, help="The task prompt.")
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help="Model context window in tokens; 0 means unknown (default: 0).",
    )
    parser.add_argument(
        "--compact-fraction",
        type=float,
        default=DEFAULT_COMPACT_FRACTION,
        help="Fraction of the context window that triggers summarization.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Output token cap per model call (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--harness",
        action="store_true",
        help="Build the agent with create_harness() from the strands-harness "
        "package (agent=strands-harness) instead of a hand-built Agent.",
    )
    args = parser.parse_args(argv)
    if args.provider == PROVIDER_BEDROCK and not args.aws_region:
        parser.error("provider=bedrock needs --aws-region or AWS_REGION.")
    if args.provider == PROVIDER_ENDPOINT and not args.endpoint:
        parser.error("provider=endpoint needs --endpoint.")
    if not (args.skill_dir / "SKILL.md").is_file():
        parser.error(f"no SKILL.md in --skill-dir {args.skill_dir}.")
    return args


def _raise_recursion_limit(limit: int = RECURSION_LIMIT) -> None:
    """Raise Python's recursion limit to ``limit``, never lowering it."""
    if sys.getrecursionlimit() < limit:
        sys.setrecursionlimit(limit)


def main() -> None:
    """Parse arguments and run the task."""
    args = _parse_args()
    _raise_recursion_limit()
    sys.exit(_run(args))


if __name__ == "__main__":
    main()
