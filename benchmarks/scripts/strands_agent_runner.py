#!/usr/bin/env python3
"""Run one benchmark task with a Strands Agents coding agent.

Strands Agents is a Python SDK with no CLI of its own, so this script plays the
part that ``pi -p`` or ``codex exec`` play for the other harnesses: the headless
runner (``run-swe-headless.py``) starts it as a subprocess, it drives one task to
completion, and it reports what happened as JSON lines on stdout.

The agent gets the stock Strands coding tools (``shell`` and ``file_editor``)
and loads the SWE skill through the Strands ``AgentSkills`` plugin, the Strands
counterpart of pi's ``--skill`` flag. The working directory is the cloned task
repo; the harness sets it when it starts this process.

stdout carries only JSON lines, one object per line:

- ``{"type": "model_call", ...}`` after every model response, with the running
  token usage so far. Strands rolls usage up after this hook fires, so the
  figure trails by one call; it exists so a killed run still leaves a trail.
- ``{"type": "tool_call", ...}`` after every tool call.
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
EXIT_OK = 0
EXIT_AGENT_ERROR = 1

SYSTEM_PROMPT = (
    "You are a software engineering agent running non-interactively: no human "
    "will answer questions, so make reasonable decisions and keep going until "
    "the task is complete. Your working directory is the cloned repository for "
    "the task. The file_editor tool needs absolute paths. Each shell call starts "
    "a fresh shell, so pass absolute paths or cd within the same command."
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

    client_args = {
        "base_url": args.endpoint.rstrip("/") + "/v1",
        "api_key": os.environ.get(API_KEY_ENV, DEFAULT_API_KEY),
    }
    return OpenAIModel(
        client_args=client_args,
        model_id=args.model,
        params={"max_tokens": args.max_tokens},
    )


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


class _EventStreamHooks:
    """Strands hook provider that mirrors model and tool activity to stdout."""

    def __init__(self) -> None:
        self.model_calls = 0

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
        _emit(
            {
                "type": "tool_call",
                "name": tool_use.get("name"),
                "input": _preview(tool_use.get("input", {})),
                "status": result.get("status"),
                "duration_s": event.duration,
            }
        )


def _build_agent(args: argparse.Namespace, hooks: _EventStreamHooks) -> Any:
    """Assemble the Strands agent: model, tools, skill plugin and hooks."""
    from strands import Agent
    from strands.vended_plugins.skills import AgentSkills
    from strands.vended_tools import file_editor, shell

    return Agent(
        model=_build_model(args),
        system_prompt=SYSTEM_PROMPT,
        tools=[shell, file_editor],
        plugins=[AgentSkills(skills=[str(args.skill_dir)])],
        conversation_manager=_build_conversation_manager(args),
        hooks=[hooks],
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
    prompt = args.prompt
    hooks = _EventStreamHooks()
    agent = _build_agent(args, hooks)
    logger.info(
        "Strands agent starting: provider=%s model=%s cwd=%s skill=%s",
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
                "model_calls": hooks.model_calls,
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
            "model_calls": hooks.model_calls,
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
    args = parser.parse_args(argv)
    if args.provider == PROVIDER_BEDROCK and not args.aws_region:
        parser.error("provider=bedrock needs --aws-region or AWS_REGION.")
    if args.provider == PROVIDER_ENDPOINT and not args.endpoint:
        parser.error("provider=endpoint needs --endpoint.")
    if not (args.skill_dir / "SKILL.md").is_file():
        parser.error(f"no SKILL.md in --skill-dir {args.skill_dir}.")
    return args


def main() -> None:
    """Parse arguments and run the task."""
    sys.exit(_run(_parse_args()))


if __name__ == "__main__":
    main()
