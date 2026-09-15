#!/usr/bin/env python3
"""Run the SWE benchmark headless: drive `claude -p /swe2` over a dataset.

Given a dataset YAML and a runner config (endpoint, model, claude flags), this
harness runs each task end to end:

  1. Clone the task's repo at its pinned ref into a temporary directory.
  2. Invoke `claude -p "/swe2 repo: ... problem: ... model: ... answers: ..."`
     non-interactively, letting the /swe2 skill produce the four design
     artifacts (github-issue.md, lld.md, review.md, testing.md) AND implement
     the change, capturing patch.diff + implementation.md.
  3. Parse the run's JSON result for the six benchmark metrics (input/output/
     cache tokens, latency, and the number of LLM turns the agent took) and
     write them to metrics.json next to the artifacts.

Routing and claude flags come from the runner config; any field may be
overridden on the command line (CLI wins).

Usage:
    uv run scripts/run-swe-headless.py --config config/runner.example.yaml
    uv run scripts/run-swe-headless.py --config config/runner.example.yaml \\
        --model qwen3-coder-30b --tasks remove-faiss
    uv run scripts/run-swe-headless.py --config config/runner.example.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404 - used with list args, no shell, hardcoded command
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml

from bedrock_pricing import cost_usd as _bedrock_cost_usd
from dataset_loader import Dataset, DatasetError, Task, load_dataset
from runner_config import (
    RunnerConfig,
    RunnerConfigError,
    load_runner_config,
    model_to_wire_id,
)
from token_accounting import cache_partition_for_agent, compute_total_tokens_processed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# The four design artifacts every /swe2 run must produce. These are the ones
# whose presence defines a "complete" design and gate the ok flag / retry.
DESIGN_ARTIFACT_FILENAMES = ("github-issue.md", "lld.md", "review.md", "testing.md")
# The /swe2 implementation artifacts (the actual code change plus its summary).
IMPLEMENTATION_ARTIFACT_FILENAMES = ("patch.diff", "implementation.md")
# Everything a full /swe2 run emits, in produced-count order.
ARTIFACT_FILENAMES = DESIGN_ARTIFACT_FILENAMES + IMPLEMENTATION_ARTIFACT_FILENAMES
GIT_CLONE_TIMEOUT_SECONDS = 300

# A task's true cost is the sum of ALL its agent invocations -- every transient
# retry and every top-up, not just the pass that happened to succeed. Each
# _run_task call overwrites metrics.json with only that pass's numbers, so these
# fields are accumulated across passes and the sums restored. Money and cache
# tokens belong here as much as turns and tokens: for a retried or topped-up run
# they are just as real and just as additive, and omitting them undercounts both
# the cost and the tokens-processed total.
ADDITIVE_COST_FIELDS = (
    "input_tokens",
    "output_tokens",
    "num_turns",
    "latency_seconds",
    "total_cost_usd",
    "cache_read_tokens",
    "cache_creation_tokens",
)
# The normalized block renames cache-write; everything else keeps its name.
MM_BLOCK_KEY = {"cache_creation_tokens": "cache_write_tokens"}

# Sanity floor for output tokens per turn, used to catch a token-accounting bug at
# run time rather than in PR review. An agent that edits files emits hundreds of
# output tokens per turn (measured: ~370-730 across claude-code and pi). A value in
# the single digits is not a model behaviour -- it means usage was read from one
# scope instead of summed across all of them, which is exactly how the pi
# per-message usage bug (#99) and the earlier claude modelUsage bug undercounted
# multi-turn runs by ~100x. The failure is silent (a plausible number in the right
# field), so only a ratio check catches it. Set well below any real per-turn rate so
# it fires on a genuine accounting fault, not on a terse run.
MIN_PLAUSIBLE_OUTPUT_TOKENS_PER_TURN = 20
# Below this turn count the ratio is noise (a 1-2 turn run can legitimately emit
# very little), so the check is skipped.
TOKEN_SANITY_MIN_TURNS = 5

# The SWE skill file, shared by both agents. Claude Code auto-loads it from the
# repo's .claude/skills tree via the matching slash command; pi is pointed at the
# same file explicitly with `--skill` so both agents run the identical task. Which
# skill (swe2 multi-agent vs swe3 single-agent) is selected per run via config.
_SKILLS_DIR = REPO_ROOT / ".claude" / "skills"


def _skill_path(config: RunnerConfig) -> Path:
    """Absolute SKILL.md path for the run's configured skill (swe2/swe3)."""
    return _SKILLS_DIR / config.skill / "SKILL.md"


# pi provider names (the `--provider` value pi expects). For a self-hosted vLLM
# endpoint pi reads a "vllm" block from its models.json; for Amazon Bedrock pi
# has a native provider backed by the bundled AWS SDK bedrock-runtime client.
PI_PROVIDER_VLLM = "vllm"
PI_PROVIDER_BEDROCK = "amazon-bedrock"
# omp uses the same provider ids as pi; named separately so the two agents can
# diverge without a silent coupling.
OMP_PROVIDER_VLLM = "vllm"
OMP_PROVIDER_BEDROCK = "amazon-bedrock"

# kiro-cli binary and the parsers for the one-line summary it prints on stderr at
# the end of a non-interactive run, e.g. "▸ Credits: 0.21 • Time: 17s". kiro-cli
# reports no token counts, so credits (its billing unit) are the cost signal; the
# harness turns them into dollars with a configurable per-credit rate. Output is
# ANSI-colored, so strip escape codes before matching. See docs/kiro-cli-setup.md.
KIRO_CLI_BIN = "kiro-cli"
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
_KIRO_CREDITS_RE = re.compile(r"Credits:\s*([0-9]*\.?[0-9]+)")
_KIRO_TIME_RE = re.compile(r"Time:\s*([0-9]+)\s*s")

# The harness scrapes vLLM's entire Prometheus /metrics surface (every family
# under this prefix) rather than a curated subset, so nothing is omitted and new
# vLLM metrics appear automatically. These series are SERVER-WIDE and CUMULATIVE:
# they aggregate every request from every client since the server started and
# carry no per-request or per-session label. See _snapshot_vllm_metrics for the
# loud single-tenant caveat that this implies. Note: in vLLM v1 a prefix-cache
# hit IS the KV-cache reuse signal -- there is no separate "KV cache hit"
# counter; the prefix-cache queries/hits counters are measured in tokens, not
# lookup events. The counters used to derive the headline hit rates:
VLLM_METRIC_PREFIX = "vllm:"
PREFIX_CACHE_QUERIES_METRIC = "vllm:prefix_cache_queries_total"
PREFIX_CACHE_HITS_METRIC = "vllm:prefix_cache_hits_total"
PROMPT_TOKENS_METRIC = "vllm:prompt_tokens_total"
PROMPT_TOKENS_CACHED_METRIC = "vllm:prompt_tokens_cached_total"
KV_CACHE_USAGE_METRIC = "vllm:kv_cache_usage_perc"
METRICS_SCRAPE_TIMEOUT_SECONDS = 10

# Gauges are point-in-time, so a before/after snapshot misses what happened
# DURING the run: KV-cache usage, for instance, reads its true value only while a
# request is in flight and drains back to 0 once the run ends. A background
# poller samples these while claude -p runs and reports the peak/mean instead.
# See _GaugePoller. These are the load/pressure gauges that actually vary.
SAMPLED_GAUGE_METRICS = (
    KV_CACHE_USAGE_METRIC,
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)
GAUGE_POLL_INTERVAL_SECONDS = 1.0


def _repo_name(repo_url: str) -> str:
    """Derive the kebab-case repo name from a clone URL.

    Args:
        repo_url: The HTTPS clone URL (with or without a trailing .git).

    Returns:
        The repository basename, e.g. "mcp-gateway-registry".
    """
    return repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")


def _safe_task_slug(task_id: str) -> str:
    """Return a filesystem-safe slug for a task id, for use in a clone path.

    The task id lands in a directory name, so anything that is not a plain
    path-segment character is replaced with ``-``. This both keeps the path the
    agent must reproduce simple and blocks path-traversal (``/`` and ``.`` runs
    cannot escape the parent). Task ids are already kebab-case slugs in practice,
    so this is a defensive no-op for well-formed input.

    Args:
        task_id: The dataset task id.

    Returns:
        A slug containing only ``[A-Za-z0-9._-]``, with leading dots/dashes and
        empty results collapsed to ``task``.

    Raises:
        ValueError: If task_id is empty.
    """
    if not task_id:
        raise ValueError("task id must not be empty")
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", task_id).lstrip(".-")
    return slug or "task"


def _clone_repo(task: Task, ref: str, clone_dir: str, log_prefix: str = "") -> Path:
    """Clone a task's repo at a ref into a temp dir named after the task.

    The checkout lands at ``<clone_dir>/swe-clone-<task-id>/<repo-name>`` so the
    /swe skill, which derives {repo-name} from the clone path's basename, gets
    the right name -- and the parent is a stable, transcribable name (the task
    id) rather than a random mktemp suffix, which agents were mis-copying
    character by character and burning turns on. The task-id parent is unique
    per task (ids are unique within a dataset), so serial and concurrent runs do
    not collide. The ``swe-clone-`` prefix is distinct from the ``swe-benchmark-
    data`` output dir so a gitignore glob can target clones precisely. A leftover
    directory from a previously killed run is removed first.

    Args:
        task: The task whose repo to clone.
        ref: The git ref (tag/branch/commit) to check out.
        clone_dir: Parent directory for the temporary clone.
        log_prefix: Optional label (e.g. ``[task=x] 3 of 12``) prepended to the
            clone log line so interleaved concurrent runs stay legible.

    Returns:
        Path to the cloned repository.

    Raises:
        RuntimeError: If the clone command fails or times out.
    """
    name = _repo_name(task.repo)
    parent = Path(clone_dir) / f"swe-clone-{_safe_task_slug(task.id)}"
    # Clear any leftover clone from a prior killed run so the fresh clone into a
    # deterministic path does not fail on a non-empty destination.
    shutil.rmtree(parent, ignore_errors=True)
    parent.mkdir(parents=True, exist_ok=True)
    dest = parent / name
    prefix = f"{log_prefix} " if log_prefix else ""
    logger.info("  %sCloning %s @ %s into %s", prefix, task.repo, ref, dest)
    try:
        subprocess.run(  # nosec B603 B607 - hardcoded git, args are dataset values, no shell
            [
                "git",
                "clone",
                "--branch",
                ref,
                "--depth",
                "1",
                task.repo,
                str(dest),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_CLONE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(parent, ignore_errors=True)
        raise RuntimeError(f"git clone timed out for {task.repo} @ {ref}") from exc
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(parent, ignore_errors=True)
        raise RuntimeError(
            f"git clone failed for {task.repo} @ {ref}: {exc.stderr.strip()[:500]}"
        ) from exc
    return dest


def _build_prompt(
    task: Task,
    clone_path: Path,
    ref: str,
    model: str,
    artifacts_dir: Path,
    agent: str = "claude",
    skill: str = "swe2",
    topup_missing: list[str] | None = None,
) -> str:
    """Build the non-interactive /swe2 prompt for a task.

    This function only **hydrates** the `/swe2` invocation with the per-run values
    the skill cannot know on its own -- it carries no behavioral instructions.
    All rules about *how* `/swe2` should run headless (use `artifacts_dir`
    verbatim, do not re-clone or `cd` out, pace/budget, subagent cap) live in the
    skill (`.claude/skills/swe2/SKILL.md`), which applies to every invocation;
    duplicating them here only risks the two copies drifting apart.

    The invocation passes the keys the skill needs to enter non-interactive mode
    (repo, problem, model, tag, answers) plus ``artifacts_dir`` (the absolute
    directory the skill writes to) and the task's problem statement and, when
    present, its reference issue URL.

    The only agent-specific difference is how the skill is triggered. Claude Code
    auto-loads the skill from the ``/swe2`` slash command, so the prompt starts
    with it. pi loads the same ``SKILL.md`` via ``--skill`` and exposes it as a
    skill named ``swe2``; it has no slash-command syntax, so the pi prompt names
    the skill in prose and passes the identical key/value payload. Both carry the
    exact same values, so the two agents run the same task.

    Args:
        task: The task to run.
        clone_path: Local path to the already-cloned repo (the sole code source).
        ref: The git ref checked out.
        model: The model name (also the artifact subfolder name).
        artifacts_dir: Absolute directory the six artifacts must be written to.
        agent: Which agent will receive the prompt ("claude" or "pi").
        topup_missing: When set, build a FOCUSED top-up prompt instead of the full
            task prompt: the design docs already exist in ``artifacts_dir`` and the
            agent is asked to produce ONLY these missing files (reading the ones
            already on disk), then stop. Used by the harness's completion loop when
            a main run finished the design but ran out of context before the
            implementation artifacts. Everything else (repo, ids, layout) is the
            same, so the topped-up artifacts belong to the same task.

    Returns:
        The prompt string to pass to the agent.
    """
    answers = task.clarifying_answers or (
        "No separate answers provided. Use your best judgment; all needed "
        "information is in the task description below."
    )
    payload = (
        f"repo: {clone_path} problem: {task.id} model: {model} "
        f'tag: {ref} artifacts_dir: {artifacts_dir} answers: "{answers.strip()}"'
    )
    if agent in ("pi", "kiro", "omp"):
        # None of pi, kiro or omp has slash commands; name the skill in prose and
        # hand it the payload. (pi loads SKILL.md via --skill; kiro and omp have no
        # --skill flag, so their _build_*_cmd inlines the SKILL.md content ahead of
        # this prompt.)
        invocation = f"Use the {skill} skill to complete this task. {payload}"
    else:
        invocation = f"/{skill} {payload}"
    if topup_missing:
        # Focused completion pass: the prior run already wrote the design docs
        # into artifacts_dir; only the listed files are missing. Ask the agent to
        # read what exists and produce ONLY those, without redoing the rest. This
        # keeps the top-up cheap (fresh, small context) so it does not hit the
        # same window wall that truncated the main run.
        missing = ", ".join(topup_missing)
        existing = ", ".join(f for f in ARTIFACT_FILENAMES if f not in topup_missing)
        lines = [
            invocation,
            "",
            "COMPLETION PASS -- do NOT restart the task.",
            f"The artifact directory ({artifacts_dir}) already contains these "
            f"finished artifacts: {existing}. Read them as needed for consistency.",
            f"Produce ONLY the missing artifact(s): {missing}. Follow the same "
            "skill rules for those artifacts (for patch.diff, implement the change "
            "the existing lld.md describes in the cloned repo and capture the diff; "
            "for implementation.md, summarize that change). Do not modify or "
            "rewrite the artifacts that already exist. When the missing files are "
            "written, stop.",
            "",
            "Task description:",
            task.problem_statement or "(see reference issue)",
        ]
        if task.problem_issue_url:
            lines += ["", f"Reference issue: {task.problem_issue_url}"]
        return "\n".join(lines)
    lines = [
        invocation,
        "",
        "Task description:",
        task.problem_statement or "(see reference issue)",
    ]
    if task.problem_issue_url:
        lines += ["", f"Reference issue: {task.problem_issue_url}"]
    return "\n".join(lines)


def _build_env(config: RunnerConfig) -> dict[str, str]:
    """Build the environment for the claude subprocess from the runner config.

    For provider=endpoint, routing pins ANTHROPIC_BASE_URL/API_KEY and disables
    Bedrock. For provider=bedrock, it flips CLAUDE_CODE_USE_BEDROCK=1 and sets
    AWS_REGION so claude talks to Amazon Bedrock natively, using the ambient AWS
    credentials; no base URL or api key is set.

    Args:
        config: The runner config.

    Returns:
        A copy of the current environment with routing overrides applied.
    """
    env = os.environ.copy()
    env["DISABLE_NON_ESSENTIAL_MODEL_CALLS"] = "1"
    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(config.max_output_tokens)
    env["CLAUDE_CODE_SUBAGENT_MODEL"] = config.model
    # Calibrate auto-compaction to the model's true window when known. Claude
    # Code cannot detect the window of a custom model on a custom base URL, so
    # without this it never compacts and the request eventually overflows the
    # endpoint's context limit (a 500 the client then retries forever).
    if config.auto_compact_window is not None:
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(config.auto_compact_window)
    if config.is_bedrock:
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        region = config.resolved_region()
        if region:
            env["AWS_REGION"] = region
        # A stray ANTHROPIC_BASE_URL in the ambient env would otherwise redirect
        # the Bedrock-mode client away from Bedrock, so clear it.
        env.pop("ANTHROPIC_BASE_URL", None)
    else:
        env["ANTHROPIC_BASE_URL"] = config.endpoint
        env["ANTHROPIC_API_KEY"] = config.api_key
        env["CLAUDE_CODE_USE_BEDROCK"] = "0"
    return env


def _build_settings_arg(config: RunnerConfig) -> str:
    """Build the value for `claude --settings`.

    A settings file's ``env`` block takes precedence over process environment
    variables, so relying on _build_env alone is not enough: a user's global
    ``~/.claude/settings.json`` (e.g. one that pins CLAUDE_CODE_USE_BEDROCK=1)
    would override our routing and the request would hit Bedrock, which rejects
    the local model id with a 400. Passing --settings overrides that global
    file, so we always supply one.

    Uses the configured ``settings_file`` when set; otherwise synthesizes an
    inline JSON settings object that pins routing at the config's endpoint.

    Args:
        config: The runner config.

    Returns:
        Either a settings file path or an inline JSON settings string.
    """
    if config.settings_file:
        return str(REPO_ROOT / config.settings_file)
    if config.is_bedrock:
        # Bedrock mode authenticates with ambient AWS credentials, so no token
        # source is needed. Pin CLAUDE_CODE_USE_BEDROCK=1 (and the region) here
        # too, so a global settings file cannot flip routing back off Bedrock.
        env: dict[str, str] = {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(config.max_output_tokens),
            "CLAUDE_CODE_SUBAGENT_MODEL": config.model,
        }
        region = config.resolved_region()
        if region:
            env["AWS_REGION"] = region
        # The settings env block overrides the process env, so mirror the
        # auto-compaction window here too when it is set.
        if config.auto_compact_window is not None:
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(config.auto_compact_window)
        return json.dumps({"env": env})
    endpoint_env = {
        "CLAUDE_CODE_USE_BEDROCK": "0",
        "ANTHROPIC_BASE_URL": config.endpoint,
        "ANTHROPIC_API_KEY": config.api_key,
        "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(config.max_output_tokens),
        "CLAUDE_CODE_SUBAGENT_MODEL": config.model,
    }
    # The settings env block overrides the process env, so mirror the
    # auto-compaction window here too when it is set.
    if config.auto_compact_window is not None:
        endpoint_env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(
            config.auto_compact_window
        )
    settings = {
        # Claude Code requires a token source even against a local endpoint that
        # ignores the value; without it the run fails with "Not logged in".
        "apiKeyHelper": f"echo {config.api_key}",
        "env": endpoint_env,
    }
    return json.dumps(settings)


def _build_claude_cmd(
    config: RunnerConfig,
    prompt: str,
    stream: bool = False,
    clone_path: Path | None = None,
) -> list[str]:
    """Assemble the `claude -p` argument vector from the runner config.

    Args:
        config: The runner config.
        prompt: The /swe prompt to run.
        stream: If True, emit newline-delimited JSON events as the run
            progresses (``--output-format stream-json``, which requires
            ``--verbose``) instead of a single buffered JSON result.
        clone_path: The task's cloned repo directory. When set, it is added as
            an allowed working directory with ``--add-dir`` so Bash commands can
            operate inside the clone. Read/Glob/Grep already reach absolute paths
            regardless; without this, Bash (ls/cd/find/grep into the clone) is
            blocked because it is sandboxed to the harness's own working dir.

    Returns:
        The command as a list of arguments (never a shell string).
    """
    output_format = "stream-json" if stream else "json"
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        config.model,
        "--output-format",
        output_format,
        "--permission-mode",
        config.permission_mode,
        "--allowedTools",
        ",".join(config.allowed_tools),
        "--max-turns",
        str(config.max_turns),
        "--settings",
        _build_settings_arg(config),
    ]
    if clone_path is not None:
        cmd += ["--add-dir", str(clone_path)]
    if stream:
        # stream-json in -p mode requires --verbose to emit per-event objects.
        cmd.append("--verbose")
    return cmd


def _write_pi_models_json(config: RunnerConfig, agent_dir: Path) -> None:
    """Write an ephemeral pi ``models.json`` pointing at the config's endpoint.

    pi resolves providers from ``<agent_dir>/models.json`` (agent_dir defaults to
    ``~/.pi/agent`` but is overridden per run via ``PI_CODING_AGENT_DIR`` so the
    benchmark never mutates a developer's global pi config). This writes a single
    ``vllm`` provider block -- the OpenAI-compatible ``/v1`` base URL and one
    anchor model whose id matches ``config.model`` -- mirroring
    ``self-hosted/vllm/scripts/run-pi.sh``. Cost is left at 0 because a
    self-hosted model has no per-token price; the real cost is hardware-derived
    separately (see cost-per-task-methodology.md).

    Args:
        config: The runner config (endpoint + model).
        agent_dir: The per-run pi agent dir to write ``models.json`` into.
    """
    # pi's baseUrl expects the OpenAI-compatible root ending in /v1.
    base = config.endpoint.rstrip("/")
    base_url = base if base.endswith("/v1") else f"{base}/v1"
    window = config.context_window or 200000
    models_json = {
        "providers": {
            "vllm": {
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": config.api_key,
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                },
                "models": [
                    {
                        "id": config.model,
                        "name": f"vLLM: {config.model}",
                        "reasoning": False,
                        "input": ["text"],
                        "contextWindow": window,
                        "maxTokens": config.max_output_tokens,
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                    }
                ],
            }
        }
    }
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "models.json").write_text(
        json.dumps(models_json, indent=2) + "\n", encoding="utf-8"
    )
    _write_pi_settings(config, agent_dir)


def _write_pi_settings(config: RunnerConfig, agent_dir: Path) -> None:
    """Write an ephemeral pi ``settings.json`` that tunes auto-compaction.

    pi has built-in auto-compaction (docs/compaction.md): it summarizes older
    messages once ``contextTokens > contextWindow - reserveTokens``. Left at pi's
    default ``reserveTokens`` of 16384, a long ``/swe2`` task on a 200K-window
    model overflows anyway -- pi lets the window fill to within 16K, then a single
    response (capped at ``maxTokens``, which we raise well above 16K) blows past
    the window and the run dies with ``stop_reason: length`` before the last
    artifacts are written (observed on the mcp-gateway-registry tasks).

    The fix is to reserve at least a full response worth of tokens (plus headroom)
    so threshold compaction fires *before* the overflow wall, keeping the run
    alive through the whole six-artifact chain -- the same role
    ``CLAUDE_CODE_AUTO_COMPACT_WINDOW`` plays for the Claude Code path. We set
    ``reserveTokens`` to ``max_output_tokens`` plus a margin, and keep pi's default
    ``keepRecentTokens``. ``contextWindow`` itself is carried in models.json.

    Args:
        config: The runner config (its ``max_output_tokens`` sizes the reserve).
        agent_dir: The per-run pi agent dir to write ``settings.json`` into.
    """
    # Reserve a full response plus ~8K headroom so compaction triggers with room
    # to spare for the reply and per-request overhead, never after overflow.
    reserve = config.max_output_tokens + 8192
    settings = {
        "compaction": {
            "enabled": True,
            "reserveTokens": reserve,
        }
    }
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )


def _write_omp_config(config: RunnerConfig, agent_dir: Path) -> None:
    """Write the per-run omp ``models.yml`` and ``config.yml`` into ``agent_dir``.

    omp is a fork of pi and honours the same ``PI_CODING_AGENT_DIR`` override, but
    its config is YAML rather than pi's ``models.json``: custom providers live in
    ``models.yml`` and settings in ``config.yml``. Writing both per run keeps the
    benchmark isolated from a developer's global ``~/.omp``.

    Compaction is sized the same way as pi's (see ``_write_pi_settings``): omp
    expresses the trigger as an absolute ``compaction.thresholdTokens`` instead of
    pi's ``reserveTokens``, so we convert, reserving a full response plus ~8K of
    headroom. Without it omp fills the window to within its default reserve and a
    single capped response overflows, killing the run before the last artifacts
    are written.

    Args:
        config: The runner config (endpoint, model, window, output cap).
        agent_dir: The per-run omp agent dir to write both files into.
    """
    base = config.endpoint.rstrip("/")
    base_url = base if base.endswith("/v1") else f"{base}/v1"
    window = config.context_window or 200000
    models_yml = {
        "providers": {
            OMP_PROVIDER_VLLM: {
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": config.api_key,
                "models": [
                    {
                        "id": config.model,
                        "name": f"vLLM: {config.model}",
                        "contextWindow": window,
                        "maxTokens": config.max_output_tokens,
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                    }
                ],
            }
        }
    }
    # Compact once the context passes (window - one full response - headroom), the
    # same effective trigger pi derives from reserveTokens.
    threshold = max(window - (config.max_output_tokens + 8192), 1)
    config_yml = {"compaction": {"enabled": True, "thresholdTokens": threshold}}
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "models.yml").write_text(
        yaml.safe_dump(models_yml, sort_keys=False), encoding="utf-8"
    )
    (agent_dir / "config.yml").write_text(
        yaml.safe_dump(config_yml, sort_keys=False), encoding="utf-8"
    )


def _build_omp_env(config: RunnerConfig, agent_dir: Path) -> dict[str, str]:
    """Build the environment for the omp subprocess.

    Mirrors ``_build_pi_env``: ``PI_CODING_AGENT_DIR`` (which omp inherits from pi)
    points at the per-run config dir, and the Bedrock path pins the region and
    resolves the ambient credential chain into explicit keys.

    Args:
        config: The runner config (provider, aws region).
        agent_dir: The per-run omp agent config dir.

    Returns:
        A copy of the environment with the omp agent dir pinned.
    """
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    if config.is_bedrock:
        region = config.resolved_region()
        if region:
            env["AWS_REGION"] = region
        _ensure_aws_sigv4_env(env)
    return env


def _build_omp_cmd(config: RunnerConfig, prompt: str) -> list[str]:
    """Assemble the ``omp -p --mode json`` argument vector.

    omp has no ``--skill`` flag (its ``--skills`` is a glob filter over discovered
    skills, not a path), so the SKILL.md the other agents load is inlined ahead of
    the task payload exactly as ``_build_kiro_cmd`` does. ``--mode json`` gives the
    pi-shaped event stream the harness already knows how to read, and
    ``--no-session`` keeps the run ephemeral.

    Args:
        config: The runner config (model, provider).
        prompt: The hydrated prompt (see ``_build_prompt`` agent="omp").

    Returns:
        The command as a list of arguments (never a shell string).
    """
    skill_md = _skill_path(config).read_text(encoding="utf-8")
    full_prompt = (
        f"{skill_md}\n\n"
        "---\n\n"
        "Follow the skill instructions above to complete the following task.\n\n"
        f"{prompt}"
    )
    if config.is_bedrock:
        model = f"{OMP_PROVIDER_BEDROCK}/{model_to_wire_id(config.model)}"
    else:
        model = f"{OMP_PROVIDER_VLLM}/{config.model}"
    # A trailing "--" ends option parsing so the prompt is always treated as the
    # positional INPUT -- essential here because the inlined SKILL.md begins with
    # "---" (YAML frontmatter), which omp would otherwise reject as an unknown
    # flag, exactly as kiro-cli does (see _build_kiro_cmd).
    cmd = [
        "omp",
        "-p",
        "--mode",
        "json",
        "--no-session",
        "--auto-approve",
        "--model",
        model,
    ]
    # omp has no turn cap, so a model that finishes the work and then loops --
    # emitting tokens without ever ending its turn -- would run until the
    # harness's own timeout_seconds and then burn a retry on an already-complete
    # task. --max-time makes omp stop itself first, which also lets the harness
    # collect whatever the run produced instead of killing it mid-write.
    if config.agent_max_time_seconds:
        cmd += [f"--max-time={config.agent_max_time_seconds}"]
    # A trailing "--" ends option parsing so the prompt is always the positional
    # INPUT (see the note above).
    return [*cmd, "--", full_prompt]


def _run_omp(
    cmd: list[str],
    env: dict[str, str],
    timeout: int,
    stream_log: Path | None = None,
) -> dict[str, Any]:
    """Run ``omp -p --mode json`` and normalize its events to a result dict.

    omp emits the same event stream as pi (``turn_start`` for turn counting, a
    final ``agent_end`` carrying per-message ``usage``), so the pi normalizer is
    reused verbatim rather than duplicated.

    The one behavioural difference that matters: omp treats an inherited stdin as
    a piped prompt and blocks waiting for EOF, ignoring the positional prompt
    entirely. ``stdin=DEVNULL`` is therefore required, not cosmetic -- without it
    the run hangs until the timeout with no output.

    Args:
        cmd: The omp command argument vector.
        env: Environment for the subprocess (pins PI_CODING_AGENT_DIR).
        timeout: Wall-clock timeout in seconds.
        stream_log: Optional file to append omp's events to as they arrive. A
            task runs for tens of minutes with no terminal output otherwise --
            ``capture_output`` only yields once the process exits -- so this is
            the only way to watch a run in flight (``tail -f`` it). omp's own
            ``~/.omp/logs`` carries lifecycle debug lines, not the event stream.

    Returns:
        The claude-shaped result dict (see ``_pi_result_from_events``).

    Raises:
        RuntimeError: If omp times out, emits no output, or emits no agent_end.
    """
    start = time.time()
    # Read stdout line by line rather than with subprocess.run so the stream can
    # be mirrored to stream_log while the task runs; run() would withhold every
    # line until exit, leaving a 30-60 minute task looking identical to a hang.
    sink = None
    if stream_log is not None:
        stream_log.parent.mkdir(parents=True, exist_ok=True)
        sink = stream_log.open("a", encoding="utf-8")
    stdout_lines: list[str] = []
    events: list[dict[str, Any]] = []
    try:
        proc = subprocess.Popen(  # nosec B603 - hardcoded 'omp', list args, no shell
            cmd,
            env=env,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        try:
            for line in proc.stdout or []:
                stdout_lines.append(line)
                if sink is not None:
                    sink.write(line)
                    sink.flush()
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    events.append(json.loads(stripped))
                except json.JSONDecodeError:
                    # omp interleaves human-readable startup notices; skip them.
                    continue
            proc.wait(timeout=max(timeout - (time.time() - start), 1))
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            raise RuntimeError(f"omp -p timed out after {timeout}s") from exc
        stderr = (proc.stderr.read() if proc.stderr else "") or ""
    finally:
        if sink is not None:
            sink.close()
    elapsed = time.time() - start

    if not "".join(stdout_lines).strip():
        raise RuntimeError(
            f"omp -p produced no output (exit {proc.returncode}): "
            f"{stderr.strip()[:500]}"
        )
    if not events:
        raise RuntimeError(
            f"omp -p output had no JSON events: {''.join(stdout_lines).strip()[:500]}"
        )
    result = _pi_result_from_events(events, elapsed)
    result["_elapsed_seconds"] = round(elapsed, 1)
    return result


def _build_pi_env(config: RunnerConfig, agent_dir: Path) -> dict[str, str]:
    """Build the environment for the pi subprocess.

    ``PI_CODING_AGENT_DIR`` points pi at the per-run config dir so the benchmark
    stays isolated from a developer's global ``~/.pi`` setup. For a vLLM endpoint,
    routing comes from the per-run ``models.json`` and no other override is needed.
    For Amazon Bedrock, pi uses the ambient AWS credential chain (env/ini/sso/...)
    and needs the region: we pin ``AWS_REGION`` from the resolved config so the run
    is reproducible regardless of the caller's shell. Credentials themselves are
    never injected here -- they come from the standard chain, so no secret is
    written or logged.

    Args:
        config: The runner config (provider, aws region).
        agent_dir: The per-run pi agent config dir.

    Returns:
        A copy of the current environment with the pi agent dir pinned (plus the
        AWS region for the bedrock path).
    """
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    if config.is_bedrock:
        region = config.resolved_region()
        if region:
            env["AWS_REGION"] = region
        _ensure_aws_sigv4_env(env)
    return env


def _ensure_aws_sigv4_env(env: dict[str, str]) -> None:
    """Populate SigV4 AWS credentials in ``env`` for pi's Bedrock provider.

    pi's ``amazon-bedrock`` provider authenticates from explicit credentials
    (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_SESSION_TOKEN`` or
    ``AWS_BEARER_TOKEN_BEDROCK``); unlike boto3 it does NOT probe the EC2 IMDS
    instance-profile chain. On an EC2 box whose only credentials come from an
    attached instance role, a run would fail with "No API key found". This
    resolves the role's short-lived credentials via ``aws configure
    export-credentials`` and injects them, so the ambient instance role Just Works.

    No-ops when credentials are already present in ``env`` (the caller's shell set
    them, or a bearer token is configured) or when the AWS CLI cannot mint any
    (off EC2 with no role) -- pi then reports its own clear auth error. Credentials
    are only placed in the child env, never logged or written to disk.

    Args:
        env: The subprocess environment to populate in place.
    """
    if env.get("AWS_ACCESS_KEY_ID") or env.get("AWS_BEARER_TOKEN_BEDROCK"):
        return
    try:
        proc = subprocess.run(
            ["aws", "configure", "export-credentials", "--format", "env-no-export"],  # nosec B603 B607 - hardcoded command, no user input
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        # No resolvable credentials (or no AWS CLI): leave env as-is and let pi
        # surface its own auth error rather than failing opaquely here.
        logger.warning(
            "could not resolve AWS credentials via 'aws configure export-credentials'; "
            "pi will rely on whatever is already in the environment"
        )
        return
    for line in proc.stdout.splitlines():
        key, _, value = line.strip().partition("=")
        if key.startswith("AWS_") and value:
            env[key] = value


def _build_pi_cmd(config: RunnerConfig, prompt: str) -> list[str]:
    """Assemble the ``pi -p`` argument vector for a /swe2 run.

    pi runs headless with ``-p`` (process the prompt and exit) and ``--mode json``
    (emit a stream of JSON-lines events the harness parses for metrics). Tools run
    without an approval gate in ``-p`` mode, which is what an unattended benchmark
    needs. The ``/swe2`` behavior is delivered by loading the same SKILL.md Claude
    Code uses, via ``--skill``. ``--no-session`` keeps the run ephemeral (no
    session file written under the agent dir).

    The ``--provider`` depends on routing: a self-hosted vLLM endpoint (resolved
    from the per-run models.json) or native Amazon Bedrock (pi signs SigV4 via the
    bundled AWS SDK; credentials come from the ambient chain, region from the env
    set in ``_build_pi_env``). The model id is the Bedrock inference-profile id
    (e.g. ``us.anthropic.claude-opus-5``) for bedrock, or the served-model-name
    for vLLM.

    Args:
        config: The runner config (model, provider).
        prompt: The hydrated /swe2 prompt (see ``_build_prompt`` agent="pi").

    Returns:
        The command as a list of arguments (never a shell string).
    """
    if config.is_bedrock:
        # pi resolves the Bedrock inference profile itself, so pass the clean
        # wire id (region prefix kept, harness "[1m]" hint stripped). For vLLM the
        # served-model-name is used verbatim.
        provider = PI_PROVIDER_BEDROCK
        model = model_to_wire_id(config.model)
    else:
        provider = PI_PROVIDER_VLLM
        model = config.model
    return [
        "pi",
        "-p",
        "--mode",
        "json",
        "--no-session",
        "--provider",
        provider,
        "--model",
        model,
        "--skill",
        str(_skill_path(config)),
        prompt,
    ]


def _build_kiro_env(config: RunnerConfig) -> dict[str, str]:
    """Environment for a kiro-cli run.

    kiro-cli authenticates through its own global sign-in under ``~/.kiro`` (AWS
    Builder ID / IAM Identity Center / Google), so -- unlike pi -- the harness
    does NOT redirect ``KIRO_HOME`` to a per-run dir: that would hide the login
    and force an interactive re-auth mid-benchmark. The model is passed on the
    command line, so there is no per-run config to write; the developer's global
    kiro config is read but never mutated.

    Args:
        config: The runner config (unused today; kept for signature parity with
            ``_build_pi_env``).

    Returns:
        A copy of the current process environment.
    """
    return os.environ.copy()


def _build_kiro_cmd(config: RunnerConfig, prompt: str) -> list[str]:
    """Assemble the ``kiro-cli chat --no-interactive`` argument vector.

    kiro-cli is the ``claude -p`` / ``codex exec`` analogue: it takes a prompt
    argument, runs to completion, and exits. It has NO ``--skill`` flag, so the
    same ``SKILL.md`` the other agents load is inlined ahead of the task payload
    in the prompt. ``--trust-all-tools`` pre-approves tool use (no operator is
    present in a benchmark); ``--model`` selects one of Kiro's managed models.
    kiro-cli cannot target a self-hosted endpoint, so there is no provider or
    endpoint to pass. See docs/kiro-cli-setup.md.

    Args:
        config: The runner config (model).
        prompt: The hydrated prompt (see ``_build_prompt`` agent="kiro").

    Returns:
        The command as a list of arguments (never a shell string).
    """
    skill_md = _skill_path(config).read_text(encoding="utf-8")
    full_prompt = (
        f"{skill_md}\n\n"
        "===TASK===\n"
        "Follow the skill instructions above to complete the following task.\n\n"
        f"{prompt}"
    )
    # A trailing "--" ends option parsing so the prompt is always treated as the
    # positional INPUT -- essential here because the inlined SKILL.md begins with
    # "---" (YAML frontmatter), which kiro-cli would otherwise reject as an
    # unknown flag.
    return [
        KIRO_CLI_BIN,
        "chat",
        "--no-interactive",
        "--trust-all-tools",
        "--model",
        config.model,
        "--",
        full_prompt,
    ]


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with a trailing Z.

    Returns:
        The timestamp, e.g. ``2026-07-22T20:41:03.512874Z``.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _claude_token_usage(result: dict[str, Any]) -> dict[str, int]:
    """Return COMPLETE token counts from a claude -p result, subagents included.

    Claude Code reports two usage views:

    * ``usage`` -- the MAIN agent's tokens only. Task subagents' tokens are NOT
      in here, so on a multi-agent run (e.g. /swe2 fanning out) it undercounts
      the real work, and the token-derived cost does not reconcile with the
      billed ``total_cost_usd``.
    * ``modelUsage`` -- a per-model rollup that DOES include subagent tokens (all
      subagents run on ``CLAUDE_CODE_SUBAGENT_MODEL`` = the benchmarked model, so
      they fold into that model's entry). Its ``costUSD`` equals ``total_cost_usd``
      to the cent, which is the proof it is complete.

    We prefer ``modelUsage`` (summed across model entries) and fall back to
    ``usage`` only when ``modelUsage`` is absent (older Claude Code). Keys are
    normalized to input/output/cache_read/cache_creation.

    Args:
        result: The parsed JSON result object from ``claude -p``.

    Returns:
        Dict with input_tokens, output_tokens, cache_read_tokens,
        cache_creation_tokens (each an int; cache fields 0 when not reported).
    """
    model_usage = result.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        agg = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }
        for entry in model_usage.values():
            if not isinstance(entry, dict):
                continue
            agg["input_tokens"] += entry.get("inputTokens") or 0
            agg["output_tokens"] += entry.get("outputTokens") or 0
            agg["cache_read_tokens"] += entry.get("cacheReadInputTokens") or 0
            agg["cache_creation_tokens"] += entry.get("cacheCreationInputTokens") or 0
        return agg
    # Fallback: main-agent usage only (undercounts a fan-out run).
    usage = result.get("usage") or {}
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
    }


def _check_token_accounting(
    metrics: dict[str, Any],
    agent: str,
    label: str,
) -> str | None:
    """Warn loudly when output tokens per turn are implausibly low.

    Guards against the token-accounting class of bug where usage is read from a
    single scope (pi's last per-message ``usage``, or claude's main-agent-only
    ``usage``) instead of summed across every message/model, undercounting a
    multi-turn run roughly in proportion to its turn count. Such a run still writes
    a well-formed metrics.json with a plausible-looking number, so nothing else in
    the pipeline notices; the ratio is the only cheap signal.

    This warns rather than fails: the artifacts and judge scores of an affected run
    are still valid (only tokens and cost are wrong), so aborting would discard good
    work. The warning names the run so it cannot be committed unnoticed.

    Args:
        metrics: The metrics dict from ``_metrics_from_result``.
        agent: Which coding agent produced the run (for the message).
        label: Task label used in log lines.

    Returns:
        The warning message if the check tripped, else None.
    """
    turns = metrics.get("num_turns") or 0
    output_tokens = metrics.get("output_tokens") or 0
    if turns < TOKEN_SANITY_MIN_TURNS:
        return None
    per_turn = output_tokens / turns
    if per_turn >= MIN_PLAUSIBLE_OUTPUT_TOKENS_PER_TURN:
        return None
    message = (
        f"{label} TOKEN ACCOUNTING SUSPECT: {output_tokens:,} output tokens over "
        f"{turns} turns = {per_turn:.1f}/turn, below the {MIN_PLAUSIBLE_OUTPUT_TOKENS_PER_TURN} "
        f"floor. An agent making edits emits hundreds per turn, so the {agent} usage "
        f"is likely being read from one scope instead of summed across all of them "
        f"(see the pi per-message usage bug, issue #99). Scores, turns, and latency "
        f"are unaffected and still valid -- but DO NOT publish the token or cost "
        f"columns from this run without re-checking the extractor."
    )
    logger.warning("!" * 100)
    logger.warning(message)
    logger.warning("!" * 100)
    return message


# An agent's prompt-token count may fall this far below the server's own counter
# before the harness calls it a gap. The server counter is server-wide, so a
# health probe or a stray request moves it by a handful of tokens even on an idle
# box; a real gap is a whole retried request, which is orders of magnitude bigger.
SERVER_PROMPT_GAP_TOLERANCE = 0.05


def _server_prompt_token_gap(
    metrics: dict[str, Any],
    vllm_prometheus: dict[str, Any],
    concurrency: int,
) -> dict[str, Any] | None:
    """Reconcile the agent's prompt tokens against vLLM's own counter.

    An agent reports the requests it accepted. A request it abandoned and retried
    still cost the server a full prefill, and for a self-hosted model that prefill
    is real GPU time the cost per task is derived from. Codex retrying a truncated
    stream produced exactly that: vLLM counted 26,022 prompt tokens across three
    requests while codex reported 8,700 (issue #183). On a healthy path the two
    agree to the token -- a measured codex turn reported 35,508 against the
    server's 35,508 -- so this field reads 0 and says so.

    Args:
        metrics: The API-reported metrics from ``_metrics_from_result``.
        vllm_prometheus: The nested vLLM Prometheus block from ``_vllm_metrics``.
        concurrency: Tasks running at once. Above 1 the server counter aggregates
            other tasks' prefill, so the comparison is not attributable and this
            returns None rather than a misleading number.

    Returns:
        A dict with both counts, the shortfall, and a verdict; None when the
        server counter is unavailable or the run was concurrent.
    """
    if concurrency > 1:
        return None
    server_prompt = (vllm_prometheus.get("counters") or {}).get(PROMPT_TOKENS_METRIC)
    if server_prompt is None:
        return None
    agent_prompt = (
        (metrics.get("input_tokens") or 0)
        + (metrics.get("cache_read_tokens") or 0)
        + (metrics.get("cache_creation_tokens") or 0)
    )
    server_prompt = int(server_prompt)
    missing = server_prompt - agent_prompt
    tolerated = int(SERVER_PROMPT_GAP_TOLERANCE * server_prompt)
    reconciled = missing <= tolerated
    if not reconciled:
        logger.warning(
            "vLLM prefilled %d prompt tokens but the agent reported %d (%d "
            "unaccounted, %.1f%%). The agent counts only the requests it "
            "accepted, so retried or abandoned requests are missing from its "
            "tokens and from any cost derived from them (issue #183).",
            server_prompt,
            agent_prompt,
            missing,
            100.0 * missing / server_prompt if server_prompt else 0.0,
        )
    return {
        "server_prompt_tokens": server_prompt,
        "agent_prompt_tokens": agent_prompt,
        "unaccounted_prompt_tokens": max(missing, 0),
        "reconciled": reconciled,
        "note": (
            "vllm:prompt_tokens_total window delta vs the agent's own prompt "
            "tokens (input + cache_read + cache_write). A shortfall is prefill "
            "the server performed for requests the agent retried and did not "
            "count, so token-derived cost understates the GPU work (issue #183)."
        ),
    }


def _metrics_from_result(result: dict[str, Any], elapsed: float) -> dict[str, Any]:
    """Extract the benchmark metrics from a claude -p JSON result.

    Token counts come from ``_claude_token_usage`` (modelUsage-first, so subagent
    tokens are included and the counts reconcile with the billed cost).

    Args:
        result: The parsed JSON result object from `claude -p`.
        elapsed: Wall-clock seconds measured around the subprocess call.

    Returns:
        A metrics dictionary keyed by the dataset's metric names.
    """
    usage = result.get("usage") or {}
    tokens = _claude_token_usage(result)
    duration_ms = result.get("duration_ms")
    latency = round(duration_ms / 1000, 1) if duration_ms else round(elapsed, 1)
    is_error = result.get("is_error", False)
    metrics = {
        "input_tokens": tokens["input_tokens"],
        "output_tokens": tokens["output_tokens"],
        "latency_seconds": latency,
        "num_turns": result.get("num_turns", 0),
        "total_cost_usd": result.get("total_cost_usd"),
        "is_error": is_error,
        # claude -p's result subtype: "success", "error_max_turns" (hit the
        # --max-turns cap), "error_during_execution", etc. Recorded so the retry
        # logic can tell an exhausted turn budget (not retryable) from a
        # transient failure (retryable).
        "result_subtype": result.get("subtype"),
        # NOTE: the agent session UUID is intentionally NOT recorded -- metrics.json
        # is now committed to git, and a per-run session id is machine/run-specific
        # noise, not useful provenance. (Nothing downstream reads it.)
    }
    # Cache-token fields: from modelUsage (subagent-inclusive) when the backend
    # reports them (Amazon Bedrock / Anthropic API). vLLM's OpenAI route reports
    # none, so they stay absent here and the real prefix-cache utilization is
    # recovered from vLLM's Prometheus /metrics ("vllm_prometheus" block) instead.
    if tokens["cache_read_tokens"] or "cache_read_input_tokens" in usage:
        metrics["cache_read_tokens"] = tokens["cache_read_tokens"]
    if tokens["cache_creation_tokens"] or "cache_creation_input_tokens" in usage:
        metrics["cache_creation_tokens"] = tokens["cache_creation_tokens"]
    # Streaming-only: peak running estimate of extended-thinking tokens for
    # reasoning models. output_tokens already includes these; this records the
    # thinking portion the model streamed as system/thinking_tokens events.
    thinking = result.get("_thinking_tokens_estimate")
    if thinking:
        metrics["thinking_tokens_estimate"] = thinking
    # agent=kiro only: kiro-cli reports credits (not tokens); record the raw
    # credits alongside the derived total_cost_usd (credits x $/credit) so the
    # cost is auditable back to what the CLI actually charged.
    if result.get("kiro_credits") is not None:
        metrics["kiro_credits"] = result["kiro_credits"]
    # Capture the error message so failures are diagnosable from metrics.json
    # without re-running the task by hand.
    if is_error:
        metrics["error"] = str(result.get("result", ""))[:1000]
        metrics["api_error_status"] = result.get("api_error_status")
    return metrics


def _parse_prometheus_counter(text: str, metric: str) -> float | None:
    """Sum the values of a Prometheus counter across all its label sets.

    vLLM exposes one series per (engine, model_name); we sum them so a
    multi-engine server still yields a single total.

    Args:
        text: The raw text body of a Prometheus /metrics scrape.
        metric: The metric name to sum (e.g. "vllm:prefix_cache_hits_total").

    Returns:
        The summed counter value, or None if the metric is absent.
    """
    return _parse_prometheus_metrics(text)[1].get(metric)


def _parse_prometheus_metrics(
    text: str,
) -> tuple[dict[str, str], dict[str, float]]:
    """Parse a Prometheus scrape into family types and summed sample values.

    Handles the whole ``vllm:`` surface, not one metric: it reads the ``# TYPE``
    declarations to learn each family's type (counter/gauge/histogram) and sums
    every concrete sample across its label sets, so a multi-engine server still
    yields one total per sample. ``_bucket`` lines are skipped -- histogram means
    are derived from ``_sum``/``_count``, and raw buckets would only bloat output.

    Args:
        text: The raw text body of a Prometheus /metrics scrape.

    Returns:
        A ``(types, samples)`` pair. ``types`` maps each ``vllm:`` family name to
        its declared type. ``samples`` maps each concrete sample name (including
        ``_sum``/``_count`` suffixes) to its value summed across all label sets.
    """
    types: dict[str, str] = {}
    samples: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            parts = line.split()
            if len(parts) >= 4 and parts[2].startswith(VLLM_METRIC_PREFIX):
                types[parts[2]] = parts[3]
            continue
        if line.startswith("#") or not line.startswith(VLLM_METRIC_PREFIX):
            continue
        name = line.partition("{")[0].split()[0]
        if name.endswith("_bucket"):
            continue  # Raw histogram buckets; means come from _sum/_count.
        try:
            value = float(line.rsplit(None, 1)[-1])
        except ValueError:
            continue
        samples[name] = samples.get(name, 0.0) + value
    return types, samples


def _snapshot_vllm_metrics(endpoint: str) -> dict[str, Any] | None:
    """Scrape vLLM's full Prometheus /metrics surface into a comparable snapshot.

    LOUD CAVEAT -- read before trusting the numbers this feeds into metrics.json:
    these vLLM metrics are SERVER-WIDE and CUMULATIVE. They aggregate every
    request from every client since the server started and carry no per-request
    or per-session label. The per-run figures the harness derives are the DELTA
    of these across a single task's window, so they are correct ONLY when that
    benchmark task is the sole traffic hitting the endpoint during its run. Any
    concurrent request (another task, a manual curl, a second claude -p, a
    dashboard) is wrongly attributed to this run. The harness runs tasks
    serially, so a run does not contend with itself, but do not run anything else
    against the endpoint while a benchmark is going.

    Args:
        endpoint: The base URL of the vLLM server (e.g. http://127.0.0.1:8000).

    Returns:
        A ``{"types": ..., "samples": ...}`` snapshot, or None if the endpoint is
        unreachable or exposes no ``vllm:`` metrics (e.g. a non-vLLM backend).
    """
    url = endpoint.rstrip("/") + "/metrics"
    try:
        response = requests.get(url, timeout=METRICS_SCRAPE_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("Could not scrape %s for vLLM metrics: %s", url, exc)
        return None
    types, samples = _parse_prometheus_metrics(response.text)
    if not samples:
        logger.debug("Endpoint %s did not expose any vLLM metrics", url)
        return None
    return {"types": types, "samples": samples}


class _GaugePoller:
    """Poll vLLM gauges in a background thread while a run is in flight.

    Gauges (e.g. ``kv_cache_usage_perc``) are point-in-time readings: a
    before/after snapshot taken around the run reads them idle, because the KV
    cache drains once the request completes. This poller samples them every
    ``GAUGE_POLL_INTERVAL_SECONDS`` while claude -p runs, so peak and mean reflect
    what actually happened DURING the run -- matching what the vLLM server log
    prints for an in-flight request.

    Use as a context manager around the claude -p call:

        with _GaugePoller(endpoint) as poller:
            result = _run_claude(...)
        summary = poller.summary()

    The peak still carries the single-tenant caveat: on a shared endpoint it
    reflects total server load, not this run alone.
    """

    def __init__(self, endpoint: str | None) -> None:
        # A None endpoint (e.g. provider=bedrock) disables polling: the thread
        # never starts and summary() reports the gauges as unavailable.
        self._url = endpoint.rstrip("/") + "/metrics" if endpoint else None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        # Per-metric list of sampled values (only successful scrapes recorded).
        self._samples: dict[str, list[float]] = {m: [] for m in SAMPLED_GAUGE_METRICS}

    def __enter__(self) -> _GaugePoller:
        if self._url:
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._url:
            self._thread.join(timeout=METRICS_SCRAPE_TIMEOUT_SECONDS)

    def _run(self) -> None:
        # Sample immediately, then every interval until stopped. wait() returns
        # True when the stop event is set, giving a prompt, drift-free exit.
        while True:
            self._sample_once()
            if self._stop.wait(GAUGE_POLL_INTERVAL_SECONDS):
                return

    def _sample_once(self) -> None:
        try:
            response = requests.get(self._url, timeout=METRICS_SCRAPE_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("Gauge poll of %s failed: %s", self._url, exc)
            return
        _, samples = _parse_prometheus_metrics(response.text)
        for metric in SAMPLED_GAUGE_METRICS:
            if metric in samples:
                self._samples[metric].append(samples[metric])

    def summary(self) -> dict[str, Any]:
        """Return per-gauge peak/mean/sample-count, or an unavailable marker.

        Returns:
            A dict describing the sampled gauges. ``available`` is False when no
            gauge was ever successfully sampled (endpoint unreachable or not
            vLLM). Otherwise each sampled gauge maps to ``{"peak", "mean",
            "samples"}``, with peak/mean None for a gauge the endpoint did not
            expose.
        """
        if not any(self._samples.values()):
            return {
                "available": False,
                "source": "vllm_prometheus_poll",
                "note": (
                    "No vLLM gauges were sampled during the run; endpoint "
                    "unreachable or not a vLLM server."
                ),
                "interval_seconds": GAUGE_POLL_INTERVAL_SECONDS,
                "gauges": {},
            }
        gauges: dict[str, Any] = {}
        for metric, values in self._samples.items():
            if values:
                gauges[metric] = {
                    "peak": round(max(values), 4),
                    "mean": round(sum(values) / len(values), 4),
                    "samples": len(values),
                }
            else:
                gauges[metric] = {"peak": None, "mean": None, "samples": 0}
        return {
            "available": True,
            "source": "vllm_prometheus_poll",
            "note": (
                "Peak/mean of gauges sampled every "
                f"{GAUGE_POLL_INTERVAL_SECONDS}s while the run was in flight. "
                "Peak still reflects total server load under the single-tenant "
                "assumption, not this run in isolation."
            ),
            "interval_seconds": GAUGE_POLL_INTERVAL_SECONDS,
            "gauges": gauges,
        }


def _sample_delta(
    before: dict[str, float], after: dict[str, float], name: str
) -> float | None:
    """Return the non-negative window delta of one sample, or None if absent.

    A sample missing from either snapshot yields None rather than 0, so the block
    distinguishes "the endpoint does not expose this" from "it happened zero
    times during the run".
    """
    if name not in before or name not in after:
        return None
    return max(0.0, after[name] - before[name])


def _num(value: float | None) -> int | float | None:
    """Render a metric value as an int when whole, else rounded, preserving None."""
    if value is None:
        return None
    return int(value) if float(value).is_integer() else round(value, 4)


def _rate(numerator: float | None, denominator: float | None) -> float | None:
    """Return numerator/denominator rounded to 4 dp, or None if not computable."""
    if numerator is None or not denominator:
        return None
    return round(numerator / denominator, 4)


def _vllm_metrics(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> dict[str, Any]:
    """Derive the nested vLLM Prometheus block for a run -- the FULL surface.

    This block is kept SEPARATE from the top-level metrics on purpose: the
    top-level fields report only what the model API returned per request,
    whereas these numbers come from a different source and a different method --
    a window delta of vLLM's SERVER-WIDE, CUMULATIVE Prometheus metrics. See
    _snapshot_vllm_metrics for the single-tenant caveat: each delta equals this
    run's activity only when the run is the sole traffic on the endpoint.

    Every ``vllm:`` metric is reported (duplicates of the top-level API numbers
    included) under its own type-named group, so the source is unambiguous:

    - ``counters``: window delta of each counter (e.g. generation/prompt tokens,
      prefix-cache queries/hits, preemptions, request successes).
    - ``histograms``: per-family ``count``/``sum`` window deltas plus the derived
      window ``mean`` (e.g. mean TTFT, mean end-to-end latency).
    - ``gauges``: an instantaneous post-run reading. Gauges are point-in-time, so
      between serial tasks they typically read idle (0); continuous sampling
      (the DuckDB collector) is the way to capture peaks.
    - ``derived``: the headline cache hit rates computed from the counters. In
      vLLM v1 a prefix-cache hit IS a KV-cache hit -- there is no separate KV-hit
      counter, and vLLM does not publish the rate itself.

    ``_created`` timestamp series are dropped as noise; histogram ``_bucket``
    lines are omitted (means come from ``_sum``/``_count``).

    Args:
        before: Snapshot taken immediately before the claude -p call.
        after: Snapshot taken immediately after it.

    Returns:
        A nested dict describing the run's vLLM-side activity. When snapshots are
        missing (endpoint does not expose /metrics), ``available`` is False and
        the groups are empty.
    """
    unavailable_note = (
        "vLLM metrics were not reachable; run against a vLLM endpoint exposing "
        "/metrics to populate this block."
    )
    available_note = (
        "Window delta of server-wide vLLM metrics; accurate only if this run was "
        "the sole traffic on the endpoint during its execution. Gauges are an "
        "instantaneous post-run reading and typically read idle between tasks."
    )
    if before is None or after is None:
        return {
            "available": False,
            "source": "vllm_prometheus_window",
            "note": unavailable_note,
            "derived": {
                "prefix_cache_hit_rate": None,
                "prompt_tokens_cached_rate": None,
            },
            "counters": {},
            "histograms": {},
            "gauges": {},
        }
    types: dict[str, str] = after["types"]
    before_s: dict[str, float] = before["samples"]
    after_s: dict[str, float] = after["samples"]
    counters: dict[str, Any] = {}
    histograms: dict[str, Any] = {}
    gauges: dict[str, Any] = {}
    for family, mtype in sorted(types.items()):
        if family.endswith("_created"):
            continue  # Prometheus per-series creation timestamps; pure noise.
        if mtype == "counter":
            counters[family] = _num(_sample_delta(before_s, after_s, family))
        elif mtype == "histogram":
            dcount = _sample_delta(before_s, after_s, family + "_count")
            dsum = _sample_delta(before_s, after_s, family + "_sum")
            histograms[family] = {
                "count": _num(dcount),
                "sum": _num(dsum),
                "mean": round(dsum / dcount, 6)
                if dcount and dsum is not None
                else None,
            }
        elif mtype == "gauge":
            gauges[family] = _num(after_s.get(family))
    return {
        "available": True,
        # Windowed deltas of server-wide metrics (single-tenant assumption), NOT
        # per-request accounting from the model API response.
        "source": "vllm_prometheus_window",
        "note": available_note,
        "derived": {
            "prefix_cache_hit_rate": _rate(
                _sample_delta(before_s, after_s, PREFIX_CACHE_HITS_METRIC),
                _sample_delta(before_s, after_s, PREFIX_CACHE_QUERIES_METRIC),
            ),
            "prompt_tokens_cached_rate": _rate(
                _sample_delta(before_s, after_s, PROMPT_TOKENS_CACHED_METRIC),
                _sample_delta(before_s, after_s, PROMPT_TOKENS_METRIC),
            ),
        },
        "counters": counters,
        "histograms": histograms,
        "gauges": gauges,
    }


def _mark_aggregate(vllm_block: dict[str, Any]) -> None:
    """Annotate a vLLM block in place as a concurrency-aggregated measurement.

    Called only when the run overlapped other tasks (concurrency > 1). The window
    deltas still measure REAL server/GPU activity -- they are not junk -- but they
    are server-wide aggregates over a window shared by every concurrent task, not
    this run in isolation: ratios (hit rates) become the aggregate rate across all
    overlapping runs, while absolute counts (token/request deltas) are summed
    across them, so each of the N overlapping files carries a near-identical,
    inflated figure. The per-run API fields in metrics_that_matter are unaffected
    and stay correct. No-op when the block is unavailable.

    Args:
        vllm_block: The dict returned by _vllm_metrics, mutated in place.
    """
    if not vllm_block.get("available"):
        return
    vllm_block["single_tenant"] = False
    vllm_block["note"] = (
        "AGGREGATE (concurrency > 1): server-wide window deltas over a period "
        "shared by other in-flight tasks. Ratios are the real aggregate rate "
        "across all overlapping runs; absolute counts are summed across them "
        "(inflated and near-duplicated per file). Not isolated to this run. Use "
        "concurrency 1 for per-run vLLM metrics. " + vllm_block["note"]
    )


def _run_claude(cmd: list[str], env: dict[str, str], timeout: int) -> dict[str, Any]:
    """Run `claude -p` and parse its JSON result.

    Args:
        cmd: The command argument vector.
        env: Environment for the subprocess.
        timeout: Wall-clock timeout in seconds.

    Returns:
        The parsed JSON result object.

    Raises:
        RuntimeError: If claude times out, exits nonzero, or emits no JSON.
    """
    start = time.time()
    try:
        proc = subprocess.run(  # nosec B603 - hardcoded 'claude', list args, no shell
            cmd,
            env=env,
            # Run from the repo root so the /swe skill's artifact paths (written
            # relative to the repo root, e.g. benchmarks/swe-benchmark-data/...)
            # resolve correctly. Without this, cwd is wherever the harness was
            # invoked (typically benchmarks/), and a model that writes a relative
            # path doubles it to benchmarks/benchmarks/... and the run scores 0/4.
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"claude -p timed out after {timeout}s") from exc
    elapsed = time.time() - start

    if not proc.stdout.strip():
        raise RuntimeError(
            f"claude -p produced no output (exit {proc.returncode}): "
            f"{proc.stderr.strip()[:500]}"
        )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"claude -p output was not JSON: {proc.stdout.strip()[:500]}"
        ) from exc
    result["_elapsed_seconds"] = round(elapsed, 1)
    return result


def _pi_result_from_events(
    events: list[dict[str, Any]], elapsed: float
) -> dict[str, Any]:
    """Normalize pi's JSON-lines event stream into the claude-shaped result dict.

    pi ``--mode json`` emits a stream of events, not one result object. The final
    ``agent_end`` carries the settled conversation, and ``stopReason`` reports why
    it stopped. Turns are counted from ``turn_start`` events.

    Token usage is PER-MESSAGE, not cumulative: each assistant message carries its
    own ``usage`` (``input``/``output``/``cacheRead``/``cacheWrite``), matching pi's
    own ``UsageTotals`` accounting (``addUsageToTotals`` is called once per message).
    Reading only the last message's usage undercounts a multi-turn run by ~100x
    (a 200-turn edit run would report only the final turn's tokens), so we SUM
    usage across every assistant message -- the same fix as sourcing claude's tokens
    from ``modelUsage`` rather than the main-agent-only ``usage``. This maps onto the
    keys ``_metrics_from_result`` reads for claude so the harness stays agent-agnostic.

    Args:
        events: The parsed JSON-lines events pi emitted, in order.
        elapsed: Wall-clock seconds measured around the subprocess call.

    Returns:
        A result dict shaped like ``claude -p``'s JSON result.

    Raises:
        RuntimeError: If the stream carried no ``agent_end`` (pi crashed or
            produced no parseable result).
    """
    num_turns = sum(1 for e in events if e.get("type") == "turn_start")
    agent_end = next(
        (e for e in reversed(events) if e.get("type") == "agent_end"), None
    )
    if agent_end is None:
        raise RuntimeError("pi emitted no agent_end event (no parseable result)")
    messages = agent_end.get("messages") or []
    assistant_msgs = [
        m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
    ]
    # Usage is summed from the WHOLE event stream, not from agent_end.messages.
    #
    # agent_end carries the SETTLED conversation, and omp resets that list on every
    # extra agent_start (context compaction, and the todo reminder that nudges the
    # agent to keep going). So agent_end reports only the messages since the last
    # restart, and every token before it is silently dropped -- by 14x to 704x on
    # output across the runs this was measured on (issue #157).
    #
    # Verified against 231 saved omp streams: on the 200 with a single agent_start
    # the two sums are IDENTICAL per message, to the token; on the 30 with more,
    # agent_end.messages is an exact SUFFIX of the stream. agent_end can only ever
    # carry less, never anything different, so summing the stream is strictly safer.
    #
    # Only assistant messages carry a usage object (toolResult / user / custom
    # message_end events have none), so no role filter is needed. turn_end mirrors
    # message_end and is skipped, or every message would count twice.
    totals = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "cost": 0.0}

    def _accumulate(usage: dict[str, Any]) -> None:
        """Add one message's usage into ``totals`` (see enclosing docstring)."""
        totals["input"] += usage.get("input") or 0
        totals["output"] += usage.get("output") or 0
        totals["cacheRead"] += usage.get("cacheRead") or 0
        totals["cacheWrite"] += usage.get("cacheWrite") or 0
        # Cost, when present, is per-message and additive (pi accrues it into
        # UsageTotals.cost the same way). Shapes vary by pi version: a bare number
        # or a {"total": ...} object -- accept both.
        cost = usage.get("cost")
        if isinstance(cost, dict):
            cost = cost.get("total")
        if isinstance(cost, (int, float)):
            totals["cost"] += cost

    counted = 0
    for event in events:
        if event.get("type") != "message_end":
            continue
        usage = (event.get("message") or {}).get("usage")
        if isinstance(usage, dict):
            _accumulate(usage)
            counted += 1
    # Fall back to the settled conversation when the stream carried no per-message
    # usage at all: pi emits no message_end, and for a single-agent_start run the
    # two sources agree exactly anyway.
    if not counted:
        for m in assistant_msgs:
            _accumulate(m.get("usage") or {})
    stop_reason = assistant_msgs[-1].get("stopReason") if assistant_msgs else None
    # remap onto the keys _metrics_from_result reads for claude.
    remapped_usage: dict[str, Any] = {
        "input_tokens": totals["input"],
        "output_tokens": totals["output"],
    }
    # Cache tokens: against a vLLM endpoint pi reports 0 (real reuse comes from the
    # Prometheus block, so we omit them rather than record a misleading 0). Against
    # Amazon Bedrock pi DOES report native prompt-cache usage (cacheRead/cacheWrite)
    # -- pass those through under the keys _metrics_from_result expects, but only
    # when non-zero so the vLLM path stays clean.
    if totals["cacheRead"]:
        remapped_usage["cache_read_input_tokens"] = totals["cacheRead"]
    if totals["cacheWrite"]:
        remapped_usage["cache_creation_input_tokens"] = totals["cacheWrite"]
    cost = totals["cost"] or None
    # An error retry (pi tried and gave up) or a non-"stop" terminal reason marks
    # the run failed so the harness's retry/failure logic can react.
    will_retry = agent_end.get("willRetry", False)
    is_error = bool(will_retry) or stop_reason not in (None, "stop", "end_turn")
    return {
        "usage": remapped_usage,
        "num_turns": num_turns,
        # pi reports 0 cost against a local vLLM endpoint (the real per-task cost is
        # hardware-derived; see cost-per-task-methodology.md), so keep None there
        # rather than a misleading 0. Against Amazon Bedrock pi reports a REAL
        # metered cost -- pass it through unchanged.
        "total_cost_usd": cost if cost else None,
        "is_error": is_error,
        # Map pi's stopReason onto the subtype the retry logic keys on. pi has no
        # turn cap (no error_max_turns analogue), so a clean stop is "success".
        "subtype": "success" if stop_reason in ("stop", "end_turn") else stop_reason,
        "duration_ms": round(elapsed * 1000),
        "result": stop_reason or "",
    }


def _run_pi(
    cmd: list[str],
    env: dict[str, str],
    timeout: int,
    stream_log: Path | None = None,
) -> dict[str, Any]:
    """Run ``pi -p --mode json`` and normalize its event stream to a result dict.

    Args:
        cmd: The pi command argument vector.
        env: Environment for the subprocess (pins PI_CODING_AGENT_DIR).
        timeout: Wall-clock timeout in seconds.
        stream_log: Optional file to append pi's events to as they arrive. A task
            runs for hours with no output otherwise, and -- worse -- a timeout
            discards the buffered stdout entirely, so a run that hits the wall
            leaves NO evidence of what the agent did. Mirroring as we read keeps
            that evidence, and lets a run in flight be followed with tail -f.

    Returns:
        The claude-shaped result dict (see ``_pi_result_from_events``).

    Raises:
        RuntimeError: If pi times out, emits no output, or emits no agent_end.
    """
    start = time.time()
    # Read stdout line by line rather than with subprocess.run so the stream can
    # be mirrored to stream_log while the task runs. run() buffers everything
    # until exit and drops it on timeout, which is how a 2-hour task can fail
    # leaving nothing to diagnose.
    sink = None
    if stream_log is not None:
        stream_log.parent.mkdir(parents=True, exist_ok=True)
        sink = stream_log.open("a", encoding="utf-8")
    stdout_lines: list[str] = []
    events: list[dict[str, Any]] = []
    try:
        proc = subprocess.Popen(  # nosec B603 - hardcoded 'pi', list args, no shell
            cmd,
            env=env,
            # Run from the repo root so the skill's artifact paths resolve, exactly
            # as for the claude path (see the note in _run_claude).
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        try:
            for line in proc.stdout or []:
                stdout_lines.append(line)
                if sink is not None:
                    sink.write(line)
                    sink.flush()
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    events.append(json.loads(stripped))
                except json.JSONDecodeError:
                    # pi may interleave non-JSON diagnostics; skip them.
                    continue
            proc.wait(timeout=max(timeout - (time.time() - start), 1))
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            raise RuntimeError(f"pi -p timed out after {timeout}s") from exc
        stderr = (proc.stderr.read() if proc.stderr else "") or ""
    finally:
        if sink is not None:
            sink.close()
    elapsed = time.time() - start

    if not "".join(stdout_lines).strip():
        raise RuntimeError(
            f"pi -p produced no output (exit {proc.returncode}): {stderr.strip()[:500]}"
        )
    if not events:
        raise RuntimeError(
            f"pi -p output had no JSON events: {''.join(stdout_lines).strip()[:500]}"
        )
    # Debug aid: dump the raw event stream when PI_RAW_EVENTS_DUMP is set, so the
    # usage-summation logic can be validated against real pi output.
    dump_path = os.environ.get("PI_RAW_EVENTS_DUMP")
    if dump_path:
        with open(dump_path, "w", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")
    result = _pi_result_from_events(events, elapsed)
    result["_elapsed_seconds"] = round(elapsed, 1)
    return result


def _kiro_result_from_output(
    output: str,
    returncode: int,
    elapsed: float,
    dollars_per_credit: float,
) -> dict[str, Any]:
    """Normalize a kiro-cli run to the claude-shaped result dict.

    kiro-cli emits ANSI-colored narration and a one-line summary, e.g.
    ``▸ Credits: 0.21 • Time: 17s``. It reports no token counts, so input/output
    tokens are 0 and ``num_turns`` is 0; the cost signal is credits, turned into
    dollars via the configured per-credit rate. Success is gated on the process
    exit code (kiro-cli returns non-zero on failure).

    Args:
        output: The captured combined stdout+stderr (carries the Credits/Time
            summary line).
        returncode: The process exit code.
        elapsed: Wall-clock seconds measured by the harness.
        dollars_per_credit: USD per credit for the cost estimate.

    Returns:
        The claude-shaped result dict, with ``kiro_credits`` added for provenance.
    """
    clean = _ANSI_ESCAPE_RE.sub("", output)
    credits_match = _KIRO_CREDITS_RE.search(clean)
    time_match = _KIRO_TIME_RE.search(clean)
    credits = float(credits_match.group(1)) if credits_match else None
    reported_s = float(time_match.group(1)) if time_match else None
    cost = round(credits * dollars_per_credit, 6) if credits is not None else None
    is_error = returncode != 0
    return {
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "num_turns": 0,
        "total_cost_usd": cost,
        "is_error": is_error,
        "subtype": "success" if not is_error else f"exit_{returncode}",
        "duration_ms": round(
            (reported_s if reported_s is not None else elapsed) * 1000
        ),
        "result": "" if is_error else "stop",
        "kiro_credits": credits,
    }


def _run_kiro(
    cmd: list[str],
    env: dict[str, str],
    timeout: int,
    dollars_per_credit: float,
) -> dict[str, Any]:
    """Run ``kiro-cli chat --no-interactive``, streaming its trace, and normalize output.

    kiro-cli streams ANSI narration on stdout and prints its ``Credits/Time``
    summary on stderr. We merge the two (``stderr=STDOUT``) and echo each line to
    this process's stderr as it arrives -- a live trace, the kiro analogue of
    Claude Code's ``--stream`` mode -- while accumulating the combined text for
    metrics parsing. The per-line timeout check mirrors ``_run_claude_streaming``.

    Args:
        cmd: The kiro-cli command argument vector.
        env: Environment for the subprocess.
        timeout: Wall-clock timeout in seconds.
        dollars_per_credit: USD per credit for the cost estimate.

    Returns:
        The claude-shaped result dict (see ``_kiro_result_from_output``).

    Raises:
        RuntimeError: If kiro-cli times out or produces no output.
    """
    start = time.time()
    proc = subprocess.Popen(  # nosec B603 - hardcoded 'kiro-cli', list args, no shell
        cmd,
        env=env,
        # Run from the repo root so the skill's artifact paths resolve, exactly as
        # for the claude and pi paths.
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout is None:  # pragma: no cover - stdout is always a pipe here
        raise RuntimeError("kiro-cli produced no stdout stream")
    captured: list[str] = []
    for line in proc.stdout:
        if time.time() - start > timeout:
            proc.kill()
            raise RuntimeError(f"kiro-cli timed out after {timeout}s")
        # Echo the model's live trace so it shows up in the harness log/terminal.
        sys.stderr.write(line)
        sys.stderr.flush()
        captured.append(line)
    proc.wait()
    elapsed = time.time() - start

    combined = "".join(captured)
    if not combined.strip():
        raise RuntimeError(f"kiro-cli produced no output (exit {proc.returncode}).")
    result = _kiro_result_from_output(
        combined, proc.returncode, elapsed, dollars_per_credit
    )
    result["_elapsed_seconds"] = round(elapsed, 1)
    return result


CODEX_BIN = "codex"

# item.completed item types that count as one agent step for num_turns: an
# assistant message or a shell command the model issued.
CODEX_STEP_ITEM_TYPES = frozenset({"agent_message", "command_execution"})


# Name of the codex provider block the harness defines for an OpenAI-compatible
# endpoint. codex resolves the base URL from the provider its config selects, so
# the harness declares one per run rather than relying on whatever
# ~/.codex/config.toml happens to set.
CODEX_ENDPOINT_PROVIDER = "harness_endpoint"


def _codex_base_url(endpoint: str | None) -> str:
    """Return the ``/v1`` base URL for codex's provider block, or fail loudly.

    The provider block travels in argv, so this value is visible to any local
    user through ``ps`` and is printed by ``--dry-run``. A URL carrying userinfo
    (``https://user:pass@host``) would put that credential there, so it is
    rejected rather than trimmed: the harness authenticates with
    ``OPENAI_API_KEY``, which stays in the environment, and no path here needs
    credentials in the URL.

    Args:
        endpoint: The configured endpoint (``--endpoint`` or the runner config).

    Returns:
        The endpoint with a single ``/v1`` suffix.

    Raises:
        ValueError: If the endpoint is missing, is not http(s), or embeds
            credentials.
    """
    raw = (endpoint or "").strip()
    if not raw:
        raise ValueError(
            "agent=codex with provider=endpoint needs an endpoint. Pass "
            "--endpoint http://127.0.0.1:8000 or set it in the runner config."
        )
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"codex endpoint must be http or https, got '{raw}'. codex 0.153.4 "
            "speaks the Responses API over HTTP only."
        )
    if parsed.username or parsed.password:
        raise ValueError(
            "codex endpoint must not embed credentials: the base URL is passed "
            "on the command line, where any local user can read it from ps. "
            "Use the endpoint host alone and put the key in api_key, which the "
            "harness passes through the OPENAI_API_KEY environment variable."
        )
    return raw.rstrip("/") + "/v1"


def _build_codex_env(config: RunnerConfig) -> dict[str, str]:
    """Build the environment for a codex exec run.

    For provider=bedrock, pins AWS_REGION so codex uses the right region.
    For provider=endpoint, sets OPENAI_API_KEY, which the provider block in
    ``_build_codex_cmd`` names as its ``env_key``. OPENAI_BASE_URL is set too,
    but only for older codex builds that still read it: codex 0.153.4 ignores
    it, which is why the base URL now travels in the provider block instead
    (issue #183).

    Args:
        config: The runner config.

    Returns:
        A copy of the current environment with routing vars set.
    """
    env = os.environ.copy()
    if config.is_bedrock:
        region = config.resolved_region()
        if region:
            env["AWS_REGION"] = region
    else:
        env["OPENAI_BASE_URL"] = (config.endpoint or "").rstrip("/") + "/v1"
        env["OPENAI_API_KEY"] = config.api_key or "local"
    return env


def _build_codex_cmd(config: RunnerConfig, clone_path: Path, prompt: str) -> list[str]:
    """Assemble the ``codex exec`` argument vector.

    codex exec runs non-interactively, outputs JSON lines, and supports
    ``--model`` to select any Bedrock or OpenAI-compatible model. ``--cd``
    sets the working directory to the cloned repo so codex file tools operate
    on the task. The SKILL.md is inlined ahead of the task payload (codex has
    no ``--skill`` flag, same as kiro).

    On ``--dangerously-bypass-approvals-and-sandbox``: every harness here runs
    unattended against a throwaway clone, so all of them pre-approve tool use
    (claude uses bypassPermissions, omp --auto-approve, kiro --trust-all-tools).
    codex additionally wraps model-issued shell commands in a bubblewrap
    sandbox, and that sandbox cannot start on the EC2 hosts this benchmark runs
    on: both ``--sandbox read-only`` and ``--sandbox workspace-write`` abort
    with ``bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted``
    (creating a loopback interface in an unprivileged user namespace is not
    permitted there), so the agent completes no shell work at all. Bypassing is
    therefore required for the run to function, not a convenience. The blast
    radius is the benchmark host itself, so run this agent only on a disposable
    instance -- the same assumption the other harnesses already make.

    ROUTING (issue #183). codex takes its base URL from the provider its config
    selects, and codex 0.153.4 ignores ``OPENAI_BASE_URL`` entirely. Exporting
    that variable alone therefore sent an endpoint run wherever
    ``~/.codex/config.toml`` pointed -- on a machine configured for the judge,
    straight to Amazon Bedrock, which answered ``404 The model
    'minicpm5-2b' does not exist``. So provider=endpoint declares its own
    provider block here. ``wire_api`` must be ``responses``: codex 0.153.4
    removed the chat-completions wire and rejects ``wire_api = "chat"``.

    A vLLM server therefore needs a tool-call parser that accepts
    Responses-shaped tools (flat ``{"type": "function", "name": ...}``).
    ``qwen3_coder`` and ``hermes`` do; ``minicpm5xml``, ``dots``, ``hy_v3``,
    ``hy_v4``, ``rust``, ``step3`` and ``step3p5`` read the nested
    chat-completions shape and abort tool extraction, which truncates the
    stream and makes codex retry every request.

    Args:
        config: The runner config (model, provider).
        clone_path: Path to the cloned task repo.
        prompt: The hydrated task prompt (from ``_build_prompt`` agent="codex").

    Returns:
        The command as a list of arguments (never a shell string).
    """
    skill_md = _skill_path(config).read_text(encoding="utf-8")
    full_prompt = (
        f"{skill_md}\n\n"
        "===TASK===\n"
        "Follow the skill instructions above to complete the following task.\n\n"
        f"{prompt}"
    )
    cmd = [
        CODEX_BIN,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd",
        str(clone_path),
        "--model",
        model_to_wire_id(config.model or ""),
    ]
    if config.is_bedrock:
        cmd += ["-c", "model_provider=amazon-bedrock"]
    else:
        base_url = _codex_base_url(config.endpoint)
        provider = CODEX_ENDPOINT_PROVIDER
        cmd += [
            "-c",
            f"model_provider={provider}",
            "-c",
            f"model_providers.{provider}.name={provider}",
            "-c",
            f"model_providers.{provider}.base_url={base_url}",
            "-c",
            f"model_providers.{provider}.wire_api=responses",
            "-c",
            f"model_providers.{provider}.env_key=OPENAI_API_KEY",
        ]
        # A self-hosted model is unknown to codex, which then warns "Model
        # metadata not found. Defaulting to fallback metadata" and sizes its
        # context from that guess. Tell it the served window when the config
        # knows it.
        if config.context_window > 0:
            cmd += ["-c", f"model_context_window={config.context_window}"]
    # Config-derived values (model, clone path, prompt) are passed as separate
    # list elements, never interpolated into a shell string, and the executable
    # is the hardcoded CODEX_BIN.
    cmd += ["--", full_prompt]
    return cmd


def _codex_result_from_events(
    events: list[dict[str, Any]],
    returncode: int,
    elapsed: float,
    model: str = "",
) -> dict[str, Any]:
    """Normalize codex JSON-lines output to the claude-shaped result dict.

    codex exec emits JSON-lines events. The ``turn.completed`` event carries
    token usage; ``item.completed`` events carry the agent's steps. Cost is
    derived from the token counts using the local Bedrock price table in
    ``bedrock_pricing``, because codex exec reports no billed cost of its own.

    codex's ``input_tokens`` is the TOTAL prompt size, with
    ``cached_input_tokens`` and ``cache_write_input_tokens`` as subsets of it
    (a measured turn: input 50768 = cached 36741 + cache_write 13817 + fresh
    210). The claude-shaped ``usage`` this harness passes around is additive
    instead -- ``input_tokens`` there means fresh, non-cached tokens, which is
    also what ``bedrock_pricing.cost_usd`` documents -- so the cached and
    cache-written portions are subtracted out here. Without that subtraction
    the cached prompt is billed twice, once at the full input rate and again
    at the cache rate, which overstates cost by roughly 80% on a
    cache-dominated agentic run. This mirrors the partition handling in
    ``token_accounting.py`` (see issue #136). Because the fields this returns
    are disjoint, every total derived from them must be additive: callers pass
    ``cache_partition=False`` via ``cache_partition_for_agent`` (issue #183).

    One ``turn.completed`` carries the whole turn, VERIFIED against the server:
    codex reported input 35508 / output 180 for a three-tool-call turn while
    vLLM's own counters moved 35508 prompt / 180 generation tokens across the
    turn's 4 requests. The counts here SUM across ``turn.completed`` events
    anyway, so a build that splits one exec into several turns (resume,
    compaction) cannot silently drop all but the last.

    Retries are the one gap: codex counts the attempt it accepted and ignores
    the ones it abandoned, so a flaky endpoint bills the GPU for prefill that
    never reaches these numbers. ``_save_metrics`` records that gap for an
    endpoint run by comparing this usage against vLLM's prompt-token counter.

    ``reasoning_output_tokens`` is deliberately NOT added to ``output_tokens``:
    it is a subset of it, not a sibling.

    Args:
        events: Parsed JSON event dicts from codex exec stdout.
        returncode: The process exit code.
        elapsed: Wall-clock seconds measured by the harness.
        model: The model id used for cost derivation.

    Returns:
        The claude-shaped result dict.
    """
    usage: dict[str, int] = {}
    last_message = ""
    agent_steps = 0
    for event in events:
        if event.get("type") == "turn.completed":
            u = event.get("usage", {})
            total_input = int(u.get("input_tokens", 0) or 0)
            cache_read = int(u.get("cached_input_tokens", 0) or 0)
            cache_write = int(u.get("cache_write_input_tokens", 0) or 0)
            # Clamp: a future codex build reporting these as siblings rather
            # than subsets must not produce a negative fresh-token count.
            fresh_input = max(total_input - cache_read - cache_write, 0)
            usage = {
                "input_tokens": usage.get("input_tokens", 0) + fresh_input,
                "output_tokens": usage.get("output_tokens", 0)
                + int(u.get("output_tokens", 0) or 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0)
                + cache_read,
                "cache_creation_input_tokens": usage.get(
                    "cache_creation_input_tokens", 0
                )
                + cache_write,
            }
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") in CODEX_STEP_ITEM_TYPES:
                agent_steps += 1
            if item.get("type") == "agent_message":
                last_message = item.get("text", "")

    cost = (
        _bedrock_cost_usd(
            model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
        )
        if model
        else None
    )

    is_error = returncode != 0
    return {
        "usage": usage,
        # codex emits exactly one turn.completed per exec, so that count says
        # nothing about how hard the agent worked. Count the agent's completed
        # steps instead -- messages and shell commands -- which is the closest
        # analogue to the turn counts the other harnesses report.
        "num_turns": agent_steps,
        "total_cost_usd": cost,
        "is_error": is_error,
        "subtype": "success" if not is_error else f"exit_{returncode}",
        "duration_ms": round(elapsed * 1000),
        "result": last_message if not is_error else f"exit_{returncode}",
    }


def _run_codex(
    cmd: list[str],
    env: dict[str, str],
    timeout: int,
    model: str = "",
) -> dict[str, Any]:
    """Run ``codex exec --json`` and normalize its JSON-lines output.

    codex exec streams JSON-lines events to stdout. We accumulate them and
    parse on completion. Each line is also echoed to stderr as a live trace.

    The deadline is enforced by a watchdog timer rather than by checking the
    clock inside the read loop: a codex process that stalls without writing a
    line leaves that loop blocked on read, so a clock check there never runs
    and the run would hang until the outer harness gave up. The watchdog kills
    the process at the deadline, which ends the read loop and lets this
    function raise. That matters because run-multi-model-benchmark.sh drives
    unattended multi-day batches.

    Args:
        cmd: The codex exec command argument vector.
        env: Environment for the subprocess.
        timeout: Wall-clock timeout in seconds.
        model: The model id used for cost derivation.

    Returns:
        The claude-shaped result dict (see ``_codex_result_from_events``).

    Raises:
        RuntimeError: If codex times out or produces no output.
    """
    start = time.time()
    proc = subprocess.Popen(  # nosec B603 B607 - hardcoded 'codex', list args, no shell
        cmd,
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout is None:  # pragma: no cover
        raise RuntimeError("codex produced no stdout stream")

    timed_out = threading.Event()

    def _kill_on_deadline() -> None:
        timed_out.set()
        proc.kill()

    watchdog = threading.Timer(timeout, _kill_on_deadline)
    watchdog.daemon = True
    watchdog.start()

    events: list[dict[str, Any]] = []
    try:
        for line in proc.stdout:
            sys.stderr.write(line)
            sys.stderr.flush()
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                # codex interleaves human-readable diagnostics with the JSON
                # event stream; skip them the way the pi/omp readers do.
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
        try:
            proc.wait(timeout=max(timeout - (time.time() - start), 1))
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            raise RuntimeError(f"codex exec timed out after {timeout}s") from exc
    finally:
        watchdog.cancel()
        # The pipe stays open when the read loop exits via an exception (a
        # killed child), so close it here rather than leaking the descriptor
        # across a multi-task run.
        proc.stdout.close()

    if timed_out.is_set():
        raise RuntimeError(f"codex exec timed out after {timeout}s")

    elapsed = time.time() - start

    if not events:
        raise RuntimeError(f"codex produced no output (exit {proc.returncode}).")
    result = _codex_result_from_events(events, proc.returncode, elapsed, model=model)
    result["_elapsed_seconds"] = round(elapsed, 1)
    return result


TOOL_RESULT_PREVIEW_CHARS = 500


def _tool_result_text(content: Any) -> str:
    """Flatten a tool_result block's content into plain text.

    The Anthropic message format allows a tool_result's ``content`` to be either
    a plain string or a list of content blocks (each a ``{"type": "text",
    "text": ...}`` mapping, though other block types may appear). This joins the
    text it can find so the trace can show what a tool actually returned.

    Args:
        content: The ``content`` field of a tool_result block.

    Returns:
        The extracted text, stripped. Empty when no text could be found.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "\n".join(texts).strip()
    return ""


def _truncate(text: str, limit: int, verbose: bool) -> str:
    """Return text as-is when verbose, else truncated to limit with a marker.

    Args:
        text: The text to (maybe) truncate.
        limit: Max characters to keep when not verbose.
        verbose: When True, never truncate.

    Returns:
        The full text (verbose) or a truncated preview with a "+N chars" tail.
    """
    if verbose or len(text) <= limit:
        return text
    return f"{text[:limit]}... (+{len(text) - limit} chars)"


def _format_stream_event(event: dict[str, Any], verbose: bool = False) -> str | None:
    """Render one stream-json event as a human-readable trace line.

    Args:
        event: A single parsed event object from `--output-format stream-json`.
        verbose: When True, print assistant text and tool results in full
            instead of truncating them (thinking is always shown in full).

    Returns:
        A summary to print, or None for events not worth showing.
    """
    etype = event.get("type")
    if etype == "system":
        subtype = event.get("subtype", "")
        # Reasoning models (e.g. Kimi K2 Thinking) stream a running estimate of
        # extended-thinking tokens as system/thinking_tokens events. Surface the
        # count instead of a bare, repeated subtype line.
        if subtype == "thinking_tokens":
            est = event.get("estimated_tokens")
            return f"[system] thinking ~{est:,} tokens" if est is not None else None
        return f"[system] {subtype}".rstrip()
    if etype == "result":
        return None  # The caller logs the final result separately.
    if etype not in ("assistant", "user"):
        return None
    blocks = (event.get("message") or {}).get("content") or []
    parts: list[str] = []
    for block in blocks:
        btype = block.get("type")
        if btype == "text" and block.get("text", "").strip():
            parts.append(f"[{etype}] {_truncate(block['text'].strip(), 200, verbose)}")
        elif btype == "thinking" and block.get("thinking", "").strip():
            # Print the full reasoning trace, not a preview: for reasoning models
            # the thinking is the interesting signal, and truncating it hides why
            # a run stalled or how it reached a decision.
            parts.append(f"[{etype}:thinking] {block['thinking'].strip()}")
        elif btype == "tool_use":
            # In verbose mode also show the tool's input arguments, so a blocked
            # or surprising command is fully visible in the trace.
            line = f"[tool] {block.get('name', '?')}"
            if verbose and block.get("input"):
                line += f" {json.dumps(block['input'], default=str)}"
            parts.append(line)
        elif btype == "tool_result":
            text = _tool_result_text(block.get("content"))
            preview = _truncate(text, TOOL_RESULT_PREVIEW_CHARS, verbose)
            marker = "[tool_result:error]" if block.get("is_error") else "[tool_result]"
            parts.append(f"{marker} {preview}" if preview else marker)
    return "\n".join(parts) if parts else None


def _run_claude_streaming(
    cmd: list[str], env: dict[str, str], timeout: int, verbose: bool = False
) -> dict[str, Any]:
    """Run `claude -p` in streaming mode, printing a live trace.

    Reads newline-delimited JSON events as they arrive, prints a short summary
    of each, and returns the final ``result`` event (the same shape
    _metrics_from_result consumes).

    Args:
        cmd: The command argument vector (must include stream-json/--verbose).
        env: Environment for the subprocess.
        timeout: Wall-clock timeout in seconds.
        verbose: When True, print assistant text and tool results in full
            instead of truncating them in the live trace.

    Returns:
        The parsed final result event.

    Raises:
        RuntimeError: If claude times out or never emits a result event.
    """
    start = time.time()
    proc = subprocess.Popen(  # nosec B603 - hardcoded 'claude', list args, no shell
        cmd,
        env=env,
        # Run from the repo root so the /swe skill's relative artifact paths
        # resolve correctly (see the note in _run_claude); otherwise a model that
        # writes a relative path doubles it to benchmarks/benchmarks/...
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    final: dict[str, Any] | None = None
    thinking_tokens = 0
    # Latest estimate of the current extended-thinking burst, held back so we log
    # ONE summary line per burst instead of a line per streamed estimate. Flushed
    # when the next non-thinking event arrives (burst ended) and at stream end.
    pending_thinking: int | None = None
    if proc.stdout is None:  # pragma: no cover - stdout is always a pipe here
        raise RuntimeError("claude -p produced no stdout stream")

    def _flush_thinking() -> None:
        nonlocal pending_thinking
        if pending_thinking is not None:
            logger.info("  [system] thinking ~%s tokens", f"{pending_thinking:,}")
            pending_thinking = None

    try:
        for line in proc.stdout:
            if time.time() - start > timeout:
                proc.kill()
                raise RuntimeError(f"claude -p timed out after {timeout}s")
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # Non-JSON progress noise; skip.
            if event.get("type") == "result":
                _flush_thinking()
                final = event
            elif event.get("subtype") == "thinking_tokens":
                # Reasoning models stream a running token estimate every few
                # tokens. Track the peak (the result event omits it) and hold the
                # latest value; do NOT log per event -- _flush_thinking prints one
                # summary line once the burst ends.
                est = event.get("estimated_tokens")
                if isinstance(est, int):
                    thinking_tokens = max(thinking_tokens, est)
                    pending_thinking = est
            else:
                _flush_thinking()
                trace = _format_stream_event(event, verbose=verbose)
                if trace:
                    logger.info("  %s", trace)
        _flush_thinking()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise RuntimeError(f"claude -p timed out after {timeout}s") from exc

    if final is None:
        stderr = (proc.stderr.read() if proc.stderr else "").strip()
        raise RuntimeError(
            f"claude -p emitted no result event (exit {proc.returncode}): "
            f"{stderr[:500]}"
        )
    final["_elapsed_seconds"] = round(time.time() - start, 1)
    # Only present when the model streamed thinking_tokens events; buffered
    # (non-streaming) runs never see these, so the field stays absent there.
    if thinking_tokens:
        final["_thinking_tokens_estimate"] = thinking_tokens
    return final


def _artifact_dir(config: RunnerConfig, dataset: Dataset, task: Task) -> Path:
    """Return the directory where the skill writes a task's artifacts.

    Layout: ``benchmarks/<output_dir>/<model>/<harness>/<skill>/<scope>/<task>/``.
    Model, harness (coding agent), and skill are each their own path level, so
    runs never collide: a pi run never overwrites a Claude Code run, and a swe3
    run never overwrites a swe2 run of the same model. swe2 and swe3 are sibling
    folders under the harness -- they differ materially in token use and accuracy,
    so each is its own dimension rather than a suffix.

    The ``<scope>`` level is the repository name, unless the dataset sets
    ``output_scope`` -- which two datasets over the *same* repository must do, or
    they share a folder and the folder-level run-summary.json of one is rebuilt
    over the other's tasks.

    Args:
        config: The runner config.
        dataset: The loaded dataset, which decides the scope folder.
        task: The task being run.

    Returns:
        The absolute artifact directory path.
    """
    return (
        REPO_ROOT
        / "benchmarks"
        / config.output_dir
        / config.model_slug
        / config.harness_slug
        / config.skill
        / dataset.scope_for(_repo_name(task.repo))
        / task.id
    )


def _summary_metrics(
    metrics: dict[str, Any],
    vllm_prometheus: dict[str, Any],
    generation_tokens_per_sec: float,
    include_vllm: bool = True,
    agent: str = "claude",
) -> dict[str, Any]:
    """Build the headline "metrics that matter" block for a run.

    This is a curated, source-resolved summary: for each metric it picks the best
    available number and records where it came from, so a reader never has to
    know whether a value lives in the top-level API fields or the nested
    ``vllm_prometheus`` block. Cache tokens prefer the model API when it reports
    them (Amazon Bedrock, the Anthropic API) and fall back to vLLM's server-side
    counters when it does not (vLLM's Anthropic route omits per-request cache
    fields). Everything sourced from ``vllm_prometheus`` inherits its single-
    tenant caveat.

    KV-cache utilization is intentionally NOT a headline metric: on a serial,
    single-tenant benchmark it barely varies (it tracks one request's working set
    as a fraction of the pool, not anything the benchmark controls), so it has no
    power to discriminate between runs. The sampled peak/mean still lives in
    ``vllm_prometheus.gauges_sampled`` as capacity telemetry -- useful alongside
    ``num_preemptions`` to judge whether a run was memory-clean, and it becomes a
    headline concern only under concurrent load.

    Args:
        metrics: The API-reported metrics from _metrics_from_result.
        vllm_prometheus: The nested vLLM Prometheus block from _vllm_metrics.
        generation_tokens_per_sec: Output-token throughput (output_tokens /
            latency_seconds), computed once by the caller so it matches the
            top-level field.
        include_vllm: When False (e.g. provider=bedrock, which has no vLLM
            server), the vLLM-derived cache fallbacks and the prefix-cache-hit
            headline are omitted, so the summary reports only what the model API
            returned.

    Returns:
        A flat summary dict of headline numbers plus a ``sources`` map naming the
        provenance of each. Values are None when no source could supply them.
    """
    counters = vllm_prometheus.get("counters", {})
    derived = vllm_prometheus.get("derived", {})
    prompt_total = counters.get(PROMPT_TOKENS_METRIC)
    prompt_cached = counters.get(PROMPT_TOKENS_CACHED_METRIC)

    # Provenance label for the agent's own reported usage (Claude Code vs pi).
    api = f"{agent}_api"

    # Cache-read: the API number if the backend reported it, else (only when a
    # vLLM server backs the run) vLLM's cached prompt tokens. Cache-write has no
    # direct vLLM counter; the freshly computed (uncached) prefill tokens are the
    # closest equivalent.
    if "cache_read_tokens" in metrics:
        cache_read: int | None = metrics["cache_read_tokens"]
        cache_read_src = f"{api}.usage.cache_read_input_tokens"
    elif include_vllm:
        cache_read = prompt_cached
        cache_read_src = f"vllm_prometheus.counters.{PROMPT_TOKENS_CACHED_METRIC}"
    else:
        cache_read = None
        cache_read_src = f"{api}.usage.cache_read_input_tokens (not reported)"
    if "cache_creation_tokens" in metrics:
        cache_write: int | None = metrics["cache_creation_tokens"]
        cache_write_src = f"{api}.usage.cache_creation_input_tokens"
    elif include_vllm and prompt_total is not None and prompt_cached is not None:
        cache_write = prompt_total - prompt_cached
        cache_write_src = (
            f"vllm_prometheus derived: {PROMPT_TOKENS_METRIC} - "
            f"{PROMPT_TOKENS_CACHED_METRIC} (uncached prefill tokens)"
        )
    else:
        cache_write = None
        cache_write_src = "unavailable (backend reports no cache-write signal)"

    note = (
        "Headline metrics resolved to the best available source for each; see "
        "'sources'. Values drawn from vllm_prometheus carry its single-tenant, "
        "server-wide caveat."
        if include_vllm
        else (
            "Headline metrics as reported by the model API (claude -p); there is "
            "no vLLM server for this provider, so no server-side cache telemetry."
        )
    )
    # total_tokens is the total tokens processed once each, from
    # compute_total_tokens_processed (issue #136). It detects whether the cache
    # fields are a PARTITION of input_tokens (self-hosted vLLM: cache already
    # inside input, so total = input + output) or ADDITIVE (Bedrock prompt
    # caching: total = input + output + cache_read + cache_write, so a
    # heavily-cached run is not understated ~100x). Adding the cache
    # unconditionally, as this used to, ~2x double-counted self-hosted partition
    # runs. An agent whose counts are already disjoint skips the detection and
    # declares itself additive (codex; issue #183), because at a ~50% cache hit
    # rate its fresh input equals its cache sum and the detector would drop the
    # whole cache read. total_cost_usd is the agent's own metered bill (null for
    # a self-hosted model, which has no per-token price).
    inp = metrics.get("input_tokens") or 0
    out = metrics.get("output_tokens") or 0
    total_tokens = compute_total_tokens_processed(
        inp,
        out,
        cache_read or 0,
        cache_write or 0,
        context=f"run-swe-headless:_summary_metrics/{agent}",
        cache_partition=cache_partition_for_agent(agent),
    )
    token_src = (
        f"{api}.modelUsage (per-model rollup; INCLUDES subagent tokens)"
        if agent == "claude"
        else f"{api}.usage"
    )
    summary = {
        "note": note,
        "input_tokens": metrics.get("input_tokens"),
        "output_tokens": metrics.get("output_tokens"),
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "total_tokens": total_tokens,
        "total_cost_usd": metrics.get("total_cost_usd"),
        "latency_seconds": metrics.get("latency_seconds"),
        "num_turns": metrics.get("num_turns"),
        "generation_tokens_per_sec": generation_tokens_per_sec,
    }
    sources = {
        "input_tokens": f"{token_src}.input",
        "output_tokens": f"{token_src}.output",
        "cache_read_tokens": cache_read_src,
        "cache_write_tokens": cache_write_src,
        "total_tokens": (
            "total tokens processed once each (issue #136): input + output, plus "
            "cache_read + cache_write ONLY when the cache is additive (not a "
            "partition of input). An agent with disjoint counts declares that "
            "instead of being detected (issue #183)."
        ),
        "total_cost_usd": (
            f"{api}.total_cost_usd (metered)"
            if metrics.get("total_cost_usd") is not None
            else "null (self-hosted: no per-token bill; cost is hardware-derived)"
        ),
        "latency_seconds": f"harness wall-clock (or {api}.duration_ms)",
        "num_turns": f"{api}.num_turns",
        "generation_tokens_per_sec": "derived: output_tokens / latency_seconds",
    }
    # The prefix-cache hit rate is a vLLM-only signal; omit it entirely rather
    # than emit a permanently-null field for a provider that has no vLLM server.
    if include_vllm:
        summary["prefix_cache_hit_rate"] = derived.get("prefix_cache_hit_rate")
        sources["prefix_cache_hit_rate"] = (
            "vllm_prometheus.derived.prefix_cache_hit_rate"
        )
    summary["sources"] = sources
    return summary


def _save_metrics(
    config: RunnerConfig,
    dataset: Dataset,
    task: Task,
    ref: str,
    metrics: dict[str, Any],
    vllm_prometheus: dict[str, Any],
) -> Path:
    """Write the run metrics to metrics.json in the artifact directory.

    The top-level fields report what the model API returned for the run
    (tokens, latency, turns, cost) plus the harness-observed UTC wall-clock
    bounds (``run_started_at`` / ``run_ended_at``). For provider=endpoint, cache
    utilization measured out-of-band from vLLM's Prometheus /metrics is kept in
    its own ``vllm_prometheus`` block, so the two sources -- and their different
    accuracy assumptions -- never mix. For provider=bedrock there is no vLLM
    server, so that block is omitted entirely and the run is limited to what
    claude -p itself reports.

    Args:
        config: The runner config.
        task: The task that was run.
        ref: The git ref used.
        metrics: The API-reported metrics from _metrics_from_result.
        vllm_prometheus: The nested vLLM Prometheus block from _vllm_metrics.

    Returns:
        Path to the written metrics.json.
    """
    out_dir = _artifact_dir(config, dataset, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = [f for f in ARTIFACT_FILENAMES if (out_dir / f).exists()]
    latency = metrics["latency_seconds"] or 0
    generation_tokens_per_sec = (
        round(metrics["output_tokens"] / latency, 1) if latency > 0 else 0
    )
    # Persist the token-accounting verdict alongside the numbers it judges, so a
    # suspect run is self-identifying on disk and not only in the run log.
    token_accounting_warning = _check_token_accounting(
        metrics, config.agent, f"[task={task.id}]"
    )
    include_vllm = not config.is_bedrock
    record = {
        "task": task.id,
        "repo": task.repo,
        "ref": ref,
        "complexity": task.complexity,
        "tags": task.tags,
        "model": config.model,
        "model_slug": config.model_slug,
        "agent": config.agent,
        # The SWE skill that drove the run (swe2 multi-agent / swe3 single-agent).
        # Recorded so a result self-identifies its skill regardless of folder path.
        "skill": config.skill,
        "provider": config.provider,
        "endpoint": config.endpoint if not config.is_bedrock else None,
        "aws_region": config.resolved_region() if config.is_bedrock else None,
        # Serving provenance: the hardware and how the model was served. Grouped
        # in one block rather than scattered top-level fields. context_window is
        # the served window (0 in config means "unset"); null values mean unknown
        # (e.g. tensor_parallel_size / precision on the Bedrock path).
        "serving": {
            "instance_type": config.resolved_instance_type(),
            "tensor_parallel_size": config.tensor_parallel_size,
            "precision": config.precision,
            "context_window": config.context_window or None,
        },
        "artifacts_produced": len(produced),
        "artifacts_expected": len(ARTIFACT_FILENAMES),
        "generation_tokens_per_sec": generation_tokens_per_sec,
        # Null on a healthy run. A string here means output-tokens-per-turn fell
        # below MIN_PLAUSIBLE_OUTPUT_TOKENS_PER_TURN, i.e. the token/cost figures in
        # this file are suspect (scores/turns/latency are not). Never publish a
        # token or cost column from a run where this is set.
        "token_accounting_warning": token_accounting_warning,
        # Endpoint runs only: the agent's prompt tokens against vLLM's own
        # counter, so retried requests the agent never counted are visible on
        # disk instead of silently deflating token-derived cost (issue #183).
        "prompt_token_reconciliation": (
            _server_prompt_token_gap(metrics, vllm_prometheus, config.concurrency)
            if include_vllm
            else None
        ),
        # The normalized, cross-agent metric block -- the single source of truth
        # for tokens/cost/turns/latency + provenance. summarize_run and the charts
        # read this. Filled to whatever extent the agent reports (nulls where a
        # signal is unavailable, never a misleading 0).
        "metrics": _summary_metrics(
            metrics,
            vllm_prometheus,
            generation_tokens_per_sec,
            include_vllm,
            agent=config.agent,
        ),
        **metrics,
    }
    # Back-compat alias: older tooling / committed readers referenced
    # "metrics_that_matter". Keep it pointing at the same normalized block for one
    # release so nothing breaks; new code should read "metrics".
    record["metrics_that_matter"] = record["metrics"]
    # vLLM Prometheus telemetry only exists for an HTTP endpoint; omit the block
    # for Amazon Bedrock rather than write a permanently-unavailable stub.
    if include_vllm:
        record["vllm_prometheus"] = vllm_prometheus
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _run_task(
    config: RunnerConfig,
    dataset: Dataset,
    task: Task,
    stream: bool = False,
    concurrent: bool = False,
    position: int = 1,
    total: int = 1,
    verbose: bool = False,
    topup_missing: list[str] | None = None,
) -> dict[str, Any]:
    """Run a single task end to end and return its outcome summary.

    Args:
        config: The runner config.
        dataset: The loaded dataset (for default-ref resolution).
        task: The task to run.
        stream: If True, print a live event trace while claude -p runs.
        concurrent: True when other tasks may run on the endpoint at the same
            time (concurrency > 1). The single-tenant assumption behind the
            window-delta metrics no longer holds, so the vLLM block is annotated
            as a server-wide aggregate (ratios blended, absolute counts summed
            across the overlapping runs) rather than passed off as per-run.
        position: This task's 1-based position in the run (for legible logs).
        total: Total number of tasks in the run.
        verbose: When True (and streaming), print assistant text and tool
            results in full instead of truncating them in the live trace.

    Returns:
        A summary dict: task id, ok flag, artifacts produced, and metrics.
    """
    ref = dataset.resolved_ref(task)
    label = f"[task={task.id}] {position} of {total}"
    logger.info("=== %s [%s] ref=%s ===", label, task.complexity, ref)

    clone_path = _clone_repo(task, ref, config.clone_dir, log_prefix=label)
    clone_parent = clone_path.parent
    try:
        prompt = _build_prompt(
            task,
            clone_path,
            ref,
            config.model_slug,
            _artifact_dir(config, dataset, task),
            agent=config.agent,
            skill=config.skill,
            topup_missing=topup_missing,
        )
        run_kind = f"top-up ({', '.join(topup_missing)})" if topup_missing else "run"
        # Build the agent-specific command + environment. pi and Claude Code run
        # the same /swe2 task but take entirely different flags and routing, so
        # this is the single branch point; everything downstream (metrics,
        # scraping, artifact checks) is agent-agnostic.
        if config.is_pi:
            # Per-run pi config dir under the clone parent so it is cleaned up
            # with the clone and never touches the developer's global ~/.pi.
            pi_agent_dir = clone_parent / "pi-agent"
            # vLLM routing needs a models.json pointing at the endpoint; the
            # native Amazon Bedrock provider is built into pi, so skip it there.
            if config.is_bedrock:
                pi_agent_dir.mkdir(parents=True, exist_ok=True)
                _write_pi_settings(config, pi_agent_dir)
            else:
                _write_pi_models_json(config, pi_agent_dir)
            cmd = _build_pi_cmd(config, prompt)
            env = _build_pi_env(config, pi_agent_dir)
            logger.info(
                "  %s Running pi -p %s (agent=pi, no turn cap)...", label, run_kind
            )
        elif config.is_omp:
            # Per-run omp config dir under the clone parent, same isolation as pi.
            # The endpoint path needs models.yml + config.yml; omp's native Bedrock
            # provider needs neither, but the compaction setting still applies.
            omp_agent_dir = clone_parent / "omp-agent"
            if config.is_bedrock:
                omp_agent_dir.mkdir(parents=True, exist_ok=True)
            else:
                _write_omp_config(config, omp_agent_dir)
            cmd = _build_omp_cmd(config, prompt)
            env = _build_omp_env(config, omp_agent_dir)
            omp_cap = (
                f"max-time {config.agent_max_time_seconds}s"
                if config.agent_max_time_seconds
                else "no time cap"
            )
            logger.info(
                "  %s Running omp -p %s (agent=omp, no turn cap, %s)...",
                label,
                run_kind,
                omp_cap,
            )
        elif config.is_kiro:
            # kiro-cli uses its own global sign-in (~/.kiro); there is no per-run
            # config dir to write and no endpoint to route. The SKILL.md is inlined
            # into the prompt by _build_kiro_cmd (kiro has no --skill flag).
            cmd = _build_kiro_cmd(config, prompt)
            env = _build_kiro_env(config)
            logger.info(
                "  %s Running kiro-cli chat %s (agent=kiro, no turn cap)...",
                label,
                run_kind,
            )
        elif config.is_codex:
            # codex exec runs non-interactively with --json output and full token
            # accounting. The SKILL.md is inlined into the prompt (codex has no
            # --skill flag, same as kiro).
            cmd = _build_codex_cmd(config, clone_path, prompt)
            env = _build_codex_env(config)
            logger.info(
                "  %s Running codex exec %s (agent=codex, no turn cap)...",
                label,
                run_kind,
            )
        else:
            cmd = _build_claude_cmd(
                config, prompt, stream=stream, clone_path=clone_path
            )
            env = _build_env(config)
            logger.info(
                "  %s Running claude -p %s (max_turns=%s)...",
                label,
                run_kind,
                config.max_turns,
            )
        # vLLM Prometheus scraping only applies to an HTTP endpoint; Amazon
        # Bedrock exposes no such surface, so skip it and leave the block marked
        # unavailable. metrics_endpoint is None for provider=bedrock.
        metrics_endpoint = config.endpoint if not config.is_bedrock else None
        # Snapshot vLLM's full server-wide metrics surface as tightly around the
        # claude -p call as possible. Each delta is this run's activity ONLY if
        # the run is the sole traffic on the endpoint (see _snapshot_vllm_metrics).
        # Keep the reads adjacent to the call to minimize the window in which
        # other traffic could be misattributed.
        vllm_before = (
            _snapshot_vllm_metrics(metrics_endpoint) if metrics_endpoint else None
        )
        # Gauges (KV-cache usage, running/waiting requests) drain to idle the
        # moment a request finishes, so a before/after snapshot always reads them
        # at ~0. Sample them in a background thread WHILE claude -p runs to capture
        # the in-flight peak/mean instead.
        # Wall-clock UTC bounds of the run, captured as tightly around the
        # claude -p call as the metric snapshots. ISO 8601 with a trailing Z.
        run_started_at = _utc_now_iso()
        with _GaugePoller(metrics_endpoint) as poller:
            if config.is_pi:
                # pi emits a JSON-lines event stream; _run_pi normalizes it to the
                # same result shape. It has no separate streaming trace mode.
                result = _run_pi(
                    cmd,
                    env,
                    config.timeout_seconds,
                    stream_log=_artifact_dir(config, dataset, task) / "pi-stream.jsonl",
                )
            elif config.is_omp:
                result = _run_omp(
                    cmd,
                    env,
                    config.timeout_seconds,
                    stream_log=_artifact_dir(config, dataset, task)
                    / "omp-stream.jsonl",
                )
            elif config.is_kiro:
                # kiro-cli streams ANSI text and prints a Credits/Time summary on
                # stderr; _run_kiro normalizes that to the same result shape.
                result = _run_kiro(
                    cmd, env, config.timeout_seconds, config.kiro_dollars_per_credit
                )
            elif config.is_codex:
                # codex exec outputs JSON-lines events; _run_codex normalizes them.
                result = _run_codex(
                    cmd, env, config.timeout_seconds, model=config.model or ""
                )
            elif stream:
                result = _run_claude_streaming(
                    cmd, env, config.timeout_seconds, verbose=verbose
                )
            else:
                result = _run_claude(cmd, env, config.timeout_seconds)
        run_ended_at = _utc_now_iso()
        vllm_after = (
            _snapshot_vllm_metrics(metrics_endpoint) if metrics_endpoint else None
        )
        metrics = _metrics_from_result(result, result.get("_elapsed_seconds", 0))
        metrics["run_started_at"] = run_started_at
        metrics["run_ended_at"] = run_ended_at
        vllm_block = _vllm_metrics(vllm_before, vllm_after)
        vllm_block["gauges_sampled"] = poller.summary()
        if concurrent:
            _mark_aggregate(vllm_block)
    finally:
        shutil.rmtree(clone_parent, ignore_errors=True)

    metrics_path = _save_metrics(config, dataset, task, ref, metrics, vllm_block)
    out_dir = metrics_path.parent
    produced = [f for f in ARTIFACT_FILENAMES if (out_dir / f).exists()]
    # Completeness is gated on the four DESIGN artifacts plus the implementation
    # patch: a /swe2 task is "ok" only when it both designed and implemented the
    # change (patch.diff present) and claude did not report an error. The banner
    # still shows the full produced count over all six artifacts.
    design_done = all((out_dir / f).exists() for f in DESIGN_ARTIFACT_FILENAMES)
    patch_done = (out_dir / "patch.diff").exists()
    ok = design_done and patch_done and not metrics["is_error"]

    # One-line outcome banner: artifacts, turns, tokens, latency, and (when
    # available) the vLLM prefix-cache hit rate for the run's window.
    cache_suffix = ""
    hit_rate = vllm_block.get("derived", {}).get("prefix_cache_hit_rate")
    if hit_rate is not None:
        queries = vllm_block["counters"].get(PREFIX_CACHE_QUERIES_METRIC)
        hits = vllm_block["counters"].get(PREFIX_CACHE_HITS_METRIC)
        scope = "aggregate" if concurrent else "single-tenant"
        cache_suffix = (
            f", prefix cache {hit_rate * 100:.1f}% hit "
            f"({hits:,}/{queries:,} tokens, {scope})"
            if hits is not None and queries is not None
            else f", prefix cache {hit_rate * 100:.1f}% hit ({scope})"
        )
    thinking_suffix = ""
    if metrics.get("thinking_tokens_estimate"):
        thinking_suffix = f" (~{metrics['thinking_tokens_estimate']:,} thinking)"
    summary = (
        f"{label} | {'OK' if ok else 'INCOMPLETE'}: "
        f"{len(produced)}/{len(ARTIFACT_FILENAMES)} artifacts, "
        f"{metrics['num_turns']} turns, "
        f"{metrics['input_tokens']:,} in / {metrics['output_tokens']:,} out{thinking_suffix} tokens, "
        f"{metrics['latency_seconds']}s{cache_suffix}"
    )
    banner = "=" * len(summary)
    logger.info(banner)
    logger.info(summary)
    logger.info(banner)
    if metrics["is_error"]:
        logger.error(
            "  %s -p reported an error (status %s): %s",
            config.agent,
            metrics.get("api_error_status"),
            metrics.get("error"),
        )
    logger.info("  Metrics: %s", metrics_path)
    return {
        "task": task.id,
        "ok": ok,
        "artifacts": len(produced),
        "design_done": design_done,
        "patch_done": patch_done,
        "metrics": metrics,
    }


def _select_tasks(dataset: Dataset, task_ids: list[str], count: int = 0) -> list[Task]:
    """Select tasks to run, preserving dataset order.

    Args:
        dataset: The loaded dataset.
        task_ids: Task ids to run; empty means all tasks.
        count: Keep only the first ``count`` selected tasks; 0 means no limit.

    Returns:
        The tasks to run.

    Raises:
        DatasetError: If a requested id is not in the dataset or count is negative.
    """
    if count < 0:
        raise DatasetError(
            f"--count must be 0 (all) or a positive integer, got {count}"
        )
    if not task_ids:
        selected = dataset.tasks
    else:
        known = {t.id for t in dataset.tasks}
        missing = [tid for tid in task_ids if tid not in known]
        if missing:
            raise DatasetError(
                f"Unknown task ids: {missing}. Available: {sorted(known)}"
            )
        selected = [t for t in dataset.tasks if t.id in set(task_ids)]
    return selected[:count] if count else selected


def _dry_run(config: RunnerConfig, dataset: Dataset, tasks: list[Task]) -> None:
    """Print the prompt and command for each task without executing anything."""
    for task in tasks:
        ref = dataset.resolved_ref(task)
        placeholder = (
            Path(config.clone_dir)
            / f"swe-clone-{_safe_task_slug(task.id)}"
            / _repo_name(task.repo)
        )
        prompt = _build_prompt(
            task,
            placeholder,
            ref,
            config.model_slug,
            _artifact_dir(config, dataset, task),
            agent=config.agent,
            skill=config.skill,
        )
        if config.is_pi:
            cmd = _build_pi_cmd(config, prompt)
        elif config.is_omp:
            cmd = _build_omp_cmd(config, prompt)
        elif config.is_kiro:
            cmd = _build_kiro_cmd(config, prompt)
        elif config.is_codex:
            cmd = _build_codex_cmd(config, placeholder, prompt)
        else:
            cmd = _build_claude_cmd(config, prompt, clone_path=placeholder)
        print(f"\n=== {task.id} [{task.complexity}] ref={ref} ===")
        print("PROMPT:")
        print(prompt)
        print("\nCOMMAND:")
        print(" ".join(cmd))


def _summary_is_retryable(summary: dict[str, Any]) -> bool:
    """Decide whether a failed task summary warrants a retry.

    A task is retried only when it failed for a TRANSIENT reason. It is NOT
    retried when it simply exhausted its turn budget: another attempt at the
    same ``max_turns`` will hit the same wall, so the fix is a larger budget,
    not a retry.

    Turn exhaustion is identified by claude -p's result subtype
    ``error_max_turns``, or, defensively, by a run that used up (near) all of its
    turns without producing the design artifacts. Everything else that left the
    task not-ok (a stream/JSON/timeout RuntimeError, an api/execution error, an
    empty result) is treated as transient and retryable.

    Args:
        summary: A task outcome dict from :func:`_run_task` (or the RuntimeError
            fallback below), possibly carrying a ``metrics`` block.

    Returns:
        True if the task should be retried, False otherwise.
    """
    if summary.get("ok"):
        return False
    metrics = summary.get("metrics") or {}
    if metrics.get("result_subtype") == "error_max_turns":
        return False
    # Defensive fallback: no subtype recorded but the run clearly ran the turn
    # budget dry (e.g. an older claude that omits the subtype). Treat a run that
    # burned >=95% of max_turns without finishing the design as turn exhaustion.
    max_turns = summary.get("max_turns")
    num_turns = metrics.get("num_turns")
    if (
        max_turns
        and num_turns is not None
        and num_turns >= 0.95 * max_turns
        and not summary.get("design_done", False)
    ):
        return False
    return True


def _clear_partial_artifacts(
    config: RunnerConfig, dataset: Dataset, task: Task
) -> None:
    """Remove a task's partially-written artifacts before a retry.

    A transiently-failed attempt may have left some artifacts (and a
    metrics.json) behind. Clearing them keeps the retry a clean run and prevents
    a stale partial file from masking what the retry actually produced.
    """
    out_dir = _artifact_dir(config, dataset, task)
    for filename in (*ARTIFACT_FILENAMES, "metrics.json"):
        (out_dir / filename).unlink(missing_ok=True)


def _missing_artifacts(config: RunnerConfig, dataset: Dataset, task: Task) -> list[str]:
    """Return the ARTIFACT_FILENAMES not yet present in the task's output dir."""
    out_dir = _artifact_dir(config, dataset, task)
    return [f for f in ARTIFACT_FILENAMES if not (out_dir / f).exists()]


def _pass_cost_value(record: dict[str, Any], key: str) -> float:
    """Read one additive cost field from a single pass's metrics record.

    Args:
        record: A parsed metrics.json (or an in-memory summary) for one pass.
        key: A member of :data:`ADDITIVE_COST_FIELDS`.

    Returns:
        The field's value, or 0 when the pass did not report it. The normalized
        block is preferred over the top-level mirror because it is what
        ``summarize_run`` reads.
    """
    block = record.get("metrics") or record.get("metrics_that_matter") or {}
    val = block.get(MM_BLOCK_KEY.get(key, key))
    if val is None:
        val = record.get(key)
    return val or 0


def _fold_pass_into_totals(
    totals: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    """Add one pass's additive cost fields into a running total.

    Args:
        totals: The running totals, keyed by :data:`ADDITIVE_COST_FIELDS`.
        record: The pass to fold in.

    Returns:
        The same ``totals`` dict, mutated.
    """
    for key in ADDITIVE_COST_FIELDS:
        totals[key] = (totals.get(key) or 0) + _pass_cost_value(record, key)
    return totals


def _write_cost_totals(
    path: Path,
    totals: dict[str, Any],
    invocations: int,
    context: str,
    topped_up: list[str] | None = None,
) -> None:
    """Restore summed multi-invocation cost into a task's metrics.json.

    The final pass left metrics.json holding only its own numbers. This writes
    the summed additive fields to the top-level fields AND to the normalized
    block ("metrics", plus its "metrics_that_matter" alias), because
    ``summarize_run`` reads the normalized block -- writing only one place would
    leave the summary showing a single pass. Best-effort: a write failure is
    logged, not fatal.

    Args:
        path: The task's metrics.json.
        totals: Summed additive fields across every agent invocation.
        invocations: How many agent invocations the task actually took.
        context: Label for the token-accounting trace.
        topped_up: Artifacts produced by a top-up pass, when any.
    """
    record = _read_json_file(path)
    if record is None:
        return
    record.update(totals)
    for block_name in ("metrics", "metrics_that_matter"):
        block = record.get(block_name)
        if not isinstance(block, dict):
            continue
        for key, value in totals.items():
            target = MM_BLOCK_KEY.get(key, key)
            if target in block:
                block[target] = value
        # total_tokens is derived, not additive; recompute it from the summed
        # parts so the partition-vs-additive rule (issue #136) is applied
        # consistently and does not ~2x double-count self-hosted partition runs.
        # The agent recorded in the file decides whether the shape is declared
        # (disjoint counts) or detected (issue #183).
        if "total_tokens" in block:
            block["total_tokens"] = compute_total_tokens_processed(
                totals.get("input_tokens") or 0,
                totals.get("output_tokens") or 0,
                totals.get("cache_read_tokens") or 0,
                totals.get("cache_creation_tokens") or 0,
                context=f"{context}/{block_name}",
                cache_partition=cache_partition_for_agent(record.get("agent")),
            )
    record["agent_invocations"] = invocations
    if topped_up is not None:
        record["topped_up_artifacts"] = topped_up
    try:
        path.write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
    except OSError:
        logger.warning("could not write summed cost totals to %s", path)


def _maybe_topup(
    config: RunnerConfig,
    dataset: Dataset,
    task: Task,
    summary: dict[str, Any],
    *,
    stream: bool,
    concurrent: bool,
    position: int,
    total: int,
    verbose: bool,
) -> dict[str, Any]:
    """Complete a design-complete task that is missing implementation artifacts.

    The outer completion loop: after the main run (and any transient retries), if
    the task is still not ``ok`` but the four DESIGN artifacts are all present, it
    is the common "ran out of context right before patch.diff" case. Up to
    ``config.max_topups`` times, re-invoke the agent in a FRESH context with a
    focused prompt that produces ONLY the missing files, reading (never rewriting)
    the ones already on disk. Existing artifacts are NOT cleared, so a top-up can
    only add. Each top-up is a separate agent invocation, recorded on the returned
    summary and in metrics.json (``agent_invocations``, ``topped_up_artifacts``),
    so a completed-but-assisted run stays distinguishable from a clean one.

    Top-up is intentionally NOT attempted when the design is incomplete: a run
    that could not finish the design docs is a genuine quality failure, not a
    truncation to be patched over.

    Args:
        summary: The outcome from the main run/retries (mutated with top-up
            provenance and replaced by the latest attempt's summary).

    Returns:
        The final summary (ok if a top-up completed the artifact set).
    """
    invocations = summary.get("attempts", 1)
    topped_up: list[str] = []
    # Seed the running cost from what is already on disk. That record ALREADY
    # holds the sum across any transient retries (_run_task_with_retries folded
    # them in before we were called), so top-ups accumulate on top of it rather
    # than restarting the count. See ADDITIVE_COST_FIELDS for why these fields.
    base = _read_json_file(_artifact_dir(config, dataset, task) / "metrics.json") or {}
    totals = {k: _pass_cost_value(base, k) for k in ADDITIVE_COST_FIELDS}
    for topup in range(1, config.max_topups + 1):
        if summary.get("ok"):
            break
        # Only design-complete tasks are eligible; a missing design doc is a real
        # failure, not a truncation to top up.
        design_done = all(
            (_artifact_dir(config, dataset, task) / f).exists()
            for f in DESIGN_ARTIFACT_FILENAMES
        )
        missing = _missing_artifacts(config, dataset, task)
        if not design_done or not missing:
            break
        logger.warning(
            "[task=%s] %s of %s: design complete but missing %s; top-up %s of %s",
            task.id,
            position,
            total,
            ", ".join(missing),
            topup,
            config.max_topups,
        )
        try:
            summary = _run_task(
                config,
                dataset,
                task,
                stream=stream,
                concurrent=concurrent,
                position=position,
                total=total,
                verbose=verbose,
                topup_missing=missing,
            )
        except RuntimeError:
            logger.exception(
                "[task=%s] %s of %s top-up failed", task.id, position, total
            )
            break
        invocations += 1
        # This top-up overwrote metrics.json with only its own pass; fold its
        # additive cost into the running totals.
        pass_metrics = (
            _read_json_file(_artifact_dir(config, dataset, task) / "metrics.json") or {}
        )
        _fold_pass_into_totals(totals, pass_metrics)
        topped_up = [
            f for f in missing if (_artifact_dir(config, dataset, task) / f).exists()
        ]

    # Record top-up provenance so the run is honestly flagged as assisted, both on
    # the in-memory summary and in the on-disk metrics.json.
    summary["max_turns"] = config.max_turns
    summary["agent_invocations"] = invocations
    summary["topped_up_artifacts"] = topped_up
    if invocations > 1:
        _annotate_metrics_topup(config, dataset, task, invocations, topped_up, totals)
    return summary


def _annotate_metrics_topup(
    config: RunnerConfig,
    dataset: Dataset,
    task: Task,
    invocations: int,
    topped_up: list[str],
    totals: dict[str, Any],
) -> None:
    """Record top-up provenance and summed cost into the task's metrics.json.

    Thin wrapper over :func:`_write_cost_totals` that also records which
    artifacts a top-up produced, so a completed-but-assisted run stays
    distinguishable from a clean one.

    Args:
        config: The runner config.
        dataset: The loaded dataset.
        task: The task whose metrics.json to annotate.
        invocations: Total agent invocations across retries and top-ups.
        topped_up: Artifacts produced by a top-up pass.
        totals: Summed additive cost fields across every invocation.
    """
    _write_cost_totals(
        _artifact_dir(config, dataset, task) / "metrics.json",
        totals,
        invocations,
        context="run-swe-headless:_annotate_metrics_topup",
        topped_up=topped_up,
    )


def _read_json_file(path: Path) -> dict[str, Any] | None:
    """Return parsed JSON at ``path``, or None if absent/unreadable/invalid."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _run_task_safe(
    config: RunnerConfig,
    dataset: Dataset,
    task: Task,
    stream: bool,
    concurrent: bool,
    position: int,
    total: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one task with transient-failure retries, returning its outcome.

    Wraps :func:`_run_task` so a single task's failure never aborts the whole
    run. Retries up to ``config.max_retries`` times, but ONLY for transient
    failures (see :func:`_summary_is_retryable`): a task that exhausted its turn
    budget is returned as-is without retrying. A RuntimeError from ``_run_task``
    (timeout, empty/non-JSON output, clone failure) is a transient failure and
    counts as an attempt.

    Used as the unit of work for both the serial loop and the thread pool.
    """
    attempts = config.max_retries + 1
    last: dict[str, Any] = {"task": task.id, "ok": False, "artifacts": 0}
    # Cost carried over from attempts that were discarded. A failed attempt still
    # burned real tokens, turns and wall-clock on the GPU, so dropping it would
    # understate the task's true cost by roughly the number of attempts it took.
    # _clear_partial_artifacts deletes metrics.json, so each attempt is folded in
    # BEFORE the wipe; the sum is restored onto the final record after the loop.
    carried: dict[str, Any] = {}
    metrics_path = _artifact_dir(config, dataset, task) / "metrics.json"
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            logger.warning(
                "[task=%s] %s of %s: transient failure, retry %s of %s",
                task.id,
                position,
                total,
                attempt - 1,
                config.max_retries,
            )
            prior = _read_json_file(metrics_path)
            if prior is not None:
                _fold_pass_into_totals(carried, prior)
            _clear_partial_artifacts(config, dataset, task)
        try:
            summary = _run_task(
                config,
                dataset,
                task,
                stream=stream,
                concurrent=concurrent,
                position=position,
                total=total,
                verbose=verbose,
            )
        except RuntimeError:
            logger.exception("[task=%s] %s of %s failed", task.id, position, total)
            # A thrown RuntimeError is transient (timeout / no output / clone
            # error); record it as a retryable attempt.
            summary = {
                "task": task.id,
                "ok": False,
                "artifacts": 0,
                "metrics": {"is_error": True, "error": "run raised RuntimeError"},
            }
        summary["max_turns"] = config.max_turns
        summary["attempts"] = attempt
        last = summary
        if summary.get("ok") or not _summary_is_retryable(summary):
            break
    else:
        logger.error(
            "[task=%s] %s of %s: still failing after %s attempt(s)",
            task.id,
            position,
            total,
            attempts,
        )
    # Fold the discarded attempts back in so metrics.json reports what the task
    # actually cost, not just its final pass. Done before any top-up, which seeds
    # its own running total from this record.
    if carried:
        totals = _fold_pass_into_totals(
            dict(carried), _read_json_file(metrics_path) or {}
        )
        _write_cost_totals(
            metrics_path,
            totals,
            last.get("attempts", attempts),
            context="run-swe-headless:_run_task_safe",
        )

    # Outer completion loop: if the task is design-complete but missing the
    # implementation artifacts, try focused top-ups to finish it (see _maybe_topup).
    if not last.get("ok") and config.max_topups > 0:
        last = _maybe_topup(
            config,
            dataset,
            task,
            last,
            stream=stream,
            concurrent=concurrent,
            position=position,
            total=total,
            verbose=verbose,
        )
    return last


def _run(
    config: RunnerConfig,
    dataset: Dataset,
    tasks: list[Task],
    stream: bool = False,
    verbose: bool = False,
) -> None:
    """Run every selected task and log a final pass/fail summary.

    Tasks run serially when ``config.concurrency`` is 1 (the default) and in a
    thread pool of that width otherwise. Concurrency > 1 overlaps runs on the
    endpoint, which invalidates the single-tenant vLLM window-delta metrics; the
    per-run blocks are flagged unreliable and a warning is logged here.
    """
    concurrency = max(1, min(config.concurrency, len(tasks)))
    target = (
        f"Amazon Bedrock ({config.resolved_region()})"
        if config.is_bedrock
        else config.endpoint
    )
    logger.info(
        "Running %s task(s) with model=%s against %s (concurrency=%s)",
        len(tasks),
        config.model,
        target,
        concurrency,
    )
    if concurrency > 1:
        logger.warning(
            "Concurrency is %s: per-run API metrics (tokens, latency, turns) stay "
            "correct, but the vllm_prometheus block becomes a server-wide "
            "AGGREGATE over the shared window (ratios blended across runs, "
            "absolute counts summed). Use concurrency 1 for per-run vLLM metrics.",
            concurrency,
        )

    total = len(tasks)
    if concurrency == 1:
        summaries = [
            _run_task_safe(
                config,
                dataset,
                task,
                stream,
                False,
                position=i,
                total=total,
                verbose=verbose,
            )
            for i, task in enumerate(tasks, start=1)
        ]
    else:
        summaries = _run_concurrent(config, dataset, tasks, stream, concurrency)

    passed = sum(1 for s in summaries if s["ok"])
    logger.info("=" * 60)
    logger.info("Done: %s/%s tasks produced all artifacts.", passed, len(summaries))
    for s in summaries:
        logger.info(
            "  %s %s (%s artifacts)",
            "OK " if s["ok"] else "FAIL",
            s["task"],
            s["artifacts"],
        )


def _run_concurrent(
    config: RunnerConfig,
    dataset: Dataset,
    tasks: list[Task],
    stream: bool,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Run tasks in a thread pool of the given width, preserving task order.

    Each task clones into its own temp dir and writes to a distinct artifact
    dir, and claude -p runs as an independent subprocess, so the work is safe to
    parallelize. Streaming is disabled here because interleaved event traces from
    concurrent tasks are unreadable.

    Returns:
        Summary dicts in the same order as ``tasks``.
    """
    if stream:
        logger.warning("Disabling --stream under concurrency; traces would interleave.")
    total = len(tasks)
    results: list[dict[str, Any]] = [{} for _ in tasks]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {
            executor.submit(
                _run_task_safe, config, dataset, task, False, True, index + 1, total
            ): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    return results


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    CLI flags override the corresponding runner-config fields.
    """
    parser = argparse.ArgumentParser(
        description="Run the SWE benchmark headless via claude -p and the /swe skill.",
        epilog=(
            "Examples:\n"
            "  uv run scripts/run-swe-headless.py --config config/runner.example.yaml\n"
            "  uv run scripts/run-swe-headless.py --config config/runner.example.yaml "
            "--model qwen3-coder-30b --tasks remove-faiss,remove-efs-from-terraform-aws-ecs\n"
            "  uv run scripts/run-swe-headless.py --config config/runner.example.yaml --dry-run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="Path to the runner config YAML file")
    parser.add_argument(
        "--agent",
        help="Override: coding agent that runs the task ('claude' for Claude "
        "Code, 'pi' for the pi coding agent, 'omp' for oh-my-pi, 'kiro' for "
        "kiro-cli, 'codex' for OpenAI Codex). All support provider=bedrock; "
        "all except kiro also support provider=endpoint.",
    )
    parser.add_argument(
        "--skill",
        help="Override: SWE skill to run ('swe2' multi-agent fan-out, the "
        "default, or 'swe3' single-agent, no subagents). Same six artifacts; "
        "swe3 results land under a '<harness>-swe3' folder so they never "
        "overwrite swe2 results.",
    )
    parser.add_argument(
        "--provider",
        help="Override: routing provider ('endpoint' for a base URL, 'bedrock' "
        "for native Amazon Bedrock)",
    )
    parser.add_argument("--endpoint", help="Override: API endpoint base URL")
    parser.add_argument("--model", help="Override: model name")
    parser.add_argument(
        "--aws-region",
        help="Override: AWS region for provider=bedrock (e.g. us-east-1)",
    )
    parser.add_argument(
        "--instance-type",
        help="Override: EC2 instance type served on (e.g. p5en.48xlarge). "
        "Defaults to the EC2 metadata service when unset.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Override: tensor-parallel size (vLLM TP) the model is served with. "
        "Recorded in the metrics.json serving block.",
    )
    parser.add_argument(
        "--precision",
        help="Override: served weight precision (e.g. BF16, FP8). Recorded in the "
        "metrics.json serving block.",
    )
    parser.add_argument("--dataset", help="Override: dataset YAML path")
    parser.add_argument(
        "--tasks", help="Override: comma-separated task ids to run (default: all)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Run only the first N selected tasks (default: 0 = all)",
    )
    parser.add_argument("--max-turns", type=int, help="Override: cap on the agent loop")
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Override: retries for a task that fails TRANSIENTLY (stream/JSON/"
        "timeout/api error). A task that ran out of turns is never retried. "
        "0 disables retries.",
    )
    parser.add_argument(
        "--max-topups",
        type=int,
        help="Override: focused top-up attempts when the main run left artifacts "
        "missing but the design docs are complete. A top-up re-invokes the agent "
        "in a fresh context to produce ONLY the missing files (existing ones are "
        "kept), flagged in metrics.json. 0 disables.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Override: per-response output-token cap (CLAUDE_CODE_MAX_OUTPUT_TOKENS). "
        "Lower it on a small-window model so the prompt has usable input room "
        "(usable input ~= context_window - max_output_tokens).",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        help="Override: the model's true context window in tokens. Calibrates "
        "auto-compaction (CLAUDE_CODE_AUTO_COMPACT_WINDOW) for custom models "
        "whose window Claude Code cannot detect. 0 leaves it unset.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="Override: wall-clock timeout for a single task's claude -p run. "
        "Raise it for a slow (e.g. dense) model that produces artifacts but "
        "does not return within the default before the harness kills it.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Override: how many tasks to run at once (default 1 = serial). "
        "Values above 1 invalidate the single-tenant vLLM metrics.",
    )
    parser.add_argument(
        "--kiro-dollars-per-credit",
        type=float,
        help="Override (agent=kiro only): USD per kiro-cli credit, used to turn "
        "the credits kiro-cli reports into a dollar cost per task. Default 0.04 "
        "(Kiro add-on/overage rate); use 0.02 for the blended included rate.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print prompts/commands without running"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print a live event trace as each task runs (uses stream-json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="With --stream, print assistant text and tool results in full "
        "instead of truncating them in the live trace",
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments, load config and dataset, and run the benchmark."""
    args = _parse_args()
    overrides: dict[str, Any] = {
        "agent": args.agent,
        "skill": args.skill,
        "provider": args.provider,
        "endpoint": args.endpoint,
        "model": args.model,
        "aws_region": args.aws_region,
        "instance_type": args.instance_type,
        "tensor_parallel_size": args.tensor_parallel_size,
        "precision": args.precision,
        "dataset": args.dataset,
        "max_turns": args.max_turns,
        "max_retries": args.max_retries,
        "max_topups": args.max_topups,
        "max_output_tokens": args.max_output_tokens,
        "context_window": args.context_window,
        "timeout_seconds": args.timeout_seconds,
        "concurrency": args.concurrency,
        "kiro_dollars_per_credit": args.kiro_dollars_per_credit,
    }
    if args.tasks:
        overrides["tasks"] = [t.strip() for t in args.tasks.split(",") if t.strip()]

    try:
        config = load_runner_config(args.config, overrides)
    except RunnerConfigError as exc:
        logger.error("Config error: %s", exc)
        sys.exit(1)

    dataset_path = config.dataset
    if not Path(dataset_path).is_absolute():
        dataset_path = str(Path(__file__).resolve().parent.parent / dataset_path)
    try:
        dataset = load_dataset(dataset_path)
        tasks = _select_tasks(dataset, config.tasks, args.count)
    except DatasetError as exc:
        logger.error("Dataset error: %s", exc)
        sys.exit(1)

    if args.dry_run:
        _dry_run(config, dataset, tasks)
        return
    if args.verbose and not args.stream:
        logger.warning("--verbose has no effect without --stream; ignoring it.")
    _run(config, dataset, tasks, stream=args.stream, verbose=args.verbose)


if __name__ == "__main__":
    main()
