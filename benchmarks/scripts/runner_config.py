#!/usr/bin/env python3
"""Load and validate the SWE benchmark runner configuration.

The runner config is a small YAML file that supplies the run-time parameters
for the headless harness: which endpoint and model to drive, which dataset to
run, where to put outputs, and how to invoke `claude -p` (permission mode,
allowed tools, turn cap). Every field can be overridden on the command line so
a committed config stays the reusable default while one-off runs stay flexible.

Run it from the ``benchmarks/`` directory with its own venv:

    uv run scripts/runner_config.py config/runner.example.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Tools the /swe skill needs to read a repo and write the four artifacts. The
# skill only reads code and writes markdown, so this stays deliberately narrow.
DEFAULT_ALLOWED_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Write",
    "Edit",
    "Bash(git clone*)",
    "Bash(git -C*)",
    "Bash(mktemp*)",
    "Task",
]
# acceptEdits lets the skill write artifacts without a prompt while still
# refusing anything not covered by the allowlist. We never default to
# bypassPermissions.
DEFAULT_PERMISSION_MODE = "acceptEdits"
# bypassPermissions is allowed because the benchmark runs against THROWAWAY
# clones (fresh git clone into a temp dir, deleted after each task, no secrets).
# It is required for /swe2 (implementation): Claude Code's built-in Bash guard
# blocks the `cd <repo> && git ...` idiom that non-Claude models emit ("changes
# directory before running git, can execute untrusted hooks"), which otherwise
# burns the whole turn budget on denied commands and prevents any patch.diff.
VALID_PERMISSION_MODES = {"default", "acceptEdits", "plan", "bypassPermissions"}
DEFAULT_MAX_TURNS = 250
DEFAULT_MAX_OUTPUT_TOKENS = 16000
# 2h per task: /swe2 implements (up to 250 turns), and a frontier model writing
# long, thorough implementations can exceed 1h on a heavy task (observed with
# Opus 5). A task that overruns is killed and marked failed.
DEFAULT_TIMEOUT_SECONDS = 7200
# Wall-clock cap handed to the agent itself, so it stops on its own well before
# the harness's own timeout_seconds kills it. An agent that finishes the work and
# then loops -- emitting tokens forever without ever ending its turn -- otherwise
# burns the full timeout AND its retry, hours per task, for a task that was
# already done. Observed tasks finish in ~4 minutes, so this leaves wide headroom
# while capping a runaway. Only agents with a native duration flag use it (omp's
# --max-time); 0 disables it.
DEFAULT_AGENT_MAX_TIME_SECONDS = 1800
# How many times to retry a task that failed for a TRANSIENT reason (stream
# error, empty/non-JSON output, timeout, an api/execution error). A task that
# simply ran out of turns (subtype "error_max_turns") is NOT retried -- more
# attempts at the same turn budget will not help; raise max_turns instead.
# 0 disables retries (one attempt per task). Default 1: a transient Bedrock API
# error ("unexpected error during processing. Try your request again") can strand
# a task mid-run with partial artifacts, and one retry recovers it (observed on
# the Sonnet 5 run).
DEFAULT_MAX_RETRIES = 1
# How many focused "top-up" attempts to make when a task's MAIN run finished but
# left some artifacts missing (e.g. the design docs are all present but the run
# ran out of context before writing patch.diff). Unlike a retry, a top-up does
# NOT wipe the existing artifacts and re-run the whole task: it re-invokes the
# agent in a fresh context with a narrow prompt to produce ONLY the missing
# files, reading the ones already on disk. It only fires when the four design
# artifacts already exist (a run that could not even finish the design is a real
# quality failure, not topped up). Each top-up is a separate agent invocation and
# is recorded in metrics.json (agent_invocations, topped_up_artifacts) so a
# completed-but-assisted run stays honestly distinguishable from a clean one.
# 0 disables top-ups.
DEFAULT_MAX_TOPUPS = 1
# The model's true context window, in tokens. Claude Code cannot learn the
# window of a custom model served over a custom base URL, so it never triggers
# auto-compaction and the conversation grows until the endpoint rejects the
# request (HTTP 500 "maximum context length is N tokens"), which the client
# then retries forever. Setting CLAUDE_CODE_AUTO_COMPACT_WINDOW to the true
# window lets auto-compaction fire before the request overflows. 0 means "leave
# unset" -- for a known Claude model or Bedrock, Claude Code already knows the
# window, so no override is needed.
DEFAULT_CONTEXT_WINDOW = 0
# Fraction of the context window at which to run auto-compaction. Kept below 1.0
# so there is headroom for the output-token reserve (max_output_tokens) and
# per-request overhead: at 0.9 of a 262144 window the compact target is 235929,
# leaving ~26k tokens on top of the 16k output reserve.
DEFAULT_AUTO_COMPACT_FRACTION = 0.9

# Where claude -p sends requests. "endpoint" routes through an OpenAI/Anthropic-
# compatible base URL (a local vLLM server, a gateway, the Anthropic API);
# "bedrock" flips claude into native Amazon Bedrock mode (CLAUDE_CODE_USE_BEDROCK=1)
# and names a Bedrock model id, so no base URL or api_key is used.
PROVIDER_ENDPOINT = "endpoint"
PROVIDER_BEDROCK = "bedrock"
# "kiro" routes through the kiro-cli agent's own managed (Bedrock-backed) models
# via its AWS/Builder-ID/Google sign-in. There is no user-supplied base URL or
# region to configure -- kiro-cli cannot target a self-hosted vLLM endpoint -- so
# this provider is only valid with agent=kiro. See docs/kiro-cli-setup.md.
PROVIDER_KIRO = "kiro"
VALID_PROVIDERS = {PROVIDER_ENDPOINT, PROVIDER_BEDROCK, PROVIDER_KIRO}
DEFAULT_PROVIDER = PROVIDER_ENDPOINT

# Which coding agent drives the task. "claude" is Claude Code (`claude -p`);
# "pi" is the pi coding agent (`pi -p --mode json`). The /swe2 task definition is
# identical for both -- only the agent binary and its invocation differ. Both
# support either provider: an OpenAI-compatible endpoint (a self-hosted vLLM
# server or a gateway) or native Amazon Bedrock (pi bundles the AWS SDK
# bedrock-runtime client, invoked as `pi --provider amazon-bedrock`).
AGENT_CLAUDE = "claude"
AGENT_PI = "pi"
# "kiro" is the kiro-cli agent (`kiro-cli chat --no-interactive`). Unlike claude
# and pi it cannot target a self-hosted endpoint: it drives Kiro's own managed
# models and requires provider=kiro. It reports credits + wall-clock time (not
# tokens) on stderr, which the harness maps to a dollar cost. See
# docs/kiro-cli-setup.md.
AGENT_KIRO = "kiro"
# "omp" is the omp coding agent (oh-my-pi), a fork of pi. It speaks the same
# JSON-lines event stream, so the harness reuses pi's result parser verbatim, but
# it differs in three ways that the omp helpers handle: its per-run config is
# YAML (`models.yml` + `config.yml`, not pi's `models.json`), it has no `--skill`
# flag so the SKILL.md is inlined into the prompt the way kiro's is, and it hangs
# waiting for EOF unless its stdin is closed. Like pi it supports
# provider=endpoint and provider=bedrock.
AGENT_OMP = "omp"
# "codex" is the OpenAI Codex agent (`codex exec --json`). It runs
# non-interactively and outputs JSON-lines events. Like kiro it has no --skill
# flag so the SKILL.md is inlined into the prompt. Supports provider=bedrock
# (via AWS_REGION + ambient credentials) and provider=endpoint (via
# OPENAI_BASE_URL / OPENAI_API_KEY). Cost is derived from token counts using
# the local bedrock_pricing table (not metered directly by the CLI).
AGENT_CODEX = "codex"
VALID_AGENTS = {AGENT_CLAUDE, AGENT_PI, AGENT_KIRO, AGENT_OMP, AGENT_CODEX}
DEFAULT_AGENT = AGENT_CLAUDE

# Artifacts are grouped by the coding agent (the "harness") that produced them,
# so a pi run never overwrites a Claude Code run of the same model: the layout is
# ``<model-slug>/<harness-slug>/<repo>/<task>/``. The harness slug is the folder
# name for each agent; "claude" -> "claude-code" (the historical Claude Code
# results, migrated under this name), "pi" -> "pi".
HARNESS_SLUGS = {
    AGENT_CLAUDE: "claude-code",
    AGENT_PI: "pi",
    AGENT_KIRO: "kiro-cli",
    AGENT_OMP: "omp",
    AGENT_CODEX: "codex",
}

# kiro-cli bills in credits, not tokens; the harness translates credits (parsed
# from kiro-cli's stderr summary line) to a dollar cost with a configurable
# per-credit rate. Kiro's add-on/overage rate is $0.04/credit; the blended
# included-allotment rate across paid tiers is $0.02/credit. See
# docs/kiro-cli-setup.md and docs/cost-per-task-methodology.md.
DEFAULT_KIRO_DOLLARS_PER_CREDIT = 0.04

# Which SWE skill drives the run. "swe3" is the DEFAULT single-agent skill: all
# work is done inline in the main loop with NO subagent fan-out, so its token/cost
# accounting is complete and comparable across harnesses (including agents like pi
# that have no subagent mechanism). "swe2" is the older multi-agent variant that
# fans out to parallel Task subagents (codebase analysis + five expert reviews) --
# same six artifacts and rigor, but its main-agent usage undercounts subagent
# tokens. Both produce the same artifacts, so everything downstream (judge,
# summarize, charts) is skill-agnostic.
#
# The default skill maps to the canonical harness folder (e.g. "claude-code"); a
# non-default skill is suffixed ("claude-code-swe2"). So with swe3 as default, a
# swe3 run lands in "claude-code/" (and overwrites older data there as models are
# re-run), while a swe2 run goes to "claude-code-swe2/". See harness_slug.
SKILL_SWE2 = "swe2"
SKILL_SWE3 = "swe3"
VALID_SKILLS = {SKILL_SWE2, SKILL_SWE3}
DEFAULT_SKILL = SKILL_SWE3

# Amazon Bedrock model ids carry a region/vendor inference-profile prefix
# (e.g. "us.anthropic.claude-opus-4-8") and may carry a bracketed context-window
# suffix (e.g. "[1m]"). The /swe skill strips both to name its artifact folder,
# so the harness must derive the same slug to find the artifacts the skill wrote.
# Matches a leading "<region>.<vendor>." such as "us.anthropic." or "eu.meta.".
_BEDROCK_PREFIX_RE = re.compile(r"^[a-z]{2}\.[a-z0-9-]+\.")
# Matches a trailing bracketed suffix such as "[1m]".
_MODEL_SUFFIX_RE = re.compile(r"\[[^\]]*\]$")
# Matches a trailing Anthropic date+version stamp such as "-20251001-v1:0", so a
# dated Bedrock id (us.anthropic.claude-haiku-4-5-20251001-v1:0) folds onto the
# same slug as its short name (claude-haiku-4-5). Only date-versioned ids match;
# plain names (claude-opus-5, glm-5.2, qwen3-coder-30b) are untouched.
_MODEL_DATE_VERSION_RE = re.compile(r"-\d{8}-v\d+:\d+$")


def model_to_slug(model: str, *, normalize_dots: bool = False) -> str:
    """Normalize a model id to the folder slug the /swe skill uses.

    Mirrors the skill's rule (SKILL.md): strip a Bedrock inference-profile
    prefix like ``us.anthropic.`` and a bracketed context-window suffix like
    ``[1m]``. By default nothing else is altered -- dots inside a version (e.g.
    ``glm-5.2``) and existing kebab-case are preserved, matching the on-disk
    folder names of the claude/pi/self-hosted runs.

    ``normalize_dots`` additionally replaces ``.`` with ``-`` (e.g.
    ``claude-haiku-4.5`` -> ``claude-haiku-4-5``). This is used only for the
    kiro agent, whose managed model names carry dots (``claude-haiku-4.5``,
    ``deepseek-3.2``), so kiro artifacts land in the same dash-style folder the
    Bedrock/self-hosted runs already use. It is NOT applied to the other agents,
    whose committed folders and charts intentionally keep the dotted names.

    Args:
        model: The raw model id (e.g. ``us.anthropic.claude-opus-4-8[1m]``).
        normalize_dots: When True, also convert ``.`` to ``-`` (kiro only).

    Returns:
        The artifact-folder slug (e.g. ``claude-opus-4-8``).
    """
    slug = _MODEL_SUFFIX_RE.sub("", model)
    slug = _BEDROCK_PREFIX_RE.sub("", slug)
    slug = _MODEL_DATE_VERSION_RE.sub("", slug)
    if normalize_dots:
        slug = slug.replace(".", "-")
    return slug


def model_to_wire_id(model: str) -> str:
    """Strip only the bracketed suffix, keeping any Bedrock region/vendor prefix.

    The ``[1m]`` style suffix is a harness convention (a context-window hint the
    Claude Code CLI understands); it is not part of a real model id. An API that
    resolves the id itself -- e.g. pi calling Amazon Bedrock through the AWS SDK
    -- needs the clean inference-profile id WITH its region prefix intact
    (``us.anthropic.claude-opus-5``), unlike ``model_to_slug`` which also drops
    the prefix to name the artifact folder.

    Args:
        model: The raw model id (e.g. ``us.anthropic.claude-opus-5[1m]``).

    Returns:
        The wire model id (e.g. ``us.anthropic.claude-opus-5``).
    """
    return _MODEL_SUFFIX_RE.sub("", model)


@lru_cache(maxsize=1)
def _imds_instance_type() -> str | None:
    """Fetch the EC2 instance type from IMDSv2, or None if not on EC2.

    Best-effort and cached: uses a short timeout, tolerates any failure
    (no metadata service, disabled IMDS, non-EC2 host), and never raises so a
    benchmark run is never blocked on this lookup.

    Returns:
        The instance type string (e.g. ``p5en.48xlarge``), or None.
    """
    base = "http://169.254.169.254/latest"
    try:
        token_req = urllib.request.Request(
            f"{base}/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(token_req, timeout=1.0).read().decode()  # nosec B310 - hardcoded IMDS link-local URL
        type_req = urllib.request.Request(
            f"{base}/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(type_req, timeout=1.0).read().decode().strip()  # nosec B310 - hardcoded IMDS link-local URL
    except (urllib.error.URLError, OSError, ValueError):
        return None


class RunnerConfigError(Exception):
    """Raised when the runner config is missing, unparseable, or invalid."""


class RunnerConfig(BaseModel):
    """Run-time parameters for the headless SWE benchmark harness."""

    model_config = ConfigDict(extra="forbid")

    # Which coding agent drives the /swe2 task: "claude" (Claude Code, the
    # default) or "pi" (the pi coding agent). The task definition is the same for
    # both; only the agent binary and how it is invoked differ. See VALID_AGENTS.
    agent: str = Field(
        default=DEFAULT_AGENT,
        description="Coding agent that runs the task: 'claude' (Claude Code) or "
        "'pi' (pi coding agent). Both support provider=endpoint or "
        "provider=bedrock.",
    )

    # Which SWE skill to run: "swe2" (default, multi-agent fan-out) or "swe3"
    # (single-agent, no subagents). Same six artifacts either way. See VALID_SKILLS.
    skill: str = Field(
        default=DEFAULT_SKILL,
        description="SWE skill: 'swe2' (default, multi-agent fan-out) or 'swe3' "
        "(single-agent, no subagents). Same artifacts; only orchestration differs.",
    )

    # Routing: how the agent reaches the model.
    #   "endpoint" (default): route through an OpenAI/Anthropic-compatible base
    #       URL (a local vLLM server, a gateway, or the Anthropic API).
    #   "bedrock": drive models directly on Amazon Bedrock via the native
    #       CLAUDE_CODE_USE_BEDROCK path; no base URL or api_key is used.
    provider: str = Field(
        default=DEFAULT_PROVIDER,
        description="How claude -p reaches the model: 'endpoint' (base URL) or "
        "'bedrock' (native Amazon Bedrock).",
    )
    endpoint: str | None = Field(
        default=None,
        description="Base URL of the OpenAI/Anthropic-compatible endpoint "
        "(e.g. http://127.0.0.1:8000). Required for provider=endpoint; ignored "
        "for provider=bedrock.",
    )
    model: str | None = Field(
        default=None,
        description="Model name/id to pass to claude --model. For provider=bedrock "
        "this is a Bedrock model id or inference profile (e.g. "
        "us.anthropic.claude-opus-4-8). Left unset in the committed config so one "
        "file serves every model; supply it with --model.",
    )
    api_key: str = Field(default="local", description="API key sent to the endpoint.")
    aws_region: str | None = Field(
        default=None,
        description="AWS region for provider=bedrock (e.g. us-east-1). Falls back "
        "to AWS_REGION/AWS_DEFAULT_REGION from the environment when unset.",
    )
    instance_type: str | None = Field(
        default=None,
        description="EC2 instance type the model is served on (e.g. p5en.48xlarge). "
        "Recorded in each run's metrics.json for hardware provenance. Falls back to "
        "the EC2 instance metadata service (IMDSv2) when unset; null if unavailable.",
    )
    tensor_parallel_size: int | None = Field(
        default=None,
        description="Tensor-parallel size (vLLM --tensor-parallel-size / TP) the "
        "model is served with. Recorded in the metrics.json serving block for "
        "provenance; null when unknown (e.g. Bedrock).",
    )
    precision: str | None = Field(
        default=None,
        description="Weight precision the model is served at, e.g. BF16 or FP8. "
        "Recorded in the metrics.json serving block for provenance; null when "
        "unknown.",
    )

    # What to run and where outputs go.
    dataset: str | None = Field(
        default=None,
        description="Path to the benchmark dataset YAML file. Left unset in the "
        "committed config so one file serves every dataset; supply it with --dataset.",
    )
    output_dir: str = Field(
        default="swe-benchmark-data",
        description="Directory (relative to repo root) where artifacts land.",
    )
    clone_dir: str = Field(
        default="/tmp",  # nosec B108 - clone parent; each repo lands in a mkdtemp subdir
        description="Parent directory for per-task temporary repo clones.",
    )
    tasks: list[str] = Field(
        default_factory=list,
        description="Task ids to run. Empty means every task in the dataset.",
    )
    concurrency: int = Field(
        default=1,
        ge=1,
        description="How many tasks to run at once. 1 (default) runs serially. "
        "Values above 1 overlap runs on the endpoint, which invalidates the "
        "single-tenant vllm_prometheus window-delta metrics for those runs.",
    )

    # How claude -p is invoked.
    permission_mode: str = Field(default=DEFAULT_PERMISSION_MODE)
    allowed_tools: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ALLOWED_TOOLS)
    )
    max_turns: int = Field(default=DEFAULT_MAX_TURNS, ge=1)
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=1)
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1)
    agent_max_time_seconds: int = Field(
        default=DEFAULT_AGENT_MAX_TIME_SECONDS,
        ge=0,
        description="Wall-clock budget passed to the agent itself so it stops "
        "before the harness timeout fires (omp --max-time). Caps a runaway "
        "generation loop that would otherwise burn timeout_seconds plus a "
        "retry. 0 disables, leaving only the harness timeout.",
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        description="Retries for a task that failed transiently (not for a "
        "turn-budget exhaustion, which is never retried). 0 disables retries.",
    )
    max_topups: int = Field(
        default=DEFAULT_MAX_TOPUPS,
        ge=0,
        description="Focused top-up attempts when the main run left artifacts "
        "missing but the design docs are complete. A top-up re-invokes the agent "
        "in a fresh context to produce ONLY the missing files (it does not wipe "
        "or redo the existing ones), and is flagged in metrics.json. 0 disables.",
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        ge=0,
        description="Model's true context window in tokens; calibrates "
        "auto-compaction for custom models. 0 leaves it unset.",
    )
    auto_compact_fraction: float = Field(
        default=DEFAULT_AUTO_COMPACT_FRACTION,
        gt=0.0,
        le=1.0,
        description="Fraction of context_window at which auto-compaction fires.",
    )
    settings_file: str | None = Field(
        default=None,
        description="Optional claude --settings JSON file (e.g. the vLLM config).",
    )
    kiro_dollars_per_credit: float = Field(
        default=DEFAULT_KIRO_DOLLARS_PER_CREDIT,
        ge=0.0,
        description="USD per kiro-cli credit, used only for agent=kiro to turn the "
        "credits it reports into a dollar cost per task. Default 0.04 (Kiro's "
        "add-on/overage rate); use 0.02 for the blended included-allotment rate. "
        "Set to your real plan's effective rate. See docs/kiro-cli-setup.md.",
    )

    @property
    def is_bedrock(self) -> bool:
        """True when the agent should route natively to Amazon Bedrock."""
        return self.provider == PROVIDER_BEDROCK

    @property
    def is_pi(self) -> bool:
        """True when the pi coding agent drives the task (instead of Claude Code)."""
        return self.agent == AGENT_PI

    @property
    def is_omp(self) -> bool:
        """True when the omp (oh-my-pi) coding agent drives the task."""
        return self.agent == AGENT_OMP

    @property
    def is_kiro(self) -> bool:
        """True when the kiro-cli agent drives the task."""
        return self.agent == AGENT_KIRO

    @property
    def is_codex(self) -> bool:
        """True when the codex agent drives the task."""
        return self.agent == AGENT_CODEX

    @property
    def harness_slug(self) -> str:
        """Folder name for the coding agent (harness) that produced a run's artifacts.

        Just the agent slug (``claude`` -> ``claude-code``, ``pi`` -> ``pi``; see
        ``HARNESS_SLUGS``). The skill is a SEPARATE path level (see ``skill`` and
        ``_artifact_dir``), so the full layout is
        ``<model>/<harness>/<skill>/<repo>/<task>/`` -- model, harness, and skill
        are each their own dimension. swe2 and swe3 are sibling folders that never
        collide.

        Returns:
            The harness folder name for this run's agent.
        """
        return HARNESS_SLUGS[self.agent]

    @property
    def auto_compact_window(self) -> int | None:
        """Token budget for CLAUDE_CODE_AUTO_COMPACT_WINDOW, or None if unset.

        Computed as ``floor(context_window * auto_compact_fraction)`` so
        auto-compaction fires with headroom below the model's true window. When
        ``context_window`` is 0 (the default) this returns None and the harness
        leaves the env var unset -- Claude Code already knows the window for a
        known Claude model or Bedrock, so no override is needed there.

        Returns:
            The compact-trigger token budget, or None when no window is set.
        """
        if self.context_window <= 0:
            return None
        return int(self.context_window * self.auto_compact_fraction)

    @property
    def model_slug(self) -> str:
        """The artifact-folder name for this model.

        ``model`` is the full id passed to ``claude --model`` (for Bedrock, an
        inference profile such as ``us.anthropic.claude-opus-4-8``). The /swe
        skill strips the vendor/region prefix and any ``[...]`` suffix to name
        its output folder, so the harness derives the same slug -- otherwise it
        looks for artifacts in a folder the skill never wrote to. See
        ``model_to_slug``.

        Returns:
            The normalized folder slug (e.g. ``claude-opus-4-8``).
        """
        # kiro's managed model names carry dots (claude-haiku-4.5); dash them so
        # kiro artifacts share the folder the Bedrock/self-hosted runs use.
        return (
            model_to_slug(self.model, normalize_dots=self.is_kiro) if self.model else ""
        )

    def resolved_region(self) -> str | None:
        """Return the AWS region for Bedrock, falling back to the environment.

        Returns:
            The configured ``aws_region``, else ``AWS_REGION`` /
            ``AWS_DEFAULT_REGION`` from the environment, else None.
        """
        return (
            self.aws_region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )

    def resolved_instance_type(self) -> str | None:
        """Return the EC2 instance type the run executes on, for provenance.

        Resolution order: the configured ``instance_type``, else the
        ``EC2_INSTANCE_TYPE`` environment variable, else the EC2 instance
        metadata service (IMDSv2). The IMDS lookup is best-effort with a short
        timeout and never raises -- off EC2 (or if metadata is disabled) it
        simply returns None, so the run is unaffected.

        Returns:
            The instance type (e.g. ``p5en.48xlarge``), or None if unknown.
        """
        if self.instance_type:
            return self.instance_type
        env = os.environ.get("EC2_INSTANCE_TYPE")
        if env:
            return env
        return _imds_instance_type()

    def validate_semantics(self) -> None:
        """Check fields the type system cannot.

        Raises:
            RunnerConfigError: If a value is present but invalid.
        """
        if self.provider not in VALID_PROVIDERS:
            raise RunnerConfigError(
                f"provider '{self.provider}' not in {sorted(VALID_PROVIDERS)}."
            )
        if self.agent not in VALID_AGENTS:
            raise RunnerConfigError(
                f"agent '{self.agent}' not in {sorted(VALID_AGENTS)}."
            )
        if self.skill not in VALID_SKILLS:
            raise RunnerConfigError(
                f"skill '{self.skill}' not in {sorted(VALID_SKILLS)}."
            )
        # pi supports both an OpenAI-compatible endpoint (a local vLLM server, a
        # gateway) and native Amazon Bedrock (it bundles the AWS SDK's
        # bedrock-runtime client + credential chain, invoked as
        # `pi --provider amazon-bedrock`). No routing combination is rejected
        # here; _validate_routing checks the fields each provider needs.
        #
        # kiro-cli is the exception: it only drives its own managed models, so
        # agent=kiro and provider=kiro must go together (neither pairs with
        # anything else).
        if (self.agent == AGENT_KIRO) != (self.provider == PROVIDER_KIRO):
            raise RunnerConfigError(
                "agent=kiro and provider=kiro must be used together: kiro-cli only "
                "drives Kiro's managed models (no endpoint/bedrock routing), and no "
                "other agent uses provider=kiro. See docs/kiro-cli-setup.md."
            )
        if not self.model:
            raise RunnerConfigError(
                "model is required. Set it in the config file or pass --model "
                "(e.g. --model qwen3-coder-30b, or a Bedrock model id such as "
                "us.anthropic.claude-opus-4-8 for provider=bedrock)."
            )
        if not self.dataset:
            raise RunnerConfigError(
                "dataset is required. Set it in the config file or pass --dataset "
                "(e.g. --dataset dataset/mcp-gateway-registry.yaml)."
            )
        if self.permission_mode not in VALID_PERMISSION_MODES:
            raise RunnerConfigError(
                f"permission_mode '{self.permission_mode}' not in "
                f"{sorted(VALID_PERMISSION_MODES)}."
            )
        self._validate_routing()

    def _validate_routing(self) -> None:
        """Validate provider-specific routing fields.

        Raises:
            RunnerConfigError: If routing fields are missing or malformed.
        """
        # kiro-cli manages its own routing and sign-in; there is no endpoint or
        # region for the harness to supply or validate.
        if self.provider == PROVIDER_KIRO:
            return
        if self.is_bedrock:
            if not self.resolved_region():
                raise RunnerConfigError(
                    "provider=bedrock requires an AWS region. Set aws_region in "
                    "the config, pass --aws-region, or export AWS_REGION."
                )
            return
        if not self.endpoint:
            raise RunnerConfigError(
                "endpoint is required for provider=endpoint. Set it in the config "
                "file or pass --endpoint (e.g. http://127.0.0.1:8000)."
            )
        if not self.endpoint.startswith(("http://", "https://")):
            raise RunnerConfigError(
                f"endpoint '{self.endpoint}' must start with http:// or https://"
            )


def _apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge CLI overrides onto raw config data (CLI wins).

    Args:
        data: The parsed YAML config mapping.
        overrides: CLI-supplied values; None entries are ignored.

    Returns:
        A new mapping with non-None overrides applied.
    """
    merged = dict(data)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def load_runner_config(
    path: str | Path | None,
    overrides: dict[str, Any] | None = None,
) -> RunnerConfig:
    """Load the runner config from YAML and apply CLI overrides.

    Args:
        path: Path to the config YAML file, or None to build purely from
            overrides (useful for CLI-only runs).
        overrides: CLI-supplied values that take precedence over the file.

    Returns:
        The validated RunnerConfig.

    Raises:
        RunnerConfigError: If the file is missing, unparseable, or invalid.
    """
    overrides = overrides or {}

    if path is None:
        raw: dict[str, Any] = {}
    else:
        file_path = Path(path)
        if not file_path.exists():
            raise RunnerConfigError(f"Runner config not found: {file_path}")
        try:
            loaded = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise RunnerConfigError(f"Failed to parse {file_path}: {exc}") from exc
        if loaded is None:
            raw = {}
        elif isinstance(loaded, dict):
            raw = loaded
        else:
            raise RunnerConfigError(f"{file_path}: top level must be a mapping")

    merged = _apply_overrides(raw, overrides)

    try:
        config = RunnerConfig.model_validate(merged)
    except ValidationError as exc:
        raise RunnerConfigError(f"Invalid runner config:\n{exc}") from exc

    config.validate_semantics()
    return config


def _summarize(config: RunnerConfig) -> None:
    """Log a short human-readable summary of the runner config."""
    logger.info("Runner config:")
    logger.info("  agent: %s", config.agent)
    logger.info("  provider: %s", config.provider)
    if config.is_bedrock:
        logger.info("  aws_region: %s", config.resolved_region())
    else:
        logger.info("  endpoint: %s", config.endpoint)
    logger.info("  model: %s", config.model)
    logger.info(
        "  serving: instance_type=%s tensor_parallel_size=%s precision=%s",
        config.resolved_instance_type(),
        config.tensor_parallel_size,
        config.precision,
    )
    logger.info("  dataset: %s", config.dataset)
    logger.info("  output_dir: %s", config.output_dir)
    logger.info("  clone_dir: %s", config.clone_dir)
    logger.info("  tasks: %s", config.tasks or "(all)")
    logger.info("  concurrency: %s", config.concurrency)
    logger.info("  permission_mode: %s", config.permission_mode)
    logger.info("  max_turns: %s", config.max_turns)
    logger.info("  max_retries: %s", config.max_retries)
    logger.info("  max_topups: %s", config.max_topups)
    if config.auto_compact_window is not None:
        logger.info(
            "  context_window: %s (auto-compact at %s, fraction %s)",
            config.context_window,
            config.auto_compact_window,
            config.auto_compact_fraction,
        )
    else:
        logger.info("  context_window: (unset -- relying on Claude Code's default)")
    logger.info("  allowed_tools: %s", ", ".join(config.allowed_tools))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate and summarize a SWE benchmark runner config.",
        epilog="Example:\n  uv run scripts/runner_config.py config/runner.example.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Path to the runner config YAML file")
    parser.add_argument(
        "--agent",
        help="Override: coding agent that runs the task (claude | pi | omp | kiro)",
    )
    parser.add_argument(
        "--provider", help="Override: routing provider (endpoint | bedrock)"
    )
    parser.add_argument("--endpoint", help="Override: API endpoint base URL")
    parser.add_argument("--model", help="Override: model name (as with the harness)")
    parser.add_argument("--dataset", help="Override: dataset YAML path")
    parser.add_argument(
        "--aws-region", help="Override: AWS region for provider=bedrock"
    )
    parser.add_argument(
        "--instance-type",
        help="Override: EC2 instance type served on (e.g. p5en.48xlarge)",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the given runner config file and print a summary."""
    args = _parse_args()
    overrides = {
        "agent": args.agent,
        "provider": args.provider,
        "endpoint": args.endpoint,
        "model": args.model,
        "dataset": args.dataset,
        "aws_region": args.aws_region,
        "instance_type": args.instance_type,
    }
    try:
        config = load_runner_config(args.config, overrides)
    except RunnerConfigError as exc:
        logger.error("Invalid runner config: %s", exc)
        sys.exit(1)
    _summarize(config)
    logger.info("Runner config is valid.")


if __name__ == "__main__":
    main()
