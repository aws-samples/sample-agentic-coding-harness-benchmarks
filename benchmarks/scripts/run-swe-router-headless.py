#!/usr/bin/env python3
"""Run the swe-router skill headless over a dataset to collect its step-1 judgments.

WHY THIS EXISTS
---------------
``eval_swe_router.py`` can replay the router's SELECTION over a dataset, but
selection is only half the skill. The other half is the judgment it opens with:
read the repo, read the task, and decide a quality floor from the consequence of
the change being wrong plus a complexity tier. That half is an LLM call, so a
script cannot fake it -- and it is the half that decides the outcome, since the
floor drives which model gets picked.

So this drives a real agent through the skill's steps 1 and 1b, once per task,
in the task's own cloned repository, and writes the ``(floor, tier)`` tuples to
JSON in the shape ``eval_swe_router.py --judged-inputs`` consumes. The two
scripts together run the whole skill end to end: judgment here, selection and
the join to measured runs there.

WHAT IS DELIBERATELY DIFFERENT FROM THE SKILL
---------------------------------------------
The skill's natural output is a prose recommendation. Here the agent is asked
for steps 1 and 1b ONLY, emitting a JSON object, and is told not to run
route.py. That is a deviation and it is on purpose: routing centrally, from one
fixed candidate list, keeps every task's selection comparable. An agent running
route.py itself would pass whatever ``--available`` it guessed, and no two tasks
would be answered on the same basis.

REPEATS
-------
A floor is a judgment, so one sample per task says little about whether the
judgment is stable. ``--repeats`` runs the whole pass N times and records every
judgment, then consolidates (median floor, modal tier) for the downstream eval.
The spread is reported per task: a task that draws 65 one run and 75 the next is
a finding about the skill, not noise to average away.

Run from the ``benchmarks/`` directory:

    uv run scripts/run-swe-router-headless.py --agent omp --provider bedrock \\
        --model us.anthropic.claude-opus-5 --repeats 3
    uv run scripts/run-swe-router-headless.py --tasks configurable-ui-title --repeats 1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import shutil
import statistics
import subprocess  # nosec B404 - list args, no shell, agent binary is hardcoded
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent.parent
BENCHMARKS_DIR = _SCRIPTS_DIR.parent

from dataset_loader import Dataset, DatasetError, Task, load_dataset  # noqa: E402
from runner_config import (  # noqa: E402
    AGENT_CLAUDE,
    AGENT_OMP,
    RunnerConfig,
    RunnerConfigError,
    load_runner_config,
)

# The router skill lives with the repo's other skills. Its SKILL.md is inlined
# into the prompt: omp has no --skill flag, and Claude Code's slash command is
# not available to a bare -p prompt against an arbitrary working directory.
SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "swe-router"
SKILL_MD = SKILL_DIR / "SKILL.md"

VALID_TIERS = ("trivial", "low", "medium", "high")
# The skill's floor table runs 55-75, and its one adjustment adds 5. Anything
# outside that band means the agent invented a scale, which is a parse failure
# rather than a judgment worth recording.
MIN_FLOOR = 55.0
MAX_FLOOR = 80.0

DEFAULT_DATASET = "dataset/mcp-gateway-registry-v2.yaml"
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_REPEATS = 3
DEFAULT_CONCURRENCY = 4
# A judgment is a short read-and-decide, not an implementation, so the agent
# needs far less room than a /swe3 run. Capping it keeps a confused run from
# burning the full timeout.
DEFAULT_AGENT_MAX_TIME_SECONDS = 600

# Matches a ```json fenced block, the shape the prompt asks for.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _load_harness() -> Any:
    """Import ``run-swe-headless.py`` by path and return the module.

    The harness owns how a repo is cloned, how omp is invoked, and how its event
    stream is turned into token counts. Re-deriving any of that here would give
    the two scripts two answers to the same question, so it is imported instead.
    Its filename carries a dash and is not a valid module identifier, hence the
    by-path import (the same approach ``preflight_check.py`` uses).

    Returns:
        The loaded harness module.

    Raises:
        RuntimeError: If the harness module cannot be loaded.
    """
    path = _SCRIPTS_DIR / "run-swe-headless.py"
    spec = importlib.util.spec_from_file_location("swe_harness", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load harness module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARNESS = _load_harness()


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with a trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_prompt(task: Task, clone_path: Path) -> str:
    """Build the step-1 prompt: the skill, the task, and the output contract.

    The skill text is passed verbatim so the agent judges by the same rules a
    real invocation would. Everything after it is scaffolding the skill cannot
    supply for itself: which task, where the repo is, and the fact that only
    steps 1 and 1b are wanted.

    Args:
        task: The task being judged.
        clone_path: The cloned repository the change would land in.

    Returns:
        The prompt string.
    """
    skill_md = SKILL_MD.read_text(encoding="utf-8")
    return "\n".join(
        [
            skill_md,
            "",
            "===TASK===",
            "",
            "Apply ONLY steps 1 and 1b of the skill above to the coding task "
            "below: establish the quality floor from the consequence of the "
            "change being wrong, and classify the task's complexity tier.",
            "",
            "Do NOT run route.py. Do NOT recommend a model. Do NOT read "
            "models.json, model-aliases.json or allowed-models.txt -- selection "
            "is handled separately and is not your job here. Do NOT write, edit "
            "or run any code in the repository.",
            "",
            f"The repository the change would land in is cloned at {clone_path} "
            "and is your working directory. Start by reading its agent map "
            "(AGENTS.md, else CLAUDE.md, else README.md) as step 1 instructs, "
            "and use the project's own language about what it treats as "
            "sensitive. Read whatever else you need to judge the task, but do "
            "not modify anything.",
            "",
            "Finish by printing EXACTLY one fenced JSON block and nothing after it:",
            "",
            "```json",
            "{",
            '  "floor": <number from the skill\'s floor table, plus the +5 '
            "adjustment if it applies>,",
            f'  "tier": "<one of: {", ".join(VALID_TIERS)}>",',
            '  "base_floor": <the table row you started from, before any adjustment>,',
            '  "adjustment": <5 if you applied the single-specific-thing '
            "adjustment, else 0>,",
            '  "consequence": "<one sentence: what happens if this change is wrong>",',
            '  "reason": "<two or three sentences: why that floor row and why '
            'that tier>"',
            "}",
            "```",
            "",
            f"Task id: {task.id}",
            "",
            "Task description:",
            task.problem_statement or "(see reference issue)",
        ]
    )


def _extract_judgment(text: str) -> dict[str, Any]:
    """Pull the judgment object out of the agent's final message.

    Prefers the last fenced JSON block (what the prompt asks for) and falls back
    to the last bare object that parses and carries both keys, so a model that
    drops the fence is still read rather than discarded.

    Args:
        text: The agent's final message.

    Returns:
        The parsed judgment.

    Raises:
        ValueError: If no block carries a usable floor and tier.
    """
    candidates = [m.group(1) for m in _JSON_FENCE_RE.finditer(text)]
    if not candidates:
        # No fence: scan for balanced objects and keep the ones that parse.
        starts = [i for i, ch in enumerate(text) if ch == "{"]
        for start in reversed(starts):
            for end in range(len(text), start, -1):
                chunk = text[start:end]
                try:
                    parsed = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and "floor" in parsed:
                    candidates = [chunk]
                    break
            if candidates:
                break
    for chunk in reversed(candidates):
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        floor, tier = parsed.get("floor"), parsed.get("tier")
        if not isinstance(floor, (int, float)) or tier not in VALID_TIERS:
            continue
        if not MIN_FLOOR <= float(floor) <= MAX_FLOOR:
            raise ValueError(
                f"floor {floor} is outside the skill's {MIN_FLOOR:.0f}-"
                f"{MAX_FLOOR:.0f} range; the agent invented a scale"
            )
        parsed["floor"] = float(floor)
        return parsed
    raise ValueError(
        f"no JSON object with a valid floor and tier in the agent's reply: "
        f"{text.strip()[-500:]!r}"
    )


def _omp_cmd(config: RunnerConfig, prompt: str) -> list[str]:
    """Assemble the ``omp -p --mode json`` argument vector for a judgment run.

    Mirrors the harness's own omp invocation (``--mode json`` for the parseable
    event stream, ``--no-session`` to stay ephemeral, a trailing ``--`` because
    the inlined SKILL.md opens with YAML frontmatter that omp would otherwise
    read as flags). It differs in one way: no ``--auto-approve``. A judgment run
    only reads, so withholding write approval is a cheap guarantee that a
    confused agent cannot edit the repository it is judging.

    Args:
        config: The runner config (model, provider).
        prompt: The step-1 prompt.

    Returns:
        The command argument vector.
    """
    if config.is_bedrock:
        model = f"{HARNESS.OMP_PROVIDER_BEDROCK}/{config.model}"
    else:
        model = f"{HARNESS.OMP_PROVIDER_VLLM}/{config.model}"
    cmd = ["omp", "-p", "--mode", "json", "--no-session", "--model", model]
    if config.agent_max_time_seconds:
        cmd += [f"--max-time={config.agent_max_time_seconds}"]
    return [*cmd, "--", prompt]


def _omp_final_text(events: list[dict[str, Any]]) -> str:
    """Concatenate the text of the last assistant message in an omp stream.

    Args:
        events: The parsed JSON-lines events omp emitted, in order.

    Returns:
        The final assistant message's text, or "" when there is none.
    """
    texts: list[str] = []
    for event in events:
        message = event.get("message") or {}
        if event.get("type") != "message_end" or message.get("role") != "assistant":
            continue
        content = message.get("content")
        parts: list[str] = []
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        joined = "\n".join(p for p in parts if p.strip())
        if joined.strip():
            texts.append(joined)
    return texts[-1] if texts else ""


def _run_omp_judgment(
    config: RunnerConfig,
    prompt: str,
    cwd: Path,
    agent_dir: Path,
) -> tuple[str, dict[str, Any]]:
    """Run one omp judgment and return its final text plus usage metrics.

    Args:
        config: The runner config.
        prompt: The step-1 prompt.
        cwd: The cloned repository, used as the agent's working directory so the
            skill's "read the repo's agent map" step resolves to the repo under
            judgement rather than to this one.
        agent_dir: Per-run omp config dir, keeping the run off the developer's
            global ``~/.omp``.

    Returns:
        The final assistant text and the claude-shaped result dict.

    Raises:
        RuntimeError: If omp times out or emits no parseable events.
    """
    if not config.is_bedrock:
        HARNESS._write_omp_config(config, agent_dir)
    else:
        agent_dir.mkdir(parents=True, exist_ok=True)
    env = HARNESS._build_omp_env(config, agent_dir)
    start = time.time()
    events: list[dict[str, Any]] = []
    # stdin=DEVNULL is required, not cosmetic: omp treats an inherited stdin as a
    # piped prompt and blocks on EOF, ignoring the positional prompt entirely.
    proc = subprocess.Popen(  # nosec B603 - hardcoded 'omp', list args, no shell
        _omp_cmd(config, prompt),
        env=env,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    try:
        for line in proc.stdout or []:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue  # omp interleaves human-readable startup notices
        proc.wait(timeout=max(config.timeout_seconds - (time.time() - start), 1))
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise RuntimeError(f"omp timed out after {config.timeout_seconds}s") from exc
    stderr = (proc.stderr.read() if proc.stderr else "") or ""
    if not events:
        raise RuntimeError(
            f"omp produced no JSON events (exit {proc.returncode}): "
            f"{stderr.strip()[:500]}"
        )
    elapsed = time.time() - start
    result = HARNESS._pi_result_from_events(events, elapsed)
    result["_elapsed_seconds"] = round(elapsed, 1)
    return _omp_final_text(events), result


def _run_claude_judgment(
    config: RunnerConfig,
    prompt: str,
    cwd: Path,
) -> tuple[str, dict[str, Any]]:
    """Run one Claude Code judgment and return its final text plus usage metrics.

    Args:
        config: The runner config.
        prompt: The step-1 prompt.
        cwd: The cloned repository, used as the working directory.

    Returns:
        The final assistant text and the claude-shaped result dict.

    Raises:
        RuntimeError: If claude times out or emits no parseable result.
    """
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        config.model,
        "--output-format",
        "json",
        "--permission-mode",
        # A judgment only reads, so the run gets no write permission at all.
        "plan",
        "--max-turns",
        str(config.max_turns),
        "--settings",
        HARNESS._build_settings_arg(config),
    ]
    env = HARNESS._build_env(config)
    start = time.time()
    try:
        proc = subprocess.run(  # nosec B603 - hardcoded 'claude', list args, no shell
            cmd,
            env=env,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"claude timed out after {config.timeout_seconds}s") from exc
    elapsed = time.time() - start
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"claude produced no parseable JSON (exit {proc.returncode}): "
            f"{proc.stdout.strip()[:300]} {proc.stderr.strip()[:300]}"
        ) from exc
    result["_elapsed_seconds"] = round(elapsed, 1)
    return result.get("result") or "", result


def _judge_task(
    config: RunnerConfig,
    dataset: Dataset,
    task: Task,
    attempt: int,
    label: str,
) -> dict[str, Any]:
    """Clone a task's repo, run one judgment in it, and return the result.

    The clone is always removed, including on failure, so a long pass cannot
    fill the disk with abandoned checkouts.

    Args:
        config: The runner config.
        dataset: The loaded dataset, for default-ref resolution.
        task: The task to judge.
        attempt: 1-based repeat index, recorded on the judgment.
        label: Log prefix.

    Returns:
        A judgment record: the tuple plus provenance and cost, or an ``error``.
    """
    ref = dataset.resolved_ref(task)
    # The harness names its clone parent after the task alone and wipes it before
    # cloning -- safe there (one run per task), unsafe here: two repeats of the
    # SAME task in flight would delete each other's checkout mid-run, and the
    # cleanup below would remove the survivor. Giving each attempt its own parent
    # makes the collision impossible rather than merely unlikely.
    clone_dir = str(Path(config.clone_dir) / f"router-attempt-{attempt}")
    clone_path = HARNESS._clone_repo(task, ref, clone_dir, log_prefix=label)
    record: dict[str, Any] = {
        "task": task.id,
        "attempt": attempt,
        "ref": ref,
        "judged_at": _utc_now_iso(),
    }
    try:
        prompt = _build_prompt(task, clone_path)
        if config.agent == AGENT_OMP:
            text, result = _run_omp_judgment(
                config, prompt, clone_path, clone_path.parent / "omp-agent"
            )
        else:
            text, result = _run_claude_judgment(config, prompt, clone_path)
        metrics = HARNESS._metrics_from_result(
            result, result.get("_elapsed_seconds", 0)
        )
        record["metrics"] = {
            "input_tokens": metrics.get("input_tokens"),
            "output_tokens": metrics.get("output_tokens"),
            "cache_read_tokens": metrics.get("cache_read_tokens"),
            "cache_creation_tokens": metrics.get("cache_creation_tokens"),
            "num_turns": metrics.get("num_turns"),
            "latency_seconds": metrics.get("latency_seconds"),
            "total_cost_usd": metrics.get("total_cost_usd"),
        }
        record.update(_extract_judgment(text))
        logger.info(
            "  %s judged floor=%s tier=%s (%ss, $%s)",
            label,
            record["floor"],
            record["tier"],
            record["metrics"]["latency_seconds"],
            record["metrics"]["total_cost_usd"],
        )
    except (RuntimeError, ValueError) as exc:
        record["error"] = str(exc)[:1000]
        logger.error("  %s FAILED: %s", label, record["error"])
    finally:
        shutil.rmtree(clone_path.parent, ignore_errors=True)
    return record


def _consolidate(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce a task's repeated judgments to the one tuple the eval will route on.

    Median floor and modal tier, because a floor is ordinal (a middle value is
    meaningful) while a tier is categorical (it is not). The spread is kept
    beside them: a task whose floor moves between runs is telling you the
    skill's judgment is unstable there, which matters more than the average.

    Args:
        judgments: Every successful judgment for one task.

    Returns:
        The consolidated tuple plus its agreement statistics.

    Raises:
        ValueError: If there are no judgments to consolidate.
    """
    if not judgments:
        raise ValueError("no successful judgments to consolidate")
    floors = [j["floor"] for j in judgments]
    tiers = [j["tier"] for j in judgments]
    tier_counts = Counter(tiers)
    modal_tier, modal_n = tier_counts.most_common(1)[0]
    best = max(
        (j for j in judgments if j["tier"] == modal_tier),
        key=lambda j: -abs(j["floor"] - statistics.median(floors)),
    )
    return {
        "floor": statistics.median(floors),
        "tier": modal_tier,
        "base_floor": best.get("base_floor"),
        "adjustment": best.get("adjustment"),
        "consequence": best.get("consequence"),
        "reason": best.get("reason"),
        "attempts": len(judgments),
        "floors_seen": sorted(floors),
        "floor_spread": max(floors) - min(floors),
        "floor_unanimous": len(set(floors)) == 1,
        "tiers_seen": dict(sorted(tier_counts.items())),
        "tier_unanimous": modal_n == len(tiers),
    }


def _collect(
    config: RunnerConfig,
    dataset: Dataset,
    tasks: list[Task],
    repeats: int,
    concurrency: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run every (task, repeat) judgment and consolidate the successful ones.

    Args:
        config: The runner config.
        dataset: The loaded dataset.
        tasks: Tasks to judge.
        repeats: How many independent judgments per task.
        concurrency: How many judgments to run at once.

    Returns:
        Every judgment record, and the consolidated ``{task_id: tuple}`` map.
    """
    jobs = [(task, attempt) for attempt in range(1, repeats + 1) for task in tasks]
    total = len(jobs)
    logger.info(
        "judging %d task(s) x %d repeat(s) = %d run(s), concurrency %d",
        len(tasks),
        repeats,
        total,
        concurrency,
    )
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _judge_task,
                config,
                dataset,
                task,
                attempt,
                f"[{task.id} #{attempt}] {i} of {total}",
            ): (task, attempt)
            for i, (task, attempt) in enumerate(jobs, start=1)
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda r: (r["task"], r["attempt"]))

    consolidated: dict[str, Any] = {}
    for task in tasks:
        good = [r for r in records if r["task"] == task.id and "error" not in r]
        if not good:
            logger.error("task %s produced no usable judgment", task.id)
            continue
        consolidated[task.id] = _consolidate(good)
    return records, consolidated


def _judgments_markdown(payload: dict[str, Any]) -> str:
    """Render the judged tuples as a markdown document.

    The table is the deliverable: for every task, what this agent and model
    decided the quality floor and complexity tier are, and why. The spread
    columns come first among the caveats because a floor that moves between
    identical runs is the single most important thing a reader can know about
    how much to trust the rest.

    Args:
        payload: The full judged-inputs mapping.

    Returns:
        The markdown source.
    """
    meta = payload["judged_by"]
    tasks = payload["tasks"]
    unstable_floor = [t for t, v in tasks.items() if not v["floor_unanimous"]]
    unstable_tier = [t for t, v in tasks.items() if not v["tier_unanimous"]]
    lines = [
        f"# What {meta['harness']} + {meta['model']} judges each task to need",
        "",
        "The `swe-router` skill opens by reading the repository and the task "
        "and deciding two things: a **quality floor**, from the consequence of "
        "the change being wrong, and a **complexity tier**. Everything the skill "
        "does afterwards is arithmetic on those two numbers. It is also the "
        "only step with no measurement behind it.",
        "",
        f"This is that step, run for real: `{meta['harness']}` driving "
        f"`{meta['model']}` over every task in "
        f"`{meta['dataset']}`, each one in its own clone of the target "
        f"repository at the task's pinned ref, {meta['repeats']} independent "
        "time(s) per task. Each run got the skill verbatim and a request for "
        "steps 1 and 1b only. None of them selected a model.",
        "",
        f"- **Repeats.** {meta['repeats']} per task. Floor is the median across "
        "them, tier the mode. Where the runs disagreed, every value seen is in "
        "the last column.",
        f"- **Stability.** The floor was unanimous on "
        f"{len(tasks) - len(unstable_floor)}/{len(tasks)} tasks, the tier on "
        f"{len(tasks) - len(unstable_tier)}/{len(tasks)}."
        + (
            f" Floor disagreed on: {', '.join(sorted(unstable_floor))}."
            if unstable_floor
            else ""
        ),
        f"- **Runs.** {meta['runs_ok']} succeeded, {meta['runs_failed']} failed. "
        f"Judging cost ${meta.get('judging_cost_usd')}.",
        f"- **Judged.** {meta['started_at']} to {meta['finished_at']}.",
        "",
        "The floor table the skill applies: 55 throwaway / 65 internal tool or "
        "docs / 70 production, user-facing / 75 auth, payments, deletion or a "
        "security path; +5 when the task turns on one specific load-bearing "
        "thing (an API contract, a portability trap, a security invariant, an "
        "exact version comparison).",
        "",
        "| Task | Tier | Floor | Base | Adj | Consequence | Spread |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for task_id, v in tasks.items():
        floors = ", ".join(f"{f:g}" for f in v["floors_seen"])
        tiers = ", ".join(f"{k}x{n}" for k, n in v["tiers_seen"].items())
        spread = (
            "unanimous"
            if v["floor_unanimous"] and v["tier_unanimous"]
            else f"floors {floors}; tiers {tiers}"
        )
        lines.append(
            f"| {task_id} | {v['tier']} | **{v['floor']:g}** "
            f"| {v.get('base_floor') or '--'} | {v.get('adjustment') or 0} "
            f"| {(v.get('consequence') or '').replace('|', '/')} | {spread} |"
        )
    lines += ["", "## Why each floor", ""]
    for task_id, v in tasks.items():
        lines += [
            f"### {task_id} -- floor {v['floor']:g}, {v['tier']}",
            "",
            (v.get("reason") or "(no reason recorded)"),
            "",
        ]
    return "\n".join(lines)


def _select_tasks(dataset: Dataset, task_ids: list[str]) -> list[Task]:
    """Select tasks to judge, preserving dataset order.

    Args:
        dataset: The loaded dataset.
        task_ids: Task ids to keep; empty means all.

    Returns:
        The selected tasks.

    Raises:
        SystemExit: If an id is not in the dataset.
    """
    if not task_ids:
        return list(dataset.tasks)
    wanted = {t.strip() for t in task_ids if t.strip()}
    known = {t.id for t in dataset.tasks}
    unknown = wanted - known
    if unknown:
        raise SystemExit(f"unknown task id(s): {sorted(unknown)}")
    return [t for t in dataset.tasks if t.id in wanted]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the swe-router skill's step 1 headless over a dataset.",
        epilog=(
            "Examples:\n"
            "  uv run scripts/run-swe-router-headless.py --agent omp --provider bedrock \\\n"
            "      --model us.anthropic.claude-opus-5 --repeats 3\n"
            "  uv run scripts/run-swe-router-headless.py --tasks configurable-ui-title "
            "--repeats 1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Runner config YAML path.")
    parser.add_argument(
        "--agent", default=AGENT_OMP, help="Coding agent: omp | claude."
    )
    parser.add_argument("--provider", default="bedrock", help="endpoint | bedrock.")
    parser.add_argument(
        "--endpoint", default=None, help="Base URL for provider=endpoint."
    )
    parser.add_argument(
        "--model",
        default="us.anthropic.claude-opus-5",
        help="Model that does the judging. Default: %(default)s.",
    )
    parser.add_argument(
        "--aws-region", default=None, help="Region for provider=bedrock."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset YAML path.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task ids.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Independent judgments per task. Default: %(default)s.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Judgments to run at once. Default: %(default)s.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Wall-clock cap per judgment. Default: %(default)s.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "metrics" / "swe-router-judged-inputs.json",
        help="Where to write the judged inputs. Default: %(default)s.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Also write the judgments as markdown here. Default: the --out "
        "path with a .md suffix, under docs/.",
    )
    parser.add_argument(
        "--render",
        type=Path,
        default=None,
        help="Render an existing judged-inputs JSON to markdown and exit, "
        "running no agent. Use to regenerate the report after editing prose.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the prompt for one task and exit."
    )
    return parser.parse_args()


def main() -> None:
    """Judge every task in the dataset and write the tuples to JSON."""
    args = _parse_args()
    if args.render:
        payload = json.loads(args.render.read_text(encoding="utf-8"))
        out_md = args.out_md or (REPO_ROOT / "docs" / f"{args.render.stem}.md")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_judgments_markdown(payload), encoding="utf-8")
        logger.info("wrote %s", out_md)
        return
    overrides = {
        "agent": args.agent,
        "provider": args.provider,
        "endpoint": args.endpoint,
        "model": args.model,
        "aws_region": args.aws_region,
        "dataset": args.dataset,
        "timeout_seconds": args.timeout_seconds,
        "agent_max_time_seconds": DEFAULT_AGENT_MAX_TIME_SECONDS,
    }
    try:
        config = load_runner_config(args.config, overrides)
    except RunnerConfigError as exc:
        raise SystemExit(f"invalid runner config: {exc}") from exc
    if config.agent not in (AGENT_OMP, AGENT_CLAUDE):
        raise SystemExit(
            f"--agent {config.agent} is not supported here; use omp or claude"
        )

    dataset_path = Path(config.dataset)
    if not dataset_path.is_absolute():
        dataset_path = BENCHMARKS_DIR / dataset_path
    try:
        dataset = load_dataset(dataset_path)
    except DatasetError as exc:
        raise SystemExit(f"dataset error: {exc}") from exc
    tasks = _select_tasks(dataset, args.tasks.split(",") if args.tasks else [])

    if args.dry_run:
        clone = Path("/tmp/example-clone")  # nosec B108 - illustrative path only
        print(_build_prompt(tasks[0], clone))
        return

    started = _utc_now_iso()
    records, consolidated = _collect(
        config, dataset, tasks, args.repeats, args.concurrency
    )
    costs = [
        r["metrics"]["total_cost_usd"]
        for r in records
        if "error" not in r and r["metrics"].get("total_cost_usd")
    ]
    payload = {
        "judged_by": {
            "harness": config.agent,
            "model": config.model,
            "provider": config.provider,
            "skill": "swe-router",
            "step": "1 and 1b (consequence floor + complexity tier) only; "
            "route.py is run separately by eval_swe_router.py",
            "dataset": str(dataset_path.relative_to(REPO_ROOT)),
            "repeats": args.repeats,
            "started_at": started,
            "finished_at": _utc_now_iso(),
            "judging_cost_usd": round(sum(costs), 4) if costs else None,
            "runs_ok": sum(1 for r in records if "error" not in r),
            "runs_failed": sum(1 for r in records if "error" in r),
        },
        "consolidation": "median floor, modal tier across repeats; per-task "
        "spread kept in floors_seen / tiers_seen",
        "tasks": consolidated,
        "judgments": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out_md = args.out_md or (REPO_ROOT / "docs" / f"{args.out.stem}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_judgments_markdown(payload), encoding="utf-8")

    unstable = [t for t, v in consolidated.items() if not v["floor_unanimous"]]
    logger.info(
        "judged %d/%d task(s); %d run(s) failed; floor unanimous on %d/%d; "
        "judging cost $%s",
        len(consolidated),
        len(tasks),
        payload["judged_by"]["runs_failed"],
        len(consolidated) - len(unstable),
        len(consolidated),
        payload["judged_by"]["judging_cost_usd"],
    )
    if unstable:
        logger.warning("floor disagreed across repeats on: %s", ", ".join(unstable))
    logger.info("wrote %s", args.out)
    logger.info("wrote %s", out_md)


if __name__ == "__main__":
    main()
