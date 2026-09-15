#!/usr/bin/env python3
"""Throughput harness: drive N concurrent agentic /swe2 sessions for a fixed window.

This is the SEPARATE, throughput-oriented sibling of ``run-swe-headless.py``. It
answers a different question -- *how much agentic-coding load can this model on
this hardware sustain, and therefore what does a task really cost?* -- so it is
kept apart from the quality harness rather than bolted onto it. It **imports** the
stable building blocks from ``run-swe-headless.py`` (clone / prompt / env / claude
command / run) and only adds the throughput loop on top; the quality harness is
not modified.

How it differs from the quality run:

  * **Concurrency is the point, not a side effect.** It holds ``--concurrency N``
    agentic sessions in flight, refilling a slot as soon as one finishes, for a
    fixed ``--duration-seconds`` window -- so a saturation curve can be built by
    sweeping N.
  * **Each slot picks a DISTINCT task at random.** Tasks are drawn at random
    WITHOUT replacement, cycling: a shuffled ordering of the dataset is consumed
    one task per slot, reshuffling once exhausted. So the N in-flight slots hold N
    *different* tasks whenever the dataset has at least N of them -- never N copies
    of the same task -- and only once every task is already in flight do repeats
    begin, spread as evenly as possible. With a multi-repo dataset that means each
    concurrent slot clones and reasons over a DIFFERENT repo, simulating N
    developers each on their own project. Each running instance still gets a
    unique slot id so its clone dir and (throwaway) artifact dir never collide.
  * **Artifacts are load, not results.** We measure server throughput (via the
    DuckDB metrics collector) and client-side per-request tokens/latency; the
    written artifacts are not scored and their dirs are cleaned up.

The real request shape (large read-heavy prompts, short outputs) comes for free
because each session is a genuine /swe2 run against a real cloned repo -- the same
workload the quality harness produces, just driven at a controlled concurrency.

Usage (normally invoked by run-throughput-sweep.sh, one concurrency per call):
    uv run scripts/run-throughput-harness.py --config config/runner.yaml \\
        --model gemma-4-31b --dataset dataset/mcp-gateway-registry.yaml \\
        --endpoint http://127.0.0.1:8000 --context-window 200000 \\
        --concurrency 5 --duration-seconds 600 --out throughput-c5.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import re
import shutil
import subprocess  # nosec B404 - list-form `git ls-files` only, never shell=True
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Live-heartbeat cadence: how often run_level logs window progress + throughput.
_HEARTBEAT_SECONDS = 30
# vLLM Prometheus counters the heartbeat reads for a live tokens/sec readout.
_GEN_TOKENS_METRIC = "vllm:generation_tokens_total"
_PROMPT_TOKENS_METRIC = "vllm:prompt_tokens_total"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Import the quality harness as a library (its filename has hyphens).
_HARNESS_PATH = _SCRIPTS_DIR / "run-swe-headless.py"
_spec = importlib.util.spec_from_file_location("run_swe_headless", _HARNESS_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - import wiring
    raise ImportError(f"cannot load harness building blocks from {_HARNESS_PATH}")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

from dataset_loader import Dataset, Task, load_dataset  # noqa: E402
from runner_config import RunnerConfig, load_runner_config  # noqa: E402


def _utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string with a trailing Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _scrape_token_counters(endpoint: str | None) -> dict[str, float] | None:
    """Read vLLM's cumulative gen/prompt token counters from ``/metrics``.

    Best-effort and fast: used only to give the heartbeat a live tokens/sec
    readout. Returns None (heartbeat degrades gracefully) if the endpoint is
    unset or unreachable -- authoritative throughput still comes from the DuckDB
    collector, not from here.

    Args:
        endpoint: The vLLM base URL (e.g. ``http://127.0.0.1:8000``).

    Returns:
        ``{"gen": <tokens>, "prompt": <tokens>}`` summed across label sets, or None.
    """
    if not endpoint:
        return None
    url = endpoint.rstrip("/") + "/metrics"
    totals = {"gen": 0.0, "prompt": 0.0}
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:  # nosec B310 - fixed http(s) metrics URL
            body = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    for line in body.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.partition(" ")
        metric = name.split("{", 1)[0]
        try:
            val = float(value)
        except ValueError:
            continue
        if metric == _GEN_TOKENS_METRIC:
            totals["gen"] += val
        elif metric == _PROMPT_TOKENS_METRIC:
            totals["prompt"] += val
    return totals


def _task_cycler(tasks: list[Task]):
    """Yield tasks in a random order WITHOUT replacement, reshuffling each pass.

    Drawing without replacement guarantees the next ``len(tasks)`` picks are all
    distinct, so N concurrent slots never hold duplicate tasks while the dataset
    still has unused ones; once every task has been handed out, a fresh shuffle
    starts the next pass. Load-slot selection only -- ``random`` is fine here.

    Args:
        tasks: The non-empty task pool to cycle through.

    Yields:
        The next ``Task`` to run, forever.
    """
    while True:
        order = list(tasks)
        random.shuffle(order)  # nosec B311 - load-slot ordering, not crypto
        yield from order


_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _safe_path_component(value: str, fallback: str) -> str:
    """Reduce a value to a single safe filename component.

    ``model_slug`` is derived from ``--model`` by ``model_to_slug``, which only
    strips a Bedrock prefix and a bracketed suffix -- it does NOT sanitize path
    separators, so a model id containing ``/`` or ``..`` survives intact. That
    matters here because the slot dir built from it is ``shutil.rmtree``d with
    ``ignore_errors=True`` when the session ends: a slug of ``../../../etc/x``
    would escape ``clone_dir`` and silently delete a tree outside it. Collapsing
    everything outside ``[A-Za-z0-9._-]`` and stripping leading dots means the
    result can only ever name a child of ``clone_dir``.

    Args:
        value: The raw value to use as a path component.
        fallback: Component to return when ``value`` reduces to nothing.

    Returns:
        A single path component safe to join onto a trusted parent directory.
    """
    return _UNSAFE_PATH_CHARS.sub("-", value).lstrip(".") or fallback


def _root_entry_names(root: Path) -> set[str]:
    """Return the names directly under ``root``, or an empty set if unreadable."""
    try:
        return {entry.name for entry in root.iterdir()}
    except OSError:
        return set()


def _is_git_tracked(root: Path, name: str) -> bool:
    """Report whether ``name`` is tracked by the git repo at ``root``.

    Fails CLOSED. Only exit 1 -- git ran, found the repo, and reported the path is
    not in the index -- counts as untracked. Exit 128 (``root`` is not a git repo),
    a missing binary, or a timeout all return True, so a file whose trackedness
    cannot be established is never deleted.

    Args:
        root: The repository root to ask about (also the subprocess cwd).
        name: A single path component directly under ``root``.

    Returns:
        True if git lists the path, or if trackedness could not be determined.
    """
    try:
        proc = subprocess.run(  # nosec B603 B607 - hardcoded 'git', list args, no shell
            ["git", "ls-files", "--error-unmatch", "--", name],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return proc.returncode != 1


def _quarantine_dir(parent: Path) -> Path | None:
    """Create a fresh 0700 directory under ``parent`` to move strays into.

    ``mkdtemp`` rather than a fixed name: ``clone_dir`` defaults to ``/tmp``, so a
    predictable path could be pre-created as a symlink by another local user and
    would then receive the moved files. ``mkdtemp`` creates exclusively with 0700
    or fails.

    Args:
        parent: Directory to create the quarantine dir inside (``clone_dir``).

    Returns:
        The new directory, or None if it could not be created.
    """
    try:
        return Path(tempfile.mkdtemp(prefix="swe-thru-stray-", dir=str(parent)))
    except OSError as exc:
        logger.warning("could not create a quarantine dir under %s: %s", parent, exc)
        return None


def _sweep_stray_root_writes(
    root: Path, before: set[str], since: float, quarantine_parent: Path
) -> list[str]:
    """Move files a load session dropped in the repo root out of the working tree.

    WHY THIS EXISTS. Load sessions run ``claude -p`` with ``cwd=REPO_ROOT``
    (``_run_claude``), which is required: the prompt is the ``/swe2`` slash
    command, resolved from ``.claude/skills/`` relative to cwd, and the root
    ``CLAUDE.md`` is auto-loaded as session context. Both are part of the request
    shape the committed throughput baselines were measured with, so cwd cannot be
    moved to the throwaway slot dir. The cost is that a session which ignores the
    absolute ``artifacts_dir`` it was given and writes a bare relative filename
    writes into the working repo instead. One such file (``github-issue.md``, from
    the ``pytest-flaky-test-detection`` task) survived the 2026-08-31 H200 sweep
    untracked and un-ignored, where a ``git add -A`` would have committed it.

    MOVED, NOT DELETED. Throughput artifacts are load, not results (the slot dir
    is already ``rmtree``d), so these files have no value -- but this runs against
    the user's own working tree, where the cost of a wrong call is someone's
    unsaved work. So each stray is moved into a throwaway quarantine dir under
    ``clone_dir`` instead of unlinked: the repo comes out clean either way, and a
    misattributed file is recoverable rather than gone.

    Four guards on top of that:
      * name came from ``iterdir`` of ``root``, so it is a single component and
        cannot traverse; the resolved parent is re-checked against ``root``.
      * only plain files, never directories or symlinks (a symlink could point
        outside the repo, and a new directory is reported instead of moved).
      * only files modified at or after the level's start, so a file that merely
        became visible is left alone.
      * only files git positively reports as untracked; an unverifiable answer
        (``root`` is not a git repo, git missing, timeout) leaves the file alone.

    A file replaced by a symlink between the check and the move is harmless:
    ``shutil.move`` on a symlink moves the link, not its target.

    Args:
        root: The repository root the sessions ran in.
        before: Entry names present in ``root`` when the level started.
        since: ``time.time()`` value marking the start of the level window.
        quarantine_parent: Directory to create the quarantine dir under, on the
            first stray found (nothing is created when the root stays clean).

    Returns:
        The names moved out of ``root``, for the level summary.
    """
    moved: list[str] = []
    quarantine: Path | None = None
    for name in sorted(_root_entry_names(root) - before):
        path = root / name
        try:
            if path.is_symlink() or not path.is_file():
                logger.warning(
                    "  stray repo-root entry %r appeared during this level and was "
                    "LEFT IN PLACE (not a plain file) -- inspect and clean up by hand",
                    name,
                )
                continue
            stat = path.stat()
            if path.resolve().parent != root.resolve():
                continue
            if stat.st_mtime < since:
                continue
        except OSError:
            continue
        if _is_git_tracked(root, name):
            logger.warning(
                "  repo-root file %r appeared during this level but is git-tracked "
                "(or trackedness is unknown) -- leaving it in place",
                name,
            )
            continue
        if quarantine is None:
            quarantine = _quarantine_dir(quarantine_parent)
            if quarantine is None:
                logger.warning(
                    "  leaving stray repo-root file %r in place: no quarantine dir",
                    name,
                )
                continue
        try:
            shutil.move(str(path), str(quarantine / name))
        except (OSError, shutil.Error) as exc:
            logger.warning("  could not move stray repo-root file %r: %s", name, exc)
            continue
        moved.append(name)
        logger.warning(
            "  moved stray repo-root write %r (%s bytes) to %s: a load session wrote "
            "a bare relative path instead of its artifacts_dir",
            name,
            stat.st_size,
            quarantine,
        )
    return moved


def _run_one_session(
    config: RunnerConfig,
    task: Task,
    ref: str,
    slot_label: str,
    deadline: float,
) -> dict[str, Any]:
    """Run one agentic /swe2 session for load and return its per-request record.

    Throughput is measured server-side (vLLM counters in DuckDB), so a session
    does NOT need to finish to count -- the tokens it generated while running are
    already in the collector's window. This session's ``claude -p`` timeout is
    therefore bounded by the remaining window (``deadline``): at window close,
    any still-running session self-terminates promptly via the existing timeout
    rather than dragging the level out by ~30 min waiting for a full agentic run.
    Such a session is recorded as ``cutoff`` (not a failure) -- it consumed real
    serving capacity for the whole window.

    Clones into a dir unique per model AND slot, so neither repeated instances of
    the same task nor concurrent sweeps of different models on one host ever
    collide, and always cleans up the clone.

    Args:
        config: The runner config (endpoint, model, timeout, ...).
        task: The dataset task to run this session on.
        ref: The git ref to clone.
        slot_label: A unique label for this in-flight instance (e.g. ``c5#12``).
        deadline: ``time.time()`` value after which the session is cut off.

    Returns:
        A record with tokens, latency, turns, and ok/cutoff/error for this session.
    """
    started = time.time()
    started_iso = _utc_now()
    # Scope the slot dir by MODEL as well as slot: several sweeps can run
    # concurrently on one host (one per GPU, each on its own port), and slot
    # labels restart at "c{N}#1" for every level, so two arms sweeping the same
    # concurrency would claim the same dir -- and the finally block below
    # rmtree's it, which would delete a sibling arm's live clone mid-session.
    slot_dir = Path(config.clone_dir) / (
        f"swe-thru-{_safe_path_component(config.model_slug, 'model')}"
        f"-{slot_label.replace('#', '-')}"
    )
    slot_dir.mkdir(parents=True, exist_ok=True)
    # Bound this session by whichever is smaller: the config timeout or the time
    # left in the window. min 1s so a just-past-deadline submit still terminates.
    session_timeout = max(1, int(min(config.timeout_seconds, deadline - started)))

    def _record(status: str, result: dict[str, Any] | None, error: str = "") -> dict:
        usage = (result or {}).get("usage") or {}
        elapsed = time.time() - started
        rec = {
            "slot": slot_label,
            "task": task.id,
            "status": status,  # "ok" | "cutoff" | "error"
            "ok": status == "ok",
            "started_at": started_iso,
            "ended_at": _utc_now(),
            "latency_seconds": round(elapsed, 1),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "num_turns": (result or {}).get("num_turns", 0),
        }
        if error:
            rec["error"] = error[:300]
        return rec

    try:
        clone_path = harness._clone_repo(
            task, ref, str(slot_dir), log_prefix=slot_label
        )
        # Artifacts go under this session's throwaway slot dir, NOT the real
        # swe-benchmark-data tree: throughput does not score artifacts, and many
        # cut-off sessions writing there would clobber the model's quality-run
        # artifacts. The dir is removed with the slot in the finally block.
        prompt = harness._build_prompt(
            task, clone_path, ref, config.model_slug, slot_dir / "artifacts"
        )
        cmd = harness._build_claude_cmd(
            config, prompt, stream=False, clone_path=clone_path
        )
        env = harness._build_env(config)
        result = harness._run_claude(cmd, env, session_timeout)
        return _record("ok" if not result.get("is_error", False) else "error", result)
    except Exception as exc:
        # A window-bounded timeout is an expected cutoff, not a failure; anything
        # else (clone error, etc.) is a real error but must not kill the sweep.
        msg = str(exc)
        if "timed out" in msg.lower():
            return _record("cutoff", None, "cut off at window close")
        logger.warning("%s failed: %s", slot_label, msg[:200])
        return _record("error", None, msg)
    finally:
        shutil.rmtree(slot_dir, ignore_errors=True)


def run_level(
    config: RunnerConfig,
    dataset: Dataset,
    tasks: list[Task],
    concurrency: int,
    duration_seconds: int,
) -> dict[str, Any]:
    """Hold ``concurrency`` agentic sessions in flight for ``duration_seconds``.

    A thread pool of width ``concurrency`` is kept saturated: each time a session
    finishes, the next task from a shuffled cycle of ``tasks`` (random WITHOUT
    replacement, reshuffled each pass) is submitted, until the wall-clock window
    elapses. Drawing without replacement means the N in-flight slots hold N
    distinct tasks whenever the dataset has at least N -- never N copies of one
    task -- so with a multi-repo dataset the slots spread across different repos,
    simulating many developers on different projects rather than one shared
    repo. Sessions still running at window close are **cut
    off** (their ``claude -p`` timeout is bounded by the remaining window) rather
    than drained to completion -- because throughput is measured server-side from
    vLLM's counters over the level's time window, a session need not finish to
    have contributed the tokens it generated. This keeps every level ~= the
    window, even for a slow model whose agentic sessions take far longer.

    ``level_started_at`` / ``level_ended_at`` bound the window so the performance
    summary can slice the DuckDB collector session to exactly this level.

    Once every session has exited, files a session leaked into the repo root are
    swept (see ``_sweep_stray_root_writes``) and listed under
    ``stray_root_writes``.

    Args:
        config: The runner config.
        dataset: The loaded dataset (for ref resolution).
        tasks: The tasks to cycle through as load.
        concurrency: How many sessions to hold in flight.
        duration_seconds: Wall-clock window to keep submitting new sessions.

    Returns:
        A level summary: config, wall-clock window bounds, and per-session records.
    """
    refs = {t.id: dataset.resolved_ref(t) for t in tasks}
    task_cycle = _task_cycler(tasks)
    records: list[dict[str, Any]] = []
    lock = threading.Lock()
    submitted = 0
    wall_start = time.time()
    deadline = wall_start + duration_seconds
    level_started = _utc_now()
    # Snapshot the repo root so writes a session leaks there (see
    # _sweep_stray_root_writes) can be told apart from what was already present.
    repo_root = Path(harness.REPO_ROOT)
    root_before = _root_entry_names(repo_root)

    logger.info(
        "=== concurrency=%s: holding %s sessions in flight for %ss ===",
        concurrency,
        concurrency,
        duration_seconds,
    )
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: set[Any] = set()
        in_flight: dict[Any, str] = {}  # future -> "slot task" for the heartbeat

        def _submit() -> None:
            nonlocal submitted
            # Random draw WITHOUT replacement (shuffled cycle) so a multi-repo
            # dataset spreads the in-flight slots across DISTINCT repos and never
            # runs N copies of one task while others sit idle; a single-repo
            # dataset is unaffected (its lone task is picked every time).
            task = next(task_cycle)
            slot = f"c{concurrency}#{submitted + 1}"
            fut = executor.submit(
                _run_one_session, config, task, refs[task.id], slot, deadline
            )
            futures.add(fut)
            in_flight[fut] = f"{slot} {task.id}"
            submitted += 1

        # Heartbeat state: emit a live progress line every _HEARTBEAT_SECONDS so
        # the log is not silent during the long window (at low concurrency no
        # session finishes until cutoff). Live tokens/sec is derived from the
        # vLLM counter delta since the last beat -- a preview of the DuckDB
        # collector's authoritative figure, not a replacement for it.
        last_beat = wall_start
        beat_counters = _scrape_token_counters(config.endpoint)

        def _heartbeat() -> None:
            nonlocal last_beat, beat_counters
            now = time.time()
            interval = now - last_beat
            counters = _scrape_token_counters(config.endpoint)
            rate = ""
            if counters and beat_counters and interval > 0:
                gen_tps = (counters["gen"] - beat_counters["gen"]) / interval
                prompt_tps = (counters["prompt"] - beat_counters["prompt"]) / interval
                rate = f" | server ~{gen_tps:.0f} gen tok/s, ~{prompt_tps:.0f} prompt tok/s"
            by_status_now: dict[str, int] = {}
            for r in records:
                by_status_now[r["status"]] = by_status_now.get(r["status"], 0) + 1
            active = sorted(in_flight.values())
            logger.info(
                "  [c%s heartbeat] %.0fs/%ss elapsed | in-flight=%s %s | done=%s %s%s",
                concurrency,
                min(now - wall_start, duration_seconds),
                duration_seconds,
                len(futures),
                active,
                len(records),
                by_status_now or "{}",
                rate,
            )
            last_beat, beat_counters = now, counters

        for _ in range(concurrency):
            _submit()

        # Refill finished slots only while inside the window. After the window
        # closes, remaining in-flight sessions self-terminate at the deadline
        # (their timeout was bounded to it), so this loop drains in seconds.
        while futures:
            done = {f for f in futures if f.done()}
            for fut in done:
                futures.discard(fut)
                in_flight.pop(fut, None)
                rec = fut.result()
                with lock:
                    records.append(rec)
                logger.info(
                    "  %s %s (%s) out=%s in %.0fs | done=%s in-flight=%s",
                    rec["status"],
                    rec["slot"],
                    rec["task"],
                    rec["output_tokens"],
                    rec["latency_seconds"],
                    len(records),
                    len(futures),
                )
                if time.time() < deadline:
                    _submit()
            if time.time() - last_beat >= _HEARTBEAT_SECONDS:
                _heartbeat()
            if not done:
                time.sleep(0.5)

    wall_seconds = round(time.time() - wall_start, 1)
    # Every session has exited by here, so nothing this harness owns is still
    # writing to the repo root.
    strays = _sweep_stray_root_writes(
        repo_root, root_before, wall_start, Path(config.clone_dir)
    )
    by_status: dict[str, int] = {}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    return {
        "concurrency": concurrency,
        "duration_seconds": duration_seconds,
        "wall_seconds": wall_seconds,
        # Window bounds for slicing the DuckDB collector session to this level.
        "level_started_at": level_started,
        "level_ended_at": _utc_now(),
        "sessions_started": submitted,
        "sessions_by_status": by_status,
        # Throughput is read from the vLLM DuckDB counters over this window, NOT
        # from these client-side token counts (which only cover sessions that
        # actually completed within the window). Kept for context.
        "client_completed_output_tokens": sum(
            r["output_tokens"] for r in records if r["status"] == "ok"
        ),
        # Files a session wrote to the repo root instead of its artifacts_dir, and
        # which this level moved out to quarantine. Recorded so the leak is visible
        # in the level JSON rather than only in the log.
        "stray_root_writes": strays,
        "sessions": records,
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Drive N concurrent /swe2 sessions for a fixed window (throughput).",
    )
    parser.add_argument("--config", help="Runner config YAML path")
    parser.add_argument("--model", help="Served model name / id")
    parser.add_argument("--endpoint", help="OpenAI/Anthropic-compatible base URL")
    parser.add_argument("--dataset", help="Dataset YAML path")
    parser.add_argument("--context-window", type=int, dest="context_window")
    parser.add_argument("--timeout-seconds", type=int, dest="timeout_seconds")
    parser.add_argument(
        "--concurrency", type=int, required=True, help="Sessions to hold in flight"
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=600,
        help="Wall-clock window to keep submitting new sessions (default 600)",
    )
    parser.add_argument(
        "--tasks", help="Comma-separated task ids to cycle (default: all in dataset)"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Write the level summary JSON here"
    )
    return parser.parse_args()


def main() -> None:
    """Run one concurrency level and write its summary JSON."""
    args = _parse_args()
    overrides: dict[str, Any] = {
        "provider": "endpoint",
        "endpoint": args.endpoint,
        "model": args.model,
        "dataset": args.dataset,
        "context_window": args.context_window,
        "timeout_seconds": args.timeout_seconds,
    }
    config = load_runner_config(args.config, {k: v for k, v in overrides.items() if v})
    dataset = load_dataset(config.dataset)
    tasks = list(dataset.tasks)
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        tasks = [t for t in tasks if t.id in wanted]
    if not tasks:
        raise SystemExit("no tasks selected to drive load")

    summary = run_level(config, dataset, tasks, args.concurrency, args.duration_seconds)
    summary["model"] = config.model
    summary["endpoint"] = config.endpoint
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    logger.info(
        "wrote %s: c=%s started=%s by_status=%s wall=%ss "
        "(throughput read from DuckDB over %s..%s)",
        args.out,
        summary["concurrency"],
        summary["sessions_started"],
        summary["sessions_by_status"],
        summary["wall_seconds"],
        summary["level_started_at"],
        summary["level_ended_at"],
    )


if __name__ == "__main__":
    main()
