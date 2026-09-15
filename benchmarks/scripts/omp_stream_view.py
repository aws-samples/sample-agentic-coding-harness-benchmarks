#!/usr/bin/env python3
"""Render an omp or pi event stream as readable text, live or after the fact.

``run-swe-headless.py`` mirrors omp's JSON-lines events to
``<artifacts_dir>/omp-stream.jsonl`` while a task runs (see ``_run_omp``). That
file is the only way to watch an omp task in flight, but one line per token
makes it unreadable raw -- a sentence arrives as fifty ``text_delta`` events.

This reassembles the stream: prose is printed as continuous text, tool calls are
announced with their arguments, tool results are summarized, and each turn ends
with its token usage. It follows the file by default, so it can be pointed at a
task that is still running.

``--latest`` follows the whole *run*, not one file. A benchmark run walks 21 tasks
per model and then swaps to the next model, writing a fresh stream file each time,
so pinning to one path means re-running this command a hundred-odd times. Instead
it drains the current file, notices when a newer stream appears, prints a banner
naming the new model and task, and carries on -- start it once and leave it up for
the whole run. It also waits rather than exiting if no stream exists yet, so it can
be started before the first task begins.

Usage:
    # Follow the run: every task of every model, hopping automatically
    uv run scripts/omp_stream_view.py --latest

    # A specific task, from the beginning, without following
    uv run scripts/omp_stream_view.py --no-follow \\
        swe-benchmark-data/qwen3.8-27b/omp/swe3/mcp-gateway-registry/remove-faiss/omp-stream.jsonl

    # Only what the model did, not what it said
    uv run scripts/omp_stream_view.py --latest --tools-only

    # Pin to one task even while it is live: pass the path instead of --latest
    uv run scripts/omp_stream_view.py .../omp-stream.jsonl

    # Also works as a filter
    tail -f .../omp-stream.jsonl | uv run scripts/omp_stream_view.py -
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator, TextIO

_SCRIPTS_DIR = Path(__file__).resolve().parent
_BENCHMARKS_DIR = _SCRIPTS_DIR.parent
DEFAULT_DATA_DIR = _BENCHMARKS_DIR / "swe-benchmark-data"
# pi and omp emit the same event stream (omp is a fork), so one viewer
# serves both; --latest searches for either.
STREAM_FILENAMES = ("omp-stream.jsonl", "pi-stream.jsonl")

# How much of a tool's arguments and result to show. Full arguments can be a
# whole file's contents, which would bury the trace it is meant to reveal.
ARGS_PREVIEW_CHARS = 220
RESULT_PREVIEW_CHARS = 400
# Poll interval when following a file that has not grown yet.
FOLLOW_POLL_SECONDS = 0.4
# How often --latest re-checks for a newer stream file. Only ever paid while the
# current file is idle, and a full rescan of the artifact tree is ~35 ms, so this
# costs nothing next to the task it is watching.
RESCAN_SECONDS = 2.0


def _latest_stream(data_dir: Path) -> Path | None:
    """Return the most recently modified omp/pi stream under ``data_dir``.

    Args:
        data_dir: The swe-benchmark-data root to search.

    Returns:
        Path to the newest stream file, or None if none exists yet.
    """
    found = [f for name in STREAM_FILENAMES for f in data_dir.rglob(name)]
    if not found:
        return None

    # stat() can race a file being written; treat an unreadable one as oldest
    # rather than crashing a viewer that is meant to run unattended for hours.
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    return max(found, key=_mtime)


def _stream_label(path: Path, data_dir: Path) -> str:
    """Describe a stream file as ``model/harness/skill/scope :: task``.

    Falls back to the bare path when it does not sit at the expected depth, so an
    older or hand-placed artifact layout still gets a readable banner.
    """
    try:
        parts = path.relative_to(data_dir).parts
    except ValueError:
        return str(path)
    if len(parts) < 2:
        return str(path)
    return f"{'/'.join(parts[:-2])} :: {parts[-2]}"


def _follow(path: Path, data_dir: Path | None = None) -> Iterator[str]:
    """Yield complete lines from ``path``, waiting when it stops growing.

    Only whole lines are yielded. ``readline`` at EOF hands back whatever has been
    flushed so far, which for a live writer is routinely half a line; yielding that
    split a single event into two fragments that both failed to parse and were
    silently dropped by the renderer. Partial reads are buffered until the newline
    arrives instead.

    Args:
        path: The file to follow.
        data_dir: When given, stop once a NEWER stream appears under this root and
            ``path`` has been drained to EOF -- this is what lets --latest hop from
            one task to the next. When None, follow ``path`` forever.

    Yields:
        Each complete line as it is appended.
    """
    with path.open("r", encoding="utf-8") as fh:
        pending = ""
        last_scan = time.monotonic()
        while True:
            chunk = fh.readline()
            if chunk:
                pending += chunk
                if pending.endswith("\n"):
                    yield pending
                    pending = ""
                continue
            # Idle: the file is drained, so this is the only safe point at which to
            # hand over to a newer task -- no event is left half-read behind us.
            if data_dir is not None and time.monotonic() - last_scan >= RESCAN_SECONDS:
                last_scan = time.monotonic()
                newest = _latest_stream(data_dir)
                if newest is not None and newest.resolve() != path.resolve():
                    return
            time.sleep(FOLLOW_POLL_SECONDS)


def _follow_run(data_dir: Path, out: TextIO, tools_only: bool) -> None:
    """Render every stream under ``data_dir`` in turn, newest first, forever.

    Waits for a stream to exist, renders it until a newer one shows up, announces
    the handover, and repeats. Each file gets its own render call so the turn
    counter restarts per task rather than climbing across the whole run.
    """
    current: Path | None = None
    waiting = False
    while True:
        path = _latest_stream(data_dir)
        if path is None:
            if not waiting:
                print(
                    f"# waiting for {' or '.join(STREAM_FILENAMES)} under {data_dir} ...",
                    file=sys.stderr,
                )
                waiting = True
            time.sleep(RESCAN_SECONDS)
            continue
        waiting = False
        if path != current:
            print(f"\n# {_stream_label(path, data_dir)}", file=sys.stderr)
            print(f"# {path}", file=sys.stderr)
            current = path
        _render(_follow(path, data_dir), out, tools_only)


def _preview(value: Any, limit: int) -> str:
    """Collapse a value to a single-line preview of at most ``limit`` chars."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " ..."


def _result_text(result: Any) -> str:
    """Extract the text of a tool result, whatever shape it arrived in."""
    if isinstance(result, dict):
        parts = [
            c.get("text", "")
            for c in result.get("content") or []
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        if parts:
            return "\n".join(parts)
    return _preview(result, RESULT_PREVIEW_CHARS)


def _usage_line(message: dict[str, Any]) -> str | None:
    """Format an assistant message's token usage, or None if it carries none."""
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    total = cost.get("total") if isinstance(cost, dict) else cost
    bits = [
        f"in={usage.get('input', 0):,}",
        f"out={usage.get('output', 0):,}",
        f"cacheRead={usage.get('cacheRead', 0):,}",
    ]
    if isinstance(total, (int, float)) and total:
        bits.append(f"${total:.4f}")
    return "  [" + "  ".join(bits) + "]"


def _render(lines: Iterator[str], out: TextIO, tools_only: bool) -> None:
    """Render an omp event stream to ``out``.

    Text deltas are written without newlines so prose reassembles as it streams;
    every other event is a discrete labelled line.

    Args:
        lines: The raw JSON-lines stream.
        out: Where to write the rendered trace.
        tools_only: Skip the model's prose, showing only tool calls and results.
    """
    turn = 0
    mid_text = False

    def end_text() -> None:
        nonlocal mid_text
        if mid_text:
            out.write("\n")
            mid_text = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")

        if etype == "turn_start":
            turn += 1
            end_text()
            out.write(f"\n{'=' * 70}\n=== turn {turn}\n{'=' * 70}\n")
        elif etype == "message_update":
            ame = event.get("assistantMessageEvent") or {}
            if ame.get("type") == "text_delta" and not tools_only:
                out.write(ame.get("delta") or "")
                mid_text = True
        elif etype == "tool_execution_start":
            end_text()
            out.write(
                f"\n  -> {event.get('toolName')}("
                f"{_preview(event.get('args'), ARGS_PREVIEW_CHARS)})\n"
            )
        elif etype == "tool_execution_end":
            end_text()
            text = _result_text(event.get("result"))
            first = text.splitlines()[0] if text.splitlines() else ""
            extra = len(text.splitlines()) - 1
            suffix = f"  (+{extra} more lines)" if extra > 0 else ""
            out.write(f"  <- {_preview(first, RESULT_PREVIEW_CHARS)}{suffix}\n")
        elif etype == "message_end":
            message = event.get("message") or {}
            if message.get("role") == "assistant":
                end_text()
                usage = _usage_line(message)
                if usage:
                    out.write(usage + "\n")
        elif etype == "agent_end":
            end_text()
            out.write("\n=== agent_end\n")
        out.flush()
    end_text()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render an omp event stream (omp-stream.jsonl) as readable text.",
        epilog="Examples:\n"
        "  uv run scripts/omp_stream_view.py --latest\n"
        "  uv run scripts/omp_stream_view.py --latest --tools-only\n"
        "  uv run scripts/omp_stream_view.py --no-follow path/to/omp-stream.jsonl\n"
        "  tail -f path/to/omp-stream.jsonl | uv run scripts/omp_stream_view.py -",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stream",
        nargs="?",
        help="Path to an omp-stream.jsonl, or '-' to read stdin. "
        "Omit and pass --latest to pick the newest automatically.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Follow the run: start at the most recently modified stream under "
        "--data-dir and hop to each new task and model as they start, so the "
        "command does not need restarting. Waits if no stream exists yet. "
        "With --no-follow, renders just the newest one and exits.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root to search with --latest (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Render what is in the file and exit, instead of following it.",
    )
    parser.add_argument(
        "--tools-only",
        action="store_true",
        help="Show only tool calls and results, skipping the model's prose.",
    )
    return parser.parse_args()


def main() -> None:
    """Resolve the stream to read and render it."""
    args = _parse_args()
    if args.stream == "-":
        _render(iter(sys.stdin), sys.stdout, args.tools_only)
        return

    data_dir = args.data_dir.expanduser().resolve()

    # --latest while following is a whole-run view, not a single file, so it owns
    # its own loop over successive streams.
    if args.latest and not args.stream and not args.no_follow:
        try:
            _follow_run(data_dir, sys.stdout, args.tools_only)
        except KeyboardInterrupt:
            print("\n(stopped)", file=sys.stderr)
        return

    if args.stream:
        path = Path(args.stream).expanduser()
    elif args.latest:
        latest = _latest_stream(data_dir)
        if latest is None:
            raise SystemExit(
                f"no {' or '.join(STREAM_FILENAMES)} under {data_dir} -- "
                "is an omp or pi run in progress?"
            )
        path = latest
    else:
        raise SystemExit("pass a stream path, '-' for stdin, or --latest")
    if not path.is_file():
        raise SystemExit(f"stream not found: {path}")

    print(f"# {path}", file=sys.stderr)
    if args.no_follow:
        with path.open("r", encoding="utf-8") as fh:
            _render(iter(fh), sys.stdout, args.tools_only)
    else:
        try:
            _render(_follow(path), sys.stdout, args.tools_only)
        except KeyboardInterrupt:
            print("\n(stopped)", file=sys.stderr)


if __name__ == "__main__":
    main()
