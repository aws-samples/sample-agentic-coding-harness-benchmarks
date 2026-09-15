#!/usr/bin/env python3
"""Evaluate one folder of SWE design artifacts with an agentic ``codex exec`` judge.

The direct-API sibling (``llm_as_judge.py``) embeds the five artifacts in a
single stateless Bedrock request and scores them in isolation. This judge runs
``codex exec`` non-interactively instead, so the model can additionally open the
candidate's repository (read-only) with its own file tools and verify the
factual claims in ``lld.md``/``testing.md`` -- and the ``patch.diff``
implementation -- against the real source before scoring. The judge prompt, the strict schema, and the score validation are
shared with ``llm_as_judge.py`` via ``judge_common.py`` so the two backends stay
comparable.

Repo grounding is the default. The artifact folder's ``metrics.json`` supplies
the ``repo`` URL and ``ref`` the artifacts were generated against; this judge
clones that repository at that ref into a reusable checkout and points codex at
it. A missing ``metrics.json`` (or a missing ``repo``/``ref``) fails loudly.

The flow:
  1. Render ``judge_prompt.txt`` with the five artifacts (shared code).
  2. Clone ``repo`` at ``ref`` from ``metrics.json`` (reusing an existing
     checkout), or use an explicit local ``--repo`` path when given.
  3. Run ``codex exec`` with that repository as a read-only working root.
  4. Validate the model's final message against the shared Pydantic schema.
  5. Atomically write ``eval.json`` and mirror it into ``metrics.json``.

The wrapper writes and validates the output; codex never writes ``eval.json``
itself, so the arithmetic and identifier guarantees match the direct path.

Batch mode (``--recursive``) points the judge at a top-level directory instead
of a single artifact folder: it walks that directory recursively, treats every
subdirectory that contains a ``metrics.json`` as one artifact folder to score,
and judges each in turn. A folder whose scoring fails is logged and skipped so
one bad folder never aborts the batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess  # nosec B404 - used with list args, no shell, hardcoded 'codex'/'git'
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse the prompt rendering, strict schema, score validation, and atomic write
# from the shared judge core so both backends score identically.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from judge_common import (  # noqa: E402
    DEFAULT_TEMPLATE_PATH,
    EvaluationResult,
    JudgeError,
    atomic_write_json,
    identify_folder,
    missing_artifacts,
    optional_file,
    parse_and_validate_result,
    render_judge_prompt,
    zero_score_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
DEFAULT_GIT_BIN = os.environ.get("GIT_BIN", "git")
DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "openai.gpt-5.6-sol")
DEFAULT_REASONING_EFFORT = os.environ.get("JUDGE_REASONING_EFFORT", "high")
DEFAULT_SANDBOX = "read-only"
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_CLONE_TIMEOUT_SECONDS = 600
# Root under which candidate repositories are cloned for repo-grounded scoring.
# Each (repo, ref) pair clones once into a stable, content-addressed subdirectory
# so repeated judge runs reuse the same checkout instead of re-cloning.
DEFAULT_CLONE_ROOT = Path(os.environ.get("JUDGE_CLONE_ROOT", "/tmp/swe-judge-repos"))  # nosec B108 - reused checkout cache, not sensitive
# Prepended to the shared judge prompt so codex knows it may ground its scoring
# in the repository at its working root. Read-only is enforced by the sandbox;
# this only tells the model the capability exists (the template already refers
# to "repository evidence available through explicitly provided read-only tools").
_REPO_PREAMBLE = (
    "You are running as a non-interactive agent with read-only access to the "
    "candidate's repository at your current working directory. Use your file "
    "tools to inspect that repository and verify the factual claims in the "
    "artifacts (paths, symbols, APIs, commands) before scoring. Do not modify "
    "any file. When an implementation artifact (patch.diff) is present, ground "
    "your implementation score in the real source: check that the diff's target "
    "files, line context, symbols, and surrounding code actually exist and match "
    "the repository, that the change is correct and consistent with the design, "
    "and that it follows the repository's own conventions. You may dry-run "
    "`git apply --check` against the working tree to confirm the patch applies, "
    "but do not leave any modification behind. Your final message must be the "
    "single strict JSON object the instructions below require, with no "
    "surrounding prose.\n\n"
)


def _build_codex_cmd(
    *,
    codex_bin: str,
    working_root: Path,
    output_file: Path,
    model: str | None,
    reasoning_effort: str | None,
    sandbox: str,
    output_schema_file: Path | None,
) -> list[str]:
    """Assemble the ``codex exec`` argument vector.

    The prompt is fed on stdin (``-``) so large embedded artifacts never hit the
    shell argument-length limit.

    Args:
        codex_bin: Path to the codex executable.
        working_root: Directory codex uses as its read-only working root.
        output_file: File codex writes its final message to (``-o``).
        model: Optional model id (``-m``); None uses the codex config default.
        reasoning_effort: Optional reasoning effort override.
        sandbox: Sandbox policy (default ``read-only``).
        output_schema_file: Optional JSON Schema file constraining the reply.

    Returns:
        The command argument vector for subprocess.
    """
    cmd = [
        codex_bin,
        "exec",
        "--json",  # Stream token_count / task_complete events to stdout as JSONL.
        "--cd",
        str(working_root),
        "--sandbox",
        sandbox,
        "--skip-git-repo-check",
        "--output-last-message",
        str(output_file),
    ]
    if model:
        cmd += ["--model", model]
    if reasoning_effort:
        # Value is parsed as TOML; a bareword like "high" is used as a literal.
        cmd += ["-c", f"model_reasoning_effort={reasoning_effort}"]
    if output_schema_file is not None:
        cmd += ["--output-schema", str(output_schema_file)]
    cmd.append("-")  # Read the prompt from stdin.
    return cmd


def _parse_codex_events(stdout: str) -> dict[str, Any]:
    """Extract token-usage metrics from codex ``--json`` stdout.

    ``codex exec --json`` streams one JSON object per line to stdout. The final
    ``turn.completed`` event carries a ``usage`` block with input/output/cached
    token counts; that is the last usage record seen, so it wins. The older
    ``token_count`` event shape (used by the rollout log) is also accepted for
    forward/backward compatibility. Missing or malformed events yield an empty
    dict rather than an error, so metrics are always best-effort and never fail
    the evaluation.

    Args:
        stdout: The full stdout captured from a ``codex exec --json`` run.

    Returns:
        A metrics dict with ``token_usage`` (and ``context_window`` when the
        rollout-style event is present); empty when nothing was parsed.
    """
    metrics: dict[str, Any] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload", event)
        if not isinstance(payload, dict):
            continue
        event_type = payload.get("type")
        if event_type == "turn.completed":
            usage = payload.get("usage")
            if isinstance(usage, dict):
                metrics["token_usage"] = usage
        elif event_type == "token_count":
            info = payload.get("info")
            if isinstance(info, dict):
                usage = info.get("total_token_usage")
                if isinstance(usage, dict):
                    metrics["token_usage"] = usage
                window = info.get("model_context_window")
                if isinstance(window, int):
                    metrics["context_window"] = window
    return metrics


def _run_codex(
    prompt: str,
    *,
    codex_bin: str,
    working_root: Path,
    model: str | None,
    reasoning_effort: str | None,
    sandbox: str,
    timeout_seconds: int,
    output_schema_file: Path | None,
) -> tuple[str, dict[str, Any]]:
    """Run ``codex exec`` once and return its final message and run metrics.

    Args:
        prompt: The fully rendered judge prompt (fed on stdin).
        codex_bin: Path to the codex executable.
        working_root: Read-only working root for repository grounding.
        model: Optional model id override.
        reasoning_effort: Optional reasoning effort override.
        sandbox: Sandbox policy.
        timeout_seconds: Wall-clock timeout for the codex run.
        output_schema_file: Optional JSON Schema file constraining the reply.

    Returns:
        A tuple of the final agent message text (expected to be the evaluation
        JSON) and a best-effort metrics dict from the streamed ``--json`` events.

    Raises:
        JudgeError: If codex is missing, times out, fails, or emits no message.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", delete=False
    ) as handle:
        output_file = Path(handle.name)
    cmd = _build_codex_cmd(
        codex_bin=codex_bin,
        working_root=working_root,
        output_file=output_file,
        model=model,
        reasoning_effort=reasoning_effort,
        sandbox=sandbox,
        output_schema_file=output_schema_file,
    )
    logger.info("running: %s", " ".join(cmd))
    start = time.monotonic()
    try:
        proc = subprocess.run(  # nosec B603 - hardcoded 'codex', list args, no shell
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        output_file.unlink(missing_ok=True)
        raise JudgeError(
            f"codex executable not found: {codex_bin}. Install codex or set CODEX_BIN."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        output_file.unlink(missing_ok=True)
        raise JudgeError(f"codex exec timed out after {timeout_seconds}s") from exc
    duration_ms = round((time.monotonic() - start) * 1000)

    try:
        message = output_file.read_text(encoding="utf-8").strip()
    except OSError:
        message = ""
    finally:
        output_file.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise JudgeError(
            f"codex exec exited {proc.returncode}: {proc.stderr.strip()[:1000]}"
        )
    if not message:
        raise JudgeError(
            "codex exec produced no final message: "
            f"{proc.stderr.strip()[:1000] or proc.stdout.strip()[:1000]}"
        )
    run_metrics = _parse_codex_events(proc.stdout)
    run_metrics["duration_ms"] = duration_ms
    return message, run_metrics


def _clone_dir(repo: str, ref: str, clone_root: Path) -> Path:
    """Return the stable, content-addressed checkout directory for (repo, ref)."""
    digest = hashlib.sha256(f"{repo}@{ref}".encode()).hexdigest()[:16]
    slug = repo.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git") or "repo"
    return clone_root / f"{slug}-{digest}"


def _run_git(args: list[str], *, git_bin: str, timeout_seconds: int) -> None:
    """Run one git command, raising JudgeError on failure or timeout."""
    cmd = [git_bin, *args]
    logger.info("running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(  # nosec B603 - hardcoded 'git', list args, no shell
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise JudgeError(
            f"git executable not found: {git_bin}. Install git or set GIT_BIN."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise JudgeError(
            f"git timed out after {timeout_seconds}s: {' '.join(cmd)}"
        ) from exc
    if proc.returncode != 0:
        raise JudgeError(
            f"git command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"{proc.stderr.strip()[:1000]}"
        )


def clone_repo_at_ref(
    repo: str,
    ref: str,
    *,
    clone_root: Path = DEFAULT_CLONE_ROOT,
    git_bin: str = DEFAULT_GIT_BIN,
    timeout_seconds: int = DEFAULT_CLONE_TIMEOUT_SECONDS,
) -> Path:
    """Clone ``repo`` at ``ref`` into a reusable checkout under ``clone_root``.

    The checkout is content-addressed by ``(repo, ref)`` so repeated judge runs
    reuse an existing clone. A pre-existing directory that already resolves to
    ``ref`` is reused as-is; a partial or mismatched directory is removed and
    re-cloned so the judge always grounds against a clean, correct source tree.

    Args:
        repo: Git remote URL (e.g. ``https://github.com/owner/name``).
        ref: Branch, tag, or commit the artifacts were generated against.
        clone_root: Parent directory for all judge checkouts.
        git_bin: Path to the git executable.
        timeout_seconds: Per-git-command wall-clock timeout.

    Returns:
        The absolute path to the checked-out repository.

    Raises:
        JudgeError: If cloning or checkout fails.
    """
    if not repo or not repo.strip():
        raise JudgeError("cannot clone: repo URL is empty")
    if not ref or not ref.strip():
        raise JudgeError("cannot clone: ref is empty")

    clone_root.mkdir(parents=True, exist_ok=True)
    target = _clone_dir(repo, ref, clone_root)

    if (target / ".git").is_dir():
        head = subprocess.run(  # nosec B603 - hardcoded 'git', list args, no shell
            [git_bin, "-C", str(target), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if head.returncode == 0 and head.stdout.strip():
            logger.info("reusing existing checkout at %s", target)
            return target
        logger.warning("removing incomplete checkout at %s", target)
        shutil.rmtree(target, ignore_errors=True)

    logger.info("cloning %s@%s into %s", repo, ref, target)
    _run_git(
        ["clone", "--quiet", repo, str(target)],
        git_bin=git_bin,
        timeout_seconds=timeout_seconds,
    )
    _run_git(
        ["-C", str(target), "checkout", "--quiet", ref],
        git_bin=git_bin,
        timeout_seconds=timeout_seconds,
    )
    return target


METRICS_FILENAME = "metrics.json"


def _discover_artifact_folders(root: str | Path) -> list[Path]:
    """Recursively find every artifact folder under ``root``.

    An artifact folder is any directory that directly contains a
    ``metrics.json`` file; that file marks the directory as holding a set of
    SWE artifacts the judge knows how to score. The walk starts at ``root``
    itself (so a single artifact folder passed directly is also discovered) and
    descends into every subdirectory.

    Args:
        root: Top-level directory to search.

    Returns:
        Sorted list of absolute artifact-folder paths (deterministic order).

    Raises:
        JudgeError: If ``root`` is not an existing directory.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise JudgeError(f"not a directory: {root_path}")
    folders = {
        metrics_file.parent
        for metrics_file in root_path.rglob(METRICS_FILENAME)
        if metrics_file.is_file()
    }
    return sorted(folders)


def _resolve_repo_ref(metrics: dict[str, Any] | None) -> tuple[str, str]:
    """Extract the required ``repo`` and ``ref`` from metrics, failing loudly."""
    if not metrics:
        raise JudgeError(
            "metrics.json is required for repo-grounded scoring but was not found "
            "in the artifact folder. It must supply the 'repo' URL and 'ref' the "
            "artifacts were generated against, or pass --repo to use a local clone."
        )
    repo = metrics.get("repo")
    ref = metrics.get("ref")
    if not isinstance(repo, str) or not repo.strip():
        raise JudgeError("metrics.json is missing a non-empty 'repo' URL")
    if not isinstance(ref, str) or not ref.strip():
        raise JudgeError("metrics.json is missing a non-empty 'ref'")
    return repo.strip(), ref.strip()


def evaluate_artifact_folder_with_codex(
    folder: str | Path,
    *,
    repo: str | Path | None = None,
    model: str | None = DEFAULT_MODEL,
    codex_bin: str = DEFAULT_CODEX_BIN,
    git_bin: str = DEFAULT_GIT_BIN,
    clone_root: Path = DEFAULT_CLONE_ROOT,
    clone_timeout_seconds: int = DEFAULT_CLONE_TIMEOUT_SECONDS,
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
    task_context: str | None = None,
    repository_context: str | None = None,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    sandbox: str = DEFAULT_SANDBOX,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    use_output_schema: bool = False,
    overwrite: bool = True,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Evaluate one artifact folder with a single agentic ``codex exec`` run.

    Repo grounding is the default. The artifact folder must contain a
    ``metrics.json`` giving the ``repo`` URL and ``ref`` the artifacts were
    generated against; this judge clones that repository at that ref into
    ``clone_root`` (reusing an existing checkout when present) and runs codex
    with the clone as its read-only working root, so the model can verify the
    factual claims in the artifacts against the real source. Passing an explicit
    local ``repo`` path overrides the clone and uses that checkout directly.

    Args:
        folder: Directory containing the four required Markdown artifacts and
            ``metrics.json``.
        repo: Optional local repository checkout to use as-is. When None (the
            default), the repository is cloned from ``metrics.json``'s ``repo``
            at ``ref``.
        model: Codex model id (default ``openai.gpt-5.6-sol``); pass None to use
            the codex config default.
        codex_bin: Path to the codex executable.
        git_bin: Path to the git executable used for cloning.
        clone_root: Parent directory for judge repository checkouts.
        clone_timeout_seconds: Per-git-command wall-clock timeout.
        template_path: Judge prompt template path (shared with the direct judge).
        task_context: Optional independent task requirements.
        repository_context: Optional independent repository evidence.
        reasoning_effort: Reasoning effort override (default ``high``); pass None
            to use the codex config default.
        sandbox: Sandbox policy for codex (default ``read-only``).
        timeout_seconds: Wall-clock timeout for the codex run.
        use_output_schema: Constrain codex output with the shared JSON Schema.
        overwrite: Allow replacing an existing ``eval.json``.
        write_outputs: Write output files when true.

    Returns:
        The validated evaluation with attached judge metadata.

    Raises:
        JudgeError: On invalid inputs, a missing ``metrics.json`` when cloning,
            a failed clone or codex run, or invalid scores.
    """
    if timeout_seconds < 1:
        raise JudgeError("timeout_seconds must be positive")

    artifact_dir = Path(folder).expanduser().resolve()
    eval_path = artifact_dir / "eval.json"
    if eval_path.exists() and not overwrite:
        raise JudgeError(f"eval.json exists and overwrite is disabled: {eval_path}")

    # A folder missing (or with empty) required artifacts is a genuine model
    # failure, not a judging error: score it 0 and record why, instead of
    # cloning the repo and calling codex only to fail. This keeps the failed
    # task in the results with an explicit MODEL FAILURE verdict.
    missing = missing_artifacts(artifact_dir)
    if missing:
        task_id, candidate_id = identify_folder(artifact_dir)
        logger.warning(
            "%s: missing artifact(s) %s -- scoring 0 (model failure)",
            artifact_dir,
            ", ".join(missing),
        )
        result = zero_score_result(
            task_id=task_id, candidate_id=candidate_id, missing=missing
        )
        result["judge"] = {
            "model": model or "codex-config-default",
            "provider": "codex-exec",
            "repo_grounded": False,
            "scored_zero_missing_artifacts": missing,
            "evaluated_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        if write_outputs:
            atomic_write_json(eval_path, result)
            metrics_path = artifact_dir / "metrics.json"
            if metrics_path.exists():
                existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                existing["evaluation"] = result
                atomic_write_json(metrics_path, existing)
        return result

    prompt, task_id, candidate_id, metrics = render_judge_prompt(
        artifact_dir,
        template_path=template_path,
        task_context=task_context,
        repository_context=repository_context,
    )

    if repo is not None:
        working_root = Path(repo).expanduser().resolve()
        if not working_root.is_dir():
            raise JudgeError(f"repo is not a directory: {working_root}")
        repo_ref: str | None = None
    else:
        repo_url, repo_ref = _resolve_repo_ref(metrics)
        working_root = clone_repo_at_ref(
            repo_url,
            repo_ref,
            clone_root=clone_root,
            git_bin=git_bin,
            timeout_seconds=clone_timeout_seconds,
        )

    prompt = _REPO_PREAMBLE + prompt

    schema_file: Path | None = None
    try:
        if use_output_schema:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".schema.json", delete=False
            ) as handle:
                json.dump(EvaluationResult.model_json_schema(), handle)
                schema_file = Path(handle.name)
        message, run_metrics = _run_codex(
            prompt,
            codex_bin=codex_bin,
            working_root=working_root,
            model=model,
            reasoning_effort=reasoning_effort,
            sandbox=sandbox,
            timeout_seconds=timeout_seconds,
            output_schema_file=schema_file,
        )
    finally:
        if schema_file is not None:
            schema_file.unlink(missing_ok=True)

    result = parse_and_validate_result(
        message, task_id=task_id, candidate_id=candidate_id
    )
    judge: dict[str, Any] = {
        "model": model or "codex-config-default",
        "provider": "codex-exec",
        "repo_grounded": True,
        # NOTE: the local working checkout path (working_root, e.g. /tmp/swe-judge-
        # repos/...) is deliberately NOT recorded -- eval.json and metrics.json are
        # committed to git, and a machine-specific /tmp path is noise, not
        # provenance. repo_ref below captures the meaningful grounding.
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if repo_ref is not None:
        judge["repo_ref"] = repo_ref
    if reasoning_effort is not None:
        judge["reasoning_effort"] = reasoning_effort
    judge.update(run_metrics)
    result["judge"] = judge

    if write_outputs:
        atomic_write_json(eval_path, result)
        if metrics is not None:
            metrics["evaluation"] = result
            atomic_write_json(artifact_dir / "metrics.json", metrics)
    return result


def evaluate_tree_with_codex(
    root: str | Path,
    *,
    overwrite: bool = True,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Judge every artifact folder found recursively under ``root``.

    Walks ``root`` with :func:`_discover_artifact_folders` and scores each
    directory that contains a ``metrics.json`` by delegating to
    :func:`evaluate_artifact_folder_with_codex`. A folder that fails to score
    (missing repo/ref, a codex or clone failure, invalid arithmetic) is logged
    and skipped so one bad folder never aborts the whole batch. When
    ``overwrite`` is False, folders that already have an ``eval.json`` are
    skipped up front rather than re-judged.

    Args:
        root: Top-level directory to search for artifact folders.
        overwrite: Re-judge folders that already have an ``eval.json``.
        **kwargs: Forwarded verbatim to
            :func:`evaluate_artifact_folder_with_codex` (model, codex_bin,
            reasoning_effort, sandbox, timeouts, etc.).

    Returns:
        A mapping of artifact-folder path (as a string) to its validated
        evaluation, for every folder that scored successfully.

    Raises:
        JudgeError: If ``root`` is not a directory, or no artifact folder is
            found under it.
    """
    folders = _discover_artifact_folders(root)
    if not folders:
        raise JudgeError(
            f"no artifact folders found under {Path(root).expanduser().resolve()}: "
            f"expected at least one subdirectory containing {METRICS_FILENAME}"
        )

    logger.info("discovered %d artifact folder(s) under %s", len(folders), root)
    results: dict[str, dict[str, Any]] = {}
    failures: list[tuple[Path, str]] = []
    for index, folder in enumerate(folders, start=1):
        if not overwrite and (folder / "eval.json").exists():
            logger.info(
                "[%d/%d] skipping (eval.json exists): %s", index, len(folders), folder
            )
            continue
        logger.info("[%d/%d] judging %s", index, len(folders), folder)
        try:
            results[str(folder)] = evaluate_artifact_folder_with_codex(
                folder, overwrite=overwrite, **kwargs
            )
        except JudgeError as exc:
            logger.error("[%d/%d] failed %s: %s", index, len(folders), folder, exc)
            failures.append((folder, str(exc)))

    logger.info(
        "batch complete: %d scored, %d failed, %d total",
        len(results),
        len(failures),
        len(folders),
    )
    return results


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Score four SWE artifacts with an agentic codex exec judge that can "
            "read the candidate repository to verify claims."
        ),
        epilog=(
            "By default the repository is cloned from the folder's metrics.json "
            "(repo + ref) into the clone root and used as codex's read-only "
            "working root.\n\n"
            "Score one folder:\n"
            "  uv run scripts/codex_judge.py \\\n"
            "    --folder swe-benchmark-data/<repo>/<task>/<model>\n\n"
            "Score every folder under a tree (each subdirectory that contains a\n"
            "metrics.json is judged):\n"
            "  uv run scripts/codex_judge.py --recursive \\\n"
            "    --folder swe-benchmark-data"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Artifact folder to score, or (with --recursive) a top-level "
        "directory to search for artifact folders.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Treat --folder as a top-level directory: recursively find every "
        "subdirectory that contains a metrics.json and judge each one.",
    )
    parser.add_argument(
        "--repo",
        help="Use this local repository checkout as-is instead of cloning from "
        "metrics.json (read-only).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Codex model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--codex-bin", default=DEFAULT_CODEX_BIN)
    parser.add_argument("--git-bin", default=DEFAULT_GIT_BIN)
    parser.add_argument(
        "--clone-root",
        default=str(DEFAULT_CLONE_ROOT),
        help=f"Parent directory for judge repository checkouts "
        f"(default: {DEFAULT_CLONE_ROOT})",
    )
    parser.add_argument(
        "--clone-timeout-seconds", type=int, default=DEFAULT_CLONE_TIMEOUT_SECONDS
    )
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE_PATH))
    parser.add_argument(
        "--task-context-file", help="File containing independent task requirements"
    )
    parser.add_argument(
        "--repository-context-file",
        help="File containing independent repository evidence",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        help=f"Reasoning effort (default: {DEFAULT_REASONING_EFFORT})",
    )
    parser.add_argument(
        "--sandbox",
        default=DEFAULT_SANDBOX,
        choices=("read-only", "workspace-write", "danger-full-access"),
        help="Codex sandbox policy (default: read-only)",
    )
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--output-schema",
        action="store_true",
        help="Constrain codex output with the shared JSON Schema (opt-in; the "
        "prompt already requires strict JSON and the wrapper validates it)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail instead of replacing an existing eval.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the codex judge, and report the result."""
    args = _build_parser().parse_args(argv)
    common = dict(
        model=args.model,
        codex_bin=args.codex_bin,
        git_bin=args.git_bin,
        clone_root=Path(args.clone_root).expanduser(),
        clone_timeout_seconds=args.clone_timeout_seconds,
        template_path=args.template,
        task_context=optional_file(args.task_context_file, "task context"),
        repository_context=optional_file(
            args.repository_context_file, "repository context"
        ),
        reasoning_effort=args.reasoning_effort,
        sandbox=args.sandbox,
        timeout_seconds=args.timeout_seconds,
        use_output_schema=args.output_schema,
        overwrite=not args.no_overwrite,
    )

    if args.recursive:
        if args.repo:
            logger.error("--repo cannot be combined with --recursive")
            return 1
        try:
            results = evaluate_tree_with_codex(args.folder, **common)
        except JudgeError as exc:
            logger.error("%s", exc)
            return 1
        return 0 if results else 1

    try:
        result = evaluate_artifact_folder_with_codex(
            folder=args.folder, repo=args.repo, **common
        )
    except JudgeError as exc:
        logger.error("%s", exc)
        return 1

    eval_path = Path(args.folder).expanduser().resolve() / "eval.json"
    logger.info("wrote %s (task_score=%.2f)", eval_path, result["task_score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
