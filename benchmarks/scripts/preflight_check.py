#!/usr/bin/env python3
"""Pre-flight helper for the end-to-end benchmark orchestrator.

Two checks, each answering a question that is cheap to ask now and expensive to
discover mid-run:

* ``--check`` / ``--clear`` enumerate the artifact directories a SWE benchmark
  run would write to, for a given dataset and model, and either report which
  already exist (so a headless run does not stall on the /swe2 skill's overwrite
  prompt) or clear them.
* ``--check-repos`` confirms every repository in the dataset can be reached at
  its pinned ref, before a single task runs. The harness clones per task, so
  without this a bad credential or a ref that does not exist fails once per
  task, hours into a batch, with an opaque ``git clone failed``. This matters
  most for private repositories on GitHub Enterprise Server, where the failure
  is usually a missing credential helper entry or an untrusted internal CA.

The directory layout mirrors the harness exactly -- it reuses the dataset
loader, ``model_to_slug`` (the folder-name normalization), and ``_repo_name``
(the repo-basename derivation) rather than re-deriving any of them here, so this
helper and the harness can never disagree about where artifacts land.

Run from the ``benchmarks/`` directory:

    uv run scripts/preflight_check.py --dataset dataset/mcp-gateway-registry.yaml \
        --model qwen3.6-35b --check
    uv run scripts/preflight_check.py --dataset dataset/mcp-gateway-registry.yaml \
        --model qwen3.6-35b --clear
    uv run scripts/preflight_check.py --dataset dataset/my-team.yaml --check-repos

Exit codes: 0 = all clear, 2 = artifact folders exist and need clearing
(``--check`` only), 1 = an error (bad dataset, bad args, or an unreachable
repository).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import shutil
import subprocess  # nosec B404 - used with list args, no shell, hardcoded 'git'
import sys
from collections.abc import Callable
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from dataset_loader import (  # noqa: E402
    Dataset,
    DatasetError,
    Task,
    load_dataset,
)
from runner_config import (  # noqa: E402
    AGENT_KIRO,
    DEFAULT_AGENT,
    DEFAULT_SKILL,
    HARNESS_SLUGS,
    VALID_SKILLS,
    model_to_slug,
)

# The output root, relative to benchmarks/, matching the harness default.
_OUTPUT_DIR = "swe-benchmark-data"

# Long enough for a slow corporate proxy or a large enterprise server to answer,
# short enough that an unreachable host does not stall the pre-flight. A hung
# check is worse than a failed one: the whole point is to fail before the batch.
GIT_LS_REMOTE_TIMEOUT_SECONDS = 60

# A ref that looks like a commit SHA cannot be listed by `git ls-remote`, which
# only reports branches and tags. For those we check that the repository answers
# at all and leave the ref to the clone.
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

# Substrings git puts in stderr, mapped to the fix. The order matters: the first
# match wins, so the specific causes come before the generic ones.
_GIT_ERROR_HINTS = (
    (
        "terminal prompts disabled",
        "git needed a username or password and no credential was available. "
        "Store one (git config --global credential.helper store, then clone once "
        "by hand) or use an SSH key or deploy key.",
    ),
    (
        "could not read Username",
        "git needed a username or password and no credential was available. "
        "Store one (git config --global credential.helper store, then clone once "
        "by hand) or use an SSH key or deploy key.",
    ),
    (
        "Authentication failed",
        "the credential git found was rejected. Check the token has read access "
        "to this repository and has not expired.",
    ),
    (
        "Permission denied",
        "the credential git found was rejected. Check the token or key has read "
        "access to this repository.",
    ),
    (
        "SSL certificate problem",
        "the server's TLS certificate was not trusted. On an internal CA, point "
        "git at the CA bundle (export GIT_SSL_CAINFO=/path/to/ca.pem). The agent "
        "CLIs are Node programs and need NODE_EXTRA_CA_CERTS set to the same "
        "bundle separately.",
    ),
    (
        "unable to access",
        "the host did not answer. Check the URL, DNS, the security group and any "
        "proxy between this machine and the server.",
    ),
    (
        "Could not resolve host",
        "the hostname did not resolve. Check the URL and this machine's DNS.",
    ),
    (
        "not found",
        "the repository path does not exist on the server, or the credential "
        "cannot see it. Check the org and repo name.",
    ),
)


def _repo_name_from_harness() -> Callable[[str], str]:
    """Load the harness module and return its ``_repo_name`` function.

    The harness file name (``run-swe-headless.py``) is not a valid module
    identifier, so import it by path rather than with a plain ``import``.

    Returns:
        The harness ``_repo_name`` callable.

    Raises:
        RuntimeError: If the harness module cannot be loaded.
    """
    path = _SCRIPTS_DIR / "run-swe-headless.py"
    spec = importlib.util.spec_from_file_location("swe_harness", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load harness module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._repo_name


def _load_tasks(
    dataset_path: str,
    only_tasks: list[str] | None = None,
) -> tuple[Dataset, list[Task]]:
    """Load the dataset and apply the same task filter the harness applies.

    Both pre-flight checks need this, so it lives in one place: a filter that
    disagreed with the harness's ``--tasks`` would check the wrong things.

    Args:
        dataset_path: Path to the dataset YAML (relative to benchmarks/).
        only_tasks: If given, restrict to these task ids. Unknown ids raise.

    Returns:
        The loaded dataset and the selected tasks, in dataset order.

    Raises:
        DatasetError: If the dataset is missing/invalid or an id is unknown.
    """
    benchmarks_dir = _SCRIPTS_DIR.parent
    resolved = Path(dataset_path)
    if not resolved.is_absolute():
        resolved = benchmarks_dir / dataset_path
    dataset = load_dataset(resolved)
    tasks = list(dataset.tasks)
    if not only_tasks:
        return dataset, tasks
    wanted = {t.strip() for t in only_tasks if t.strip()}
    known = {t.id for t in tasks}
    unknown = wanted - known
    if unknown:
        raise DatasetError(
            f"unknown task id(s): {sorted(unknown)}; dataset has {sorted(known)}"
        )
    return dataset, [t for t in tasks if t.id in wanted]


def _ref_is_sha(ref: str) -> bool:
    """Return True when a ref looks like a commit SHA rather than a branch or tag."""
    return bool(_SHA_RE.match(ref))


def _diagnose_git_error(stderr: str) -> str | None:
    """Return a plain-English fix for a known git failure, or None if unrecognized.

    Args:
        stderr: The stderr git produced.

    Returns:
        The hint to print alongside git's own message, or None.
    """
    for needle, hint in _GIT_ERROR_HINTS:
        if needle.lower() in stderr.lower():
            return hint
    return None


def _check_repo_ref(repo: str, ref: str) -> str | None:
    """Check one repository answers at one ref, without cloning it.

    Uses ``git ls-remote``, which needs the same credentials a clone does but
    transfers no objects, so checking twenty repositories costs seconds. Terminal
    prompting is disabled: this runs unattended ahead of a long batch, and git
    waiting forever on a password prompt would be worse than a clear failure.

    Args:
        repo: The repository URL from the dataset.
        ref: The resolved ref (tag, branch, or commit) for the task.

    Returns:
        None when the repository is reachable at that ref, else a one-line
        explanation of what went wrong and how to fix it.
    """
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    args = ["git", "ls-remote", "--exit-code", repo]
    if not _ref_is_sha(ref):
        args += [f"refs/heads/{ref}", f"refs/tags/{ref}"]
    try:
        subprocess.run(  # nosec B603 B607 - hardcoded git, args are dataset values, no shell
            args,
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_LS_REMOTE_TIMEOUT_SECONDS,
            env=env,
        )
    except FileNotFoundError:
        return "git is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return (
            f"git ls-remote timed out after {GIT_LS_REMOTE_TIMEOUT_SECONDS}s. The "
            "host is unreachable, or something is waiting on a credential prompt."
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        hint = _diagnose_git_error(stderr)
        if hint is None and exc.returncode == 2 and not _ref_is_sha(ref):
            # ls-remote exits 2 when it matched no refs, which for a named ref
            # means the tag or branch is simply not there.
            hint = (
                f"the repository is reachable but has no branch or tag named "
                f"'{ref}'. Check the ref, and remember a task should pin the "
                "release BEFORE its fix so the defect is present in the tree."
            )
        message = hint or "git ls-remote failed."
        return f"{message}" + (f" git said: {stderr[:300]}" if stderr else "")
    return None


def _target_dirs(
    dataset_path: str,
    model: str,
    only_tasks: list[str] | None = None,
    agent: str = DEFAULT_AGENT,
    skill: str = DEFAULT_SKILL,
) -> list[Path]:
    """Return the artifact directory for every task in the dataset.

    Args:
        dataset_path: Path to the dataset YAML (relative to benchmarks/).
        model: The model id passed to the harness (full id, not the slug).
        only_tasks: If given, restrict to these task ids (the same filter the
            harness applies with --tasks), so a scoped run only checks/clears
            the folders it will actually write. Unknown ids raise DatasetError.
        agent: The coding agent (claude|pi); selects the harness folder level so
            the check/clear targets the same tree the harness will write.

    Returns:
        Absolute artifact directories, one per selected task, in dataset order.

    Raises:
        DatasetError: If the dataset is missing/invalid or an id is unknown.
    """
    benchmarks_dir = _SCRIPTS_DIR.parent
    dataset, tasks = _load_tasks(dataset_path, only_tasks)
    repo_name = _repo_name_from_harness()
    # kiro's managed model names carry dots; dash them so the preflight clears the
    # same dash-style folder the harness will write to (see model_to_slug).
    slug = model_to_slug(model, normalize_dots=(agent == AGENT_KIRO))
    # Layout: <model>/<harness>/<skill>/<scope>/<task> -- skill is its own level, so
    # check the exact tree the harness will write to. The scope follows the
    # dataset (output_scope, else the repo name), matching _artifact_dir.
    harness = HARNESS_SLUGS[agent]
    root = benchmarks_dir / _OUTPUT_DIR
    dirs = [
        root
        / slug
        / harness
        / skill
        / dataset.scope_for(repo_name(task.repo))
        / task.id
        for task in tasks
    ]
    _assert_inside_root(root, dirs)
    return dirs


def _assert_inside_root(root: Path, dirs: list[Path]) -> None:
    """Fail closed if a target folder resolves outside the results root.

    ``--clear`` deletes every non-empty target folder, and the path parts come
    from a dataset task id and the --model argument. A value such as
    ``../../scripts`` would otherwise point the delete at source code.

    Raises:
        DatasetError: If any folder is the root itself or lies outside it.
    """
    resolved_root = root.resolve()
    for d in dirs:
        resolved = d.resolve()
        if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
            raise DatasetError(
                f"Refusing to use target folder {d}: it resolves to {resolved}, "
                f"outside {resolved_root}. Check the task id and --model for '..' "
                "or absolute paths."
            )


def _existing(dirs: list[Path]) -> list[Path]:
    """Return the subset of dirs that exist and hold any file from an earlier run.

    The design artifacts make the skill stop at its overwrite prompt, but a
    failed attempt can leave only metrics.json and an agent stream file. Those
    leak into the next run too: the harness appends to an existing stream file
    (issue #196), so any file counts. An empty folder does not.
    """
    return [d for d in dirs if d.is_dir() and any(d.iterdir())]


def _run_check(dirs: list[Path]) -> int:
    """Report existing artifact folders. Returns the process exit code."""
    existing = _existing(dirs)
    if not existing:
        logger.info("OK: no existing artifact folders for this model; safe to run.")
        logger.info("Would write %d task folder(s):", len(dirs))
        for d in dirs:
            logger.info("  %s", d)
        return 0
    logger.warning(
        "%d of %d target folder(s) already hold output from an earlier run, which "
        "would stall the skill's overwrite prompt or mix into this run's files:",
        len(existing),
        len(dirs),
    )
    for d in existing:
        logger.warning("  EXISTS: %s", d)
    logger.warning("Clear them with --clear (or rename them to keep the prior run).")
    return 2


def _run_check_repos(dataset_path: str, only_tasks: list[str] | None) -> int:
    """Check every repository in the dataset is reachable at its pinned ref.

    Each distinct (repo, ref) pair is checked once, so a dataset with twenty
    tasks over two repositories makes two network calls rather than twenty.

    Args:
        dataset_path: Path to the dataset YAML (relative to benchmarks/).
        only_tasks: If given, restrict to these task ids.

    Returns:
        The process exit code: 0 when every repository answered, 1 otherwise.

    Raises:
        DatasetError: If the dataset is missing/invalid or an id is unknown.
    """
    dataset, tasks = _load_tasks(dataset_path, only_tasks)
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for task in tasks:
        pair = (task.repo, dataset.resolved_ref(task))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    logger.info("Checking %d repository/ref pair(s) are reachable...", len(pairs))
    failures: list[tuple[str, str, str]] = []
    for repo, ref in pairs:
        problem = _check_repo_ref(repo, ref)
        if problem is None:
            logger.info("  OK: %s @ %s", repo, ref)
        else:
            logger.error("  UNREACHABLE: %s @ %s -- %s", repo, ref, problem)
            failures.append((repo, ref, problem))

    if not failures:
        logger.info("All %d repository/ref pair(s) reachable.", len(pairs))
        return 0
    logger.error(
        "%d of %d repository/ref pair(s) could not be reached. The harness clones "
        "per task, so this would fail once per task, hours into a run.",
        len(failures),
        len(pairs),
    )
    logger.error(
        "Private repositories need a credential available to this shell without "
        "prompting. See docs/benchmark-your-own-repo.md for the setup."
    )
    return 1


def _run_clear(dirs: list[Path]) -> int:
    """Remove existing artifact folders. Returns the process exit code."""
    existing = _existing(dirs)
    if not existing:
        logger.info("Nothing to clear: no existing artifact folders for this model.")
        return 0
    for d in existing:
        shutil.rmtree(d)
        logger.info("cleared %s", d)
    logger.info("Cleared %d folder(s).", len(existing))
    return 0


def main() -> None:
    """Parse arguments and run the requested pre-flight action."""
    parser = argparse.ArgumentParser(
        description="Check or clear the artifact folders a benchmark run would write to.",
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset YAML path (relative to benchmarks/)."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id (the full id passed to the harness). Required for --check "
        "and --clear, which resolve a per-model artifact folder; not used by "
        "--check-repos, which only talks to the dataset's repositories.",
    )
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        choices=sorted(HARNESS_SLUGS),
        help="Coding agent (claude|pi); selects the harness folder level so the "
        "check/clear targets the same tree the harness writes. Default: claude.",
    )
    parser.add_argument(
        "--skill",
        default=DEFAULT_SKILL,
        choices=sorted(VALID_SKILLS),
        help="SWE skill (swe2|swe3); a non-default skill appends to the harness "
        "folder (e.g. claude-code-swe3) so the check/clear targets the same tree. "
        "Default: swe2.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check", action="store_true", help="Report existing folders (exit 2 if any)."
    )
    group.add_argument(
        "--clear", action="store_true", help="Remove existing artifact folders."
    )
    group.add_argument(
        "--check-repos",
        action="store_true",
        help="Check every repository in the dataset is reachable at its pinned "
        "ref, before any task runs (exit 1 if any is not).",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task ids to scope to (default: all tasks in the "
        "dataset). Matches the harness's --tasks so a scoped run only "
        "checks/clears the folders it will write.",
    )
    args = parser.parse_args()

    only_tasks = (
        [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None
    )

    if args.check_repos:
        try:
            sys.exit(_run_check_repos(args.dataset, only_tasks))
        except DatasetError as exc:
            logger.error("Dataset error: %s", exc)
            sys.exit(1)

    if not args.model:
        logger.error("--model is required for --check and --clear.")
        sys.exit(1)
    try:
        dirs = _target_dirs(
            args.dataset, args.model, only_tasks, args.agent, args.skill
        )
    except DatasetError as exc:
        logger.error("Dataset error: %s", exc)
        sys.exit(1)

    sys.exit(_run_check(dirs) if args.check else _run_clear(dirs))


if __name__ == "__main__":
    main()
