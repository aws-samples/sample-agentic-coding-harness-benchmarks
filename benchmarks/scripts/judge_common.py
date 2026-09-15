#!/usr/bin/env python3
"""Shared core for the SWE artifact judges.

Both judge backends -- the direct Bedrock Mantle call (``llm_as_judge.py``) and
the agentic ``codex exec`` run (``codex_judge.py``) -- score the same five
artifacts (four design documents plus the /swe2 implementation) against the same
rubric and must produce identically-shaped, identically-validated ``eval.json``
output. That common ground lives here:

  * the strict score schema (``EvaluationResult`` and friends),
  * prompt rendering from ``judge_prompt.txt`` (``render_judge_prompt``),
  * parsing and validating a model's reply (``parse_and_validate_result``),
  * the atomic ``eval.json`` writer (``atomic_write_json``),
  * small file helpers (``read_text``, ``optional_file``).

Each backend imports these and adds only its own transport (an HTTP request vs.
a codex subprocess) plus its judge-metadata block.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from string import Template
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# The four design artifacts every run must produce. These -- and only these --
# drive the missing-artifact folder-zeroing check: a run that fails to produce
# one of them is a genuine candidate failure and scores 0 for the whole folder.
ARTIFACT_FILES = {
    "github_issue": "github-issue.md",
    "lld": "lld.md",
    "review": "review.md",
    "testing": "testing.md",
}
# The /swe2 implementation artifact, scored as a fifth artifact on the same
# 0-100 scale. It is OPTIONAL: a design-only (/swe) run, or a /swe2 run that did
# not land a patch, simply scores the implementation artifact 0 (empty content)
# while the four design artifacts are still judged normally. It therefore does
# NOT belong to ARTIFACT_FILES / the folder-zeroing set.
IMPLEMENTATION_FILES = {
    "summary": "implementation.md",
    "patch": "patch.diff",
}

# Alias resolution. Weaker models often produce the right artifact CONTENT but
# under a different filename (e.g. EXPERT_REVIEW.md, expert_review.md,
# low-level-design.md, GITHUB_ISSUE_SPEC.md) instead of the canonical name. Rather
# than score such a run 0 for a naming miss, resolve each canonical artifact to a
# file on disk by matching a normalized stem against a per-artifact alias pattern.
# The canonical name always wins if present; scaffolding files (README, the
# clarifying-answers input) are never matched.
#
# Normalization: lowercase, strip the extension, and collapse any run of
# non-alphanumeric characters (-, _, space) to a single separator, so
# "Low-Level_Design" and "low level design" both normalize to "low level design".
_ARTIFACT_ALIAS_PATTERNS: dict[str, re.Pattern[str]] = {
    # e.g. github-issue, github issue, github_issue_spec, issue-spec, issue
    "github_issue": re.compile(r"^(github[ ]?issue([ ]spec)?|issue([ ]spec)?)$"),
    # e.g. lld, low-level-design, low level design, lowleveldesign, design-doc
    "lld": re.compile(r"^(lld|low[ ]?level[ ]?design|design([ ]doc)?)$"),
    # e.g. review, expert-review, expert_review, code-review
    "review": re.compile(r"^((expert|code)[ ]?)?review$"),
    # e.g. testing, testing-plan, test-plan, tests
    "testing": re.compile(r"^(test(ing)?([ ]plan)?|tests)$"),
}
# Files that are NEVER an artifact even if a pattern might loosely match.
_NON_ARTIFACT_STEMS = {"readme", "answers", "metrics", "eval", "run-summary"}


def _normalize_stem(filename: str) -> str:
    """Lowercase the filename's stem and collapse separators to single spaces."""
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9]+", " ", stem).strip()


def resolve_artifact(artifact_dir: Path, key: str) -> Path | None:
    """Resolve a canonical artifact ``key`` to an existing file in ``artifact_dir``.

    Resolution order:
      1. The canonical filename (e.g. ``review.md``) if it exists -- always wins.
      2. Otherwise any ``*.md`` whose normalized stem matches the artifact's alias
         pattern. On multiple matches (e.g. ``EXPERT_REVIEW.md`` AND
         ``expert_review.md``), prefer an all-lowercase filename, then the shortest
         name, then lexicographic order -- fully deterministic.

    Args:
        artifact_dir: The resolved task artifact directory.
        key: An ``ARTIFACT_FILES`` key (github_issue, lld, review, testing).

    Returns:
        The resolved ``Path``, or None if neither the canonical name nor any alias
        is present.
    """
    canonical = artifact_dir / ARTIFACT_FILES[key]
    if canonical.exists():
        return canonical
    pattern = _ARTIFACT_ALIAS_PATTERNS.get(key)
    if pattern is None:
        return None
    candidates: list[Path] = []
    for path in artifact_dir.glob("*.md"):
        stem = _normalize_stem(path.name)
        if stem in _NON_ARTIFACT_STEMS:
            continue
        if pattern.match(stem):
            candidates.append(path)
    if not candidates:
        return None
    # Prefer all-lowercase names (islower() is False for MixedCase/UPPER), then
    # shortest, then lexicographic -- deterministic on collisions.
    candidates.sort(key=lambda p: (not p.name.islower(), len(p.name), p.name))
    return candidates[0]


# Cap the embedded patch so a large diff cannot blow the judge context. The head
# of the patch carries the substantive change; the tail is truncated with a
# marker so the judge knows content was elided rather than absent.
MAX_PATCH_CHARS = 200_000
DEFAULT_TEMPLATE_PATH = Path(__file__).with_name("judge_prompt.txt")
Score = Annotated[int, Field(strict=True, ge=0, le=25)]


class JudgeError(Exception):
    """Raised when judge inputs, model output, or score data are invalid."""


class ArtifactScore(BaseModel):
    """Validated scores for one artifact."""

    model_config = ConfigDict(extra="forbid")

    completeness: Score
    correctness: Score
    specificity: Score
    risk_awareness: Score
    total: Annotated[int, Field(strict=True, ge=0, le=100)]
    notes: str

    @model_validator(mode="after")
    def total_is_correct(self) -> "ArtifactScore":
        expected = (
            self.completeness
            + self.correctness
            + self.specificity
            + self.risk_awareness
        )
        if self.total != expected:
            raise ValueError(f"total is {self.total}; expected {expected}")
        return self


class ScoreSet(BaseModel):
    """The fixed five-artifact score set.

    The first four are the design artifacts (github issue, LLD, review, testing
    plan); ``implementation`` scores the /swe2 code change (``patch.diff`` plus
    ``implementation.md``) on the same 0-100 scale. For a design-only run the
    implementation artifact is scored 0.
    """

    model_config = ConfigDict(extra="forbid")

    github_issue: ArtifactScore
    lld: ArtifactScore
    review: ArtifactScore
    testing: ArtifactScore
    implementation: ArtifactScore


class EvaluationResult(BaseModel):
    """Strict model-produced evaluation before judge metadata is attached."""

    model_config = ConfigDict(extra="forbid")

    task: str
    model: str
    scores: ScoreSet
    task_score: float
    verdict: str

    @model_validator(mode="after")
    def task_score_is_correct(self) -> "EvaluationResult":
        totals = [
            self.scores.github_issue.total,
            self.scores.lld.total,
            self.scores.review.total,
            self.scores.testing.total,
            self.scores.implementation.total,
        ]
        expected = round(sum(totals) / len(totals), 2)
        if abs(self.task_score - expected) > 0.001:
            raise ValueError(f"task_score is {self.task_score}; expected {expected}")
        return self


def missing_artifacts(folder: str | Path) -> list[str]:
    """Return the required artifact filenames that are missing or empty.

    A required artifact that does not exist -- or exists but is blank -- is a
    genuine candidate (model) failure: the run did not produce that design
    document. This lets the judge score such a folder 0 rather than erroring out
    and dropping it from the results.

    Args:
        folder: The artifact directory.

    Returns:
        The missing/empty artifact filenames (e.g. ``["github-issue.md"]``),
        empty if all four are present and non-empty.
    """
    artifact_dir = Path(folder).expanduser().resolve()
    missing: list[str] = []
    for key, filename in ARTIFACT_FILES.items():
        # Accept the canonical name or a recognized alias (see resolve_artifact).
        path = resolve_artifact(artifact_dir, key)
        try:
            if path is None or not path.read_text(encoding="utf-8").strip():
                missing.append(filename)
        except (FileNotFoundError, OSError):
            missing.append(filename)
    return missing


def zero_score_result(
    *, task_id: str, candidate_id: str, missing: list[str]
) -> dict[str, Any]:
    """Build a valid zero-score evaluation for a folder missing artifacts.

    The result is schema-shaped exactly like a judged one (all criteria 0, all
    totals 0, ``task_score`` 0.0) so downstream tooling treats it uniformly, but
    the verdict names the missing artifacts as the reason -- a genuine model
    failure, not a judging error.

    Args:
        task_id: The task identifier for the ``task`` field.
        candidate_id: The candidate (model) identifier for the ``model`` field.
        missing: The missing/empty artifact filenames to name in the verdict.

    Returns:
        A validated, JSON-ready evaluation dict with ``task_score`` 0.0.
    """
    zero_artifact = {
        "completeness": 0,
        "correctness": 0,
        "specificity": 0,
        "risk_awareness": 0,
        "total": 0,
        "notes": "Artifact not produced by the candidate run.",
    }
    verdict = (
        "MODEL FAILURE: the candidate run did not produce the required "
        f"artifact(s): {', '.join(missing)}. Scored 0."
    )
    result = EvaluationResult.model_validate(
        {
            "task": task_id,
            "model": candidate_id,
            "scores": {
                "github_issue": dict(zero_artifact),
                "lld": dict(zero_artifact),
                "review": dict(zero_artifact),
                "testing": dict(zero_artifact),
                "implementation": dict(zero_artifact),
            },
            "task_score": 0.0,
            "verdict": verdict,
        }
    )
    return result.model_dump(mode="json")


def identify_folder(folder: str | Path) -> tuple[str, str]:
    """Resolve (task_id, candidate_id) for a folder the way the judge does.

    Mirrors the identifier resolution in :func:`render_judge_prompt`: prefer
    ``metrics.json`` fields, else the ``<model>/<repo>/<task>`` folder layout.

    Args:
        folder: The artifact directory.

    Returns:
        A tuple of (task id, candidate id).
    """
    artifact_dir = Path(folder).expanduser().resolve()
    metrics_path = artifact_dir / "metrics.json"
    metadata = (
        _load_json_object(metrics_path, "metrics.json") if metrics_path.exists() else {}
    )
    task_id = metadata.get("task") or artifact_dir.name
    candidate_id = metadata.get("model") or artifact_dir.parent.parent.name
    return str(task_id), str(candidate_id)


def read_text(path: Path, label: str) -> str:
    """Read a non-empty UTF-8 text file, raising JudgeError on any problem.

    Args:
        path: File to read.
        label: Human-readable name used in error messages.

    Returns:
        The file's text.

    Raises:
        JudgeError: If the file is missing, unreadable, or empty.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise JudgeError(f"missing {label}: {path}") from exc
    except OSError as exc:
        raise JudgeError(f"could not read {label} {path}: {exc}") from exc
    if not content.strip():
        raise JudgeError(f"{label} is empty: {path}")
    return content


def optional_file(path: str | None, label: str) -> str | None:
    """Read a file when a path is given, else return None.

    Args:
        path: Optional file path.
        label: Human-readable name used in error messages.

    Returns:
        The file's text, or None when no path was supplied.
    """
    return read_text(Path(path).expanduser().resolve(), label) if path else None


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise JudgeError(f"missing {label}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise JudgeError(f"could not parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise JudgeError(f"{label} must contain a top-level JSON object: {path}")
    return value


def _read_implementation(artifact_dir: Path) -> str:
    """Assemble the /swe2 implementation artifact text for the judge.

    Combines ``implementation.md`` (the human summary) and ``patch.diff`` (the
    actual code change) into one labeled block. Both are OPTIONAL: a design-only
    run has neither, and this returns an empty string so the judge scores the
    implementation artifact 0 without erroring. The patch is truncated to
    ``MAX_PATCH_CHARS`` so a large diff cannot blow the judge context.

    Args:
        artifact_dir: The resolved artifact directory.

    Returns:
        The combined implementation text, or an empty string when neither the
        summary nor the patch is present/non-empty.
    """
    parts: list[str] = []
    # implementation.md may appear case-variant (IMPLEMENTATION.md); patch.diff is
    # the sole *.diff. Both optional -> missing content just scores this artifact 0.
    summary_path = artifact_dir / IMPLEMENTATION_FILES["summary"]
    if not summary_path.exists():
        impl_variants = sorted(
            (
                p
                for p in artifact_dir.glob("*.md")
                if _normalize_stem(p.name) == "implementation"
            ),
            key=lambda p: (not p.name.islower(), len(p.name), p.name),
        )
        if impl_variants:
            summary_path = impl_variants[0]
    patch_path = artifact_dir / IMPLEMENTATION_FILES["patch"]
    if not patch_path.exists():
        diffs = sorted(artifact_dir.glob("*.diff"), key=lambda p: (len(p.name), p.name))
        if diffs:
            patch_path = diffs[0]
    try:
        summary = summary_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        summary = ""
    try:
        patch = patch_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        patch = ""
    if summary:
        parts.append("## Implementation summary (implementation.md)\n\n" + summary)
    if patch:
        if len(patch) > MAX_PATCH_CHARS:
            patch = (
                patch[:MAX_PATCH_CHARS]
                + f"\n\n[... patch truncated at {MAX_PATCH_CHARS} chars for length ...]"
            )
        parts.append("## Code change (patch.diff)\n\n```diff\n" + patch + "\n```")
    return "\n\n".join(parts)


def _default_task_context(metadata: dict[str, Any]) -> str:
    for key in ("task_context", "problem_statement", "task_description"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return (
        "No independent task statement was supplied. Evaluate requirement coverage "
        "only where established by the task identifier, repository context, or "
        "internally consistent artifacts, and report this evidence gap."
    )


def _default_repository_context(metadata: dict[str, Any]) -> str:
    context = {
        key: metadata[key]
        for key in ("repo", "ref", "complexity", "tags")
        if key in metadata
    }
    return (
        json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True)
        if context
        else "No independent repository context was supplied."
    )


def render_judge_prompt(
    folder: str | Path,
    *,
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
    task_context: str | None = None,
    repository_context: str | None = None,
) -> tuple[str, str, str, dict[str, Any] | None]:
    """Load an artifact folder and render ``judge_prompt.txt``.

    Args:
        folder: Directory containing the four required Markdown artifacts, plus
            optionally the /swe2 implementation artifact (``implementation.md`` +
            ``patch.diff``); when absent, the implementation is judged 0.
        template_path: Judge prompt template path.
        task_context: Optional independent task requirements. Defaults from
            ``metrics.json`` when present, else a documented evidence-gap notice.
        repository_context: Optional independent repository evidence. Defaults
            from ``metrics.json`` fields when present.

    Returns:
        A tuple of (rendered prompt, task id, candidate id, metrics-or-None).

    Raises:
        JudgeError: If the folder, artifacts, or template are invalid.
    """
    artifact_dir = Path(folder).expanduser().resolve()
    if not artifact_dir.is_dir():
        raise JudgeError(f"artifact folder is not a directory: {artifact_dir}")

    metrics_path = artifact_dir / "metrics.json"
    metrics = (
        _load_json_object(metrics_path, "metrics.json")
        if metrics_path.exists()
        else None
    )
    metadata = metrics or {}
    # Prefer identifiers recorded in metrics.json. Fall back to the folder
    # layout, which is <model>/<repo>/<task>/: the leaf is the task and the
    # grandparent is the model.
    task_id = metadata.get("task") or artifact_dir.name
    candidate_id = metadata.get("model") or artifact_dir.parent.parent.name
    if not isinstance(task_id, str) or not task_id.strip():
        raise JudgeError("task identifier must be a non-empty string")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise JudgeError("candidate identifier must be a non-empty string")

    # Resolve each artifact to its canonical name or a recognized alias, so a run
    # that produced the right content under a variant filename is still judged.
    artifacts = {
        name: read_text(
            resolve_artifact(artifact_dir, name) or (artifact_dir / filename),
            filename,
        )
        for name, filename in ARTIFACT_FILES.items()
    }
    # The implementation artifact is optional: empty string -> judge scores it 0.
    implementation = _read_implementation(artifact_dir)
    template = Template(
        read_text(Path(template_path).expanduser().resolve(), "prompt template")
    )
    values = {
        "TASK_ID_JSON": json.dumps(task_id, ensure_ascii=False),
        "CANDIDATE_ID_JSON": json.dumps(candidate_id, ensure_ascii=False),
        "TASK_CONTEXT_JSON": json.dumps(
            task_context
            if task_context is not None
            else _default_task_context(metadata),
            ensure_ascii=False,
        ),
        "REPOSITORY_CONTEXT_JSON": json.dumps(
            repository_context
            if repository_context is not None
            else _default_repository_context(metadata),
            ensure_ascii=False,
        ),
        "GITHUB_ISSUE_JSON": json.dumps(artifacts["github_issue"], ensure_ascii=False),
        "LLD_JSON": json.dumps(artifacts["lld"], ensure_ascii=False),
        "REVIEW_JSON": json.dumps(artifacts["review"], ensure_ascii=False),
        "TESTING_JSON": json.dumps(artifacts["testing"], ensure_ascii=False),
        "IMPLEMENTATION_JSON": json.dumps(implementation, ensure_ascii=False),
    }
    try:
        prompt = template.substitute(values)
    except (KeyError, ValueError) as exc:
        raise JudgeError(f"invalid prompt template {template_path}: {exc}") from exc
    return prompt, task_id, candidate_id, metrics


def parse_and_validate_result(
    text: str, *, task_id: str, candidate_id: str
) -> dict[str, Any]:
    """Parse a model reply into a validated evaluation dict.

    Tolerates a single fenced code block wrapping the JSON. Enforces the strict
    schema (criteria 0-25, totals = sums, task_score = mean of the five artifact
    totals) and that the returned identifiers match the submission exactly.

    Args:
        text: The model's reply text.
        task_id: The task id the reply must echo.
        candidate_id: The candidate id the reply must echo.

    Returns:
        The validated evaluation as a JSON-ready dict.

    Raises:
        JudgeError: If the reply is not valid JSON, fails the schema, or the
            identifiers do not match.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        raw = json.loads(candidate)
        result = EvaluationResult.model_validate(raw)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise JudgeError(f"judge returned an invalid evaluation: {exc}") from exc
    if result.task != task_id:
        raise JudgeError(f"judge returned task {result.task!r}; expected {task_id!r}")
    if result.model != candidate_id:
        raise JudgeError(
            f"judge returned model {result.model!r}; expected candidate {candidate_id!r}"
        )
    return result.model_dump(mode="json")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write JSON to ``path`` atomically (temp file + fsync + os.replace).

    Args:
        path: Destination file.
        value: JSON-serializable mapping to write.

    Raises:
        JudgeError: If the file cannot be written.
    """
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise JudgeError(f"could not write {path}: {exc}") from exc
