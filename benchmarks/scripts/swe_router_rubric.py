"""The swe-router task rubric: scoring contract, policy and consolidation.

The rubric replaces the skill's two free-form judgments (a consequence floor
and a complexity tier) with six dimensions scored 0, 1 or 2. A model scores the
dimensions; this module turns the scores into the two numbers routing needs:

* **required quality**: the score a result must reach, from blast radius and
  verification (``routing-policy.json``, ``required_quality``);
* **difficulty bucket**: which column of measured scores to compare against
  it, from the other four dimensions (``difficulty``).

The model never sees the numbers. It describes the task; the policy prices it.

Everything here is a pure function so the judgment runner
(``run-swe-router-headless.py --mode rubric``) and the replay
(``eval_swe_router.py --rubric-inputs``) share one implementation.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# The rubric and policy ship with the skill, so the skill's copy is the only one.
_SKILL_DIR = Path(__file__).resolve().parents[2] / "vend" / "swe-router"
DEFAULT_RUBRIC = _SKILL_DIR / "rubric.md"
DEFAULT_POLICY = _SKILL_DIR / "routing-policy.json"

DIMENSIONS = (
    "hidden_contract",
    "silent_invariant",
    "coordinated_surfaces",
    "ambiguity",
    "verification",
    "blast_radius",
)
SCORES = (0, 1, 2)
CONFIDENCE_LEVELS = ("high", "medium", "low")

# Phase 0 replays through the existing route.py, which knows only its four tier
# names. The buckets are mapped onto those column keys, in order; the report
# names them by bucket. Phase 1 teaches route.py the bucket names directly.
BUCKET_ORDER = ("easy", "moderate", "hard", "very_hard")
BUCKET_TO_ROUTE_TIER = dict(zip(BUCKET_ORDER, ("trivial", "low", "medium", "high")))

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class RubricError(ValueError):
    """A rubric judgment or policy that cannot be used."""


class RequiredQuality(BaseModel):
    """How blast radius and verification set the score a result must reach."""

    by_blast_radius: dict[int, float]
    add_when_verification_is_2: float = Field(default=0, ge=0, le=20)
    meaning: dict[str, str] = Field(default_factory=dict)

    @field_validator("by_blast_radius")
    @classmethod
    def _covers_every_score(cls, v: dict[int, float]) -> dict[int, float]:
        if sorted(v) != list(SCORES):
            raise ValueError(
                f"by_blast_radius must define {list(SCORES)}, got {sorted(v)}"
            )
        if any(not 0 < q <= 100 for q in v.values()):
            raise ValueError("required quality values must be in (0, 100]")
        return v


class Difficulty(BaseModel):
    """Which dimensions add up to difficulty, and how the sum is bucketed.

    ``absolute`` uses the fixed ``buckets`` ranges. ``relative`` splits the
    difficulty sums of the benchmark's own tasks into quarters, so "hard" means
    harder than most of the work that was measured. Models score the rubric
    generously, so fixed ranges put almost every task in the top two buckets;
    quarters keep the ranking whatever the overall level. ``buckets`` stays the
    fallback when no benchmark tasks are available to take quarters of.

    On the 21-task v2 set, quarters sorted tasks better but left columns of 3-12
    tasks, too few for steady per-model means, and routed worse than absolute
    buckets. Prefer ``relative`` only for a task set large enough to fill four
    columns.
    """

    dimensions: list[str]
    bucketing: Literal["absolute", "relative"] = "absolute"
    buckets: dict[str, tuple[int, int]]

    @model_validator(mode="after")
    def _check(self) -> Difficulty:
        unknown = set(self.dimensions) - set(DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown difficulty dimension(s): {sorted(unknown)}")
        if list(self.buckets) != list(BUCKET_ORDER):
            raise ValueError(f"buckets must be {list(BUCKET_ORDER)} in that order")
        expected = 0
        for name, (lo, hi) in self.buckets.items():
            if lo != expected or hi < lo:
                raise ValueError(
                    f"bucket {name} must start at {expected} and not be empty"
                )
            expected = hi + 1
        if expected - 1 != 2 * len(self.dimensions):
            raise ValueError(
                f"buckets must cover 0..{2 * len(self.dimensions)}, end at {expected - 1}"
            )
        return self


class RoutingPolicy(BaseModel):
    """The numbers a platform team owns: required quality and difficulty buckets."""

    schema_version: str
    required_quality: RequiredQuality
    difficulty: Difficulty
    tie_band_points: float = Field(default=3.0, ge=0)
    min_completion_in_bucket: float = Field(default=1.0, ge=0, le=1)
    when_unsure: str = "stay_on_current_model"


def load_policy(path: Path = DEFAULT_POLICY) -> RoutingPolicy:
    """Load and validate a routing policy.

    Raises:
        RubricError: If the file is missing, not JSON, or fails validation.
    """
    try:
        return RoutingPolicy.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RubricError(f"no routing policy at {path}") from exc
    except ValueError as exc:
        raise RubricError(f"invalid routing policy {path}: {exc}") from exc


def required_quality(scores: dict[str, int], policy: RoutingPolicy) -> float:
    """The score a result must reach for this task."""
    rq = policy.required_quality
    base = rq.by_blast_radius[scores["blast_radius"]]
    return base + (rq.add_when_verification_is_2 if scores["verification"] == 2 else 0)


def difficulty_sum(scores: dict[str, int], policy: RoutingPolicy) -> int:
    """Sum of the difficulty dimensions."""
    return sum(scores[d] for d in policy.difficulty.dimensions)


def bucket(scores: dict[str, int], policy: RoutingPolicy) -> str:
    """The difficulty bucket for these scores."""
    total = difficulty_sum(scores, policy)
    for name, (lo, hi) in policy.difficulty.buckets.items():
        if lo <= total <= hi:
            return name
    raise RubricError(f"difficulty sum {total} falls in no bucket")


def relative_edges(sums: list[int]) -> tuple[float, float, float]:
    """Quartile cut points of the benchmark tasks' difficulty sums.

    Raises:
        RubricError: With fewer than four sums, where quarters mean nothing.
    """
    if len(sums) < 4:
        raise RubricError(f"relative buckets need at least 4 tasks, got {len(sums)}")
    q1, q2, q3 = statistics.quantiles(sums, n=4, method="inclusive")
    return q1, q2, q3


def relative_bucket(total: int, edges: tuple[float, float, float]) -> str:
    """Place a difficulty sum among quartile cut points.

    A sum equal to a cut point falls in the lower bucket, so a tie never
    promotes a task.
    """
    for name, edge in zip(BUCKET_ORDER, edges):
        if total <= edge:
            return name
    return BUCKET_ORDER[-1]


def build_prompt(task_id: str, problem: str, clone_path: Path, rubric_text: str) -> str:
    """The rubric-pass prompt: the rubric, where the repo is, and the output contract."""
    fields = ",\n".join(
        f'    "{d}": {{"score": <0, 1 or 2>, "evidence": "<file, doc, test or sentence>"}}'
        for d in DIMENSIONS
    )
    return "\n".join(
        [
            rubric_text,
            "",
            "===TASK===",
            "",
            "Score the coding task below with the rubric above. Do NOT recommend a "
            "model and do NOT write, edit or run any code in the repository.",
            "",
            f"The repository the change would land in is cloned at {clone_path} and "
            "is your working directory. Read its map and whatever else the rubric "
            "asks for; do not modify anything.",
            "",
            "Finish by printing EXACTLY one fenced JSON block and nothing after it:",
            "",
            "```json",
            "{",
            '  "scores": {',
            fields,
            "  },",
            f'  "confidence": "<one of: {", ".join(CONFIDENCE_LEVELS)}>",',
            '  "summary": "<two sentences: what the task needs in this repository>"',
            "}",
            "```",
            "",
            f"Task id: {task_id}",
            "",
            "Task description:",
            problem or "(see reference issue)",
        ]
    )


def _candidates(text: str) -> list[str]:
    """Fenced JSON blocks in the text, else bare objects that parse."""
    fenced = [m.group(1) for m in _JSON_FENCE_RE.finditer(text)]
    if fenced:
        return fenced
    found = []
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        for end in range(len(text), start, -1):
            try:
                if isinstance(json.loads(text[start:end]), dict):
                    found.append(text[start:end])
                    break
            except json.JSONDecodeError:
                continue
    return found


def _validate(parsed: dict[str, Any]) -> dict[str, Any]:
    """Check one parsed judgment; return it with scores flattened to ints."""
    raw = parsed.get("scores")
    if not isinstance(raw, dict) or set(raw) != set(DIMENSIONS):
        raise RubricError(f"scores must name exactly {list(DIMENSIONS)}")
    scores, evidence = {}, {}
    for dim in DIMENSIONS:
        entry = raw[dim]
        value = entry.get("score") if isinstance(entry, dict) else entry
        if isinstance(value, bool) or value not in SCORES:
            raise RubricError(f"{dim} score must be 0, 1 or 2, got {value!r}")
        scores[dim] = int(value)
        evidence[dim] = (
            str(entry.get("evidence", "")) if isinstance(entry, dict) else ""
        )
    confidence = parsed.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        raise RubricError(
            f"confidence must be one of {CONFIDENCE_LEVELS}, got {confidence!r}"
        )
    return {
        "scores": scores,
        "evidence": evidence,
        "confidence": confidence,
        "summary": str(parsed.get("summary", "")),
    }


def extract_judgment(text: str) -> dict[str, Any]:
    """Pull the rubric judgment out of the agent's final message.

    Takes the last block that validates, so a model that thinks aloud in JSON
    before its final answer is still read correctly.

    Raises:
        RubricError: If no block holds six valid scores and a confidence.
    """
    errors = []
    for chunk in reversed(_candidates(text)):
        try:
            return _validate(json.loads(chunk))
        except (json.JSONDecodeError, RubricError) as exc:
            errors.append(str(exc))
    detail = errors[0] if errors else "no JSON object found"
    raise RubricError(f"no usable rubric judgment ({detail}): {text.strip()[-400:]!r}")


def consolidate(
    judgments: list[dict[str, Any]], policy: RoutingPolicy
) -> dict[str, Any]:
    """Reduce a task's repeated judgments to the scores routing will use.

    Each dimension takes the lower median of its repeats, so it stays one of
    0, 1 or 2. Required quality and bucket are computed from those scores, and
    beside them the values each repeat produced on its own: a task whose
    bucket moves between identical runs is a finding about the rubric.

    Raises:
        RubricError: If there are no judgments.
    """
    if not judgments:
        raise RubricError("no successful judgments to consolidate")
    scores = {
        d: statistics.median_low(j["scores"][d] for j in judgments) for d in DIMENSIONS
    }
    per_run_quality = [required_quality(j["scores"], policy) for j in judgments]
    per_run_bucket = [bucket(j["scores"], policy) for j in judgments]
    confidence = Counter(j["confidence"] for j in judgments).most_common(1)[0][0]
    return {
        "scores": scores,
        "required_quality": required_quality(scores, policy),
        "difficulty_sum": difficulty_sum(scores, policy),
        "bucket": bucket(scores, policy),
        "confidence": confidence,
        "summary": judgments[-1].get("summary", ""),
        "attempts": len(judgments),
        "scores_seen": {
            d: sorted(j["scores"][d] for j in judgments) for d in DIMENSIONS
        },
        "quality_seen": sorted(per_run_quality),
        "buckets_seen": dict(Counter(per_run_bucket)),
        "quality_unanimous": len(set(per_run_quality)) == 1,
        "bucket_unanimous": len(set(per_run_bucket)) == 1,
    }
