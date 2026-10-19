#!/usr/bin/env python3
"""Pick a model for a coding task from six rubric scores.

The skill reads the repository and the task and scores the six questions in
``rubric.md``, 0 to 2 each. This does the rest, so the selection is arithmetic
rather than something a language model works out in prose and might get wrong:

    python3 route.py --rubric '{"hidden_contract": 1, "silent_invariant": 0,
        "coordinated_surfaces": 2, "ambiguity": 1, "verification": 1,
        "blast_radius": 1}' --confidence high

It prints JSON: the recommendation, the runners-up, and why anything was
excluded. Standard library only, so it runs wherever the skill is installed.

The rule, in full:

1. Turn the scores into two numbers with ``routing-policy.json``. Four
   difficulty scores add up to a **bucket** (the task's difficulty level:
   easy, moderate, hard, very hard). ``blast_radius`` and ``verification`` set
   the **required quality**, the score a model must reach.
2. If the scorer reported low confidence, stop and keep the developer on the
   model they have.
3. Start from every model in ``models.json``. Drop anything the organisation
   does not allow, if ``allowed-models.txt`` is present beside this file, at
   the repository root, or in ``.claude/``. That file is also the availability
   list: whoever maintains it is saying every entry is a model the developer's
   agent can select. ``--available`` narrows it further, rarely.
4. Read each survivor's score in the task's bucket -- not its overall mean.
   A model with no benchmark task in that bucket is read from the next harder
   bucket it has, which can only understate it.
5. Skip a model that failed more of that bucket's benchmark tasks than the
   policy's ``min_completion_in_bucket`` allows. A column mean leaves failures
   out, so this is the only place they count.
6. Keep the ones at or above the required quality and take the cheapest.
   Models within the tie band of each other are treated as equal, because with
   a handful of tasks per bucket a smaller gap does not survive dropping one.

``--tier`` and ``--floor`` still work, reading the old per-tier columns, for
one release.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Fail with a message the skill can read out, not a traceback, on a Python too
# old for this file. The installer checks the same floor.
if sys.version_info < (3, 8):  # pragma: no cover - depends on the interpreter
    print('{"status": "error", "error": "route.py needs Python 3.8 or later"}')
    sys.exit(2)

_HERE = Path(__file__).resolve().parent

# Five or six tasks per tier, one run each. Dropping a single task reverses the
# order of two models 82% of the time when they sit within 1 point, 53% within
# 2, and 47% within 3 -- but only 5% past 5 points. Three is where a stated
# ordering stops being a coin flip.
DEFAULT_TIE_BAND = 3.0

TIERS = ("trivial", "low", "medium", "high")

# The rubric. Kept in step with rubric.md, which the skill's model reads.
DIMENSIONS = (
    "hidden_contract",
    "silent_invariant",
    "coordinated_surfaces",
    "ambiguity",
    "verification",
    "blast_radius",
)
SCORES = (0, 1, 2)
BUCKETS = ("easy", "moderate", "hard", "very_hard")
CONFIDENCE_LEVELS = ("high", "medium", "low")
DEFAULT_POLICY = _HERE / "routing-policy.json"
SCHEMA_MAJORS = ("1", "2")

# Where an organisation's allow-list might sit, nearest the developer first: a
# team overriding the shipped default is the point of having one.
ALLOW_LIST_NAMES = ("allowed-models.txt",)
ALLOW_LIST_DIRS = (Path.cwd(), Path.cwd() / ".claude", _HERE)


class RouteError(Exception):
    """Raised when the inputs or the data files are unusable."""


def normalize(name: str) -> str:
    """Reduce a model name to a comparison key.

    Assistants, policy files and inference profiles all spell the same model
    differently. This strips the parts that are routing or packaging rather
    than identity.

    Args:
        name: A model name from any source.

    Returns:
        A lowercase key with separators and decoration removed.
    """
    n = name.strip().lower()
    n = re.sub(r"\[[^\]]*\]$", "", n)  # [1m] context-window hint
    n = re.sub(r"-\d{8}-v\d+:\d+$", "", n)  # -20251001-v1:0 snapshot stamp
    n = n.rsplit("/", 1)[-1]  # anthropic/ provider prefix
    n = re.sub(r"^[a-z]{2}\.[a-z0-9-]+\.", "", n)  # us.anthropic. region prefix
    return re.sub(r"[\s_.-]", "", n)


def load_models(path: Path) -> dict[str, Any]:
    """Load the vended measurements.

    Args:
        path: Path to models.json.

    Returns:
        The parsed payload.

    Raises:
        RouteError: If the file is missing, unparseable, or a schema major this
            code does not understand.
    """
    if not path.is_file():
        raise RouteError(f"no models.json at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RouteError(f"{path} is not valid JSON: {exc}") from exc
    major = str(data.get("schema_version", "")).split(".", 1)[0]
    if major not in SCHEMA_MAJORS:
        raise RouteError(
            f"models.json is schema {data.get('schema_version')}; this router "
            f"understands major versions {', '.join(SCHEMA_MAJORS)}. Update the router."
        )
    return data


def load_policy(path: Path) -> dict[str, Any]:
    """Load and check the routing policy.

    Args:
        path: Path to routing-policy.json.

    Returns:
        The parsed policy.

    Raises:
        RouteError: If the file is missing, unparseable, or does not define a
            quality for every blast radius and a range for every bucket.
    """
    if not path.is_file():
        raise RouteError(f"no routing policy at {path}")
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RouteError(f"{path} is not valid JSON: {exc}") from exc
    try:
        by_radius = policy["required_quality"]["by_blast_radius"]
        dims = policy["difficulty"]["dimensions"]
        buckets = policy["difficulty"]["buckets"]
    except (KeyError, TypeError) as exc:
        raise RouteError(f"{path} is missing {exc}") from exc
    if sorted(by_radius) != [str(v) for v in SCORES]:
        raise RouteError(f"{path}: by_blast_radius must define 0, 1 and 2")
    if not set(dims) <= set(DIMENSIONS):
        raise RouteError(f"{path}: unknown difficulty dimension in {dims}")
    if list(buckets) != list(BUCKETS):
        raise RouteError(f"{path}: buckets must be {', '.join(BUCKETS)}, in order")
    # A platform team edits this file by hand, so check the shapes the
    # arithmetic relies on and say which entry is wrong.
    for name, edges in buckets.items():
        if not (
            isinstance(edges, list)
            and len(edges) == 2
            and all(isinstance(e, int) and not isinstance(e, bool) for e in edges)
        ):
            raise RouteError(f"{path}: bucket {name} must be [low, high] integers")
    numbers = [
        *by_radius.values(),
        policy["required_quality"].get("add_when_verification_is_2", 0),
    ]
    if not all(
        isinstance(n, (int, float)) and not isinstance(n, bool) for n in numbers
    ):
        raise RouteError(f"{path}: required quality values must be numbers")
    return policy


def parse_rubric(text: str) -> dict[str, int]:
    """Read the six rubric scores from JSON.

    Accepts ``{"hidden_contract": 1, ...}``, the same with each value as
    ``{"score": 1, "evidence": "..."}``, or either wrapped in ``{"scores": ...}``.

    Args:
        text: The JSON the skill passed.

    Returns:
        Dimension -> score.

    Raises:
        RouteError: Unless all six dimensions are present, each 0, 1 or 2.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RouteError(f"--rubric is not valid JSON: {exc}") from exc
    if isinstance(data, dict) and isinstance(data.get("scores"), dict):
        data = data["scores"]
    if not isinstance(data, dict) or set(data) != set(DIMENSIONS):
        raise RouteError(f"--rubric must score exactly: {', '.join(DIMENSIONS)}")
    scores = {}
    for dim in DIMENSIONS:
        value = data[dim]
        if isinstance(value, dict):
            value = value.get("score")
        if isinstance(value, bool) or value not in SCORES:
            raise RouteError(f"--rubric {dim} must be 0, 1 or 2, got {value!r}")
        scores[dim] = int(value)
    return scores


def required_quality(scores: dict[str, int], policy: dict[str, Any]) -> float:
    """The score a model must reach: blast radius, plus a margin when untested."""
    rq = policy["required_quality"]
    base = float(rq["by_blast_radius"][str(scores["blast_radius"])])
    extra = float(rq.get("add_when_verification_is_2", 0))
    return base + (extra if scores["verification"] == 2 else 0.0)


def difficulty_sum(scores: dict[str, int], policy: dict[str, Any]) -> int:
    """Sum of the policy's difficulty dimensions."""
    return sum(scores[d] for d in policy["difficulty"]["dimensions"])


def bucket_for(scores: dict[str, int], policy: dict[str, Any]) -> str:
    """The task's difficulty level, from the policy's bucket ranges.

    Raises:
        RouteError: If the policy's ranges leave this sum uncovered.
    """
    total = difficulty_sum(scores, policy)
    for name, (lo, hi) in policy["difficulty"]["buckets"].items():
        if lo <= total <= hi:
            return name
    raise RouteError(f"difficulty sum {total} falls in no bucket of the policy")


def build_alias_index(models: list[dict], aliases_path: Path) -> dict[str, str]:
    """Return a lookup from any spelling of a model to its canonical slug.

    Args:
        models: The model entries from models.json.
        aliases_path: Path to model-aliases.json; missing is not an error, the
            canonical names still resolve.

    Returns:
        Normalized name -> canonical model slug.
    """
    index = {normalize(m["model"]): m["model"] for m in models}
    if aliases_path.is_file():
        try:
            aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return index
        for canonical, names in (aliases.get("aliases") or {}).items():
            for name in names:
                index.setdefault(normalize(name), canonical)
    return index


def find_allow_list(explicit: Path | None) -> Path | None:
    """Return the allow-list to use, or None when there is no policy.

    Args:
        explicit: A path given on the command line, which wins outright.

    Returns:
        The path to read, or None.

    Raises:
        RouteError: If an explicitly named file does not exist. Silently
            ignoring it would apply no policy while the caller believed one was
            in force.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise RouteError(f"--allowed-file {explicit} does not exist")
        return explicit
    for directory in ALLOW_LIST_DIRS:
        for name in ALLOW_LIST_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def parse_allow_list(path: Path) -> list[str]:
    """Return the model names listed in an allow-list file.

    One model per line. Everything from a ``#`` to end of line is a comment,
    and blank lines are skipped, so the file can carry as much explanation as
    its maintainer wants without any of it being mistaken for policy. That last
    property is why this is a plain list rather than markdown: a commented-out
    example cannot accidentally permit a model.

    Args:
        path: The allow-list file.

    Returns:
        The model names, in file order.

    Raises:
        RouteError: If the file names no models. An empty list permits nothing,
            which is almost never what someone meant to write.
    """
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.append(name)
    if not names:
        raise RouteError(
            f"{path} lists no models. An allow-list with no entries permits "
            "nothing; delete the file to permit everything."
        )
    return names


def _completion_rate(completion: str | None) -> float | None:
    """``"9/10"`` -> 0.9; None when the count is missing or malformed."""
    try:
        done, total = (int(x) for x in (completion or "").split("/"))
    except ValueError:
        return None
    return done / total if total else None


def _score_at(
    m: dict, level: str, columns: str, fallback: tuple[str, ...]
) -> tuple[float | None, str | None]:
    """A model's score at ``level``, else at the first ``fallback`` level it has.

    Returns:
        The score and the level it came from, or (None, None).
    """
    table = m.get(columns) or {}
    for candidate in (level, *fallback):
        if table.get(candidate) is not None:
            return table[candidate], candidate
    return None, None


def select(
    models: list[dict],
    *,
    tier: str,
    floor: float,
    tie_band: float = DEFAULT_TIE_BAND,
    columns: str = "score_by_complexity",
    completions: str = "completion_by_complexity",
    fallback: tuple[str, ...] = (),
    min_completion: float = 0.0,
) -> dict[str, Any]:
    """Choose the cheapest candidate whose score at ``tier`` clears ``floor``.

    Args:
        models: Candidate model entries, already filtered.
        tier: The column to read: a difficulty bucket, or a legacy tier.
        floor: Minimum acceptable score there (the required quality).
        tie_band: Score difference treated as no difference.
        columns: The models.json field holding per-level scores.
        completions: The models.json field holding per-level completion.
        fallback: Levels to read, in order, when a model has no score at
            ``tier``. The rubric path passes the harder buckets.
        min_completion: Skip a model that finished less than this share of the
            benchmark tasks at the level its score came from.

    Returns:
        A dict with the pick, the models that cleared the floor, those that
        did not, and those skipped for failing tasks.
    """
    scored, unscored, borrowed, incomplete = [], [], {}, []
    for m in models:
        score, level = _score_at(m, tier, columns, fallback)
        if score is None:
            # No measurement at this level; fall back to the overall mean and
            # record that, rather than dropping the model without saying so.
            score, level = m.get("score"), None
            if score is None:
                continue
            unscored.append(m["model"])
        elif level != tier:
            borrowed[m["model"]] = level
        rate = _completion_rate((m.get(completions) or {}).get(level or tier))
        if rate is not None and rate < min_completion:
            incomplete.append((score, m, level))
            continue
        scored.append((score, m, level))

    clears = [(s, m, lv) for s, m, lv in scored if s >= floor]
    misses = sorted(
        ((s, m, lv) for s, m, lv in scored if s < floor), key=lambda x: -x[0]
    )

    pick = None
    tied = []
    if clears:
        cheapest = min(clears, key=lambda x: (x[1]["cost_per_task_usd"], -x[0]))
        # Anything within the tie band AND no more expensive is equally good;
        # prefer the higher score among those, since the cost is the same or
        # better and the score difference is not measurable either way.
        band = [
            e
            for e in clears
            if e[1]["cost_per_task_usd"] <= cheapest[1]["cost_per_task_usd"]
            and abs(e[0] - cheapest[0]) <= tie_band
        ]
        pick = max(band, key=lambda x: x[0]) if band else cheapest
        tied = [e[1]["model"] for e in band if e[1]["model"] != pick[1]["model"]]

    def row(entry: tuple[float, dict, str | None]) -> dict[str, Any]:
        score, m, level = entry
        completion = (m.get(completions) or {}).get(level or tier)
        rate = _completion_rate(completion)
        return {
            "model": m["model"],
            "score": round(score, 2),
            "cost_per_task_usd": m["cost_per_task_usd"],
            "completion": completion,
            "finished_every_task": rate is None or rate == 1.0,
            # How far above the floor, and whether that margin is bigger than
            # the noise. A model at 71 against a floor of 70 has not reliably
            # cleared it, and saying "clears the floor" flat would oversell it.
            "margin_over_floor": round(score - floor, 2),
            "margin_is_meaningful": (score - floor) > tie_band,
        }

    return {
        "recommended": row(pick) if pick else None,
        "tied_with": tied,
        "cleared_floor": [
            row(e) for e in sorted(clears, key=lambda x: x[1]["cost_per_task_usd"])
        ],
        "below_floor": [row(e) for e in misses],
        "skipped_for_failures": [row(e) for e in incomplete],
        "scored_from_harder_bucket": borrowed,
        "scored_from_overall_mean": unscored,
    }


def _rubric_inputs(
    rubric: dict[str, int], policy_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Turn rubric scores into the bucket, required quality and policy knobs."""
    policy = load_policy(policy_path)
    return policy, {
        "rubric": rubric,
        "difficulty_sum": difficulty_sum(rubric, policy),
        "bucket": bucket_for(rubric, policy),
        "required_quality": required_quality(rubric, policy),
    }


def route(
    *,
    tier: str | None = None,
    floor: float | None = None,
    available: list[str] | None,
    models_path: Path,
    aliases_path: Path,
    allowed_file: Path | None,
    no_allow_list: bool,
    tie_band: float | None = None,
    rubric: dict[str, int] | None = None,
    confidence: str | None = None,
    policy_path: Path = DEFAULT_POLICY,
    min_completion: float | None = None,
) -> dict[str, Any]:
    """Run the whole selection and return a result the caller can print.

    Give either ``rubric`` (the six scores; the policy sets the bucket and the
    required quality) or the legacy ``tier`` and ``floor``.

    Args:
        tier: Legacy: one of TIERS.
        floor: Legacy: the quality floor.
        available: Models to narrow the candidate set to, or None.
        models_path: models.json.
        aliases_path: model-aliases.json.
        allowed_file: Explicit allow-list path, or None to search.
        no_allow_list: Ignore any allow-list.
        tie_band: Points below which two models tie; None takes the policy's
            value on the rubric path and DEFAULT_TIE_BAND otherwise.
        rubric: The six rubric scores.
        confidence: The scorer's confidence; "low" stops the routing when the
            policy says to stay on the current model when unsure.
        policy_path: routing-policy.json.
        min_completion: Override the policy's ``min_completion_in_bucket``.

    Raises:
        RouteError: On unusable inputs.
    """
    head: dict[str, Any] = {}
    select_args: dict[str, Any] = {}
    if rubric is not None:
        if confidence is not None and confidence not in CONFIDENCE_LEVELS:
            raise RouteError(
                f"--confidence must be one of {', '.join(CONFIDENCE_LEVELS)}"
            )
        policy, head = _rubric_inputs(rubric, policy_path)
        tier, floor = head["bucket"], head["required_quality"]
        if tie_band is None:
            tie_band = float(policy.get("tie_band_points", DEFAULT_TIE_BAND))
        if min_completion is None:
            min_completion = float(policy.get("min_completion_in_bucket", 1.0))
        select_args = {
            "columns": "score_by_difficulty",
            "completions": "completion_by_difficulty",
            "fallback": BUCKETS[BUCKETS.index(tier) + 1 :],
        }
        head["confidence"] = confidence
        if confidence == "low" and policy.get("when_unsure") == "stay_on_current_model":
            return {
                **head,
                "status": "unsure",
                "recommended": None,
                "reason": "the rubric scores were marked low confidence, and the "
                "policy keeps the developer on their current model when unsure",
            }
    else:
        if tier not in TIERS:
            raise RouteError(f"--tier must be one of {', '.join(TIERS)}, got {tier!r}")
        if floor is None:
            raise RouteError("--floor is required with --tier")
        head = {"tier": tier}
    tie_band = DEFAULT_TIE_BAND if tie_band is None else tie_band
    select_args["min_completion"] = min_completion or 0.0

    data = load_models(models_path)
    if rubric is not None and not any(
        m.get("score_by_difficulty") for m in data["models"]
    ):
        raise RouteError(
            f"{models_path} has no score_by_difficulty columns; the rubric needs "
            "schema 2 or later. Update the skill."
        )
    all_models = data["models"]
    index = build_alias_index(all_models, aliases_path)

    def resolve(names: list[str]) -> tuple[list[str], list[str]]:
        known, unknown = [], []
        for n in names:
            slug = index.get(normalize(n))
            (known if slug else unknown).append(slug or n)
        return known, unknown

    candidates = list(all_models)
    excluded: dict[str, Any] = {}

    allow_path = None if no_allow_list else find_allow_list(allowed_file)
    if allow_path is not None:
        allowed, unmeasured = resolve(parse_allow_list(allow_path))
        allowed_set = set(allowed)
        excluded["not_allowed"] = [
            m["model"] for m in candidates if m["model"] not in allowed_set
        ]
        excluded["allowed_but_not_measured"] = unmeasured
        candidates = [m for m in candidates if m["model"] in allowed_set]

    if available is not None:
        have, not_measured = resolve(available)
        have_set = set(have)
        excluded["not_available"] = [
            m["model"] for m in candidates if m["model"] not in have_set
        ]
        excluded["available_but_not_measured"] = not_measured
        candidates = [m for m in candidates if m["model"] in have_set]

    result: dict[str, Any] = {
        **head,
        "floor": floor,
        "tie_band": tie_band,
        "allow_list": str(allow_path) if allow_path else None,
        "candidates_considered": [m["model"] for m in candidates],
        "excluded": excluded,
        "provenance": data.get("provenance"),
    }

    if not candidates:
        result["recommended"] = None
        result["status"] = "no_candidates"
        result["reason"] = _empty_reason(excluded, allow_path is not None, available)
        return result

    result.update(
        select(candidates, tier=tier, floor=floor, tie_band=tie_band, **select_args)
    )
    result["status"] = "ok" if result["recommended"] else "nothing_clears_floor"
    if result["status"] == "nothing_clears_floor":
        best = result["below_floor"][0] if result["below_floor"] else None
        skipped = [r["model"] for r in result["skipped_for_failures"]]
        result["reason"] = (
            f"nothing available reaches {floor} on {tier} tasks; closest is "
            f"{best['model']} at {best['score']}, short by "
            f"{round(floor - best['score'], 2)}"
            if best
            else f"nothing available has a score for {tier} tasks"
        ) + (
            f"; skipped for failing {tier} tasks: {', '.join(skipped)}"
            if skipped
            else ""
        )
    # Sanity: the pick must not be beaten on both axes by another candidate.
    if result.get("recommended"):
        p = result["recommended"]
        for other in result["cleared_floor"]:
            if (
                other["model"] != p["model"]
                and other["score"] >= p["score"]
                and other["cost_per_task_usd"] < p["cost_per_task_usd"]
            ):  # pragma: no cover - defensive, select() cannot produce this
                raise RouteError(
                    f"internal error: {p['model']} is dominated by {other['model']}"
                )
    return result


def _empty_reason(
    excluded: dict[str, Any], had_allow_list: bool, available: list[str] | None
) -> str:
    """Explain which filter emptied the candidate set.

    The three causes need different actions from the developer, so a single
    "no models" message would be useless.
    """
    if had_allow_list and excluded.get("allowed_but_not_measured"):
        names = ", ".join(excluded["allowed_but_not_measured"])
        if available is None or not excluded.get("available_but_not_measured"):
            return (
                f"your organisation allows {names}, and this benchmark has "
                "measured none of them. Stay on your current model."
            )
    if available is not None and excluded.get("available_but_not_measured"):
        names = ", ".join(excluded["available_but_not_measured"])
        return (
            f"your assistant offers {names}, none of which this benchmark "
            "measured. Stay on your current model."
        )
    if available is not None and excluded.get("not_available"):
        names = ", ".join(excluded["not_available"][:5])
        return (
            f"models are allowed and measured ({names}) but your assistant "
            "cannot select any of them. Ask whoever runs your platform for "
            "access -- this is not a model switch you can make yourself."
        )
    return "no model satisfies every constraint. Stay on your current model."


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Pick a model for a coding task from six rubric scores.",
        epilog=(
            "Example:\n"
            "  python3 route.py --confidence high --rubric "
            '\'{"hidden_contract": 1, "silent_invariant": 0, '
            '"coordinated_surfaces": 2, "ambiguity": 1, '
            '"verification": 1, "blast_radius": 1}\'\n'
            "Legacy, for one release:\n"
            "  python3 route.py --tier high --floor 70"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--rubric",
        help="The six rubric scores as JSON, each 0, 1 or 2. See rubric.md.",
    )
    p.add_argument(
        "--confidence",
        choices=CONFIDENCE_LEVELS,
        help="How sure the scorer is. 'low' keeps the developer on their model.",
    )
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--tier", choices=TIERS, help="Legacy: a difficulty tier.")
    p.add_argument("--floor", type=float, help="Legacy: a quality floor.")
    p.add_argument(
        "--available",
        help="Comma-separated models to narrow the allow-list further. Omit "
        "unless the developer says a permitted model is not reachable; the "
        "allow-list is already the set of models they can select.",
    )
    p.add_argument("--allowed-file", type=Path, help="Override the allow-list path.")
    p.add_argument(
        "--no-allow-list",
        action="store_true",
        help="Ignore any allow-list on disk.",
    )
    p.add_argument(
        "--tie-band",
        type=float,
        default=None,
        help="Points below which two models tie. Default: the policy's, or "
        f"{DEFAULT_TIE_BAND} with --tier.",
    )
    p.add_argument("--models", type=Path, default=_HERE / "models.json")
    p.add_argument("--aliases", type=Path, default=_HERE / "model-aliases.json")
    args = p.parse_args(argv)
    if args.rubric is None and (args.tier is None or args.floor is None):
        p.error("give --rubric, or the legacy --tier and --floor")
    if args.rubric is not None and (args.tier is not None or args.floor is not None):
        p.error("--rubric sets the bucket and required quality; drop --tier/--floor")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the router and print JSON. Returns a process exit code."""
    args = _parse_args(argv)
    try:
        result = route(
            tier=args.tier,
            floor=args.floor,
            rubric=parse_rubric(args.rubric) if args.rubric is not None else None,
            confidence=args.confidence,
            policy_path=args.policy,
            available=(
                [s.strip() for s in args.available.split(",") if s.strip()]
                if args.available
                else None
            ),
            models_path=args.models,
            aliases_path=args.aliases,
            allowed_file=args.allowed_file,
            no_allow_list=args.no_allow_list,
            tie_band=args.tie_band,
        )
    except RouteError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result.get("recommended") else 1


if __name__ == "__main__":
    sys.exit(main())
