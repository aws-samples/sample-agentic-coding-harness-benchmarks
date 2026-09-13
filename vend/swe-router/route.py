#!/usr/bin/env python3
"""Pick a model, given a quality floor and a task difficulty tier.

The skill decides two things by reading the repo: how good the output has to be
(the floor) and how hard the task is (the tier). This does the rest, so the
selection is arithmetic rather than something a language model works out in
prose and might get wrong.

    python3 route.py --floor 70 --tier high

It prints JSON: the recommendation, the runners-up, and why anything was
excluded. Standard library only, so it runs wherever the skill is installed.

The rule, in full:

1. Start from every model in ``models.json``.
2. Drop anything the organisation does not allow, if ``allowed-models.txt`` is
   present beside this file, at the repository root, or in ``.claude/``. That
   file is also the availability list: whoever maintains it is saying every
   entry is a model the developer's agent can select.
3. Drop anything ``--available`` leaves out, if it was given. It rarely is --
   it exists for the case where a developer says one listed model is not
   wired up for them.
4. Read each survivor's score at the task's tier -- not its overall mean, which
   is a ranking that holds at no tier.
5. Keep the ones at or above the floor and take the cheapest. Models within
   ``--tie-band`` points of each other are treated as equal, because with five
   or six tasks per tier a smaller gap does not survive dropping one task.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent

# Five or six tasks per tier, one run each. Dropping a single task reverses the
# order of two models 82% of the time when they sit within 1 point, 53% within
# 2, and 47% within 3 -- but only 5% past 5 points. Three is where a stated
# ordering stops being a coin flip.
DEFAULT_TIE_BAND = 3.0

TIERS = ("trivial", "low", "medium", "high")

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
    if major != "1":
        raise RouteError(
            f"models.json is schema {data.get('schema_version')}; this router "
            "understands major version 1. Update the router."
        )
    return data


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


def select(
    models: list[dict],
    *,
    tier: str,
    floor: float,
    tie_band: float = DEFAULT_TIE_BAND,
) -> dict[str, Any]:
    """Choose the cheapest candidate whose score at ``tier`` clears ``floor``.

    Args:
        models: Candidate model entries, already filtered.
        tier: One of TIERS.
        floor: Minimum acceptable score at that tier.
        tie_band: Score difference treated as no difference.

    Returns:
        A dict with the pick, the models that cleared the floor, and those that
        did not.
    """
    scored, unscored = [], []
    for m in models:
        score = (m.get("score_by_complexity") or {}).get(tier)
        if score is None:
            # No measurement at this tier; fall back to the overall mean and
            # record that, rather than dropping the model without saying so.
            score = m.get("score")
            if score is None:
                continue
            unscored.append(m["model"])
        scored.append((score, m))

    clears = [(s, m) for s, m in scored if s >= floor]
    misses = sorted(((s, m) for s, m in scored if s < floor), key=lambda x: -x[0])

    pick = None
    tied = []
    if clears:
        cheapest = min(clears, key=lambda x: (x[1]["cost_per_task_usd"], -x[0]))
        # Anything within the tie band AND no more expensive is equally good;
        # prefer the higher score among those, since the cost is the same or
        # better and the score difference is not measurable either way.
        band = [
            (s, m)
            for s, m in clears
            if m["cost_per_task_usd"] <= cheapest[1]["cost_per_task_usd"]
            and abs(s - cheapest[0]) <= tie_band
        ]
        pick = max(band, key=lambda x: x[0]) if band else cheapest
        tied = [m["model"] for s, m in band if m["model"] != pick[1]["model"]]

    def row(entry: tuple[float, dict]) -> dict[str, Any]:
        score, m = entry
        completion = (m.get("completion_by_complexity") or {}).get(tier)
        done, total = (
            (completion.split("/") + [None])[:2] if completion else (None, None)
        )
        return {
            "model": m["model"],
            "score": round(score, 2),
            "cost_per_task_usd": m["cost_per_task_usd"],
            "completion": completion,
            "finished_every_task": completion is None or done == total,
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
        "scored_from_overall_mean": unscored,
    }


def route(
    *,
    tier: str,
    floor: float,
    available: list[str] | None,
    models_path: Path,
    aliases_path: Path,
    allowed_file: Path | None,
    no_allow_list: bool,
    tie_band: float,
) -> dict[str, Any]:
    """Run the whole selection and return a result the caller can print.

    Raises:
        RouteError: On unusable inputs.
    """
    if tier not in TIERS:
        raise RouteError(f"--tier must be one of {', '.join(TIERS)}, got {tier!r}")

    data = load_models(models_path)
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
        "tier": tier,
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

    result.update(select(candidates, tier=tier, floor=floor, tie_band=tie_band))
    result["status"] = "ok" if result["recommended"] else "nothing_clears_floor"
    if result["status"] == "nothing_clears_floor":
        best = result["below_floor"][0] if result["below_floor"] else None
        result["reason"] = (
            f"nothing available reaches {floor} on {tier} tasks; closest is "
            f"{best['model']} at {best['score']}, short by "
            f"{round(floor - best['score'], 2)}"
            if best
            else f"nothing available has a score for {tier} tasks"
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
        description="Pick a model given a quality floor and a difficulty tier.",
    )
    p.add_argument("--tier", required=True, choices=TIERS)
    p.add_argument("--floor", required=True, type=float)
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
    p.add_argument("--tie-band", type=float, default=DEFAULT_TIE_BAND)
    p.add_argument("--models", type=Path, default=_HERE / "models.json")
    p.add_argument("--aliases", type=Path, default=_HERE / "model-aliases.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the router and print JSON. Returns a process exit code."""
    args = _parse_args(argv)
    try:
        result = route(
            tier=args.tier,
            floor=args.floor,
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
