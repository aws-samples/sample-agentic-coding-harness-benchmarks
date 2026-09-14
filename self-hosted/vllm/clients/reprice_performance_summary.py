#!/usr/bin/env python3
"""Reprice an existing performance-summary.json against current pricing.json.

``build_performance_summary`` derives cost from the sweep's DuckDB, so the honest
way to change a rate is to rebuild from the DB. That is not always possible: the
DuckDBs are gitignored and local, so a sweep run months ago on a since-terminated
instance leaves only its committed summary. Rebuilding is therefore unavailable
for most arms, while the rate they were priced at is now wrong.

Repricing is exact rather than approximate, which is what makes this safe. Every
cost field in the summary is a linear function of the hourly rate:

    cost_per_token = ($/hr / 3600) / tokens_per_second

The measured quantity is ``tokens_per_second``, and it does not depend on price.
So scaling every cost by ``new_rate / old_rate`` yields exactly what a rebuild
from the same DuckDB would produce. Nothing measured is touched: throughput,
TTFT, TPOT, queue time, KV-cache occupancy and the concurrency levels are copied
through byte for byte.

The one thing that can change shape is which level is *cheapest*, and it cannot:
a single positive scale factor preserves ordering, so ``min_*`` fields are
recomputed from the scaled levels and must still land on the same level. The tool
asserts that, so a bug in the scaling cannot quietly move the headline.

TP is inferred from the recorded rate against pricing.json's old rate for the
instance, then verified to be a whole number of GPUs; pass ``--tp`` to override.

Usage:
    # Reprice one arm (dry run first -- prints the diff, writes nothing):
    uv run python -m clients.reprice_performance_summary \
        --summary benchmark-output/throughput/glm-5.2/performance-summary.json \
        --old-rate-full 41.1424 --dry-run

    # Reprice every p5en arm that was priced at the old placeholder rate:
    uv run python -m clients.reprice_performance_summary \
        --summary 'benchmark-output/throughput/*/performance-summary.json' \
        --instance p5en.48xlarge --old-rate-full 41.1424
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Any

from .pricing import instance_hourly, resolve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Every per-level field that is a linear function of $/hr. Anything not listed is
# a measurement (or a weight) and is copied through untouched -- so a field added
# to build_performance_summary later is left ALONE rather than silently scaled,
# which is the safe direction to fail.
_LEVEL_COST_KEYS = (
    "blended_cost_per_token_usd",
    "split_cost_per_input_token_usd",
    "split_cost_per_output_token_usd",
    "blended_cost_per_1m_tokens_usd",
    "split_cost_per_1m_output_tokens_usd",
    "split_cost_per_1m_input_tokens_usd",
    "task_cost_blended_usd",
    "task_cost_split_usd",
)

# The summary's ``min_*`` fields are, by build_performance_summary's own
# definition, the minimum of a per-level field taken verbatim. So they are
# RECOMPUTED from the scaled levels rather than scaled themselves: scaling an
# already-rounded aggregate compounds its rounding error and can leave the
# headline disagreeing with the level it is supposed to have come from.
_MIN_FIELD_SOURCES = {
    "min_blended_cost_per_1m_tokens_usd": "blended_cost_per_1m_tokens_usd",
    "min_task_cost_blended_usd": "task_cost_blended_usd",
    "min_task_cost_split_usd": "task_cost_split_usd",
}

# Dollar-denominated display fields are rounded; per-token fields stay at full
# precision, as the builder leaves them. Rounding to a fixed 4 places (rather
# than reproducing each field's original precision) keeps the reprice auditable:
# divide any new figure by the old one and you get the factor back.
_ROUND = {
    "blended_cost_per_1m_tokens_usd": 4,
    "split_cost_per_1m_output_tokens_usd": 4,
    "split_cost_per_1m_input_tokens_usd": 4,
    "task_cost_blended_usd": 4,
    "task_cost_split_usd": 4,
}


class RepriceError(RuntimeError):
    """Raised when a summary cannot be repriced safely."""


def _infer_tp(old_full_rate: float, recorded_rate: float, gpus: int) -> int:
    """Infer the TP a summary was priced at, or raise if it is not a whole box share.

    ``recorded_rate = old_full_rate * tp / gpus``, so ``tp`` follows. A fractional
    result means the recorded rate did not come from this full rate, which makes
    the whole reprice unsound -- so raise instead of rounding.
    """
    if old_full_rate <= 0:
        raise RepriceError(f"old full rate must be positive, got {old_full_rate}")
    raw = recorded_rate * gpus / old_full_rate
    tp = round(raw)
    if abs(raw - tp) > 1e-6 or not 1 <= tp <= gpus:
        raise RepriceError(
            f"recorded rate {recorded_rate} is not a whole-GPU share of "
            f"{old_full_rate} across {gpus} GPUs (implied tp={raw:.4f}). "
            f"Pass --tp explicitly if you know the right value."
        )
    return tp


def _is_share_of(full_rate: float, recorded_rate: float, gpus: int) -> bool:
    """True if ``recorded_rate`` is a whole-GPU share of ``full_rate``.

    Used to detect an already-repriced file so the tool is idempotent over a glob
    that mixes finished and unfinished arms.
    """
    if full_rate <= 0:
        return False
    raw = float(recorded_rate) * gpus / full_rate
    return abs(raw - round(raw)) < 1e-6 and 1 <= round(raw) <= gpus


def _scale(value: Any, factor: float, key: str) -> Any:
    """Scale one numeric cost field, preserving the summary's rounding."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    scaled = float(value) * factor
    digits = _ROUND.get(key)
    return round(scaled, digits) if digits is not None else scaled


def reprice(summary: dict, new_rate: float, factor: float) -> dict:
    """Return a repriced copy of ``summary``: costs scaled, measurements intact.

    Args:
        summary: The parsed performance-summary.json.
        new_rate: The rate to record in ``dollars_per_hour``.
        factor: ``new_rate / old_rate``.

    Returns:
        A new dict. The input is not mutated.

    Raises:
        RepriceError: If scaling moved which concurrency level is cheapest, which
            a single positive factor cannot do and so indicates a bug.
    """
    out = dict(summary)
    out["dollars_per_hour"] = round(new_rate, 4)

    levels_in = summary.get("levels") or []
    cheapest_before = _cheapest_level(levels_in)

    levels_out = []
    for level in levels_in:
        new_level = dict(level)
        for key in _LEVEL_COST_KEYS:
            if key in new_level:
                new_level[key] = _scale(new_level[key], factor, key)
        levels_out.append(new_level)
    out["levels"] = levels_out

    cheapest_after = _cheapest_level(levels_out)
    if cheapest_before != cheapest_after:
        raise RepriceError(
            f"scaling moved the cheapest level from c={cheapest_before} to "
            f"c={cheapest_after}; a single positive factor cannot do that"
        )

    for key, source in _MIN_FIELD_SOURCES.items():
        if key not in out:
            continue
        candidates = [
            lvl[source]
            for lvl in levels_out
            if isinstance(lvl.get(source), (int, float)) and lvl[source]
        ]
        out[key] = min(candidates) if candidates else None

    # Record the basis so a repriced file is self-describing rather than looking
    # like a fresh build at a rate whose DuckDB no longer exists.
    out["_repriced"] = {
        "from_dollars_per_hour": round(summary.get("dollars_per_hour", 0.0), 4),
        "to_dollars_per_hour": round(new_rate, 4),
        "factor": round(factor, 6),
        "basis": (
            "Costs rescaled linearly from the rate above; every measured "
            "quantity (throughput, TTFT, TPOT, queue, KV) is unchanged from the "
            "original sweep. Equivalent to rebuilding from the same DuckDB at "
            "the new rate. See pricing.json for the rate basis."
        ),
    }
    return out


def _cheapest_level(levels: list[dict]) -> int | None:
    """Return the concurrency of the cheapest blended level, or None."""
    priced = [
        lvl
        for lvl in levels
        if isinstance(lvl.get("blended_cost_per_token_usd"), (int, float))
    ]
    if not priced:
        return None
    return min(priced, key=lambda lvl: lvl["blended_cost_per_token_usd"]).get(
        "concurrency"
    )


def reprice_file(
    path: Path,
    old_full_rate: float,
    instance_filter: str | None,
    tp_override: int | None,
    dry_run: bool,
) -> bool:
    """Reprice one summary file in place. Returns True if it was (or would be) changed."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    instance = summary.get("instance_type")
    if instance_filter and instance != instance_filter:
        logger.info("  skip %s: instance %s", path.parent.name, instance)
        return False
    recorded = summary.get("dollars_per_hour")
    if not isinstance(recorded, (int, float)):
        logger.warning("  skip %s: no dollars_per_hour", path.parent.name)
        return False

    gpus = _gpus_for(instance)
    # Check "already repriced" BEFORE inferring TP against the old rate, or a
    # second run would fail: a rate that is a whole-GPU share of the NEW full
    # rate is generally a fractional share of the old one. The tool has to be
    # safe to re-run over a glob that mixes done and not-yet-done arms.
    if tp_override is None and _is_share_of(instance_hourly(instance), recorded, gpus):
        logger.info("  skip %s: already at $%.4f/hr", path.parent.name, float(recorded))
        return False

    tp = tp_override or _infer_tp(old_full_rate, float(recorded), gpus)
    new_rate = resolve(instance, tp)
    if abs(new_rate - float(recorded)) < 1e-9:
        logger.info("  skip %s: already at $%.4f/hr", path.parent.name, new_rate)
        return False

    factor = new_rate / float(recorded)
    repriced = reprice(summary, new_rate, factor)
    logger.info(
        "  %-30s tp=%d  $%.4f -> $%.4f/hr (x%.4f)  $/1M %.2f -> %.2f  $/task %.4f -> %.4f",
        path.parent.name,
        tp,
        recorded,
        new_rate,
        factor,
        summary.get("min_blended_cost_per_1m_tokens_usd", float("nan")),
        repriced.get("min_blended_cost_per_1m_tokens_usd", float("nan")),
        summary.get("min_task_cost_blended_usd", float("nan")),
        repriced.get("min_task_cost_blended_usd", float("nan")),
    )
    if not dry_run:
        path.write_text(json.dumps(repriced, indent=2) + "\n", encoding="utf-8")
    return True


def _gpus_for(instance: str) -> int:
    """Return gpus_per_instance for ``instance`` from pricing.json."""
    from .pricing import _entry  # noqa: PLC0415 - internal lookup, same module family

    gpus = int(_entry(instance).get("gpus_per_instance") or 0)
    if gpus <= 0:
        raise RepriceError(f"instance {instance} has no gpus_per_instance")
    return gpus


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Path or glob to performance-summary.json file(s)",
    )
    parser.add_argument(
        "--old-rate-full",
        type=float,
        required=True,
        help="The FULL-INSTANCE rate the summaries were priced at (e.g. 41.1424), "
        "used to infer each arm's TP from its recorded partial-box rate",
    )
    parser.add_argument(
        "--instance",
        default=None,
        help="Only reprice summaries whose instance_type matches (e.g. p5en.48xlarge)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Override the inferred TP (applies to every matched file)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change and write nothing",
    )
    return parser.parse_args()


def main() -> None:
    """Reprice every matched summary against current pricing.json."""
    args = _parse_args()
    paths = sorted(Path(p) for p in glob.glob(args.summary))
    if not paths:
        raise SystemExit(f"no files matched {args.summary!r}")
    logger.info(
        "repricing %d summary file(s)%s%s",
        len(paths),
        f" on {args.instance}" if args.instance else "",
        " (DRY RUN)" if args.dry_run else "",
    )
    changed = sum(
        reprice_file(p, args.old_rate_full, args.instance, args.tp, args.dry_run)
        for p in paths
    )
    logger.info(
        "%s %d of %d file(s)",
        "would reprice" if args.dry_run else "repriced",
        changed,
        len(paths),
    )


if __name__ == "__main__":
    main()
