#!/usr/bin/env python3
"""Single-source EC2 pricing lookup for cost-per-token / cost-per-task math.

Reads ``self-hosted/vllm/pricing.json`` so no script hardcodes a dollar figure.
``dollars_per_hour`` IS the rate a run is charged, prorated by
``tp / gpus_per_instance`` when a model is served with tensor parallelism below
the instance's GPU count (it uses only part of the box) -- e.g. a TP=4 model on
an 8-GPU p5en.48xlarge is charged half the instance rate.

There is deliberately NO discount multiplier. pricing.json used to carry one, and
p5en was based at on-demand with a 0.35 PLACEHOLDER discount: a guess sitting in
the same field, and flowing into the same charts, as measured prices. Every
instance is now based at its real published 3-year commitment rate, with the
alternatives recorded in that entry's ``rates`` map for reference only. To price
at a different term, move the value into ``dollars_per_hour`` and regenerate the
summaries; nothing multiplies it on the way through.

Usage:
    from pricing import resolve, instance_hourly
    rate = resolve("p5en.48xlarge", tp=4)   # half-box $/hr
    full = instance_hourly("g6e.12xlarge")  # whole-instance $/hr
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# pricing.json lives at the vLLM dir root: clients/ -> parent.
_PRICING_PATH = Path(__file__).resolve().parent.parent / "pricing.json"


class PricingError(ValueError):
    """Raised when an instance type is missing from pricing.json."""


@lru_cache(maxsize=1)
def _load() -> dict:
    """Load and cache pricing.json."""
    if not _PRICING_PATH.is_file():
        raise PricingError(f"pricing.json not found at {_PRICING_PATH}")
    return json.loads(_PRICING_PATH.read_text(encoding="utf-8"))


def _entry(instance_type: str) -> dict:
    """Return the pricing.json entry for ``instance_type`` or raise."""
    instances = _load().get("instances", {})
    entry = instances.get(instance_type)
    if entry is None:
        raise PricingError(
            f"instance '{instance_type}' not in pricing.json "
            f"(have: {sorted(instances)}). Add it there, do not hardcode a rate."
        )
    return entry


def _effective_full(entry: dict) -> float:
    """Whole-instance $/hr: ``dollars_per_hour`` exactly as published.

    Nothing is applied on top. A ``discount`` key here would silently change
    every derived cost in the repo, so reject it rather than honour or ignore it:
    the rate must be a real published price, not a base times an assumption.

    Raises:
        PricingError: If the entry still carries a ``discount`` key.
    """
    if "discount" in entry:
        raise PricingError(
            "pricing.json entry carries a 'discount' key, which is no longer "
            "supported: set dollars_per_hour to the real published rate and "
            "record the alternatives under 'rates'."
        )
    return float(entry["dollars_per_hour"])


def instance_hourly(instance_type: str) -> float:
    """Return the whole-instance $/hr for ``instance_type``.

    This is ``dollars_per_hour`` verbatim: the entry's published 3-year
    commitment rate, with nothing applied on top.

    Args:
        instance_type: e.g. ``g6e.12xlarge`` or ``p5en.48xlarge``.

    Returns:
        Effective dollars per hour for the full instance.

    Raises:
        PricingError: If the instance type is not in pricing.json.
    """
    return _effective_full(_entry(instance_type))


def resolve(instance_type: str, tp: int | None = None) -> float:
    """Return the effective $/hr for a run, accounting for a partial-box TP.

    A model served at ``tp`` GPUs on an instance with more GPUs uses only that
    fraction of the box, so it is charged ``dollars_per_hour * tp /
    gpus_per_instance``. With ``tp`` unset (or >= the instance's GPU count) the
    full-instance rate is returned.

    Args:
        instance_type: The EC2 instance type.
        tp: Tensor-parallel size (GPUs the model actually used). None = full box.

    Returns:
        Effective dollars per hour to attribute to this model's run.

    Raises:
        PricingError: If the instance type is not in pricing.json, or its entry
            still carries an unsupported ``discount`` key.
    """
    entry = _entry(instance_type)
    full = _effective_full(entry)
    gpus = int(entry.get("gpus_per_instance") or 0)
    if tp is None or gpus <= 0 or tp >= gpus:
        return full
    return full * tp / gpus
