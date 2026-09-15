"""Amazon Bedrock price table and cost helpers for the benchmark harness.

Used by the codex agent path to derive total_cost_usd from token counts,
since codex exec does not report a billed cost itself.

Provenance of the rates:
- Rates were read directly from https://aws.amazon.com/bedrock/pricing/ on 2026-08-31.
- Tier: Global CRIS (cross-region inference, global profile) — the tier used
  by codex exec routing through bedrock-mantle.
- Context window tier: Long Context Window (1M) — confirmed from actual run
  token counts (all tasks exceeded 272K input tokens).
- Cache write is the 30-minute TTL rate.

All prices are per 1M tokens in USD.

Public API:
    cost_usd(model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
    PRICES, PRICES_AS_OF
"""

from __future__ import annotations

PRICES_AS_OF = "2026-08-31"

# USD per 1M tokens — Global CRIS, Long Context Window (1M), Standard tier.
# Sourced from https://aws.amazon.com/bedrock/pricing/ on PRICES_AS_OF.
PRICES: dict[str, dict[str, float]] = {
    # GPT-5.6 Terra — high-capability variant
    "openai.gpt-5.6-terra": {
        "input": 4.00,
        "cache_write": 5.00,
        "cache_read": 0.40,
        "output": 18.00,
    },
    # GPT-5.6 Luna — cost-efficient variant
    "openai.gpt-5.6-luna": {
        "input": 0.40,
        "cache_write": 0.50,
        "cache_read": 0.04,
        "output": 1.80,
    },
}

_PER_1M = 1_000_000.0


def _rates(model: str) -> dict[str, float] | None:
    """Return the price row for a model id, or None if unknown.

    Matching strips any 'us.' or 'global.' inference-profile prefix so both
    'openai.gpt-5.6-terra' and 'us.openai.gpt-5.6-terra' resolve correctly.
    """
    # Strip common inference-profile prefixes
    clean = model
    for prefix in ("us.", "global.", "eu.", "ap."):
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]
            break
    return PRICES.get(clean) or PRICES.get(model)


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Compute total cost in USD for a single run.

    Returns None when the model is not in the price table rather than
    returning a misleading 0.

    Args:
        model: The model id (with or without inference-profile prefix).
        input_tokens: Fresh (non-cached) input tokens.
        output_tokens: Output tokens.
        cache_read_tokens: Tokens served from cache (cache read).
        cache_write_tokens: Tokens written to cache (cache write).

    Returns:
        Total cost in USD, or None if the model is not priced.
    """
    rates = _rates(model)
    if rates is None:
        return None
    total = (
        input_tokens * rates["input"] / _PER_1M
        + output_tokens * rates["output"] / _PER_1M
        + cache_read_tokens * rates["cache_read"] / _PER_1M
        + cache_write_tokens * rates["cache_write"] / _PER_1M
    )
    return round(total, 6)
