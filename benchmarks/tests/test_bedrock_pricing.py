"""Tests for the Bedrock price table used to derive codex run costs.

codex exec reports token counts but no billed cost, so the harness prices a
run locally. These cover the rate lookup (including inference-profile
prefixes) and the fresh-vs-cached token contract the caller must honour.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from bedrock_pricing import PRICES, PRICES_AS_OF, cost_usd  # noqa: E402


class CostUsdTest(unittest.TestCase):
    def test_known_model_prices_each_token_class(self) -> None:
        # terra: input 4.00, output 18.00, cache_read 0.40, cache_write 5.00.
        cost = cost_usd(
            "openai.gpt-5.6-terra",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=1_000_000,
            cache_write_tokens=1_000_000,
        )
        self.assertAlmostEqual(cost, 4.00 + 18.00 + 0.40 + 5.00, places=6)

    def test_unknown_model_returns_none_not_zero(self) -> None:
        # A silent 0 would look like a free run on the cost/quality frontier.
        self.assertIsNone(cost_usd("not-a-model", 100, 100))

    def test_inference_profile_prefix_is_stripped(self) -> None:
        bare = cost_usd("openai.gpt-5.6-luna", 1000, 1000)
        for prefix in ("us.", "global.", "eu.", "ap."):
            self.assertEqual(cost_usd(f"{prefix}openai.gpt-5.6-luna", 1000, 1000), bare)

    def test_zero_tokens_costs_nothing(self) -> None:
        self.assertEqual(cost_usd("openai.gpt-5.6-luna", 0, 0, 0, 0), 0.0)

    def test_cache_read_is_cheaper_than_fresh_input(self) -> None:
        # The whole point of passing fresh (non-cached) input separately.
        fresh = cost_usd("openai.gpt-5.6-terra", 1_000_000, 0)
        cached = cost_usd("openai.gpt-5.6-terra", 0, 0, cache_read_tokens=1_000_000)
        assert fresh is not None and cached is not None
        self.assertLess(cached, fresh)

    def test_price_table_rows_are_complete(self) -> None:
        for model, rates in PRICES.items():
            for key in ("input", "output", "cache_read", "cache_write"):
                self.assertIn(key, rates, f"{model} missing {key}")
                self.assertGreaterEqual(rates[key], 0.0)

    def test_prices_carry_an_as_of_date(self) -> None:
        # Rates move; an undated table cannot be audited against the source.
        self.assertRegex(PRICES_AS_OF, r"^\d{4}-\d{2}-\d{2}$")


if __name__ == "__main__":
    unittest.main()
