"""Tests for the partition-aware token accounting helper (issue #136)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_PATH = _SCRIPTS_DIR / "token_accounting.py"
_spec = importlib.util.spec_from_file_location("token_accounting", _PATH)
assert _spec is not None and _spec.loader is not None
ta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ta)


class ComputeTotalTokensTest(unittest.TestCase):
    def test_partition_self_hosted_swe3_not_double_counted(self) -> None:
        # cache_read + cache_write == input_tokens (self-hosted vLLM): cache is a
        # partition of input, so total = input + output.
        total = ta.compute_total_tokens_processed(4_141_291, 23_657, 4_070_208, 71_083)
        self.assertEqual(total, 4_141_291 + 23_657)

    def test_partition_within_tolerance(self) -> None:
        # kimi-style run: cache_read + cache_write is 0.58% above input -- still
        # inside the 5% band, so treated as a partition.
        total = ta.compute_total_tokens_processed(
            11_629_254, 50_000, 11_500_000, 196_640
        )
        self.assertEqual(total, 11_629_254 + 50_000)

    def test_additive_self_hosted_swe2(self) -> None:
        # input tiny, cache huge (pi/swe2 self-hosted): cache is additive.
        total = ta.compute_total_tokens_processed(167_435, 50_000, 6_934_001, 0)
        self.assertEqual(total, 167_435 + 50_000 + 6_934_001)

    def test_additive_bedrock_prompt_cache(self) -> None:
        # Bedrock prompt caching: input ~2, cache ~180K -- additive, must be kept.
        total = ta.compute_total_tokens_processed(2, 1_000, 180_000, 500)
        self.assertEqual(total, 2 + 1_000 + 180_000 + 500)

    def test_no_cache_is_input_plus_output(self) -> None:
        # claude-code self-hosted: cache fields are zero, input holds everything.
        total = ta.compute_total_tokens_processed(6_955_183, 100_000, 0, 0)
        self.assertEqual(total, 6_955_183 + 100_000)

    def test_none_values_treated_as_zero(self) -> None:
        total = ta.compute_total_tokens_processed(None, None, None, None)  # type: ignore[arg-type]
        self.assertEqual(total, 0)

    def test_declared_disjoint_keeps_cache_at_a_50_percent_hit_rate(self) -> None:
        # codex reports the total prompt with the cache as subsets, and the
        # harness subtracts them out before this call, so the fields are
        # disjoint. At a ~50% cache hit rate fresh input equals the cache sum,
        # which is the detector's partition signature: left to detect, the total
        # would lose the whole 50_000-token cache read and halve the derived
        # cost (issue #183).
        detected = ta.compute_total_tokens_processed(50_000, 900, 50_000, 0)
        self.assertEqual(detected, 50_000 + 900)
        declared = ta.compute_total_tokens_processed(
            50_000, 900, 50_000, 0, cache_partition=False
        )
        self.assertEqual(declared, 50_000 + 900 + 50_000)

    def test_declared_partition_overrides_an_additive_signature(self) -> None:
        total = ta.compute_total_tokens_processed(
            2, 1_000, 180_000, 500, cache_partition=True
        )
        self.assertEqual(total, 2 + 1_000)


class CachePartitionForAgentTest(unittest.TestCase):
    def test_codex_counts_are_declared_disjoint(self) -> None:
        self.assertIs(ta.cache_partition_for_agent("codex"), False)
        self.assertIs(ta.cache_partition_for_agent("CODEX"), False)

    def test_other_agents_are_detected_from_the_data(self) -> None:
        for agent in ("claude", "pi", "omp", "kiro", "", None):
            self.assertIsNone(ta.cache_partition_for_agent(agent))


class IsCachePartitionTest(unittest.TestCase):
    def test_exact_partition(self) -> None:
        self.assertTrue(ta._is_cache_partition_of_input(1000, 1000))

    def test_additive_is_not_partition(self) -> None:
        self.assertFalse(ta._is_cache_partition_of_input(5, 2050))

    def test_zero_cache_is_not_partition(self) -> None:
        self.assertFalse(ta._is_cache_partition_of_input(1000, 0))

    def test_zero_input_is_not_partition(self) -> None:
        self.assertFalse(ta._is_cache_partition_of_input(0, 1000))


if __name__ == "__main__":
    unittest.main()
