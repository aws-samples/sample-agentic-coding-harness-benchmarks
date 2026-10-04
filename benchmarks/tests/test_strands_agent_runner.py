"""Tests for the Strands runner's SDK-free helpers and argument checks.

The SDK is imported lazily inside the runner, so these run without the
optional ``strands`` dependency group installed.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import strands_agent_runner as runner  # noqa: E402


class UsageDictTest(unittest.TestCase):
    def test_missing_keys_default_to_zero(self) -> None:
        usage = runner._usage_dict({"inputTokens": 5})
        self.assertEqual(
            usage,
            {
                "inputTokens": 5,
                "outputTokens": 0,
                "totalTokens": 0,
                "cacheReadInputTokens": 0,
                "cacheWriteInputTokens": 0,
            },
        )

    def test_none_usage_is_all_zero(self) -> None:
        self.assertEqual(sum(runner._usage_dict(None).values()), 0)


class PreviewTest(unittest.TestCase):
    def test_long_value_is_truncated_to_one_line(self) -> None:
        text = runner._preview("a\n" * 500, limit=10)
        self.assertEqual(text, "a a a a a ...")


class ParseArgsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.skill_dir = Path(self._tmp.name)
        (self.skill_dir / "SKILL.md").write_text("---\nname: x\n---\n")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _args(self, *extra: str) -> list[str]:
        return [
            "--model",
            "m",
            "--skill-dir",
            str(self.skill_dir),
            "--prompt",
            "p",
            *extra,
        ]

    def test_bedrock_with_region_parses(self) -> None:
        args = runner._parse_args(
            self._args("--provider", "bedrock", "--aws-region", "us-east-1")
        )
        self.assertEqual(args.aws_region, "us-east-1")

    def test_endpoint_without_url_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            runner._parse_args(self._args("--provider", "endpoint"))

    def test_missing_skill_md_is_rejected(self) -> None:
        (self.skill_dir / "SKILL.md").unlink()
        with self.assertRaises(SystemExit):
            runner._parse_args(
                self._args("--provider", "endpoint", "--endpoint", "http://x")
            )


if __name__ == "__main__":
    unittest.main()
