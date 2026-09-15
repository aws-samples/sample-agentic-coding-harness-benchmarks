"""Tests for the SWE benchmark runner config loader."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from runner_config import (  # noqa: E402
    RunnerConfigError,
    load_runner_config,
    model_to_slug,
    model_to_wire_id,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SHIPPED_CONFIG = _REPO_ROOT / "benchmarks" / "config" / "runner.example.yaml"

_MINIMAL = """\
endpoint: http://127.0.0.1:8000
model: test-model
dataset: dataset/example.yaml
"""


def _write(text: str) -> Path:
    """Write config text to a temp file and return its path."""
    temp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    temp.write(text)
    temp.close()
    return Path(temp.name)


class LoadRunnerConfigTest(unittest.TestCase):
    def test_shipped_config_needs_model_and_dataset(self) -> None:
        # The shipped template intentionally leaves model and dataset unset so
        # one file serves every run; they must come from --model / --dataset.
        with self.assertRaisesRegex(RunnerConfigError, "model is required"):
            load_runner_config(_SHIPPED_CONFIG)

    def test_shipped_config_loads_with_cli_model_and_dataset(self) -> None:
        config = load_runner_config(
            _SHIPPED_CONFIG,
            {"model": "qwen3-coder-30b", "dataset": "dataset/example.yaml"},
        )
        self.assertEqual(config.model, "qwen3-coder-30b")
        # The shipped config uses bypassPermissions: /swe2 (implementation) needs
        # it so Claude Code's built-in Bash guard does not block the `cd <repo> &&
        # git ...` idiom, against throwaway clones. See runner_config.py.
        self.assertEqual(config.permission_mode, "bypassPermissions")
        self.assertIn("Read", config.allowed_tools)

    def test_missing_dataset_raises(self) -> None:
        text = "endpoint: http://127.0.0.1:8000\nmodel: m\n"
        with self.assertRaisesRegex(RunnerConfigError, "dataset is required"):
            load_runner_config(_write(text))

    def test_defaults_applied(self) -> None:
        config = load_runner_config(_write(_MINIMAL))
        self.assertEqual(config.api_key, "local")
        self.assertEqual(config.permission_mode, "acceptEdits")
        self.assertEqual(config.max_turns, 250)
        self.assertEqual(config.tasks, [])
        self.assertEqual(config.concurrency, 1)
        self.assertEqual(config.agent, "claude")
        self.assertEqual(config.skill, "swe3")
        self.assertEqual(config.max_retries, 1)
        self.assertEqual(config.max_topups, 1)

    def test_max_topups_override(self) -> None:
        config = load_runner_config(_write(_MINIMAL), {"max_topups": 3})
        self.assertEqual(config.max_topups, 3)
        with self.assertRaises(RunnerConfigError):
            load_runner_config(_write(_MINIMAL), {"max_topups": -1})

    def test_concurrency_override_and_floor(self) -> None:
        config = load_runner_config(_write(_MINIMAL), {"concurrency": 4})
        self.assertEqual(config.concurrency, 4)
        with self.assertRaises(RunnerConfigError):
            load_runner_config(_write(_MINIMAL), {"concurrency": 0})

    def test_cli_overrides_win(self) -> None:
        config = load_runner_config(
            _write(_MINIMAL),
            {"model": "override-model", "max_turns": 10, "tasks": ["a", "b"]},
        )
        self.assertEqual(config.model, "override-model")
        self.assertEqual(config.max_turns, 10)
        self.assertEqual(config.tasks, ["a", "b"])

    def test_max_output_tokens_override(self) -> None:
        # Lowered on the CLI for a small-window model so the prompt has input
        # room; None must not clobber the config/default value.
        config = load_runner_config(_write(_MINIMAL), {"max_output_tokens": 4096})
        self.assertEqual(config.max_output_tokens, 4096)
        default = load_runner_config(_write(_MINIMAL), {"max_output_tokens": None})
        self.assertEqual(default.max_output_tokens, 16000)

    def test_none_overrides_are_ignored(self) -> None:
        config = load_runner_config(_write(_MINIMAL), {"model": None, "endpoint": None})
        self.assertEqual(config.model, "test-model")

    def test_missing_file_raises(self) -> None:
        with self.assertRaisesRegex(RunnerConfigError, "not found"):
            load_runner_config("/nonexistent/runner.yaml")

    def test_bypass_permissions_accepted(self) -> None:
        # bypassPermissions is now a valid mode: /swe2 (implementation) requires
        # it against throwaway clones so Claude Code's Bash guard does not block
        # the `cd <repo> && git ...` idiom. It must load without error.
        text = _MINIMAL + "permission_mode: bypassPermissions\n"
        config = load_runner_config(_write(text))
        self.assertEqual(config.permission_mode, "bypassPermissions")

    def test_invalid_permission_mode_rejected(self) -> None:
        text = _MINIMAL + "permission_mode: nonsense\n"
        with self.assertRaisesRegex(RunnerConfigError, "permission_mode"):
            load_runner_config(_write(text))

    def test_bad_endpoint_scheme_rejected(self) -> None:
        text = "endpoint: 127.0.0.1:8000\nmodel: m\ndataset: d.yaml\n"
        with self.assertRaisesRegex(RunnerConfigError, "http"):
            load_runner_config(_write(text))

    def test_unknown_field_rejected(self) -> None:
        text = _MINIMAL + "bogus_field: 1\n"
        with self.assertRaises(RunnerConfigError):
            load_runner_config(_write(text))

    def test_config_from_overrides_only(self) -> None:
        config = load_runner_config(
            None,
            {"endpoint": "http://localhost:9000", "model": "m", "dataset": "d.yaml"},
        )
        self.assertEqual(config.endpoint, "http://localhost:9000")

    def test_default_provider_is_endpoint(self) -> None:
        config = load_runner_config(_write(_MINIMAL))
        self.assertEqual(config.provider, "endpoint")
        self.assertFalse(config.is_bedrock)


_BEDROCK = """\
provider: bedrock
model: us.anthropic.claude-opus-4-8
dataset: dataset/example.yaml
aws_region: us-east-1
"""


class BedrockProviderTest(unittest.TestCase):
    def test_bedrock_config_loads_without_endpoint(self) -> None:
        config = load_runner_config(_write(_BEDROCK))
        self.assertTrue(config.is_bedrock)
        self.assertIsNone(config.endpoint)
        self.assertEqual(config.resolved_region(), "us-east-1")

    def test_bedrock_region_falls_back_to_env(self) -> None:
        text = "provider: bedrock\nmodel: m\ndataset: d.yaml\n"
        with mock.patch.dict(os.environ, {"AWS_REGION": "eu-west-1"}, clear=False):
            config = load_runner_config(_write(text))
            self.assertEqual(config.resolved_region(), "eu-west-1")

    def test_bedrock_without_region_fails(self) -> None:
        text = "provider: bedrock\nmodel: m\ndataset: d.yaml\n"
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("AWS_REGION", "AWS_DEFAULT_REGION")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RunnerConfigError, "requires an AWS region"):
                load_runner_config(_write(text))

    def test_unknown_provider_rejected(self) -> None:
        text = "provider: azure\nmodel: m\ndataset: d.yaml\n"
        with self.assertRaisesRegex(RunnerConfigError, "provider"):
            load_runner_config(_write(text))

    def test_endpoint_provider_still_requires_endpoint(self) -> None:
        text = "model: m\ndataset: d.yaml\n"
        with self.assertRaisesRegex(RunnerConfigError, "endpoint is required"):
            load_runner_config(_write(text))

    def test_cli_can_switch_to_bedrock(self) -> None:
        config = load_runner_config(
            _write(_MINIMAL),
            {"provider": "bedrock", "aws_region": "us-west-2"},
        )
        self.assertTrue(config.is_bedrock)
        self.assertEqual(config.resolved_region(), "us-west-2")


class PiBedrockTest(unittest.TestCase):
    """pi supports native Amazon Bedrock (not only the vLLM endpoint)."""

    def test_pi_with_bedrock_is_allowed(self) -> None:
        # pi bundles the AWS SDK bedrock-runtime client, so agent=pi +
        # provider=bedrock is a valid combination (it used to be rejected).
        config = load_runner_config(
            _write(_BEDROCK),
            {"agent": "pi"},
        )
        self.assertTrue(config.is_pi)
        self.assertTrue(config.is_bedrock)
        self.assertEqual(config.resolved_region(), "us-east-1")

    def test_pi_with_bedrock_still_requires_region(self) -> None:
        text = "provider: bedrock\nagent: pi\nmodel: m\ndataset: d.yaml\n"
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("AWS_REGION", "AWS_DEFAULT_REGION")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RunnerConfigError, "requires an AWS region"):
                load_runner_config(_write(text))


_KIRO = "agent: kiro\nprovider: kiro\nmodel: claude-sonnet-5\ndataset: d.yaml\n"


class KiroConfigTest(unittest.TestCase):
    """kiro-cli drives its own managed models: agent=kiro must pair with
    provider=kiro, needs no endpoint or region, and reports credits (turned into
    dollars via kiro_dollars_per_credit)."""

    def test_kiro_config_loads_without_endpoint_or_region(self) -> None:
        config = load_runner_config(_write(_KIRO))
        self.assertTrue(config.is_kiro)
        self.assertFalse(config.is_bedrock)
        self.assertEqual(config.harness_slug, "kiro-cli")

    def test_kiro_default_dollars_per_credit(self) -> None:
        config = load_runner_config(_write(_KIRO))
        self.assertEqual(config.kiro_dollars_per_credit, 0.04)

    def test_kiro_dollars_per_credit_override(self) -> None:
        config = load_runner_config(_write(_KIRO), {"kiro_dollars_per_credit": 0.02})
        self.assertEqual(config.kiro_dollars_per_credit, 0.02)

    def test_kiro_model_slug_dashes_dots(self) -> None:
        # kiro's managed names carry dots; the slug dashes them so kiro shares
        # the dash-style folder (claude-haiku-4.5 -> claude-haiku-4-5).
        text = "agent: kiro\nprovider: kiro\nmodel: claude-haiku-4.5\ndataset: d.yaml\n"
        self.assertEqual(
            load_runner_config(_write(text)).model_slug, "claude-haiku-4-5"
        )

    def test_non_kiro_model_slug_keeps_dots(self) -> None:
        # Self-hosted dotted slugs (glm-5.2, deepseek-v3.2) must be preserved for
        # claude/pi so existing committed data and charts are not orphaned.
        self.assertEqual(
            load_runner_config(_write(_MINIMAL), {"model": "glm-5.2"}).model_slug,
            "glm-5.2",
        )
        self.assertEqual(model_to_slug("claude-haiku-4.5"), "claude-haiku-4.5")
        self.assertEqual(
            model_to_slug("claude-haiku-4.5", normalize_dots=True), "claude-haiku-4-5"
        )

    def test_kiro_agent_requires_kiro_provider(self) -> None:
        text = "agent: kiro\nprovider: endpoint\nendpoint: http://x:8000\nmodel: m\ndataset: d.yaml\n"
        with self.assertRaisesRegex(RunnerConfigError, "used together"):
            load_runner_config(_write(text))

    def test_kiro_provider_requires_kiro_agent(self) -> None:
        text = "agent: pi\nprovider: kiro\nmodel: m\ndataset: d.yaml\n"
        with self.assertRaisesRegex(RunnerConfigError, "used together"):
            load_runner_config(_write(text))


class SkillConfigTest(unittest.TestCase):
    """The skill field selects swe3 (default) or swe2; it is a SEPARATE path level,
    so harness_slug is agent-only and never encodes the skill."""

    def test_default_skill_is_swe3(self) -> None:
        config = load_runner_config(_write(_MINIMAL))
        self.assertEqual(config.skill, "swe3")

    def test_harness_slug_is_agent_only_regardless_of_skill(self) -> None:
        # Skill lives in its own path segment, so it never appears in harness_slug.
        for skill in ("swe2", "swe3"):
            self.assertEqual(
                load_runner_config(_write(_MINIMAL), {"skill": skill}).harness_slug,
                "claude-code",
            )
            self.assertEqual(
                load_runner_config(
                    _write(_MINIMAL), {"agent": "pi", "skill": skill}
                ).harness_slug,
                "pi",
            )

    def test_invalid_skill_rejected(self) -> None:
        with self.assertRaisesRegex(RunnerConfigError, "skill"):
            load_runner_config(_write(_MINIMAL), {"skill": "swe9"})


class ModelSlugTest(unittest.TestCase):
    def test_bedrock_prefix_and_suffix_stripped(self) -> None:
        self.assertEqual(
            model_to_slug("us.anthropic.claude-opus-4-8[1m]"), "claude-opus-4-8"
        )

    def test_bedrock_prefix_stripped_without_suffix(self) -> None:
        self.assertEqual(
            model_to_slug("us.anthropic.claude-opus-4-8"), "claude-opus-4-8"
        )

    def test_other_region_and_vendor_prefix_stripped(self) -> None:
        self.assertEqual(model_to_slug("eu.meta.llama3-70b"), "llama3-70b")

    def test_dated_haiku_folds_onto_short_slug(self) -> None:
        # A dated Bedrock id must slug to the same short folder as its short name,
        # so a re-run lands in the existing claude-haiku-4-5/ tree.
        self.assertEqual(
            model_to_slug("us.anthropic.claude-haiku-4-5-20251001-v1:0"),
            "claude-haiku-4-5",
        )

    def test_dated_suffix_only_strips_date_versioned_ids(self) -> None:
        # Plain version names (no -YYYYMMDD-vN:M) are untouched.
        self.assertEqual(model_to_slug("us.anthropic.claude-opus-5"), "claude-opus-5")
        self.assertEqual(model_to_slug("glm-5.2"), "glm-5.2")

    def test_mantle_prefix_preserved(self) -> None:
        # Mantle names use a single vendor token (no 2-letter region), so the
        # inference-profile regex must not touch them.
        self.assertEqual(
            model_to_slug("moonshotai.kimi-k2-thinking"),
            "moonshotai.kimi-k2-thinking",
        )

    def test_version_dot_preserved(self) -> None:
        self.assertEqual(model_to_slug("glm-5.2"), "glm-5.2")

    def test_plain_name_unchanged(self) -> None:
        self.assertEqual(model_to_slug("qwen3-coder-30b"), "qwen3-coder-30b")

    def test_config_model_slug_property(self) -> None:
        config = load_runner_config(
            _write(_MINIMAL),
            {
                "provider": "bedrock",
                "aws_region": "us-east-1",
                "model": "us.anthropic.claude-opus-4-8",
            },
        )
        self.assertEqual(config.model, "us.anthropic.claude-opus-4-8")
        self.assertEqual(config.model_slug, "claude-opus-4-8")

    def test_wire_id_keeps_prefix_strips_suffix(self) -> None:
        # The wire id (what pi passes to the Bedrock API) keeps the region/vendor
        # prefix but drops the harness "[1m]" context-window hint.
        self.assertEqual(
            model_to_wire_id("us.anthropic.claude-opus-5[1m]"),
            "us.anthropic.claude-opus-5",
        )

    def test_wire_id_without_suffix_unchanged(self) -> None:
        self.assertEqual(
            model_to_wire_id("us.anthropic.claude-opus-5"),
            "us.anthropic.claude-opus-5",
        )

    def test_wire_id_plain_name_unchanged(self) -> None:
        self.assertEqual(model_to_wire_id("qwen3-coder-30b"), "qwen3-coder-30b")


class AutoCompactWindowTest(unittest.TestCase):
    def test_unset_by_default(self) -> None:
        config = load_runner_config(_write(_MINIMAL))
        self.assertEqual(config.context_window, 0)
        self.assertIsNone(config.auto_compact_window)

    def test_computed_from_window_and_fraction(self) -> None:
        config = load_runner_config(_write(_MINIMAL), {"context_window": 262144})
        self.assertEqual(config.auto_compact_fraction, 0.9)
        self.assertEqual(config.auto_compact_window, 235929)

    def test_custom_fraction_applied(self) -> None:
        text = _MINIMAL + "context_window: 100000\nauto_compact_fraction: 0.8\n"
        config = load_runner_config(_write(text))
        self.assertEqual(config.auto_compact_window, 80000)

    def test_cli_context_window_override_wins(self) -> None:
        text = _MINIMAL + "context_window: 131072\n"
        config = load_runner_config(_write(text), {"context_window": 262144})
        self.assertEqual(config.auto_compact_window, 235929)

    def test_zero_window_leaves_it_unset(self) -> None:
        config = load_runner_config(_write(_MINIMAL), {"context_window": 0})
        self.assertIsNone(config.auto_compact_window)

    def test_negative_window_rejected(self) -> None:
        with self.assertRaises(RunnerConfigError):
            load_runner_config(_write(_MINIMAL), {"context_window": -1})

    def test_fraction_above_one_rejected(self) -> None:
        text = _MINIMAL + "context_window: 100000\nauto_compact_fraction: 1.5\n"
        with self.assertRaises(RunnerConfigError):
            load_runner_config(_write(text))


if __name__ == "__main__":
    unittest.main()
