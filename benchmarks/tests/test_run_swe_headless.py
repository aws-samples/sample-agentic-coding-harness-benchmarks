"""Tests for the headless SWE harness helper functions.

These cover the pure, side-effect-free helpers (repo-name derivation, prompt
construction, metric extraction, artifact-path resolution). The subprocess and
git-clone paths are not exercised here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

# The harness filename uses hyphens, so import it by path rather than name.
_HARNESS_PATH = _SCRIPTS_DIR / "run-swe-headless.py"
_spec = importlib.util.spec_from_file_location("run_swe_headless", _HARNESS_PATH)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

import unittest  # noqa: E402
from unittest import mock  # noqa: E402

from dataset_loader import Dataset, DatasetError, Task  # noqa: E402
from runner_config import RunnerConfig  # noqa: E402


def _task(**overrides: object) -> Task:
    """Build a Task with sensible defaults for testing."""
    data: dict[str, object] = {
        "id": "remove-faiss",
        "repo": "https://github.com/agentic-community/mcp-gateway-registry",
        "complexity": "medium",
        "tags": ["python"],
        "problem_statement": "Remove FAISS from the codebase.",
    }
    data.update(overrides)
    return Task.model_validate(data)


def _config(**overrides: object) -> RunnerConfig:
    """Build a RunnerConfig with sensible defaults for testing."""
    data: dict[str, object] = {
        "endpoint": "http://127.0.0.1:8000",
        "model": "qwen3.6-35b",
        "dataset": "dataset/example.yaml",
    }
    data.update(overrides)
    return RunnerConfig.model_validate(data)


def _ds(**overrides: object) -> Dataset:
    """Build a one-task Dataset matching _task(), for the scope-aware helpers."""
    data: dict[str, object] = {
        "schema_version": "1.0",
        "name": "d",
        "title": "D",
        "description": "test",
        "default_ref": "1.24.4",
        "metrics": ["input_tokens", "output_tokens", "num_turns"],
        "complexity_levels": ["low", "medium", "high"],
        "tasks": [_task().model_dump()],
    }
    data.update(overrides)
    return Dataset.model_validate(data)


class RepoNameTest(unittest.TestCase):
    def test_derives_basename(self) -> None:
        self.assertEqual(
            harness._repo_name("https://github.com/foo/mcp-gateway-registry"),
            "mcp-gateway-registry",
        )

    def test_strips_git_suffix_and_trailing_slash(self) -> None:
        self.assertEqual(harness._repo_name("https://github.com/foo/bar.git/"), "bar")


class SafeTaskSlugTest(unittest.TestCase):
    def test_kebab_case_id_unchanged(self) -> None:
        # Well-formed dataset ids pass through untouched -- this is the common
        # case and what makes the clone path transcribable by the agent.
        self.assertEqual(harness._safe_task_slug("remove-faiss"), "remove-faiss")

    def test_dots_and_underscores_preserved(self) -> None:
        self.assertEqual(harness._safe_task_slug("Foo_Bar.1"), "Foo_Bar.1")

    def test_slashes_replaced(self) -> None:
        self.assertEqual(harness._safe_task_slug("a/b"), "a-b")

    def test_path_traversal_neutralized(self) -> None:
        # Leading dots/dashes are stripped so a crafted id cannot escape the
        # clone parent (e.g. "../etc" must not become a traversal).
        self.assertEqual(harness._safe_task_slug("../etc"), "etc")
        self.assertEqual(harness._safe_task_slug(".."), "task")

    def test_empty_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            harness._safe_task_slug("")


class BuildPromptTest(unittest.TestCase):
    def test_prompt_has_all_swe_keys(self) -> None:
        prompt = harness._build_prompt(
            _task(),
            Path("/tmp/x/mcp-gateway-registry"),
            "1.24.4",
            "qwen3.6-35b",
            Path("/tmp/art"),
        )
        for key in ("repo:", "problem:", "model:", "answers:"):
            self.assertIn(key, prompt)
        self.assertIn("remove-faiss", prompt)

    def test_prompt_includes_issue_url_when_present(self) -> None:
        prompt = harness._build_prompt(
            _task(problem_issue_url="https://github.com/foo/bar/issues/1"),
            Path("/tmp/x/bar"),
            "main",
            "m",
            Path("/tmp/art"),
        )
        self.assertIn("Reference issue:", prompt)

    def test_prompt_has_fallback_answers_when_absent(self) -> None:
        prompt = harness._build_prompt(
            _task(clarifying_answers=None),
            Path("/tmp/x/r"),
            "main",
            "m",
            Path("/tmp/art"),
        )
        self.assertIn("best judgment", prompt)

    def test_prompt_invokes_swe2_skill(self) -> None:
        prompt = harness._build_prompt(
            _task(),
            Path("/tmp/x/mcp-gateway-registry"),
            "1.24.4",
            "m",
            Path("/tmp/art"),
        )
        # The harness drives /swe2 (design + implementation), not /swe.
        self.assertTrue(prompt.startswith("/swe2 "))
        # The absolute artifacts dir is passed through verbatim as a drift guard.
        self.assertIn("/tmp/art", prompt)

    def test_prompt_invokes_swe3_slash_command(self) -> None:
        # skill=swe3 drives the /swe3 slash command for the claude agent.
        prompt = harness._build_prompt(
            _task(),
            Path("/tmp/x/mcp-gateway-registry"),
            "1.24.4",
            "m",
            Path("/tmp/art"),
            skill="swe3",
        )
        self.assertTrue(prompt.startswith("/swe3 "))

    def test_pi_prompt_names_the_selected_skill(self) -> None:
        # pi has no slash commands; the prose names whichever skill is selected.
        prompt = harness._build_prompt(
            _task(),
            Path("/tmp/x/mcp-gateway-registry"),
            "1.24.4",
            "m",
            Path("/tmp/art"),
            agent="pi",
            skill="swe3",
        )
        self.assertIn("Use the swe3 skill", prompt)


class SkillPathTest(unittest.TestCase):
    def test_skill_path_points_at_configured_skill(self) -> None:
        p2 = harness._skill_path(_config(skill="swe2"))
        p3 = harness._skill_path(_config(skill="swe3"))
        self.assertEqual(p2.parent.name, "swe2")
        self.assertEqual(p3.parent.name, "swe3")
        self.assertEqual(p2.name, "SKILL.md")

    def test_pi_cmd_skill_flag_follows_config(self) -> None:
        cmd = harness._build_pi_cmd(_config(agent="pi", skill="swe3"), "prompt")
        skill_arg = cmd[cmd.index("--skill") + 1]
        self.assertTrue(skill_arg.endswith("swe3/SKILL.md"))


class KiroHarnessTest(unittest.TestCase):
    """kiro-cli command assembly and stderr-based metrics normalization."""

    def test_kiro_cmd_shape_and_terminator(self) -> None:
        cmd = harness._build_kiro_cmd(
            _config(agent="kiro", provider="kiro", model="claude-sonnet-5"),
            "PROMPT-BODY",
        )
        self.assertEqual(
            cmd[:6],
            [
                "kiro-cli",
                "chat",
                "--no-interactive",
                "--trust-all-tools",
                "--model",
                "claude-sonnet-5",
            ],
        )
        # A "--" terminator sits immediately before the single prompt positional,
        # so the inlined SKILL.md (which starts with "---") is never parsed as a
        # flag by kiro-cli.
        self.assertEqual(cmd[6], "--")
        self.assertEqual(len(cmd), 8)
        prompt_arg = cmd[7]
        self.assertIn("PROMPT-BODY", prompt_arg)  # task prompt inlined
        self.assertIn("swe3", prompt_arg)  # SKILL.md content inlined

    def test_kiro_result_parses_credits_and_time(self) -> None:
        # ANSI-colored stderr with the "Credits: N • Time: Ns" summary line.
        stderr = "\x1b[38;5;8m\n ▸ Credits: 0.21 • Time: 17s\n\x1b[0m"
        result = harness._kiro_result_from_output(stderr, 0, 20.0, 0.04)
        self.assertFalse(result["is_error"])
        self.assertEqual(result["subtype"], "success")
        self.assertEqual(result["kiro_credits"], 0.21)
        self.assertAlmostEqual(result["total_cost_usd"], 0.0084, places=6)
        # duration comes from the reported Time (17s), not the harness elapsed.
        self.assertEqual(result["duration_ms"], 17000)
        self.assertEqual(result["usage"], {"input_tokens": 0, "output_tokens": 0})

    def test_kiro_result_nonzero_exit_is_error(self) -> None:
        result = harness._kiro_result_from_output("boom", 1, 5.0, 0.04)
        self.assertTrue(result["is_error"])
        self.assertEqual(result["subtype"], "exit_1")

    def test_kiro_result_missing_credits_is_graceful(self) -> None:
        result = harness._kiro_result_from_output("no summary here", 0, 12.0, 0.04)
        self.assertIsNone(result["kiro_credits"])
        self.assertIsNone(result["total_cost_usd"])
        self.assertEqual(result["duration_ms"], 12000)  # falls back to elapsed

    def test_kiro_credits_flow_into_metrics(self) -> None:
        result = harness._kiro_result_from_output(
            " ▸ Credits: 4.7 • Time: 183s", 0, 183.0, 0.04
        )
        metrics = harness._metrics_from_result(result, 183.0)
        self.assertEqual(metrics["kiro_credits"], 4.7)
        self.assertAlmostEqual(metrics["total_cost_usd"], 0.188, places=3)

    def test_kiro_prompt_names_the_skill(self) -> None:
        prompt = harness._build_prompt(
            _task(),
            Path("/tmp/x/repo"),
            "master",
            "m",
            Path("/tmp/art"),
            agent="kiro",
            skill="swe3",
        )
        self.assertIn("Use the swe3 skill", prompt)


class AnnotateMetricsTopupTest(unittest.TestCase):
    """The summed top-up totals must land in BOTH top-level and metrics_that_matter,
    and MUST include total_cost_usd and cache tokens (not just turns/tokens)."""

    def test_summed_totals_written_to_both_levels(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(output_dir=tmp, model="m")
            art = harness._artifact_dir(cfg, _ds(), _task())
            art.mkdir(parents=True)
            # A metrics.json holding only the LAST (top-up) pass's numbers, with the
            # cache-write rename (cache_write_tokens) inside metrics_that_matter.
            (art / "metrics.json").write_text(
                json.dumps(
                    {
                        "num_turns": 141,
                        "input_tokens": 282,
                        "output_tokens": 51993,
                        "total_cost_usd": 12.0,
                        "cache_read_tokens": 1000,
                        "cache_creation_tokens": 100,
                        "metrics_that_matter": {
                            "num_turns": 141,
                            "input_tokens": 282,
                            "output_tokens": 51993,
                            "cache_read_tokens": 1000,
                            "cache_write_tokens": 100,
                        },
                    }
                ),
                encoding="utf-8",
            )
            # Totals summed across both passes (original + this top-up).
            totals = {
                "input_tokens": 1086,
                "output_tokens": 299445,
                "num_turns": 542,
                "latency_seconds": 4567.7,
                "total_cost_usd": 56.78,
                "cache_read_tokens": 64490345,
                "cache_creation_tokens": 500,
            }
            harness._annotate_metrics_topup(
                cfg, _ds(), _task(), 2, ["patch.diff"], totals
            )
            rec = json.loads((art / "metrics.json").read_text(encoding="utf-8"))
            # Top-level: cost + cache summed (cost is read from here by summarize).
            self.assertAlmostEqual(rec["total_cost_usd"], 56.78)
            self.assertEqual(rec["num_turns"], 542)
            self.assertEqual(rec["cache_read_tokens"], 64490345)
            # metrics_that_matter (what summarize reads for tokens/turns/cache):
            mm = rec["metrics_that_matter"]
            self.assertEqual(mm["num_turns"], 542)
            self.assertEqual(mm["output_tokens"], 299445)
            self.assertEqual(mm["cache_read_tokens"], 64490345)
            # cache-write renamed in mm and still summed:
            self.assertEqual(mm["cache_write_tokens"], 500)
            self.assertEqual(rec["agent_invocations"], 2)

    def test_topup_prompt_asks_only_for_missing_and_keeps_existing(self) -> None:
        prompt = harness._build_prompt(
            _task(),
            Path("/tmp/x/mcp-gateway-registry"),
            "1.24.4",
            "m",
            Path("/tmp/art"),
            topup_missing=["patch.diff", "implementation.md"],
        )
        # A top-up is a completion pass, not a restart: it names the missing files
        # and tells the agent to keep the ones already on disk.
        self.assertIn("COMPLETION PASS", prompt)
        self.assertIn("patch.diff", prompt)
        self.assertIn("implementation.md", prompt)
        self.assertIn("do not modify or rewrite", prompt.lower())
        # The design docs are named as already-finished, not requested.
        self.assertIn("github-issue.md", prompt)


class MissingArtifactsTest(unittest.TestCase):
    def test_lists_only_absent_files(self) -> None:
        # Absolute output_dir: Path("/repo/benchmarks") / "/abs/tmp" == "/abs/tmp",
        # so _artifact_dir lands under the tmp tree regardless of the repo root.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(output_dir=tmp, model="m")
            art = harness._artifact_dir(cfg, _ds(), _task())
            art.mkdir(parents=True)
            # Design done, implementation missing (the common pi truncation).
            for f in harness.DESIGN_ARTIFACT_FILENAMES:
                (art / f).write_text("x", encoding="utf-8")
            missing = harness._missing_artifacts(cfg, _ds(), _task())
            self.assertEqual(missing, ["patch.diff", "implementation.md"])


class ArtifactFilenamesTest(unittest.TestCase):
    def test_full_set_is_six_design_plus_implementation(self) -> None:
        # /swe2 emits the four design docs plus patch.diff + implementation.md.
        self.assertEqual(len(harness.DESIGN_ARTIFACT_FILENAMES), 4)
        self.assertEqual(
            harness.IMPLEMENTATION_ARTIFACT_FILENAMES,
            ("patch.diff", "implementation.md"),
        )
        self.assertEqual(len(harness.ARTIFACT_FILENAMES), 6)
        for name in ("patch.diff", "implementation.md"):
            self.assertIn(name, harness.ARTIFACT_FILENAMES)


class SummaryIsRetryableTest(unittest.TestCase):
    def test_ok_task_is_not_retried(self) -> None:
        self.assertFalse(harness._summary_is_retryable({"ok": True}))

    def test_turn_exhaustion_subtype_is_not_retried(self) -> None:
        summary = {
            "ok": False,
            "metrics": {"result_subtype": "error_max_turns", "num_turns": 250},
        }
        self.assertFalse(harness._summary_is_retryable(summary))

    def test_near_full_turns_without_design_is_not_retried(self) -> None:
        # Defensive fallback when no subtype is recorded: burned >=95% of the
        # budget and never finished the design -> treat as turn exhaustion.
        summary = {
            "ok": False,
            "design_done": False,
            "max_turns": 100,
            "metrics": {"result_subtype": None, "num_turns": 99},
        }
        self.assertFalse(harness._summary_is_retryable(summary))

    def test_transient_api_error_is_retried(self) -> None:
        summary = {
            "ok": False,
            "design_done": False,
            "max_turns": 250,
            "metrics": {"result_subtype": "error_during_execution", "num_turns": 12},
        }
        self.assertTrue(harness._summary_is_retryable(summary))

    def test_runtime_error_fallback_is_retried(self) -> None:
        # The RuntimeError branch of _run_task_safe produces this shape.
        summary = {
            "ok": False,
            "max_turns": 250,
            "metrics": {"is_error": True, "error": "run raised RuntimeError"},
        }
        self.assertTrue(harness._summary_is_retryable(summary))


class MetricsFromResultTest(unittest.TestCase):
    def test_extracts_six_metrics(self) -> None:
        result = {
            "num_turns": 12,
            "duration_ms": 45000,
            "total_cost_usd": 0.12,
            "is_error": False,
            "session_id": "abc",
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 500,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 50,
            },
        }
        metrics = harness._metrics_from_result(result, elapsed=99.0)
        self.assertEqual(metrics["input_tokens"], 1000)
        self.assertEqual(metrics["output_tokens"], 500)
        self.assertEqual(metrics["cache_read_tokens"], 200)
        self.assertEqual(metrics["cache_creation_tokens"], 50)
        self.assertEqual(metrics["num_turns"], 12)
        # duration_ms wins over the measured elapsed time.
        self.assertEqual(metrics["latency_seconds"], 45.0)

    def test_falls_back_to_elapsed_without_duration(self) -> None:
        metrics = harness._metrics_from_result({"usage": {}}, elapsed=7.25)
        self.assertEqual(metrics["latency_seconds"], 7.2)
        self.assertEqual(metrics["num_turns"], 0)

    def test_prefers_modelusage_over_main_agent_usage(self) -> None:
        # modelUsage includes subagent tokens; usage is main-agent-only. On a
        # fan-out run the harness MUST use modelUsage or it undercounts + fails to
        # reconcile with total_cost_usd. Here modelUsage output (1988) >> usage (349).
        result = {
            "num_turns": 5,
            "total_cost_usd": 0.107,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 349,
                "cache_read_input_tokens": 31299,
                "cache_creation_input_tokens": 611,
            },
            "modelUsage": {
                "us.anthropic.claude-haiku-4-5-20251001-v1:0": {
                    "inputTokens": 494,
                    "outputTokens": 1988,
                    "cacheReadInputTokens": 124537,
                    "cacheCreationInputTokens": 67660,
                    "costUSD": 0.107,
                }
            },
        }
        m = harness._metrics_from_result(result, elapsed=1.0)
        self.assertEqual(m["input_tokens"], 494)
        self.assertEqual(m["output_tokens"], 1988)
        self.assertEqual(m["cache_read_tokens"], 124537)
        self.assertEqual(m["cache_creation_tokens"], 67660)

    def test_sums_modelusage_across_multiple_models(self) -> None:
        result = {
            "modelUsage": {
                "model-a": {"inputTokens": 100, "outputTokens": 200},
                "model-b": {"inputTokens": 5, "outputTokens": 10},
            }
        }
        m = harness._metrics_from_result(result, elapsed=1.0)
        self.assertEqual(m["input_tokens"], 105)
        self.assertEqual(m["output_tokens"], 210)

    def test_falls_back_to_usage_when_no_modelusage(self) -> None:
        # Older Claude Code without modelUsage: use main-agent usage.
        result = {"usage": {"input_tokens": 7, "output_tokens": 8}}
        m = harness._metrics_from_result(result, elapsed=1.0)
        self.assertEqual(m["input_tokens"], 7)
        self.assertEqual(m["output_tokens"], 8)


class ArtifactDirTest(unittest.TestCase):
    def test_path_follows_model_harness_skill_convention(self) -> None:
        # Layout: <output>/<model>/<harness>/<skill>/<repo>/<task> -- skill (default
        # swe3) is its own level between harness and repo.
        path = harness._artifact_dir(
            _config(output_dir="swe-benchmark-data"), _ds(), _task()
        )
        self.assertEqual(
            path.parts[-6:],
            (
                "swe-benchmark-data",
                "qwen3.6-35b",
                "claude-code",
                "swe3",
                "mcp-gateway-registry",
                "remove-faiss",
            ),
        )

    def test_swe2_lands_in_its_own_level(self) -> None:
        path = harness._artifact_dir(
            _config(output_dir="swe-benchmark-data", skill="swe2"), _ds(), _task()
        )
        self.assertEqual(
            path.parts[-3:], ("swe2", "mcp-gateway-registry", "remove-faiss")
        )

    def test_output_scope_replaces_the_repo_level(self) -> None:
        # Two datasets over the SAME repo must not share a scope folder: the
        # folder-level run-summary.json would be rebuilt over both task sets.
        path = harness._artifact_dir(
            _config(output_dir="swe-benchmark-data"),
            _ds(output_scope="mcp-gateway-registry-v2"),
            _task(),
        )
        self.assertEqual(
            path.parts[-3:], ("swe3", "mcp-gateway-registry-v2", "remove-faiss")
        )

    def test_repo_level_is_unchanged_without_output_scope(self) -> None:
        # Every existing dataset leaves output_scope unset, so no committed
        # result path may move.
        default = harness._artifact_dir(
            _config(output_dir="swe-benchmark-data"), _ds(), _task()
        )
        self.assertEqual(default.parts[-2], "mcp-gateway-registry")


class BuildClaudeCmdTest(unittest.TestCase):
    def test_never_uses_bypass_permissions(self) -> None:
        cmd = harness._build_claude_cmd(_config(), "prompt")
        joined = " ".join(cmd)
        self.assertNotIn("bypassPermissions", joined)
        self.assertNotIn("dangerously-skip-permissions", joined)
        self.assertIn("acceptEdits", cmd)

    def test_includes_json_output_and_max_turns(self) -> None:
        cmd = harness._build_claude_cmd(_config(max_turns=42), "prompt")
        self.assertIn("json", cmd)
        self.assertIn("42", cmd)

    def test_stream_uses_stream_json_and_verbose(self) -> None:
        cmd = harness._build_claude_cmd(_config(), "prompt", stream=True)
        self.assertIn("stream-json", cmd)
        self.assertIn("--verbose", cmd)

    def test_non_stream_omits_verbose(self) -> None:
        cmd = harness._build_claude_cmd(_config(), "prompt", stream=False)
        self.assertNotIn("--verbose", cmd)
        self.assertIn("json", cmd)

    def test_always_passes_settings(self) -> None:
        # --settings must always be present so it overrides a user's global
        # ~/.claude/settings.json (e.g. one that pins Bedrock routing).
        cmd = harness._build_claude_cmd(_config(), "prompt")
        self.assertIn("--settings", cmd)

    def test_add_dir_when_clone_path_given(self) -> None:
        from pathlib import Path

        clone = Path("/tmp/swe-abc/mcp-gateway-registry")
        cmd = harness._build_claude_cmd(_config(), "prompt", clone_path=clone)
        self.assertIn("--add-dir", cmd)
        self.assertEqual(cmd[cmd.index("--add-dir") + 1], str(clone))

    def test_no_add_dir_without_clone_path(self) -> None:
        cmd = harness._build_claude_cmd(_config(), "prompt")
        self.assertNotIn("--add-dir", cmd)


class BuildPiCmdTest(unittest.TestCase):
    def test_vllm_endpoint_uses_vllm_provider_and_raw_model(self) -> None:
        cmd = harness._build_pi_cmd(_config(agent="pi"), "prompt")
        self.assertEqual(cmd[:5], ["pi", "-p", "--mode", "json", "--no-session"])
        self.assertEqual(cmd[cmd.index("--provider") + 1], "vllm")
        self.assertEqual(cmd[cmd.index("--model") + 1], "qwen3.6-35b")
        self.assertIn("--skill", cmd)

    def test_bedrock_uses_bedrock_provider_and_wire_id(self) -> None:
        # pi + Bedrock: the native amazon-bedrock provider, and the model is the
        # clean inference-profile id (prefix kept, harness "[1m]" hint stripped).
        cmd = harness._build_pi_cmd(
            _config(
                agent="pi",
                provider="bedrock",
                aws_region="us-east-1",
                endpoint=None,
                model="us.anthropic.claude-opus-5[1m]",
            ),
            "prompt",
        )
        self.assertEqual(cmd[cmd.index("--provider") + 1], "amazon-bedrock")
        self.assertEqual(cmd[cmd.index("--model") + 1], "us.anthropic.claude-opus-5")

    def test_bedrock_env_pins_region_and_injects_creds(self) -> None:
        # For Bedrock, _build_pi_env pins the region and resolves SigV4 creds via
        # `aws configure export-credentials` (pi does not probe EC2 IMDS). Mock the
        # CLI so the test needs no AWS setup.
        fake = mock.Mock(
            stdout="AWS_ACCESS_KEY_ID=AKIAX\nAWS_SECRET_ACCESS_KEY=sk\nAWS_SESSION_TOKEN=tok\n"
        )
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("AWS_ACCESS_KEY_ID", "AWS_BEARER_TOKEN_BEDROCK")
        }
        with mock.patch.dict(os.environ, clean_env, clear=True):
            with mock.patch.object(harness.subprocess, "run", return_value=fake):
                env = harness._build_pi_env(
                    _config(
                        agent="pi",
                        provider="bedrock",
                        aws_region="us-east-1",
                        endpoint=None,
                        model="us.anthropic.claude-opus-5",
                    ),
                    Path("/tmp/pi-agent"),
                )
        self.assertEqual(env["AWS_REGION"], "us-east-1")
        self.assertEqual(env["AWS_ACCESS_KEY_ID"], "AKIAX")
        self.assertEqual(env["AWS_SESSION_TOKEN"], "tok")

    def test_bedrock_env_keeps_existing_creds(self) -> None:
        # If the caller already exported creds, do not shell out to the AWS CLI.
        with mock.patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "PRESET"}, clear=False):
            with mock.patch.object(harness.subprocess, "run") as run:
                env = harness._build_pi_env(
                    _config(
                        agent="pi",
                        provider="bedrock",
                        aws_region="us-east-1",
                        endpoint=None,
                        model="us.anthropic.claude-opus-5",
                    ),
                    Path("/tmp/pi-agent"),
                )
        run.assert_not_called()
        self.assertEqual(env["AWS_ACCESS_KEY_ID"], "PRESET")

    def test_vllm_env_does_not_resolve_aws_creds(self) -> None:
        # The vLLM endpoint path never shells out to resolve AWS credentials
        # (whatever AWS_REGION the caller's shell already exports is irrelevant).
        with mock.patch.object(harness.subprocess, "run") as run:
            env = harness._build_pi_env(_config(agent="pi"), Path("/tmp/pi-agent"))
        run.assert_not_called()
        self.assertEqual(env["PI_CODING_AGENT_DIR"], "/tmp/pi-agent")


def _pi_events(usage: dict, stop: str = "stop") -> list[dict]:
    """Build a minimal pi event stream ending in agent_end with the given usage."""
    return _pi_events_multi([usage], stop=stop)


def _pi_events_multi(usages: list[dict], stop: str = "stop") -> list[dict]:
    """Build a pi event stream with one assistant message per usage dict.

    pi usage is PER-MESSAGE; the last assistant message carries the terminal
    stopReason. One turn_start per assistant message so num_turns lines up.
    """
    turns: list[dict] = []
    msgs: list[dict] = [{"role": "user"}]
    for i, usage in enumerate(usages):
        turns.append({"type": "turn_start"})
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "x"}],
                "usage": usage,
                # only the final message carries the terminal stop reason.
                "stopReason": stop if i == len(usages) - 1 else "end_turn",
            }
        )
    return [
        *turns,
        {"type": "agent_end", "messages": msgs, "willRetry": False},
    ]


class PiResultFromEventsTest(unittest.TestCase):
    def test_vllm_usage_no_cache_no_cost(self) -> None:
        # vLLM: pi reports 0 cache + 0 cost; both stay absent/None so they are not
        # misleading (real reuse comes from the Prometheus block).
        events = _pi_events(
            {
                "input": 100,
                "output": 20,
                "cacheRead": 0,
                "cacheWrite": 0,
                "cost": {"total": 0},
            }
        )
        result = harness._pi_result_from_events(events, elapsed=1.0)
        self.assertEqual(result["usage"]["input_tokens"], 100)
        self.assertNotIn("cache_read_input_tokens", result["usage"])
        self.assertNotIn("cache_creation_input_tokens", result["usage"])
        self.assertIsNone(result["total_cost_usd"])

    def test_bedrock_usage_maps_cache_and_real_cost(self) -> None:
        # Bedrock: pi reports native prompt-cache tokens and a real metered cost;
        # both must flow through (mapped to the keys _metrics_from_result reads).
        events = _pi_events(
            {
                "input": 2,
                "output": 5,
                "cacheRead": 1000,
                "cacheWrite": 3102,
                "cost": {"total": 0.0195},
            }
        )
        result = harness._pi_result_from_events(events, elapsed=2.0)
        self.assertEqual(result["usage"]["cache_read_input_tokens"], 1000)
        self.assertEqual(result["usage"]["cache_creation_input_tokens"], 3102)
        self.assertEqual(result["total_cost_usd"], 0.0195)
        # And _metrics_from_result surfaces them under its cache keys.
        metrics = harness._metrics_from_result(result, elapsed=2.0)
        self.assertEqual(metrics["cache_read_tokens"], 1000)
        self.assertEqual(metrics["cache_creation_tokens"], 3102)

    def test_sums_usage_across_all_turns_not_just_last(self) -> None:
        # pi usage is PER-MESSAGE, not cumulative. Reading only the last message
        # (the old bug) undercounts a multi-turn run ~100x. Every assistant
        # message's tokens/cost must be summed; stopReason comes from the last.
        events = _pi_events_multi(
            [
                {
                    "input": 3000,
                    "output": 300,
                    "cacheRead": 0,
                    "cacheWrite": 100,
                    "cost": {"total": 0.5},
                },
                {
                    "input": 2,
                    "output": 250,
                    "cacheRead": 90000,
                    "cacheWrite": 40,
                    "cost": {"total": 0.3},
                },
                {
                    "input": 1,
                    "output": 200,
                    "cacheRead": 96000,
                    "cacheWrite": 60,
                    "cost": {"total": 0.2},
                },
            ]
        )
        result = harness._pi_result_from_events(events, elapsed=5.0)
        # summed, NOT the last message's 1 / 200 / 96000 / 60.
        self.assertEqual(result["usage"]["input_tokens"], 3003)
        self.assertEqual(result["usage"]["output_tokens"], 750)
        self.assertEqual(result["usage"]["cache_read_input_tokens"], 186000)
        self.assertEqual(result["usage"]["cache_creation_input_tokens"], 200)
        self.assertAlmostEqual(result["total_cost_usd"], 1.0)
        self.assertEqual(result["num_turns"], 3)
        self.assertEqual(result["subtype"], "success")

    def test_accepts_bare_number_cost_shape(self) -> None:
        # Some pi versions report usage.cost as a bare number, not {"total": ...}.
        events = _pi_events_multi(
            [
                {
                    "input": 10,
                    "output": 5,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "cost": 0.4,
                },
                {
                    "input": 10,
                    "output": 5,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "cost": 0.6,
                },
            ]
        )
        result = harness._pi_result_from_events(events, elapsed=1.0)
        self.assertAlmostEqual(result["total_cost_usd"], 1.0)
        self.assertEqual(result["usage"]["output_tokens"], 10)


class BuildSettingsArgTest(unittest.TestCase):
    def test_inline_json_pins_routing_when_no_file(self) -> None:
        import json

        arg = harness._build_settings_arg(_config(endpoint="http://127.0.0.1:8000"))
        settings = json.loads(arg)
        self.assertEqual(settings["env"]["CLAUDE_CODE_USE_BEDROCK"], "0")
        self.assertEqual(settings["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8000")

    def test_uses_settings_file_when_configured(self) -> None:
        arg = harness._build_settings_arg(
            _config(settings_file="self-hosted/vllm/config/claude-code.json")
        )
        self.assertTrue(arg.endswith("self-hosted/vllm/config/claude-code.json"))

    def test_auto_compact_window_set_in_settings_env(self) -> None:
        import json

        arg = harness._build_settings_arg(_config(context_window=262144))
        settings = json.loads(arg)
        # 0.9 * 262144 = 235929: the settings env block must carry the window so
        # it wins over the process env, which Claude Code otherwise overrides.
        self.assertEqual(settings["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "235929")

    def test_auto_compact_window_absent_when_unset(self) -> None:
        import json

        arg = harness._build_settings_arg(_config())
        settings = json.loads(arg)
        self.assertNotIn("CLAUDE_CODE_AUTO_COMPACT_WINDOW", settings["env"])


class BuildEnvTest(unittest.TestCase):
    def test_auto_compact_window_set_in_process_env(self) -> None:
        env = harness._build_env(_config(context_window=262144))
        self.assertEqual(env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "235929")

    def test_auto_compact_window_absent_when_unset(self) -> None:
        env = harness._build_env(_config())
        self.assertNotIn("CLAUDE_CODE_AUTO_COMPACT_WINDOW", env)


def _dataset(n: int) -> Dataset:
    """Build a dataset with n tasks (ids task-0..task-{n-1})."""
    return Dataset.model_validate(
        {
            "schema_version": "1.0",
            "name": "d",
            "title": "D",
            "description": "test",
            "default_ref": "main",
            "metrics": ["input_tokens", "output_tokens", "num_turns"],
            "complexity_levels": ["low", "medium", "high"],
            "tasks": [
                {
                    "id": f"task-{i}",
                    "repo": "https://github.com/foo/bar",
                    "complexity": "low",
                    "tags": ["x"],
                    "problem_statement": "do the thing",
                }
                for i in range(n)
            ],
        }
    )


class SelectTasksTest(unittest.TestCase):
    def test_count_zero_returns_all(self) -> None:
        tasks = harness._select_tasks(_dataset(3), [], count=0)
        self.assertEqual([t.id for t in tasks], ["task-0", "task-1", "task-2"])

    def test_count_takes_first_n_in_order(self) -> None:
        tasks = harness._select_tasks(_dataset(3), [], count=1)
        self.assertEqual([t.id for t in tasks], ["task-0"])

    def test_count_larger_than_dataset_returns_all(self) -> None:
        tasks = harness._select_tasks(_dataset(2), [], count=99)
        self.assertEqual(len(tasks), 2)

    def test_count_applies_after_task_id_filter(self) -> None:
        tasks = harness._select_tasks(_dataset(4), ["task-1", "task-3"], count=1)
        self.assertEqual([t.id for t in tasks], ["task-1"])

    def test_negative_count_raises(self) -> None:
        with self.assertRaises(DatasetError):
            harness._select_tasks(_dataset(2), [], count=-1)


class FormatStreamEventTest(unittest.TestCase):
    def test_tool_use_event(self) -> None:
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": "Read"}]},
        }
        self.assertEqual(harness._format_stream_event(event), "[tool] Read")

    def test_assistant_text_event(self) -> None:
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Working on it"}]},
        }
        self.assertIn("Working on it", harness._format_stream_event(event) or "")

    def test_result_event_is_skipped(self) -> None:
        self.assertIsNone(harness._format_stream_event({"type": "result"}))

    def test_empty_content_returns_none(self) -> None:
        event = {"type": "assistant", "message": {"content": []}}
        self.assertIsNone(harness._format_stream_event(event))

    def test_tool_result_string_content_is_printed(self) -> None:
        event = {
            "type": "user",
            "message": {
                "content": [{"type": "tool_result", "content": "3 matches found"}]
            },
        }
        line = harness._format_stream_event(event)
        self.assertEqual(line, "[tool_result] 3 matches found")

    def test_tool_result_block_list_content_is_printed(self) -> None:
        event = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "content": [{"type": "text", "text": "line one\nline two"}],
                    }
                ]
            },
        }
        line = harness._format_stream_event(event)
        self.assertIn("line one", line or "")
        self.assertIn("line two", line or "")

    def test_tool_result_is_truncated(self) -> None:
        big = "x" * (harness.TOOL_RESULT_PREVIEW_CHARS + 50)
        event = {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "content": big}]},
        }
        line = harness._format_stream_event(event) or ""
        self.assertIn("+50 chars", line)
        self.assertLess(len(line), len(big))

    def test_verbose_shows_full_tool_result(self) -> None:
        big = "x" * (harness.TOOL_RESULT_PREVIEW_CHARS + 50)
        event = {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "content": big}]},
        }
        line = harness._format_stream_event(event, verbose=True) or ""
        self.assertIn(big, line)
        self.assertNotIn("chars)", line)

    def test_verbose_shows_full_assistant_text(self) -> None:
        big = "y" * 500
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": big}]},
        }
        line = harness._format_stream_event(event, verbose=True) or ""
        self.assertIn(big, line)

    def test_non_verbose_truncates_assistant_text(self) -> None:
        big = "y" * 500
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": big}]},
        }
        line = harness._format_stream_event(event) or ""
        self.assertIn("+300 chars", line)

    def test_tool_result_error_marker(self) -> None:
        event = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "content": "boom", "is_error": True}
                ]
            },
        }
        self.assertEqual(
            harness._format_stream_event(event), "[tool_result:error] boom"
        )

    def test_empty_tool_result_still_shows_marker(self) -> None:
        event = {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "content": ""}]},
        }
        self.assertEqual(harness._format_stream_event(event), "[tool_result]")


_PROM_SAMPLE = """\
# HELP vllm:prefix_cache_queries_total Queries
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 100.0
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 40.0
vllm:prefix_cache_hits_total{engine="1",model_name="m"} 10.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.5
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_count{engine="0",model_name="m"} 4.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="m"} 16.0
vllm:e2e_request_latency_seconds_bucket{le="0.3",engine="0"} 1.0
# TYPE vllm:generation_tokens_created gauge
vllm:generation_tokens_created{engine="0",model_name="m"} 1.78e+09
notvllm:ignored_total{x="y"} 999.0
"""


class ParsePrometheusMetricsTest(unittest.TestCase):
    def test_reads_types_for_vllm_families_only(self) -> None:
        types, _ = harness._parse_prometheus_metrics(_PROM_SAMPLE)
        self.assertEqual(types["vllm:prefix_cache_queries_total"], "counter")
        self.assertEqual(types["vllm:kv_cache_usage_perc"], "gauge")
        self.assertEqual(types["vllm:e2e_request_latency_seconds"], "histogram")

    def test_sums_samples_across_label_sets(self) -> None:
        _, samples = harness._parse_prometheus_metrics(_PROM_SAMPLE)
        # Two engine series for hits (40 + 10) are summed.
        self.assertEqual(samples["vllm:prefix_cache_hits_total"], 50.0)
        self.assertEqual(samples["vllm:prefix_cache_queries_total"], 100.0)

    def test_keeps_histogram_sum_and_count_but_skips_buckets(self) -> None:
        _, samples = harness._parse_prometheus_metrics(_PROM_SAMPLE)
        self.assertEqual(samples["vllm:e2e_request_latency_seconds_count"], 4.0)
        self.assertEqual(samples["vllm:e2e_request_latency_seconds_sum"], 16.0)
        self.assertNotIn("vllm:e2e_request_latency_seconds_bucket", samples)

    def test_ignores_non_vllm_series(self) -> None:
        _, samples = harness._parse_prometheus_metrics(_PROM_SAMPLE)
        self.assertNotIn("notvllm:ignored_total", samples)

    def test_counter_helper_returns_summed_value(self) -> None:
        val = harness._parse_prometheus_counter(
            _PROM_SAMPLE, "vllm:prefix_cache_hits_total"
        )
        self.assertEqual(val, 50.0)

    def test_counter_helper_returns_none_when_absent(self) -> None:
        self.assertIsNone(harness._parse_prometheus_counter(_PROM_SAMPLE, "nope"))


def _snap(**samples: float) -> dict[str, object]:
    """Build a snapshot dict, inferring a type for each sample name."""
    types: dict[str, str] = {}
    for name in samples:
        if name.endswith(("_count", "_sum")):
            types[name.rsplit("_", 1)[0]] = "histogram"
        elif name.endswith("_total"):
            types[name] = "counter"
        else:
            types[name] = "gauge"
    return {"types": types, "samples": dict(samples)}


_FULL_BEFORE = _snap(
    **{
        "vllm:prefix_cache_queries_total": 100.0,
        "vllm:prefix_cache_hits_total": 40.0,
        "vllm:prompt_tokens_total": 1000.0,
        "vllm:prompt_tokens_cached_total": 600.0,
        "vllm:generation_tokens_total": 500.0,
        "vllm:kv_cache_usage_perc": 0.1,
        "vllm:e2e_request_latency_seconds_count": 10.0,
        "vllm:e2e_request_latency_seconds_sum": 30.0,
    }
)
_FULL_AFTER = _snap(
    **{
        "vllm:prefix_cache_queries_total": 300.0,
        "vllm:prefix_cache_hits_total": 140.0,
        "vllm:prompt_tokens_total": 3000.0,
        "vllm:prompt_tokens_cached_total": 2000.0,
        "vllm:generation_tokens_total": 750.0,
        "vllm:kv_cache_usage_perc": 0.0,
        "vllm:e2e_request_latency_seconds_count": 14.0,
        "vllm:e2e_request_latency_seconds_sum": 46.4,
    }
)


class VllmMetricsTest(unittest.TestCase):
    def test_derives_prefix_cache_hit_rate(self) -> None:
        m = harness._vllm_metrics(_FULL_BEFORE, _FULL_AFTER)
        self.assertTrue(m["available"])
        self.assertEqual(m["source"], "vllm_prometheus_window")
        self.assertEqual(m["derived"]["prefix_cache_hit_rate"], 0.5)

    def test_derives_prompt_tokens_cached_rate(self) -> None:
        m = harness._vllm_metrics(_FULL_BEFORE, _FULL_AFTER)
        self.assertEqual(m["derived"]["prompt_tokens_cached_rate"], 0.7)

    def test_reports_every_counter_delta_including_generation_tokens(self) -> None:
        m = harness._vllm_metrics(_FULL_BEFORE, _FULL_AFTER)
        self.assertEqual(m["counters"]["vllm:generation_tokens_total"], 250)
        self.assertEqual(m["counters"]["vllm:prefix_cache_hits_total"], 100)

    def test_histogram_reports_window_mean(self) -> None:
        m = harness._vllm_metrics(_FULL_BEFORE, _FULL_AFTER)
        hist = m["histograms"]["vllm:e2e_request_latency_seconds"]
        self.assertEqual(hist["count"], 4)
        self.assertEqual(hist["sum"], 16.4)
        self.assertEqual(hist["mean"], 4.1)

    def test_gauge_is_instantaneous_post_run_reading(self) -> None:
        m = harness._vllm_metrics(_FULL_BEFORE, _FULL_AFTER)
        # Gauge reports the "after" value, not a delta.
        self.assertEqual(m["gauges"]["vllm:kv_cache_usage_perc"], 0)

    def test_missing_snapshot_marks_unavailable(self) -> None:
        m = harness._vllm_metrics(None, None)
        self.assertFalse(m["available"])
        self.assertIsNone(m["derived"]["prefix_cache_hit_rate"])
        self.assertEqual(m["counters"], {})
        self.assertEqual(m["histograms"], {})

    def test_zero_queries_gives_null_rate_not_divide_by_zero(self) -> None:
        before = _snap(
            **{
                "vllm:prefix_cache_queries_total": 5.0,
                "vllm:prefix_cache_hits_total": 5.0,
            }
        )
        m = harness._vllm_metrics(before, before)
        self.assertEqual(m["counters"]["vllm:prefix_cache_queries_total"], 0)
        self.assertIsNone(m["derived"]["prefix_cache_hit_rate"])

    def test_drops_created_timestamp_series(self) -> None:
        before = _snap(**{"vllm:generation_tokens_created": 1.0})
        after = _snap(**{"vllm:generation_tokens_created": 2.0})
        m = harness._vllm_metrics(before, after)
        self.assertNotIn("vllm:generation_tokens_created", m["gauges"])


class SummaryMetricsTest(unittest.TestCase):
    def test_passes_through_api_tokens_latency_and_turns(self) -> None:
        metrics = {
            "input_tokens": 245870,
            "output_tokens": 6370,
            "latency_seconds": 49.7,
            "num_turns": 14,
        }
        s = harness._summary_metrics(metrics, harness._vllm_metrics(None, None), 128.2)
        self.assertEqual(s["input_tokens"], 245870)
        self.assertEqual(s["output_tokens"], 6370)
        self.assertEqual(s["latency_seconds"], 49.7)
        self.assertEqual(s["num_turns"], 14)
        self.assertEqual(s["generation_tokens_per_sec"], 128.2)

    def test_falls_back_to_vllm_for_cache_tokens_when_api_silent(self) -> None:
        # vLLM does not report per-request cache tokens, so the summary should
        # fall back to prompt_tokens_cached and derive cache-write from the gap.
        vllm = harness._vllm_metrics(
            _snap(
                **{
                    "vllm:prompt_tokens_total": 0.0,
                    "vllm:prompt_tokens_cached_total": 0.0,
                }
            ),
            _snap(
                **{
                    "vllm:prompt_tokens_total": 246414.0,
                    "vllm:prompt_tokens_cached_total": 208032.0,
                }
            ),
        )
        s = harness._summary_metrics({"input_tokens": 1, "output_tokens": 1}, vllm, 0.0)
        self.assertEqual(s["cache_read_tokens"], 208032)
        self.assertEqual(s["cache_write_tokens"], 246414 - 208032)
        self.assertIn("vllm_prometheus", s["sources"]["cache_read_tokens"])

    def test_prefers_api_cache_tokens_when_present(self) -> None:
        metrics = {"cache_read_tokens": 999, "cache_creation_tokens": 111}
        s = harness._summary_metrics(metrics, harness._vllm_metrics(None, None), 0.0)
        self.assertEqual(s["cache_read_tokens"], 999)
        self.assertEqual(s["cache_write_tokens"], 111)
        self.assertIn("claude_api", s["sources"]["cache_read_tokens"])

    def test_surfaces_prefix_hit_rate(self) -> None:
        vllm = harness._vllm_metrics(
            _snap(
                **{
                    "vllm:prefix_cache_queries_total": 0.0,
                    "vllm:prefix_cache_hits_total": 0.0,
                }
            ),
            _snap(
                **{
                    "vllm:prefix_cache_queries_total": 100.0,
                    "vllm:prefix_cache_hits_total": 84.0,
                }
            ),
        )
        s = harness._summary_metrics({}, vllm, 0.0)
        self.assertEqual(s["prefix_cache_hit_rate"], 0.84)

    def test_omits_kv_cache_utilization_from_headline(self) -> None:
        # KV-cache utilization is intentionally NOT a headline metric; the sampled
        # peak/mean lives in vllm_prometheus.gauges_sampled instead.
        vllm = harness._vllm_metrics(
            _snap(**{"vllm:kv_cache_usage_perc": 0.0}),
            _snap(**{"vllm:kv_cache_usage_perc": 0.5}),
        )
        s = harness._summary_metrics({}, vllm, 0.0)
        self.assertNotIn("kv_cache_utilization_perc", s)
        self.assertNotIn("kv_cache_utilization_perc", s["sources"])


class MarkAggregateTest(unittest.TestCase):
    def test_flags_available_block_as_aggregate(self) -> None:
        block = harness._vllm_metrics(_FULL_BEFORE, _FULL_AFTER)
        original_note = block["note"]
        harness._mark_aggregate(block)
        self.assertFalse(block["single_tenant"])
        self.assertIn("AGGREGATE", block["note"])
        self.assertIn(original_note, block["note"])
        # The measured numbers themselves are left untouched.
        self.assertEqual(block["derived"]["prefix_cache_hit_rate"], 0.5)

    def test_noop_when_block_unavailable(self) -> None:
        block = harness._vllm_metrics(None, None)
        harness._mark_aggregate(block)
        self.assertNotIn("single_tenant", block)
        self.assertNotIn("AGGREGATE", block["note"])


class GaugePollerTest(unittest.TestCase):
    def test_summary_reports_peak_and_mean_of_sampled_values(self) -> None:
        poller = harness._GaugePoller("http://127.0.0.1:8000")
        poller._samples["vllm:kv_cache_usage_perc"] = [0.1, 0.5, 0.3]
        summary = poller.summary()
        self.assertTrue(summary["available"])
        self.assertEqual(summary["source"], "vllm_prometheus_poll")
        kv = summary["gauges"]["vllm:kv_cache_usage_perc"]
        self.assertEqual(kv["peak"], 0.5)
        self.assertEqual(kv["mean"], 0.3)
        self.assertEqual(kv["samples"], 3)

    def test_summary_marks_unavailable_when_nothing_sampled(self) -> None:
        poller = harness._GaugePoller("http://127.0.0.1:8000")
        summary = poller.summary()
        self.assertFalse(summary["available"])
        self.assertEqual(summary["gauges"], {})

    def test_summary_reports_null_for_gauge_never_seen(self) -> None:
        poller = harness._GaugePoller("http://127.0.0.1:8000")
        poller._samples["vllm:kv_cache_usage_perc"] = [0.2]
        summary = poller.summary()
        # A gauge the endpoint never exposed is null, not absent.
        running = summary["gauges"]["vllm:num_requests_running"]
        self.assertIsNone(running["peak"])
        self.assertEqual(running["samples"], 0)


class MetricsErrorTest(unittest.TestCase):
    def test_captures_error_message_on_failure(self) -> None:
        result = {
            "is_error": True,
            "api_error_status": 400,
            "result": "API Error (qwen3.6-35b): 400 The provided model identifier is invalid..",
            "usage": {},
        }
        metrics = harness._metrics_from_result(result, elapsed=0.2)
        self.assertTrue(metrics["is_error"])
        self.assertEqual(metrics["api_error_status"], 400)
        self.assertIn("invalid", metrics["error"])

    def test_no_error_field_on_success(self) -> None:
        metrics = harness._metrics_from_result({"is_error": False, "usage": {}}, 1.0)
        self.assertNotIn("error", metrics)


class TestCheckTokenAccounting(unittest.TestCase):
    """The run-time guard against undercounted token accounting.

    The extractor bug (#99) is fixed; this guard exists so a future regression in
    either agent's usage extraction is caught during the run instead of in review.
    Cases use real numbers from affected and healthy runs.
    """

    def test_flags_the_real_kimi_undercount(self) -> None:
        # kimi-k2.7-code pi (PR #96): 1 output token recorded over 106 turns.
        warning = harness._check_token_accounting(
            {"num_turns": 106, "output_tokens": 1}, "pi", "[task=ssrf]"
        )
        assert warning is not None
        self.assertIn("TOKEN ACCOUNTING SUSPECT", warning)
        self.assertIn("0.0/turn", warning)

    def test_flags_the_real_deepseek_undercount(self) -> None:
        # deepseek-v3.2 pi (PR #97): 542 output over 69 turns = 7.9/turn. Subtler
        # than kimi's but still an order of magnitude below plausible.
        warning = harness._check_token_accounting(
            {"num_turns": 69, "output_tokens": 542}, "pi", "[task=remove-efs]"
        )
        assert warning is not None
        self.assertIn("7.9/turn", warning)

    def test_passes_a_healthy_post_fix_pi_run(self) -> None:
        # nemotron-ultra-550b pi, post-fix: 47,996 output over 240 turns = ~200/turn.
        self.assertIsNone(
            harness._check_token_accounting(
                {"num_turns": 240, "output_tokens": 47996}, "pi", "[task=remove-faiss]"
            )
        )

    def test_passes_a_healthy_claude_run(self) -> None:
        # minimax-m2.5 claude-code: 21,383 output over 83 turns = ~258/turn.
        self.assertIsNone(
            harness._check_token_accounting(
                {"num_turns": 83, "output_tokens": 21383},
                "claude",
                "[task=remove-faiss]",
            )
        )

    def test_skips_short_runs_where_the_ratio_is_noise(self) -> None:
        # A 2-turn run can legitimately emit very little; do not cry wolf.
        self.assertIsNone(
            harness._check_token_accounting(
                {"num_turns": 2, "output_tokens": 5}, "pi", "[task=tiny]"
            )
        )

    def test_handles_missing_and_zero_fields(self) -> None:
        # A failed run may report no turns at all; must not divide by zero.
        self.assertIsNone(harness._check_token_accounting({}, "pi", "[task=x]"))
        self.assertIsNone(
            harness._check_token_accounting(
                {"num_turns": 0, "output_tokens": 0}, "pi", "[task=x]"
            )
        )
        self.assertIsNone(
            harness._check_token_accounting(
                {"num_turns": None, "output_tokens": None}, "pi", "[task=x]"
            )
        )


if __name__ == "__main__":
    unittest.main()


class TestMultiInvocationCostAccounting(unittest.TestCase):
    """A task's recorded cost must be the sum of every agent invocation.

    A transient retry throws away the failed attempt's artifacts, but that
    attempt still burned real tokens, turns and wall-clock. Recording only the
    successful pass understates the task's true cost by roughly the number of
    attempts it took -- which matters because cost per task is the headline
    number this repo publishes.
    """

    def test_pass_value_prefers_normalized_block(self) -> None:
        record = {
            "input_tokens": 111,
            "metrics_that_matter": {"input_tokens": 222},
        }
        self.assertEqual(harness._pass_cost_value(record, "input_tokens"), 222)

    def test_pass_value_falls_back_to_top_level(self) -> None:
        record = {"input_tokens": 111}
        self.assertEqual(harness._pass_cost_value(record, "input_tokens"), 111)

    def test_pass_value_is_zero_when_absent(self) -> None:
        self.assertEqual(harness._pass_cost_value({}, "input_tokens"), 0)

    def test_pass_value_reads_renamed_cache_write(self) -> None:
        record = {"metrics": {"cache_write_tokens": 77}}
        self.assertEqual(harness._pass_cost_value(record, "cache_creation_tokens"), 77)

    def test_fold_sums_across_passes(self) -> None:
        totals: dict[str, object] = {}
        harness._fold_pass_into_totals(totals, {"input_tokens": 100, "num_turns": 3})
        harness._fold_pass_into_totals(totals, {"input_tokens": 50, "num_turns": 4})
        self.assertEqual(totals["input_tokens"], 150)
        self.assertEqual(totals["num_turns"], 7)

    def test_write_cost_totals_updates_both_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(
                json.dumps(
                    {
                        "input_tokens": 10,
                        "metrics_that_matter": {
                            "input_tokens": 10,
                            "cache_write_tokens": 1,
                            "total_tokens": 11,
                        },
                    }
                ),
                encoding="utf-8",
            )
            totals = {"input_tokens": 30, "cache_creation_tokens": 5}
            harness._write_cost_totals(path, totals, 2, context="test")
            rec = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(rec["input_tokens"], 30)
            self.assertEqual(rec["metrics_that_matter"]["input_tokens"], 30)
            # The cache-write rename is honored inside the normalized block.
            self.assertEqual(rec["metrics_that_matter"]["cache_write_tokens"], 5)
            # total_tokens is derived, so it is recomputed from the summed parts.
            self.assertNotEqual(rec["metrics_that_matter"]["total_tokens"], 11)
            self.assertEqual(rec["agent_invocations"], 2)

    def test_write_cost_totals_is_noop_without_a_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "metrics.json"
            harness._write_cost_totals(missing, {"input_tokens": 1}, 2, context="t")
            self.assertFalse(missing.exists())

    def test_retry_records_the_sum_of_both_attempts(self) -> None:
        """Regression: a retried task reported only its final pass's cost.

        The first attempt wrote six artifacts to a sibling folder (a dataset
        with an output_scope), scored 0/6, and was retried. Its ~2.1M input
        tokens vanished from metrics.json, halving the reported cost.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(output_dir=tmp, max_retries=1, max_topups=0)
            art = harness._artifact_dir(cfg, _ds(), _task())
            art.mkdir(parents=True, exist_ok=True)
            metrics_path = art / "metrics.json"

            attempts = {"n": 0}

            def _fake_run_task(*args: object, **kwargs: object) -> dict[str, object]:
                attempts["n"] += 1
                first = attempts["n"] == 1
                metrics_path.write_text(
                    json.dumps(
                        {
                            "input_tokens": 2_000_000 if first else 3_000_000,
                            "output_tokens": 1_000 if first else 2_000,
                            "num_turns": 37 if first else 56,
                            "latency_seconds": 100.0 if first else 120.0,
                        }
                    ),
                    encoding="utf-8",
                )
                # The first attempt produced no artifacts the harness can see.
                return {
                    "task": _task().id,
                    "ok": not first,
                    "artifacts": 0 if first else 6,
                }

            with mock.patch.object(harness, "_run_task", _fake_run_task):
                harness._run_task_safe(
                    cfg,
                    _ds(),
                    _task(),
                    stream=False,
                    concurrent=False,
                    position=1,
                    total=1,
                )

            rec = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(attempts["n"], 2, "the task should have retried once")
            # Both attempts, not just the successful one.
            self.assertEqual(rec["input_tokens"], 5_000_000)
            self.assertEqual(rec["output_tokens"], 3_000)
            self.assertEqual(rec["num_turns"], 93)
            self.assertAlmostEqual(rec["latency_seconds"], 220.0)
            self.assertEqual(rec["agent_invocations"], 2)

    def test_single_attempt_cost_is_left_untouched(self) -> None:
        """A task that succeeds first time must not be rewritten or annotated."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(output_dir=tmp, max_retries=1, max_topups=0)
            art = harness._artifact_dir(cfg, _ds(), _task())
            art.mkdir(parents=True, exist_ok=True)
            metrics_path = art / "metrics.json"

            def _fake_run_task(*args: object, **kwargs: object) -> dict[str, object]:
                metrics_path.write_text(
                    json.dumps({"input_tokens": 42, "num_turns": 7}), encoding="utf-8"
                )
                return {"task": _task().id, "ok": True, "artifacts": 6}

            with mock.patch.object(harness, "_run_task", _fake_run_task):
                harness._run_task_safe(
                    cfg,
                    _ds(),
                    _task(),
                    stream=False,
                    concurrent=False,
                    position=1,
                    total=1,
                )

            rec = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(rec["input_tokens"], 42)
            self.assertNotIn("agent_invocations", rec)


class TestOmpMaxTime(unittest.TestCase):
    """omp has no turn cap, so a runaway generation loop needs a wall-clock cap.

    Without one, a model that finishes the work and then keeps emitting tokens
    runs until the harness timeout and then burns a retry, costing hours on a
    task whose artifacts were already complete.
    """

    def test_max_time_flag_is_passed(self) -> None:
        cmd = harness._build_omp_cmd(
            _config(agent_max_time_seconds=1800), "do the thing"
        )
        self.assertIn("--max-time=1800", cmd)

    def test_zero_disables_the_flag(self) -> None:
        cmd = harness._build_omp_cmd(_config(agent_max_time_seconds=0), "do it")
        self.assertFalse([a for a in cmd if a.startswith("--max-time")])

    def test_prompt_stays_positional_after_the_separator(self) -> None:
        """The cap must not displace the trailing "--" that ends option parsing."""
        cmd = harness._build_omp_cmd(_config(agent_max_time_seconds=600), "PROMPT")
        # The final argument is the inlined SKILL.md followed by the prompt.
        self.assertTrue(cmd[-1].endswith("PROMPT"))
        self.assertEqual(cmd[-2], "--")
        self.assertLess(cmd.index("--max-time=600"), cmd.index("--"))


def _omp_events(usages: list[dict], tail: int = 1, stop: str = "stop") -> list[dict]:
    """Build an omp stream whose agent_end covers only the last ``tail`` messages.

    omp emits a ``message_end`` per assistant message AND a ``turn_end`` mirroring
    it, then one ``agent_end`` carrying the settled conversation. An extra
    ``agent_start`` (compaction, or the todo reminder) resets that conversation, so
    ``agent_end`` reports only what came after it -- which is the shape that lost
    tokens in issue #157.

    Args:
        usages: One per-message usage dict, in order.
        tail: How many trailing messages survive into ``agent_end``.
        stop: Terminal stopReason on the final message.

    Returns:
        The event stream, oldest first.
    """
    events: list[dict] = [{"type": "agent_start"}]
    msgs: list[dict] = []
    for i, usage in enumerate(usages):
        if i == len(usages) - tail and tail < len(usages):
            events.append({"type": "agent_start"})
        message = {
            "role": "assistant",
            "usage": usage,
            "stopReason": stop if i == len(usages) - 1 else "end_turn",
        }
        msgs.append(message)
        events.append({"type": "turn_start"})
        events.append({"type": "message_end", "message": message})
        # turn_end mirrors message_end; counting both would double every message.
        events.append({"type": "turn_end", "message": message})
    events.append({"type": "agent_end", "messages": msgs[-tail:], "willRetry": False})
    return events


class OmpAgentEndTruncationTest(unittest.TestCase):
    """omp usage must come from the stream, not from the truncated agent_end."""

    @staticmethod
    def _usage(output: int) -> dict:
        return {
            "input": 2,
            "output": output,
            "cacheRead": 1000,
            "cacheWrite": 10,
            "cost": {"total": 0.25},
        }

    def test_sums_stream_when_agent_start_truncates_agent_end(self) -> None:
        # An extra agent_start resets the message list, so agent_end carries only
        # the final message. Reading it (the issue #157 bug) reported 40 output
        # tokens for a 4-turn run; the stream holds all four.
        events = _omp_events([self._usage(n) for n in (100, 200, 300, 40)], tail=1)
        result = harness._pi_result_from_events(events, elapsed=9.0)
        self.assertEqual(result["usage"]["output_tokens"], 640)
        self.assertEqual(result["usage"]["input_tokens"], 8)
        self.assertEqual(result["usage"]["cache_read_input_tokens"], 4000)
        self.assertEqual(result["usage"]["cache_creation_input_tokens"], 40)
        self.assertAlmostEqual(result["total_cost_usd"], 1.0)
        self.assertEqual(result["num_turns"], 4)
        self.assertEqual(result["subtype"], "success")

    def test_turn_end_does_not_double_count(self) -> None:
        # Every message_end has a matching turn_end carrying the same usage.
        events = _omp_events([self._usage(100), self._usage(200)], tail=2)
        result = harness._pi_result_from_events(events, elapsed=3.0)
        self.assertEqual(result["usage"]["output_tokens"], 300)

    def test_untruncated_stream_matches_agent_end_exactly(self) -> None:
        # The 200 single-agent_start streams this was measured on agree to the
        # token, so summing the stream must not change a healthy run.
        usages = [self._usage(n) for n in (100, 200, 300)]
        from_stream = harness._pi_result_from_events(_omp_events(usages, tail=3), 1.0)
        from_agent_end = harness._pi_result_from_events(_pi_events_multi(usages), 1.0)
        self.assertEqual(from_stream["usage"], from_agent_end["usage"])

    def test_falls_back_to_agent_end_without_message_end_events(self) -> None:
        # pi emits no message_end, so the settled conversation stays the source.
        events = _pi_events_multi([self._usage(100), self._usage(200)])
        result = harness._pi_result_from_events(events, elapsed=1.0)
        self.assertEqual(result["usage"]["output_tokens"], 300)


def _codex_turn(
    input_tokens: int,
    cached: int,
    cache_write: int,
    output_tokens: int,
    reasoning: int = 0,
) -> dict[str, object]:
    """Build a codex turn.completed event with the usage shape codex emits."""
    return {
        "type": "turn.completed",
        "usage": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached,
            "cache_write_input_tokens": cache_write,
            "output_tokens": output_tokens,
            "reasoning_output_tokens": reasoning,
        },
    }


def _codex_item(item_type: str, text: str = "") -> dict[str, object]:
    """Build a codex item.completed event."""
    return {"type": "item.completed", "item": {"type": item_type, "text": text}}


class CodexHarnessTest(unittest.TestCase):
    """codex exec command assembly, env routing, and usage normalization."""

    def test_codex_cmd_shape_and_terminator(self) -> None:
        cmd = harness._build_codex_cmd(
            _config(agent="codex", provider="bedrock", aws_region="us-east-1"),
            Path("/tmp/clone"),
            "PROMPT-BODY",
        )
        self.assertEqual(cmd[0], "codex")
        self.assertEqual(cmd[1], "exec")
        self.assertIn("--json", cmd)
        self.assertIn("--cd", cmd)
        self.assertEqual(cmd[cmd.index("--cd") + 1], "/tmp/clone")
        # provider=bedrock pins codex's native Bedrock provider.
        self.assertIn("model_provider=amazon-bedrock", cmd)
        # A "--" terminator sits immediately before the single prompt
        # positional, so the inlined SKILL.md (which starts with "---") is
        # never parsed as a flag.
        self.assertEqual(cmd[-2], "--")
        self.assertIn("PROMPT-BODY", cmd[-1])
        self.assertIn("swe3", cmd[-1])

    def test_codex_endpoint_declares_its_own_provider_block(self) -> None:
        # codex resolves the base URL from the provider its config selects and
        # ignores OPENAI_BASE_URL, so an endpoint run that passes no provider
        # block reaches whatever ~/.codex/config.toml points at -- Amazon
        # Bedrock on a judge-configured machine (issue #183).
        cmd = harness._build_codex_cmd(
            _config(
                agent="codex",
                provider="endpoint",
                endpoint="http://127.0.0.1:8000/",
                context_window=262144,
            ),
            Path("/tmp/clone"),
            "P",
        )
        self.assertNotIn("model_provider=amazon-bedrock", cmd)
        provider = harness.CODEX_ENDPOINT_PROVIDER
        self.assertIn(f"model_provider={provider}", cmd)
        self.assertIn(
            f"model_providers.{provider}.base_url=http://127.0.0.1:8000/v1", cmd
        )
        # codex 0.153.4 removed the chat-completions wire and rejects it.
        self.assertIn(f"model_providers.{provider}.wire_api=responses", cmd)
        self.assertIn(f"model_providers.{provider}.env_key=OPENAI_API_KEY", cmd)
        self.assertIn("model_context_window=262144", cmd)

    def test_codex_endpoint_omits_context_window_when_unset(self) -> None:
        cmd = harness._build_codex_cmd(
            _config(agent="codex", provider="endpoint"),
            Path("/tmp/clone"),
            "P",
        )
        self.assertFalse(any(c.startswith("model_context_window=") for c in cmd))

    def test_codex_endpoint_rejects_credentials_in_the_url(self) -> None:
        # The provider block travels in argv, where ps exposes it to any local
        # user, so a URL carrying userinfo is refused rather than trimmed.
        with self.assertRaisesRegex(ValueError, "must not embed credentials"):
            harness._build_codex_cmd(
                _config(
                    agent="codex",
                    provider="endpoint",
                    endpoint="https://user:secret@proxy.example.com",
                ),
                Path("/tmp/clone"),
                "P",
            )

    def test_codex_endpoint_rejects_a_non_http_scheme(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be http or https"):
            harness._codex_base_url("ws://127.0.0.1:8000")

    def test_codex_endpoint_requires_a_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs an endpoint"):
            harness._codex_base_url("")

    def test_codex_base_url_appends_v1_once(self) -> None:
        self.assertEqual(
            harness._codex_base_url("http://127.0.0.1:8000/"),
            "http://127.0.0.1:8000/v1",
        )

    def test_codex_env_bedrock_pins_region(self) -> None:
        env = harness._build_codex_env(
            _config(agent="codex", provider="bedrock", aws_region="us-west-2")
        )
        self.assertEqual(env["AWS_REGION"], "us-west-2")
        self.assertNotIn("OPENAI_BASE_URL", env)

    def test_codex_env_endpoint_sets_openai_vars(self) -> None:
        env = harness._build_codex_env(
            _config(
                agent="codex", provider="endpoint", endpoint="http://127.0.0.1:4000/"
            )
        )
        # Trailing slash is stripped before /v1 is appended.
        self.assertEqual(env["OPENAI_BASE_URL"], "http://127.0.0.1:4000/v1")
        self.assertEqual(env["OPENAI_API_KEY"], "local")

    def test_codex_usage_subtracts_cached_from_input(self) -> None:
        # Measured shape: codex input_tokens is the TOTAL, with cached and
        # cache_write as subsets of it (50768 = 36741 + 13817 + 210 fresh).
        events = [_codex_turn(50768, 36741, 13817, 925, reasoning=403)]
        result = harness._codex_result_from_events(events, 0, 1.0)
        usage = result["usage"]
        self.assertEqual(usage["input_tokens"], 210)
        self.assertEqual(usage["cache_read_input_tokens"], 36741)
        self.assertEqual(usage["cache_creation_input_tokens"], 13817)
        # reasoning_output_tokens is a subset of output_tokens, not a sibling,
        # so it must not be added on top.
        self.assertEqual(usage["output_tokens"], 925)

    def test_codex_usage_never_goes_negative(self) -> None:
        # If a future codex build reported these as siblings rather than
        # subsets, fresh input must clamp at 0 rather than go negative.
        events = [_codex_turn(100, 900, 900, 10)]
        result = harness._codex_result_from_events(events, 0, 1.0)
        self.assertEqual(result["usage"]["input_tokens"], 0)

    def test_codex_usage_sums_across_turns(self) -> None:
        # One exec emits one turn.completed today, but a build that splits an
        # exec into several turns (resume, compaction) must not lose all but
        # the last (issue #183).
        events = [
            _codex_turn(1_000, 600, 200, 50),
            _codex_turn(2_000, 1_500, 100, 70),
        ]
        usage = harness._codex_result_from_events(events, 0, 1.0)["usage"]
        self.assertEqual(usage["input_tokens"], 200 + 400)
        self.assertEqual(usage["output_tokens"], 120)
        self.assertEqual(usage["cache_read_input_tokens"], 2_100)
        self.assertEqual(usage["cache_creation_input_tokens"], 300)

    def test_codex_cost_does_not_double_count_cache(self) -> None:
        # gpt-5.6-terra: input 4.00, cache_write 5.00, cache_read 0.40 per 1M.
        events = [_codex_turn(50768, 36741, 13817, 925)]
        result = harness._codex_result_from_events(
            events, 0, 1.0, model="openai.gpt-5.6-terra"
        )
        expected = (
            210 * 4.00 + 925 * 18.00 + 36741 * 0.40 + 13817 * 5.00
        ) / 1_000_000.0
        self.assertAlmostEqual(result["total_cost_usd"], expected, places=6)
        # The naive formula (charging the full input again) would be far higher.
        naive = (50768 * 4.00 + 925 * 18.00 + 36741 * 0.40 + 13817 * 5.00) / 1_000_000.0
        self.assertLess(result["total_cost_usd"], naive)

    def test_codex_cost_is_none_for_unpriced_model(self) -> None:
        events = [_codex_turn(100, 0, 0, 10)]
        result = harness._codex_result_from_events(
            events, 0, 1.0, model="some-unpriced-model"
        )
        self.assertIsNone(result["total_cost_usd"])

    def test_codex_num_turns_counts_agent_steps(self) -> None:
        events = [
            _codex_item("agent_message", "first"),
            _codex_item("command_execution"),
            _codex_item("reasoning"),
            _codex_item("agent_message", "last"),
            _codex_turn(100, 0, 0, 10),
        ]
        result = harness._codex_result_from_events(events, 0, 1.0)
        # Messages and shell commands count; reasoning items do not.
        self.assertEqual(result["num_turns"], 3)
        self.assertEqual(result["result"], "last")

    def test_codex_nonzero_exit_is_error(self) -> None:
        result = harness._codex_result_from_events([_codex_turn(100, 0, 0, 10)], 1, 5.0)
        self.assertTrue(result["is_error"])
        self.assertEqual(result["subtype"], "exit_1")
        self.assertEqual(result["result"], "exit_1")

    def test_codex_missing_turn_completed_is_graceful(self) -> None:
        result = harness._codex_result_from_events(
            [_codex_item("agent_message", "hi")], 0, 2.0
        )
        self.assertEqual(result["usage"], {})
        self.assertFalse(result["is_error"])


class CodexTokenTotalsTest(unittest.TestCase):
    """The normalized block must keep a codex run's cache read (issue #183)."""

    def test_summary_total_keeps_cache_at_a_50_percent_hit_rate(self) -> None:
        # fresh input equals the cache sum here, which is the detector's
        # partition signature. codex counts are disjoint, so the total must add
        # the cache read rather than drop it.
        metrics = {
            "input_tokens": 50_000,
            "output_tokens": 900,
            "cache_read_tokens": 50_000,
            "cache_creation_tokens": 0,
            "latency_seconds": 10.0,
            "num_turns": 7,
            "total_cost_usd": None,
        }
        codex = harness._summary_metrics({**metrics}, {}, 90.0, True, agent="codex")
        self.assertEqual(codex["total_tokens"], 50_000 + 900 + 50_000)
        # A detected agent with the same shape still reads as a partition.
        claude = harness._summary_metrics({**metrics}, {}, 90.0, True, agent="claude")
        self.assertEqual(claude["total_tokens"], 50_000 + 900)


class ServerPromptTokenGapTest(unittest.TestCase):
    """Retried requests are server prefill the agent never counted."""

    def _metrics(self) -> dict[str, object]:
        return {
            "input_tokens": 1_972,
            "cache_read_tokens": 33_536,
            "cache_creation_tokens": 0,
        }

    def _vllm(self, prompt_tokens: float) -> dict[str, object]:
        return {"counters": {harness.PROMPT_TOKENS_METRIC: prompt_tokens}}

    def test_healthy_run_reconciles(self) -> None:
        # Measured: codex reported 35,508 prompt tokens and vLLM counted 35,508.
        gap = harness._server_prompt_token_gap(self._metrics(), self._vllm(35_508), 1)
        assert gap is not None
        self.assertTrue(gap["reconciled"])
        self.assertEqual(gap["unaccounted_prompt_tokens"], 0)

    def test_retried_requests_show_up_as_unaccounted_prefill(self) -> None:
        # Measured on a retrying endpoint: 26,022 server prompt tokens against
        # 8,700 reported, i.e. two abandoned requests the agent did not count.
        gap = harness._server_prompt_token_gap(
            {"input_tokens": 8_700, "cache_read_tokens": 0}, self._vllm(26_022), 1
        )
        assert gap is not None
        self.assertFalse(gap["reconciled"])
        self.assertEqual(gap["unaccounted_prompt_tokens"], 26_022 - 8_700)

    def test_concurrent_run_is_not_attributable(self) -> None:
        self.assertIsNone(
            harness._server_prompt_token_gap(self._metrics(), self._vllm(99_999), 4)
        )

    def test_missing_counter_returns_none(self) -> None:
        self.assertIsNone(harness._server_prompt_token_gap(self._metrics(), {}, 1))


class CodexTimeoutTest(unittest.TestCase):
    """The watchdog must fire even when the child never writes a line."""

    def test_silent_process_is_killed_at_the_deadline(self) -> None:
        # `sleep` produces no stdout, which is the case that used to hang: the
        # read loop blocked forever, so a clock check inside it never ran.
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "timed out after 1s"):
            harness._run_codex(["sleep", "30"], dict(os.environ), timeout=1)
        # It must actually stop at the deadline, not run the full 30s.
        self.assertLess(time.monotonic() - start, 15)

    def test_no_output_from_a_fast_exit_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "produced no output"):
            harness._run_codex(["true"], dict(os.environ), timeout=30)
