#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# run-swe-benchmark.sh - Convenience wrapper around the headless SWE harness.
#
# Drives claude -p through the /swe2 skill for every task in a dataset, using a
# runner config for endpoint/model/flags. All task data (repos, problem
# statements, clarifying answers, ground truth) lives in the dataset YAML - not
# in this script. This wrapper only forwards arguments to run-swe-headless.py.
#
# Prerequisites:
#   - benchmarks/.venv set up: (cd benchmarks && uv sync)
#   - The model served and reachable at the endpoint in the runner config
#     (e.g. a local vLLM server on http://127.0.0.1:8000).
#
# Usage:
#   ./run-swe-benchmark.sh [--config <path>] [extra run-swe-headless.py args...]
#
# Examples:
#   ./run-swe-benchmark.sh
#   ./run-swe-benchmark.sh --config config/runner.yaml
#   ./run-swe-benchmark.sh --model qwen3-coder-30b --tasks remove-faiss
#   ./run-swe-benchmark.sh --dry-run
#
# Environment variables:
#   CONFIG   Runner config path (default: config/runner.yaml). Copy
#            config/runner.example.yaml to config/runner.yaml first, or set
#            CONFIG. Overridden by an explicit --config argument.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${CONFIG:-config/runner.yaml}"

# If the caller passed an explicit --config, do not also inject the default.
inject_config=1
for arg in "$@"; do
  if [[ "$arg" == "--config" ]]; then
    inject_config=0
    break
  fi
done

cd "$BENCH_DIR"

if [[ "$inject_config" -eq 1 ]]; then
  exec uv run scripts/run-swe-headless.py --config "$CONFIG" "$@"
fi
exec uv run scripts/run-swe-headless.py "$@"
