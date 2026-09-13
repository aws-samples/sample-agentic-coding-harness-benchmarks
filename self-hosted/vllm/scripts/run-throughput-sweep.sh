#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# run-throughput-sweep.sh -- sweep agentic /swe concurrency to find a model's
# sustainable throughput on this EC2 instance, so a realistic cost-per-token and
# cost-per-task can be derived (fixed $/hr instance cost / measured tokens/hr).
#
# For each concurrency level it:
#   1. starts a DuckDB metrics collector session NAMED for the level
#      ({model}_c{N}), capturing the vLLM server-side saturation view;
#   2. drives {N} concurrent agentic /swe sessions for a fixed wall-clock window
#      via the throughput harness (benchmarks/scripts/run-throughput-harness.py),
#      writing a client-side level summary JSON;
#   3. stops the collector session.
#
# All levels share one DuckDB, one named session each, so throughput can be
# sliced per concurrency later. This does NOT score artifacts (throughput, not
# quality) and does NOT start the vLLM server (serve it first).
#
# Usage:
#   ./scripts/run-throughput-sweep.sh --model gemma-4-31b [options]
#
# Options (with defaults):
#   --model NAME             served-model-name (required)
#   --dataset PATH           dataset YAML relative to benchmarks/ (multi-repo-throughput)
#   --concurrencies "L..."   space-separated levels (default: "1 2 5 7 10 15 20";
#                            the c=1 baseline is the uncontended reference -- if its
#                            TTFT is not small, the server is backed up and the run
#                            should be discarded)
#   --duration-seconds N     wall-clock window per level (default: 600)
#   --context-window N       served window, calibrates auto-compaction (default: 200000)
#   --endpoint URL           vLLM base URL (default: http://127.0.0.1:8000)
#   --out-dir DIR            where level JSONs + the DuckDB land
#                            (default: self-hosted/vllm/benchmark-output/throughput/{model})
# ---------------------------------------------------------------------------

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VLLM_DIR="$( dirname "$SCRIPT_DIR" )"
REPO_ROOT="$( cd "$VLLM_DIR/../.." && pwd )"
BENCHMARKS_DIR="$REPO_ROOT/benchmarks"

MODEL=""
# Multi-repo throughput dataset: 25 tasks across 25 different repos, so N
# concurrent slots each clone a DIFFERENT codebase (see the harness's distinct-
# task-per-slot selection). Override with --dataset for a single-repo run.
DATASET="dataset/multi-repo-throughput.yaml"
CONCURRENCIES="1 2 5 7 10 15 20"
DURATION_SECONDS="600"
CONTEXT_WINDOW="200000"
ENDPOINT="http://127.0.0.1:8000"
OUT_DIR=""
# Extra seconds to let in-flight sessions drain after the window before the
# collector session stops, so its window fully covers the load.
DRAIN_SECONDS="120"

info() { printf '\033[0;36m[info]\033[0m  %s\n' "$1"; }
ok()   { printf '\033[0;32m[ok]\033[0m    %s\n' "$1"; }
warn() { printf '\033[0;33m[warn]\033[0m  %s\n' "$1"; }
step() { printf '\n\033[1;35m=== %s ===\033[0m\n' "$1"; }
die()  { printf '\033[0;31m[FAIL]\033[0m  %s\n' "$1" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)            MODEL="${2:?}"; shift 2 ;;
    --dataset)          DATASET="${2:?}"; shift 2 ;;
    --concurrencies)    CONCURRENCIES="${2:?}"; shift 2 ;;
    --duration-seconds) DURATION_SECONDS="${2:?}"; shift 2 ;;
    --context-window)   CONTEXT_WINDOW="${2:?}"; shift 2 ;;
    --endpoint)         ENDPOINT="${2:?}"; shift 2 ;;
    --out-dir)          OUT_DIR="${2:?}"; shift 2 ;;
    -h|--help)          sed -n '4,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                  die "unknown flag: $1 (see --help)" ;;
  esac
done

[[ -n "$MODEL" ]] || die "--model is required"
[[ -z "$OUT_DIR" ]] && OUT_DIR="$VLLM_DIR/benchmark-output/throughput/$MODEL"

command -v uv >/dev/null 2>&1 || die "uv not on PATH"
mkdir -p "$OUT_DIR"
# Resolve OUT_DIR to an ABSOLUTE path: the throughput harness runs from
# BENCHMARKS_DIR (a different cwd), so a relative --out-dir would otherwise make
# its --out land under benchmarks/ instead of here. mkdir first so realpath works.
OUT_DIR="$( cd "$OUT_DIR" && pwd )"
DB="$OUT_DIR/throughput-metrics.duckdb"

# The server must already be serving the requested model.
step "Pre-flight"
SERVED="$(curl -s -m 5 "$ENDPOINT/v1/models" 2>/dev/null \
  | python3 -c 'import sys,json;print(",".join(m["id"] for m in json.load(sys.stdin).get("data",[])))' 2>/dev/null || true)"
[[ ",$SERVED," == *",$MODEL,"* ]] || die "vLLM is serving [$SERVED], not '$MODEL' at $ENDPOINT. Serve it first."
ok "vLLM serving '$MODEL' at $ENDPOINT"
info "Levels: $CONCURRENCIES | ${DURATION_SECONDS}s/level | DuckDB: $DB"

for N in $CONCURRENCIES; do
  step "Concurrency $N"
  SESSION="${MODEL}_c${N}"
  LEVEL_JSON="$OUT_DIR/throughput-c${N}.json"
  COLLECTOR_LOG="$OUT_DIR/collector-c${N}.log"
  # Collector runs foreground-in-background and self-terminates after the window
  # plus drain, tagging every scrape with this level's session name.
  COLLECT_DURATION=$(( DURATION_SECONDS + DRAIN_SECONDS ))
  info "Starting collector session '$SESSION' (${COLLECT_DURATION}s) -> $DB"
  ( cd "$VLLM_DIR" && uv run python -m clients.collect_metrics \
      --base-url "$ENDPOINT" --database "$DB" --interval 1 \
      --duration "$COLLECT_DURATION" --session-name "$SESSION" \
      >"$COLLECTOR_LOG" 2>&1 ) &
  COLLECTOR_PID=$!
  sleep 2  # let the collector register its session before load starts

  info "Driving $N concurrent /swe sessions for ${DURATION_SECONDS}s ..."
  ( cd "$BENCHMARKS_DIR" && uv run python scripts/run-throughput-harness.py \
      --config config/runner.yaml --model "$MODEL" --endpoint "$ENDPOINT" \
      --dataset "$DATASET" --context-window "$CONTEXT_WINDOW" \
      --concurrency "$N" --duration-seconds "$DURATION_SECONDS" \
      --out "$LEVEL_JSON" ) \
    || warn "throughput harness reported an error at c=$N (continuing)"

  info "Waiting for collector session '$SESSION' to finish its window ..."
  wait "$COLLECTOR_PID" 2>/dev/null || true
  ok "Level $N done: $LEVEL_JSON"
done

step "Done"
ok "Swept [$CONCURRENCIES] for $MODEL. Per-level JSON + DuckDB in: $OUT_DIR"
info "Next: build the performance summary + dashboard from these:"
info "  cd $VLLM_DIR && uv run python -m clients.build_performance_summary --model $MODEL --db $DB --levels-dir $OUT_DIR"
