#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# rerun-pi-bedrock-fixed.sh -- re-run every Bedrock pi benchmark with the fixed
# harness (PR #99: pi token usage is now summed across all turns, not read from
# the last message). The pre-fix runs undercounted tokens/cost ~100x; this
# regenerates them with correct figures. Scores/turns/latency were already fine.
#
# Covers the 5 (model, skill) pi runs that used Amazon Bedrock:
#   haiku-4-5/swe3, opus-4-8/swe3, opus-5/swe3, opus-5/swe2, sonnet-5/swe3
#
# Fully non-interactive: run-e2e-benchmark.sh --yes clears the existing (buggy)
# artifact folders before re-running, so nothing prompts. One run's failure does
# not abort the batch; a per-run log is written and its tail echoed.
#
# Usage:
#   ./scripts/rerun-pi-bedrock-fixed.sh            # foreground
#   ./scripts/rerun-pi-bedrock-fixed.sh --detach   # detached, prints log paths
#
# Env overrides:
#   DATASET   dataset YAML relative to benchmarks/ (default mcp-gateway-registry)
#   LOG_DIR   where per-run logs land (default benchmarks/logs/rerun-pi-<ts>)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET="${DATASET:-dataset/mcp-gateway-registry.yaml}"
DETACH=0
[[ "${1:-}" == "--detach" ]] && DETACH=1

# (model id, skill) pairs -- one entry per existing Bedrock pi run.
RUNS=(
    "us.anthropic.claude-haiku-4-5-20251001-v1:0|swe3"
    "us.anthropic.claude-opus-4-8[1m]|swe3"
    "us.anthropic.claude-opus-5[1m]|swe3"
    "us.anthropic.claude-opus-5[1m]|swe2"
    "us.anthropic.claude-sonnet-5|swe3"
)

TS="$(date -u +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-$BENCH_DIR/logs/rerun-pi-$TS}"
mkdir -p "$LOG_DIR"

if [[ "$DETACH" -eq 1 && -z "${RERUN_PI_DETACHED:-}" ]]; then
    export RERUN_PI_DETACHED=1
    DRIVER_LOG="$LOG_DIR/driver.log"
    echo "Launching detached. Driver log: $DRIVER_LOG"
    setsid bash "$BENCH_DIR/scripts/rerun-pi-bedrock-fixed.sh" >"$DRIVER_LOG" 2>&1 &
    echo "PID $!"
    echo "Watch with:  tail -f $DRIVER_LOG"
    echo "Per-run logs will appear under: $LOG_DIR"
    exit 0
fi

echo "=============================================================="
echo "RE-RUN pi x Bedrock with fixed token summation -- ${#RUNS[@]} runs"
echo "dataset=$DATASET   log dir: $LOG_DIR"
echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="

SUMMARY=()
i=0
for RUN in "${RUNS[@]}"; do
    i=$((i + 1))
    MODEL="${RUN%%|*}"
    SKILL="${RUN##*|}"
    SLUG="$(echo "${MODEL}_${SKILL}" | tr -c 'A-Za-z0-9._-' '_')"
    LOG="$LOG_DIR/${i}-${SLUG}.log"
    echo
    echo "-------- [$i/${#RUNS[@]}] $MODEL  skill=$SKILL --------"
    echo "log: $LOG"

    start=$(date -u +%s)
    if ( cd "$BENCH_DIR" && ./scripts/run-e2e-benchmark.sh \
            --provider bedrock --agent pi --skill "$SKILL" \
            --model "$MODEL" --dataset "$DATASET" --yes ) >"$LOG" 2>&1; then
        status="OK"
    else
        status="FAILED (rc=$?)"
    fi
    elapsed=$(( $(date -u +%s) - start ))

    echo "result: $status  (${elapsed}s)"
    echo "---- tail of $LOG ----"
    tail -n 25 "$LOG" || true
    echo "---- end tail ----"
    SUMMARY+=("[$i/${#RUNS[@]}] $status  ${elapsed}s  $MODEL ($SKILL)")
done

echo
echo "=============================================================="
echo "re-run batch complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "${SUMMARY[@]}"
echo "logs: $LOG_DIR"
echo "=============================================================="
