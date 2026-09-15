#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run-swe3-pi-bedrock.sh -- fill the swe3 gaps for the 4 Bedrock models on the
# pi harness, one model after the other, fully non-interactive.
#
# These 4 models already have claude-code/swe3 data; only pi/swe3 is missing:
#   claude-haiku-4-5, claude-opus-4-8, claude-opus-5, claude-sonnet-5
#
# Each model runs the full end-to-end benchmark (harness + judge) via
# run-e2e-benchmark.sh with --agent pi --skill swe3 --yes, so nothing prompts.
# One model's failure does not abort the batch; a per-model log is written and
# the tail is echoed. Run detached with --detach (re-execs under nohup/setsid).
#
# Usage:
#   ./scripts/run-swe3-pi-bedrock.sh            # run in the foreground
#   ./scripts/run-swe3-pi-bedrock.sh --detach   # run detached, print log paths
#
# Env overrides:
#   DATASET   dataset YAML relative to benchmarks/ (default mcp-gateway-registry)
#   LOG_DIR   where per-model logs land (default benchmarks/logs/swe3-pi-<ts>)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET="${DATASET:-dataset/mcp-gateway-registry.yaml}"
DETACH=0
[[ "${1:-}" == "--detach" ]] && DETACH=1

# The 4 Bedrock model ids whose pi/swe3 run is missing.
MODELS=(
    "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    "us.anthropic.claude-opus-4-8[1m]"
    "us.anthropic.claude-opus-5[1m]"
    "us.anthropic.claude-sonnet-5"
)

# A fixed timestamp for this batch (avoids per-line date churn in the log dir).
TS="$(date -u +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-$BENCH_DIR/logs/swe3-pi-$TS}"
mkdir -p "$LOG_DIR"

# --- Re-exec detached if asked, then tell the caller how to watch it. --------
if [[ "$DETACH" -eq 1 && -z "${SWE3_PI_DETACHED:-}" ]]; then
    export SWE3_PI_DETACHED=1
    DRIVER_LOG="$LOG_DIR/driver.log"
    echo "Launching detached. Driver log: $DRIVER_LOG"
    setsid bash "$BENCH_DIR/scripts/run-swe3-pi-bedrock.sh" >"$DRIVER_LOG" 2>&1 &
    echo "PID $!"
    echo "Watch with:  tail -f $DRIVER_LOG"
    echo "Per-model logs will appear under: $LOG_DIR"
    exit 0
fi

echo "=============================================================="
echo "swe3 x pi x Bedrock -- ${#MODELS[@]} models, dataset=$DATASET"
echo "log dir: $LOG_DIR"
echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="

SUMMARY=()
i=0
for MODEL in "${MODELS[@]}"; do
    i=$((i + 1))
    # A filesystem-safe slug for the per-model log filename.
    SLUG="$(echo "$MODEL" | tr -c 'A-Za-z0-9._-' '_')"
    LOG="$LOG_DIR/${i}-${SLUG}.log"
    echo
    echo "-------- [$i/${#MODELS[@]}] $MODEL --------"
    echo "log: $LOG"

    start=$(date -u +%s)
    # run-e2e handles preflight (creds, clear stale folders via --yes) + judge.
    if ( cd "$BENCH_DIR" && ./scripts/run-e2e-benchmark.sh \
            --provider bedrock --agent pi --skill swe3 \
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
    SUMMARY+=("[$i/${#MODELS[@]}] $status  ${elapsed}s  $MODEL")
done

echo
echo "=============================================================="
echo "batch complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "${SUMMARY[@]}"
echo "logs: $LOG_DIR"
echo "=============================================================="
