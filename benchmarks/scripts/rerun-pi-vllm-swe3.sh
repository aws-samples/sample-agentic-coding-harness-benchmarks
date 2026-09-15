#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# rerun-pi-vllm-swe3.sh -- hands-off, detached pi/swe3 benchmark for the three
# self-hosted models that fit this g6e.12xlarge (4x L40S). For each model it:
#   1. starts a vLLM server (vllm-serve.sh blocks until /v1/models is ready),
#   2. runs the full end-to-end pi/swe3 benchmark (harness + codex judge),
#   3. stops vLLM and frees the GPUs,
# then moves to the next model. One model's failure does not abort the batch.
#
# These are the pi runs whose tokens were undercounted by the pre-#99 harness
# (gemma-4-31b/swe2, qwen3-coder-30b/swe2 already on disk) plus qwen3.6-35b; we
# run them under swe3 with the fixed harness so token/cost figures are correct.
#
# Fully non-interactive: run-e2e --yes clears any stale artifact folders.
#
# Usage:
#   ./scripts/rerun-pi-vllm-swe3.sh            # foreground
#   ./scripts/rerun-pi-vllm-swe3.sh --detach   # detached, prints log paths
#
# Env overrides:
#   DATASET   dataset YAML relative to benchmarks/ (default mcp-gateway-registry)
#   LOG_DIR   per-model logs dir (default benchmarks/logs/pi-vllm-swe3-<ts>)
# ---------------------------------------------------------------------------
set -uo pipefail   # NOT -e: one model failing must not abort the batch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/.." && pwd)"
VLLM_SCRIPTS="$REPO_ROOT/self-hosted/vllm/scripts"

DATASET="${DATASET:-dataset/mcp-gateway-registry.yaml}"
DETACH=0
[[ "${1:-}" == "--detach" ]] && DETACH=1

# One entry per model: "SERVED_NAME|HF_MODEL|TOOL_PARSER|MAX_MODEL_LEN".
# All fit 4x L40S at TP=4 (see self-hosted/vllm/models/<model>.md).
RUNS=(
    "qwen3-coder-30b|Qwen/Qwen3-Coder-30B-A3B-Instruct|qwen3_coder|200000"
    "gemma-4-31b|google/gemma-4-31B-it|gemma4|200000"
    "qwen3.6-35b|Qwen/Qwen3.6-35B-A3B|qwen3_coder|200000"
)

TS="$(date -u +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-$BENCH_DIR/logs/pi-vllm-swe3-$TS}"
mkdir -p "$LOG_DIR"

if [[ "$DETACH" -eq 1 && -z "${PI_VLLM_DETACHED:-}" ]]; then
    export PI_VLLM_DETACHED=1
    DRIVER_LOG="$LOG_DIR/driver.log"
    echo "Launching detached. Driver log: $DRIVER_LOG"
    setsid bash "$SCRIPT_DIR/rerun-pi-vllm-swe3.sh" >"$DRIVER_LOG" 2>&1 &
    echo "PID $!"
    echo "Watch with:  tail -f $DRIVER_LOG"
    echo "Per-model logs under: $LOG_DIR"
    exit 0
fi

echo "=============================================================="
echo "pi x vLLM x swe3 -- ${#RUNS[@]} self-hosted models on 4x L40S"
echo "dataset=$DATASET   log dir: $LOG_DIR"
echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="

# Always leave the GPUs free on exit (normal end, error, or kill).
cleanup() { ( cd "$VLLM_SCRIPTS" && ./vllm-serve.sh --stop >/dev/null 2>&1 || true ); }
trap cleanup EXIT

SUMMARY=()
i=0
for RUN in "${RUNS[@]}"; do
    i=$((i + 1))
    IFS='|' read -r SERVED HF_MODEL PARSER MAXLEN <<< "$RUN"
    LOG="$LOG_DIR/${i}-${SERVED}.log"
    echo
    echo "======== [$i/${#RUNS[@]}] $SERVED ========"
    echo "log: $LOG"
    start=$(date -u +%s)
    status="OK"

    # 1. Stop any prior server, then serve this model (blocks until ready).
    echo "  [$SERVED] starting vLLM (MODEL=$HF_MODEL parser=$PARSER len=$MAXLEN)..." | tee -a "$LOG"
    ( cd "$VLLM_SCRIPTS" && ./vllm-serve.sh --stop >/dev/null 2>&1 || true )
    if ( cd "$VLLM_SCRIPTS" && MODEL="$HF_MODEL" SERVED_NAME="$SERVED" TP=4 \
            MAX_MODEL_LEN="$MAXLEN" GPU_MEM_UTIL=0.90 TOOL_PARSER="$PARSER" \
            ./vllm-serve.sh ) >>"$LOG" 2>&1; then
        # 2. Run the pi/swe3 benchmark end to end.
        echo "  [$SERVED] vLLM ready; running pi/swe3 benchmark..." | tee -a "$LOG"
        if ( cd "$BENCH_DIR" && ./scripts/run-e2e-benchmark.sh \
                --provider vllm --agent pi --skill swe3 \
                --model "$SERVED" --dataset "$DATASET" \
                --tensor-parallel-size 4 --precision BF16 --yes ) >>"$LOG" 2>&1; then
            status="OK"
        else
            status="BENCHMARK FAILED (rc=$?)"
        fi
    else
        status="VLLM START FAILED (rc=$?)"
    fi

    # 3. Stop vLLM, free GPUs before the next model.
    echo "  [$SERVED] stopping vLLM..." | tee -a "$LOG"
    ( cd "$VLLM_SCRIPTS" && ./vllm-serve.sh --stop >/dev/null 2>&1 || true )
    sleep 5

    elapsed=$(( $(date -u +%s) - start ))
    echo "result: $status  (${elapsed}s)"
    echo "---- tail of $LOG ----"; tail -n 20 "$LOG" || true; echo "---- end tail ----"
    SUMMARY+=("[$i/${#RUNS[@]}] $status  ${elapsed}s  $SERVED (pi/swe3)")
done

echo
echo "=============================================================="
echo "batch complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "${SUMMARY[@]}"
echo "logs: $LOG_DIR"
echo "=============================================================="
