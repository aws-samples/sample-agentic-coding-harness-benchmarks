#!/usr/bin/env bash
# =============================================================================
# Run several models through the same benchmark, one after another
# =============================================================================
#
# run-e2e-benchmark.sh runs ONE model. Benchmarking a dataset means running the
# same thing for three or five models and waiting hours between each, which in
# practice gets done with a throwaway for-loop that is rewritten (and re-debugged)
# every time. This is that loop, kept.
#
# Models run SEQUENTIALLY, never in parallel. On a self-hosted endpoint they would
# otherwise contend for the same GPU and each other's KV cache, making both the
# latency and the vllm_prometheus block meaningless; on Bedrock they would race for
# the same account throughput quota. Sequential also keeps each judge pass clean.
#
# Every run is independent: one model failing does not stop the rest, and the exit
# code of each is reported in the log so a partial batch is diagnosable afterwards.
#
# Usage:
#   ./scripts/run-benchmark-batch.sh --provider bedrock \
#       --dataset dataset/mcp-gateway-registry-v2.yaml \
#       --models 'us.anthropic.claude-sonnet-5[1m],us.anthropic.claude-opus-5[1m]'
#
#   # only some tasks (e.g. ones newly added to an already-run dataset)
#   ./scripts/run-benchmark-batch.sh --provider bedrock \
#       --dataset dataset/mcp-gateway-registry-v2.yaml \
#       --models 'us.anthropic.claude-haiku-4-5-20251001-v1:0' \
#       --tasks 'task-a,task-b'
#
#   # a self-hosted model on the local vLLM server
#   ./scripts/run-benchmark-batch.sh --provider vllm \
#       --dataset dataset/mcp-gateway-registry-v2.yaml --models qwen3.8-27b
#
# Detach it, because a batch runs for hours or days and must outlive the SSH
# session that started it:
#
#   LOG="logs/batch-$(date -u +%Y%m%dT%H%M%SZ).log"
#   setsid nohup ./scripts/run-benchmark-batch.sh ... > "$LOG" 2>&1 < /dev/null &
#
# Options:
#   --provider   bedrock | litellm | vllm            (required)
#   --dataset    dataset YAML, relative to benchmarks/ (required)
#   --models     comma-separated model ids, run in order (required)
#   --tasks      comma-separated task ids; default is every task in the dataset
#   --agent      coding agent: pi (default), claude, omp, kiro
#   --skill      swe3 (default) or swe2
#   --aws-region region for the codex judge (default us-east-1). Needed even on
#                the vllm path: the model is local but the judge is on Bedrock.
# =============================================================================
set -u

cd "$(dirname "$0")/.."

PROVIDER=""; DATASET=""; MODELS=""; TASKS=""
AGENT="pi"; SKILL="swe3"; REGION="us-east-1"

die() { printf '\033[0;31m[error]\033[0m %s\n' "$1" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --provider)   PROVIDER="${2:?--provider needs a value}"; shift 2 ;;
        --dataset)    DATASET="${2:?--dataset needs a value}"; shift 2 ;;
        --models)     MODELS="${2:?--models needs a value}"; shift 2 ;;
        --tasks)      TASKS="${2:?--tasks needs a value}"; shift 2 ;;
        --agent)      AGENT="${2:?--agent needs a value}"; shift 2 ;;
        --skill)      SKILL="${2:?--skill needs a value}"; shift 2 ;;
        --aws-region) REGION="${2:?--aws-region needs a value}"; shift 2 ;;
        -h|--help)    sed -n '2,55p' "$0"; exit 0 ;;
        *)            die "unknown option: $1 (see --help)" ;;
    esac
done

[[ -n "$PROVIDER" ]] || die "--provider is required (bedrock | litellm | vllm)"
[[ -n "$DATASET"  ]] || die "--dataset is required"
[[ -n "$MODELS"   ]] || die "--models is required"

# The judge runs `codex exec` against Bedrock regardless of where the model is
# served, so the region is exported for every provider, not just bedrock.
export AWS_REGION="$REGION"

TASK_ARG=()
[[ -n "$TASKS" ]] && TASK_ARG=(--tasks "$TASKS")

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
echo "=== BATCH START $(date -u +%FT%TZ): ${#MODEL_LIST[@]} model(s), provider=$PROVIDER, dataset=$DATASET"
[[ -n "$TASKS" ]] && echo "=== scoped to tasks: $TASKS"

FAILED=()
for MODEL in "${MODEL_LIST[@]}"; do
    echo
    echo "=============================================================="
    echo "=== START $MODEL at $(date -u +%FT%TZ)"
    echo "=============================================================="
    ./scripts/run-e2e-benchmark.sh \
        --provider "$PROVIDER" --agent "$AGENT" --skill "$SKILL" \
        --model "$MODEL" "${TASK_ARG[@]}" \
        --dataset "$DATASET" --yes
    RC=$?
    echo "=== END $MODEL rc=$RC at $(date -u +%FT%TZ)"
    # Keep going: a later model's run does not depend on an earlier one, and
    # losing four runs because the first failed is worse than a partial batch.
    [[ $RC -eq 0 ]] || FAILED+=("$MODEL (rc=$RC)")
done

echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "=== BATCH DONE $(date -u +%FT%TZ): all ${#MODEL_LIST[@]} model(s) succeeded"
else
    echo "=== BATCH DONE $(date -u +%FT%TZ): ${#FAILED[@]} of ${#MODEL_LIST[@]} model(s) FAILED"
    for f in "${FAILED[@]}"; do echo "===   $f"; done
    exit 1
fi
