# Qwen3-Coder-30B-A3B-Instruct — serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model.

| | |
|---|---|
| **HF repo** | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| **Model card** | [huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| **Type** | MoE — 30.5B total, **3B active per token** |
| **BF16 weights** | ~61 GB |
| **Fits 4×L40S (184 GB)?** | ✅ comfortably — leaves ~120 GB for KV cache |
| **Tool-call parser** | `qwen3_coder` (the script default) |
| **Native context** | **262144 (256K)** — extensible to ~1M with YaRN |
| **Role** | ⭐ the repo default — the throughput-benchmark workhorse |

## Serve it

This is the script default model, so the bare command serves it — but at the script's own 32768 window (a throughput default):

```bash
cd self-hosted/vllm/scripts
./vllm-serve.sh
```

Wrapper with the **recommended 200K window** for this 256K-native model (set `MAX_MODEL_LEN` explicitly — it is far above the script's 32768 default):

```bash
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct" \
SERVED_NAME="qwen3-coder-30b" \
TP=4 \
PORT=8000 \
MAX_MODEL_LEN=200000 \
GPU_MEM_UTIL=0.90 \
TOOL_PARSER="qwen3_coder" \
  ./vllm-serve.sh
```

Exact vLLM command, including the same log destination as the wrapper:

```bash
cd self-hosted/vllm
mkdir -p logs
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=/opt/pytorch/cuda
export HF_HOME=/opt/dlami/nvme/hf-cache
export HF_HUB_CACHE="$HF_HOME/hub"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

~/vllm-env/bin/vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen3-coder-30b \
  --max-model-len 200000 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  2>&1 | tee logs/vllm-serve.log
```

## Why it's the default

Only the **3B active parameters** run per token even though 30B are resident, so per-token compute (and therefore cost) tracks the active count, not the total. That makes it fast and leaves a large VRAM budget for the KV cache — exactly the regime the [hosting-strategy cost model](../../../README.md) depends on, where a fixed-cost GPU node beats per-token API pricing under concurrent load.

## Tuning notes

- **Tool calling:** use the `qwen3_coder` parser (default). This is the correct parser for the *Coder* models; do **not** use `hermes` here. Agentic clients (opencode, Claude Code) require it — see the [README tool-calling section](../README.md#tool-calling---enable-auto-tool-choice---tool-call-parser-qwen3_coder).
- **Concurrency headroom:** with ~120 GB free after weights, this model holds the most concurrent requests of any model in this folder. Raise `GPU_MEM_UTIL` toward `0.95` for even more KV cache if you are not seeing OOM headroom warnings at boot.
- **Context window — native 256K; we serve 200000 (200K) by default here.** Per the [HF model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) this model is **262144 (256K) native**, extensible to ~1M with YaRN. `MAX_MODEL_LEN` does *not* auto-expand to native — it is a hard ceiling you set. The spelled-out command above pins it to **200000**: below native (so **no `ROPE_SCALING`**) while still leaving KV-cache headroom for concurrency on 4×L40S. To trade context for maximum concurrency (a pure throughput benchmark), drop it — e.g. `MAX_MODEL_LEN=32768`, the bare script default:
  ```bash
  # smaller window = more concurrent requests fit in KV cache
  MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct SERVED_NAME=qwen3-coder-30b \
    MAX_MODEL_LEN=32768 ./vllm-serve.sh
  ```
  **Tradeoffs:** context length scales KV-cache VRAM linearly, so a larger window means proportionally fewer concurrent requests. Anything **≤256K needs no `ROPE_SCALING`** (it's under native); only **past 256K** does YaRN come in. On 4×L40S the real ceiling is KV-cache VRAM, not the model — watch the `Maximum concurrency` line at boot and lower `MAX_MODEL_LEN` if it reports `1x` or OOMs. See [Long context past 32K](../README.md#long-context-and-rope_scaling-yarn).

## Drive it as a coding agent

opencode's config points at the served name by default:

```bash
cd self-hosted/vllm/scripts
./opencode-setup.sh --launch      # interactive TUI, backed by vllm/qwen3-coder-30b
```
