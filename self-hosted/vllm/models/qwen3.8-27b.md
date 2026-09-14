# Qwen3.8-27B (FP8) — serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model.

| | |
|---|---|
| **HF repo** | `Qwen/Qwen3.8-27B-FP8` (BF16 original: `Qwen/Qwen3.8-27B`) |
| **Model card** | [huggingface.co/Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) |
| **Type** | dense — 27B, **every parameter active per token**; hybrid attention (48 linear + 16 full-attention layers, `full_attention_interval: 4`); **multimodal** (`Qwen3_5ForConditionalGeneration`, vision tower included) |
| **BF16 weights** | ~55.6 GB |
| **FP8 weights** | ~29 GB (e4m3, dynamic activation scale; vision blocks left unquantized) |
| **Fits 1x L40S (45 GiB, g6e.4xlarge)?** | FP8 yes. **BF16 no** — see below |
| **Fits 4x L40S (184 GB)?** | Yes, either precision |
| **Tool-call parser** | `qwen3_coder` (its chat template emits `<tool_call><function=...><parameter=...>`, not hermes JSON) |
| **Native context** | **262144 (256K)** |
| **Minimum vLLM** | **0.27.1** — earlier versions do not register `Qwen3_5ForConditionalGeneration` |
| **Role** | dense 27B at single-GPU economics; the first model here benchmarked on a 1-GPU node |

## Why FP8 on a single L40S

BF16 does not fit, and it is not close:

| | |
|---|---|
| L40S usable VRAM | 45.0 GiB |
| BF16 weights | ~51.8 GiB |
| Shortfall | **~6.8 GiB over the whole card**, before KV cache, activations, or CUDA context |

BF16 needs two or more GPUs (at TP=2 it is ~26 GiB/GPU and comfortable). On a `g6e.12xlarge` (4x L40S) serve the BF16 repo at TP=4 instead; on this box FP8 is the only option. FP8 is consistent with how the rest of this repo is benchmarked — GLM-5.2, Qwen3-Coder-480B and Devstral-2-123B are all FP8 — but the results must say so, because it is a different artifact from the BF16 model.

## Serve it

```bash
cd self-hosted/vllm/scripts
export PATH="$HOME/vllm-env/bin:$PATH"                              # ninja, see gotcha 2
export LIBRARY_PATH="$HOME/vllm-env/lib/python3.12/site-packages/nvidia/cu13/lib:$LIBRARY_PATH"

MODEL="Qwen/Qwen3.8-27B-FP8" \
SERVED_NAME="qwen3.8-27b" \
TP=1 \
PORT=8000 \
MAX_MODEL_LEN=65536 \
GPU_MEM_UTIL=0.90 \
TOOL_PARSER="qwen3_coder" \
EXTRA_ARGS="--max-num-seqs 32 --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
            --chat-template ../config/qwen3.8-27b-chat-template.jinja" \
  ./vllm-serve.sh
```

`MAX_MODEL_LEN=65536` is a deliberate cut from the 256K native window: it is the smallest window this repo will accept for agentic tasks (the multi-model runner skips anything under 64000), and on 45 GiB it is what leaves any KV headroom at all. See the measured numbers below.

## Four gotchas

The first three fail at engine init with an unhelpful `Engine core initialization failed`; the real cause is further up the log. The fourth is worse: the server starts fine and every request fails.

1. **`max_num_seqs (256) exceeds available Mamba cache blocks (173)`.** The 48 linear-attention layers each need a Mamba-style state block per decode sequence, and that pool is sized independently of the KV cache. vLLM's default `max_num_seqs` of 256 exceeds it, so CUDA-graph capture refuses to proceed. Cap it: `--max-num-seqs 32` (also the Anyscale value, and far above any concurrency this repo sweeps).
2. **`FileNotFoundError: 'ninja'`.** `vllm-install.sh` installs ninja *into the venv* (`~/vllm-env/bin/ninja`) but `vllm-serve.sh` does not put that directory on `PATH`, so FlashInfer's JIT cannot find it. Export the venv `bin` before serving. Only bites on JIT paths, which is why the 4x L40S reference node never hit it.
3. **`/usr/bin/ld: cannot find -lcudart`.** FlashInfer JIT-compiles the FP8-KV prefill kernel (head_dim 256 + e4m3 has no prebuilt kernel) and links `-lcudart` from `/opt/pytorch/cuda/lib64`, which is empty on this AMI. The runtime actually lives in the pip package, and ships only the versioned `libcudart.so.13`, so the linker cannot resolve `-lcudart`:

   ```bash
   V=~/vllm-env/lib/python3.12/site-packages/nvidia/cu13/lib
   ln -sf libcudart.so.13 "$V/libcudart.so"
   export LIBRARY_PATH="$V:$LIBRARY_PATH"
   ```

   The first serve after this pays a one-off JIT compile; later starts reuse `~/.cache/flashinfer`.

4. **`Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.`** The stock chat template validates `reasoning_effort` against `('xhigh', 'medium', 'low')`, and **Claude Code sends `high`** — so every `/v1/messages` request returns 500 and the model generates nothing. This does **not** stop the server, and the throughput harness records the dead sessions as ordinary `cutoff`s, so a sweep can run to completion and report ~0 tok/s with no obvious error. Always check `server ~N gen tok/s` in the harness heartbeat is non-zero before trusting a run.

   [`config/qwen3.8-27b-chat-template.jinja`](../config/qwen3.8-27b-chat-template.jinja) is the stock template with a two-line change: `high` is added to the accepted set and mapped to the same instruction text as `xhigh`. Serve with `--chat-template` pointing at it (already in the command above).

   This is a **deviation from the stock template** and belongs in any write-up of results from this model: the prompt Qwen ships does not accept the effort level Claude Code asks for, and we chose to accept it as `xhigh` rather than have the agent silently downgrade.

## Optimization knobs

Assessed against Anyscale's [llm-serving-for-coding-agents, part 3](https://github.com/anyscale/llm-serving-for-coding-agents/tree/main/part3-optimize), which optimizes the sibling Qwen3.6-27B for coding agents on a single GPU. Their measurements are on an RTX PRO 6000 (Blackwell, SM120); this node is an L40S (Ada, SM89), so the weight-format knob does not transfer.

| Knob | Here | Why |
|---|---|---|
| **FP8 KV cache** (`--kv-cache-dtype fp8`) | **On** | The biggest win on this box. Measured below: **1.81x more KV tokens**. L40S has native FP8. Anyscale report 6.53x concurrency at 256K from the same knob. |
| **CUDA graphs** | **On** (vLLM default) | Do not pass `--enforce-eager`. Anyscale measure ~2.87x decode on Blackwell; it is free here. |
| **Chunked prefill** (`--max-num-batched-tokens 8192`) | **On** | Agentic prompts arrive as very long prefills; chunking keeps one big prompt from stalling the batch. Anyscale's value. |
| **Prefix caching** | **On** (repo default) | Agentic sessions replay a growing transcript, so prefix hits are the dominant saving. Note vLLM flags Mamba-layer prefix caching as experimental in `align` mode for this architecture. |
| **`max_num_seqs 32`** | **On** | Required here — see gotcha 1 — and the Anyscale value. |
| **MTP speculative decoding** | **Off** | Available: this checkpoint ships `mtp.safetensors` (22 MTP tensors), and vLLM registers `Qwen3_5MTP`. Anyscale measure 121 vs 65 tok/s single-stream, but explicitly say to turn it off for **saturated high-concurrency traffic** — which is exactly what a throughput sweep is. Worth a separate single-stream latency run. |
| **NVFP4 weights** | **Not possible** | NVFP4 needs Blackwell; L40S is Ada. Anyscale keep FP8 as their documented fallback "for older FP8-capable GPUs", which is what we run. |
| **RunAI Streamer** (`load_format=runai_streamer`) | **Off** | Cold-start optimization only, and it cannot be combined with MTP ([vllm#42060](https://github.com/vllm-project/vllm/issues/42060)). Irrelevant to a steady-state throughput measurement. |
| **torch.compile cache** | Local only | Anyscale restore a prebuilt cache from S3, but it is keyed to their exact GPU, vLLM version and graph set, so it is not reusable here. The local `~/.cache` still shortens repeat starts. |
| **Prefix-aware routing** | **Not applicable** | Multi-replica routing; this is a single replica. |

## Measured on 1x L40S (g6e.4xlarge)

At `MAX_MODEL_LEN=65536`, `GPU_MEM_UTIL=0.90`, TP=1:

| KV dtype | Available KV | KV cache size | Max concurrency at 64K |
|---|---|---|---|
| bf16 (default) | 8.29 GiB | 125,974 tokens | 1.92x |
| **fp8 (used)** | 8.02 GiB | **228,010 tokens** | **3.48x** |

KV is 64 KiB/token — high because `head_dim` is 256, but paid on only the 16 full-attention layers, which is what makes a 64K window viable on 45 GiB at all.

**Expect this model to be KV-bound on this node.** Roughly 3.5 full-length sessions fit at once, so the throughput sweep should flatten by about c=3-5 and the cost per task will be dominated by the single-GPU KV ceiling, not by the model. That is a Regime A result in the terms of [cost-per-task-methodology.md](../../../docs/cost-per-task-methodology.md): no vertical headroom, so scale horizontally or move to a larger-VRAM node rather than pushing concurrency.

## Benchmark it

```bash
# Throughput and hardware-derived cost (server must already be running)
cd self-hosted/vllm
./scripts/run-throughput-sweep.sh --model qwen3.8-27b --context-window 65536
```

Instance pricing for `g6e.4xlarge` is in [pricing.json](../pricing.json) at the 3-year commitment rate ($1.298/hr), the same basis as `g6e.12xlarge`.
