# MiniMax-M2.5 - serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model. On the 8x H200 (p5en.48xlarge) node also read [`.claude/skills/vllm-setup/p5en-h200-cuda-fixes.md`](../../../.claude/skills/vllm-setup/p5en-h200-cuda-fixes.md) - this FP8 model needs the CUDA-JIT fixes documented there.

| | |
|---|---|
| **HF repo** | `MiniMaxAI/MiniMax-M2.5` (FP8) |
| **Model card** | [huggingface.co/MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) |
| **Type** | MoE - 256 experts (8 active per token); `minimax_m2` architecture |
| **FP8 weights** | ~466 GB (125 shards, block-quantized fp8 e4m3, `weight_block_size=[128,128]`) |
| **Minimum hardware** | 4x H200 141GB (uses half of a p5en.48xlarge) |
| **Fits 4x L40S (184 GB)?** | No |
| **Tool-call parser** | `minimax_m2` |
| **Reasoning parser** | `minimax_m2` (this IS a reasoning model - thinking is separated into its own block) |
| **Native context** | **196608 (192K)** |
| **Role** | Fast mid-tier open-weight coding/reasoning model - much higher throughput than the frontier 8x-H200 models |

## Serve it

**Benchmarked config:** instance `p5en.48xlarge` (8x H200 141GB) · **TP=4** (4 of 8 GPUs) · precision **FP8** (`MiniMaxAI/MiniMax-M2.5`) · 196608 (192K) window.

MiniMax-M2.5 on the 8x H200 (p5en.48xlarge) node, using **4 GPUs** (TP=4 - see "Why TP=4" below). Requires `--trust-remote-code`. On this DLAMI, export the CUDA environment (ninja on PATH; unversioned `libcudart.so`/`libcuda.so`/`libnvrtc.so`) from [`p5en-h200-cuda-fixes.md`](../../../.claude/skills/vllm-setup/p5en-h200-cuda-fixes.md) **before** running this, or the FP8-kernel JIT fails at engine init.

```bash
MODEL="MiniMaxAI/MiniMax-M2.5" \
SERVED_NAME="minimax-m2.5" \
TP=4 \
PORT=8000 \
MAX_MODEL_LEN=196608 \
GPU_MEM_UTIL=0.92 \
TOOL_PARSER="minimax_m2" \
REASONING_PARSER="minimax_m2" \
EXTRA_ARGS="--trust-remote-code" \
  ./vllm-serve.sh
```

## Why TP=4 (not 8)

This is a **block-quantized FP8 MoE** model, which imposes two tensor-parallel divisibility constraints that intersect at TP=4:

1. **KV heads:** `num_key_value_heads = 8`, so TP must divide 8 -> {1, 2, 4, 8}. Violating this aborts with `assert self.total_num_kv_heads % tp_size == 0`.
2. **FP8 MoE block:** the per-GPU MoE shard `moe_intermediate_size / TP = 1536 / TP` must be divisible by the FP8 weight block size 128 -> TP in {1, 2, 4, 6}. Violating this aborts with `ValueError: The output_size of gate's and up's weight = N is not divisible by weight quantization block_n = 128` (TP=8 gives 192, not divisible by 128).

The only multi-GPU value satisfying both is **TP=4**. Because it uses only 4 of the 8 H200s (~116 GB/GPU of weights), **two replicas fit on one p5en** (GPUs 0-3 and 4-7 via `CUDA_VISIBLE_DEVICES`, different `PORT`).

## Instance and access

| | |
|---|---|
| **Instance type** | p5en.48xlarge (8x H200 141GB); this model uses 4 of the 8 GPUs |
| **Region** | us-east-2 |

## Runtime observed

At TP=4, `GPU_MEM_UTIL=0.92`, 192K window: GPU KV cache ~1.18M tokens, ~6.0x max concurrency. The whole 5-task `mcp-gateway-registry` benchmark ran in ~10 minutes (per-task latency 87-170s) - roughly 7-10x faster than the frontier 8x-H200 models (Kimi/GLM), reflecting the smaller active-parameter count.

## Tuning notes

- **DeepGemm / FlashInfer JIT:** the FP8 path JIT-compiles CUDA kernels at startup that need `ninja` on PATH and unversioned `libcudart.so`/`libcuda.so`/`libnvrtc.so`. Apply Fixes 1-3 from the p5en note.
- **Reasoning model:** pass `REASONING_PARSER=minimax_m2` so thinking is separated into a `"type": "thinking"` block rather than leaking into visible output. Give clients a generous `max_tokens` - the model thinks before answering, so a small output cap can truncate it mid-reasoning before any final content is emitted.
- **HF_TOKEN:** strongly recommended; the FP8 model is ~466 GB (125 shards).

## Comparison with other frontier models on this node

| Model | FP8 size | TP | Served window | Notes |
|-------|----------|----|----|-------|
| Kimi-K2.7-Code | ~555 GB | 8 | 131072 | frontier, slow |
| GLM-5.2 | ~750 GB | 8 | 300000 | frontier, slow |
| Qwen3-Coder-480B | ~482 GB | 4 | 200000 | frontier coder |
| MiniMax-M2.5 | ~466 GB | 4 | 196608 | fast, mid-tier quality |
