# Qwen3-Coder-480B-A35B-Instruct-FP8 - serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model. On the 8x H200 (p5en.48xlarge) node also read [`.claude/skills/vllm-setup/p5en-h200-cuda-fixes.md`](../../../.claude/skills/vllm-setup/p5en-h200-cuda-fixes.md) - this FP8 model needs the CUDA-JIT fixes documented there.

| | |
|---|---|
| **HF repo** | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` (FP8) / `Qwen/Qwen3-Coder-480B-A35B-Instruct` (BF16) |
| **Model card** | [huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8) |
| **Type** | MoE - 480B total, **35B active per token** (160 experts, 8 active); `qwen3_moe` architecture |
| **FP8 weights** | ~482 GB (49 shards, block-quantized fp8 e4m3, `weight_block_size=[128,128]`) |
| **BF16 weights** | ~960 GB (does NOT fit 4x H200; needs TP=8) |
| **Minimum hardware** | 4x H200 141GB for FP8 (uses half of a p5en.48xlarge) |
| **Fits 4x L40S (184 GB)?** | No |
| **Tool-call parser** | `qwen3_coder` |
| **Reasoning parser** | none (Instruct model, not a reasoning model) |
| **Native context** | **262144 (256K)** - VRAM at TP=4 caps the practical window well below this (see below) |
| **Role** | Frontier open-weight coding model - the largest of the Qwen3-Coder family |

## Serve it

**Benchmarked config:** instance `p5en.48xlarge` (8x H200 141GB) · **TP=4** (4 of 8 GPUs) · precision **FP8** (`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`) · 200000 (200K) window.

Qwen3-Coder-480B-FP8 on the 8x H200 (p5en.48xlarge) node, using **4 GPUs** (TP=4 - see "Why TP=4" below). Requires `--trust-remote-code`. On this DLAMI, export the CUDA environment (ninja on PATH; unversioned `libcudart.so`/`libcuda.so`/`libnvrtc.so`) from [`p5en-h200-cuda-fixes.md`](../../../.claude/skills/vllm-setup/p5en-h200-cuda-fixes.md) **before** running this, or FP8-kernel JIT fails at engine init.

```bash
MODEL="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" \
SERVED_NAME="qwen3-coder-480b" \
TP=4 \
PORT=8000 \
MAX_MODEL_LEN=200000 \
GPU_MEM_UTIL=0.95 \
TOOL_PARSER="qwen3_coder" \
EXTRA_ARGS="--trust-remote-code" \
  ./vllm-serve.sh
```

## Why TP=4 (not 8)

This is a **block-quantized FP8 MoE** model, which imposes two tensor-parallel divisibility constraints that intersect at TP=4:

1. **KV heads:** `num_key_value_heads = 8`, so TP must divide 8 -> {1, 2, 4, 8}. Violating this aborts with `assert self.total_num_kv_heads % tp_size == 0`.
2. **FP8 MoE block:** the per-GPU MoE shard `moe_intermediate_size / TP = 2560 / TP` must be divisible by the FP8 weight block size 128 -> TP in {1, 2, 4}. Violating this aborts with `ValueError: The output_size of gate's and up's weight = N is not divisible by weight quantization block_n = 128` (TP=8 gives 320, not divisible by 128).

The only multi-GPU value satisfying both is **TP=4**. Because it uses only 4 of the 8 H200s (~120 GB/GPU of weights), **two replicas fit on one p5en** (GPUs 0-3 and 4-7 via `CUDA_VISIBLE_DEVICES`, different `PORT`). The BF16 variant has no FP8 block constraint, so it would run at TP=8 - but at ~960 GB it does not fit at TP=4.

## Context window reality

Native context is 256K, but at TP=4 the ~120 GB/GPU of weights leave a limited KV-cache budget:

| `max-model-len` | Fits at TP=4, GPU_MEM_UTIL=0.95? | Notes |
|----------------|-------|-------|
| 200,000 | Yes | Current config - KV cache ~255K tokens, ~1.28x concurrency |
| 262,144 (full) | Marginal | Native window; may reduce concurrency below 1x. Test before relying on it. |

To serve the full 256K at higher concurrency, use TP=8 with the BF16 weights (more GPUs, no FP8 block limit) or add `--kv-cache-dtype fp8`.

## Instance and access

| | |
|---|---|
| **Instance type** | p5en.48xlarge (8x H200 141GB); this model uses 4 of the 8 GPUs |
| **Region** | us-east-2 |

## Tuning notes

- **DeepGemm / FlashInfer JIT:** as with the other FP8 models on this node, the FP8 path JIT-compiles CUDA kernels at startup that need `ninja` on PATH and unversioned `libcudart.so`/`libcuda.so`/`libnvrtc.so`. Apply Fixes 1-3 from the p5en note.
- **No reasoning parser:** this is an Instruct (non-thinking) model - do not pass `REASONING_PARSER`.
- **HF_TOKEN:** strongly recommended; the FP8 model is ~482 GB (49 shards).
- **Startup time:** first boot downloads ~482 GB + weight load + torch.compile + CUDA graph capture. Subsequent boots (weights cached) are faster.
- **Prefix caching:** enabled by default; effective for the `/swe` benchmark's constant system prompt across turns.

## Comparison with other frontier models on this node

| Model | Params (active) | FP8 size | TP | Served window |
|-------|----------------|----------|----|----|
| Kimi-K2.7-Code | 1,058.6B (MoE) | ~555 GB | 8 | 131072 |
| GLM-5.2 | 744B (40B) | ~750 GB | 8 | 300000 |
| Qwen3-Coder-480B | 480B (35B) | ~482 GB | 4 | 200000 |
| MiniMax-M2.5 | ~230B (10B) | ~466 GB | 4 | 196608 |
