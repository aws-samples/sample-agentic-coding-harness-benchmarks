# MiniMax-M3 - serving guidelines

> [!IMPORTANT]
> **Not yet served on this node.** Every other guide here records a configuration that booted and ran a benchmark; this one is written ahead of the first bring-up. Be clear about which parts are which:
>
> - **Checked**: the model's dimensions and quantization (read from `config.json`), its weight sizes (HF API), and what the installed vLLM 0.28.0 supports (read from the package). The TP=8 conclusion follows from those.
> - **Not checked**: that it serves, the startup time, the KV-cache size, the concurrency, and every runtime number. `MAX_MODEL_LEN` and `GPU_MEM_UTIL` below are proposals, not measurements.
>
> Correct this on first bring-up and delete this note once the model has actually run.

| | |
|---|---|
| **HF repo (BF16)** | `MiniMaxAI/MiniMax-M3` |
| **HF repo (FP8)** | `MiniMaxAI/MiniMax-M3-MXFP8` - **MXFP8**, microscaling FP8, not the block-FP8 used by M2.5 and GLM-5.3 |
| **Model card** | [huggingface.co/MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) |
| **Recipe** | [recipes.vllm.ai/MiniMaxAI/MiniMax-M3](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3) |
| **Type** | **Vision-language MoE**: `model_type` is `minimax_m3_vl`, a CLIP vision tower plus a 60-layer MoE text model (128 experts, 4 active per token). MSA sparse attention with an index cache |
| **Weights size** | **MXFP8 444 GB (31 shards)**; BF16 854 GB (59 shards). Read from the HF API |
| **Quantization** | `mxfp8`, dynamic activation scheme, `weight_block_size=[1,32]` - far finer than the `[128,128]` block FP8 of M2.5 and GLM-5.3 |
| **Minimum hardware** | 8x H200/H20 recommended by the recipe; compute capability >= 9.0 (Hopper or newer) |
| **Fits 4x L40S (184 GB)?** | No |
| **Tool-call parser** | `minimax_m3` |
| **Reasoning parser** | `minimax_m3` (a reasoning model; thinking is separated into its own block) |
| **Native context** | 1048576 (1M), from `text_config.max_position_embeddings`; cap it with `--max-model-len` |

## What this build actually supports

The recipe says M3 "has not yet shipped in a stable vLLM release - use the dedicated Docker image". That is **out of date for the vLLM pinned here (0.28.0)**, which carries M3 support natively. Verified in the installed package, so no Docker image is needed:

| What | Where |
|---|---|
| Model config | `vllm/transformers_utils/configs/minimax_m3.py` |
| Reasoning parser | `vllm/reasoning/minimax_m3_reasoning_parser.py` |
| Tool-call parser name | `minimax_m3` registered under `entrypoints/openai/tool_parsers/` |
| Sparse-attention kernels | `minimax_m3_index_decode`, `minimax_m3_index_decode_score` |

Re-check this after a vLLM upgrade or on a different box before assuming it holds.

## Serve it

`--block-size 128` is **mandatory**: the MSA sparse attention and its index cache require it, and the default 16 will not work. `vllm-serve.sh` has no dedicated variable for it, so it rides in `EXTRA_ARGS`, exactly as GLM-5.3's `--kv-cache-dtype` and speculative config do.

On a p5en.48xlarge, source the node's environment first ([p5en-h200-cuda-fixes.md](../../../.claude/skills/vllm-setup/p5en-h200-cuda-fixes.md)) so the CUDA JIT fixes and `NCCL_NVLS_ENABLE=0` are in place. Without the NCCL setting the server dies at distributed init, before any weights load.

```bash
cd self-hosted/vllm/scripts

MODEL="MiniMaxAI/MiniMax-M3-MXFP8" \
SERVED_NAME="minimax-m3" \
TP=8 PORT=8000 MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.90 \
TOOL_PARSER="minimax_m3" REASONING_PARSER="minimax_m3" \
EXTRA_ARGS="--trust-remote-code --block-size 128" \
  ./vllm-serve.sh
```

`--enable-auto-tool-choice` is added by `vllm-serve.sh` whenever a tool parser is set, so it does not belong in `EXTRA_ARGS`.

`MAX_MODEL_LEN` is set to 262144 rather than the native 1M so the KV cache fits alongside the weights. The repo's harness gate wants at least 200K, so this clears it with room. Raise it only after confirming the reported `GPU KV cache size` still supports a workable concurrency.

## TP=8 is legal, and here is why it is not obvious

The dimensions below come from `config.json` on the MXFP8 repo. Note they live under **`text_config`**, not at the top level, because this is a VL model:

| Field | Value |
|---|---|
| `num_hidden_layers` | 60 |
| `num_attention_heads` | 64 |
| `num_key_value_heads` | **4** |
| `head_dim` | 128 |
| `hidden_size` | 6144 |
| `intermediate_size` | 3072 |
| `num_local_experts` / `num_experts_per_tok` | 128 / 4 |

**Four KV heads on eight GPUs looks like a blocker and is not.** The rule that forces M2.5 to TP=4 is `num_key_value_heads % TP == 0`, and 4 % 8 is not 0. But vLLM only applies that test when there are at least as many KV heads as ranks; below that it replicates KV heads instead, requiring `TP % num_key_value_heads == 0`. That is `8 % 4 == 0`, so **TP=8 is legal** (`model_executor/layers/linear.py`, the `tp_size >= self.total_num_kv_heads` branch).

The MoE constraint also passes, and much more easily than for the block-FP8 models: MXFP8's `weight_block_size` is `[1,32]`, so the per-rank shard `3072 / 8 = 384` needs to divide by 32, which it does. The `[128,128]` blocks of M2.5 and GLM-5.3 are what make TP a tight fit there; MXFP8's finer granularity removes that pressure.

So the recipe's TP=8 recommendation holds here. Verify it still holds after a vLLM upgrade, since the replication branch is an implementation detail, not a guarantee.

## It is a multimodal model

`minimax_m3_vl` carries a CLIP vision tower, a multimodal projector and image/video token handling. The `/swe3` benchmark is text-only, so the vision stack is dead weight for this workload: it occupies memory and its layers appear in the quantization config's `ignored_layers`. Nothing needs disabling, but do not compare its memory footprint or throughput against a text-only model of the same parameter count without noting this.

## Benchmark it

Add a registry row to [run-multi-model-benchmark.sh](../../../benchmarks/scripts/run-multi-model-benchmark.sh) in the documented column order (`served_name | HF repo | max_model_len | tool_parser | tp | fits | gpu_mem_util | reasoning_parser | extra_args | extra_env`):

```text
"minimax-m3|MiniMaxAI/MiniMax-M3-MXFP8|262144|minimax_m3|8|p5en.48xl|0.90|minimax_m3|--trust-remote-code --block-size 128|"
```

**Never edit that script while a benchmark is running.** Bash reads a script incrementally, so an in-place edit can make the live shell execute garbage. Wait for the batch to finish, or copy the script.

Then:

```bash
./scripts/run-multi-model-benchmark.sh minimax-m3 \
  --agent omp --skill swe3 \
  --dataset dataset/mcp-gateway-registry-v2.yaml \
  --judge-mode async --timeout-seconds 7200 --dollars-per-hour 27.72
```

## Runtime observed

Nothing yet. Record on first bring-up: weights size and download time, startup duration, `GPU KV cache size`, maximum concurrency at the served window, and the mean minutes per `/swe3` task. Those are what the next person needs, and what the cost-per-task derivation reads.

## Comparison with the other frontier models on this node

| Model | Quantization | TP | Served window | Status |
|-------|--------------|----|---------------|--------|
| Kimi-K2.7-Code | block FP8 | 8 | 131072 | measured |
| GLM-5.3 | block FP8 + FP8 KV cache | 8 | 300000 | measured |
| Qwen3-Coder-480B | block FP8 | 4 | 200000 | measured |
| MiniMax-M2.5 | block FP8 | 4 | 196608 | measured |
| **MiniMax-M3** | **MXFP8 (444 GB)** | **8** | **262144 (proposed)** | **not yet served** |
