# GLM-5.3 — serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model.

| | |
|---|---|
| **HF repo** | `zai-org/GLM-5.3` — **natively FP8**, no `-FP8` suffix. BF16 lives under `zai-org/GLM-5.3-BF16` |
| **Model card** | [huggingface.co/zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) |
| **vLLM recipe** | [recipes.vllm.ai/zai-org/GLM-5.3](https://recipes.vllm.ai/zai-org/GLM-5.3) |
| **Type** | MoE — 743B total, **39B active per token**; `GlmMoeDsaForCausalLM` (DSA sparse attention), 78 layers, 256 routed experts (8 active) + 1 shared, MLA (`kv_lora_rank` 512), 1 MTP layer |
| **FP8 weights** | ~756 GB (141 safetensor shards) |
| **BF16 weights** | ~1,500 GB (does NOT fit 8×H200 — multi-node only) |
| **Minimum hardware** | 8×H200 141GB (p5en.48xlarge) or 8×H20; 8×B200 180GB for the full 1M window |
| **Fits 4×L40S (184 GB)?** | No |
| **Requires** | vLLM >= 0.28.0, `transformers` >= 5.15.0 |
| **Tool-call parser** | `glm47` |
| **Reasoning parser** | `glm47` (the vLLM recipe says `glm45`; **both names are aliases for the same `Glm47MoeParserReasoningAdapter`** — do not go looking for a difference) |
| **Native context** | **1,048,576 (1M)** |
| **Role** | Frontier open-weights coding model — 88.2 on Terminal-Bench 2.1, 28.3 on Terminal-Bench 3.0, 66.9 on DeepSWE |

**GLM-5.3 is the same base model as [GLM-5.2](glm-5.2.md); every gain comes from post-training.** Architecture, shape, and serving flags are identical (both are `GlmMoeDsaForCausalLM`, 78 layers, 256 experts, hidden 6144, vocab 154880), so everything already learned about serving 5.2 on this node carries over unchanged. Three things actually differ: the checkpoint is FP8 by default (the plain repo name, not a `-FP8` variant), `reasoning_effort` gained a third level, and the vLLM recipe now recommends MTP speculative decoding plus an FP8 KV cache.

## Serve it

> **Not yet run on this node.** Unlike the GLM-5.2 file, the config below is not a benchmarked config — it is the [official vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3) adapted to this repo's script. Treat the context-window numbers as arithmetic, not measurement.

**Intended config:** instance `p5en.48xlarge` (8×H200 141GB) · **TP=8** (all 8 GPUs) · precision **FP8** (native) · FP8 KV cache · 5-token MTP.

**Only one of these models can be resident at a time.** GLM-5.3's weights are ~94.5 GB per GPU, so a running GLM-5.2 server must be stopped first — there is no room for both on one node.

> **The p5en CUDA prerequisites are identical to GLM-5.2's and still apply.** This DLAMI has **no `/usr/local/cuda`** (nvcc lives at `/opt/pytorch/cuda`), and the FP8 path JIT-compiles kernels needing `ninja` on PATH plus three unversioned CUDA libs (`libcudart.so`, `libcuda.so`, `libnvrtc.so`) in `$CUDA_HOME/lib64`. Export the environment from [`.claude/skills/vllm-setup/p5en-h200-cuda-fixes.md`](../../../.claude/skills/vllm-setup/p5en-h200-cuda-fixes.md) (Fixes 1+2+3) **before** serving, or engine init fails with `cannot find -lnvrtc`.

```bash
cd self-hosted/vllm

MODEL="zai-org/GLM-5.3" \
SERVED_NAME="glm-5.3" \
TP=8 \
PORT=8000 \
MAX_MODEL_LEN=300000 \
GPU_MEM_UTIL=0.95 \
TOOL_PARSER="glm47" \
REASONING_PARSER="glm47" \
EXTRA_ARGS="--trust-remote-code --kv-cache-dtype fp8 --speculative-config.method mtp --speculative-config.num_speculative_tokens 5" \
  ./scripts/vllm-serve.sh
```

`vllm-serve.sh` has no dedicated env var for the KV-cache dtype or the speculative config, which is why both ride in `EXTRA_ARGS`.

Or the raw vLLM command:

```bash
cd self-hosted/vllm
mkdir -p logs

vllm serve zai-org/GLM-5.3 \
  --tensor-parallel-size 8 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name glm-5.3 \
  --max-model-len 300000 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 5 \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --enable-prefix-caching \
  --trust-remote-code \
  2>&1 | tee logs/vllm-serve.log
```

## Instance and access

Same node as GLM-5.2 — see [glm-5.2.md](glm-5.2.md#instance-and-access) for SSH and tunnel commands.

| | |
|---|---|
| **Instance type** | p5en.48xlarge (8×H200 141GB, 1.13 TB VRAM) |
| **Cost** | ~$63.30/hr on-demand (AWS Price List API, us-east-1; see [pricing.json](../pricing.json)) |
| **Disk** | ~756 GB of weights. `HF_HOME` defaults to `/opt/dlami/nvme/hf-cache` (27 TB free) — do not override it to the root disk, which has < 10 GB free. That NVMe scratch is **ephemeral**: weights re-download after an instance stop. |

## Context window reality

The model is 1M-native, but KV cache VRAM is the binding constraint. Weights take ~88 GiB of each GPU's 140.4 GiB, and MLA means the KV cache is **replicated on every TP rank** rather than sharded — so per-GPU KV cost is the whole context, roughly **88 KiB/token at BF16** and **44 KiB/token at FP8**.

Extrapolating from GLM-5.2's measured ceiling on this node (307,840 tokens with ~26.5 GiB free per GPU at `0.95`), GLM-5.3's ~1 GiB/GPU of extra weight shifts things slightly:

| `max-model-len` | KV dtype | Fits? | Notes |
|----------------|----------|-------|-------|
| 300,000 | fp8 | Yes, comfortably | The config above. Roughly half the KV budget of BF16. |
| ~285,000 | bf16 | Borderline | 5.2 topped out at 307,840; 5.3 is slightly heavier, so BF16 KV likely lands just under 300K. |
| ~570,000 | fp8 | Probably | Arithmetic only. Untested. |
| 1,000,000 | fp8 | No | Needs ~43 GiB/GPU of KV vs ~25 GiB free. The recipe calls for 8×B200 (180 GB) here. |

`--max-num-seqs` is the other lever: it bounds how many sequences share the KV budget at once. The recipe suggests starting at 32 when pushing toward very long windows. **Confirm the real ceiling from vLLM's own KV-cache estimate at boot rather than from this table.**

## Thinking / reasoning effort

GLM-5.3 has **three** levels (5.2 had two), via `reasoning_effort`:
- **`max`** (default) — deepest reasoning; what the published benchmark numbers use
- **`high`** — balanced depth and latency
- **`low`** — lightest, lowest token cost

The chat template falls back to `max` for any value that is not exactly `low` or `high`. Pass it via the top-level OpenAI `reasoning_effort` field or `chat_template_kwargs`. Thinking is always on — the generation prompt opens a `<think>` block unconditionally; `--reasoning-parser glm47` keeps it out of visible output.

One 5.3-specific chat-template default: **`clear_thinking` defaults to `false`.** For chat-style use the model card says to pass `clear_thinking=true` explicitly.

## Tool calling

Uses the `glm47` parser, same as 5.2. Tool calls come back as structured `tool_use` blocks via the Anthropic messages API (`/v1/messages`).

## Tuning notes

- **MTP speculative decoding:** the checkpoint ships `num_nextn_predict_layers: 1` with `index_share_for_mtp_iteration: true`, and the recipe drafts **5** tokens from that single layer. This is the main throughput lever and is new relative to how 5.2 is served here. Note that synthetic throughput benchmarks under-report its benefit, because MTP acceptance rates are low on random prompts — measure on real agent traffic.
- **DeepGEMM is required** for FP8 performance (already present on this node — `deep_gemm.py` shows PDL and E8M0 enabled in the serve log).
- **Download speed:** ~756 GB. The repo is ungated, but without `HF_TOKEN` downloads are rate-limited; set a token.
- **Startup time:** expect 5.2's profile or worse — roughly 15-20 min on a cold cache (download dominates), ~5-8 min once weights are local, plus MTP adding a little to graph capture.
- **Prefix caching:** keep it on for `/swe` and `/implement` benchmarks, where the system prompt and skill instructions are constant across turns. Add `--no-enable-prefix-caching` only for clean throughput measurement.
- **NVFP4 on Blackwell:** `Inferact/GLM-5.3-NVFP4` re-quantizes only the MoE expert linears (~465 GB), leaving attention, shared experts, embeddings, and early dense layers in BF16. Irrelevant on H200, relevant if this ever moves to B200.

## Comparison with other frontier models

Model-card numbers (`max` reasoning effort throughout):

| Model | Terminal-Bench 2.1 | Terminal-Bench 3.0 | DeepSWE (v1.1) | CyberGym |
|-------|-------------------|--------------------|----------------|----------|
| **GLM-5.3** | 88.2 | 28.3 | 66.9 | **84.5** |
| GLM-5.2 | 81.0 | 4.6 | 46.2 | 77.2 |
| Kimi K3 | 88.3 | 17.4 | 67.5 | 80.0 |
| Claude Opus 4.8 | 85.0 | 21.1 | 58.0 | 78.1 |
| Fable 5 (w/ fallback) | 88.0 | 33.7 | 69.7 | 83.8 |
| GPT-5.6 Sol | **88.8** | **34.6** | **72.7** | 83.6 |

The Terminal-Bench 3.0 jump (4.6 to 28.3) is the headline: on the harder agentic benchmark 5.2 was barely functional and 5.3 is competitive with closed frontier models. Z-AI also reports emergent cyber capability — SOTA on CyberGym, with the largest gains further up the exploitation chain.
