# Kimi-K2.7-Code — serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model.

| | |
|---|---|
| **HF repo** | `moonshotai/Kimi-K2.7-Code` |
| **Model card** | [huggingface.co/moonshotai/Kimi-K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code) |
| **Type** | MoE — 1,058.6B total, compressed-tensors (FP8/Marlin quantized) |
| **Weights on disk** | ~1 TB |
| **Minimum hardware** | 8×H200 141GB (p5en.48xlarge) |
| **Fits 4×L40S (184 GB)?** | ❌ |
| **Tool-call parser** | `kimi_k2` |
| **Reasoning parser** | `kimi_k2` |
| **Native context** | **131,072 (128K)** |
| **Role** | Frontier code-focused model from the Kimi K2 family — coding-optimized variant of K2.6 |

## Serve it

**Benchmarked config:** instance `p5en.48xlarge` (8×H200 141GB) · **TP=8** (all 8 GPUs) · precision **FP8** (compressed-tensors / Marlin, as published) · 131072 (128K) window.

Kimi-K2.7-Code on 8×H200 (p5en.48xlarge). Requires `--trust-remote-code` and benefits from `CUDA_HOME` set for DeepGemm JIT.

> **Verified on this repo's p5en.48xlarge node (2026-08):** the "Serve it" block below is correct as-is, but the DLAMI here has **no `/usr/local/cuda`** (nvcc lives at `/opt/pytorch/cuda`) and `ninja` exists only inside the vLLM venv, so the DeepGemm JIT cannot link. Export the environment from [`.claude/skills/vllm-setup/p5en-h200-cuda-fixes.md`](../../../.claude/skills/vllm-setup/p5en-h200-cuda-fixes.md) (Fixes 1+2) **before** running either command below, or the server fails at engine init with `cannot find -lcudart`. Unlike GLM-5.2, Kimi does **not** need the `libnvrtc.so` symlink (Fix 3), so Fixes 1+2 are enough. The `/usr/local/cuda` exports in the raw command and in "Tuning notes" assume a different DLAMI layout and do not apply to this node -- the venv also lives on the NVMe here (`/opt/dlami/nvme/vllm-env`), not `$HOME`, because the 29 GB root disk cannot hold torch plus the CUDA wheels.
>
> [`benchmarks/scripts/run-multi-model-benchmark.sh`](../../../benchmarks/scripts/run-multi-model-benchmark.sh) applies all of this itself (`_apply_p5en_cuda_env`), so a run driven through that script needs none of these exports by hand.

```bash
MODEL="moonshotai/Kimi-K2.7-Code" \
SERVED_NAME="kimi-k2.7-code" \
TP=8 \
PORT=8000 \
MAX_MODEL_LEN=131072 \
GPU_MEM_UTIL=0.90 \
TOOL_PARSER="kimi_k2" \
REASONING_PARSER="kimi_k2" \
EXTRA_ARGS="--trust-remote-code" \
  ./vllm-serve.sh
```

Or the raw vLLM command (what actually runs on the instance):

```bash
cd self-hosted/vllm
mkdir -p logs
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$HOME/vllm-env/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_TOKEN=<your-token>

vllm serve moonshotai/Kimi-K2.7-Code \
  --tensor-parallel-size 8 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name kimi-k2.7-code claude-sonnet-4-20250514 us.anthropic.claude-opus-4-6-v1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --enable-prefix-caching \
  --trust-remote-code \
  2>&1 | tee logs/vllm-serve.log
```

## Instance and access

| | |
|---|---|
| **Instance type** | p5en.48xlarge (8×H200 141GB, 1.13 TB VRAM) |
| **Region** | us-east-2 |
| **Cost** | ~$63.30/hr on-demand (AWS Price List API, us-east-1; see [pricing.json](../pricing.json)), lower via capacity block |
| **SSH** | `ssh -i ~/.ssh/qwen36-key.pem ubuntu@<IP>` |
| **Tunnel** | `ssh -i ~/.ssh/qwen36-key.pem -L 8000:127.0.0.1:8000 ubuntu@<IP>` |

## Disk requirements

The model weights are ~1TB on disk. Ensure at least **1.5TB free** before downloading (HF uses temp files during download requiring ~2× the final size). The instance EBS should be 2TB.

If disk fills during download:
```bash
# Delete other cached models to free space
rm -rf ~/.cache/huggingface/hub/models--zai-org--GLM-5.2-FP8
```

Check `HF_HOME` before running that: on this repo's p5en node the cache is on the ephemeral NVMe (`/opt/dlami/nvme/hf-cache/hub/...`), not under `$HOME`, so the command above frees nothing there. Evicting GLM-5.2 also costs a ~750 GB re-download if a later run needs it — on a 27 TB NVMe both models fit, so only evict when the volume is genuinely full.

## Quantization

Kimi-K2.7-Code uses `compressed-tensors` format (Marlin WNA16 MoE backend). vLLM detects this automatically — no `--quantization` flag needed. The log should show:
```
Using CompressedTensorsWNA16MarlinMoEMethod
Using 'MARLIN' WNA16 MoE backend
```

## Thinking / reasoning

Uses the `kimi_k2` reasoning parser. Thinking is separated into a `"type": "thinking"` content block, keeping visible output clean. The model has strong reasoning capabilities for complex code tasks.

## Tool calling

Uses the `kimi_k2` tool parser. Tool calls are returned as structured `tool_use` blocks via the Anthropic messages API (`/v1/messages`).

## Tuning notes

- **Slow tokenizer warning:** Expected — Kimi K2 uses a custom tokenizer without a fast Rust implementation. Does not affect inference speed, only tokenization of prompts.
- **DeepGemm:** Same as GLM-5.2 — needs `CUDA_HOME` pointing at a real `bin/nvcc`, and `ninja` on PATH from the vLLM venv (`vllm-install.sh` installs it there, and `vllm-serve.sh` does not add that directory itself). On this repo's p5en node `CUDA_HOME` is `/opt/pytorch/cuda`, not `/usr/local/cuda` — see the callout under "Serve it".
- **HF_TOKEN:** Strongly recommended for faster downloads. Without it, the 1TB download is rate-limited.
- **Startup time:** First boot downloads ~1TB + weight loading + torch.compile. Allow 20-30 minutes. Subsequent boots (weights cached): ~8-10 minutes.
- **Attention backend:** Uses `FLASH_ATTN_MLA` (Multi-head Latent Attention) — same efficient attention as DeepSeek V3 family.
- **Prefix caching:** Enabled by default. Effective for repeated system prompts across benchmark runs.

## Comparison

| Model | Params (active) | Disk size | Architecture |
|-------|----------------|-----------|--------------|
| Kimi-K2.7-Code | 1,058B MoE | ~1 TB | DeepSeek V3 + MLA |
| Kimi-K2.6 | 1,058B MoE | ~1 TB | Same arch, general-purpose |
| GLM-5.2-FP8 | 744B (40B active) | ~750 GB | DeepSeek V3 + IndexShare |
| DeepSeek-V3-0324 | 685B MoE | ~685 GB FP8 | Original DeepSeek V3 |
