# Kimi-K3 — serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model.

| | |
|---|---|
| **HF repo** | `moonshotai/Kimi-K3` |
| **Model card** | [huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) |
| **Type** | MoE — 2.8T total, 104B active (16 of 896 experts + 2 shared); native **MXFP4** weights / MXFP8 activations (quantization-aware trained) |
| **Weights on disk** | ~1.4 TB (native MXFP4) |
| **Minimum hardware** | Fits **8×B300 (p6-b300.48xlarge, ~2.15 TiB)** at TP=8 — **verified serving on this repo's node**. Also runs as a 16×B200 TEP16 slice of a P6e-GB200 (the vLLM recipe target); 8×B200 (1.5 TB) is too small, but 8×B300 (268 GB/GPU) is not. |
| **Fits 8×B300 (2.15 TiB)?** | **Yes (verified)** — weights load at **192.6 GiB/GPU** (~1.54 TB total), leaving ~32 GiB/GPU for KV. 8×**B200** (192 GB) would not fit; the B300's 268 GB cards are the difference. |
| **Tool-call parser** | `kimi_k3` (**verified** — vLLM 0.28 ships `kimi_k3_tool_parser`. Do **not** use `kimi_k2`: it mis-parses K3 and the chat `content` comes back empty) |
| **Reasoning parser** | `kimi_k3` (**verified** — vLLM 0.28 ships `kimi_k3_reasoning_parser`; K3 is a thinking model, so give it enough `max_tokens` to finish reasoning before the answer) |
| **Native context** | **1,048,576 (1M)** |
| **Role** | Frontier open-weight MoE from the Kimi K3 family — 2.8T QAT-MXFP4, 1M context |

## Serve it

### Verified: single node, 8×B300 (p6-b300.48xlarge), TP=8

This is what actually served on this repo's p6-b300 node (2026-09, vLLM 0.28.0, stock `~/vllm-env` — **no special container, no multi-node, no expert-parallel needed**). B300 cards are 268 GB, so all 8 hold the ~1.54 TB of weights at TP=8.

```bash
# B300/NVSwitch node fixes — see .claude/skills/vllm-setup/b300-cuda-fixes.md
export CUDA_HOME=/opt/pytorch/cuda
export PATH="$HOME/vllm-env/bin:$CUDA_HOME/bin:$PATH"          # Fix: ninja + nvcc on PATH (FlashInfer JIT)
LINKDIR=/opt/dlami/nvme/cuda-link                              # Fix: unversioned CUDA libs for the JIT link step
VENV_CU13="$HOME/vllm-env/lib/python3.14/site-packages/nvidia/cu13/lib"
mkdir -p "$LINKDIR" "$CUDA_HOME/lib64/stubs"
ln -sf "$VENV_CU13/libcudart.so.13" "$LINKDIR/libcudart.so";  ln -sf "$VENV_CU13/libcudart.so.13" "$CUDA_HOME/lib64/libcudart.so"
ln -sf "$VENV_CU13/libnvrtc.so.13"  "$LINKDIR/libnvrtc.so";   ln -sf "$VENV_CU13/libnvrtc.so.13"  "$CUDA_HOME/lib64/libnvrtc.so"
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$LINKDIR/libcuda.so"; ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_HOME/lib64/libcuda.so"; ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_HOME/lib64/stubs/libcuda.so"
export LIBRARY_PATH="$LINKDIR:$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$VENV_CU13:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LINKDIR:$CUDA_HOME/lib64:$VENV_CU13:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_TOKEN="$(cat ~/.hf_token)"
# NVLS left ENABLED — K3 requires multicast. If NCCL/init fails with CUDA error 401, reset the GPUs (b300-cuda-fixes.md Fix 1), do NOT set NCCL_NVLS_ENABLE=0.

MODEL="moonshotai/Kimi-K3" SERVED_NAME="kimi-k3" \
TP=8 PORT=8000 MAX_MODEL_LEN=131072 GPU_MEM_UTIL=0.90 \
TOOL_PARSER="kimi_k3" REASONING_PARSER="kimi_k3" \
EXTRA_ARGS="--trust-remote-code" \
  ./vllm-serve.sh
```

**Verified runtime (this node):** MoE backend `DEEPGEMM_MXFP4`, attention `FLASHINFER_MLA` + `FlashKDA` (Kimi Delta Attention — K3 is a hybrid linear-attention model, not plain DeepSeek-V3+MLA), fused allreduce `mnnvl`. Model load **192.6 GiB/GPU** (~1.54 TB). **GPU KV cache ~1.35M tokens, max concurrency 10.3× at 131,072.**

**First-boot cost is large and one-time (cached after):** on this brand-new arch (sm_103a) the first launch spends ~30 min single-threaded `ptxas` compiling the trtllm fused-MoE kernel, then ~24 min FlashInfer autotuning `trtllm_fp4_block_scale_moe` (22 configs), then CUDA-graph capture (83 sizes). Budget **~60–75 min for the very first boot**; subsequent boots reuse the caches and are far shorter. Raise `MAX_MODEL_LEN` (up to 1M native) only as KV headroom allows.

### Alternative: 16×B200 multi-node (P6e-GB200 / vLLM recipe)

**Proposed config:** UltraServer `P6e-GB200` (B200 192GB, one NVLink domain) · **TEP16** — `--tensor-parallel-size 16 --enable-expert-parallel` across 16 GPUs · precision **native MXFP4** (MXFP8 activations, as published) · **FP8 KV cache** · 256K window to start.

Kimi K3 runs on Blackwell's native FP4 datapath, so MXFP4 is the intended format here — do not dequantize to FP8/BF16 (2.8 TB / 5.6 TB) unless you have a specific reason; you spend more memory for no quality gain over the QAT baseline. Requires the K3 container (CUDA 13, r580+ driver) and vLLM 0.27.1+.

> **Two things this node needs before the command works.** (1) **Multi-node:** 16 B200s span more than one EC2 instance inside the NVL72 domain, so this is a multi-node launch — bring up a Ray cluster (head + workers) first, then run `vllm serve` on the head; TP/EP spans nodes over NVLink. Pin to a single instance's GPUs only if you drop `--tensor-parallel-size` to that count. (2) **Container/driver:** use `vllm/vllm-openai:kimi-k3` (CUDA 13, r580+); the stock DLAMI CUDA on many nodes is older. If your DLAMI needs CUDA-path fixes for JIT (nvcc/ninja), export them before launch the same way the other models in this repo do — see [`.claude/skills/vllm-setup/`](../../../.claude/skills/vllm-setup/) and adapt for the CUDA 13 container.
>
> The pre-release recipe still moves flag names; confirm the TEP16 pairing and whether your build wants an explicit `--data-parallel-size` for attention against `vllm serve --help` in the `:kimi-k3` image and the [recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3).

```bash
MODEL="moonshotai/Kimi-K3" \
SERVED_NAME="kimi-k3" \
TP=16 \
PORT=8000 \
MAX_MODEL_LEN=262144 \
GPU_MEM_UTIL=0.90 \
TOOL_PARSER="kimi_k2" \
REASONING_PARSER="kimi_k2" \
EXTRA_ARGS="--enable-expert-parallel --all2all-backend flashinfer_nvlink_one_sided --moe-backend deep_gemm_mega_moe --kv-cache-dtype fp8 --load-format fastsafetensors --trust-remote-code" \
  ./vllm-serve.sh
```

Or the raw vLLM command (what actually runs on the head node):

```bash
cd self-hosted/vllm
mkdir -p logs

# K3 serving env (from the vLLM K3 day-0 blog + recipe)
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_USE_V2_MODEL_RUNNER=1
export NCCL_DMABUF_ENABLE=0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HF_TOKEN=<your-token>

vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --all2all-backend flashinfer_nvlink_one_sided \
  --moe-backend deep_gemm_mega_moe \
  --kv-cache-dtype fp8 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name kimi-k3 claude-sonnet-4-20250514 us.anthropic.claude-opus-4-6-v1 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --trust-remote-code \
  2>&1 | tee logs/vllm-serve.log
```

Run it through the K3 container instead of a host venv:

```bash
docker run --gpus all --rm -it --network host --ipc host \
  -v $HF_HOME:$HF_HOME -e HF_HOME=$HF_HOME -e HF_TOKEN=$HF_TOKEN \
  -e VLLM_USE_RUST_FRONTEND=1 -e VLLM_ALLREDUCE_USE_FLASHINFER=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e NCCL_DMABUF_ENABLE=0 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  vllm/vllm-openai:kimi-k3 \
  --model moonshotai/Kimi-K3 --tensor-parallel-size 16 --enable-expert-parallel \
  --all2all-backend flashinfer_nvlink_one_sided --moe-backend deep_gemm_mega_moe \
  --kv-cache-dtype fp8 --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 \
  --enable-prefix-caching --load-format fastsafetensors --trust-remote-code
```

## Instance and access

| | |
|---|---|
| **Instance type** | P6e-GB200 UltraServer (B200 192GB; NVL72 domain — x36 = 36 GPU / 6.66 TB, x72 = 72 GPU / 13.3 TB) |
| **Region** | us-east-1 *(verify — P6e-GB200 is offered in limited AZs, usually via Capacity Blocks)* |
| **Cost** | Reserved through EC2 **Capacity Blocks** / UltraServer reservation — *verify the hourly rate via the AWS Price List API before committing; it is not a simple on-demand number* |
| **SSH** | `ssh -i ~/.ssh/qwen36-key.pem ubuntu@<head-IP>` |
| **Tunnel** | `ssh -i ~/.ssh/qwen36-key.pem -L 8000:127.0.0.1:8000 ubuntu@<head-IP>` |

## Download the model

The weights are ~1.4 TB. Pull them with the HF CLI before the first serve (or the server will download on boot, which is slower and harder to monitor):

```bash
export HF_TOKEN=<your-token>
export HF_HOME=/opt/dlami/nvme/hf-cache   # put the cache on NVMe, NOT the root disk
pip install -U "huggingface_hub[cli]" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1        # fast parallel download

hf download moonshotai/Kimi-K3 --local-dir-use-symlinks False
```

Point the server at the same `HF_HOME` so it reuses the cache.

## Disk requirements

The weights are ~1.4 TB on disk. Ensure at least **3 TB free** before downloading — HF uses temp files during download needing ~2× the final size. Put the cache on the instance NVMe, not the small root volume.

If disk fills during download:
```bash
# Delete other cached models to free space (check HF_HOME first!)
rm -rf $HF_HOME/hub/models--moonshotai--Kimi-K2.7-Code
```

Check `HF_HOME` before running that: if the cache is on the ephemeral NVMe (`/opt/dlami/nvme/hf-cache/hub/...`) rather than `$HOME`, the default-path command frees nothing. Evicting a model also costs a full re-download if a later run needs it — on a large NVMe several models fit, so evict only when the volume is genuinely full.

## Quantization

Kimi K3 ships **native MXFP4** (weights) with MXFP8 activations, quantization-aware trained from the SFT stage on. vLLM detects the checkpoint format automatically — **no `--quantization` flag, and no weight `--dtype`**. Confirm the FP4 MoE path in the log at startup (an MXFP4 / FP4 MoE method line). Do not pass `--quantization fp8` or `bf16`; that dequantizes upward and defeats the point of running on Blackwell.

## Thinking / reasoning

Uses the `kimi_k2`-family reasoning parser *(verify K3's exact name)*. Thinking is separated into a `"type": "thinking"` content block, keeping visible output clean. K3 has strong reasoning for complex code tasks and a 1M context.

## Tool calling

Uses the `kimi_k2`-family tool parser *(verify K3's exact name)*. Tool calls come back as structured `tool_use` blocks via the Anthropic messages API (`/v1/messages`).

## Tuning notes

- **Interconnect:** on the NVL72 domain use `--all2all-backend flashinfer_nvlink_one_sided` (NVLink); switch to `deepep_v2` only for RDMA/cross-rack. Keep `--moe-backend deep_gemm_mega_moe` for the expert-parallel MoE path.
- **Precision:** stay in native MXFP4. FP8 (~2.8 TB) or BF16 (~5.6 TB) only add memory pressure for no quality gain over the QAT baseline.
- **Multi-node:** a 16-GPU TEP16 slice spans instances — launch a Ray cluster first, then `vllm serve` on the head. Set `VLLM_ENGINE_READY_TIMEOUT_S=3600` (already above) so the long init does not time out.
- **HF_TOKEN + hf_transfer:** strongly recommended — the ~1.4 TB pull is rate-limited without a token and much slower without `HF_HUB_ENABLE_HF_TRANSFER=1`.
- **Startup time:** first boot = ~1.4 TB download + weight load + compile. Allow 30-45 minutes. Cached weights: ~10-15 minutes.
- **Attention backend:** MLA (Multi-head Latent Attention), DeepSeek-V3-style — vLLM selects the MLA prefill backend for this arch.
- **Context:** set `--max-model-len 262144` to start; push toward `1048576` once KV headroom on the 16-GPU slice is confirmed, or keep it lower for more concurrent sequences.
- **Prefix caching:** enabled — effective for repeated system prompts across benchmark runs.

## Comparison

| Model | Params (active) | Disk size | Precision | Architecture |
|-------|-----------------|-----------|-----------|--------------|
| Kimi-K3 | 2.8T (104B) | ~1.54 TB (in VRAM; HF ~1.56 TB) | native MXFP4 (QAT) | KDA (Kimi Delta Attention, hybrid linear) + MLA, 896 experts |
| Kimi-K2.7-Code | 1,058B (32B) | ~1 TB | FP8 (compressed-tensors) | DeepSeek V3 + MLA |
| GLM-5.3 | 744B (40B) | ~465 GB | NVFP4 | DeepSeek V3 + IndexShare |
| DeepSeek-V4 | ~1.6T (49B) | ~ FP8 | FP8 | DeepSeek V3 lineage |

*Numbers for Kimi K3 read from the model card and the vLLM day-0 writeup; parser names and the P6e-GB200 price are marked "verify" and should be confirmed on your node before a benchmark run.*
