# DeepSeek-V3.2 -- serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model.

| | |
|---|---|
| **HF repo** | `deepseek-ai/DeepSeek-V3.2` |
| **Model card** | [huggingface.co/deepseek-ai/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) |
| **Type** | MoE -- 685B total, 37B active per token |
| **Weights on disk** | ~652 GB (FP8) |
| **Minimum hardware** | 8×H200 141GB (p5en.48xlarge) |
| **Fits 4×L40S (184 GB)?** | No |
| **Tool-call parser** | `deepseek_v32` |
| **Reasoning parser** | none (not a thinking model) |
| **Native context** | **131,072 (128K)** |
| **Role** | Frontier open-weight coding and reasoning model from DeepSeek |

## Serve it

**Benchmarked config:** instance `p5en.48xlarge` (8×H200 141GB) · **TP=8** (all 8 GPUs) · precision **FP8** · 131072 (128K) window.

```bash
MODEL="deepseek-ai/DeepSeek-V3.2" \
SERVED_NAME="deepseek-v3.2" \
TP=8 \
PORT=8000 \
MAX_MODEL_LEN=131072 \
GPU_MEM_UTIL=0.90 \
TOOL_PARSER="deepseek_v32" \
  ./vllm-serve.sh
```

Or the raw vLLM command:

```bash
cd self-hosted/vllm
mkdir -p logs
export CUDA_HOME=/opt/pytorch/cuda
export PATH=/opt/pytorch/cuda/bin:$HOME/vllm-env/bin:$PATH
export LD_LIBRARY_PATH=/opt/pytorch/cuda/lib:/opt/pytorch/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_TOKEN=<your-token>

vllm serve deepseek-ai/DeepSeek-V3.2 \
  --tensor-parallel-size 8 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name deepseek-v3.2 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v32 \
  --enable-prefix-caching \
  2>&1 | tee logs/vllm-serve.log
```

## Instance and access

| | |
|---|---|
| **Instance type** | p5en.48xlarge (8×H200 141GB, 1.13 TB VRAM) |
| **Region** | us-east-2 |
| **Cost** | ~$63.30/hr on-demand; lower via capacity block |
| **SSH** | `ssh -i ~/.ssh/mcp-gateway-key.pem ubuntu@<IP>` |

## Disk requirements

Weights are ~652 GB on disk. Ensure at least **1.3 TB free** before downloading. The NVMe scratch at `/opt/dlami/nvme` (27 TB on p5en) is the right location.

## Architecture notes

DeepSeek-V3.2 uses Multi-head Latent Attention (MLA) which compresses the KV cache significantly. vLLM uses the `FLASH_ATTN_MLA` backend automatically. The `deepseek_v32` tool parser handles DeepSeek's native function-calling format.

## Tuning notes

- **No `--trust-remote-code` needed** -- the architecture is natively supported in vLLM.
- **Startup time:** First boot downloads ~652 GB + weight loading. Allow 20-30 minutes. Subsequent boots (weights cached): ~5-8 minutes.
- **Prefix caching:** Highly effective for benchmark runs where the system prompt and skill are constant across turns.
- **CUDA fixes on p5en:** Ensure the CUDA symlinks from `p5en-h200-cuda-fixes.md` are applied before serving.
