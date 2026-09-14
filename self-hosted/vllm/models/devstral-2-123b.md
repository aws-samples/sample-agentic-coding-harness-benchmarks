# Devstral-2-123B -- serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model.

| | |
|---|---|
| **HF repo** | `mistralai/Devstral-2-123B-Instruct-2512` |
| **Model card** | [huggingface.co/mistralai/Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512) |
| **Type** | Dense transformer -- 123B parameters (all active) |
| **Weights on disk** | ~128 GB (FP8, native quantization) |
| **Minimum hardware** | 4×H200 141GB (half of p5en.48xlarge, TP=4) |
| **Fits 4×L40S (184 GB)?** | No (weights fit but context window shrinks to ~64K) |
| **Tool-call parser** | `mistral` |
| **Reasoning parser** | none |
| **Native context** | **262,144 (256K)** |
| **Role** | Mistral AI's coding-focused dense model; strong on agentic tasks |

## Serve it

**Benchmarked config:** instance `p5en.48xlarge` (using 4 of 8 H200 GPUs, TP=4) · precision **FP8** (native) · 262144 (256K) window.

```bash
MODEL="mistralai/Devstral-2-123B-Instruct-2512" \
SERVED_NAME="devstral-2-123b" \
TP=4 \
PORT=8000 \
MAX_MODEL_LEN=262144 \
GPU_MEM_UTIL=0.90 \
TOOL_PARSER="mistral" \
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

vllm serve mistralai/Devstral-2-123B-Instruct-2512 \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name devstral-2-123b \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser mistral \
  --enable-prefix-caching \
  2>&1 | tee logs/vllm-serve.log
```

## Instance and access

| | |
|---|---|
| **Instance type** | p5en.48xlarge (using 4 of 8×H200 141GB) |
| **Region** | us-east-2 |
| **Cost** | ~$63.30/hr on-demand (whole node); TP=4 frees 4 GPUs for concurrent workloads |
| **SSH** | `ssh -i ~/.ssh/mcp-gateway-key.pem ubuntu@<IP>` |

## Disk requirements

Weights are ~128 GB on disk. Ensure at least **256 GB free** before downloading.

## Architecture notes

Devstral-2-123B is a dense model (all 123B parameters active per token), unlike the MoE models in this directory. It ships natively in FP8 format -- no additional quantization needed, and no `--quantization` flag required for vLLM.

Using TP=4 (half the p5en node) leaves the other 4 GPUs available for a second model or a second benchmark run in parallel.

## Tuning notes

- **No `--trust-remote-code` needed** -- Mistral architecture is natively supported.
- **Startup time:** First boot downloads ~128 GB + weight loading. Allow 5-10 minutes. Subsequent boots: ~2-3 minutes.
- **TP=4 vs TP=8:** Quality is identical; TP=4 is preferred to free GPUs. TP=8 is fine if you want more KV cache headroom for very long contexts.
- **Prefix caching:** Effective for repeated system prompts.
