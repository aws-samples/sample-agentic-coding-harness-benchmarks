# Gemma-4-31B-it — serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model.

| | |
|---|---|
| **HF repo** | `google/gemma-4-31B-it` |
| **Model card** | [huggingface.co/google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) |
| **Type** | Dense, multimodal (text + image/audio input) — `Gemma4ForConditionalGeneration` |
| **BF16 weights** | ~63 GB (2 safetensors shards) |
| **Fits 4×L40S (184 GB)?** | ✅ comfortably — leaves ~120 GB for KV cache |
| **Tool-call parser** | `gemma4` (ships with vLLM 0.25.1; required for agentic clients) |
| **Native context** | **262144 (256K)** |
| **Role** | Google's instruction-tuned Gemma 4; a dense alternative to the Qwen MoE models on this node |

## Serve it

This model is **256K-native**, so we default it to a **200K (200000) window** — below native (no rope scaling needed), and it clears the benchmark's 200K context-window gate. Set `MAX_MODEL_LEN` explicitly since it is far above the script's 32768 default. The tool-call parser is `gemma4` (verify against the model card at serve time):

```bash
cd self-hosted/vllm/scripts
MODEL=google/gemma-4-31B-it SERVED_NAME=gemma-4-31b MAX_MODEL_LEN=200000 \
  TOOL_PARSER=gemma4 ./vllm-serve.sh
```

Wrapper with every model-specific parameter spelled out:

```bash
MODEL="google/gemma-4-31B-it" \
SERVED_NAME="gemma-4-31b" \
TP=4 \
PORT=8000 \
MAX_MODEL_LEN=200000 \
GPU_MEM_UTIL=0.90 \
TOOL_PARSER="gemma4" \
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

~/vllm-env/bin/vllm serve google/gemma-4-31B-it \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name gemma-4-31b \
  --max-model-len 200000 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser gemma4 \
  --enable-prefix-caching \
  2>&1 | tee logs/vllm-serve.log
```

## Notes

- **Multimodal model, used here for text.** `Gemma4ForConditionalGeneration` accepts image and audio input, but the SWE benchmark drives it purely as a text LLM through Claude Code. The extra vision/audio towers add to the resident weights but are not exercised by `/swe`.
- **Dense, not MoE.** Unlike the Qwen3-Coder / Qwen3.6 MoE models on this node, every parameter is active per token, so per-token compute is higher for the same weight budget — expect lower throughput than a 3B-active MoE of similar size. It still fits comfortably at ~63 GB.
- **Verify vLLM support before a long download.** This model needs a vLLM new enough to implement `Gemma4ForConditionalGeneration` and to ship the `gemma4` tool parser; both are present in **0.25.1** (the version on this node). On an older vLLM the serve will fail to load the architecture, or tool calls will not parse.

## Tuning notes

- **Tool calling:** use the `gemma4` parser. Agentic clients (Claude Code, opencode) require `--enable-auto-tool-choice --tool-call-parser gemma4`; without it, tool calls will not be parsed and every `/swe` task fails.
- **Context window — native 256K; we serve 200000 (200K) here.** Per the [HF model card](https://huggingface.co/google/gemma-4-31B-it) the native window is **262144 (256K)**. `MAX_MODEL_LEN` is a hard ceiling you set, not auto-expanded to native; 200000 stays below native (no `ROPE_SCALING` needed) while leaving KV-cache headroom on 4×L40S. Watch the `Maximum concurrency` line at boot; lower `MAX_MODEL_LEN` if it reports `1x` or OOMs.
- **Attention:** the model uses sliding-window attention on most layers (window 1024) with periodic full-attention layers, which keeps the KV cache smaller than a fully-global-attention model at the same context length.
