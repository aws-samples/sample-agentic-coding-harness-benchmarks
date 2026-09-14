# Qwen3.6-35B-A3B — serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model.

| | |
|---|---|
| **HF repo** | `Qwen/Qwen3.6-35B-A3B` |
| **Model card** | [huggingface.co/Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| **Type** | MoE — 35.9B total, **3B active per token** |
| **BF16 weights** | ~72 GB |
| **Fits 4×L40S (184 GB)?** | ✅ |
| **Tool-call parser** | `qwen3_coder` (verify against the model card at serve time) |
| **Native context** | **262144 (256K)** — extensible to ~1M with YaRN |
| **Role** | 3.6-generation MoE — newer than the 30B coder default, same 3B-active economics |

## Serve it

This model is **256K-native**, so we default it to a **200K (200000) window** — below native (no rope scaling needed) yet a very long context. Set `MAX_MODEL_LEN` explicitly since it is far above the script's own 32768 default:

```bash
cd self-hosted/vllm/scripts
MODEL=Qwen/Qwen3.6-35B-A3B SERVED_NAME=qwen3.6-35b MAX_MODEL_LEN=200000 ./vllm-serve.sh
```

Wrapper with every model-specific parameter spelled out:

```bash
MODEL="Qwen/Qwen3.6-35B-A3B" \
SERVED_NAME="qwen3.6-35b" \
TP=4 \
PORT=8000 \
MAX_MODEL_LEN=200000 \
GPU_MEM_UTIL=0.90 \
TOOL_PARSER="qwen3_coder" \
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

~/vllm-env/bin/vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen3.6-35b \
  --max-model-len 200000 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  2>&1 | tee logs/vllm-serve.log
```

> **Why 200000 and not the full 256K?** `MAX_MODEL_LEN` is a hard ceiling you set — it does not auto-expand to native. 200K stays just under the 256K native window (so no rope scaling) while leaving a little KV-cache headroom; the full 256K would consume so much KV cache it may not boot at useful concurrency. If you hit OOM or see `Maximum concurrency ... 1x` at boot, lower it (e.g. `131072` or `65536`). See the long-context tuning note below.
>
> **Single-request benchmarking — use the full 262144.** The 200K default and the "may not boot at useful concurrency" caution above are written for multi-tenant serving, where every concurrent request needs its own KV-cache slice. The benchmark harness runs at `concurrency: 1` (see `benchmarks/config/runner.yaml`), so only one request is ever in flight and that caution does not apply — serve the full native window instead:
>
> ```bash
> MODEL="Qwen/Qwen3.6-35B-A3B" SERVED_NAME="qwen3.6-35b" TP=4 PORT=8000 \
>   MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.90 TOOL_PARSER="qwen3_coder" ./vllm-serve.sh
> ```
>
> Measured on 4xL40S at boot: **GPU KV cache 2.15M tokens, max concurrency 8.21x** — no OOM, no `1x` warning. This matters because a real SWE task can exceed 200K in a single request: Claude Code reserves `max_output_tokens` (16000) on top of the prompt, so a 184,001-token prompt hits `184001 + 16000 = 200001`, one token over a 200K window. 262144 gives ~46K more input headroom at zero cost here.
>
> Serving the full window is only half the fix — the harness must also tell Claude Code the window so it auto-compacts *before* overflowing it, otherwise the conversation still grows into the ceiling (now 262145) and the client retries the resulting 500 forever. The orchestrator does this automatically on the vllm path (it reads the live `max_model_len` and passes `--context-window`); see [Context window and auto-compaction](../../../benchmarks/docs/harness-reference.md#context-window-and-auto-compaction).

## How it compares

Same **3B-active MoE economics** as the 30B coder default — per-token compute tracks the 3B active count, not the 35.9B total — so throughput and cost per token stay in the favorable regime the [strategy cost model](../../../README.md) depends on. The extra ~11 GB of weights over the 30B (~72 GB vs ~61 GB) leaves marginally less room for the KV cache, so expect slightly lower max concurrency at the same context length. Use this file's model when you want the newer 3.6-generation quality without giving up the MoE throughput advantage.

## Tuning notes

- **Tool-call parser:** `qwen3_coder` is the expected parser for this MoE family, but **confirm against the HF model card** when you first serve it — if agentic tool calls fail to parse, try `hermes`. Run `vllm serve --help` for the full parser list.
- **Long context — this model is native 256K, so most of the time you need NO rope scaling.** Per the [HF model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), the native window is **262144 tokens (256K)**, extensible to ~1,010,000 tokens with YaRN. That is very different from the 32K-native Qwen3 models in this folder:
  - **Up to 256K — just raise `MAX_MODEL_LEN`, no `ROPE_SCALING`.** YaRN is static and can hurt short-prompt quality, so only enable it when you genuinely exceed native. On 4×L40S the real limit here is KV-cache VRAM, not the model — 256K context will not fit at meaningful concurrency, so pick the largest window your VRAM allows and stop there. To test a longer window (e.g. 128K, comfortably below native):
    ```bash
    MODEL=Qwen/Qwen3.6-35B-A3B SERVED_NAME=qwen3.6-35b \
      MAX_MODEL_LEN=131072 ./vllm-serve.sh
    ```
    The exact `vllm serve` command this builds — export the two DLAMI fixes first (the wrapper sets these for you) or the engine may fail to boot on this node:
    ```bash
    cd self-hosted/vllm
    mkdir -p logs
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export CUDA_HOME=/opt/pytorch/cuda

    ~/vllm-env/bin/vllm serve Qwen/Qwen3.6-35B-A3B \
      --tensor-parallel-size 4 \
      --host 127.0.0.1 \
      --port 8000 \
      --served-model-name qwen3.6-35b \
      --max-model-len 131072 \
      --gpu-memory-utilization 0.90 \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --enable-prefix-caching \
      2>&1 | tee logs/vllm-serve.log
    ```
  - **Past 256K (up to ~1M) — enable YaRN through `--hf-overrides`.** On vLLM `0.25.1`, the old `--rope-scaling` server flag is no longer accepted. Patch the Hugging Face config instead with `--hf-overrides '{"rope_scaling": ...}'`, and set `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` so vLLM permits a `MAX_MODEL_LEN` above the model config's native 262144. The model card's base config is `factor 4.0` on `original_max_position_embeddings 262144` (→ ~1,010,000 tokens); use a smaller factor for a smaller target (e.g. `2.0` for ~512K). This is academic on this node — 4×L40S has nowhere near the VRAM to hold a >256K KV cache. **Do not use the bare-number `ROPE_SCALING=4` shorthand here** — it would infer the wrong native window (`MAX_MODEL_LEN ÷ 4`). The fixed wrapper accepts the same full JSON and maps it to `--hf-overrides`; the direct command below shows the exact vLLM invocation:
    ```bash
    cd self-hosted/vllm
    mkdir -p logs
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export CUDA_HOME=/opt/pytorch/cuda
    export HF_HOME=/opt/dlami/nvme/hf-cache
    export HF_HUB_CACHE=/opt/dlami/nvme/hf-cache/hub
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

    ~/vllm-env/bin/vllm serve Qwen/Qwen3.6-35B-A3B \
      --tensor-parallel-size 4 \
      --host 127.0.0.1 \
      --port 8000 \
      --served-model-name qwen3.6-35b \
      --max-model-len 1010000 \
      --gpu-memory-utilization 0.90 \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --enable-prefix-caching \
      --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}' \
      2>&1 | tee logs/vllm-serve.log
    ```
    Keep the JSON single-quoted so the shell doesn't split it.
  - **KV-cache reality:** context length is 4× the KV cache per request when you 4× the window, so concurrency drops proportionally. Because this is a 3B-active MoE (~72 GB weights), it still leaves more KV-cache room than the dense [Qwen3-32B](qwen3-32b.md) or the tight-fit [Qwen3-Coder-Next](qwen3-coder-next.md) — the better long-context choice of the three — but VRAM, not the model's native window, is your ceiling here. If vLLM reports `Maximum concurrency ... 1x` or can't fit even one request's KV cache at boot, lower `MAX_MODEL_LEN`. See [PagedAttention + KV cache](../README.md#pagedattention--kv-cache) and [Long context past 32K](../README.md#long-context-and-rope_scaling-yarn).

  > **Heads-up on the `ROPE_SCALING` shorthand.** The bare-number form assumes native = `MAX_MODEL_LEN ÷ factor`, which is wrong for this model unless those values happen to match its native 262144 window. Pass the full JSON shown above. The wrapper maps that JSON to `--hf-overrides` on vLLM `0.25.1`.
- **Concurrency:** between the 30B coder and the dense 32B — more headroom than the dense model (3B active), slightly less than the 30B (larger weights).

## Naming note

Distinct from [Qwen3-32B](qwen3-32b.md), which is a **dense** 32.8B model from the earlier generation. This is a 35B **MoE** from the 3.6 generation — different architecture and different economics.

Also distinct from [qwen3.6-35b-a3b-fp8.md](qwen3.6-35b-a3b-fp8.md), which serves the **FP8** weights of this same model on a **single L40S** (`g6e.4xlarge`, TP=1) and is served as `qwen3.6-35b-fp8`. Same model, different precision and hardware, so its numbers are a separate result — do not merge them into this file's. This file's BF16 4x L40S configuration is the one the published results were measured on.
