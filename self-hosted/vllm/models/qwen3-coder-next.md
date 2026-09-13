# Qwen3-Coder-Next (80B) — serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model.

| | |
|---|---|
| **HF repo** | `Qwen/Qwen3-Coder-Next` |
| **Model card** | [huggingface.co/Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) |
| **Type** | MoE — 79.6B total, **3B active per token** |
| **BF16 weights** | ~160 GB |
| **Fits 4×L40S (184 GB)?** | ✅ **tight** — leaves only ~24 GB for KV cache + CUDA graphs |
| **Tool-call parser** | `qwen3_coder` |
| **Native context** | **262144 (256K)** — but VRAM caps you *far* below this (see below) |
| **Role** | largest model that fits the node — top-tier quality, low concurrency |
| **For agentic coding (>=200K window)** | needs a **g6e.48xlarge** (8xL40S, 384 GB) or larger — a g6e.12xlarge (4xL40S) only fits ~16K, which is unusable for repo-scale tasks |

> **Hardware requirement for benchmarking / agentic coding.** On a **g6e.12xlarge (4xL40S, 184 GB)** this model fits only a ~16384-token window (the ~160 GB of weights leave ~1 GB for KV cache). That is far too small for agentic coding: the `/swe` prompt alone is ~12K tokens, so the very first request overflows the window and every task fails on turn 1 (auto-compaction cannot help — the first message already does not fit). To serve a **200K or 256K** window this model needs a larger-VRAM node — a **g6e.48xlarge (8xL40S, 384 GB)** or bigger — where the weights leave enough room for a repo-scale KV cache. The 16384 config below is only useful for short-prompt smoke tests, **not** the SWE benchmark. For agentic coding on a 4xL40S node, use the 256K-native [Qwen3.6-35B-A3B](qwen3.6-35b-a3b.md) or [Qwen3-Coder-30B](qwen3-coder-30b.md) instead.

## Serve it

This model needs **two** non-defaults, both because it is large *and* a hybrid Mamba model on a VRAM-tight node: reduce the context window (`MAX_MODEL_LEN`) so the KV cache fits, and cap concurrent sequences (`MAX_NUM_SEQS`) so vLLM can allocate a Mamba state-cache block per sequence:

```bash
cd self-hosted/vllm/scripts
MODEL=Qwen/Qwen3-Coder-Next SERVED_NAME=qwen3-coder-next \
  MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 ./vllm-serve.sh
```

Wrapper with every model-specific parameter spelled out:

```bash
MODEL="Qwen/Qwen3-Coder-Next" \
SERVED_NAME="qwen3-coder-next" \
TP=4 \
PORT=8000 \
MAX_MODEL_LEN=16384 \
MAX_NUM_SEQS=128 \
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

~/vllm-env/bin/vllm serve Qwen/Qwen3-Coder-Next \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen3-coder-next \
  --max-model-len 16384 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  2>&1 | tee logs/vllm-serve.log
```

## The tight-fit warning — read this first

At ~160 GB of BF16 weights on 184 GB of total VRAM, only ~24 GB remains for the KV cache and CUDA graphs **after** vLLM's `--gpu-memory-utilization 0.90` reservation. That has real consequences:

- **Keep `MAX_MODEL_LEN` low** (16384 is a safe starting point). The full 32768 may not leave enough KV cache to boot, or will cap concurrency to a handful of requests.
- **Watch the boot log** for the `Available KV cache memory` / `Maximum concurrency` lines (see [PagedAttention + KV cache](../README.md#pagedattention--kv-cache)). If concurrency reports near 1×, lower `MAX_MODEL_LEN` further.
- **Do not naively raise `GPU_MEM_UTIL`** past `0.90` here — the weights already dominate, and the CUDA-graph capture (~1 GiB) plus activation memory need the remaining headroom. Pushing it risks an OOM at capture time.
- **Cap `MAX_NUM_SEQS` — this model is a hybrid Mamba architecture.** Qwen3-Coder-Next has Mamba / linear-attention (GDN) layers, and vLLM allocates **one Mamba state-cache block per in-flight sequence**. On this VRAM-tight node there are only ~134 such blocks, but vLLM's default `max_num_seqs` is 256, so boot aborts at CUDA-graph capture with:
  > `ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (134). Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed.`

  Set `MAX_NUM_SEQS` at or below the block count the error prints (the serve commands above use `128`). This is a *different* limit from `MAX_MODEL_LEN` — the KV cache and the Mamba cache are separate pools, and on this model both bind. The block count scales with `GPU_MEM_UTIL` and `MAX_MODEL_LEN`, so if you change either, re-read the number in the error and adjust `MAX_NUM_SEQS` to match.

Despite only 3B active parameters per token (fast, MoE economics), this model's value is **quality at the top end**, not concurrency. Use the 30B coder default when you need throughput.

### Disk: ~160 GB of weights won't fit on the root disk

Separate from VRAM: the **download** is ~160 GB, and the DLAMI root disk is only ~193 GB — nowhere near enough once anything else (e.g. the 30B, at ~57 GB) is also cached. A naive run fills the root disk and the download stalls with `Not enough free disk space`. `vllm-serve.sh` handles this by defaulting `HF_HOME` to the DLAMI's large local-NVMe scratch (`/opt/dlami/nvme/hf-cache`, 3.5 TB here) when it exists — the boot log prints `Weights cache: <dir> (<free>)` so you can confirm before the download starts. **Caveat:** that NVMe scratch is **ephemeral** — wiped on instance stop/terminate — so the 160 GB re-downloads after any stop. If you need the cache to survive a stop, set `HF_HOME` to a path on a persistent EBS volume with ≥200 GB free instead.

## Benchmarking

**The SWE benchmark wants a >=200K context window** (agentic coding tasks routinely need 100K-250K input tokens per request), so this model is best benchmarked on a **g6e.48xlarge (8xL40S) or larger** where a 200K-256K window fits. The `/benchmark` skill reads the served `max_model_len` after boot and gates on it: >=200K proceeds, a window somewhat below 200K warns and asks the user to confirm, and a tiny window like this model's ~16K on a 4xL40S is not benchmarkable (every task overflows on turn 1). On a large-VRAM node, serve at a wide window (e.g. `MAX_MODEL_LEN=200000`) and run:

```bash
cd benchmarks
./scripts/run-e2e-benchmark.sh --provider vllm --model qwen3-coder-next \
  --dataset dataset/mcp-gateway-registry.yaml --yes
```

**Do not attempt the benchmark on a g6e.12xlarge (4xL40S).** There the model is VRAM-bound to a ~16384 window; the `/swe` prompt alone (~12K tokens) plus the output reserve overflows it, so every task fails on turn 1. This was confirmed empirically: on a 4xL40S at 16384 with `--max-output-tokens 4096`, task 1 failed 0/4 with `maximum context length is 16384 tokens ... prompt contains at least 12289 input tokens`. Auto-compaction cannot rescue it because the first message already does not fit. For agentic coding on a 4xL40S node, benchmark the 256K-native [Qwen3.6-35B-A3B](qwen3.6-35b-a3b.md) or [Qwen3-Coder-30B](qwen3-coder-30b.md) instead.

The `--max-output-tokens` flag (added to both the harness and the orchestrator) lets you lower the per-response output cap on the CLI without editing `runner.yaml` -- useful for short-window smoke tests, but it does not make this model benchmarkable on a small-VRAM node.

## Tuning notes

- **Tool calling:** `qwen3_coder` parser (same family as the 30B coder). Required for agentic clients.
- **Context window — native 256K, but VRAM is the binding constraint, not the model.** Per the [HF model card](https://huggingface.co/Qwen/Qwen3-Coder-Next) this model is **262144 (256K) native**, extensible to ~1M with YaRN — but that is irrelevant here: with only ~24 GB of VRAM left after the 160 GB of weights, the KV cache caps you *far* below native. That is why the serve command pins `MAX_MODEL_LEN=16384`, the opposite of the other MoE models where the window is a throughput choice. **You will almost never raise it** — 16384 is already near the ceiling this node's KV cache allows for this model, and `ROPE_SCALING` is pointless (you're nowhere near the 256K native window, so there is nothing to extend). **Tradeoff:** if you need a genuinely long window, use [Qwen3.6-35B-A3B](qwen3.6-35b-a3b.md) or [Qwen3-Coder-30B](qwen3-coder-30b.md) instead — both are also 256K-native but leave far more KV-cache room. See [Long context past 32K](../README.md#long-context-and-rope_scaling-yarn).
- **Concurrency:** the lowest of any model in this folder, by design — the weights leave little KV cache.
