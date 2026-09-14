# Qwen3.6-35B-A3B (FP8, single L40S) — serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model.

> [!IMPORTANT]
> **This is a different artifact from [qwen3.6-35b-a3b.md](qwen3.6-35b-a3b.md), and its results must not be merged with that model's.** That file serves the **BF16** weights across **4x L40S** (`g6e.12xlarge`, TP=4); this one serves the **FP8** weights on a **single L40S** (`g6e.4xlarge`, TP=1). Different precision, different hardware, different hourly cost. Serve this one as `qwen3.6-35b-fp8` so its artifacts land in their own `swe-benchmark-data/qwen3.6-35b-fp8/` folder and never overwrite the BF16 runs.

| | |
|---|---|
| **HF repo** | `Qwen/Qwen3.6-35B-A3B-FP8` (BF16 original: `Qwen/Qwen3.6-35B-A3B`) |
| **Model card** | [huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) |
| **Served as** | **`qwen3.6-35b-fp8`** (the BF16/4-GPU guide uses `qwen3.6-35b`) |
| **Type** | MoE — 35.9B total, **3B active per token** (256 experts, 8 per token); hybrid attention (**30 linear + 10 full-attention** layers of 40, `full_attention_interval: 4`); **multimodal** (`Qwen3_5MoeForConditionalGeneration`, vision tower included) |
| **BF16 weights** | ~72 GB |
| **FP8 weights** | **37.5 GB (34.9 GiB)** — e4m3; vision blocks left unquantized |
| **Fits 1x L40S (45 GiB, g6e.4xlarge)?** | **FP8 yes.** BF16 no — see below |
| **Fits 4x L40S (184 GB)?** | Yes, either precision — use [qwen3.6-35b-a3b.md](qwen3.6-35b-a3b.md) |
| **Tool-call parser** | `qwen3_coder` |
| **Native context** | **262144 (256K)** |
| **KV cache** | **10 KiB/token at `--kv-cache-dtype fp8`** (20 KiB at bf16) |
| **Role** | 3.6-generation MoE at single-GPU economics — the BF16 guide's model on one twelfth of the GPUs |

## Why FP8, and why not FP4

BF16 does not fit a single L40S, and it is not close:

| | |
|---|---|
| L40S usable VRAM | 45.0 GiB |
| BF16 weights | ~67 GiB |
| Shortfall | **~22 GiB over the whole card**, before KV cache, activations or CUDA context |

FP8 at 34.9 GiB does fit, with room left for the cache. vLLM selects `MarlinFP8ScaledMMLinearKernel` on this GPU: L40S is Ada (**compute capability 8.9**) and has native FP8, so this is a supported path, not an emulation fallback.

**FP4 is not an option on this GPU, even though a checkpoint exists.** [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) is 23.4 GB and would fit with room to spare, but NVFP4 needs **Blackwell** (sm_100+) and this card is Ada. vLLM 0.29 does register `modelopt_fp4` and `nvfp4_per_token`, so the software is present and the hardware is not — the same wall [qwen3.8-27b.md](qwen3.8-27b.md) documents. There is no INT4 AWQ/GPTQ build of this model either, which *would* have run on Ada; the community 4-bit builds are GGUF (llama.cpp) and MLX (Apple Silicon), neither of which vLLM serves.

## Serve it

```bash
cd self-hosted/vllm/scripts
export PATH="$HOME/vllm-env/bin:$PATH"                              # ninja, see gotchas
export LIBRARY_PATH="$HOME/vllm-env/lib/python3.12/site-packages/nvidia/cu13/lib:$LIBRARY_PATH"

MODEL="Qwen/Qwen3.6-35B-A3B-FP8" \
SERVED_NAME="qwen3.6-35b-fp8" \
TP=1 \
PORT=8000 \
MAX_MODEL_LEN=262144 \
GPU_MEM_UTIL=0.92 \
TOOL_PARSER="qwen3_coder" \
EXTRA_ARGS="--max-num-seqs 32 --kv-cache-dtype fp8" \
  ./vllm-serve.sh
```

The weights download once (37.5 GB); point `HF_HOME` at a volume with room for them.

**`SERVED_NAME` must stay `qwen3.6-35b-fp8`.** The harness writes artifacts to `swe-benchmark-data/<served name>/<harness>/<skill>/...`, so reusing `qwen3.6-35b` would silently mix these single-GPU FP8 runs into the published 4-GPU BF16 results.

## Why the KV cache is not the constraint here

Unusually for a 35B model on one 45 GiB card, **the weights are the ceiling and the cache is cheap**. KV is paid only on full-attention layers, and this model has few of them with narrow heads:

```
10 full-attention layers x 2 (K and V) x 2 KV heads x 256 head_dim x 1 byte (fp8)
  = 10,240 bytes = 10 KiB per token
```

That is roughly a **quarter** of the sibling [Qwen3.8-27B](qwen3.8-27b.md)'s ~37 KiB/token, which pays 16 full-attention layers across 4 KV heads. The practical consequence:

Measured cost is ~11 KiB/token once block allocation is counted, so a full-length request needs:

| Window | KV at fp8 (measured rate) |
|---|---|
| 65,536 | 0.7 GiB |
| 131,072 | 1.4 GiB |
| 262,144 (native) | 2.8 GiB |

So the full 256K native window costs under 3 GiB of cache, and the card has 4.16 GiB free after weights. Pick the window from the VRAM left after the 34.9 GiB of weights, not from a fear of the cache.

## Measured on 1x L40S (g6e.4xlarge)

Both windows booted on this node at `GPU_MEM_UTIL=0.92`, TP=1, `--kv-cache-dtype fp8`, vLLM 0.29.0, driver 595.91.07:

| `MAX_MODEL_LEN` | Available KV | GPU KV cache size | Max concurrency |
|---|---|---|---|
| 131,072 | 4.16 GiB | 395,115 tokens | **3.01x** |
| **262,144 (native)** | 4.16 GiB | 413,075 tokens | **1.58x** |

Measured KV cost is **~11 KiB/token**, against the ~10 KiB the architecture predicts; the difference is block allocation. Weights occupy ~35.9 GB of the card, leaving the 4.16 GiB the table reports.

**The full 256K native window fits, so use it for benchmarking.** The harness runs at `concurrency: 1`, so the 1.58x figure is ample, and Claude Code reserves `max_output_tokens` on top of the prompt — a 246,145-token prompt plus a 16,000-token reserve would overflow a 256K ceiling, which is the failure mode the [BF16 guide](qwen3.6-35b-a3b.md) documents for its 200K default. Drop to 131,072 only when serving several developers at once.

Serving the window is half the job: the harness must also tell Claude Code the window so it auto-compacts before overflowing. The orchestrator does this on the vllm path (it reads the live `max_model_len` and passes `--context-window`); see [Context window and auto-compaction](../../../benchmarks/docs/harness-reference.md#context-window-and-auto-compaction).

Verified working at 262,144: plain completions, and tool calls parsed by `qwen3_coder` (`finish_reason: tool_calls`, arguments `{"city": "Tokyo"}`).

**It also drives `codex` over vLLM's Responses API, which most parsers do not.** codex 0.153.4 speaks only the Responses API and sends flat tools (`{"type": "function", "name": "exec_command", ...}`). `qwen3_coder` never inspects those tool objects, so it works: a measured three-tool-call turn produced `response.function_call_arguments.done` plus a terminal `response.completed`, ran three separate shell commands with no retries, and reported input 35,508 / output 180 against vLLM's own 35,508 prompt / 180 generation counters. Seven other parsers -- `minicpm5xml`, `dots`, `hy_v3`, `hy_v4`, `rust`, `step3`, `step3p5` -- read the nested chat-completions shape (`tool.function.name`), abort tool extraction, and truncate the stream, which makes codex retry every request five times and fail the turn (issue #183).

## Environment failures before you reach the model

Two node-level failures hit this box first, and both are documented in full under [qwen3.8-27b.md](qwen3.8-27b.md): the **`ninja` PATH** failure (`vllm-install.sh` puts ninja in the venv, `vllm-serve.sh` does not add it to `PATH`) and the **`libcudart` symlink** FlashInfer's JIT needs for the FP8-KV kernel. The serve command above exports both. Neither is specific to this model, but the ninja one killed its first boot after loading all 35.9 GB of weights, so budget for it on a fresh box.

## The one gotcha that is this model's own

**It reasons inline, which costs output tokens and can truncate a tool call.** The chain of thought arrives as ordinary text in `content`: no `<think>` delimiter, no `reasoning_content` field.

**Tool calls parse correctly, and `qwen3_coder` is the right parser.** Measured on this node with the same weather-tool request:

| Request | Result | Completion tokens |
|---|---|---|
| `max_tokens: 800` | parsed, `finish_reason: tool_calls` | 175 |
| `max_tokens: 800`, `enable_thinking: false` | parsed, `tool_calls` | **26** |
| `max_tokens: 150`, five cities | **4 of 5 parsed**; one returned `finish_reason: length` and no tool call | 124-150 |

The failure is truncation, not parsing. Reasoning length varies run to run, so a tight cap loses the tool call intermittently rather than every time -- the worst kind of flake to debug from a benchmark log. Its signature is `finish_reason: length` with no `tool_calls`. **No `--reasoning-parser` helps:** with no delimiter in the output there is nothing for one to split on.

Two consequences:

- **Give this model a generous `max_output_tokens`.** Expect output-token counts, and so cost per task, to run higher than a non-reasoning model of the same size.
- **Thinking stays on. That is the default for this repo, and the serve command above sets nothing to change it.** Reasoning is how the model does the work, and the models it is measured against on the frontier reason too, so turning it off here would compare a different thing.

> [!NOTE]
> **`enable_thinking: false` is a cost lever, not a setting to leave on.** Passed through `chat_template_kwargs`, it returned the same tool call in **26 completion tokens against 175** -- about 7x less output on that request. It is worth measuring on a real dataset, because output tokens drive cost per task. But it changes how the model works a problem, so any run with thinking off is a **separate result**: record it as its own configuration and never merge its scores into a table built with thinking on.

`--max-num-seqs 32` is set in the command above as a precaution: the 30 linear-attention layers each need a state block per decode sequence from a pool sized independently of the KV cache, and vLLM's default of 256 exceeds that pool on the sibling 27B. It did not fail here, but 32 is far above any concurrency this repo sweeps and costs nothing.

## Benchmarking it

```bash
# Quality (the server must already be running)
cd benchmarks
uv run scripts/run-swe-headless.py --config config/runner.yaml \
    --provider vllm --model qwen3.6-35b-fp8 \
    --dataset dataset/mcp-gateway-registry-v2.yaml
```

Instance pricing for `g6e.4xlarge` is in [pricing.json](../pricing.json) at the 3-year commitment rate ($1.298/hr), against $4.533/hr for the `g6e.12xlarge` the BF16 guide uses. That ratio, not the quality delta alone, is the reason this variant is worth measuring separately.

## Naming note

Three neighbouring files are easy to confuse:

- **This file** — Qwen3.6-35B-A3B **FP8 on 1x L40S**, served as `qwen3.6-35b-fp8`.
- [qwen3.6-35b-a3b.md](qwen3.6-35b-a3b.md) — the same model in **BF16 on 4x L40S**, served as `qwen3.6-35b`. The published results are from that one.
- [qwen3-32b.md](qwen3-32b.md) — a **dense** 32.8B model from the earlier generation, unrelated architecture.
