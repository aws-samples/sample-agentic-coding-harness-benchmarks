# MiniCPM5-2B — serving guidelines

> Per-model serving notes for the vLLM path. See the [directory README](../README.md) for the full install and configuration reference; this file only covers what is specific to **this** model.

| | |
|---|---|
| **HF repo** | `openbmb/MiniCPM5-2B` |
| **Model card** | [huggingface.co/openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) · [OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM) |
| **Served as** | **`minicpm5-2b`** |
| **Type** | Dense `LlamaForCausalLM` — 2.52B total, 1.98B non-embedding; 42 layers, 16 query heads / **2 KV heads** (GQA 8:1), head_dim 128 |
| **Weights** | **5.03 GB** BF16, one safetensors shard |
| **Native context** | **131072 (128K)**, `rope_theta` 5e6, `rope_scaling: null` (no YaRN needed) |
| **KV cache** | **42 KiB/token** at bf16 (21 KiB at `--kv-cache-dtype fp8`) |
| **Tool-call parser** | **`minicpm5`** (vLLM 0.29+) |
| **Reasoning parser** | **`qwen3`** — see [Thinking](#thinking-is-on-and-it-is-verbose) |
| **Fits 1x L40S (46 GB, g6e.4xlarge / g6e.xlarge)?** | **Yes, with the whole 128K window and room for 6 concurrent full-length sequences** |
| **Fits 1x L4 (24 GB, g6.xlarge)?** | Yes — see [Smaller boxes](#smaller-boxes) |
| **License** | Apache-2.0 |
| **Role** | The cheapest thing that can hold a tool-calling loop: a scout and utility model, not a patch author |

## Serve it

```bash
cd self-hosted/vllm/scripts

MODEL="openbmb/MiniCPM5-2B" \
SERVED_NAME="minicpm5-2b" \
TP=1 \
PORT=8000 \
MAX_MODEL_LEN=131072 \
GPU_MEM_UTIL=0.90 \
TOOL_PARSER="minicpm5" \
REASONING_PARSER="qwen3" \
  ./vllm-serve.sh
```

No `--trust-remote-code`, no chat-template override, no rope scaling, no `MAX_NUM_SEQS` cap. The architecture is stock Llama, so vLLM needs nothing custom to load it. The weights download once at 5.03 GB, which is small enough that `HF_HOME` placement does not matter the way it does for the 37-72 GB models on this node.

## The tool parser both upstream pages get wrong

The HuggingFace card and the GitHub README steer you to SGLang for tool calling, naming `--tool-call-parser minicpm5` as an SGLang flag and describing vLLM as merely working. That is out of date. **vLLM 0.29 ships `vllm/tool_parsers/minicpm5xml_tool_parser.py` and registers it under the name `minicpm5`**, and it is the parser this model needs, because the model does not emit JSON tool calls. Its chat template emits XML:

```
<function name="read_file"><param name="path">src/auth/session.py</param></function>
```

with a CDATA wrapper for multi-line or `<`/`&`-bearing values. `MiniCPM5XMLToolParser` converts that to OpenAI-format `tool_calls`. Serve without `TOOL_PARSER=minicpm5` and the XML lands in the response text as prose, so every agent harness sees a model that describes tool calls instead of making them.

Verified on this node:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "model":"minicpm5-2b","messages":[{"role":"user","content":"Read src/auth/session.py. Use the tools."}],
 "tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object",
   "properties":{"path":{"type":"string"}},"required":["path"]}}}],"tool_choice":"auto"}'
```

returns `finish_reason: "tool_calls"` with `{"name":"read_file","arguments":"{\"path\": \"src/auth/session.py\"}"}`.

## Thinking is on, and it is verbose

The model thinks by default and there is no `minicpm5` reasoning parser in vLLM. It does not need one: `<think>` and `</think>` are real special tokens in this tokenizer (ids 8 and 9), the same tags `BaseThinkingReasoningParser` handles, so **`REASONING_PARSER=qwen3` accepts this model unchanged** and routes thinking into its own field. Leave it unset and the think block leaks into the response text.

Two things to know about the field and the volume:

- **vLLM 0.29 names the field `reasoning`, not `reasoning_content`.** Reading the wrong key returns an empty string and looks exactly like a parser that is not working.
- **A 2.5B model thinks at a length that belongs to a much larger one.** On a two-step arithmetic word problem it spent 899 reasoning tokens and hit `finish_reason: "length"` at a 900-token cap without ever reaching an answer. The harness default of `max_output_tokens: 16000` ([runner.yaml](../../../benchmarks/config/runner.yaml)) leaves ample room, but any caller that caps output in the hundreds will get empty `content` back and read it as a broken model.

## Measured on 1x L40S (g6e.4xlarge)

Serving at `MAX_MODEL_LEN=131072`, `GPU_MEM_UTIL=0.90`, TP=1:

| | |
|---|---|
| Weights resident | **4.75 GiB** |
| GPU KV cache | **856,480 tokens** |
| Max concurrency at the full 131,072-token window | **6.53x** |
| Total VRAM in use | 41.7 GB of 46.1 GB |
| Single-request generation | ~116 tokens/sec (smoke test, batch of one) |

The 856K-token cache is the number that makes this model interesting for fan-out work: six agents can each hold a full 128K context at once on one GPU, or several hundred short triage sessions can share it. That single-request 116 tokens/sec is not a throughput figure — use the [`/throughput`](../../../.claude/skills/throughput/SKILL.md) sweep for a number the cost model can consume.

## Smaller boxes

Weights plus activations come to roughly 7 GiB, so the whole 128K window fits well inside a 24 GB card. On a g6.xlarge (1x L4, 24 GB) expect roughly 13 GiB free for KV, which is about 318K tokens at bf16 — still 2.4 full-length sequences. The L4 costs less per hour but has about a third of the L40S's memory bandwidth (300 GB/s vs 864 GB/s) and about a third of its BF16 tensor throughput, and agentic coding is prefill-bound ([cost-per-task methodology](../../../docs/cost-per-task-methodology.md)), so the cheaper box is likely the more expensive one per task. Measure before assuming either way.

## Throughput and cost (1x L40S, g6e.4xlarge)

Seven-level agentic concurrency sweep, 600s per level, priced at the repo's 3-year commitment basis of $1.298/hr:

| Concurrency | Output tok/s | Prefill tok/s | TTFT mean (ms) | TPOT mean (ms) | $/1M blended |
|---|---|---|---|---|---|
| 1 | 28.8 | 4,644 | 7,590 | 13.0 | 0.080 |
| 2 | 79.5 | 3,084 | 3,059 | 19.0 | 0.110 |
| **5** | 76.8 | 8,286 | 4,065 | 47.3 | **0.040** |
| **7** | **82.5** | 6,597 | 6,555 | 68.5 | 0.050 |
| 10 | 65.9 | 7,699 | 9,602 | 107.7 | 0.050 |
| 15 | 40.8 | 4,904 | 32,677 | 253.4 | 0.070 |
| 20 | 37.5 | 4,944 | 72,708 | 280.5 | 0.070 |

**Peak sustained output is 82.5 tokens/sec at concurrency 7**; the cheapest point is concurrency 5 at **$0.04 per 1M blended tokens**, or **$0.0036 per task** at this model's measured 83,753 output tokens per task.

Throughput peaks at 7 and falls away after 10, while TTFT climbs from 6.6s to 32.7s between 7 and 15 and TPOT quadruples. **Concurrency 7 to 10 is the usable ceiling on this card**; past it you pay in latency for throughput you do not get.

Two caveats on this data:

- **The c=1 TTFT of 7.6s is high**, and the sweep's own guidance treats a large uncontended TTFT as a signal to discard the run. Here the cause is prefill volume rather than queueing: agentic prompts run to tens of thousands of tokens and prefill measured 4,644 tokens/sec, which accounts for the latency directly. The c=1 output figure of 28.8 tokens/sec is correspondingly prefill-dominated and understates decode.
- **KV-cache usage, running and waiting request gauges came back empty** for every level, so the saturation story rests on the throughput and latency curves alone. Worth fixing in the collector before quoting a concurrency ceiling with confidence.

## Reproducing it

```bash
# Quality (the server must already be running)
cd benchmarks
./scripts/run-e2e-benchmark.sh --provider vllm --model minicpm5-2b \
    --dataset dataset/mcp-gateway-registry-v2.yaml --agent omp --skill swe3 --yes

# Throughput
cd self-hosted/vllm
./scripts/run-throughput-sweep.sh --model minicpm5-2b --context-window 131072
```

> [!WARNING]
> `run-e2e-benchmark.sh` has **no stray-repo-root-write sweep**. That protection lives in `run-multi-model-benchmark.sh` (quality) and `run-throughput-harness.py` (throughput), so the single-model path the `/benchmark` skill drives is unprotected. This run leaked `fix_config.py`, `lld.md`, `patch.diff` and a `registry/` tree into the repository root, none of which the `.gitignore` name backstops cover. Check `git status` before staging, and never `git add -A` after a single-model run.

## Naming note

`SERVED_NAME` must stay `minicpm5-2b`. The harness writes artifacts to `swe-benchmark-data/<served name>/<harness>/<skill>/<scope>/`, so the served name is what separates this model's results from every other model's.
