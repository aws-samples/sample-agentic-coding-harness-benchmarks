# How do I wire omp to a model?

Three ways to point `omp` (oh-my-pi) at a model: an open-weight model on Amazon Bedrock, an open-weight model you serve yourself with vLLM, or a model on Bedrock directly. Each ends with an `omp -p` call you can run by hand.

For installing omp and the three ways it differs from `pi`, see [omp-setup.md](../omp-setup.md). This page assumes `omp --version` already answers.

> [!NOTE]
> **Every command below was run before this page shipped**, on a `g6e.4xlarge` (1x L40S) with `omp/18.1.15`. Each `omp -p` example returned `OK`: the vLLM one against a live `qwen3.6-35b-fp8` server at a 262,144-token window, the Bedrock one against `claude-haiku-4-5`, and the open-weight-on-Bedrock one against `qwen.qwen3-coder-30b-a3b-instruct` through the LiteLLM mantle proxy.

## The two facts everything follows from

**omp's config is YAML, and it lives wherever `PI_CODING_AGENT_DIR` points.** omp is a fork of pi and honours that variable, but where pi reads `models.json`, omp reads `models.yml` for providers and `config.yml` for settings. Pointing the variable at a scratch directory keeps an experiment away from your own `~/.omp`, which is what the harness does per run.

**A model is named `provider/model`.** The provider is a key in `models.yml` for a custom endpoint, or the built-in `amazon-bedrock` for Bedrock. So `--model vllm/qwen3.6-35b-fp8` and `--model amazon-bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`.

Two smaller ones that bite:

- **omp hangs on an inherited stdin.** Redirect on every call: the examples below all end with `< /dev/null`.
- **Put `--` before the prompt.** omp parses options until it sees `--`. A prompt that starts with `---`, which any inlined `SKILL.md` does because of its YAML frontmatter, is otherwise read as an unknown flag.

## Open-weight model you serve with vLLM

**1. Serve it.** Take `MODEL`, `SERVED_NAME`, `TP`, `MAX_MODEL_LEN` and `TOOL_PARSER` from the model's guide under [self-hosted/vllm/models/](../../self-hosted/vllm/models/) rather than inventing them. For `qwen3.6-35b-fp8`:

```bash
cd self-hosted/vllm/scripts
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

**2. Declare the server as a provider.** Write `models.yml` into a scratch agent dir:

```yaml
# /tmp/omp-vllm/.omp/models.yml
providers:
  vllm:
    baseUrl: http://127.0.0.1:8000/v1
    api: openai-completions
    apiKey: local
    models:
      - id: qwen3.6-35b-fp8
        name: "vLLM: qwen3.6-35b-fp8"
        contextWindow: 262144
        maxTokens: 16000
        cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}
```

**3. Size compaction to the window.** Without this omp fills the context to within its default reserve, and one capped response overflows the endpoint mid-run. Reserve a full response plus about 8K of headroom -- for a 262,144 window and a 16,000-token cap, that is `262144 - (16000 + 8192) = 237952`:

```yaml
# /tmp/omp-vllm/.omp/config.yml
compaction:
  enabled: true
  thresholdTokens: 237952
```

**4. Run it.**

```bash
PI_CODING_AGENT_DIR=/tmp/omp-vllm/.omp \
  omp -p --mode json --no-session --auto-approve \
  --model vllm/qwen3.6-35b-fp8 \
  -- "Reply with exactly: OK" < /dev/null
```

`--mode json` gives the pi-shaped event stream; the final `message_end` event carries usage. The verified run reported `"provider":"vllm","model":"qwen3.6-35b-fp8"` with 18,108 input and 6 output tokens.

**`--no-session` keeps the run ephemeral** so nothing lands in your session history, and **`--auto-approve`** runs tools without an approval gate, which an unattended run needs.

## Open-weight model on Bedrock (Qwen, Kimi, DeepSeek)

omp speaks the OpenAI chat-completions wire through `api: openai-completions`, so it needs an OpenAI-compatible endpoint in front of Bedrock. The repo ships one: [bedrock-mantle-proxy.sh](../../benchmarks/scripts/bedrock-mantle-proxy.sh) starts a LiteLLM proxy over Bedrock's OpenAI-compatible `bedrock-mantle` endpoint, minting a 12-hour bearer token from your ambient AWS credentials and holding it server-side.

**1. Start the proxy.**

```bash
cd benchmarks
./scripts/bedrock-mantle-proxy.sh          # installs deps, mints the token, listens on :4000
./scripts/bedrock-mantle-proxy.sh --status
```

Models come from [config/litellm-mantle.yaml](../../benchmarks/config/litellm-mantle.yaml); `curl -s http://127.0.0.1:4000/v1/models` lists them.

**2. Declare it as a provider, with real rates.**

```yaml
# /tmp/omp-mantle/.omp/models.yml
providers:
  mantle:
    baseUrl: http://127.0.0.1:4000/v1
    api: openai-completions
    apiKey: local
    models:
      - id: qwen.qwen3-coder-30b-a3b-instruct
        name: "Bedrock: qwen3-coder-30b"
        contextWindow: 262144
        maxTokens: 16000
        cost: {input: 0.1545, output: 0.618, cacheRead: 0, cacheWrite: 0}
```

**3. Run it.**

```bash
PI_CODING_AGENT_DIR=/tmp/omp-mantle/.omp \
  omp -p --mode json --no-session --auto-approve \
  --model mantle/qwen.qwen3-coder-30b-a3b-instruct \
  -- "Reply with exactly: OK" < /dev/null
```

**The `cost` block is not decoration: omp prices the run from it.** The verified run reported `"cost":{"input":0.002824878,"output":0.000001236,...,"total":0.002826114}` for 18,284 input and 2 output tokens, which is exactly those per-1M rates applied. Leave the zeros in for a self-hosted model, where per-token cost is meaningless and the repo derives cost from GPU-seconds instead ([cost-per-task-methodology.md](../cost-per-task-methodology.md)). Put the real Bedrock rates in for a metered model, and omp's own numbers are usable.

The rates above are Qwen3-Coder-30B's Standard tier, the same ones in [bedrock_pricing.py](../../benchmarks/scripts/bedrock_pricing.py).

## Model on Bedrock directly

Bedrock is built in, so no `models.yml` is needed. Name the inference profile after `amazon-bedrock/`:

```bash
omp -p --mode json --no-session --auto-approve \
  --model amazon-bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
  -- "Reply with exactly: OK" < /dev/null
```

Authentication comes from the ambient AWS credential chain; set `AWS_REGION` if your default region does not host the model. Nothing is written to disk and no key is passed on the command line.

## Which route serves which model

| Model | Route | Extra service |
|---|---|---|
| Anthropic (or any native Bedrock model) | `--model amazon-bedrock/<inference-profile>` | none |
| Any model on your own vLLM server | a `models.yml` provider at `:8000` | none |
| Open-weight on Bedrock (Qwen, Kimi, DeepSeek) | a `models.yml` provider at the mantle proxy | `bedrock-mantle-proxy.sh` |

## Driving a benchmark with this wiring

The harness writes `models.yml` and `config.yml` per run into a scratch `PI_CODING_AGENT_DIR`, so a benchmark run needs only `--agent omp` and a provider. See [harness-reference.md](../../benchmarks/docs/harness-reference.md#choosing-the-agent) for the agent's flags and cost accounting, and the [benchmark skill](../../.claude/skills/benchmark/SKILL.md) for the end-to-end flow.

## Related

- [omp-setup.md](../omp-setup.md) -- installing omp and the three ways it differs from pi.
- [faq/wiring-codex-to-models.md](wiring-codex-to-models.md) -- the same three routes for codex, which needs the Responses API and therefore a different bridge.
- [cost-per-task-methodology.md](../cost-per-task-methodology.md) -- why a metered Bedrock bill and a hardware-derived self-hosted figure do not compare as raw dollars.
