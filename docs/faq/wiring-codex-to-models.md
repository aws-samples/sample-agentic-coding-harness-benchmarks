# How do I wire codex to a model?

Three ways to point `codex` at a model: an open-weight model on Amazon Bedrock, an open-weight model you serve yourself with vLLM, or an OpenAI model on Bedrock. Each ends with a `codex exec` call you can run by hand.

For installing codex and wiring it to Bedrock, see [codex-setup.md](../codex-setup.md). This page assumes `codex --version` already answers.

> [!NOTE]
> **Every command below was run before this page shipped**, on a `g6e.4xlarge` (1x L40S) with `codex-cli 0.153.4`. Each `codex exec` example returned `OK`: the Bedrock one against `openai.gpt-5.6-luna`, the vLLM one against a live `qwen3.6-35b-fp8` server at a 262,144-token window, and the bridge one against `qwen.qwen3-coder-30b-a3b-v1:0` through LiteLLM on port 4002. Where a command emits a warning anyway, this page says so.

## The one fact everything follows from

**codex speaks the OpenAI Responses API and nothing else.** codex 0.153.4 removed the chat-completions wire and rejects the fallback:

```
Error loading config.toml: `wire_api = "chat"` is no longer supported.
How to fix: set `wire_api = "responses"` in your provider config.
```

So whatever serves the model must answer `POST /v1/responses`. Bedrock does that for the `openai.*` family only, which is why an open-weight Bedrock model needs a bridge and your own vLLM server does not.

**Close stdin on every call.** `codex exec` reads stdin even when the prompt arrives as an argument, and blocks forever on an open, empty one. Every example below ends with `< /dev/null`.

## Open-weight model on Bedrock (Qwen, Kimi, DeepSeek)

Bedrock will not answer Responses for these:

```
The model 'qwen.qwen3-coder-30b-a3b-v1:0' does not support the '/openai/v1/responses' API
```

The model works there; it just does not answer that API. Put a LiteLLM proxy in front that speaks Responses to codex and Converse to Bedrock, using LiteLLM's native `bedrock/` provider, which authenticates from ambient AWS credentials with no bearer token.

**1. Write the proxy config.**

```yaml
# litellm-responses-bedrock.yaml
model_list:
  - model_name: qwen.qwen3-coder-30b-a3b-instruct
    litellm_params:
      model: bedrock/qwen.qwen3-coder-30b-a3b-v1:0
      aws_region_name: us-east-2

litellm_settings:
  drop_params: true
```

**2. Start it.**

```bash
uv run --with 'litellm[proxy]' litellm --config litellm-responses-bedrock.yaml \
    --host 127.0.0.1 --port 4002
```

**3. Point codex at it.**

```bash
export OPENAI_API_KEY=local     # the proxy ignores the value; codex requires the variable

codex exec --json --skip-git-repo-check \
  --model qwen.qwen3-coder-30b-a3b-instruct \
  -c model_provider=litellm \
  -c model_providers.litellm.name=litellm \
  -c model_providers.litellm.base_url=http://127.0.0.1:4002/v1 \
  -c model_providers.litellm.wire_api=responses \
  -c model_providers.litellm.env_key=OPENAI_API_KEY \
  -- "Reply with exactly: OK" < /dev/null
```

**Two traps.**

The committed [litellm-mantle.yaml](../../benchmarks/config/litellm-mantle.yaml) **cannot** serve this path. It registers each model as `openai/<id>` against the `bedrock-mantle` endpoint, so LiteLLM forwards `/v1/responses` untouched and mantle rejects it for a non-`openai.*` model. That config exists for Claude Code, which speaks the Anthropic Messages API. Write a separate config with the `bedrock/` provider.

Pin nothing older than current LiteLLM. The range the mantle script uses, `litellm[proxy]>=1.72,<1.84`, fails the bridge with `500 'NoneType' object has no attribute 'encode'`.

### Prompt caching does not happen on this route

Bedrock Converse caching is opt-in per request: the caller must place `cachePoint` blocks in the message content. codex has no cache-control concept and LiteLLM's bridge adds none, so `cache_read` is 0 and every token is billed fresh.

Two separate limits hide behind that zero:

| Model | `cachePoint` on Converse |
|---|---|
| `qwen.qwen3-coder-30b-a3b-v1:0` | `AccessDeniedException: You invoked an unsupported model or your request did not allow prompt caching` |
| `us.anthropic.claude-haiku-4-5` | cold call writes `cacheWriteInputTokens: 5204`; warm call reads `cacheReadInputTokens: 5204` |

Qwen cannot cache on Bedrock at all, and a model that can still will not through this bridge. A self-hosted vLLM server is the opposite: its prefix caching is automatic and server-side, needing nothing from the client.

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

**2. Point codex at it.**

```bash
export OPENAI_API_KEY=local     # vLLM ignores the value; codex requires the variable

codex exec --json --skip-git-repo-check \
  --model qwen3.6-35b-fp8 \
  -c model_provider=vllm \
  -c model_providers.vllm.name=vllm \
  -c model_providers.vllm.base_url=http://127.0.0.1:8000/v1 \
  -c model_providers.vllm.wire_api=responses \
  -c model_providers.vllm.env_key=OPENAI_API_KEY \
  -c model_context_window=262144 \
  -- "Reply with exactly: OK" < /dev/null
```

**The base URL must travel in the provider block.** codex 0.153.4 ignores `OPENAI_BASE_URL` and takes its base URL from whichever provider its config selects. Exporting that variable alone sends the call wherever `~/.codex/config.toml` points, which on a judge-configured machine is Bedrock, producing the misleading `404 The model 'minicpm5-2b' does not exist` (issue #183).

**The tool-call parser must accept Responses-shaped tools.** The Responses API sends a flat tool definition, `{"type": "function", "name": ...}`; chat completions nests it under `"function"`. Seven of vLLM 0.29.0's parsers read only the nested shape, abort tool extraction on the flat one, end the stream with no `response.completed`, and make codex retry every request five times before failing the turn:

| Works with codex | Breaks with codex |
|---|---|
| `qwen3_coder`, `hermes` | `minicpm5xml`, `dots`, `hy_v3`, `hy_v4`, `rust`, `step3`, `step3p5` |

**Pass `model_context_window`.** A self-hosted model is unknown to codex, so it sizes the conversation from fallback metadata unless told the real window. Passing the flag does **not** silence the warning -- the run above still logged `Model metadata for 'qwen3.6-35b-fp8' not found. Defaulting to fallback metadata` with `model_context_window=262144` set. Treat that line as noise; what matters is that the window codex plans against matches the one vLLM booted.

## OpenAI model on Bedrock

Nothing to bridge. Bedrock answers Responses natively for the `openai.*` family, so codex needs only its own provider and a region:

```bash
codex exec --json --skip-git-repo-check \
  --model openai.gpt-5.6-luna \
  -c model_provider=amazon-bedrock \
  -c model_providers.amazon-bedrock.aws.region=us-east-2 \
  -- "Reply with exactly: OK" < /dev/null
```

Authentication comes from the ambient AWS credential chain: no proxy, no bearer token. Set the same two values in `~/.codex/config.toml` to make them the default for every call, which is how the judge is configured ([codex-setup.md](../codex-setup.md)).

**This is the one route where codex gets prompt caching.** The call above reported `cache_write_input_tokens: 8761` on a cold run, so a repeated prefix is read back cheaply on the next call. Bedrock's Responses implementation handles the cache itself, with nothing for the caller to place -- unlike Converse behind the LiteLLM bridge, where caching needs `cachePoint` blocks that nothing in the chain inserts.

## Which route serves which model

| Model | Route | Bridge needed |
|---|---|---|
| `openai.*` on Bedrock | native `amazon-bedrock` provider | no |
| Any model on your own vLLM server | provider block at `:8000` | no, but the tool parser must be Responses-safe |
| Open-weight on Bedrock (Qwen, Kimi, DeepSeek) | LiteLLM with the native `bedrock/` provider | yes |
| Anthropic models on Bedrock | native `amazon-bedrock` provider | no |

## Driving a benchmark with this wiring

The harness builds these provider blocks itself, so a benchmark run needs only `--agent codex` and a provider. See [harness-reference.md](../../benchmarks/docs/harness-reference.md#choosing-the-agent) for the agent's flags and cost accounting, and the [benchmark skill](../../.claude/skills/benchmark/SKILL.md) for the end-to-end flow.

## Related

- [codex-setup.md](../codex-setup.md) -- installing codex, wiring it to Bedrock, and its failure-mode table.
- [agent-cli-bedrock-setup.md](../../benchmarks/docs/agent-cli-bedrock-setup.md) -- wiring both CLIs to Bedrock and proving it with a live call.
- [self-hosted/vllm/README.md](../../self-hosted/vllm/README.md) -- installing vLLM and the serving-config reference.
