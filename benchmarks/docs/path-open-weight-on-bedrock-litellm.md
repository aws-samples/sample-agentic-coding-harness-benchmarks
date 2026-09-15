# Path 2 - open-weight models on Amazon Bedrock via a LiteLLM proxy

Use this path to benchmark **non-Anthropic (open-weight) models hosted on Amazon Bedrock** -- Moonshot AI's Kimi, Qwen, DeepSeek, Mistral, MiniMax, GLM, GPT-OSS, and so on. `claude -p` only speaks the Anthropic Messages API, and Bedrock's native Anthropic route rejects these models (see [Path 1](path-anthropic-on-bedrock.md#anthropic-only)). The fix is a [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) that we run in front of Bedrock: it translates between the Anthropic Messages format Claude Code speaks and the OpenAI Chat Completions format the open-weight Bedrock models speak, so **any open-weight model on Bedrock can be wired into Claude Code** and driven through the harness with `provider: endpoint`.

The [scripts/bedrock-mantle-proxy.sh](../scripts/bedrock-mantle-proxy.sh) helper starts the proxy for you. For everything common to all paths -- dataset format, runner config, metrics file, and the judge -- see the [harness reference](harness-reference.md).

## Use the `bedrock-mantle` endpoint, not the Converse path

There are two ways a proxy can reach a non-Anthropic Bedrock model, and only one preserves tool calls -- which the agentic `/swe` run depends on:

- **Converse (`litellm bedrock/<model>`) -- broken for agentic runs.** This path returns the model's *native* tool-call tokens (e.g. Kimi's `<|tool_calls_section|>...`) as plain **text** with `stop_reason: end_turn`. Claude Code only acts on structured `tool_use` blocks, so it never calls a tool and the `/swe` run stalls at one turn with 0 artifacts.
- **`bedrock-mantle` (`litellm openai/<model>`) -- works.** [`bedrock-mantle`](https://docs.aws.amazon.com/bedrock/latest/userguide/inference.html) is Bedrock's OpenAI-compatible Chat Completions endpoint (`bedrock-mantle.us-east-1.api.aws/v1`). Third-party models on it support tool calling natively, so LiteLLM gets **structured** tool calls back and translates them into Anthropic `tool_use` blocks the agent can act on (`stop_reason: tool_use`).

The proxy config [config/litellm-mantle.yaml](../config/litellm-mantle.yaml) maps every mantle model to its `openai/<model-id>` on that endpoint.

## Prerequisites

- AWS credentials configured for `us-east-1` (the only region where `bedrock-mantle` is available today; the proxy uses the same ambient `boto3`/AWS CLI chain).
- The target model enabled in the Bedrock console (Model access).
- `uv` available (the script installs `litellm[proxy]` and `aws-bedrock-token-generator` on demand).

## Run it

Run these in order.

### 1. Start the proxy

It mints a 12h Bedrock bearer token from your AWS credentials, injects it as `MANTLE_API_KEY`, and binds `127.0.0.1:4000`. Leave it running:

```bash
cd benchmarks
./scripts/bedrock-mantle-proxy.sh            # start on :4000
./scripts/bedrock-mantle-proxy.sh --status   # check health + token age
./scripts/bedrock-mantle-proxy.sh --refresh  # remint the token (restart to apply)
./scripts/bedrock-mantle-proxy.sh --stop     # stop it
```

To benchmark a model not already in [config/litellm-mantle.yaml](../config/litellm-mantle.yaml), add a `model_list` entry (copy an existing block, change the id). Discover exact ids with:

```bash
aws bedrock list-foundation-models --region us-east-1 \
    --query "modelSummaries[?contains(providerName,'Moonshot')].[modelId,modelName]" \
    --output table
```

### 2. Smoke-test that tool calls come back structured

Do this before spending a full run. This is the exact shape Claude Code sends -- a `tool_use` content block and `stop_reason: tool_use` in the reply confirm the whole path works:

```bash
curl -s http://127.0.0.1:4000/v1/messages \
    -H "Content-Type: application/json" \
    -H "x-api-key: sk-anything" \
    -H "anthropic-version: 2023-06-01" \
    -d '{"model":"moonshotai.kimi-k2-thinking","max_tokens":256,
         "tools":[{"name":"write_file","description":"Write a file.",
           "input_schema":{"type":"object","properties":{"path":{"type":"string"},
           "content":{"type":"string"}},"required":["path","content"]}}],
         "messages":[{"role":"user","content":"Create hello.txt containing HELLO. Use write_file."}]}'
```

### 3. Run the harness through the `endpoint` provider

Point it at the proxy (not `--provider bedrock`). The `--model` must match a `model_name` in the proxy config; the harness derives the `{model-name}` artifact subfolder from it. Start with one task, then run the full dataset:

```bash
cd benchmarks

# One-task confirmation
uv run scripts/run-swe-headless.py --config config/runner.yaml \
    --provider endpoint --endpoint http://127.0.0.1:4000 \
    --model moonshotai.kimi-k2-thinking \
    --dataset dataset/mcp-gateway-registry.yaml --count 1 --stream

# Full dataset
uv run scripts/run-swe-headless.py --config config/runner.yaml \
    --provider endpoint --endpoint http://127.0.0.1:4000 \
    --model moonshotai.kimi-k2-thinking \
    --dataset dataset/mcp-gateway-registry.yaml
```

## Notes

- **Auth.** The `api_key` comes from the config (`api_key: local` in `runner.example.yaml`); the proxy holds the real Bedrock token, so clients send a throwaway value and the harness turns that field into the `apiKeyHelper` Claude Code needs so it never hits `Not logged in`.
- **Metrics.** Because this is the `endpoint` path, the harness will attempt to scrape Prometheus `/metrics` from the proxy -- LiteLLM does not expose vLLM's metric surface, so the `vllm_prometheus` block is simply empty (`available: false`); the per-run API metrics (tokens, latency, turns) are captured correctly.
- **No prompt caching.** Prompt caching is not available across the translation layer, so multi-turn `/swe` runs re-send the full context each turn and input-token counts climb accordingly.
- **Model-name prefixes differ by transport.** On the mantle endpoint both Kimi models use the `moonshotai.` prefix (e.g. `moonshotai.kimi-k2-thinking`, `moonshotai.kimi-k2.5`). The native Converse foundation-model listing uses a different prefix (`moonshot.`), so use the mantle ids from `config/litellm-mantle.yaml` here.

See the [harness reference](harness-reference.md#common-invocations) for the full set of `--count`, `--tasks`, `--concurrency`, `--stream`, and `--verbose` options, which behave the same on every path.
