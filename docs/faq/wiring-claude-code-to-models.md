# How do I wire Claude Code to a model?

Three ways to point `claude` at a model: an open-weight model on Amazon Bedrock, an open-weight model you serve yourself with vLLM, or a model on Bedrock directly. Each ends with a `claude -p` call you can run by hand.

For wiring `claude` and the `codex` judge to Bedrock on a fresh box, see [agent-cli-bedrock-setup.md](../../benchmarks/docs/agent-cli-bedrock-setup.md). This page assumes `claude --version` already answers.

> [!NOTE]
> **Every command below was run before this page shipped**, on a `g6e.4xlarge` (1x L40S) with Claude Code `2.1.266`. Each `claude -p` example returned `OK`: the Bedrock one against `claude-haiku-4-5`, the vLLM one against a live `qwen3.6-35b-fp8` server, and the open-weight-on-Bedrock one against `qwen.qwen3-coder-30b-a3b-instruct` through the LiteLLM mantle proxy.

## The three facts everything follows from

**Claude Code speaks the Anthropic Messages API.** Anything else must answer `POST /v1/messages`. vLLM does, which is why a self-hosted model needs no bridge here; Bedrock's non-Anthropic models do not, which is why they need one.

**`--settings` wins, and you always want to pass it.** A settings object's `env` block takes precedence over process environment variables, including anything in your global `~/.claude/settings.json`. A global file pinning `CLAUDE_CODE_USE_BEDROCK=1` will otherwise redirect a run meant for a local endpoint straight to Bedrock, which rejects the local model id. Passing `--settings` is what reliably wins.

**Claude Code cannot detect a custom model's context window.** Against a custom `ANTHROPIC_BASE_URL` it has no window to compact against, so on a long task the conversation grows until the endpoint rejects it, and Claude Code treats that 500 as transient and retries it forever. Set `CLAUDE_CODE_AUTO_COMPACT_WINDOW` to the served window on every endpoint run.

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

vLLM answers the Anthropic Messages API as well as the OpenAI one, so nothing sits between Claude Code and the server. Confirm it before blaming the agent:

```bash
curl -s http://127.0.0.1:8000/v1/messages \
  -H 'content-type: application/json' -H 'x-api-key: local' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"qwen3.6-35b-fp8","max_tokens":16,"messages":[{"role":"user","content":"Reply with exactly: OK"}]}'
```

**2. Run it.**

```bash
claude -p "Reply with exactly: OK" \
  --model qwen3.6-35b-fp8 \
  --output-format json \
  --settings '{"apiKeyHelper":"echo local","env":{
      "CLAUDE_CODE_USE_BEDROCK":"0",
      "ANTHROPIC_BASE_URL":"http://127.0.0.1:8000",
      "ANTHROPIC_API_KEY":"local",
      "CLAUDE_CODE_AUTO_COMPACT_WINDOW":"262144",
      "DISABLE_NON_ESSENTIAL_MODEL_CALLS":"1"}}' < /dev/null
```

**`apiKeyHelper` is required even though the server ignores the value.** Without a token source Claude Code refuses with "Not logged in". `echo local` satisfies it.

**Do not trust `total_cost_usd` on this route.** The verified run against a local server reported `"total_cost_usd": 0.1138` for a model that costs nothing per token, because Claude Code prices from its own assumptions for an unrecognized model. It also reported `"contextWindow": 200000` while the server was serving 262,144. For a self-hosted model the repo derives cost from GPU-seconds instead, never from this field ([cost-per-task-methodology.md](../cost-per-task-methodology.md)).

## Open-weight model on Bedrock (Qwen, Kimi, DeepSeek)

Bedrock's non-Anthropic models do not answer the Messages API. Worse, sent through Bedrock's Converse path they return their native tool-call tokens as plain text, so Claude Code never sees a structured `tool_use` block and an agentic run stalls at one turn with zero artifacts. The fix is a proxy that translates Anthropic Messages to OpenAI chat completions and on to Bedrock's OpenAI-compatible `bedrock-mantle` endpoint, which parses those tokens into real tool calls. The repo ships it.

**1. Start the proxy.**

```bash
cd benchmarks
./scripts/bedrock-mantle-proxy.sh          # installs deps, mints a 12h token, listens on :4000
./scripts/bedrock-mantle-proxy.sh --status
```

It mints the bearer token from your ambient AWS credentials and holds it server-side; clients send a throwaway key. Models come from [config/litellm-mantle.yaml](../../benchmarks/config/litellm-mantle.yaml), and `curl -s http://127.0.0.1:4000/v1/models` lists them.

**2. Run it.** Same shape as the vLLM route, pointed at the proxy:

```bash
claude -p "Reply with exactly: OK" \
  --model qwen.qwen3-coder-30b-a3b-instruct \
  --output-format json \
  --settings '{"apiKeyHelper":"echo local","env":{
      "CLAUDE_CODE_USE_BEDROCK":"0",
      "ANTHROPIC_BASE_URL":"http://127.0.0.1:4000",
      "ANTHROPIC_API_KEY":"local",
      "CLAUDE_CODE_AUTO_COMPACT_WINDOW":"262144",
      "DISABLE_NON_ESSENTIAL_MODEL_CALLS":"1"}}' < /dev/null
```

**`DISABLE_NON_ESSENTIAL_MODEL_CALLS=1` is doing real work here.** Without it, the same command emitted `[claude-code:unrecognized_model] {"model":"qwen.qwen3-coder-30b-a3b-instruct","query_source":"generate_session_title"}` before the answer: Claude Code makes side calls, such as generating a session title, and an unrecognized model id makes them fail noisily. With the flag set the run returned `"result":"OK","subtype":"success","num_turns":1`.

Setting `ANTHROPIC_API_KEY` also prints a warning that claude.ai connectors are disabled because an auth source takes precedence over your claude.ai login. That is expected on this route and harmless.

## Model on Bedrock directly

Anthropic models need no proxy. Flip Bedrock mode on and name the inference profile:

```bash
claude -p "Reply with exactly: OK" \
  --model us.anthropic.claude-haiku-4-5-20251001-v1:0 \
  --output-format json \
  --settings '{"env":{"CLAUDE_CODE_USE_BEDROCK":"1","AWS_REGION":"us-east-2"}}' < /dev/null
```

No `apiKeyHelper` and no base URL: Bedrock mode authenticates from the ambient AWS credential chain. Pin the region in the settings block, not just the environment, so a global settings file cannot flip routing.

**This route reports real costs and real caching.** The verified run returned `"total_cost_usd": 0.0253` with `"cache_creation_input_tokens": 20064` -- Claude Code inserts its own cache-control markers, so a repeated prefix is written once and read back cheaply on later turns. That is why published Claude Code runs on Bedrock show millions of cache-read tokens against a tiny fresh-input count.

## Which route serves which model

| Model | Route | Extra service |
|---|---|---|
| Anthropic on Bedrock | `CLAUDE_CODE_USE_BEDROCK=1` + inference profile | none |
| Any model on your own vLLM server | `ANTHROPIC_BASE_URL` at `:8000` | none, vLLM answers `/v1/messages` |
| Open-weight on Bedrock (Qwen, Kimi, DeepSeek) | `ANTHROPIC_BASE_URL` at the mantle proxy | `bedrock-mantle-proxy.sh` |

## Driving a benchmark with this wiring

The harness builds the settings object and the environment per run, so a benchmark run needs only `--provider` and a model; `claude` is the default agent. See [harness-reference.md](../../benchmarks/docs/harness-reference.md#choosing-the-agent) for its flags, the permission model, and the auto-compaction detail, and the [benchmark skill](../../.claude/skills/benchmark/SKILL.md) for the end-to-end flow.

## Related

- [agent-cli-bedrock-setup.md](../../benchmarks/docs/agent-cli-bedrock-setup.md) -- wiring both CLIs to Bedrock and proving it with a live call.
- [faq/wiring-codex-to-models.md](wiring-codex-to-models.md) and [faq/wiring-omp-to-models.md](wiring-omp-to-models.md) -- the same three routes for the other two harnesses.
- [cost-per-task-methodology.md](../cost-per-task-methodology.md) -- why `total_cost_usd` is trustworthy on Bedrock and meaningless against a self-hosted endpoint.
