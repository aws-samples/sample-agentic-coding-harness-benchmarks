# Installing codex and pointing it at a model

`codex` does two jobs here, and they are configured separately. It is the **judge** that scores every benchmark run, always against Amazon Bedrock, and it is one of five **coding harnesses** (`--agent codex`) that can drive a task on any hosting path. This page installs it, proves it works, and states the one constraint that decides which models it can reach. The per-path wiring recipes live in the FAQ: [How do I wire codex to a model?](faq/wiring-codex-to-models.md)

## Install

```bash
npm install -g @openai/codex
codex --version    # >= 0.144 for the native amazon-bedrock provider
```

The `setup-machine` skill installs it as part of the core stack, so on a box bootstrapped with [setup-machine.sh](../.claude/skills/setup-machine/setup-machine.sh) it is already present. Installing the CLI does **not** wire it to Bedrock.

## Wire it to Bedrock and prove it

codex ships a native `amazon-bedrock` provider that authenticates from the AWS credential chain. No proxy, no bearer token; the LiteLLM route older notes describe is legacy. The config lives at `~/.codex/config.toml`:

```toml
model_provider = "amazon-bedrock"
model_providers.amazon-bedrock.aws.region = "us-east-2"
model = "openai.gpt-5.6-sol"
```

Full walk-through, including the `claude` side: [agent-cli-bedrock-setup.md](../benchmarks/docs/agent-cli-bedrock-setup.md).

**Prove it with a real call before starting a long run.** Working AWS credentials are not sufficient: an unconfigured codex ignores them and 401s against `api.openai.com`, and the failure surfaces only after the harness has finished generating.

```bash
codex exec --skip-git-repo-check "Reply with exactly: JUDGE OK"
```

## The constraint that decides everything else

**codex speaks the OpenAI Responses API and nothing else.** codex 0.153.4 removed the chat-completions wire and rejects the fallback:

```
Error loading config.toml: `wire_api = "chat"` is no longer supported.
How to fix: set `wire_api = "responses"` in your provider config.
```

Amazon Bedrock answers `/v1/responses` for the **`openai.*` family only**. Everything downstream follows:

| Model you want to drive | Provider | Reachable directly? |
|---|---|---|
| `openai.*` on Bedrock | `bedrock` | Yes, this same native provider |
| Any model on your own vLLM server | `endpoint` (`--provider vllm`) | Yes, if its tool parser accepts Responses-shaped tools |
| Open-weight on Bedrock (Qwen, Kimi, DeepSeek) | `endpoint` (`--provider litellm`) | No. Bedrock rejects Responses for these; a LiteLLM bridge is required |
| Anthropic models | `bedrock` | Yes, though Claude Code is the better-measured harness for them |

The commands for each row are in the FAQ: [How do I wire codex to a model?](faq/wiring-codex-to-models.md)

## Known failure modes

Each of these cost real debugging time. The FAQ carries the detail and the fix.

| Symptom | Cause |
|---|---|
| Run hangs before the first request, log stops at `Reading additional input from stdin...` | `codex exec` blocks on an open, empty stdin. Launch every run with `< /dev/null` |
| `404 The model '<name>' does not exist` on an endpoint run | codex 0.153.4 ignores `OPENAI_BASE_URL` and used the config's provider, which on a judge-configured box is Bedrock (issue #183) |
| Every request retried five times, stream ends with no `response.completed` | The vLLM tool parser reads the nested chat-completions tool shape and aborts on the flat Responses shape. Use `qwen3_coder` or `hermes` |
| `Model metadata not found. Defaulting to fallback metadata` | codex does not know a self-hosted model's window. Pass `--context-window` |
| `does not support the '/openai/v1/responses' API` | An open-weight Bedrock model reached without the bridge |
| `total_cost_usd` is null | The model has no row in [bedrock_pricing.py](../benchmarks/scripts/bedrock_pricing.py). codex reports tokens but no billed cost, so an unpriced model records null rather than a misleading zero |

## Related

- [faq/wiring-codex-to-models.md](faq/wiring-codex-to-models.md) -- the per-path recipes, the parser table, and why prompt caching does not happen on the Bedrock bridge.
- [agent-cli-bedrock-setup.md](../benchmarks/docs/agent-cli-bedrock-setup.md) -- wiring both CLIs to Bedrock.
- [harness-reference.md](../benchmarks/docs/harness-reference.md#choosing-the-agent) -- every supported agent, its providers, and its cost basis.
