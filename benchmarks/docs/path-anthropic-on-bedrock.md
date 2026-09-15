# Path 1 - Anthropic models directly on Amazon Bedrock

Use this path to benchmark the **Anthropic model family** (Claude Opus, Sonnet, Haiku) served on Amazon Bedrock. `claude -p` speaks the Anthropic Messages API natively, and Bedrock has a first-class Anthropic path, so no proxy or extra infrastructure is involved -- the harness points `claude -p` straight at Bedrock.

This is the simplest of the three paths. For everything that is common to all paths -- dataset format, runner config, metrics file, and the judge -- see the [harness reference](harness-reference.md).

## How it works

With `provider: bedrock`, the harness:

- flips `CLAUDE_CODE_USE_BEDROCK=1`,
- sets `AWS_REGION` from `aws_region` (falling back to `AWS_REGION` / `AWS_DEFAULT_REGION` in the environment),
- clears any stray `ANTHROPIC_BASE_URL` so nothing redirects the client off Bedrock, and
- authenticates with your **ambient AWS credentials** (the standard `boto3`/AWS CLI chain: environment variables, `~/.aws/credentials`, an SSO session, or an instance/role profile), so no `api_key` or `apiKeyHelper` is set.

`model` is a Bedrock model id or inference profile, e.g. `us.anthropic.claude-opus-4-8`. The harness strips the vendor/region prefix and any `[...]` suffix to derive the `{model-name}` artifact subfolder, so `us.anthropic.claude-opus-4-8` writes its artifacts under `claude-opus-4-8/`.

The harness still passes `claude --settings` here (that is how it wins over a global `~/.claude/settings.json` -- see [How `--settings` pins routing](harness-reference.md#how---settings-pins-routing)). Concretely, if your global settings file pins `CLAUDE_CODE_USE_BEDROCK=1` and you try to route to a *local* endpoint instead, merely exporting `CLAUDE_CODE_USE_BEDROCK=0` is silently overridden; the `--settings` object is what pins routing deterministically. On this path it pins it *to* Bedrock.

## Metrics on this path

Because Bedrock exposes no Prometheus `/metrics` surface, the `vllm_prometheus` block is **omitted entirely** from `metrics.json` (and the vLLM-only `prefix_cache_hit_rate` drops out of `metrics_that_matter`) rather than written as a permanently-unavailable stub -- so a Bedrock run is limited to what `claude -p` itself reports. The per-run API metrics (tokens, latency, turns, cost) are captured exactly as on the endpoint paths, and against Bedrock the cache-token fields are populated straight from the model API.

## Prerequisites

- AWS credentials configured for the target region (`aws sts get-caller-identity` should succeed).
- The requested Anthropic model id enabled in the Bedrock console (Model access).

## Run it

```bash
cd benchmarks
uv run scripts/run-swe-headless.py --config config/runner.yaml \
    --provider bedrock --aws-region us-east-1 \
    --model us.anthropic.claude-opus-4-8 \
    --dataset dataset/mcp-gateway-registry.yaml
```

Start with the trivial sanity dataset to confirm credentials and model access before a full run:

```bash
cd benchmarks
uv run scripts/run-swe-headless.py --config config/runner.yaml \
    --provider bedrock --aws-region us-east-1 \
    --model us.anthropic.claude-opus-4-8 \
    --dataset dataset/hello-world.yaml --stream
```

See the [harness reference](harness-reference.md#common-invocations) for the full set of `--count`, `--tasks`, `--concurrency`, `--stream`, and `--verbose` options, which behave the same on every path.

## Anthropic-only

`provider: bedrock` works **only** for `us.anthropic.claude-*` models. `claude -p` always speaks the Anthropic Messages API, and this path sends that straight to Bedrock's Anthropic route -- so pointing `--provider bedrock` at a non-Anthropic Bedrock model (Moonshot/Kimi, Meta Llama, Mistral, etc.) fails fast, e.g. `400 Request metadata contains a value that violates the regular expression`. To benchmark those models, front Bedrock with a LiteLLM proxy: see [Path 2 - open-weight models on Amazon Bedrock via a LiteLLM proxy](path-open-weight-on-bedrock-litellm.md).
