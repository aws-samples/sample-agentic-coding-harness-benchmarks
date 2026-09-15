# Path 3 - self-hosted open-weight models on EC2 with vLLM

Use this path to benchmark an **open-weight model you serve yourself** on an EC2 GPU instance with [vLLM](https://docs.vllm.ai) -- Qwen3-Coder, GLM, Kimi, DeepSeek, and so on -- rather than consuming it through Bedrock. You bring up vLLM on a multi-GPU node, expose its OpenAI/Anthropic-compatible API, and the harness wires that endpoint directly into Claude Code with `provider: endpoint`. This is the **throughput path**: a fixed-cost GPU node running many concurrent requests, which is exactly the regime where the `vllm_prometheus` cache and utilization metrics become meaningful.

Serving the model is documented in full in [self-hosted/vllm/README.md](../../self-hosted/vllm/README.md); this guide covers only how the benchmark harness talks to it. For everything common to all paths -- dataset format, runner config, metrics file, and the judge -- see the [harness reference](harness-reference.md).

> **Want the whole flow end to end?** [end-to-end-self-hosted-run.md](end-to-end-self-hosted-run.md) is an ordered run-book -- pre-flight checks, serve the model, capture a live GPU metrics time series into DuckDB, run the benchmark against `mcp-gateway-registry`, and score the artifacts with the judge.

## How it works

vLLM serves an OpenAI/Anthropic-compatible API (including the `/v1/messages` route Claude Code uses) bound to `127.0.0.1:8000` on the EC2 host, with prefix caching always enabled. The harness runs with `provider: endpoint` pointed at that base URL and passes a Claude Code `--settings` object that pins `CLAUDE_CODE_USE_BEDROCK=0`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, and an `apiKeyHelper`. That `--settings` object is what keeps the run on your endpoint: if your global `~/.claude/settings.json` pins `CLAUDE_CODE_USE_BEDROCK=1` (a common setup), merely exporting `CLAUDE_CODE_USE_BEDROCK=0` is silently overridden, the request goes to Amazon Bedrock, and Bedrock rejects the local model id with `400 The provided model identifier is invalid` (see [How `--settings` pins routing](harness-reference.md#how---settings-pins-routing)).

The harness sources the settings two ways:

- If the config sets `settings_file`, it passes that file (e.g. the vLLM [self-hosted/vllm/config/claude-code.json](../../self-hosted/vllm/config/claude-code.json)).
- Otherwise it synthesizes an inline settings object from the config's `endpoint` and `api_key`.

The `apiKeyHelper` is required even against a local server that ignores the key's value: without a token source Claude Code aborts with `Not logged in - Please run /login`. The synthesized settings set it to `echo <api_key>`, so the config's `api_key` field doubles as that token. Because routing is synthesized from `endpoint`/`api_key`, runs work with `settings_file` left unset (commented out) -- keep it set only when you need the extra options in the vLLM settings file.

`--model` must be the `served-model-name` vLLM was launched with (e.g. `qwen3-coder-30b`); it also becomes the `{model-name}` artifact subfolder.

## Metrics on this path

This is the only path where the full `vllm_prometheus` block is populated. The harness scrapes vLLM's Prometheus `/metrics` surface before and after each run and samples the gauges while the run is in flight, so `metrics.json` carries prefix-cache hit rates, per-run token/latency window deltas, and the in-flight KV-cache utilization peak alongside the model-API metrics. See [The metrics file](harness-reference.md#the-metrics-file) for the full structure -- and the **single-tenant caveat**: those numbers are only this run's activity if the run was the sole traffic on the endpoint, so run benchmarks against a dedicated, otherwise-idle server when they matter.

## Prerequisites

- A vLLM server running and reachable at the config's `endpoint`. Bring it up with [self-hosted/vllm/README.md](../../self-hosted/vllm/README.md) (or the `/vllm-setup` skill).
- If the server runs on a remote EC2 host, open the SSH tunnel first so `127.0.0.1:8000` on your machine forwards to the instance (see [Connect a client (SSH tunnel)](../../self-hosted/vllm/README.md#connect-a-client-ssh-tunnel)). vLLM binds loopback only; there is no public ingress.

## Run it

```bash
cd benchmarks

# One-task confirmation against the local (or tunneled) vLLM endpoint
uv run scripts/run-swe-headless.py --config config/runner.yaml \
    --provider endpoint --endpoint http://127.0.0.1:8000 \
    --model qwen3-coder-30b \
    --dataset dataset/mcp-gateway-registry.yaml --count 1 --stream

# Full dataset
uv run scripts/run-swe-headless.py --config config/runner.yaml \
    --provider endpoint --endpoint http://127.0.0.1:8000 \
    --model qwen3-coder-30b \
    --dataset dataset/mcp-gateway-registry.yaml
```

Since `provider: endpoint` and `endpoint: http://127.0.0.1:8000` are the template defaults, once `config/runner.yaml` points at your server you can drop the `--provider`/`--endpoint` flags and just pass `--model` and `--dataset`.

To exercise the throughput path (and get meaningful aggregate cache/utilization metrics), raise concurrency -- but note the trade-off for per-run vLLM numbers described in [Running tasks concurrently](harness-reference.md#running-tasks-concurrently):

```bash
uv run scripts/run-swe-headless.py --config config/runner.yaml \
    --model qwen3-coder-30b \
    --dataset dataset/mcp-gateway-registry.yaml --concurrency 3
```

See the [harness reference](harness-reference.md#common-invocations) for the full set of `--count`, `--tasks`, `--concurrency`, `--stream`, and `--verbose` options, which behave the same on every path.
