# kiro-cli setup and benchmark-integration notes

Install and configuration reference for **kiro-cli** as a coding agent (harness) for this benchmark. This file covers how to install kiro-cli, how it authenticates, how to drive it headlessly, and -- importantly -- the two ways it differs from the `pi` and `claude-code` harnesses that constrain how it can be wired into the benchmark. Read the "Benchmark integration status" section before starting the harness wiring in issue #73.

## What kiro-cli is

kiro-cli is the command-line agent from [kiro.dev](https://kiro.dev), and is the successor to the Amazon Q Developer CLI (the installed binary still ships a `q` shim and reports internal `fig_*` / AWS SDK components). It runs an interactive or non-interactive coding agent in the terminal, backed by **Kiro's managed models** (powered by Amazon Bedrock), selected through the CLI rather than a user-supplied endpoint.

> **As of this version (kiro-cli 2.18.1), kiro-cli supports only Amazon Bedrock-backed managed models.** There is no support for any other backend -- no self-hosted vLLM, no OpenAI-compatible base URL, no third-party provider. Model selection is limited to the managed list returned by `kiro-cli chat --list-models`, all served through Amazon Bedrock. This is the key constraint that shapes the benchmark integration below.

Verified version at time of writing: **kiro-cli 2.18.1**.

## Prerequisites

- Linux, macOS, or Windows. This repo's node is Linux (Amazon Linux / Ubuntu on EC2).
- An AWS sign-in for kiro-cli: a free **Builder ID** (or Google/GitHub social login), or a **pro** IAM Identity Center license. kiro-cli will not run a chat turn until you are logged in.
- Network egress to `cli.kiro.dev`, `prod.download.cli.kiro.dev`, and the AWS sign-in endpoints.

## Install

The official installer covers macOS, Linux, and Windows:

```bash
curl -fsSL https://cli.kiro.dev/install | bash
```

It installs three binaries into `~/.local/bin` (ensure that is on your `PATH`):

- `kiro-cli` -- the launcher and subcommand entry point.
- `kiro-cli-chat` -- the chat/agent backend (`kiro-cli chat` dispatches to it).
- `kiro-cli-term` -- the terminal integration.

To inspect the installer before running it (recommended on a shared node):

```bash
curl -fsSL https://cli.kiro.dev/install -o /tmp/kiro-install.sh
less /tmp/kiro-install.sh          # review, then:
bash /tmp/kiro-install.sh
```

### Verify

```bash
export PATH="$HOME/.local/bin:$PATH"
kiro-cli --version        # -> kiro-cli 2.18.1
kiro-cli --help-all       # full subcommand list
```

## Authenticate

kiro-cli requires a sign-in before any chat turn. Any command that needs a model (for example `kiro-cli chat ... --list-models`) will otherwise drop into an interactive device-code login and block.

The **free** license accepts a Builder ID or a social login (Google or GitHub). Signing in with a **Google account** via the free-license device flow is confirmed working on this node -- no AWS account of your own is required.

```bash
# Free license: Builder ID, or Google / GitHub social login (Google confirmed working):
kiro-cli login --license free --use-device-flow

# Pro (IAM Identity Center):
kiro-cli login --license pro \
  --identity-provider https://<your-start-url>.awsapps.com/start \
  --region us-east-1 --use-device-flow

kiro-cli whoami           # confirm the signed-in identity
```

`--use-device-flow` prints a URL and a code to confirm in a browser -- use it on headless servers where a browser redirect cannot be handled.

`KIRO_HOME` redirects the global `~/.kiro` directory (config, settings, session store) to another location -- useful for keeping a per-run or per-profile config isolated from a developer's global setup:

```bash
KIRO_HOME=/path/to/run-config kiro-cli whoami
```

## Headless (non-interactive) use

The non-interactive form takes a prompt argument, selects a model with `--model`, and pre-approves tools (no operator is present to confirm tool calls):

```bash
kiro-cli chat --no-interactive --trust-all-tools --model claude-sonnet-5 "Find all TODO comments in src/"
```

`--model` is optional (kiro-cli falls back to the `auto` default), but the benchmark always passes it so a run is pinned to a known model. List the model names you can pass to `--model` (requires login):

```bash
kiro-cli chat --list-models              # human-readable
kiro-cli chat --list-models --format json # machine-readable (name, context, rate_multiplier)
```

Relevant `kiro-cli chat` flags:

| Flag | Purpose |
|---|---|
| `--no-interactive` | Run without an interactive session; requires a prompt argument. |
| `--model <MODEL>` | Select a model from Kiro's managed list (see `--list-models`). |
| `--effort <LEVEL>` | Reasoning effort: `low`, `medium`, `high`, `xhigh`, `max`. |
| `--agent <AGENT>` | Use a named agent config (see `kiro-cli agent`). |
| `--trust-all-tools` | Auto-approve every tool call (required when non-interactive). |
| `--trust-tools=<names>` | Auto-approve only specific tools, e.g. `fs_read,fs_write`. |
| `--list-models -f json` | List available managed models as JSON (requires login). |

Prompt context can also be piped in on stdin:

```bash
cat build-error.log | kiro-cli chat --no-interactive --trust-all-tools "Explain this failure and suggest a fix"
```

### Headless output and metrics

A non-interactive chat turn streams **ANSI-colored narration to stdout** (the agent's edits and tool calls) -- not JSON. It does **not** report input/output token counts. It **does** print a one-line run summary to **stderr** on completion:

```
 ▸ Credits: 0.21 • Time: 17s
```

So the two capturable per-run metrics are **Credits** (Kiro's billing unit) and **wall-clock time**; success is gated on the exit code and the presence of expected artifacts. Credits, not tokens, are the cost signal for a kiro-cli harness.

### Available managed models

`kiro-cli chat --list-models --format json` (requires login) returns the managed models and a per-model `rate_multiplier` in Credits -- a relative cost weight. As of this writing the list includes the same families this benchmark uses, for example:

| Model | Context | Rate (Credit) |
|---|---|---|
| qwen3-coder-next | 256k | 0.05 |
| gpt-5.6-luna | 272k | 0.10 |
| deepseek-3.2 | 164k | 0.25 |
| minimax-m2.5 | 196k | 0.25 |
| claude-haiku-4.5 | 200k | 0.40 |
| glm-5 | 200k | 0.50 |
| auto (default) | 1M | 1.00 |
| claude-sonnet-5 | 1M | 1.30 |
| claude-opus-5 | 1M | 2.20 |
| gpt-5.6-sol | 272k | 2.40 |

(Full list also includes claude-opus-4.5/4.6/4.7/4.8, claude-sonnet-4/4.5/4.6, gpt-5.6-terra, minimax-m2.1, and internal-only entries. `default_model` is `auto`.)

### Translating credits to dollars

kiro-cli bills in **credits**, so a dollar cost per task is `credits_consumed x $/credit`. The per-run credits come from the stderr summary line and **already include the model's `rate_multiplier`** (a run on claude-opus-5 at 2.2 burns credits faster than one on qwen3-coder-next at 0.05), so do not multiply by the rate again.

Kiro's published pricing gives two defensible per-credit rates:

| Basis | $/credit | Derivation |
|---|---|---|
| Blended (included monthly allotment) | **$0.02** | Every paid tier is the same rate: Pro $20/1,000, Pro+ $40/2,000, Pro Max $100/5,000, Power $200/10,000 |
| Marginal (add-on / overage) | **$0.04** | "Add-on credits $0.04/credit" once the monthly allotment is spent |

```
cost_per_task_usd = credits_consumed_for_the_run x DOLLARS_PER_CREDIT
```

Treat `DOLLARS_PER_CREDIT` as a **configurable rate**, the same way the self-hosted GPU rate is a documented commitment term in this repo, swappable for your own (see [cost-per-task-methodology.md](cost-per-task-methodology.md)). The default is the **$0.04 marginal** rate -- the honest "what does one more task cost" figure -- with $0.02 available for an all-you-can-use blended view. Example: a run reporting `Credits: 0.21` costs `0.21 x $0.04 = $0.0084` (or `$0.0042` at the blended rate). Trivial tasks cost cents; a real swe task at 50-300 turns consumes far more credits.

Kiro credits are a **third cost basis**, alongside metered Bedrock dollars and hardware-derived self-hosted GPU-seconds. As with those, compare within a hosting basis rather than reading raw dollars across bases; the credit-to-dollar conversion depends on your Kiro plan.

### Caveat: Kiro is a per-developer subscription, and this figure ignores that

Kiro's real pricing is a **per-developer monthly subscription** ([kiro.dev/pricing](https://kiro.dev/pricing/)): Free ($0/mo, 50 credits), Pro ($20/mo, 1,000), Pro+ ($40/mo, 2,000), Pro Max ($100/mo, 5,000), Power ($200/mo, 10,000). Those credits are **included in the seat**; the **$0.04/credit** default applies **only to overage** beyond the monthly allotment. So the `credits x $0.04` cost the harness reports treats **every credit as if it were add-on overage** -- the conservative worst case. A developer working within their monthly allotment has effectively already paid for those credits via the seat; the amortized rate is nearer the blended **$0.02/credit**.

This matters when comparing kiro-cli to the **pi** and **Claude Code** harnesses: driving those through **Amazon Bedrock is pure usage-based, per-token** billing -- no seat, no monthly commitment. kiro-cli instead bundles a **fixed monthly seat with an included credit allotment**. A fair total-cost comparison models kiro's **seat cost + expected monthly volume** against the others' metered/hardware spend, rather than treating the single per-task credit-dollar figure as equivalent. See [cost-per-task-methodology.md](cost-per-task-methodology.md).

## Benchmark integration status

kiro-cli is **not yet wired** into the benchmark harness. Two properties of the tool differ from the `pi` and `claude-code` harnesses and shape how it can be integrated (tracked in issue #73):

1. **No self-hosted / OpenAI-compatible endpoint.** Unlike `pi` (which points at a vLLM endpoint via `--provider vllm`), kiro-cli talks only to Kiro's managed, Bedrock-backed models and authenticates through AWS. There is no base-URL or custom-endpoint setting. **A "kiro-cli against self-hosted vLLM" run -- the original framing of issue #73 -- is therefore not possible.** kiro-cli can only be benchmarked driving its own managed models.

2. **No token counts, but a credits + time signal.** `--format json` applies to `--list-models`/`--list-sessions` only; a non-interactive chat turn streams ANSI-colored text to stdout (not JSON) and does **not** report input/output token counts. It **does** print `▸ Credits: <n> • Time: <s>s` to stderr on completion (see "Headless output and metrics"). An integration would normalize **credits** (parsed from stderr), not tokens, as the cost metric, gating success on the exit code and artifact presence. This differs from `pi`/`claude-code`, which emit token- and dollar-level accounting.

The practical consequence: if kiro-cli is added as a third harness, it would be a **managed-model** harness (its own Kiro models, priced in **Kiro credits** -- a third cost basis alongside metered Bedrock dollars and hardware-derived self-hosted GPU-seconds, so compare within a hosting basis as the repo already does). Per-model credit weights come from `--list-models` (`rate_multiplier`); per-run credits come from the stderr summary line. It cannot be a self-hosted-vLLM harness. Confirm the scope before mirroring the `pi` wiring described in issue #73.

## References

- Install and CLI docs: https://kiro.dev/docs/cli/
- Headless mode: https://kiro.dev/docs/cli/headless/
- Models: https://kiro.dev/docs/models/
- Benchmark harness reference: [benchmarks/docs/harness-reference.md](../benchmarks/docs/harness-reference.md)
