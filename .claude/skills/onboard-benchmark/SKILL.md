---
name: onboard-benchmark
description: "Guide someone through standing this benchmark up on THEIR OWN infrastructure: their models (reached through a corporate FM gateway, Amazon Bedrock, or a self-hosted server), their private code repositories (including GitHub Enterprise Server), and their own dataset of tasks. Establishes the machine, determines the gateway's API shape and picks the harness that can speak it, proves the agent AND the judge reach a model with a real call, gets private-repo credentials and internal CA certificates working, authors a dataset from their closed issues, makes every pre-flight check pass, and runs one smoke task end to end. Ends by handing off to /benchmark for the real batch and to build_vended_models.py to package the result as a swe-router skill. Use when someone wants to benchmark their own repos, run this in their own AWS account, set it up against their own gateway, or asks why a pre-flight check is failing on their infrastructure."
license: Apache-2.0
metadata:
  author: Amit Arora
  version: "1.0"
---

# Onboard Benchmark Skill

Take someone from "we have models and repositories" to "one scored run on our own code, and a plan for the rest". Everything published in this repository was produced this way; this skill is that path made repeatable for somebody else's infrastructure.

## When to use this, and when not to

| Situation | Skill |
|---|---|
| Fresh box, nothing installed | [`setup-machine`](../setup-machine/SKILL.md) first, then this |
| Their own models, their own repos, never run here before | **This skill** |
| Everything already wired, run one more model | [`benchmark`](../benchmark/SKILL.md) |
| Stand up a vLLM server on their GPUs | [`vllm-setup`](../vllm-setup/SKILL.md) |
| Results exist; put them in front of developers | [`swe-router`](../swe-router/SKILL.md) and `build_vended_models.py` |

This skill assumes model hosting **already exists**. It does not provision GPUs or serve a model.

## The shape of the work

Ten steps. Steps 1 to 5 are wiring and each one ends in a proof; steps 6 to 10 produce a result. Do them in order: every step after the first depends on the one before it, and skipping a proof is what turns a twenty-minute setup into an afternoon.

Tell the person the honest timeline up front, from measured runs: one 21-task model takes **110 to 449 minutes** of wall clock depending on the model, plus roughly 50 minutes of judging. A single task is 5 to 21 minutes. So today produces one scored task and a working pipeline, and the frontier comes later.

---

## Step 1 - Establish the machine, and confirm it reaches both sides

**Ask where this will run before anything else, and confirm that machine reaches two separate things.** Every task makes two network calls that must both succeed from the same host: a `git clone` from the code host, and a model call to the gateway. Those usually live in different places, and a box that reaches one but not the other cannot run a single task.

Say plainly what qualifies, because people assume this needs dedicated infrastructure: anywhere Python runs and dependencies can be installed. An EC2 instance, a build or CI runner, or a development workstation or laptop all work. No GPU is required when the models are hosted elsewhere.

Verify from the machine itself, never from a laptop or a network diagram:

```bash
git ls-remote https://<their-git-host>/<org>/<repo> >/dev/null && echo "git OK"
curl -sS -o /dev/null -w 'gateway HTTP %{http_code}\n' "$GATEWAY_URL/v1/models" \
  -H "Authorization: Bearer $GATEWAY_API_KEY"
```

If only one side is reachable, stop and say so. A subnet, security-group, peering or PrivateLink change is not something to work around from here, and it can take days to approve. Offer the useful fallback: run the first pass against one **public** repository so the pipeline is proven end to end while the network request is in flight, then swap the dataset over.

**There is a third network dependency, and it is easy to forget: a package index.** Installing anything at all needs one, and a locked-down host often has neither:

- **PyPI**, for `uv sync` in both projects. Allowlist `pypi.org` **and** `files.pythonhosted.org`; wheels come from the latter, so permitting only the first fails partway through the install. On an internal index, set `UV_DEFAULT_INDEX` (`UV_INDEX_URL` on older uv).
- **`registry.npmjs.org`**, because the agent and the judge are npm packages (`@anthropic-ai/claude-code`, `@openai/codex`). On an internal mirror, `npm config set registry <url>`.
- **The installer and OS sources**: `astral.sh` (uv), `deb.nodesource.com` (Node 22), `cli.github.com` (gh), `omp.sh` (omp), plus the distro's apt or dnf mirrors.

Behind a proxy, `HTTPS_PROXY` and `NO_PROXY` need setting before the installer runs, and `NO_PROXY` must include the gateway and the git host or those calls get sent to the proxy too. If egress is closed entirely, the options are an internal mirror, a proxy, or a pre-baked image; say which applies rather than retrying a failing install.

Then install the dependencies. No GPU is needed when the models are hosted elsewhere: this box clones repositories, runs a coding-agent CLI as a subprocess, and writes JSON, so it is bound by network and by the gateway's rate limit.

```bash
.claude/skills/setup-machine/setup-machine.sh --check
.claude/skills/setup-machine/setup-machine.sh --install --with-omp \
    --git-name "..." --git-email "..."
```

`--with-omp` matters: the core installer covers `claude`, `codex` and `pi`, and `omp` is a separate third-party installer behind that flag. Four vCPU and 16 GB is enough for three to five concurrent tasks.

Installing the CLIs does **not** point them at any model. That is steps 3 and 4, and it is where onboarding actually fails.

## Step 2 - Determine the gateway contract, then pick the harness

Two facts decide everything downstream. **Ask; do not assume, and do not infer from the vendor's name.**

**2a. Which API shape does the gateway expose?** This selects the harness, because the agents speak different protocols:

| Gateway exposes | Harness to use | Why |
|---|---|---|
| OpenAI Chat Completions (`POST /v1/chat/completions`) | `omp` (or `pi`) | The harness writes them an `openai-completions` provider block |
| Anthropic Messages (`POST /v1/messages`) | `claude` | Claude Code drives `ANTHROPIC_BASE_URL` and needs the Anthropic route |
| OpenAI Responses (`POST /v1/responses`) | `codex` | Recent codex speaks only the Responses API, not chat-completions |
| Both chat-completions and Anthropic | `omp` | It produced this repository's headline results |
| Native Amazon Bedrock, no gateway | any of `claude`, `pi`, `omp`, `codex`, `strands` | Use `--provider bedrock` and ambient AWS credentials |

`strands` (the [Strands Agents SDK](../../../docs/strands-setup.md)) is a further option on either provider. Unlike the others it has no CLI binary, so there is nothing to install on PATH: it runs in-process and the orchestrator installs its optional dependency group with `uv sync --group strands`.

Two notes on `codex` as the harness. It is also the judge, so running `--agent codex` has one tool on both sides of the run; keep the judge model different from the model under test. And on a self-hosted vLLM endpoint it needs a Responses-safe tool-call parser (`qwen3_coder` or `hermes` are verified; seven others crash on the flat Responses tool shape, issue #183).

Settle it with a real call rather than a document:

```bash
curl -sS "$GATEWAY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $GATEWAY_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"<their-model-id>","messages":[{"role":"user","content":"say ok"}],"max_tokens":16}'
```

**2b. How does it authenticate?** The harness sends a **static** key. A long-lived bearer token or API key works. SigV4, OIDC, or a token that rotates mid-run does **not**, and that is a blocker rather than a workaround: say so plainly and offer the two real options, which are a small local proxy that holds the rotating credential and presents a static one, or a code change. Do not attempt to make the harness refresh a credential.

**2c. Collect the exact model ids** as the gateway spells them, not the marketing names. If it serves `/v1/models`, that listing is the answer.

State the decision back before moving on: the harness, the provider mode, and the model ids.

## Step 3 - Wire the agent to the gateway, and prove it

Create the runner config once, and keep the token out of it.

```bash
cd benchmarks
uv sync
cp config/runner.example.yaml config/runner.yaml
```

Edit `config/runner.yaml`:

```yaml
provider: endpoint
endpoint: https://gateway.example.com
api_key_env: GATEWAY_API_KEY     # names the variable; the token itself never lands in the file
context_window: 200000           # the served window, so auto-compaction is calibrated
```

`api_key_env` wins over `api_key` and fails at config load when the variable is unset, rather than hours later inside the agent with an authentication error that never mentions it. `config/runner.yaml` is gitignored; `api_key: local` is correct only for a self-hosted vLLM server, which ignores the value.

Validate the config, then prove the CLI itself reaches a model:

```bash
export GATEWAY_API_KEY=...
uv run scripts/runner_config.py config/runner.yaml \
    --model <their-model-id> --dataset dataset/hello-world.yaml

# omp against the gateway
omp -p --mode json --model <provider>/<their-model-id> -- "Reply with the single word ok"
```

A successful `curl` in step 2 does not prove this. The CLI has its own configuration layer, and this is the call that exercises it.

**`context_window` is not optional on a gateway.** Claude Code and omp cannot detect the window of a model behind a custom base URL, so without it the conversation grows until the endpoint rejects the request, and the client retries that rejection forever. Set it to the real served window. On a self-hosted vLLM path the orchestrator reads it from `/v1/models` automatically; a gateway usually will not tell you, so ask.

## Step 4 - Wire the judge, and prove it separately

The judge is a **second, independent** model call and it is the step most often left until it is too late. `codex_judge.py` runs `codex exec` with the candidate's repository checked out read-only, so the judge can verify that files and functions the design cites actually exist. Note that `codex` can also be the harness (step 2), and the two roles are separate calls with separate models; if they chose `--agent codex`, the judge model must differ from the model under test.

Two things to settle:

- **Which model judges.** It must be a strong reasoning model and it must not be the model under test. Set it with `JUDGE_MODEL`; the default is `openai.gpt-5.6-sol`, which their gateway will probably not serve.
- **How `codex` reaches it.** Routing lives in codex's own configuration, not in this repository. See [benchmarks/docs/agent-cli-bedrock-setup.md](../../../benchmarks/docs/agent-cli-bedrock-setup.md).

```bash
export JUDGE_MODEL=<their-judge-model-id>
codex exec --skip-git-repo-check "Reply with exactly: JUDGE OK"
```

**Working AWS credentials are not sufficient.** An unconfigured `codex` ignores the environment entirely and fails against its own default endpoint. If this call does not return, stop here: the harness would run for hours and then have nothing to score it with.

If `codex` genuinely cannot be pointed at their models, [llm_as_judge.py](../../../benchmarks/scripts/llm_as_judge.py) accepts a `--base-url` and is the fallback. Say what it costs them: it scores the artifacts in isolation with no repository access, so it cannot catch a design citing a file that does not exist, and its scores are not comparable with the published ones.

## Step 5 - Repository access, and prove every repo

The harness clones per task using whatever credentials the shell already has, so the requirement is a credential that works **without prompting**. An unattended run would hang on a password prompt rather than fail.

```bash
# A read-scoped token, recorded once
git config --global credential.helper store
git clone --depth 1 https://ghe.example.com/org/repo /tmp/once && rm -rf /tmp/once
```

An `insteadOf` rewrite or a deploy key works equally well. A read scope is enough: the harness edits a throwaway clone, captures the result as `patch.diff`, and never pushes.

Say which trade-off they are taking: `credential.helper store` writes the token **in plaintext** to `~/.git-credentials` (git creates it `0600`), which suits a single-purpose benchmark box and not a shared one. The `insteadOf` form keeps it in the environment; a deploy key or a platform credential manager avoids an on-disk secret. **Never offer `GIT_SSL_NO_VERIFY` as the fix for a TLS error** -- trust the CA instead.

**Internal certificate authority: two variables, not one.** They fail separately for the same reason, which is what makes this expensive to diagnose:

```bash
export GIT_SSL_CAINFO=/etc/pki/tls/certs/internal-ca.pem      # git
export NODE_EXTRA_CA_CERTS=/etc/pki/tls/certs/internal-ca.pem # the agent CLIs are Node programs
```

Once a dataset exists (step 6), check every repository and ref at once:

```bash
uv run python scripts/preflight_check.py --dataset dataset/<theirs>.yaml --check-repos
```

This uses `git ls-remote`, so it transfers no objects and costs seconds. It reports a missing credential, a rejected token, an untrusted CA, an unresolvable host and a tag that does not exist as separate diagnoses, each with the fix. Run it before every batch, not just the first.

**The judge clones too**, into its own read-only checkout, so it needs the same read credential.

## Step 6 - Author the dataset

The highest-value step, and the one where a first attempt usually goes wrong. Start from the annotated template:

```bash
cp dataset/workshop-template.yaml dataset/<their-team>.yaml
```

Four rules to state explicitly:

1. **Pin the release *before* the fix.** For each task, find the closed issue, find the release that shipped its fix, and pin the release before that one. Pin a ref that already contains the fix and the agent finds the work already done, so the task measures nothing and every model scores about the same on it. This is the rule that decides whether the dataset is worth running.
2. **`ground_truth` is never shown to the agent.** It exists so a human can check the judge. Say this unprompted: the first question a sceptical engineer asks is whether the model was handed the answer.
3. **Spread across complexity tiers.** Five or six tasks per tier across `trivial`, `low`, `medium`, `high`. A single overall mean hides that the ranking between two models flips by tier (see [docs/model-selection-by-complexity.md](../../../docs/model-selection-by-complexity.md)). Ten to twenty tasks before trusting a frontier; one task today.
4. **Set `output_scope`** whenever two datasets target the same repository. `run-summary.json` sits at the scope level and is rebuilt from every task folder under it, so sharing a scope silently averages two task sets into one number.

Write the `problem_statement` like a well-scoped ticket: what to change, the constraints, what "done" means, enough that an agent can act with nobody from their team present.

```bash
uv run scripts/dataset_loader.py dataset/<their-team>.yaml     # shape
uv run scripts/preflight_check.py --dataset dataset/<their-team>.yaml --check-repos   # reachability
```

## Step 7 - Smoke run, one task, end to end

Prove the whole chain before committing hours to it.

```bash
cd benchmarks
./scripts/run-e2e-benchmark.sh --provider vllm --model <their-model-id> \
    --endpoint https://gateway.example.com \
    --dataset dataset/<their-team>.yaml \
    --agent omp --skill swe3 --count 1 --yes
```

`--provider vllm` is the OpenAI-compatible **endpoint** path and is the right choice for a gateway despite the name; pass `--endpoint` to override the local default. Use `--provider bedrock` only for models reached natively on Amazon Bedrock.

Walk the output tree with them, because this is where the mental model forms:

```
benchmarks/swe-benchmark-data/<model>/<harness>/<skill>/<scope>/<task>/
  github-issue.md  lld.md  review.md  testing.md  patch.diff  implementation.md
  metrics.json     eval.json
```

Six artifacts. `metrics.json` is what it cost, `eval.json` is how good it was, and model, harness and skill are each their own level so a second harness never overwrites the first.

If `hello-world.yaml` was going to be the smoke dataset, check first that the box has public egress: it clones a public GitHub repository, and failing there for a reason unrelated to their setup is the worst kind of failure to debug in front of someone.

## Step 8 - Decide the cost basis, and write it down

This repository documents two cost bases, and a gateway-served model is **neither**. Settle it before any dollar figure gets quoted, or the first chart will be wrong in a way nobody notices:

- **The gateway re-bills per token** -> use that rate; it is a metered basis like Amazon Bedrock.
- **The gateway fronts hardware they own** -> it is the hardware-derived basis, which means `instance $/hr` divided by measured throughput at a stated concurrency. Never wall-clock times hourly rate, which charges for the time the agent spent thinking and assumes one developer owns the whole box.

Record the choice and the rate alongside the results. [docs/cost-per-task-methodology.md](../../../docs/cost-per-task-methodology.md) has the derivation, including why dollars from different bases must not be compared as raw numbers.

## Step 9 - Scale to the real batch

Hand off to the [`benchmark`](../benchmark/SKILL.md) skill for one model, or go straight to the batch runner for several:

```bash
./scripts/run-multi-model-benchmark.sh <model-a> <model-b> \
    --agent omp --skill swe3 --judge-mode async
```

Four things to tell them:

- **`--judge-mode async`** scores the finished model while the next one generates. The judge uses no GPU, so the overlap is free; inline leaves roughly 50 minutes idle per 21-task model.
- **Detach it.** `setsid nohup ... > log 2>&1 < /dev/null &` leaves the run with no controlling terminal, so no `SIGHUP` reaches it and its output lands on disk rather than in a pipe that dies with the session.
- **`--concurrency N`** is the wall-clock lever, bounded by the gateway's rate limit. Start at 1 and raise it once a run has completed cleanly. Above 1, the per-run vLLM cache metrics become a server-wide aggregate.
- **Scores from different datasets never merge.** Different tasks, refs and difficulty mixes mean separate tables.

## Step 10 - Package the result for developers

The frontier is the point, and a frontier nobody reads changes no spending.

```bash
uv run scripts/build_vended_models.py     # regenerates vend/swe-router/models.json
```

Then install `swe-router` into a repository their developers work in, and **have them edit `allowed-models.txt`**: the skill treats every name in it as a model a developer can select, so listing an unreachable model wastes the cheap option and omitting a reachable one wastes the saving.

The vended file carries provenance (date, harness, skill, dataset, judge) so a stale copy is visible as stale. Say the honest number from this repository's own evidence: routing came out 45.4% cheaper for 4.6% less quality over 21 tasks, and the caveats live in the judgment step rather than the arithmetic ([docs/swe-router-evaluation.md](../../../docs/swe-router-evaluation.md)).

Close by naming an owner for the frontier and a cadence. New models and harness changes land constantly, so a frontier measured today is stale within weeks. A benchmark with no owner is run once and cited for a year.

---

## Failure modes, in the order they actually happen

| Symptom | Cause | Fix |
|---|---|---|
| Gateway reachable but `git clone` is not, or the reverse | The machine sits on one side of the network only | Step 1. Needs a network change, not a workaround; run against a public repo meanwhile |
| `uv sync` resolves then fails downloading | `pypi.org` allowlisted but `files.pythonhosted.org` is not | Allowlist both, or point `UV_DEFAULT_INDEX` at the internal index |
| `npm` install of the agent or judge hangs or 403s | No route to `registry.npmjs.org` | Internal mirror via `npm config set registry`, or a proxy |
| Everything installs, then the gateway call goes to the proxy | `NO_PROXY` omits the gateway and git host | Add both to `NO_PROXY` |
| `codex` 401s against its own default endpoint | Provider config missing; ambient AWS credentials are ignored | [agent-cli-bedrock-setup.md](../../../benchmarks/docs/agent-cli-bedrock-setup.md), then re-prove with a real call |
| `git clone` prompts for a password | No credential helper entry, so an unattended run hangs | Step 5 |
| TLS failure in git but not the agent, or the reverse | Only one of the two CA variables is set | `GIT_SSL_CAINFO` **and** `NODE_EXTRA_CA_CERTS` |
| `omp CLI not found on PATH` | Installer put it in `~/.local/bin` | Add to `PATH`, or re-run setup with `--with-omp` |
| Endpoint 500s about maximum context length, forever | `context_window` unset, so auto-compaction never fires | Set it to the served window (step 3) |
| Every task scores about the same on a task | `ref` pins a release that already contains the fix | Re-pin to the release before it (step 6) |
| Run stalls with no output | Pre-existing artifact folders block the headless overwrite prompt | `preflight_check.py --clear`, or pass `--yes` |
| Authentication error hours into a batch | A rotating credential expired mid-run | Not supportable directly; see step 2b |

## Stay inside the lines

- **Never print or echo a token**, theirs or otherwise. Refer to the variable name.
- **Prove each of steps 3, 4 and 5 with a real call** before moving on. A configuration file that looks right is not evidence.
- **Do not guess the gateway's API shape, its auth model, or a model id.** Ask, or test.
- **Do not widen the agent's permissions** to make a run pass. The harness runs an autonomous agent inside a cloned third-party repository, so the permission mode and tool allowlist are a security boundary.
- **Do not promise a frontier from one session.** One scored task and a working pipeline is the honest deliverable; state the measured timeline from the start.
