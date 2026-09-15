# Wiring the agent CLIs to Amazon Bedrock

The harness shells out to two CLIs that must reach **Amazon Bedrock** on their own: `claude` (or `pi` / `omp` / `kiro-cli`) produces the artifacts, and `codex` scores them as the judge. Neither is configured by the harness, the `/benchmark` skill, or any script here -- they read their own config, so a machine can pass every pre-flight check and still fail the moment a model is invoked.

**`aws sts get-caller-identity` succeeding is not enough.** The pre-flight in [end-to-end-self-hosted-run.md](end-to-end-self-hosted-run.md) only proves the instance can reach AWS. A CLI that has not been pointed at Bedrock will ignore those credentials entirely and call its vendor's public API instead. The failure looks like this, on a box whose exec role is perfectly healthy:

```text
ERROR: unexpected status 401 Unauthorized: Missing bearer or basic authentication in header,
       url: https://api.openai.com/v1/responses
```

That is `codex` talking to OpenAI, not Bedrock. Nothing about it mentions Bedrock or AWS, which is what makes it slow to diagnose.

This page is the copy-pasteable fix for both CLIs. It is condensed from [aarora79/claude-codex-bedrock-ec2](https://github.com/aarora79/claude-codex-bedrock-ec2), which carries the fuller version (VS Code extension setup, the legacy LiteLLM route, region discovery); when the two disagree, that repo is upstream.

## Prerequisite: the exec role

Both CLIs use the standard AWS SDK credential chain, so an EC2 instance role with Bedrock access needs no keys on disk:

```bash
aws sts get-caller-identity
```

The principal needs `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` (plus `bedrock:ListInferenceProfiles` if you use inference profiles). For the judge it also needs the **`bedrock-mantle`** OpenAI-compatible endpoint; the managed policy `arn:aws:iam::aws:policy/AmazonBedrockLimitedAccess` covers both.

## Codex (the judge)

Codex ships a native `amazon-bedrock` provider that talks to `bedrock-mantle` and authenticates from the credential chain. **No proxy and no bearer token are required** -- the LiteLLM route older notes describe is legacy.

```bash
npm install -g @openai/codex
codex --version    # must be >= 0.144 for the native provider
```

```bash
mkdir -p ~/.codex
cat > ~/.codex/config.toml <<'EOF'
model_provider = "amazon-bedrock"
model_providers.amazon-bedrock.aws.region = "us-east-2"
model = "openai.gpt-5.6-sol"
EOF
```

Verify before trusting a benchmark run to it:

```bash
codex exec --skip-git-repo-check "Reply with exactly: JUDGE OK"
```

Two things that decide whether this works:

- **The region must host the model.** The GPT-5.6 models are region-scoped, and asking a region that does not host one returns a 404 (`The model '...' does not exist`), not a helpful message. `openai.gpt-5.6-sol` -- the judge's default ([codex_judge.py](../scripts/codex_judge.py), overridable with `JUDGE_MODEL`) -- runs in **us-east-1 and us-east-2 only**; `openai.gpt-5.6-terra` and `openai.gpt-5.6-luna` add us-west-2.
- **Do not trust `list-foundation-models` for this.** `aws bedrock list-foundation-models --region us-west-2` returns `openai.gpt-5.6-sol`, yet the `bedrock-mantle` endpoint in that region 404s on the same id. The two surfaces do not agree, so the listing is not evidence the judge can reach a model. The `codex exec` call below is the only check that settles it.
- **Scope the region in `config.toml`, not `AWS_REGION`.** `model_providers.amazon-bedrock.aws.region` pins Codex to one region without disturbing the ambient environment -- which matters here, because the same box may point Claude Code at a different region and drives a local vLLM server that reads `AWS_*` for its own reasons.

`--skip-git-repo-check` is only needed outside a git repo or trusted folder; the judge passes its own flags. If Codex warns that `bubblewrap` is missing it falls back to a bundled copy, which is harmless for `--sandbox read-only` judging; `sudo apt install bubblewrap` silences it.

## Codex (the agent, on the Bedrock path)

The same binary also drives tasks as a harness, with `--agent codex`. It reads the `~/.codex/config.toml` above, so a codex that already judges can already run tasks; the harness passes `-c model_provider=amazon-bedrock` and pins `AWS_REGION` from `aws_region` in the runner config, which overrides the region in the file for that run.

Two things differ from every other harness here.

**Its sandbox cannot start on these EC2 hosts, so the harness bypasses it.** Codex normally wraps model-issued shell commands in bubblewrap. Both `--sandbox read-only` and `--sandbox workspace-write` abort before running anything:

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

Creating a loopback interface in an unprivileged user namespace is not permitted on these instances, and installing the distro `bubblewrap` package does not change it -- the bundled and system copies fail identically. The agent then completes no shell work at all and the task produces nothing. The harness therefore passes `--dangerously-bypass-approvals-and-sandbox`, which is what makes the run function rather than a convenience. Every harness here already pre-approves tool use because no operator is watching and the repos are throwaway clones; codex is the one whose flag also removes an OS-level boundary, so **run it only on a disposable instance**. Do not "harden" this back to `--sandbox workspace-write` without re-testing on the target host: the result is a silent zero-artifact run, not a safer one.

**It reports tokens, not cost.** `codex exec` bills nothing back, so the harness prices each run from [bedrock_pricing.py](../scripts/bedrock_pricing.py) (rates dated in the module, per 1M tokens). A model missing from that table yields a null cost rather than a misleading zero, so add a row before benchmarking a new codex model or its runs will not plot on the frontier.

Note that codex's `input_tokens` is the **total** prompt, with `cached_input_tokens` and `cache_write_input_tokens` as subsets of it -- the opposite of the additive shape the rest of this harness uses. The runner subtracts them before pricing. Charging the raw `input_tokens` and then adding the cache lines again overstates cost by roughly 80% on a cache-dominated run, which is the same double-count [token_accounting.py](../scripts/token_accounting.py) guards against (issue #136).

## omp (the agent, on the Bedrock path)

`omp` needs no config file for Bedrock. It ships an `amazon-bedrock` provider, addresses models as `amazon-bedrock/<wire model id>`, and resolves credentials from the standard chain, an EC2 instance role included. All it needs from you is a region:

```bash
export AWS_REGION=us-west-2    # or wherever your Anthropic inference profiles live
omp -p --mode json --no-session \
    --model amazon-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    -- "Reply with exactly: OMP OK" </dev/null
```

A working call reports `"provider":"amazon-bedrock"` and `"api":"bedrock-converse-stream"` in its `message_end` event. Redirect stdin from `/dev/null`: omp reads an inherited stdin as a piped prompt and blocks on EOF, ignoring the positional one ([omp-setup.md](../../docs/omp-setup.md)).

Two traps when you persist that export:

- **`~/.bashrc` returns early for non-interactive shells.** Ubuntu's stock `~/.bashrc` opens with a `case $- in *i*) ;; *) return;; esac` guard, so an export appended to the end reaches interactive shells only. Put it in `~/.profile` as well, or a script and a `systemd --user` unit both start without it.
- **The harness sets the region itself.** `run-swe-headless.py` pins `AWS_REGION` per run from `aws_region` in the runner config, so `--provider bedrock` runs do not depend on your shell at all. The export matters for driving `omp` by hand.

Unlike `codex`, `omp` cannot pin its region in a config file, so this one is ambient. Check what `AWS_REGION` holds before blaming a model id.

## Claude Code (the agent, on the Bedrock path)

Only needed when Claude Code itself must call Bedrock -- that is `--provider bedrock`. On the `vllm` and `litellm` paths the harness points Claude Code at the local server or proxy instead, and only the judge uses Bedrock.

```bash
npm install -g @anthropic-ai/claude-code
```

```bash
cat >> ~/.bashrc <<'EOF'
export CLAUDE_CODE_USE_BEDROCK=1
export AWS_REGION=us-east-1
export ANTHROPIC_MODEL='us.anthropic.claude-opus-4-8[1m]'
export ANTHROPIC_SMALL_FAST_MODEL='us.anthropic.claude-haiku-4-5-20251001'
EOF
source ~/.bashrc
```

Confirm with `/status` inside `claude`.

The **VS Code extension does not read `~/.bashrc`** (it is not a login shell), so put the same variables in the `env` block of `~/.claude/settings.json` and fully restart VS Code -- a window reload is usually not enough, because the extension host reads that file on boot. Merge into the file if it already exists. On Remote-SSH it must be the copy on the remote box.

## Checking both before a run

```bash
command -v claude codex
aws sts get-caller-identity >/dev/null && echo "aws ok"
codex exec --skip-git-repo-check "Reply with exactly: JUDGE OK"
```

The third line is the one that matters: it is the only check that proves the judge can actually reach Bedrock. Run it before starting a long benchmark, because `--skip-judge` is the fallback if it fails, and discovering that after a multi-hour harness run means scoring a second time.

## Related

- [aarora79/claude-codex-bedrock-ec2](https://github.com/aarora79/claude-codex-bedrock-ec2) -- upstream source for this page.
- [end-to-end-self-hosted-run.md](end-to-end-self-hosted-run.md) -- the full manual run-book; its pre-flight assumes what this page sets up.
- [harness-reference.md](harness-reference.md#running-the-codex-judge) -- what the judge does with the model once it can reach it.
- [path-anthropic-on-bedrock.md](path-anthropic-on-bedrock.md) -- Path 1, where Claude Code itself runs on Bedrock.
