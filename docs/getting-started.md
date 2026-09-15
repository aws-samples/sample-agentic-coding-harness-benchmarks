# Getting started

What to install, and the order to do it in, from a fresh box to a first benchmark run.

## Prerequisites

> **On a fresh machine, start with the [`/setup-machine` skill](../.claude/skills/setup-machine/SKILL.md).** It inspects the box, prints exactly what it will install and why, installs it, and summarizes -- so you do not have to work through the list below by hand. It also installs the GPU stack (vLLM, nvtop, nvitop) only when a GPU is present, and puts the vLLM venv and its caches on the large ephemeral NVMe when the root disk is too small.

- An **AWS account** with [Amazon Bedrock model access](https://console.aws.amazon.com/bedrock/home#/modelaccess) enabled for the models you want (Paths 1 and 2).
- **AWS credentials** configured locally (`aws configure`, an IAM role, or AWS SSO).
- **[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)** installed.
- **[uv](https://docs.astral.sh/uv/)** and **Python 3.10+** for the harness.
- For Path 3: permission to launch an **EC2 GPU instance** (for example `g6e.12xlarge`).

> The `bedrock-mantle` endpoint used for Path 2 (third-party models) is available in **`us-east-1`**.

## Get started

1. **Set up the machine -- do this first on any new box.** Run the **`/setup-machine` skill** from Claude Code (or its script directly). It reports what the instance is, lists every missing dependency with the reason each one is needed, installs them, and prints a summary table:

   ```bash
   # Dry run: report only, install nothing
   .claude/skills/setup-machine/setup-machine.sh --check

   # Install everything missing (git identity is required, never guessed)
   .claude/skills/setup-machine/setup-machine.sh --install \
       --git-name "Your Name" --git-email "you@example.com"
   ```

Add `--with-omp` / `--with-kiro` to include those two harnesses (opt-in: both ship third-party install scripts, and kiro-cli needs an interactive sign-in). See [.claude/skills/setup-machine/SKILL.md](../.claude/skills/setup-machine/SKILL.md) for the full component list and flags.

2. **Set up the harness** (its own isolated virtual environment):

   ```bash
   cd benchmarks
   uv sync
   cp config/runner.example.yaml config/runner.yaml
   ```

3. **Wire the agent CLIs to Amazon Bedrock.** Installing `claude` and `codex` does not configure them -- an unconfigured `codex` silently calls `api.openai.com` and 401s mid-run. Follow [benchmarks/docs/agent-cli-bedrock-setup.md](../benchmarks/docs/agent-cli-bedrock-setup.md).

4. **Run a benchmark.** The fastest way is the **`/benchmark` skill** from Claude Code, which drives the whole flow interactively -- pre-flight checks, the harness run over a dataset, and the judge -- for any of the three paths. It even manages the vLLM server and metrics collector for the self-hosted path:

   ```
   /benchmark provider=vllm model=qwen3.6-35b dataset=dataset/mcp-gateway-registry.yaml
   ```

Prefer a script? The same flow runs headless via [benchmarks/scripts/run-e2e-benchmark.sh](../benchmarks/scripts/run-e2e-benchmark.sh) (`--provider bedrock|litellm|vllm --model ... --dataset ...`).

5. **Pick a path and follow its guide** for the setup details each one needs -- every guide ends with a copy-pasteable run command:
- [Path 1 - Anthropic models directly on Amazon Bedrock](../benchmarks/docs/path-anthropic-on-bedrock.md)
- [Path 2 - open-weight models on Amazon Bedrock via a LiteLLM proxy](../benchmarks/docs/path-open-weight-on-bedrock-litellm.md)
- [Path 3 - self-hosted open-weight models on EC2 with vLLM](../benchmarks/docs/path-self-hosted-vllm.md)

6. **Read the shared mechanics** once (they apply to every path): the [harness reference](../benchmarks/docs/harness-reference.md) covers the dataset format, the runner config, running the benchmark, the metrics file, and the judge.

For Path 3 you must first stand up the vLLM server itself -- see [self-hosted/vllm/README.md](../self-hosted/vllm/README.md) (or let the `/benchmark` skill start it for you).


---

[< Back to the README](../README.md)
