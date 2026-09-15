<h1 align="center">Agentic Coding Harness and Benchmarks</h1>

<p align="center">
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT--0-yellow.svg" alt="License: MIT-0"></a>
<a href="https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html"><img src="https://img.shields.io/badge/Amazon-Bedrock-blue" alt="Bedrock"></a>
<a href="./"><img src="https://img.shields.io/badge/Models-45%20from%2011%20providers-orange" alt="Models: 45"></a>
</p>

<p align="center">
<a href="docs/benchmark-your-own-repo.md">Benchmark your own repo</a> |
<a href="docs/hosting-paths.md">Hosting paths</a> |
<a href="docs/how-a-run-works.md">How a run works</a> |
<a href="docs/getting-started.md">Getting started</a> |
<a href="docs/cost-per-task-methodology.md">Cost methodology</a>
</p>

<p align="center">
<video src="https://github.com/user-attachments/assets/84f60f0f-cd7d-4513-b90b-37a0ffb5feed" controls width="820">
Your browser cannot play this video inline. <a href="https://github.com/user-attachments/assets/84f60f0f-cd7d-4513-b90b-37a0ffb5feed">Download the swe-router explainer</a>.
</video>
</p>

<p align="center"><em>swe-router in about 70 seconds: the right model, handed to the developer for each task, while the platform team benchmarks once and spends less.</em></p>

> **This is sample code intended for demonstration and learning purposes only.**
> It is not meant for production use. Review and harden all scripts, configurations,
> and IAM permissions before using in any production or sensitive environment.

## What this is

This repository holds a benchmark harness and a skill that reads what the harness measures.

1. **The benchmark harness** drives real coding tasks from real repositories through a coding agent, across models and hosting paths, and an independent judge scores each result. It produces a **cost/quality Pareto frontier for your code**: the models where nothing else scores higher for less money.
2. **[`swe-router`](vend/swe-router/README.md)** installs into a developer's coding assistant. Before a substantial task it reads that frontier and names the cheapest model that clears the bar for the work in front of them. It recommends a model and stops. The developer switches.

A platform team runs the benchmark on a schedule, and every developer's assistant reads the result before each task.

## Step 1 — Benchmark, and get a frontier

One command per model. The `/benchmark` skill runs the pre-flight checks, the harness over a dataset, and the judge:

```
/benchmark provider=bedrock model=claude-opus-5 dataset=dataset/mcp-gateway-registry-v2.yaml agent=omp
```

`agent` names the coding agent that drives the task and defaults to `claude`. The same flow runs headless from [`run-e2e-benchmark.sh`](benchmarks/scripts/run-e2e-benchmark.sh):

```bash
./benchmarks/scripts/run-e2e-benchmark.sh \
  --provider bedrock \
  --model claude-opus-5 \
  --dataset benchmarks/dataset/mcp-gateway-registry-v2.yaml \
  --agent omp \
  --skill swe3
```

Repeat across your model list. The harness writes per-task `metrics.json` and `eval.json` under `benchmarks/swe-benchmark-data/`; the generators then plot the cost/quality frontier for your repo and model set.

## Step 2 — Developers install the skill

Five files copied into a skills directory. The skill imports nothing and needs no build step. Run this from the root of the repository you want it in, and it lands in `.claude/skills/swe-router`:

```bash
curl -sL https://raw.githubusercontent.com/aws-samples/sample-agentic-coding-harness-benchmarks/main/vend/swe-router/install.sh | bash
```

To install it once for every repository instead, send it to your home directory:

```bash
curl -sL https://raw.githubusercontent.com/aws-samples/sample-agentic-coding-harness-benchmarks/main/vend/swe-router/install.sh | bash -s -- --dir ~/.claude/skills
```

Re-run either command to upgrade; your edited `allowed-models.txt` is kept.

`swe-router` engages on its own before a substantial task. It sets a quality floor from what happens if the change is wrong, classifies how hard the task is, and takes the cheapest model that clears that floor at that tier. **Edit `allowed-models.txt`.** The skill treats every name in it as a model the developer can select, so listing one your team cannot reach costs them the cheaper option.

Install notes and the file-by-file breakdown: **[vend/swe-router/README.md](vend/swe-router/README.md)**.

---

## Behind the two steps

### Why measure it yourself

A vendor's benchmark reports how their model does on their tasks. A public leaderboard may already be saturated, because models get tuned toward well-known test sets. Neither one tells you what a model costs to run *your* code, and that is the number a budget answers to.

**[Why this exists](docs/why-this-exists.md)**

### The three hosting paths

Anthropic models direct on Bedrock, open-weight models on Bedrock through a LiteLLM proxy, or a model you self-host on an EC2 GPU node with vLLM. All three run the same agent, tasks, skill and scoring. Only the place the model runs changes.

| Path | Models | Cost model | Best for |
|---|---|---|---|
| Bedrock (Anthropic) | Claude Opus, Sonnet, Haiku | Pay-per-token | Native Anthropic models, prompt caching |
| Bedrock (LiteLLM proxy) | 38 third-party models (Qwen, DeepSeek, Kimi, Mistral, GLM, Gemma…) | Pay-per-token | Model variety, zero infrastructure |
| Self-hosted vLLM on EC2 | Any open-weight model | Fixed hourly GPU cost | Data sovereignty, high-volume, air-gapped |

**[The three hosting paths](docs/hosting-paths.md)**

### What a single benchmark run does

Clone the repo at a pinned ref, drive the agent through the task, record tokens, latency and turns, score the artifacts against a rubric, then discard the clone.

**[What a single run does](docs/how-a-run-works.md)** · [Harness reference](benchmarks/docs/harness-reference.md)

### Benchmark your own repositories

The harness works against any GitHub repository. Write a dataset YAML naming your own repos and pinned refs, run it, and the same generators build your frontier. Git ignores your run artifacts, so private code never lands in version control.

**[Benchmark your own repositories](docs/benchmark-your-own-repo.md)**

### Getting started

On a fresh box, the `/setup-machine` skill inspects the instance, reports every missing dependency with the reason it needs it, and installs them. It adds the GPU stack only when the box has a GPU.

**[Getting started](docs/getting-started.md)** · [Repository structure](docs/repository-structure.md)

---

## Documentation map

| Document | What it covers |
|---|---|
| [docs/why-this-exists.md](docs/why-this-exists.md) | Why measure harness × model on your own repositories, and what each benchmark measures. |
| [docs/hosting-paths.md](docs/hosting-paths.md) | The three hosting paths, what each is best for, and the proxy that makes path 2 work. |
| [docs/how-a-run-works.md](docs/how-a-run-works.md) | One benchmark run end to end: clone, drive the agent, record metrics, score artifacts. |
| [docs/benchmark-your-own-repo.md](docs/benchmark-your-own-repo.md) | Dataset format and steps to build a frontier on your own code. |
| [docs/getting-started.md](docs/getting-started.md) | Prerequisites and setup sequence, from a fresh box to a first benchmark run. |
| [docs/repository-structure.md](docs/repository-structure.md) | What lives where in this repository. |
| [docs/cost-per-task-methodology.md](docs/cost-per-task-methodology.md) | How cost numbers are derived: the two cost lenses, prompt-caching accounting, and why agentic coding is prefill-bound. |
| [docs/serving-optimization-notes.md](docs/serving-optimization-notes.md) | Portable vLLM serving defaults and why we do not tune prefill knobs per model. |
| [docs/gpu-selection-h200-vs-l40s.md](docs/gpu-selection-h200-vs-l40s.md) | Which GPU to serve on: H200 slice vs L40S, same model and config. |
| [docs/vision.md](docs/vision.md) | The north star: a cost-aware harness that routes each task to the right model on the frontier automatically. |
| [benchmarks/README.md](benchmarks/README.md) | The benchmark harness landing page: the three hosting paths, how a run works, datasets. |
| [benchmarks/docs/harness-reference.md](benchmarks/docs/harness-reference.md) | Full harness reference: config, the `/swe2`/`/swe3` flow, context-window handling, and LLM-as-judge scoring. |
| [benchmarks/docs/path-anthropic-on-bedrock.md](benchmarks/docs/path-anthropic-on-bedrock.md) | Path 1 setup: benchmarking Anthropic models directly on Amazon Bedrock. |
| [benchmarks/docs/path-open-weight-on-bedrock-litellm.md](benchmarks/docs/path-open-weight-on-bedrock-litellm.md) | Path 2 setup: open-weight models on Amazon Bedrock through the LiteLLM proxy. |
| [benchmarks/docs/path-self-hosted-vllm.md](benchmarks/docs/path-self-hosted-vllm.md) | Path 3 setup: self-hosting a model on vLLM and pointing the harness at it. |
| [benchmarks/docs/end-to-end-self-hosted-run.md](benchmarks/docs/end-to-end-self-hosted-run.md) | Full run-book for an end-to-end self-hosted benchmark. |
| [self-hosted/vllm/README.md](self-hosted/vllm/README.md) | Standing up a vLLM server: install, tensor parallelism, tool-call parsers, serving-config reference. |
| [self-hosted/vllm/models/](self-hosted/vllm/models/) | Per-model serving guides for every supported model (HF repo, context window, TP size, tool parser, hardware fit). |
| [docs/omp-setup.md](docs/omp-setup.md) | Setting up the `omp` coding agent harness. |
| [docs/kiro-cli-setup.md](docs/kiro-cli-setup.md) | The kiro-cli harness: install, sign-in, headless use, and Bedrock-managed constraint. |
| [.claude/skills/setup-machine/SKILL.md](.claude/skills/setup-machine/SKILL.md) | **Start here on a new machine.** What `/setup-machine` installs and why. |
| [.claude/skills/swe-router/SKILL.md](.claude/skills/swe-router/SKILL.md) | The `/swe-router` skill: how it picks the cheapest model that clears the quality bar. |
| [docs/release-notes/](docs/release-notes/) | Release notes per version. |
| [CONTRIBUTING.md](CONTRIBUTING.md) / [SECURITY.md](SECURITY.md) / [SUPPORT.md](SUPPORT.md) | How to contribute, report a vulnerability, and get help. |

## See also

- [Claude Code docs](https://docs.anthropic.com/en/docs/claude-code) — official Claude Code documentation
- [benchmarks/README.md](benchmarks/README.md) — harness landing page
- [self-hosted/vllm/README.md](self-hosted/vllm/README.md) — standing up a self-hosted vLLM server (Path 3)

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
