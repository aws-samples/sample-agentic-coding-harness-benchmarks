<h1 align="center">Agentic Coding Harness and Benchmarks</h1>

<p align="center">
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT--0-yellow.svg" alt="License: MIT-0"></a>
<a href="https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html"><img src="https://img.shields.io/badge/Amazon-Bedrock-blue" alt="Bedrock"></a>
<a href="./"><img src="https://img.shields.io/badge/Models-45%20from%2011%20providers-orange" alt="Models: 45"></a>
</p>

<p align="center">
<a href="https://github.com/aarora79/agentic-coding-harness-benchmarks">GitHub Repo</a> |
<a href="docs/benchmark-your-own-repo.md">Benchmark your own repo</a> |
<a href="docs/results-swe3.md">Results (/swe3)</a> |
<a href="docs/agentic-coding-swe-comparison-swe3.md">Harness Comparison</a> |
<a href="docs/slides/agentic-coding-benchmarks-presentation.pdf">Slide Deck</a> |
<a href="docs/slides/agentic-coding-benchmarks-exec.pdf">Executive Brief</a> |
<a href="docs/cost-per-task-methodology.md">Cost Methodology</a>
</p>

<p align="center">
<video src="https://github.com/user-attachments/assets/84f60f0f-cd7d-4513-b90b-37a0ffb5feed" controls width="820">
Your browser cannot play this video inline. <a href="https://github.com/user-attachments/assets/84f60f0f-cd7d-4513-b90b-37a0ffb5feed">Download the swe-router explainer</a> or open the <a href="docs/slides/agentic-coding-benchmarks-presentation.pdf">slide deck</a>.
</video>
</p>

<p align="center"><em>swe-router in about 70 seconds: the right model, handed to the developer for each task, while the platform team benchmarks once and spends less.</em></p>

> **This is sample code intended for demonstration and learning purposes only.**
> It is not meant for production use. Review and harden all scripts, configurations,
> and IAM permissions before using in any production or sensitive environment.

> [!WARNING]
> **These results are still moving -- expect the numbers to change for a few days.**
> We re-run benchmarks most hours and keep learning how
> different **coding-harness x model** combinations behave: their real token usage,
> caching, cost, and how each model drives (or under-drives) a long-horizon agentic
> task. As we learn, methodology improves (counting subagent tokens, switching
> the default skill to a single-agent variant) and figures get re-measured. Treat
> the current numbers as a **live snapshot**, not final -- they should settle over
> the coming days.

## What this is

This repository holds a benchmark harness and a skill that reads what the harness measures.

1. **The benchmark harness** drives real coding tasks from real repositories through a coding agent, across models and hosting paths, and an independent judge scores each result. It produces a **cost/quality Pareto frontier for your code**: the models where nothing else scores higher for less money.
2. **[`swe-router`](vend/swe-router/README.md)** installs into a developer's coding assistant. Before a substantial task it reads that frontier and names the cheapest model that clears the bar for the work in front of them. It recommends a model and stops. The developer switches.

A platform team runs the benchmark on a schedule, and every developer's assistant reads the result before each task. The diagram below is also a [slide-ready HTML version](docs/slides/swe-router-workflow.html) and an [ASCII version](docs/diagram-ascii.md).

![Measure once, spend less on every task: a platform team benchmarks its approved models on its own repo to build a frontier, which ships as the swe-router skill that every developer's coding assistant reads before each task](docs/images/measure-once-spend-less-every-week.png)

## Step 1 — Benchmark, and get a frontier

One command per model. The `/benchmark` skill runs the pre-flight checks, the harness over a dataset, and the judge:

```
/benchmark provider=bedrock model=claude-opus-5 dataset=dataset/mcp-gateway-registry-v2.yaml agent=omp
```

`agent` names the coding agent that drives the task and defaults to `claude`. The run charted below used `omp` on the `/swe3` skill, which is the default skill. The same flow runs headless from [`run-e2e-benchmark.sh`](benchmarks/scripts/run-e2e-benchmark.sh) (`--provider bedrock|litellm|vllm --model ... --dataset ... --agent claude|pi|omp|kiro --skill swe2|swe3`). Repeat across your model list, then the generators plot the frontier.

![Cost vs. quality Pareto frontier, omp harness on /swe3](docs/images/cost-quality-omp-swe3.png)

On ours, 18 models over 21 tasks: `claude-opus-5` scores highest at **82.83** for **$11.95** a task, `glm-5.3` lands 2.7 points behind it at $7.04, and `kimi-k3` a further point back but cheaper at **79.17 for $4.33**. The cheapest model above 78 is `qwen3.8-27b`, which scores **78.48 at $1.47** and beats `claude-sonnet-5` (76.97, $4.67) at under a third of the cost.

## Step 2 — Developers install the skill

Five files copied into a skills directory. The skill imports nothing and needs no build step. Run this from the root of the repository you want it in, and it lands in `.claude/skills/swe-router`:

```bash
curl -sL https://raw.githubusercontent.com/aarora79/agentic-coding-harness-benchmarks/main/vend/swe-router/install.sh | bash
```

To install it once for every repository instead, send it to your home directory:

```bash
curl -sL https://raw.githubusercontent.com/aarora79/agentic-coding-harness-benchmarks/main/vend/swe-router/install.sh | bash -s -- --dir ~/.claude/skills
```

Re-run either command to upgrade; your edited `allowed-models.txt` is kept.

`swe-router` engages on its own before a substantial task. It sets a quality floor from what happens if the change is wrong, classifies how hard the task is, and takes the cheapest model that clears that floor at that tier. **Edit `allowed-models.txt`.** The skill treats every name in it as a model the developer can select, so listing one your team cannot reach costs them the cheaper option.

Install notes, the file-by-file breakdown and the measured results: **[vend/swe-router/README.md](vend/swe-router/README.md)**.

---

## Behind the two steps

### Results

18 models over 21 tasks. Each task comes from a closed issue in a real repository, pinned to the release before the fix shipped, so the defect is present in the tree the agent clones. An independent judge scores every run 0-100, and the tasks split across four complexity tiers. Earlier runs on other harnesses and datasets stay published as background. They use different task sets, so their scores do not merge into the headline table.

**[All results](docs/results-index.md)** · [Headline run (omp, /swe3, v2)](docs/harness-omp-swe3.md) · [Which model for which task?](docs/model-selection-by-complexity.md) · [Cost per task methodology](docs/cost-per-task-methodology.md) · [v1 dataset](docs/results-swe3.md) · [/swe2 multi-agent](docs/results-swe2.md) · [Cross-harness comparison](docs/agentic-coding-swe-comparison-swe3.md)

### Does routing pay for itself?

We ran `swe-router` against its own evidence. Every model has run every task, so for whatever the skill picks we can look up what that model scored and cost on that task. Over 21 tasks it came out **45.4% cheaper for 4.6% less quality**. Three caveats decide how much to trust that number, and all three come from the judgment step, not the arithmetic.

**[Does routing pay for itself?](docs/swe-router-evaluation.md)** · [The routing result, joined to the measured runs](docs/swe-router-evaluation-judged.md) · [What the model judged each task to need](docs/swe-router-judged-inputs.md) · [The skill itself](.claude/skills/swe-router/SKILL.md)

### Why measure it yourself

A vendor's benchmark reports how their model does on their tasks. A public leaderboard may already be saturated, because models get tuned toward well-known test sets. Neither one tells you what a model costs to run *your* code, and that is the number a budget answers to.

**[Why this exists](docs/why-this-exists.md)**

### The three hosting paths

Anthropic models direct on Bedrock, open-weight models on Bedrock through a LiteLLM proxy, or a model you self-host on an EC2 GPU node with vLLM. All three run the same agent, tasks, skill and scoring. Only the place the model runs changes.

**[The three hosting paths](docs/hosting-paths.md)**

### What a single benchmark run does

Clone the repo at a pinned ref, drive the agent through the task, record tokens, latency and turns, score the six artifacts against a rubric, then discard the clone.

**[What a single run does](docs/how-a-run-works.md)** · [Harness reference](benchmarks/docs/harness-reference.md)

### Benchmark your own repositories

The harness works against any GitHub repository. Write a dataset YAML naming your own repos and pinned refs, run it, and the same generators build your frontier. Git ignores your run artifacts, so private code never lands in version control.

**[Benchmark your own repositories](docs/benchmark-your-own-repo.md)**

### Getting started

On a fresh box, the `/setup-machine` skill inspects the instance, reports every missing dependency with the reason it needs it, and installs them. It adds the GPU stack only when the box has a GPU.

**[Getting started](docs/getting-started.md)** · [Repository structure](docs/repository-structure.md)

## Documentation map

Where to read more, by topic:

| Document | What it covers |
|----------|----------------|
| [docs/why-this-exists.md](docs/why-this-exists.md) | Why measure harness x model on your own repositories, what the two benchmarks (quality and throughput) each measure, and why they only mean something combined. |
| [docs/results-index.md](docs/results-index.md) | Every published run: the headline table, the per-(harness, skill) documents, and why results from different datasets must not be merged. |
| [docs/swe-router-evaluation.md](docs/swe-router-evaluation.md) | Whether routing pays for itself, run against the benchmark's own measurements -- the cost/quality trade and the three caveats in the judgment step. |
| [docs/hosting-paths.md](docs/hosting-paths.md) | The three places a model can run (Bedrock native, Bedrock via LiteLLM, self-hosted vLLM), what each is best for, and the proxy that makes path 2 work. |
| [docs/how-a-run-works.md](docs/how-a-run-works.md) | One benchmark run end to end: clone at a pinned ref, drive the agent, record metrics, score six artifacts. |
| [docs/benchmark-your-own-repo.md](docs/benchmark-your-own-repo.md) | The dataset format and the steps to build a frontier on your own code, with tips for writing tasks that produce comparable runs. |
| [docs/getting-started.md](docs/getting-started.md) | Prerequisites and the setup sequence, from a fresh box to a first benchmark run. |
| [docs/repository-structure.md](docs/repository-structure.md) | What lives where in this repository. |
| [.claude/skills/setup-machine/SKILL.md](.claude/skills/setup-machine/SKILL.md) | **Start here on a new machine.** What `/setup-machine` inspects and installs, why each component is needed, where the vLLM venv lands on a small root disk, and what it deliberately does not do. |
| [docs/results-swe3.md](docs/results-swe3.md) / [docs/results-swe2.md](docs/results-swe2.md) | Full benchmark results per skill: task-by-task tables, per-model leaderboard, cost/quality frontier, hardware, and what the data says. |
| [.claude/skills/swe-router/SKILL.md](.claude/skills/swe-router/SKILL.md) | The `/swe-router` skill: how it sets a quality floor from the consequence of a change being wrong, picks the tier table to read it against, and selects the cheapest model that clears it. Advisory -- it recommends and stops. |
| [docs/swe-router-judged-inputs.md](docs/swe-router-judged-inputs.md) | What `omp` + `claude-opus-5` judged each of the 21 v2 tasks to need: floor, tier, the reasoning, and how much those moved across three identical runs. |
| [docs/swe-router-evaluation-judged.md](docs/swe-router-evaluation-judged.md) | Whether routing on those judgments beats running one model on everything: per-task picks joined to the measured runs, with cost and quality deltas. |
| [docs/vision.md](docs/vision.md) | The north star: a cost-aware harness that routes each task (and each phase) to the right model on the frontier -- frontier / workhorse / budget -- switching automatically. |
| [benchmarks/README.md](benchmarks/README.md) | The benchmark harness landing page: the three hosting paths, how a run works, and how to reproduce the results above. |
| [benchmarks/docs/harness-reference.md](benchmarks/docs/harness-reference.md) | Full harness reference: config, the `/swe2` flow, context-window/auto-compaction, and the LLM-as-judge scoring. |
| [docs/agentic-coding-swe-comparison-swe3.md](docs/agentic-coding-swe-comparison-swe3.md) | Claude Code against pi on the same models and tasks: per-metric win tallies, the frontier, and model-tier buying guidance. Claude Code is a few points more accurate on some models; pi is far more token-efficient, so faster and, self-hosted, cheaper. |
| [benchmarks/docs/path-anthropic-on-bedrock.md](benchmarks/docs/path-anthropic-on-bedrock.md) | Path 1 setup: benchmarking the Anthropic family (Claude Opus/Sonnet/Haiku) directly on Amazon Bedrock. |
| [benchmarks/docs/path-open-weight-on-bedrock-litellm.md](benchmarks/docs/path-open-weight-on-bedrock-litellm.md) | Path 2 setup: open-weight models on Amazon Bedrock through the LiteLLM proxy. |
| [benchmarks/docs/path-self-hosted-vllm.md](benchmarks/docs/path-self-hosted-vllm.md) | Path 3 setup: self-hosting a model on vLLM and pointing the harness at it. |
| [docs/kiro-cli-setup.md](docs/kiro-cli-setup.md) | The kiro-cli harness: install, sign-in, headless use, the Bedrock-managed-only constraint, and how its Kiro-credit spend is calculated. Results: [harness-kiro-cli-swe3.md](docs/harness-kiro-cli-swe3.md). |
| [benchmarks/docs/end-to-end-self-hosted-run.md](benchmarks/docs/end-to-end-self-hosted-run.md) | The full manual run-book for an end-to-end self-hosted benchmark. |
| [self-hosted/vllm/README.md](self-hosted/vllm/README.md) | Standing up a vLLM server: install, tensor parallelism, tool-call parsers, and the serving-config reference. |
| [self-hosted/vllm/models/](self-hosted/vllm/models/) | Per-model serving guides (HF repo, context window, TP size, tool parser, hardware fit) for every benchmarked model. |
| [docs/agentic-coding-throughput-comparison.md](docs/agentic-coding-throughput-comparison.md) | Serving-economics comparison across models: throughput, saturation, and hardware-derived cost per token / per task. |
| [docs/gpu-selection-h200-vs-l40s.md](docs/gpu-selection-h200-vs-l40s.md) | Which GPU to serve on: one H200 slice of a p5en vs a whole g6e.4xlarge (1x L40S), same model and config. Why the H200 slice is 41% cheaper per unit of work despite costing 2.6x per hour, and what it costs to serve N developers. Public on-demand prices, no discounts. |
| [docs/cost-per-task-methodology.md](docs/cost-per-task-methodology.md) | How the cost numbers are derived: the two cost lenses, prompt-caching accounting (API vs self-hosted), and why agentic coding is prefill-bound. |
| [docs/serving-optimization-notes.md](docs/serving-optimization-notes.md) | Portable vLLM serving defaults and why we do not tune the prefill knobs per model. |
| [docs/release-notes/](docs/release-notes/) | Release notes per version, newest first, and the versioning scheme: new dataset or model is a minor, a methodology change or new functionality is a major. |
| [CONTRIBUTING.md](CONTRIBUTING.md) / [SECURITY.md](SECURITY.md) / [SUPPORT.md](SUPPORT.md) | How to contribute, report a vulnerability, and get help. |

## See also

- [Claude Code docs](https://docs.anthropic.com/en/docs/claude-code) -- official Claude Code documentation
- [benchmarks/README.md](benchmarks/README.md) -- the harness landing page
- [self-hosted/vllm/README.md](self-hosted/vllm/README.md) -- standing up a self-hosted vLLM server (Path 3)

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
