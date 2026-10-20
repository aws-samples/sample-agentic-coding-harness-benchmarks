# SWE Router

A skill that tells you which model to switch your coding assistant to, based on measured scores and cost rather than vendor claims.

It recommends and stops. It does not run your task, change a setting, or write code.

> **Model names in this copy are indicative placeholders, not the models that were run.** Closed-source models are named by tier and generation (`closed-frontier-gen5`, `closed-mid-gen6`), open-weight models by parameter count and dense/MoE architecture (`open-moe-743b-a39b`, `open-dense-27b`). Generation numbers ascend with recency and are relative to this file, not any vendor's versioning. Every score, cost, completion count and frontier flag is the measured value from the run — only the names are substituted. Because nothing here names a model you can actually select, **run the benchmark on your own repository to get results under real model names**; see [Regenerating the data](#regenerating-the-data).

## Why it exists

Most developers pick the strongest model and leave it there. That is a defensible default and an expensive one. On the benchmark behind this skill, `closed-mid-gen5` lands within 5 points of `closed-frontier-gen5` on **10 of 21 tasks** at **61% lower cost per task**, and matches or beats it outright on 5 of them.

Knowing which tasks those are needs measurement. This skill carries the measurements.

## Install

```bash
curl -sL https://raw.githubusercontent.com/aws-samples/sample-agentic-coding-harness-benchmarks/main/vend/swe-router/install.sh | bash
```

That writes the skill to `.claude/skills/swe-router` in the current directory. To install it once for every repository you work in, point it at your home directory instead:

```bash
curl -sL https://raw.githubusercontent.com/aws-samples/sample-agentic-coding-harness-benchmarks/main/vend/swe-router/install.sh | bash -s -- --dir ~/.claude/skills
```

`--ref` pins a tag or commit, and `--force` replaces an `allowed-models.txt` or `routing-policy.json` you have already edited. Pinning is also the supply-chain answer: `--ref <commit-sha>` installs a fixed revision you can read on GitHub first, rather than whatever `main` holds on the day you run it. Run `install.sh --help` for the rest. Re-running the installer upgrades the skill and keeps your edited `allowed-models.txt` and `routing-policy.json`, so an upgrade never silently replaces your policy.

If you would rather read a script than pipe it into a shell, download it first:

```bash
curl -sL -O https://raw.githubusercontent.com/aws-samples/sample-agentic-coding-harness-benchmarks/main/vend/swe-router/install.sh
less install.sh && bash install.sh
```

Seven files land in the skill directory:

| File | |
|---|---|
| `SKILL.md` | the skill itself |
| `rubric.md` | the six questions the skill answers about a task |
| `route.py` | the selection, as code. Standard library only, no install step |
| `routing-policy.json` | how answers become a bucket and a required quality. **Yours to edit.** |
| `models.json` | the measurements: 23 models, scores and cost per bucket |
| `model-aliases.json` | model names as different assistants spell them |
| `allowed-models.txt` | which models your organisation permits. **Edit this one.** |

**It needs `python3`, 3.8 or later, on PATH, and no Python packages.** `route.py` uses only the standard library. The installer checks for it before downloading anything and stops with instructions if it is missing; `route.py` itself reports a too-old Python as an error rather than a traceback.

The path differs by assistant, which is what `--dir` is for. The skill has no other dependencies and imports nothing outside the standard library, so a directory of files is the whole install, and copying that directory by hand works just as well as the installer.

### Restricting it to models your organisation allows

`allowed-models.txt` ships with the skill, allowing the five models on the measured frontier. **Edit it.** As written it is a starting point, not a policy: one of those five, `open-dense-27b`, is self-hosted, and if you do not serve it the skill will recommend a model nobody can select. The file says so at the top and lists what to add.

The file is also the candidate set, not just a filter. The skill treats every entry as a model the developer can actually select, and does not inspect the agent's model picker or provider config to check — a model reached through a proxy or driven by a second agent is invisible to that kind of check, so second-guessing the list mostly costs developers the cheaper option. That puts the whole promise on whoever maintains the file: **list a model and a developer can select it; if they cannot reach it, it does not belong on the list.**

Put your own copy beside the skill, at your repository root, or in `.claude/`; the one nearest the developer's repository wins. The format is one model per line, with `#` starting a comment — so a model you are not ready to enable is commented out rather than described as forbidden somewhere a parser might read it.

The list is a hard constraint. The skill intersects it with what the benchmark measured **before** it ranks anything, so it never recommends a model you are not permitted to use. When the intersection comes out empty it says which test emptied it — nothing allowed was measured, or nothing measured is allowed — and tells the developer to stay put rather than inventing a recommendation.

Delete the file and every model is permitted.

It fires on its own before a substantial coding task — a feature, a bug fix, a refactor, a migration — and stays quiet for typos, questions, and work already under way. You can also ask it directly: *"which model should I use for this?"*

If it is firing when you do not want it, narrow the `Run it BEFORE…` sentence in `SKILL.md`; that one sentence is the whole trigger.

## What the numbers mean

`models.json` holds 23 models measured on **21 software-engineering tasks** drawn from real closed issues in one open-source repository. Each task asks the model to take a problem from description to a working patch, producing six artifacts: a GitHub issue spec, a low-level design, an expert review, a testing plan, the patch, and an implementation summary. An LLM judge with access to the repository scores each artifact.

| Field | Meaning |
|---|---|
| `score` | Mean 0-100 across the tasks the model completed. Context; the skill selects on `score_by_difficulty` |
| `cost_per_task_usd` | Mean cost of one task |
| `hosting` | `Bedrock` or `self-hosted`. Reported, never used to rank |
| `tasks_completed` / `tasks_total` | Three models did not finish all 21 |
| `on_combined_frontier` | Nothing beats it on both axes across all 23 models. Context, not a selection key: it comes from overall means and can disagree with the per-bucket ranking |
| `on_hosting_frontier` | The same, computed within one hosting basis, which is the apples-to-apples version |
| `score_by_difficulty` | Mean score per difficulty bucket: the number the required quality is compared against |
| `completion_by_difficulty` | How many tasks it finished per bucket, where failure shows |
| `score_by_complexity` / `completion_by_complexity` | The same per hand-set dataset tier, for the legacy `--tier` path |

### Read these caveats before you trust a number

**The names are placeholders.** As the note at the top says, no `model` value in `models.json` identifies a real model: closed-source entries carry a tier and a generation, open-weight entries a parameter count and an architecture. That is enough to tell you the class of model a recommendation points at, and the hardware an open-weight one would need, and it is not enough to switch to. The numbers beside each name are the measured ones. Run the benchmark on your own repository to get the same table under real model names.

**One repository.** Every task comes from a Python/FastAPI and React service with nginx, Terraform, Helm and bash around it. Rankings travel better than absolute scores. If you write Rust game engines, treat the ordering as a starting point and the numbers as indicative.

**One run per model per task, 2 to 10 per bucket.** Small differences are noise. In the earlier per-tier analysis, dropping a single task from a tier of five or six reversed the order of two models 82% of the time when they sat within 1 point of each other, 53% within 2, and 47% within 3, but only 5% past 5 points and never past 8. The skill treats anything under 3 points as a tie and takes the cheaper model. The moderate bucket holds only 2 tasks; read its column as a rough guide.

**Self-hosted costs are measured under load.** Hosted figures come from a metered bill. Self-hosted figures come from the server's hourly price divided by throughput measured at a stated concurrency — a platform team serving a group of developers, not one person with an idle GPU. Both are a cost per task, and the skill ranks them together.

**These numbers move.** A fix to token accounting once changed `closed-frontier-gen5` from $7.63 to $11.95 per task while every score stayed identical. Check `provenance.measured_on`, the date of the newest run in the file, and refetch if it looks old. `provenance.source_sha256` identifies the exact data: it is the first 12 hex digits of the SHA-256 of the frontier file it was built from, so two copies with the same value hold the same numbers.

## How it decides

The skill asks your assistant's model six questions about the task, from [`rubric.md`](rubric.md), and `route.py` turns the answers into a model. The rubric file carries each question, its scoring guide and the arithmetic; the walk-through below follows the same order.

```
   the task + the repository (AGENTS.md / CLAUDE.md / README.md first)
                     │
                     ▼
        six rubric answers, 0-2 each, with evidence
                     │
       ┌─────────────┴──────────────┐
       ▼                            ▼
  hidden_contract              verification
  silent_invariant             blast_radius
  coordinated_surfaces              │
  ambiguity                         ▼
       │                   REQUIRED QUALITY
       ▼                   60 / 70 / 75 by blast radius,
  sum 0-8 ──▶ BUCKET       +5 when nothing would catch a mistake
  easy / moderate /                 │
  hard / very hard                  │
       │                            │
       ▼                            │
  each model's mean score           │
  on benchmark tasks in ◀── compare ┘
  that bucket (models.json)
       │
       ▼
  allowed-models.txt ── keep permitted, measured models
       │
       ▼
  skip any that failed a benchmark task in the bucket
       │
       ▼
  cheapest at or above the required quality
  (gaps under 3 points count as ties)
```

A **bucket** is the task's difficulty level. The four difficulty answers add up to a sum from 0 to 8: 0-1 is easy, 2-3 moderate, 4-5 hard, 6-8 very hard. The two consequence answers set the **required quality**, the benchmark score a model must reach. Your platform team owns both mappings in [`routing-policy.json`](routing-policy.json).

### Why two separate numbers

How hard a task is and how bad a mistake would be are different questions. A one-line change to a password check is easy and dangerous. A twenty-file refactor of an internal dashboard is hard and harmless. Difficulty picks which column of measured scores to read; consequence sets the bar that column has to clear. Mixing them is how a team ends up paying for the top model to write a docs page.

### Why difficulty picks the column

Models lose points at different rates as tasks get harder, and some start failing them. On the benchmark, `closed-efficient-gen6` scores 83.2 on moderate tasks and 65.3 on very hard ones, with one very hard task unfinished. `closed-mid-gen6` drops from 86.5 to 77.2 and finishes all of them. An overall mean hides both facts.

Each benchmark task got its bucket from the same six questions, answered three times by `closed-frontier-gen6`, so "hard" means the same thing in the column and in your request. The rubric scores live in the benchmark repository, in `docs/metrics/swe-router-rubric-<dataset>-<judge>.json`; the skill never reads them, only the columns built from them.

### Why a failure skips a model

A column mean leaves out the tasks a model failed, so a model that finished 8 of 9 very hard tasks looks as good as one that finished 9 of 9. The policy's `min_completion_in_bucket` (default 1.0) skips any model that failed a benchmark task in the task's bucket. `route.py` names what it skipped.

### The columns

Every measured model, cheapest first. A fraction marks a bucket where the model failed some tasks; with the default policy it is skipped there. The benchmark has no task in the easy bucket, so an easy task reads the moderate column, which can only understate a model.

| Model | $/task | moderate (2 tasks) | hard (10) | very hard (9) |
|---|---:|---:|---:|---:|
| `closed-efficient-gen6` | 0.14 | 83.2 | 76.5 | 65.3 (8/9) |
| `open-moe-35b-a3b` | 0.26 | 62.6 | 62.3 | 55.1 |
| `open-dense-2.5b` | 0.40 | 42.6 | 47.2 (9/10) | 37.4 (8/9) |
| `open-moe-30b-a3b-code` | 0.47 | 52.4 | 45.4 | 37.2 |
| `open-moe-230b-a10b` | 0.48 | 58.9 | 58.2 | 52.7 |
| `closed-efficient-gen1` | 0.76 | 62.7 | 57.0 | 53.8 |
| `open-dense-123b` | 0.79 | 65.7 | 55.2 (6/10) | 44.5 (8/9) |
| `open-dense-31b` | 0.87 | 68.6 | 63.0 | 54.2 |
| `open-dense-27b` | 1.47 | 83.6 | 81.3 | 73.6 (8/9) |
| `open-moe-685b-a37b` | 2.15 | 66.7 | 64.9 | 56.9 |
| `open-moe-1.06t-code` | 2.47 | 79.9 | 71.9 | 62.1 (7/9) |
| `closed-mid-gen6` | 2.62 | 86.5 | 80.1 | 77.2 |
| `open-moe-430b-vl` | 3.51 | 74.3 | 60.9 | 57.0 |
| `open-moe-2.8t-a104b-hosted` | 3.87 | 83.6 | 73.1 | 70.2 |
| `closed-frontier-gen1` | 4.18 | 68.0 | 70.9 | 60.9 |
| `open-moe-2.8t-a104b` | 4.33 | 84.8 | 81.0 | 75.9 |
| `closed-frontier-gen6` | 4.44 | 89.7 | 80.4 | 80.2 |
| `closed-mid-gen5` | 4.67 | 81.8 | 78.1 | 74.6 |
| `closed-frontier-gen2` | 4.95 | 78.8 | 73.8 | 65.3 |
| `closed-frontier-gen4` | 5.32 | 81.1 | 74.9 | 73.0 |
| `open-moe-743b-a39b` | 7.04 | 86.8 | 81.0 | 77.7 |
| `closed-frontier-gen3` | 7.35 | 81.0 | 76.5 | 73.4 |
| `closed-frontier-gen5` | 11.95 | 87.4 | 84.3 | 80.2 |

### Where the numbers stop

The hardest tasks measured are bounded changes inside one repository: a rate-limiting subsystem, server-side OAuth token storage. Nothing here is a rewrite, a language port, or a change spanning several services. Asked about work beyond that range, the skill recommends the strongest permitted model and says the measurements do not reach that far.

## Running the selection directly

`route.py` is the whole decision procedure, and it works without the skill:

```bash
python3 route.py --confidence high --rubric '{"hidden_contract": 1, "silent_invariant": 0, "coordinated_surfaces": 2, "ambiguity": 1, "verification": 1, "blast_radius": 1}'
```

It prints JSON with the bucket, the required quality, the pick, the runners-up cheapest first, how far above the bar the pick sits and whether that beats the noise, and what was excluded or skipped. `--confidence low` returns `status: unsure` and keeps the developer on their model. `--policy` points at another policy file, `--allowed-file` at another allow-list, and `--no-allow-list` ignores it.

The old interface, `python3 route.py --tier high --floor 70`, still works for one release. It reads the dataset's hand-set tiers (`score_by_complexity`) and does not skip models for failures.

### What it delivers, measured

Replaying the router over the 21 benchmark tasks, each routed from columns that leave that task out, then looking up the chosen model's **actual score and cost on that task**, against running `closed-frontier-gen5` on everything ($251.04 for the 21 tasks, mean score 82.83):

| Router | Cost saved | Quality change | Tasks below the required quality |
|---|---:|---:|---:|
| Previous skill (floor and tier judged in prose) | 70.5% | −5.06 | 9 |
| Rubric | 75.3% | −3.75 | 4 |
| **Rubric with the failure rule (this version)** | **74.7%** | **−3.22** | **4** |

On a second repository, cli/cli in Go, columns built from this benchmark alone routed the tasks with no adjustment and erred towards the stronger model. The failure rule cannot foresee a failure the benchmark never recorded: `closed-efficient-gen6` finished all 10 hard tasks here and then failed a hard cli/cli task by writing its output to the wrong folder.

These come from replaying the router against its own evidence with [`scripts/eval_swe_router.py`](../../benchmarks/scripts/eval_swe_router.py) in the benchmark repository: for whatever the skill picks, every model has already run every task, so the score and cost it would have got are a lookup rather than a new run.

## Regenerating the data

Inside the benchmark repository:

```bash
cd benchmarks
uv run scripts/build_vended_models.py           # rewrite models.json
uv run scripts/build_vended_models.py --check   # CI: fail if out of date
```

The generator emits **real** model names, which is what you want: run it against your own measurements and `models.json` names models your developers can select. The copy committed here is the one exception — its names were replaced with placeholders after generation, so regenerating it in this repository replaces the placeholders with whatever your run measured.

Scores and costs come from `docs/metrics/pareto-frontier-omp-swe3.json` and the run summaries behind it. Each benchmark task's bucket comes from the dataset's rubric pass, `docs/metrics/swe-router-rubric-<dataset>-<judge>.json`, recomputed from its stored answers with `routing-policy.json`. A new model needs only a rebuild; a new dataset, or a change to `rubric.md`, needs a new rubric pass first:

```bash
uv run scripts/run-swe-router-headless.py --mode rubric --agent omp --provider bedrock \
    --model <your-judge-model-id> --dataset dataset/<dataset>.yaml --repeats 3
```

## Source

Measurements, harness and methodology: [aws-samples/sample-agentic-coding-harness-benchmarks](https://github.com/aws-samples/sample-agentic-coding-harness-benchmarks).

Whether these measurements carry over to a different repository is still open (issue #221): every task behind them comes from one Python/FastAPI and React service. Rankings travel further than absolute scores, so re-measure on your own code before trusting the numbers rather than the shape.
