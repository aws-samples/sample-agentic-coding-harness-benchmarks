# SWE Router

A skill that tells you which model to switch your coding assistant to, based on measured scores and cost rather than vendor claims.

It recommends and stops. It does not run your task, change a setting, or write code.

## Why it exists

Most developers pick the strongest model and leave it there. That is a defensible default and an expensive one. On the benchmark behind this skill, `claude-sonnet-5` lands within 5 points of `claude-opus-5` on **10 of 21 tasks** at **61% lower cost per task**, and matches or beats it outright on 5 of them.

Knowing which tasks those are needs measurement. This skill carries the measurements.

## Install

```bash
curl -sL https://raw.githubusercontent.com/aws-samples/sample-agentic-coding-harness-benchmarks/main/vend/swe-router/install.sh | bash
```

That writes the skill to `.claude/skills/swe-router` in the current directory. To install it once for every repository you work in, point it at your home directory instead:

```bash
curl -sL https://raw.githubusercontent.com/aws-samples/sample-agentic-coding-harness-benchmarks/main/vend/swe-router/install.sh | bash -s -- --dir ~/.claude/skills
```

`--ref` pins a tag or commit, and `--force` replaces an `allowed-models.txt` you have already edited. Pinning is also the supply-chain answer: `--ref <commit-sha>` installs a fixed revision you can read on GitHub first, rather than whatever `main` holds on the day you run it. Run `install.sh --help` for the rest. Re-running the installer upgrades the skill and keeps your edited `allowed-models.txt`, so an upgrade never silently replaces your model policy.

If you would rather read a script than pipe it into a shell, download it first:

```bash
curl -sL -O https://raw.githubusercontent.com/aws-samples/sample-agentic-coding-harness-benchmarks/main/vend/swe-router/install.sh
less install.sh && bash install.sh
```

Five files land in the skill directory:

| File | |
|---|---|
| `SKILL.md` | the skill itself |
| `route.py` | the selection, as code. Standard library only — no install step |
| `models.json` | the measurements — 18 models, scores and cost per tier |
| `model-aliases.json` | model names as different assistants spell them |
| `allowed-models.txt` | which models your organisation permits. **Edit this one.** |

The path differs by assistant, which is what `--dir` is for. The skill has no dependencies and imports nothing — a directory of files is the whole install, and copying that directory by hand works just as well as the installer.

### Restricting it to models your organisation allows

`allowed-models.txt` ships with the skill, allowing the five models on the measured frontier. **Edit it.** As written it is a starting point, not a policy: four of those five are self-hosted, and if you do not serve them the skill will recommend a model nobody can select. The file says so at the top and lists what to add.

The file is also the candidate set, not just a filter. The skill treats every entry as a model the developer can actually select, and does not inspect the agent's model picker or provider config to check — a model reached through a proxy or driven by a second agent is invisible to that kind of check, so second-guessing the list mostly costs developers the cheaper option. That puts the whole promise on whoever maintains the file: **list a model and a developer can select it; if they cannot reach it, it does not belong on the list.**

Put your own copy beside the skill, at your repository root, or in `.claude/`; the one nearest the developer's repository wins. The format is one model per line, with `#` starting a comment — so a model you are not ready to enable is commented out rather than described as forbidden somewhere a parser might read it.

The list is a hard constraint. The skill intersects it with what the benchmark measured **before** it ranks anything, so it never recommends a model you are not permitted to use. When the intersection comes out empty it says which test emptied it — nothing allowed was measured, or nothing measured is allowed — and tells the developer to stay put rather than inventing a recommendation.

Delete the file and every model is permitted.

It fires on its own before a substantial coding task — a feature, a bug fix, a refactor, a migration — and stays quiet for typos, questions, and work already under way. You can also ask it directly: *"which model should I use for this?"*

If it is firing when you do not want it, narrow the `Run it BEFORE…` sentence in `SKILL.md`; that one sentence is the whole trigger.

## What the numbers mean

`models.json` holds 18 models measured on **21 software-engineering tasks** drawn from real closed issues in one open-source repository. Each task asks the model to take a problem from description to a working patch, producing six artifacts: a GitHub issue spec, a low-level design, an expert review, a testing plan, the patch, and an implementation summary. An LLM judge with access to the repository scores each artifact.

| Field | Meaning |
|---|---|
| `score` | Mean 0-100 across the tasks the model completed. Context; the skill selects on `score_by_complexity` |
| `cost_per_task_usd` | Mean cost of one task |
| `hosting` | `Bedrock` or `self-hosted`. Reported, never used to rank |
| `tasks_completed` / `tasks_total` | Three models did not finish all 21 |
| `on_combined_frontier` | Nothing beats it on both axes across all 18 models. Context, not a selection key — it comes from overall means and can disagree with the per-tier ranking |
| `on_hosting_frontier` | The same, computed within one hosting basis, which is the apples-to-apples version |
| `score_by_complexity` | Mean score per tier — the number the floor is compared against |
| `completion_by_complexity` | How many tasks it finished per tier, where failure shows |

### Read these caveats before you trust a number

**One repository.** Every task comes from a Python/FastAPI and React service with nginx, Terraform, Helm and bash around it. Rankings travel better than absolute scores. If you write Rust game engines, treat the ordering as a starting point and the numbers as indicative.

**One run per model per task, 5 or 6 per tier.** Small differences are noise. Dropping a single task from a tier reverses the order of two models 82% of the time when they sit within 1 point of each other, 53% within 2, and 47% within 3 — but only 5% past 5 points, and never past 8. The skill treats anything under 3 points at a tier as a tie and takes the cheaper model.

**Self-hosted costs are measured under load.** Hosted figures come from a metered bill. Self-hosted figures come from the server's hourly price divided by throughput measured at a stated concurrency — a platform team serving a group of developers, not one person with an idle GPU. Both are a cost per task, and the skill ranks them together.

**These numbers move.** A fix to token accounting once changed `claude-opus-5` from $7.63 to $11.95 per task while every score stayed identical. Check `provenance.measured_on` and refetch if it looks old.

## How it decides

Two questions, asked separately, because they answer different things.

**How good does this have to be?** That is about consequence. A one-line change to an auth path can ruin someone's week; a fifty-line change to a scratch script cannot. Consequence sets a **floor** — the score a model has to reach before it is worth considering.

**How hard is this?** That is about the work itself. It does not change the floor. It changes **which measured score** gets compared against the floor, because models fall off at very different rates as tasks get harder.

Keeping them apart is the whole trick. Collapsing them — treating a big task as a high-stakes one — is how you end up paying for the top model to write a docs page.

```
   AGENTS.md ──▶ read the repo's own map
   CLAUDE.md         (fall back: README.md, CONTRIBUTING.md)
       │
       ▼
   the task ─────┬──────────────────────┐
                 │                      │
                 ▼                      ▼
      what happens if           how hard is it?
      this is wrong?                    │
                 │                      │
                 ▼                      ▼
        ┌────────────────┐   one table per tier, pick the matching one
        │ prototype   55 │   ┌──────┐┌──────┐┌──────┐┌──────┐
        │ internal    65 │   │triv. ││ low  ││medium││ high │
        │ production  70 │   │      ││      ││      ││      │
        │ security    75 │   └──────┘└──────┘└──────┘└──────┘
        └────────────────┘   beyond that range ──▶ strongest available,
                 │                                  say so, stop
                 │                      │
            +5 if it turns              │  picks which table
            on one exact fact           │
                 │                      │
                 ▼                      ▼
              FLOOR  ◀── read against ── that tier's table
                              │
                              ▼
        allowed-models.txt ── the candidate set: what you
                    │           permit AND can select
          ┌─────────┴─────────┐
          ▼                   ▼
    in models.json      not measured
          │              (named, excluded)
          │
          ├── empty? say which test
          │   emptied it, then stop
          ▼
        cheapest at or above the floor
        gaps under 3 pts count as ties
                    │
         ┌──────────┼──────────────┐
         ▼          ▼              ▼
     switch to   already        nothing
      model X    on it          clears it
         │          │              │
      say what   stay put      stay put, and
      it costs                 name the shortfall
```

### Why the floor comes from consequence, not size

The natural instinct is to reach for a stronger model when a task looks big. The measurements do not support it.

On the benchmark, five tasks all classed as `trivial` — every one a single-file change — ranged from a **0.4-point** gap between the best and second-best model to a **13.8-point** gap. Three were mechanical: write a docs index, add two config passthroughs, narrow a render condition. On those, the expensive model bought nothing. Two turned on knowing one exact thing — that BSD `sed -i` swallows its next argument, that a version check has to compare rather than test for existence. On those it was worth 10 to 14 points.

Same size. Opposite answers. What separated them was whether the task hinged on getting one specific thing right, which is why that gets a +5 on the floor and size gets nothing.

### Why difficulty picks the table

Models do not degrade in parallel, so the running order changes from one table to the next. Six of the eighteen, ranked within each tier (`!` marks a tier where the model failed at least one task):

```
  trivial              low                  medium               high
  ─────────────────    ─────────────────    ─────────────────    ─────────────────
  glm-5.3    85.0      opus-5     86.2      opus-5     81.8      opus-5     79.8
  opus-5     83.8      glm-5.3    82.6      qwen3.8    78.7      glm-5.3    74.7
  qwen3.8    81.4      qwen3.8    80.9      glm-5.3    78.5      sonnet-5   73.7
  sonnet-5   76.1      sonnet-5   80.5      sonnet-5   77.5      qwen3.8    71.5 !
  kimi-k2.7  72.4      kimi-k2.7  71.7      kimi-k2.7  69.6      kimi-k2.7  58.5 !
  gemma-4    66.5      gemma-4    63.3      gemma-4    59.3      gemma-4    50.0
```

Read `qwen3.8` across the four. Third on trivial and low, second on medium, then **fourth on high, behind `sonnet-5`**, having failed one hard task. Its whole-dataset average of 78.48 beats sonnet's 76.97, which is why it sits on the published frontier and sonnet does not. On hard work that ordering is reversed.

`glm-5.3` does the opposite: it tops the `trivial` table at 85.0, above `opus-5`, then slips behind `opus-5` on every harder tier, and behind `qwen3.8` on medium.

**A recommendation built on the overall average is reading a table that does not exist.** There is no tier where those averages are the ranking.

The `!` marks carry information no score can. A failure is excluded from the mean rather than averaged in, so `qwen3.8` scoring 71.5 on `high` is its average over the four hard tasks it finished — the fifth is simply absent. `completion_by_complexity` is where that shows, and the skill reports it rather than quoting the average of the ones that worked.

How much difficulty matters depends on the pair. Between `claude-opus-5` and `claude-sonnet-5` it explains 4% of the gap; between `claude-opus-5` and `gemma-4-31b`, 43%. Two frontier models degrade together. A frontier model and a small one do not.

### The four tables

`score_by_complexity` is one table per tier, and this is what they look like. Each is sorted cheapest first, which is the order the skill reads them in: **scan down until the score clears your floor, and stop.** That first hit is the recommendation.

A bold completion count means the model failed at least one task at that tier, which no score can show — a failure is excluded from the mean rather than averaged into it.

**`trivial`** — scan down until the score clears your floor.

| $/task | Model | Score | Finished |
|---:|---|---:|:-:|
| $0.26 | `qwen3.6-35b` | 58.4 | 5/5 |
| $0.47 | `qwen3-coder-30b` | 45.6 | 5/5 |
| $0.48 | `minimax-m2.5` | 54.7 | 5/5 |
| $0.76 | `claude-haiku-4-5` | 55.4 | 5/5 |
| $0.79 | `devstral-2-123b` | 56.5 | **4/5** |
| $0.87 | `gemma-4-31b` | 66.5 | 5/5 |
| $1.47 | `qwen3.8-27b` | 81.4 | 5/5 |
| $2.15 | `deepseek-v3.2` | 64.4 | 5/5 |
| $2.47 | `kimi-k2.7-code` | 72.4 | 5/5 |
| $3.51 | `minimax-m3` | 65.7 | 5/5 |
| $4.18 | `claude-opus-4-5` | 68.8 | 5/5 |
| $4.33 | `kimi-k3` | 81.8 | 5/5 |
| $4.67 | `claude-sonnet-5` | 76.1 | 5/5 |
| $4.95 | `claude-opus-4-6-v1` | 74.2 | 5/5 |
| $5.32 | `claude-opus-4-8` | 73.8 | 5/5 |
| $7.04 | `glm-5.3` | 85.0 | 5/5 |
| $7.35 | `claude-opus-4-7` | 77.8 | 5/5 |
| $11.95 | `claude-opus-5` | 83.8 | 5/5 |

**`low`** — scan down until the score clears your floor.

| $/task | Model | Score | Finished |
|---:|---|---:|:-:|
| $0.26 | `qwen3.6-35b` | 62.5 | 5/5 |
| $0.47 | `qwen3-coder-30b` | 48.4 | 5/5 |
| $0.48 | `minimax-m2.5` | 60.2 | 5/5 |
| $0.76 | `claude-haiku-4-5` | 57.8 | 5/5 |
| $0.79 | `devstral-2-123b` | 61.3 | **3/5** |
| $0.87 | `gemma-4-31b` | 63.3 | 5/5 |
| $1.47 | `qwen3.8-27b` | 80.9 | 5/5 |
| $2.15 | `deepseek-v3.2` | 63.9 | 5/5 |
| $2.47 | `kimi-k2.7-code` | 71.7 | 5/5 |
| $3.51 | `minimax-m3` | 60.6 | 5/5 |
| $4.18 | `claude-opus-4-5` | 66.2 | 5/5 |
| $4.33 | `kimi-k3` | 84.2 | 5/5 |
| $4.67 | `claude-sonnet-5` | 80.5 | 5/5 |
| $4.95 | `claude-opus-4-6-v1` | 75.4 | 5/5 |
| $5.32 | `claude-opus-4-8` | 81.1 | 5/5 |
| $7.04 | `glm-5.3` | 82.6 | 5/5 |
| $7.35 | `claude-opus-4-7` | 77.0 | 5/5 |
| $11.95 | `claude-opus-5` | 86.2 | 5/5 |

**`medium`** — scan down until the score clears your floor.

| $/task | Model | Score | Finished |
|---:|---|---:|:-:|
| $0.26 | `qwen3.6-35b` | 59.9 | 6/6 |
| $0.47 | `qwen3-coder-30b` | 40.8 | 6/6 |
| $0.48 | `minimax-m2.5` | 57.2 | 6/6 |
| $0.76 | `claude-haiku-4-5` | 57.8 | 6/6 |
| $0.79 | `devstral-2-123b` | 51.0 | **4/6** |
| $0.87 | `gemma-4-31b` | 59.3 | 6/6 |
| $1.47 | `qwen3.8-27b` | 78.7 | 6/6 |
| $2.15 | `deepseek-v3.2` | 62.1 | 6/6 |
| $2.47 | `kimi-k2.7-code` | 69.6 | 6/6 |
| $3.51 | `minimax-m3` | 56.8 | 6/6 |
| $4.18 | `claude-opus-4-5` | 67.9 | 6/6 |
| $4.33 | `kimi-k3` | 76.8 | 6/6 |
| $4.67 | `claude-sonnet-5` | 77.5 | 6/6 |
| $4.95 | `claude-opus-4-6-v1` | 68.3 | 6/6 |
| $5.32 | `claude-opus-4-8` | 73.3 | 6/6 |
| $7.04 | `glm-5.3` | 78.5 | 6/6 |
| $7.35 | `claude-opus-4-7` | 73.6 | 6/6 |
| $11.95 | `claude-opus-5` | 81.8 | 6/6 |

**`high`** — scan down until the score clears your floor.

| $/task | Model | Score | Finished |
|---:|---|---:|:-:|
| $0.26 | `qwen3.6-35b` | 56.0 | 5/5 |
| $0.47 | `qwen3-coder-30b` | 35.8 | 5/5 |
| $0.48 | `minimax-m2.5` | 51.4 | 5/5 |
| $0.76 | `claude-haiku-4-5` | 53.4 | 5/5 |
| $0.79 | `devstral-2-123b` | 40.8 | 5/5 |
| $0.87 | `gemma-4-31b` | 50.0 | 5/5 |
| $1.47 | `qwen3.8-27b` | 71.5 | **4/5** |
| $2.15 | `deepseek-v3.2` | 56.0 | 5/5 |
| $2.47 | `kimi-k2.7-code` | 58.5 | **3/5** |
| $3.51 | `minimax-m3` | 59.7 | 5/5 |
| $4.18 | `claude-opus-4-5` | 62.0 | 5/5 |
| $4.33 | `kimi-k3` | 74.3 | 5/5 |
| $4.67 | `claude-sonnet-5` | 73.7 | 5/5 |
| $4.95 | `claude-opus-4-6-v1` | 65.1 | 5/5 |
| $5.32 | `claude-opus-4-8` | 70.8 | 5/5 |
| $7.04 | `glm-5.3` | 74.7 | 5/5 |
| $7.35 | `claude-opus-4-7` | 74.4 | 5/5 |
| $11.95 | `claude-opus-5` | 79.8 | 5/5 |

### Where the numbers stop

The hardest tasks measured are bounded changes inside one repository — a rate-limiting subsystem, server-side OAuth token storage. Nothing here is a rewrite, a language port, or a change spanning several services.

Asked about work beyond that range, the skill recommends the strongest model available and tells you the measurements do not reach that far. Extrapolating a cheap model onto a job three orders of magnitude larger than anything tested is not a saving.

## Running the selection directly

`route.py` is the whole decision procedure, and it works without the skill:

```bash
python3 route.py --tier high --floor 70
```

It prints JSON with the pick, the runners-up cheapest-first, how far above the floor the pick sits and whether that margin beats the noise, whether the model finished every task at that tier, and what policy excluded. `--allowed-file` points at another allow-list and `--no-allow-list` ignores it; `--available` narrows the set further, for the case where a developer says one permitted model is not wired up for them.

### What it delivers, measured

Running the router over the 21 benchmark tasks, using each task's real difficulty and then checking the recommended model's **actual score on that specific task**:

| Floor | $/task | Cleared the floor in reality |
|---:|---:|---|
| 55 | $0.26 | 16/21 (76%) |
| 60 | $1.04 | 17/21 (81%) |
| 65 | $1.33 | 17/21 (81%) |
| 70 | $1.47 | 18/21 (86%) |
| 75 | $3.05 | 18/21 (86%) |

`claude-opus-5` on the same 21 tasks clears a floor of 70 on **21 of 21**, at **$11.95** per task.

So the honest summary is not "the same quality for less". It is **86% of the bar for an eighth of the price**. Whether that trade is right depends on what a miss costs you, which is the question the floor was supposed to encode.

The gap has a cause worth knowing: a tier score is a *mean*, so a model sitting near the floor will fall below it on the harder half of that tier by construction. Asking for headroom fixes most of it:

| Floor | Headroom | Hit rate | $/task |
|---:|---:|---|---:|
| 70 | +0 | 18/21 (86%) | $1.47 |
| 70 | +5 | 20/21 (95%) | $3.05 |
| 65 | +3 | 19/21 (90%) | $1.47 |
| 65 | +8 | 20/21 (95%) | $2.24 |

Going from 86% to 95% at a floor of 70 costs $1.58 per task and still runs at a quarter of always-opus. If a miss is expensive for you, raise the floor rather than distrusting the router.

## Regenerating the data

Inside the benchmark repository:

```bash
cd benchmarks
uv run scripts/build_vended_models.py           # rewrite models.json
uv run scripts/build_vended_models.py --check   # CI: fail if out of date
```

`models.json` is generated from `docs/metrics/pareto-frontier-omp-swe3.json` and never edited by hand.

## Source

Measurements, harness and methodology: [aws-samples/sample-agentic-coding-harness-benchmarks](https://github.com/aws-samples/sample-agentic-coding-harness-benchmarks). Design notes and open questions: issue #161.
