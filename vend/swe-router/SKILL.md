---
name: swe-router
description: "Checks whether the current model is the right one for the coding task about to be done, and names a cheaper or stronger one when it is not. Uses measured scores and cost per task from a real 21-task benchmark rather than vendor claims, and usually recommends switching DOWN, because most developers run the top model for everything. Advisory only: it recommends a model in a few lines and stops -- it does not run the task, change any setting, or write code. The recommendation ENDS the turn: print it and hand back, so the developer can switch model before any work starts. Run it BEFORE starting a substantial coding task (a feature, a bug fix, a refactor, a migration), and whenever someone asks which model to use, whether a cheaper one would do, or whether they are overpaying. Do NOT run it for trivial requests, for questions, or once work on a task has already started."
license: Apache-2.0
metadata:
  author: Amit Arora
  version: "0.1.0"
---

# SWE Router

Tell the developer which model to switch to. Recommend, then stop — the recommendation is the end of your turn.

You do not run their task. You do not change a setting. You do not write code. The output is a recommendation, the reasoning behind it, and then control back to the developer.

## Why this runs here and not in a gateway

A gateway routes a request that already exists. By then it can see the prompt text, a token count, maybe a model hint. What it cannot see is the thing that decides this choice: what breaks if the change is wrong.

That is knowable only while the work is still an idea. An auth path, a payments flow, a docs page -- the developer knows which one they are about to touch, and nothing on the wire recovers it. So the decision belongs where the task is framed, before a request exists. That is here.

## Do not run when

Stop immediately, say nothing, and get on with what was asked, if any of these hold:

- **The request is small.** A typo, a rename, one obvious line. The advice would cost more attention than the task.
- **Work has already started.** Switching model mid-task loses the context built so far, which is worth more than the price difference.
- **It is a question, not a change.** "How does this work" needs an answer, not a model.
- **The developer already chose.** If they named a model, they have decided. Do not second-guess it unasked.
- **You ran already for this task.** Once per task. A second opinion on the same work is noise.

Everything below assumes a substantial coding task about to begin: a feature, a bug fix, a refactor, a migration. When one is, be quick — the whole output is a few lines, and a developer who wanted a lecture would have asked for one.

## What you have to work with

These sit beside this one:

- **`models.json`** — 18 models measured on a 21-task benchmark: mean score out of 100, cost per task in dollars, hosting, and how many tasks each finished. Read the `provenance` block: it names the date, harness, dataset and judge.
- **`model-aliases.json`** — the same models under the names different assistants use, plus the rules for matching them.
- **`route.py`** — runs the selection. Standard library only, so `python3 route.py` works wherever the skill is installed.
- **`allowed-models.txt`** — the organisation's approved model list, and a hard constraint when present. One model per line, `#` starts a comment. `route.py` finds and reads it; you never open it. It ships allowing the five models on the measured frontier, which a platform team is expected to edit.

Read both before you answer. Never quote a score or a price from memory; if a model is not in `models.json` it has not been measured, and you say so rather than guessing.

## Three steps

### 1. Understand the repo, then the task, then set a quality floor

**Start with the repo's own map.** Read `AGENTS.md` at the repository root. If there is none, read `CLAUDE.md`. If neither exists, fall back to `README.md` and `CONTRIBUTING.md`.

That file is written for an agent working in this codebase, so it is the fastest route to what matters: the stack, the directory layout, where the tests live, what the project treats as risky. Use it to navigate rather than crawling the tree — a repo map read in one file beats twenty guesses at where things are.

Two things to watch for. `AGENTS.md` sometimes points at other documents rather than holding the detail itself, so follow the links it names. And it describes the project's intent, not necessarily its current state — if it contradicts what you see in the code, trust the code.

**Then look at the task.** What is being changed, which files, how much of the system it touches, and whether the repo map flags that area as sensitive.

Then ask the question that decides the answer: **what happens if this change is wrong?**

| The change is… | Floor | Because |
|---|---|---|
| a throwaway prototype, a spike, a scratch script | **55** | nobody merges it; wrong is cheap |
| an internal tool, a test fixture, a docs page | **65** | a human reads it before it matters |
| a production service, anything users touch | **70** | it ships; a defect reaches someone |
| auth, payments, data deletion, a security path | **75** | wrong is expensive and slow to find |

The repo map often answers this for you. A project that calls a directory security-critical, or names a path as user-facing, has already told you the consequence of getting it wrong there. Use its language.

Ask the developer only when the repo does not settle it. One question, not four.

**The floor comes from consequence, not size.** A one-line change to an auth path needs a good model. A three-file mechanical edit does not. Do not raise the floor because the task looks big, or lower it because it looks small.

**One adjustment worth making.** If the task hinges on getting a single specific thing right — an API contract, a portability trap, a security invariant, an exact version comparison — raise the floor by 5. That is where the measured gap between models opens up.

### 1b. Also classify how hard the task is

The floor is one half. The other half is **which table to read it against**.

`score_by_complexity` is four tables, one per difficulty tier. Classifying the task picks the table; the floor then decides which row in it you take. A model has a different score in each, so reading the wrong table gives the wrong answer.

Place the task in one of four tiers:

| Tier | Looks like |
|---|---|
| `trivial` | a docs page, one render condition, two config passthroughs |
| `low` | a default changed across a couple of files, one small feature |
| `medium` | a feature touching several files, a config surface, a template rewrite |
| `high` | a subsystem: rate limiting, server-side token storage, a new authenticated endpoint |

Take that tier's table from `score_by_complexity`, and read `completion_by_complexity` for the same tier beside it. Models do not degrade at the same rate between tables, and some stop finishing tasks at all in the harder ones:

| Model | overall | on `low` | on `high` | finished on `high` |
|---|---:|---:|---:|:-:|
| `claude-opus-5` | 82.83 | 86.2 | 79.8 | 5/5 |
| `claude-sonnet-5` | 76.97 | 80.5 | 73.7 | 5/5 |
| `qwen3.8-27b` | 78.48 | 80.9 | 71.5 | **4/5** |
| `kimi-k2.7-code` | 69.13 | 71.7 | 58.5 | **3/5** |
| `gemma-4-31b` | 59.74 | 63.3 | 50.0 | 5/5 |

Two things the overall mean hides. `qwen3.8-27b` outscores `claude-sonnet-5` overall (78.48 against 76.97) and trails it by 2.2 points on hard work — the ranking between them flips with the tier. And it did not finish one hard task at all, which the mean cannot show because a failure is excluded from it rather than averaged in.

**A completion rate below the full count is a warning, not a rounding detail.** A model that finishes 4 of 5 hard tasks fails one in five outright. Say so when you recommend it, and prefer a model that finishes when the floor is close either way.

Where `score_by_complexity` has no entry for a tier, fall back to the overall `score` and say you did.

**When the task is beyond the measured range, say so.** The hardest tasks here are bounded single-repo changes. A language port, a framework migration, a rewrite, or anything spanning many services is outside what these numbers cover. Do not extrapolate: recommend the strongest model available and tell the developer the measurements do not reach that far. A weak model on an out-of-range task is not a saving.

### 2. Build the candidate set

Two things have to be true of a model before it can be recommended: the organisation permits it, and this benchmark measured it. **`route.py` applies both** — it reads `allowed-models.txt` and checks each name against `models.json`. Do not parse those files yourself; a second reading of them is a second answer waiting to disagree with the first.

**Take `allowed-models.txt` as the list of models this assistant can select.** That is the contract with whoever maintains it: a platform team putting a model on that list is saying developers can reach it from the agent they run. Every entry is a candidate, whatever it is and wherever it runs.

So do not narrow the list a second time. Do not ask the assistant what its picker offers, do not go looking at the provider config, and **do not pass `--available`**. Guessing at what is reachable is how a permitted model gets dropped for no reason — and the guess is usually wrong, because a self-hosted model reached through a proxy or a second agent does not show up anywhere you would think to look.

The one exception is the developer telling you otherwise. If they say a listed model is not wired up for them, pass the ones that are through `--available` and say you did. Their word overrides the file; your inference does not.

**When nothing survives, `route.py` returns `status: "no_candidates"` with a `reason`.** Read it out. Either nothing permitted was measured, or nothing measured is permitted — both belong to whoever maintains the list, and neither is something you route around. Stop there. Do not fall back to a model that failed one of the tests.

**Getting a model is not your problem.** If a self-hosted model is on the list, someone stood up the server. If it is not, it is not a candidate. Never tell anyone to provision hardware.

**Do not treat a model differently because of where it runs.** What it costs to run is already in `cost_per_task_usd`, which is what you rank on. A self-hosted figure comes from the server's hourly price divided by throughput measured under concurrent load — a platform team serving a group of developers, which is how self-hosting is actually done — so it is a cost per task on the same footing as a metered one. Rank them together and say nothing about hosting unless the developer asks.

### 3. Run the selection

You have the two inputs. `route.py` does the rest, so the arithmetic is arithmetic rather than something you work out in prose:

```bash
python3 route.py --tier high --floor 70
```

Two arguments, and that is the whole invocation. It finds `allowed-models.txt` on its own and ranks every permitted model that the benchmark measured. `--allowed-file` overrides the path; `--available` narrows the set further, for the one case in step 2 where the developer says a listed model is not reachable; `--no-allow-list` ignores policy entirely, which you should not do unasked.

It prints JSON. The fields that matter:

| Field | Use |
|---|---|
| `status` | `ok`, `nothing_clears_floor`, or `no_candidates` |
| `recommended` | the model, its score at that tier, and cost per task |
| `margin_over_floor` / `margin_is_meaningful` | how far above the floor, and whether that is more than noise |
| `finished_every_task` / `completion` | whether it finished every task at this tier |
| `cleared_floor` | the runners-up, cheapest first — the next step up is the second entry |
| `reason` | why nothing was recommended, when nothing was |
| `excluded` | what policy or availability removed, so you can name it |

**Report what it returns; do not re-derive it.** If the numbers in the JSON disagree with your reading of `models.json`, the JSON is right.

Run it once. If you find yourself running it twice with different floors to get a nicer answer, stop — the floor came from the consequence of being wrong, and that has not changed.

**Treat a gap under 3 points at a tier as no gap at all.** Each tier holds 5 or 6 tasks, run once each, so a small difference between two models is sampling noise rather than a finding. Dropping a single task from a tier reverses the order of two models 82% of the time when they sit within 1 point, 53% within 2, and 47% within 3. Past 5 points it reverses 5% of the time, and past 8 it never does.

So when the candidates are within 3 points of each other at the task's tier, they are indistinguishable: **take the cheaper one and say they were tied**. Do not present a 1.4-point difference as a reason to prefer a model.

The same applies to the floor itself. A model scoring 71 against a floor of 70 has not reliably cleared it. Recommend it if nothing better is available, and say it sits inside the margin.

That is the whole selection rule. There is no separate step that drops dominated models, because taking the cheapest model above the floor already yields a non-dominated answer — anything that beat it on both axes would have been cheaper, and would have been picked instead.

**Read the tier's table, never the frontier flags.** `models.json` marks each model with `on_combined_frontier` and `on_hosting_frontier` — nothing else beats it on both score and cost across the whole dataset. That is worth mentioning when you recommend a model, and it is not how you choose one. Those flags come from overall means, so they can disagree with the tier that matters: `qwen3.8-27b` is on the combined frontier and still trails `claude-sonnet-5` on high-complexity work. `on_hosting_frontier` compares within one hosting basis, `on_combined_frontier` across both.

The ranking is worked out here in any case, over the models this developer can actually select, at the tier this task actually sits in. A published frontier answers neither question.

Then say it plainly, in the block under *The format of the answer* — that layout is required, not a suggestion. **The first line names the recommended model**, every time — that is the output, and burying it under the reasoning wastes the developer's attention. Say the model `route.py` returned in `recommended`, not a hedge about what you would have picked under other conditions:

- **A candidate clears the floor** → `Recommendation: switch to <model>`. Give its score at the task's tier and its cost per task, say what they are on now and what changes. Mention if it is on the frontier, as context. If a better model exists above it, say what the next step up would cost per extra point, so they can overrule you.
- **They are already on it** → `Recommendation: stay on <model>`. Still name it, and still show the runners-up you ranked it against, so they can see the call was made rather than skipped. "Stay where you are" is a real answer and it is often the right one.
- **Nothing clears the floor** → say that on the first line. Name the closest, how far short it falls, and let them decide. Do not promote a model past its measured score to produce a recommendation.

The recommendation is the one `route.py` returned over the permitted models. Do not qualify it with a model you left out, and never present a model as unavailable on your own inference — if you narrowed the set at all, it was because the developer told you to, and you say which of their words you acted on.

Never invent a switch. A router that always recommends a change is a router that recommends noise.

## Then stop and hand back

The recommendation ends the turn. Print it and stop: no tool call after it, no "meanwhile, I have started on...", no first file written while the developer is still reading which model to use.

A recommendation they cannot act on is not a recommendation. Switching model costs them one command, but only before the work begins — carry straight on and the context is built under the old model, so the advice is dead the moment it is printed. It also reads as though you asked the question and then overruled the answer.

So the basis line is the last thing in the turn. The developer switches or does not, and tells you to go. Then you work.

This holds when the answer is to stay, too. `stay on <model>` still ends the turn: they may disagree with the floor, the tier, or the model, and they get the chance to say so before anything is written.

## Expect to recommend downward

Most developers run the top model for everything. That is what this skill is for.

On the benchmark, if someone is on `claude-opus-5`, then `claude-sonnet-5` lands within 5 points on **10 of 21 tasks** at **61% lower cost**, and matches or beats it outright on 5. Recommending down is the common case, not the exception.

Three cautions that go with it:

- **A downgrade inside the noise is a free saving, not a risk.** If the cheaper model is within 3 points at the tier, the measurements do not distinguish them, so the cheaper one is the right call and the saving is real. What needs care is the opposite: do not describe a 2-point difference as a quality loss the developer is accepting, because it is not measurable.
- **Cheaper is not a free win at low floors.** The weakest models are far behind on every kind of task, easy ones included. On the benchmark the cheapest model scores 42.58 against the best at 82.83 — that is a different outcome, not the same result for less.
- **Downgrades get riskier as the task gets harder.** How much a cheaper model costs you depends on the pair *and* the tier. `claude-sonnet-5` trails `claude-opus-5` by 5.7 points on easy work and 6.1 on hard, so that swap holds up everywhere. `qwen3.8-27b` beats `claude-sonnet-5` by 0.4 on easy work and trails it by 2.2 on hard, while also failing one hard task in five. Check the tier column and the completion count before recommending down on difficult work.

## Say what the recommendation rests on

Every recommendation carries its basis. Not a footnote, a line the developer reads:

> Measured on one repository — a Python/FastAPI and React service with nginx, Terraform and bash around it — over 21 design-and-implement tasks, scored by an LLM judge. Rankings travel better than absolute scores. If your codebase is very different, treat this as a starting point.

State the measurement date from `provenance.measured_on` too. These numbers move: a fix to token accounting once changed `claude-opus-5` from $7.63 to $11.95 per task while every score stayed the same. A stale price with a visible date is honest; a stale price presented as current is not.

**Say that the numbers are not about this task.** Every score and every dollar figure in the block is a mean over the benchmark's 21 tasks — past work, on somebody else's repository, under a different harness. None of it is a forecast for the change in front of the developer. The cost especially: what this task costs depends on how long it runs, how much of the repo it reads and how many turns it takes, and that is not knowable before it starts. So put it in the basis, plainly:

> Scores and costs are that benchmark's per-task averages, not an estimate for this task. What this one costs depends on how long it runs.

A developer who reads `$11.95 per task` and expects a bill of $11.95 for their change has been misled, and it is the number in the block most likely to be taken literally. The comparison between two models is the point. The absolute figure is scaffolding under the comparison.

## The format of the answer

Use this layout. Not something in its spirit — this one, with these labels, in this order:

```
Recommendation: <switch to|stay on> <model>

  Task           <one line: what is about to be built>
  Complexity     <trivial|low|medium|high>
  Consequence    <why the floor is where it is>
  Floor          <number>

  You are on     <model>   <score> on <tier> tasks / $<cost> per task
  Recommended    <model>   <score> on <tier> tasks / $<cost> per task  (<n/n finished>)
  Saving         <percent>% per task, <points> points of headroom above the floor

  Next step up   <model>   <score> for $<cost>  (+<points> points over the pick)

Basis: 21 design-and-implement tasks on one Python/React service repo, <harness>
harness, judged by an LLM. Measured <date>. Scores and costs are that benchmark's
per-task averages, not an estimate for this task. Rankings travel better than
absolute scores.
```

Six rules hold it together:

- **The first line names the model**, prefixed `Recommendation:`. Nothing above it.
- **Two indented spaces, labels in one column, model names in the next, numbers after.** It is read by eye, in a terminal, in about four seconds.
- **Every score carries its tier** — `86.2 on low tasks`, never a bare `86.2`. The reader has to know which table it came from.
- **Show the completion count** beside a recommended model, and always when it is short of the full count: `(4/5 finished)` is a one-in-five outright failure and belongs in front of them.
- **Drop a row with nothing to say.** No `Saving` when the answer is to stay, no `Next step up` when they are already on the strongest permitted model. Pluralise to `Next steps up` when you list more than one. Do not add rows of prose.
- **The basis is the last thing in the turn.** Nothing after it — see *Then stop and hand back*.

A worked one, where the answer is to switch:

```
Recommendation: switch to claude-sonnet-5

  Task           add an env passthrough to two compose files
  Complexity     trivial
  Consequence    internal tooling, a human reviews it before it matters
  Floor          65

  You are on     claude-opus-5   83.8 on trivial tasks / $11.95 per task
  Recommended    claude-sonnet-5 76.1 on trivial tasks / $4.67 per task  (5/5 finished)
  Saving         61% per task, 11.1 points of headroom above the floor

  Next step up   claude-opus-5   83.8 for $11.95  (+7.7 points over the pick,
                 outside the +/-3 noise band, so it is a real gap)

Basis: 21 design-and-implement tasks on one Python/React service repo, omp
harness, judged by an LLM. Measured 2026-09-08. Scores and costs are that
benchmark's per-task averages, not an estimate for this task. Rankings travel
better than absolute scores.
```

And when the answer is to stay. Same layout, same labels — it still names the
model on the first line and still shows what it was ranked against:

```
Recommendation: stay on qwen3.8-27b

  Task           add per-caller rate limiting to the proxy
  Complexity     high
  Consequence    a production service, a defect reaches someone
  Floor          70

  You are on     qwen3.8-27b     71.5 on high tasks / $1.47 per task  (4/5 finished)
                 cheapest permitted model above the floor, and it sits
                 1.5 above it -- inside the noise band, not clear of it;
                 it also failed one of the five hard tasks

  Next step up   glm-5.3         74.7 on high tasks / $7.04 per task
                 3.2 points for $5.57 more, just outside +/-3

  Below the floor  qwen3.6-35b   56.0
                   gemma-4-31b   50.0

Basis: 21 design-and-implement tasks on one Python/React service repo, omp
harness, judged by an LLM. Measured 2026-09-08. Scores and costs are that
benchmark's per-task averages, not an estimate for this task. Rankings travel
better than absolute scores.
```

Short. The numbers they need to disagree with you, and nothing else.

## Stay inside the lines

- Recommend a model. Do not switch it, do not run the task, do not write the code.
- End the turn on the recommendation. Do not start the task in the same turn, whether the answer was switch or stay.
- Do not quote a number that is not in `models.json`.
- Do not present a benchmark average as this task's cost or score. It is what the model averaged on 21 other tasks, not a forecast for this one.
- Do not compare a floor against an overall mean when `score_by_complexity` has the tier.
- Do not filter or rank on `on_combined_frontier` or `on_hosting_frontier`. Report them; select on the tier score.
- Do not present a gap under 3 points at a tier as a difference. It is noise.
- Do not recommend a cheaper model for a task beyond the measured range.
- Do not score a model that was not measured.
- Do not tell anyone to buy or provision hardware.
- If `models.json` has a `schema_version` whose major differs from `1`, stop and say the data is newer than these instructions.
