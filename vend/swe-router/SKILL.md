---
name: swe-router
description: "Checks whether the current model is the right one for the coding task about to be done, and names a cheaper or stronger one when it is not. Scores the task on a six-question rubric, then picks the cheapest permitted model whose measured benchmark score on tasks of the same difficulty clears the quality the task needs. Usually recommends switching DOWN, because most developers run the top model for everything. Advisory only: it recommends a model in a few lines and stops -- it does not run the task, change any setting, or write code. The recommendation ENDS the turn: print it and hand back, so the developer can switch model before any work starts. Run it BEFORE starting a substantial coding task (a feature, a bug fix, a refactor, a migration), and whenever someone asks which model to use, whether a cheaper one would do, or whether they are overpaying. Do NOT run it for trivial requests, for questions, or once work on a task has already started."
license: Apache-2.0
metadata:
  author: Amit Arora
  version: "0.2.0"
---

# SWE Router

Tell the developer which model to switch to. Recommend, then stop: the recommendation is the end of your turn.

You do not run their task. You do not change a setting. You do not write code. The output is a recommendation, the numbers behind it, and then control back to the developer.

## Why this runs here and not in a gateway

A gateway routes a request that already exists. By then it can see the prompt text, a token count, maybe a model hint. It cannot see what decides this choice: what breaks if the change is wrong, and what the repository already does to catch it.

You can see both, because you are in the repository while the work is still an idea. So the decision belongs here, before a request exists.

## Do not run when

Stop at once, say nothing, and get on with what was asked, if any of these hold:

- **The request is small.** A typo, a rename, one obvious line. The advice would cost more attention than the task.
- **Work has already started.** Switching model mid-task loses the context built so far, which is worth more than the price difference.
- **It is a question, not a change.** "How does this work" needs an answer, not a model.
- **The developer already chose.** If they named a model, they have decided. Do not second-guess it unasked.
- **You ran already for this task.** Once per task. A second opinion on the same work is noise.

Everything below assumes a substantial coding task about to begin: a feature, a bug fix, a refactor, a migration. Be quick. The whole output is a few lines.

## What you have to work with

These sit beside this file:

- **`rubric.md`**: six questions about the task, each answered 0, 1 or 2, with what each answer means. You answer them.
- **`route.py`**: turns your six answers into a model. Standard library only, so `python3 route.py` works wherever the skill is installed.
- **`routing-policy.json`**: how `route.py` turns answers into numbers. A platform team may have edited it; you never need to open it.
- **`models.json`**: every measured model's mean benchmark score in each difficulty bucket, its cost per task, and how many benchmark tasks in each bucket it finished. Its `provenance` block names the date, dataset, harness and judge.
- **`model-aliases.json`**: the same models under the names different assistants use.
- **`allowed-models.txt`**: the models your organisation permits. `route.py` reads it; you never open it.

Never quote a score or a price from memory. If a model is not in `models.json`, it has not been measured, and you say so rather than guess.

## Three steps

### 1. Score the task with the rubric

Read `rubric.md` and do what it says, in order: read the repository's map (`AGENTS.md`, else `CLAUDE.md`, else `README.md`, plus `SECURITY.md` if present), read the task, look at the code and tests it touches, then answer the six questions. Each answer is 0, 1 or 2, with one line of evidence: a file, a test, a document, or a sentence of the task.

The six questions do two jobs, and you do not need to work out either:

| Questions | What `route.py` makes of them |
|---|---|
| hidden_contract, silent_invariant, coordinated_surfaces, ambiguity | added up, they give the task's **bucket**: its difficulty level, one of easy, moderate, hard or very hard |
| verification, blast_radius | the **required quality**: the benchmark score a model must reach for this task |

Do not try to steer either number. Answer each question about the task as it would land in this repository, and let the policy price it. Size is not a signal, as `rubric.md` says: a one-line change to a password check scores higher on blast radius than a twenty-file change to an internal dashboard.

Then rate your confidence in the six answers as a whole: **high** when the repository and the task settle every one, **medium** when one or two rest on inference, **low** when you could not find what you needed. Low is a real answer. It keeps the developer on their current model, which is the right call when you cannot tell.

Ask the developer only when the repository cannot settle blast radius, the one answer that rests on how the change will be used. One question, not four.

**When the task is beyond the measured range, say so.** The hardest benchmark tasks are bounded changes inside one repository. A language port, a framework migration, a rewrite, or work spanning several services is outside what these numbers cover. Recommend the strongest permitted model and tell the developer the measurements do not reach that far. Do not run `route.py` for it.

### 2. Run the selection

Pass your six answers and your confidence:

```bash
python3 route.py --confidence high --rubric '{"hidden_contract": 1, "silent_invariant": 0, "coordinated_surfaces": 2, "ambiguity": 1, "verification": 1, "blast_radius": 1}'
```

**If `python3` is missing or the command fails to start, stop.** Tell the developer the skill needs Python 3.8 or later on PATH, with no packages, and recommend nothing. Do not work out the bucket, the required quality or the pick yourself: the arithmetic lives in `route.py` so that it is never done in prose.

That is the whole invocation. `route.py` finds `allowed-models.txt` and `routing-policy.json` on its own. It works out the bucket and the required quality, keeps the permitted models the benchmark measured, skips any that failed a benchmark task in that bucket, and takes the cheapest whose score in that bucket clears the required quality.

It prints JSON. The fields that matter:

| Field | Use |
|---|---|
| `status` | `ok`, `nothing_clears_floor`, `no_candidates`, or `unsure` |
| `bucket`, `difficulty_sum` | the task's difficulty level, and the sum of the four answers behind it |
| `required_quality` | the score a model must reach, from blast radius and verification |
| `recommended` | the model, its score in that bucket, its cost per task, and how many of the bucket's tasks it finished |
| `margin_over_floor` / `margin_is_meaningful` | how far above the required quality, and whether that is more than noise |
| `cleared_floor` | every model that cleared, cheapest first; the next step up is the second entry |
| `skipped_for_failures` | models that would have cleared but failed a benchmark task in this bucket |
| `scored_from_harder_bucket` | models read from a harder bucket because the benchmark had no task in this one |
| `reason` | why nothing was recommended, when nothing was |
| `excluded` | what the allow-list removed, so you can name it |

**Report what it returns; do not re-derive it.** If the JSON disagrees with your reading of `models.json`, the JSON is right. Run it once. If you catch yourself running it again with different answers to get a nicer result, stop: the task has not changed.

### 3. Read the result and say it

**The candidate set is the allow-list.** A platform team that puts a model on `allowed-models.txt` is saying developers can reach it from this assistant. Do not narrow the list a second time: do not inspect the model picker, do not read provider config, and do not pass `--available`. Guessing at what is reachable drops permitted models for no reason. The one exception is the developer telling you a listed model is not wired up for them; then pass the ones that are through `--available` and say you did.

**When `status` is `no_candidates`,** read out the `reason`. Either nothing permitted was measured, or nothing measured is permitted. Both belong to whoever maintains the list. Stop there.

**When `status` is `unsure`,** your confidence was low. Recommend staying on the current model and say which answer you could not settle.

**Treat a gap under 3 points as no gap.** Each bucket holds a handful of benchmark tasks, run once each, so a small difference between two models is noise. When candidates sit within 3 points, take the cheaper one and say they tied. The same goes for the required quality: a model 1 point above it has not reliably cleared it. `margin_is_meaningful` tells you which case you are in; recommend the model anyway if nothing better is permitted, and say it sits inside the margin.

**Select on the bucket, never on the frontier flags.** `on_combined_frontier` and `on_hosting_frontier` in `models.json` come from overall means. Mention them as context; do not choose by them.

**Do not treat a model differently because of where it runs.** `cost_per_task_usd` already holds what it costs. A self-hosted figure is the server's hourly price divided by throughput measured under load, so it sits on the same footing as a metered price. Rank them together and say nothing about hosting unless asked. Never tell anyone to provision hardware.

Then say it, in the layout under *The format of the answer*:

- **A candidate clears** → `Recommendation: switch to <model>`. Give its score in the bucket and its cost per task, what they are on now, and what changes. If a stronger model sits above it, say what the next step up costs per extra point, so they can overrule you.
- **They are already on it** → `Recommendation: stay on <model>`. Still show what it was ranked against.
- **Nothing clears** → say so on the first line. Name the closest model and how far short it falls, and name any model skipped for failing tasks in the bucket. Let them decide.

Never invent a switch. A router that always recommends a change recommends noise.

## Then stop and hand back

The recommendation ends the turn. Print it and stop: no tool call after it, no "meanwhile, I have started on...", no first file written while the developer is still reading which model to use.

Switching model costs the developer one command, but only before the work begins. Carry on under the old model and the advice is dead the moment it is printed. This holds when the answer is to stay, too: they may disagree with an answer, and they get the chance to say so before anything is written.

## Expect to recommend downward

Most developers run the top model for everything. That is what this skill is for. On most tasks a cheaper model's score in the bucket clears the required quality, so recommending down is the common case.

Two cautions go with it:

- **A downgrade inside the noise is a free saving.** If the cheaper model is within 3 points, the measurements do not tell them apart. Do not describe that gap as a quality loss.
- **Downgrades get riskier as the bucket gets harder.** Models lose points at different rates as tasks get harder, and some start failing tasks outright. `route.py` already reads the bucket and skips models with failures there; report what it skipped.

## Say what the recommendation rests on

Every recommendation carries its basis, as a line the developer reads. Build it from `models.json`: the dataset in `provenance.dataset`, the number of benchmark tasks, the harness in `provenance.harness`, the date in `provenance.measured_on`, and the repository description in `measurement_basis.single_repository_warning`.

Say that the numbers are not about this task. Every score and every dollar figure is a mean over benchmark tasks on somebody else's repository. None of it forecasts the change in front of the developer. The cost especially: what this task costs depends on how long it runs. A developer who reads `$4.44 per task` and expects that bill for their change has been misled.

## The format of the answer

Use this layout, with these labels, in this order:

```
Recommendation: <switch to|stay on> <model>

  Task           <one line: what is about to be built>
  Difficulty     <bucket> (rubric sum <n> of 8)
  Required       <required quality> (blast radius <0-2>, verification <0-2>)

  You are on     <model>   <score> on <bucket> tasks / $<cost> per task
  Recommended    <model>   <score> on <bucket> tasks / $<cost> per task  (<n/n finished>)
  Saving         <percent>% per task, <points> points above the required quality

  Next step up   <model>   <score> for $<cost>  (+<points> points over the pick)

Basis: <n> benchmark tasks on <repository description>, <harness> harness,
judged by an LLM. Measured <date>. Scores and costs are that benchmark's
per-task averages for <bucket> tasks, not an estimate for this task.
```

The rules that hold it together:

- **The first line names the model**, prefixed `Recommendation:`. Nothing above it.
- **Two indented spaces, labels in one column, model names in the next, numbers after.** It is read by eye, in a terminal, in about four seconds.
- **Every score carries its bucket**: `80.1 on hard tasks`, never a bare `80.1`.
- **Show the completion count** beside the recommended model.
- **Drop a row with nothing to say.** No `Saving` when the answer is to stay, no `Next step up` when they are on the strongest permitted model. Add a `Skipped` row when `skipped_for_failures` names a model that would otherwise have been picked.
- **The basis is the last thing in the turn.**

A worked example, with made-up model names and numbers:

```
Recommendation: switch to mid-model

  Task           add per-caller rate limiting to the proxy
  Difficulty     very_hard (rubric sum 6 of 8)
  Required       70 (blast radius 1, verification 1)

  You are on     large-model   80.2 on very_hard tasks / $4.44 per task
  Recommended    mid-model     77.2 on very_hard tasks / $2.62 per task  (9/9 finished)
  Saving         41% per task, 7.2 points above the required quality
  Skipped        small-model   failed 1 of 9 very_hard tasks

  Next step up   large-model   80.2 for $4.44  (+3.0 points over the pick,
                 inside the +/-3 noise band, so no measurable gain)

Basis: 21 benchmark tasks on one Python/FastAPI and React service, omp
harness, judged by an LLM. Measured 2026-10-11. Scores and costs are that
benchmark's per-task averages for very_hard tasks, not an estimate for this task.
```

## Stay inside the lines

- Recommend a model. Do not switch it, run the task, or write the code.
- End the turn on the recommendation, whether the answer was switch or stay.
- Do not compute the bucket or the required quality yourself; `route.py` does.
- Do not quote a number that is not in `models.json` or the `route.py` output.
- Do not present a benchmark average as this task's cost or score.
- Do not present a gap under 3 points as a difference.
- Do not recommend a cheaper model for a task beyond the measured range.
- Do not filter or rank on `on_combined_frontier` or `on_hosting_frontier`.
- Do not tell anyone to buy or provision hardware.
- If `models.json` has a `schema_version` whose major is not `2`, stop and say the data and these instructions do not match.
