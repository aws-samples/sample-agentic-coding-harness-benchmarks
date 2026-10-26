# How the swe-router rubric picks a model

`swe-router` recommends which model a developer should use for the coding task in front of them. This page explains **why the rubric asks the six questions it asks**, and how code turns the answers into a model. It is the reasoning behind [`vend/swe-router/rubric.md`](../vend/swe-router/rubric.md), which states the questions without arguing for them.

This is the design the skill ships. It began as a proposal evaluated against an earlier version that judged a consequence floor and a complexity tier in prose; that version was replaced because the judgement moved between identical runs, which [swe-router-evaluation.md](swe-router-evaluation.md) documents.

> [!NOTE]
> **Terms.**
> **Required quality** (the bar): the score a result must reach, from 0 to 100, on the artifact judge's scale.
> **Bucket**: easy, moderate, hard or very hard.
> **Column**: each model's mean measured score on the benchmark tasks in one bucket.
> **Artifact judge**: `codex_judge.py`, which scored every model's work in the benchmark. The rubric never re-runs it.

## The idea in one paragraph

The developer's own model reads the task and the repository and scores six properties of the task, from 0 to 2, citing evidence for each. Code turns two of the scores into a bar and the other four into a bucket. The router then recommends the cheapest allowed model whose measured score in that bucket's column reaches the bar. The model describes the task; it never sees a price, a model name or a threshold.

```text
 task + repository map
        │
        ▼
 the developer's model scores six dimensions, 0-2, with evidence   (rubric.md)
        │
        ├── hidden contract, silent invariant, coordinated surfaces, ambiguity
        │        sum 0-8  ──▶  BUCKET  easy 0-1 · moderate 2-3 · hard 4-5 · very hard 6-8
        │
        └── blast radius, verification
                 ──▶  BAR  60 · 70 · 75  (+5 when nothing would catch a mistake)
                                   (routing-policy.json holds every number)
        │
        ▼
 models.json: each model's measured score on benchmark tasks in that bucket
        │
        ▼
 the cheapest allowed model whose score in that column reaches the bar
```

## Step 1: read the repository, then the task

The model reads `AGENTS.md` at the repository root, else `CLAUDE.md`, else `README.md`, and follows the documents that file links to. It also reads `SECURITY.md` when one exists, because that file says what the project treats as security-sensitive. Then it reads the task and the parts of the repository the task touches: the files it names or implies, the tests around them, and their callers.

Size counts for nothing. The rubric tells the model that line count, file count and task length move no score: a one-line change to an authorisation check can score 2, and a twenty-file mechanical change can score 0.

## Step 2: score six dimensions

| Dimension | The question | Feeds |
|---|---|---|
| Hidden contract | Does something outside the task, such as an earlier issue, a design or another service, define the right behaviour? | bucket |
| Silent invariant | Could a wrong change look like it works, because it breaks a security, data or concurrency guarantee without an error? | bucket |
| Coordinated surfaces | How many separate places (API, command line, user interface, deployment config) must change and agree? | bucket |
| Ambiguity | Does a narrower fix look complete but miss the real problem? | bucket |
| Verification | Would the repository's tests catch a wrong change? | bar |
| Blast radius | What does a mistake cost once shipped, and can a revert undo it? | bar |

`vend/swe-router/rubric.md` defines each dimension: where to look, what 0, 1 and 2 mean, and a common mistake. Every score needs evidence: a file, a document, a test or a sentence of the task. The model also reports its confidence as high, medium or low.

## Step 3: turn scores into a bar and a bucket

`routing-policy.json` holds the numbers, and code applies them.

**The bar** comes from what a mistake would cost:

| Blast radius | Bar | What that score means on the judge's scale |
|---|---:|---|
| 0: throwaway or internal | 60 | bottom of "strong": usable; review will fix real gaps |
| 1: users notice; a revert fixes it | 70 | middle of "strong": ready to implement, with limited gaps |
| 2: security, money or data; hard to undo | 75 | top of "strong": few gaps |

Verification at 2, meaning nothing would catch a mistake, adds 5. The judge gives each of 20 criteria up to 25 points, and its anchors say 16 to 20 per criterion means "implementation-ready in most respects with limited gaps"; a task score of 64 to 80 corresponds to that band.

**The bucket** comes from how demanding the task is: the sum of the four difficulty scores, from 0 to 8.

The bar and the bucket stay separate on purpose. A hard task does not need a better result; it makes every model's result worse, by different amounts. The bucket picks which measured scores to compare, and the bar says how high they must reach.

## Step 4: pick the model

`route.py` reads each allowed model's mean score in the task's bucket from `models.json` and recommends the cheapest model whose score reaches the bar. It treats scores within 3 points of each other as equal. When nothing reaches the bar, it says so and names the shortfall.

`routing-policy.json` sets `when_unsure`, and the skill honours it: when the scoring model reports low confidence, the router returns status `unsure` and keeps the developer on their current model rather than guessing.

`models.json` builds each column from benchmark tasks scored with the same rubric. The skill and its data therefore share one definition of "hard".

## Which file holds what

| File | Read by | Owned by | Changes per customer? |
|---|---|---|---|
| `rubric.md` | the model | this repository | no |
| `routing-policy.json` | code | the customer's platform team | yes: their bar levels and bucket edges |
| `models.json` | code | generated from the customer's benchmark | yes: their measured scores per bucket |
| `allowed-models.txt` | code | the customer's platform team | yes: the models their developers can select |
| `SKILL.md` | the model | this repository | no |

`rubric.md` and `routing-policy.json` are vended with the skill, so the installer carries them. It keeps an edited `routing-policy.json` on upgrade exactly as it keeps `allowed-models.txt`, because overwriting either would silently replace an organisation's policy.

## How a customer adapts it

1. Benchmark their allowed models on their own repositories (`onboard-benchmark`, then `benchmark`).
2. Score their benchmark tasks with the rubric: `run-swe-router-headless.py --mode rubric --dataset <theirs>`. This reads task descriptions only and re-runs no model.
3. Rebuild `models.json` from their results, grouped by bucket.
4. Optionally edit `routing-policy.json` and `allowed-models.txt`.

The rubric and `SKILL.md` stay the same for every customer.

## What the evaluation found

Model classes below follow this repository's convention: closed-source models by tier and generation, open-weight models by parameter count and architecture.

Replayed over the 21 tasks of `mcp-gateway-registry-v2` with every model available, the rubric **saved 75.3% of the cost** of running the closed frontier model on every task, for **3.75 points** of lost quality. The prose-judged floor-and-tier version it replaced saved 70.5% for 5.06 points, so the rubric is both cheaper and more accurate -- the gain came from making the judgement reproducible, not from better arithmetic.

Three results worth keeping in view, because they bound how much to trust it:

- **A cheap judge is good enough to score the rubric.** The efficient model judged about as well as the frontier model at roughly 4% of the cost per judgement. The expensive model is needed for the *work*, not for describing the work.
- **Two refinements did no better than plain buckets.** Relative bucketing and a fitted line per model both failed to beat absolute buckets: 21 tasks run once each cannot separate them. The shipped policy therefore uses `"bucketing": "absolute"`, the simplest of the three.
- **Difficulty costs models different amounts.** The efficient model lost about 4 points per step of rubric difficulty; stronger models lost 1 to 2. This is the whole reason the bucket selects *which column to read* rather than raising the bar: a hard task does not demand a better result, it makes every model's result worse, unevenly.

## What shipped

The design above is live. `rubric.md` and `routing-policy.json` are vended with the skill, `route.py` reads bucket names and honours `when_unsure`, and `models.json` carries a `score_by_difficulty` column per bucket built from a rubric pass over the benchmark tasks.

Two caveats on the figures, both unchanged by shipping:

- They come from **one repository** -- a Python/FastAPI and React service with nginx, Terraform, Helm and bash around it -- and **one run per model per task**. Whether they carry to a different codebase is open.
- The saving is measured against always using the most expensive model. A team already using a mid-tier model by default will see a smaller gap.

Re-measure on your own repositories before trusting the numbers rather than the shape: [How a customer adapts it](#how-a-customer-adapts-it) above is the procedure.
