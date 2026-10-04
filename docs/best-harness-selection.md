# Picking the best harness per model (the combined cost/quality chart)

The per-harness charts answer "which model should I pick, given that I have already chosen a harness". The combined chart -- [`docs/images/cost-quality-combined-swe3.png`](images/cost-quality-combined-swe3.png), emitted by [`benchmarks/scripts/plot_cost_quality_combined.py`](../benchmarks/scripts/plot_cost_quality_combined.py) -- answers the question a buyer actually has: across everything measured, what is the best available per dollar, and which harness gets me there.

That requires collapsing each model's two runs (Claude Code and pi) into a single point, which is only partly well-defined: a harness can be cheaper on one axis and better on the other. This page documents exactly how that choice is made, so no reader has to infer it from the picture.

## The rule, in one line

**Both axes, never one alone: Pareto dominance decides where it can, and the lower cost per point breaks the tie where it cannot.**

Ranking by score alone would systematically plot the pricier harness; ranking by cost alone would plot the weaker one. Neither is a defensible answer to "which harness should I run this model on".

## Step 1: Pareto dominance

A harness run wins outright when it is **no worse on both axes and better on at least one** -- at least the score, at most the cost. This settles 8 of the 13 models measured under more than one harness (of 16 models on the chart; the other 3 ran under a single harness). Nothing is traded away in these cases; the losing run is worse at everything.

With the self-hosted cost correction (see [cost-per-task-methodology.md](cost-per-task-methodology.md) and issue #136, which roughly halved every self-hosted `$/task`), **pi is now the winner for all 12 both-harness models** -- 7 by outright dominance below, 5 on cost/point in Step 2. Several self-hosted models that Claude Code used to win now flip to pi, and three move up from the cost/point tie-break into outright dominance because pi is now both cheaper and higher-scoring.

| Model | Winner | Score | Cost/task | Loser | Score | Cost/task |
|---|---|---:|---:|---|---:|---:|
| Efficient Gen 3 | pi | 47.12 | $0.64 | claude-code | 41.08 | $0.80 |
| Frontier Gen 3 | pi | 75.72 | $8.28 | claude-code | 70.76 | $24.05 |
| 671B MoE | pi | 54.44 | $1.71 | claude-code | 53.72 | $5.26 |
| MoE 2.8T/104B active | pi | 70.76 | $5.98 | claude-code | 65.6 | $12.50 |
| 1T MoE (coder) | pi | 60.68 | $5.52 | claude-code | 55.44 | $6.26 |
| 550B dense | pi | 55.20 | $3.91 | claude-code | 53.68 | $5.43 |
| 27B dense | omp | 70.32 | $3.49 | pi | 66.40 | $3.53 |
| 35B dense | pi | 52.30 | $0.44 | claude-code | 48.16 | $0.71 |

## Step 2: cost per point, as the tie-break

For the remaining 5 models neither run dominates -- one is cheaper, the other scores higher -- so the winner is the one with the lower **cost per point** (`cost per task / mean score`), the same value-efficiency lens the comparison docs report as `Cost/point`.

| Model | Winner | Score | Cost/task | $/point | Runner-up | Score | Cost/task | $/point |
|---|---|---:|---:|---:|---|---:|---:|---:|
| Frontier Gen 2 | pi | 60.68 | $4.60 | 0.0758 | claude-code | 69.24 | $9.90 | 0.1430 |
| Balanced Gen 3 | pi | 66.52 | $3.81 | 0.0573 | claude-code | 68.04 | $24.64 | 0.3621 |
| 123B MoE (coder) | pi | 47.64 | $0.76 | 0.0159 | claude-code | 49.52 | $1.14 | 0.0229 |
| 456B MoE | pi | 45.08 | $0.47 | 0.0104 | claude-code | 48.36 | $0.86 | 0.0179 |
| 480B MoE (coder) | pi | 43.96 | $3.11 | 0.0708 | claude-code | 46.32 | $4.03 | 0.0870 |

## What this does NOT mean

**The plotted point is not always the model's highest score.** This is the most important caveat on the chart, and it follows directly from the tie-break.

Frontier Gen 2 is the sharpest case: Claude Code scores **8.6 points higher** (69.24 vs 60.68) but costs **2.2x more** ($9.90 vs $4.60), so pi wins on value and the chart plots the 60.68. Read the combined chart as *the best value each model offers*, not as *the ceiling each model can reach*. When you need the ceiling, read the per-harness charts.

The alternative -- always plotting the higher score -- was rejected because it misleads more often than it helps: it would put Balanced Gen 3 on the chart at **$24.64** instead of **$3.81** in exchange for 1.5 points, which is not a trade any buyer would make.

## Models measured under one harness only

31B dense (pi), Frontier Gen 2 alt provider (pi) and 30B dense coder (pi) have no second run to compare against, so they pass through unchanged. They are on the chart for completeness, not because their harness was chosen -- if another harness would do better for them, we have not measured it.

## Cost bases are still not interchangeable

Selecting a harness does nothing to fix the underlying cost-basis split: a metered Bedrock bill and a hardware-derived self-hosted figure are **not comparable as raw dollars** (see [cost-per-task-methodology.md](cost-per-task-methodology.md)). The combined frontier drawn on the chart is therefore **cross-hosting and directional only**. The machine-readable output also carries a `bedrock_frontier` and a `self_hosted_frontier` -- the honest like-for-like comparisons -- and those are what any cost claim should quote.

## The machine-readable record

Every decision above is written to [`docs/metrics/pareto-frontier-combined-swe3.json`](metrics/pareto-frontier-combined-swe3.json) under `harness_selection`: for each model the `winner`, the `runners_up` that were set aside (with their scores, costs, and cost/point), a plain-language `verdict`, and whether it was `decided_by` dominance or cost_per_point. Nothing the chart drops is lost -- if you disagree with a call, the losing run is one file away.

## Reproducing it

```bash
cd benchmarks
uv run scripts/plot_cost_quality_combined.py            # light; also writes the frontier JSON
uv run scripts/plot_cost_quality_combined.py --dark     # dark
```

Useful flags: `--harnesses claude-code,pi` selects which harnesses to merge (a harness with no scorable run is skipped with a warning), `--skill swe2` switches skill, and `--log-x` draws cost on a log axis, which spreads the sub-$1 models out of the left margin at the price of an axis that no longer reads directly against the per-harness charts.

Selection logic is covered by [`benchmarks/tests/test_plot_cost_quality_combined.py`](../benchmarks/tests/test_plot_cost_quality_combined.py), including a test that pins the case where the tie-break picks the lower-scoring harness.
