# Why this exists

Why this repo measures harness x model on real repositories, and what the quality and throughput benchmarks each contribute to a cost per task.

## Why this exists

Enterprises are adopting coding agents and models at scale, and the bill grows with every developer and every task. The two big levers on that bill -- **which harness** drives the work and **which model** it drives -- get chosen on gut feel or on public leaderboards that may already be **saturated**: models can be tuned toward well-known public test sets, so a high headline number is a poor predictor of performance on a team's actual, messy, long-horizon coding work.

This repo measures what decides the bill instead: **harness x model, on real agentic software-engineering tasks against real repositories**, reporting all three axes a buyer trades off -- **cost, latency, and accuracy**. Crossing harnesses with models gives real **optionality**: the same model can be a few points more accurate under one agent yet several times cheaper and faster under another. With those numbers in hand, an organization can make an **informed, defensible decision** about the cost/latency/accuracy trade-off for its own workload -- and, very often, **cut its coding bill** by picking a cheaper harness-and-model pairing that is more than good enough, rather than defaulting to the most expensive option. That is the deliverable: an evidence base for smart, budget-aware choices on work that looks like yours, not like a leaderboard.

## Overview

This repository is a **benchmark and harness for measuring how well different LLMs perform real-world software-engineering tasks** when driven by a coding agent. It supports **four coding agents (harnesses)** today -- [Claude Code](https://docs.anthropic.com/en/docs/claude-code), Anthropic's command-line coding agent; [pi](https://github.com/earendil-works/pi-coding-agent), a lightweight open-source agent; [oh-my-pi](https://github.com/can1357/oh-my-pi) (`omp`, [omp.sh](https://omp.sh)), a fork of pi -- see [omp setup](omp-setup.md); and [kiro-cli](https://kiro.dev) (the successor to the Amazon Q Developer CLI), which drives Kiro's own managed, Amazon Bedrock-backed models -- with [opencode](https://opencode.ai) being added soon. Claude Code and pi are each wired to run with a model hosted in any of **three different places**, so you can put many models through the *same* tasks with the *same* harness and compare them on both quality and cost; kiro-cli instead runs Kiro's managed models directly (it cannot target a self-hosted endpoint -- see [kiro-cli setup](kiro-cli-setup.md)). Pick the harness per run with `--agent claude` (default), `--agent pi`, `--agent omp`, or `--agent kiro`, and the skill with `--skill swe2`/`--skill swe3`; results are kept separate on disk (`<model>/<harness>/<skill>/<repo>/<task>`) so neither the agents nor the two skills ever overwrite each other.

It runs **two complementary benchmarks**, and combining them is the whole point:

1. **Quality** -- how well a model does a real coding task (scored 0-100 by an independent LLM judge).
2. **Throughput** -- how many tokens per second a self-hosted model sustains on a given GPU instance, which turns the instance's hourly price into a **hardware-derived cost per task**.

Quality alone tells you which model is best; cost alone tells you which is cheapest. Plotting one against the other yields the **cost/quality Pareto frontier** (the chart below) -- the set of models where nothing else is both better *and* cheaper. That frontier is the deliverable: it is what lets you choose a model for a real budget, and it exists only because this repo measures both halves.

**The two benchmarks must be combined over the *real agentic coding tasks*, not over a synthetic input:output token ratio.** Agentic coding is a **prefill-heavy, long-horizon** workload: each task replays a large, growing transcript as fresh input on every turn and emits a much smaller edit, so the real input:output ratio runs ~150:1 up to ~660:1 -- far more lopsided than the ~3:1 or ~4:1 assumed by generic pricing. A model's cost per task therefore depends on *how* it drives the task (how many turns, how much context it re-reads, whether prefix caching hits), which a lab-style token-count estimate cannot capture. This repo measures throughput on that same prefill-heavy shape and multiplies it by the tokens each run processed, so the cost on the frontier is the cost of the *work as it happens* -- see [cost-per-task-methodology.md](cost-per-task-methodology.md).


---

[< Back to the README](../README.md)
