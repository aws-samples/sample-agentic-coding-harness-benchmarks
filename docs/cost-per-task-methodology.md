# Cost per task: methodology, the two lenses, and what agentic coding does to it

How the throughput skill turns a fixed instance price into a **cost per token** and **cost per task** for a self-hosted (vLLM) model, why there are two ways to express it, and what the agentic-coding workload shape means for user experience and for scaling. The core model is **`run_cost = GPU-seconds x $/second`** — price tokens by dividing them by measured throughput at a stated concurrency, never by wall-clock; and the only lever that lowers that cost on a KV-bound model is KV-cache headroom, which trades against context window. A worked GLM-5.2 example runs through both. Companion to [serving-optimization-notes.md](serving-optimization-notes.md); produced by [clients/build_performance_summary.py](../self-hosted/vllm/clients/build_performance_summary.py) and surfaced by [clients/build_performance_dashboard.py](../self-hosted/vllm/clients/build_performance_dashboard.py) and [clients/cost_for_task.py](../self-hosted/vllm/clients/cost_for_task.py).

> [!IMPORTANT]
> **Self-hosted GPU pricing basis (read before quoting any self-hosted dollar figure).** Every self-hosted cost on this page and in the charts is derived from an hourly rate in [`self-hosted/vllm/pricing.json`](../self-hosted/vllm/pricing.json). Both instance families are based at their **3-year commitment rate**, so the whole fleet is one commitment term and the cost columns are comparable across it:
> - **g6e.12xlarge**: **$4.533/hr** (on-demand $10.493, 1-year $6.61).
> - **g6e.4xlarge**: **$1.298/hr** (on-demand $3.004, 1-year $1.893).
> - **p5en.48xlarge**: **$27.72/hr** (on-demand $63.296, 1-year $40.43).
>
> `dollars_per_hour` **is** the rate charged -- nothing is applied on top -- then prorated by `tp / gpus_per_instance` for a partial-box run, so a TP=4 model on the 8-GPU p5en is charged $13.86/hr and a TP=1 model $3.465/hr. The alternative terms sit in each entry's `rates` map for reference only.
>
> **To price at a different term**, move that value into `dollars_per_hour` and regenerate: every cost number, chart, and frontier scales linearly, so [`clients/reprice_performance_summary.py`](../self-hosted/vllm/clients/reprice_performance_summary.py) rescales a committed summary exactly, without needing the (local, gitignored) sweep DuckDB.
>
> There is deliberately **no discount multiplier**. `pricing.json` used to carry one, and p5en was based at on-demand times a `0.35` **placeholder** -- an assumption sitting in the same field, and flowing into the same charts, as measured prices. `pricing.py` now rejects a leftover `discount` key rather than honouring it, so the concept cannot creep back.

### Which AWS discount instrument these rates name

The g6e rates are a "**3-year commitment rate**" without naming an instrument, because the instrument does not change the number: AWS's [own comparison](https://docs.aws.amazon.com/savingsplans/latest/userguide/sp-ris.html) puts an EC2 Instance Savings Plan and a Standard RI in the same **up to 72% off** tier, and a Compute Savings Plan and a Convertible RI in the same **up to 66%** tier. Pick either; the rate lands in the same place. The p5en rates are quoted as **EC2 Instance Savings Plan** rates because that is where they were read from, and they sit at the same ratios to on-demand as g6e (1-year 63.9%, 3-year 43.8%) -- which is the check that the two families are on the same basis.

Reserved Instances are not retired. AWS [recommends Savings Plans over them](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-reserved-instances.html) and still documents buying, modifying, exchanging and reselling RIs. Two capabilities remain RI-only: reselling an unwanted Standard RI on the RI Marketplace, and a zonal RI's capacity reservation. Savings Plans provide no capacity, so pair one with an On-Demand Capacity Reservation if supply is tight.

For a **steady inference endpoint** like the coding-agent workload these benchmarks model, an **EC2 Instance Savings Plan on the GPU family** is the closest fit: the model has to fit the GPU, so cross-family flexibility buys little and costs about six points of discount. Commit only the always-on baseline. Benchmark sweeps like the ones behind these tables are bursty and interruption-tolerant, so they belong on on-demand or Spot rather than under a commitment. Pull real rates with `aws savingsplans describe-savings-plans-offering-rates --service-codes AmazonEC2 --filters name=instanceType,values=<type>` rather than estimating them.

## The starting point: a fixed-cost machine, not a per-token bill

A self-hosted model has **no per-token price**. You rent a GPU instance by the hour (e.g. `g6e.12xlarge` at $4.533/hr on a 3-year commitment) and it processes whatever tokens it can. So the only honest cost is derived, not quoted:

```
dollars_per_second = dollars_per_hour / 3600
```

Everything below attributes that fixed $/second to tokens. The number is real and defensible — it is what the hardware actually costs to run — unlike the fictional token-priced `total_cost_usd` the quality harness records for self-hosted models.

## The two lenses (why there are two)

On a fixed-cost machine the $/hr is **not itemized per token**, so "cost per token" depends on how you attribute the GPU-second. We report both; they answer different questions.

### Lens A — blended / measured (no assumption)

Every processed token — prompt or generation — costs the same slice of GPU time:

```
blended_cost_per_token = dollars_per_second / (prompt_tokens_per_second + generation_tokens_per_second)
```

Both throughputs are measured server-side from vLLM counters over the concurrency window. Input and output cost the **same** per token. This makes no pricing assumption — it just divides the bill by the work done. **This is the primary lens** (see "Why blended is the honest headline" below).

### Lens B — lab-style split (one convention `w`)

The commercial-API shape, where input tokens are billed cheaper than output. We introduce one convention: an input token counts as `w` of an output token when splitting GPU time (default `w = 0.25`).

```
cost_per_output_token = dollars_per_second / (generation_tps + w * prompt_tps)
cost_per_input_token  = w * cost_per_output_token
```

`w` is a **chosen convention, not a measurement** — it exists only to produce the familiar input-cheaper-than-output shape for comparison with hosted APIs. It is a secondary view.

### Cost per task (either lens)

A "task" is defined by its token counts `N` input : `M` output, taken from real agentic runs (`metrics.json` of the model's `/swe3` sessions):

```
task_cost = cost_per_input_token * N + cost_per_output_token * M      # Lens B
task_cost = blended_cost_per_token * (N + M)                          # Lens A (per-token equal)
```

Here `N`/`M` are **all processed tokens** — for the blended lens, `N + M` means every token the server handled: fresh input + output **plus cache-read + cache-write**. This matters because the blended rate was measured over every token vLLM processed (cached prefixes included), so the count it multiplies must include them too; and because it is the only token basis that agrees across harnesses (pi reports cache tokens separately, Claude Code folds them into `input` — see the [appendix](#appendix-prompt-caching-and-why-self-hosted-vs-api-costs-are-not-measured-the-same-way)). Pricing only fresh `input + output` would undercount a pi run by ~half while leaving a Claude Code run unchanged.

The dashboard exposes `N`, `M`, and `w` as adjustable inputs so you can price any task shape. [clients/cost_for_task.py](../self-hosted/vllm/clients/cost_for_task.py) does the same from the CLI for any task run **outside** the harness — because per-token cost is a property of the model + hardware + load, not of the tasks the sweep happened to run.

### Cost is GPU-seconds times dollars-per-second — NOT wall-clock times dollars-per-hour

Substitute the blended rate into the task-cost formula and it collapses to something more intuitive:

```
run_cost = (dollars_per_second / combined_tokens_per_second) * tokens_processed
         = dollars_per_second * (tokens_processed / combined_tokens_per_second)
         = dollars_per_second * GPU-seconds
```

The blended lens is just **"how many GPU-seconds did this work occupy, times the price of a GPU-second."** That is the whole model. Two things flow from it:

- **Throughput converts token *work* into GPU-seconds.** `tokens_processed / combined_tok_per_sec` is the time the GPU actually spent crunching tokens — with the idle gaps (agent thinking, tool calls, network) removed. This is why we divide by *measured* throughput and never by wall-clock.
- **Concurrency selects *which* throughput you divide by,** and that is where amortization lives. At concurrency 1 the box sustains a low combined rate (one lonely stream); at its saturation concurrency it sustains a much higher one, because idle gaps in one session are backfilled by other sessions' tokens. Pricing at the saturated rate assumes you keep the box busy — i.e. the hourly cost is shared across concurrent users. **The self-hosted cost is therefore an operating-point assumption, not a free number.**

**Why not just pro-rate wall-clock?** Because a single agentic session leaves the GPU idle most of the time. Wall-clock pricing charges you full box price for that idle time and, worse, assumes one user owns the whole box. See the GLM-5.2 worked example below.

#### Worked example: GLM-5.2, 5 SWE tasks on p5en.48xlarge (8xH200, $27.72/hr = the 3-year EC2 Instance Savings Plan rate for the full box)

From the throughput sweep (the throughput sweep's `performance-summary.json`) and the SWE run (the run's `run-summary.json`). The sweep's cheapest sustainable point is **concurrency 5**, where the server sustains **15,679 prompt + 189 decode = 15,867 combined tok/s** and the KV cache is already 100% full. The 5 tasks processed **41,519,145 tokens** (input+output). On this self-hosted vLLM run the `cache_read`/`cache_write` tokens are a *partition* of `input_tokens`, not additions to it, so they are already counted inside input and are NOT added again (adding them back would ~2x the count; see issue #136 and the appendix). This is measured over **6,288 wall-clock seconds**.

| Costing method | Calculation | Result | What it assumes |
|---|---|--:|---|
| **Blended / GPU-seconds (what we use)** | `41,519,145 / 15,867 = 2,617 s = 0.727 hr; x $27.72` | **$20.15** | box kept busy at c=5 (shared across ~11 concurrent requests) |
| Wall-clock pro-rate (rejected) | `6,288 s = 1.747 hr; x $27.72` | $48.42 | one user owns the box; idle time billed |
| GPU-seconds at concurrency 1 (rejected) | `41,519,145 / 7,019 = 1.64 hr; x $27.72` | $45.55 | dedicated box, serial single user |

The $20.15 figure is the **c=5** number. Note it is *lower* than the naive wall-clock estimate ($48.42) — because dividing by measured throughput removes the idle wall-clock (6,288 s elapsed vs 2,617 s of actual token-crunching, ~1.0 hr idle) the agent spent thinking and tool-calling. And it is below the c=1 figure ($45.55) — the gap between those two is exactly the amortization concurrency buys. If you cannot actually run the box at c=5 (e.g. you dedicate it to one serial developer), your true cost is nearer $46, not $20. Always state the operating point.

Every figure in that table is linear in the hourly rate, so the whole example rescales by one factor: at 1-year ($40.43/hr) the blended figure is $29.39, and at on-demand ($63.296/hr) it is $46.01. That linearity is what [`clients/reprice_performance_summary.py`](../self-hosted/vllm/clients/reprice_performance_summary.py) exploits.

### The trade-off between the lenses

- **Blended** is assumption-free and workload-honest, but it prices input and output identically — which looks unfamiliar next to a commercial API invoice.
- **Split** looks like an API bill, but its output-heavy pricing (`cost_out = 4x cost_in` at `w=0.25`) **understates the true cost of an input-heavy workload**. Agentic coding is exactly that (see below), so the split lens's headline "cost per output token" can be misleading — a task that is 98% input tokens is cheap under split's logic but is really consuming the machine's prefill capacity. Report both; **trust blended for this workload.**

## What agentic coding does to all of this

Agentic coding has an extreme request shape: **very large, read-heavy prompts and small outputs.** Measured on the **pi** agent driving the single-agent `/swe3` skill (5 real runs each on `mcp-gateway-registry`; token counts from each run's `metrics.json`), the input:output ratio is extremely lopsided, because a single agent replays the whole growing transcript on every turn. (These are pi numbers, not Claude Code: only pi splits prompt tokens into `input + cache_read`, so its prompt-side count is directly comparable across models — see the [appendix](#appendix-prompt-caching-and-why-self-hosted-vs-api-costs-are-not-measured-the-same-way).)

| Model (pi `/swe3`) | Turns/task | Prompt tokens/task (input+cache) | Output/task | Ratio |
|---|--:|--:|--:|--:|
| glm-5.2 | 75 | ~16.44M | ~107.1K | **153:1** |
| qwen3.6-35b | 41 | ~7.54M | ~41.9K | **180:1** |
| deepseek-v3.2 | 84 | ~8.87M | ~33.1K | **268:1** |
| kimi-k2.7-code | 117 | ~20.36M | ~50.4K | **404:1** |
| nemotron-ultra-550b | 112 | ~27.43M | ~41.4K | **663:1** |

The model reads the repo, tool-calls, reasons over big files, and emits a comparatively tiny design/patch — so the ratio sits in the **~150:1 to ~660:1** band. This is the extreme prefill-heavy end of the workload spectrum, and it is why the server is prefill/KV-bound (see [References](#references) for external corroboration at 180-220:1).

**What drives the spread is turn count, not reasoning-token output.** The ratio's numerator grows with **turns**: every turn re-feeds the entire growing transcript as fresh prompt, so a model that takes 112 turns (nemotron) accumulates ~27M prompt tokens while a model that finishes in 41 (qwen3.6-35b) accumulates ~7.5M. It is *not* that reasoning models emit more output — the opposite: the highest-ratio models (nemotron, kimi) emit the *fewest* output tokens per turn (~370-430), while glm-5.2 and qwen3.6-35b emit the *most* (~1,000-1,400/turn) and have the *lowest* ratios. There is no separate reasoning/thinking token field in the metrics — whatever a model streams as thinking is already inside `output_tokens` — so a verbose reasoner would *lower* the ratio (bigger denominator), not raise it. The lever to cut these models' cost is therefore **fewer turns** (better tool use, less thrashing), not suppressing reasoning.

That shape has three consequences:

1. **The server is prefill-heavy: it spends far more compute on reading prompts than on generating.** Prompt throughput (~9-13K tok/s) dwarfs generation (~90-145 tok/s). But "prefill-heavy" is not the same as "saturated" — on a healthy server the prefill still completes in a few seconds (measured **prefill ~2-4s mean** across the sweep), because the prompt is processed in large batched chunks, not token-by-token.

2. **User experience = time-to-first-token, and on a healthy server it is fine and degrades gracefully.** Because each request must prefill ~100K+ tokens before the first output token, **TTFT is the felt latency**. Measured (median) it is **~1-2s uncontended and rises to ~5-8s at concurrency 20** — good, and it degrades gracefully, not off a cliff. TTFT = **queue wait + prefill**; the decomposition shows prefill is a flat ~2-4s and queue wait stays near zero until higher concurrency (p50 0s up to c=7, ~2s by c=20). Report **TTFT as p50/p90**, never the mean: the mean is distorted by a few cold-cache outliers and by how many requests complete in the window, which can make it move in the wrong direction.

   Full curve (qwen3.6-35b, multi-repo, 25 repos, 10-min windows per level, healthy server):

   | c | gen t/s | prompt t/s | TTFT p50 | TTFT p90 | queue p50 | prefill mean | blended $/1M | task $ |
   |--:|--:|--:|--:|--:|--:|--:|--:|--:|
   | 1 | 145 | 12202 | 2s | 20s | 0s | 4s | 0.10 | 0.16 |
   | 2 | 114 | 13252 | 1s | 20s | 0s | 3s | 0.10 | 0.14 |
   | 5 | 87 | 9441 | 2s | 20s | 0s | 3s | 0.14 | 0.21 |
   | 7 | 111 | 10168 | 5s | 20s | 0s | 3s | 0.12 | 0.19 |
   | 10 | 103 | 10326 | 5s | 40s | 1s | 3s | 0.12 | 0.19 |
   | 15 | 109 | 10667 | 8s | 80s | 2s | 3s | 0.12 | 0.18 |
   | 20 | 134 | 11705 | 5s | 40s | 2s | 2s | 0.11 | 0.16 |

   > **Watch the server state when you measure.** An earlier run of this same sweep reported TTFT p50 pegged at the histogram ceiling (>640s) with queue-wait means of ~135-240s. That was **not** the model's true behavior — the server was in a backed-up state (a stale scheduler backlog from prior experimentation): it completed ~7x fewer requests per window (23 vs 171 at c=10) because they sat queued. A clean re-run gave the healthy 1-8s numbers above. The lesson: the c=1 baseline and the queue-vs-prefill split exist precisely to catch this — if the c=1 TTFT is not a small number, or queue-wait dominates prefill at low concurrency, the server is not in a clean state and the run should be discarded.

3. **Report cost on the blended (per processed token) lens.** Generation tok/s alone is a poor headline for an input-heavy workload; blended cost counts prompt + generation, i.e. the work the machine actually does. Blended cost stayed in a tight **$0.10-0.14/1M** band across the whole sweep, cheapest at low concurrency, which is the stable, comparable figure.

## The only lever that lowers self-hosted cost: KV-cache headroom (which trades against context window)

Since `run_cost = GPU-seconds x $/sec` and `$/sec` is fixed by the instance, the **only** way to lower cost is to shrink GPU-seconds — i.e. raise the sustained combined throughput. And throughput is capped by whichever runs out first: compute (prefill-bound) or KV-cache space (memory-bound). For a large model on a memory-tight box, it is the **KV cache**, and this is where the interesting trade-off lives.

**The caching benefit is already in the throughput — do not try to credit it again.** vLLM prefix caching (98-99% hit rate on the GLM SWE run) lets the server skip prefill recompute for the repeated conversation prefix; that is *why* prompt throughput is ~15.7K tok/s rather than a fraction of it. Higher throughput → fewer GPU-seconds → lower cost. The cache payoff shows up as the low blended rate, not as a per-token discount — see the [appendix](#appendix-prompt-caching-and-why-self-hosted-vs-api-costs-are-not-measured-the-same-way) for why applying both would double-count.

**But prefix cache and live-request KV compete for the same HBM, and that caps concurrency.** GLM-5.2-FP8 is ~744 GB of weights on a 1,128 GB p5en, leaving only ~380 GB for KV. At a **300K-token context window** each concurrent agentic session reserves a large KV slab, so the box saturates KV at only ~11 running requests (**concurrency 5** — where the sweep shows `kv_cache_usage.peak = 1.00` and `requests_waiting` starts climbing). Beyond c=5 throughput does *not* rise; extra sessions just queue. So GLM on p5en has **no vertical headroom** — it is already at its cheapest sustainable point at c=5.

**The one real lever is a smaller context window.** Reduce the per-session KV footprint (e.g. 300K → 128K) and more sessions fit before KV saturates → higher combined throughput → fewer GPU-seconds per task → lower `$/task`. On a KV-bound model this is essentially the *only* knob short of cheaper hardware (reserved/spot) or a smaller model.

**There is no free lunch — the window is also an accuracy lever.** Agentic coding *uses* long context: input-per-call climbs from ~50K early in a session to ~200K deep in it as the transcript, tool outputs, and read files accumulate (this is the ~150:1 input:output ratio, corroborated by external agentic-coding data at 180-220:1 — the GLM run's 153:1 is normal, not anomalous; see [References](#references)). Truncate the window and the model loses earlier reasoning and file context, which can lower task quality. So **cheaper serving (more KV headroom → more concurrency → higher throughput) is bought with a shorter context window that risks lower accuracy.** The right operating point is workload-specific: measure score vs window, don't assume.

## Scaling: it depends whether the model is KV-bound or has headroom

**Two regimes, and you must know which one you are in — do not assume vertical headroom.**

**Regime A — KV-bound (e.g. GLM-5.2 on p5en, above):** the model nearly fills the box, KV saturates at low concurrency, and there is *no* vertical headroom. Concurrency is capped by memory, not latency budget. Your levers are context-window reduction, cheaper hardware, a smaller/right-sized model (a small-active MoE that fits at TP=4 leaves far more KV room and serves many more concurrent sessions), or horizontal replicas — **not** "push concurrency higher on this box."

**Regime B — headroom (e.g. qwen3.6-35b on g6e.12xlarge):** on a **healthy** server this workload is not at a hard ceiling — TTFT p50 is only 5-8s at c=20, KV cache never saturates, and there are no preemptions. So there is genuine vertical headroom: this instance can take more concurrency before latency becomes a problem, and the serving defaults (chunked prefill on, batch budget auto) already use it well. Push concurrency up until p90 TTFT or KV pressure crosses your latency budget — that is the per-instance operating point (the dashboard's recommended-concurrency banner picks the cheapest-blended point; temper it with the p90 TTFT you can tolerate).

**How to tell which regime you are in:** look at `kv_cache_usage.peak` across the sweep. If it hits 1.00 at low concurrency and throughput stops rising there, you are KV-bound (Regime A). If it stays well under 1.0 as concurrency climbs, you have headroom (Regime B).

Beyond that per-instance limit, scale **horizontally**: the workload is embarrassingly parallel (N developers on N repos are N independent sessions, nothing to synchronize), so add replicas behind a load balancer. The **blended cost per token is roughly flat across replicas** — each instance has the same $/hr and the same throughput profile — so cost scales linearly with load and the per-task cost measured on one instance is the per-task cost at fleet scale: **measure once, multiply by replicas for capacity.**

## A third cost basis: managed-model credits (kiro-cli)

The two lenses above turn a GPU's hourly price into a cost per task for **self-hosted** models. A hosted API (Anthropic on Bedrock) uses its **metered per-token bill**. The **kiro-cli** harness introduces a third basis again: kiro-cli drives Kiro's managed, Bedrock-backed models and bills in **credits**, not tokens or GPU-seconds. See [kiro-cli-setup.md](kiro-cli-setup.md) for install and the harness constraints.

**What kiro-cli reports.** A non-interactive kiro-cli run emits no token counts. It prints a one-line summary to stderr on completion -- `▸ Credits: <n> • Time: <s>s` -- so the per-run cost signal is **credits consumed** (and wall-clock time). The credits figure already includes the model's `rate_multiplier` (from `kiro-cli chat --list-models`: claude-opus-5 at 2.2 burns credits faster than qwen3-coder-next at 0.05), so it is not multiplied by the rate again.

**Credits to dollars.**

```
cost_per_task_usd = credits_consumed_for_the_run x DOLLARS_PER_CREDIT
```

Kiro's published pricing gives two defensible per-credit rates:

| Basis | $/credit | Derivation |
|---|--:|---|
| Blended (included monthly allotment) | **$0.02** | Every paid tier is the same rate: Pro $20/1,000, Pro+ $40/2,000, Pro Max $100/5,000, Power $200/10,000 |
| Marginal (add-on / overage) | **$0.04** | "Add-on credits $0.04/credit" once the monthly allotment is spent |

`DOLLARS_PER_CREDIT` is a **configurable rate**, the same stance this page takes on the self-hosted GPU rate (a documented commitment term you swap for your own). The default is the **$0.04 marginal** rate -- the honest "what does one more task cost" figure -- with $0.02 available for an all-you-can-use blended view. Example: a run reporting `Credits: 0.21` costs `0.21 x $0.04 = $0.0084` (or `$0.0042` blended); a real swe task at 50-300 turns consumes far more.

**How the harness records it.** `run-swe-headless.py` captures kiro-cli's output, strips the ANSI color codes, and regex-parses the `Credits:` value from the summary line. It multiplies that by `kiro_dollars_per_credit` -- a config knob (default 0.04) settable in `runner.yaml` or with `--kiro-dollars-per-credit` -- to get the run's `total_cost_usd`. Each task's `metrics.json` stores **both** the raw `kiro_credits` (provenance) and the derived `total_cost_usd`, and `summarize_run.py` averages `total_cost_usd` across the run's tasks into the reported `$/task` (the `mean_cost_usd_excl_failed` field). So the dollar figure is always traceable back to the exact credits kiro-cli charged.

**Important: Kiro is a per-developer monthly subscription, not pure usage-based pricing -- and this figure ignores that.** Kiro sells seats (see [kiro.dev/pricing](https://kiro.dev/pricing/)): Free ($0/mo, 50 credits), Pro ($20/mo, 1,000), Pro+ ($40/mo, 2,000), Pro Max ($100/mo, 5,000), Power ($200/mo, 10,000). Those credits are **included in the seat**, and the $0.04/credit rate applies **only to add-on/overage credits once the monthly allotment is spent**. The `credits x $0.04` cost this repo reports therefore treats **every credit as marginal overage** -- as if the monthly allotment were already exhausted -- which is the conservative worst case. For a developer working **within** their allotment, the marginal dollar cost of one more task is effectively already paid by the seat (up to the cap); the amortized rate is closer to the blended **$0.02/credit** (seat price / included credits). Set `kiro_dollars_per_credit` to reflect your plan and expected volume.

**This makes the cross-harness dollar comparison structurally uneven, not just a different unit.** pi and Claude Code on Bedrock are **pure usage-based, per-token** billing -- no seat, no monthly commitment; you pay only for the tokens you consume. Kiro bundles a **fixed monthly seat plus an included credit allotment**. So comparing kiro's per-task credit cost against Bedrock metered dollars compares a *subscription-plus-credits* model against a *usage-based* one. For a real total-cost comparison, model kiro's **monthly seat cost + expected task volume** against the others' metered (Bedrock) or hardware-derived (self-hosted) spend -- do not read the single per-task dollar figure as directly equivalent.

**Do not compare raw dollars across the three bases.** Metered Bedrock dollars, hardware-derived self-hosted GPU-seconds, and Kiro credits are measured on different footings (and, per the note above, the credit-to-dollar conversion depends on your Kiro plan and whether you are within your monthly allotment). As with the metered-vs-self-hosted comparison, treat any cross-basis dollar tie as an order-of-magnitude result and state the provenance; compare within a basis.

## Summary

- Cost is derived from `instance $/hr / measured tokens/sec` — real, not a quoted price. The formula collapses to **`GPU-seconds x $/second`**: price the tokens by dividing them by *measured throughput*, never by wall-clock (which over-charges idle agent-thinking time and assumes one user owns the box). Worked example: GLM-5.2's 5 SWE tasks cost **$29.90 at c=5**, vs $72 naive wall-clock and $68 at c=1 — the operating point *is* the number, so always state it.
- **Blended** (per processed token, input == output) is the honest primary lens; **split** (`w`-weighted, API-shaped) is a familiar-but-misleading secondary lens for input-heavy work. Caching is already baked into the blended rate (it raises throughput) — do not also discount cache-read tokens per-token, that double-counts.
- Agentic coding is input-heavy (~50:1 early, ~150-220:1 deep in a session), so the server is prefill-heavy — but on a healthy instance TTFT is a few seconds and degrades gracefully; **report TTFT as p50/p90 (not mean) and decompose queue vs prefill**, and always include a c=1 baseline to catch a backed-up server.
- **The only lever that lowers self-hosted cost is raising sustained throughput, which on a KV-bound model means more KV headroom — bought by shrinking the context window, at a possible accuracy cost. No free lunch.** Check `kv_cache_usage.peak`: if it pegs at 1.00 at low concurrency (Regime A, e.g. GLM-5.2 on p5en) there is no vertical headroom — right-size the model/window or scale horizontally; if it stays low (Regime B, e.g. qwen3.6-35b on g6e) push concurrency up to your latency budget first.
- Blended per-task cost is flat across replicas: plan capacity from the measured per-task cost — **measure once, multiply by replicas.**

## Companion: picking the best harness per model

The combined cost/quality chart plots one point per model rather than one per model-and-harness, which means something has to choose between a model's Claude Code run and its pi run. That choice uses **both** axes -- Pareto dominance first, then the lower cost per point as the tie-break --. Note the plotted point is therefore not always the model's highest score.

## References

Public data on the input-heavy / prefill-heavy shape of agentic-coding workloads — the external corroboration for the ~150:1 input:output ratio and the "prefill-bound, not decode-bound" framing used above:

- [Together.ai — Benchmarking inference at scale: coding agents](https://www.together.ai/blog/coding-agent-benchmarks) — per-request accumulated context ~80-100K input vs ~450 avg output; implied **~180:1 to 220:1**.
- [Requesty — The Coding Agent Economy](https://www.requesty.ai/coding-agent-economy) — avg input per call ~84K (Claude Code), ~95K (OpenCode); output "negligible."
- [Applied Compute — Benchmarking inference on agentic workloads](https://www.appliedcompute.com/research/inference-benchmark) — per single assistant turn ~10K input : ~200-300 output (**~33:1 to 50:1** early-session).
- [dstack — Benchmarking prefill-decode ratios](https://dstack.ai/blog/benchmarking-pd-ratios/) — prefill/decode contrast; reasoning workloads sit at the opposite (~1:3) regime.

The ratio climbs through a session because each tool call replays the growing transcript as fresh input while generating only a small edit — ~30-50:1 early, ~180-220:1+ deep in a session — which is why the server is prefill/KV-bound and why KV-cache headroom (not decode speed) is the cost lever.

## Appendix: prompt caching, and why self-hosted vs API costs are not measured the same way

*Measurement-plumbing detail, not part of the core cost model. It explains how `total tokens processed` — the count the blended cost multiplies — is derived, and why it depends on whether the backend's cache fields are a partition of `input_tokens` or additions to it. Numbers below are from the `/swe3` runs on `mcp-gateway-registry`.*

Comparing a self-hosted model's `$/task` against a hosted API model's (e.g. Claude on Bedrock) is the most error-prone part of this analysis, because **the two paths account for cached tokens completely differently.** This is not a modeling choice — it is what each backend reports back to the client.

- **Anthropic API / Bedrock** implements explicit prompt caching and returns `cache_read_input_tokens` / `cache_creation_input_tokens` in every response's `usage`. Claude Code records those, so on a Bedrock run the reused context lands in `cache_read_tokens` (billed at ~10% of the input rate) and the fresh, full-price `input_tokens` is tiny. Example from an Opus-4.8 `/swe3` task: **`input_tokens: 461`, `cache_read_tokens: 25,260,499`** — ~99.99% of the prompt served from cache. Opus's per-task cost is therefore dominated by *output* tokens (verbose, priced high), not input.
- **On the self-hosted path the two harnesses report cache tokens differently, because they read different fields off vLLM's OpenAI-compatible response:**
  - **Claude Code** does not populate the Anthropic-specific cache fields for a vLLM endpoint, so it records **`cache_read_tokens: 0`** and books the entire (re-fed, growing) conversation as fresh `input_tokens`. Example from a GLM-5.2 Claude Code `/swe3` run: **`input_tokens: 86,412,298`, `cache_read_tokens: 0`.**
  - **pi** *does* surface vLLM's prefix-cache accounting, so it reports a `prefix_cache_hit_rate` and splits the prompt into `cache_read_tokens` + `cache_write_tokens`, both of which are a **partition of** `input_tokens` — not additions to it. The same GLM-5.2 model under pi `/swe3`: **`input_tokens: 40,983,637`, `cache_read_tokens: 40,663,808`** — i.e. `cache_read` is **99.2% of `input`** (the prefix-cache hit rate), because `input_tokens` already counts the full prompt and `cache_read`/`cache_write` merely describe how much of it was served from cache. `cache_read + cache_write ≈ input_tokens` is the signature of this partition.

**The two harnesses do NOT process the same number of tokens, and the total-processed basis depends on how each backend accounts for cache.** Claude Code's GLM-5.2 prompt total is `input` ≈ 86.4M (it folds cache into input, `cache_read` = 0); pi's is `input` ≈ 41.0M (of which ~99% was served from cache). These are genuinely different amounts of work — the two runs took different numbers of turns / grew conversations of different lengths — not the same prompt "categorized differently," which is what an earlier version of this note wrongly claimed by adding pi's `cache_read` on top of its `input` (`input + cache_read` ≈ 81.6M) to force an apparent match. That double-counted the cached prompt and inflated every self-hosted total (and therefore cost) by ~2x (issue #136). The correct **total tokens processed** is partition-aware:

- **Self-hosted vLLM (cache is a partition of `input`):** `total = input + output`. The cache is already inside `input`, so it must not be added again. (GLM-5.2 pi `/swe3`: 41.0M input + 0.5M output ≈ 41.5M, not 82.7M.)
- **Anthropic / Bedrock (cache is additive to `input`):** `total = input + output + cache_read + cache_write`. Here `input_tokens` is only the fresh, uncached tokens (often ~2), and the reused prompt lives separately in `cache_read`, so it must be added.

The blended `$/token` rate is measured server-side over each token counted **once** (`vllm:prompt_tokens_total` + `vllm:generation_tokens_total`), so the count it multiplies must also count each token once — which is exactly what the partition-aware total does.

**This does NOT mean either model failed to cache.** vLLM's `--enable-prefix-caching` (on by default here) caches the KV of repeated prefixes server-side and reuses them across the growing agentic conversation. pi surfaces that reuse in `cache_read_tokens`; Claude Code does not, so its client-side `input_tokens` overstates the fresh work — but the GPU did the same thing under both.

**The caching is measured server-side, and `/swe3` pi captures it directly.** Each pi `/swe3` task records a true server-side `prefix_cache_hit_rate` (from vLLM's `vllm:prompt_tokens_cached_total` / `prompt_tokens_total`). Over the `/swe3` runs on `mcp-gateway-registry` the mean prompt-cache hit rate — the self-hosted analogue of Anthropic's `cache_read_tokens` fraction — was uniformly high across **every** self-hosted model, on both node types:

| Model (self-hosted, pi `/swe3`) | Node | Mean `prefix_cache_hit_rate` |
|---|---|--:|
| qwen3-coder-480b | p5en.48xlarge | **98.8%** |
| glm-5.2 | p5en.48xlarge | **98.7%** |
| minimax-m2.5 | p5en.48xlarge | **98.3%** |
| deepseek-v3.2 | p5en.48xlarge | **98.2%** |
| devstral-2-123b | p5en.48xlarge | **98.2%** |
| kimi-k2.7-code | p5en.48xlarge | **97.5%** |
| nemotron-ultra-550b | p5en.48xlarge | **96.7%** |
| gemma-4-31b | g6e.12xlarge | **96.6%** |
| qwen3-coder-30b | g6e.12xlarge | **95.9%** |
| qwen3.6-35b | g6e.12xlarge | **95.2%** |

So on `/swe3`, **~95-99% of the prompt tokens Claude Code would have counted as fresh input were actually served from vLLM's prefix cache** — exactly the tokens Anthropic would have reported (and discounted) as `cache_read`. The single-agent `/swe3` shape replays a stable prefix on every turn, so it caches uniformly well across models and node types. To estimate an API-style billable-input for a self-hosted run: `billable_input ~= input_tokens x (1 - hit_rate)`.

**Why the hardware-derived cost is still fair despite the 0 in the Claude Code client.** The blended `$/token` already bakes the caching in: it comes from *measured throughput* (tokens/sec the server actually sustained), and that throughput was achieved *with* prefix caching active. So a self-hosted `$/task` is not penalized for the un-credited `input_tokens` — the cheap per-token rate reflects a GPU that was mostly reusing cached prefills. The client-side token *count* is inflated; the *cost* is not.

**Bottom line for cross-path comparison.** When a hosted-API model (small billable input, cached) lands near a self-hosted model (huge counted input, cheap per token) on `$/task`, treat it as an **order-of-magnitude** result, not an exact tie — the two token counts are measured on different bases. State the provenance (metered API bill vs hardware-derived) alongside the number.
