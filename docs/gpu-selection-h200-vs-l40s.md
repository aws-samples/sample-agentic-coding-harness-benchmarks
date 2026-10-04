# GPU choice: H200 slices against L40S instances

**Host these models on p5en H200 slices, not on g6e L40S instances.** Every model measured on both families serves more agentic-coding load per dollar on one H200 than on an L40S box. A g6e instance runs these models correctly, so it stays a legitimate option. It also saturates at one or two concurrent sessions and costs 1.4x to 3x more per token, which rules it out for an endpoint a team shares.

**The illustration below is a 27B dense (FP8) model**, worked end to end so the mechanism rests on measurements: a p5en.48xlarge costs 21x more per hour than a g6e.4xlarge and serves 48x more concurrent coding sessions at the same 10-second first-token wait, so each developer costs 56% less, $0.26/hr against $0.60/hr. That test changes one thing: the 27B dense (FP8) model runs at TP=1 on one L40S, then at TP=1 on **one of the eight H200s** in a p5en, charged at one eighth of the box. Same model, same flags, same dataset, same window.

The same conclusion was then confirmed on three further models against a whole 4-GPU g6e.12xlarge; see [every model lands the same way](#every-model-lands-the-same-way). All prices are public on-demand with no discount applied to any box.

For the cross-model throughput tables see [agentic-coding-throughput-comparison.md](agentic-coding-throughput-comparison.md). For the cost model see [cost-per-task-methodology.md](cost-per-task-methodology.md).

> [!NOTE]
> **Terms this doc uses.**
> - **Prefill** is the server reading the prompt. **Decode** is generating tokens one at a time. Agentic coding sends huge prompts and gets short replies (about 160:1 here), so prefill does most of the work.
> - **KV cache** holds the attention state for every token of every session in flight. Its size decides how many sessions one GPU can hold.
> - **Admitted** means vLLM gave a request KV cache and started running it. A request the server cannot admit sits in a queue. `requests_running` and `requests_waiting` count each group.
> - **TTFT** (time to first token) is how long a developer waits before output starts. **TPOT** (time per output token) is the gap between tokens after that.
> - **Blended cost** charges prompt and generated tokens the same, because they use the same GPU slice: `($/hr / 3600) / tokens per second`.

## Headline

| | g6e.4xlarge (1x L40S) | p5en slice (1x H200, TP=1) | ratio |
|---|--:|--:|--:|
| $/hr, public on-demand | $3.004 | $7.912 (1/8 of $63.296) | 2.63x |
| Peak generation tok/s | 71.03 @ c=10 | **356.72 @ c=30**, still climbing | **5.02x** |
| Best total tok/s (prefill + decode) | 1,547 @ c=5 | **6,870 @ c=30** | **4.44x** |
| Sessions admitted at once | **7**, hard wall | **30+** at matched config; **87** at `--max-num-seqs 128` | **4.3x** to **12x** |
| TPOT at c=1 | 61.6 ms | **11.9 ms** | 5.18x faster |
| Cheapest blended $/1M tokens | $0.539 | **$0.320** | **41% cheaper** |
| Cheapest $/task (8M in : 50K out) | $4.341 | **$2.575** | **41% cheaper** |

The H200 slice costs 2.6x more per hour and does 5x the work, so it wins on cost per token while serving 4x the users. Every figure in this table holds the serving config identical across the two GPUs. Tuning the H200's scheduler for its own hardware pushes it further, to 454 gen tok/s and $0.235/1M, at a latency cost [the ceiling section](#the-h200s-ceiling-measured) sets out.

> [!IMPORTANT]
> **Every price in this doc is public on-demand, us-east-1, with no discount applied to either box.** That is a deliberate choice, made so the comparison is like for like: $3.004/hr for a g6e.4xlarge and $63.296/hr for a p5en.48xlarge. Note that these are **not** the repo's configured rates, which put both families on their 3-year commitment price (~43% of on-demand), so every cost figure here is about 2.3x the matching number in the throughput summaries and the cross-model tables. Because the two boxes now carry the same commitment term, that difference is a single scale factor and no ratio in this doc moves.
>
> **Nobody actually runs a steady inference fleet at on-demand.** In practice you would put the always-on baseline under a 1- or 3-year commitment -- AWS sells that as an EC2 Instance Savings Plan or a Standard Reserved Instance, both in its "up to 72% off" tier -- and pay roughly 43% of the rates above. **This doc models none of that.** Discounting both boxes by the same factor leaves every ratio, every percentage and every break-even in this doc unchanged, because $/hr enters the cost model linearly; it only scales the absolute dollars down. Discounting them by *different* factors, which is what happens if you commit to one family and not the other, moves the conclusion, and this doc deliberately does not go there.

### What it costs to serve N developers

Both boxes are priced at the concurrency each sustains under a 10-second first-token budget: 5 sessions on a g6e.4xlarge, 30 on one H200, 240 on a full p5en. Cost per developer-hour is **$0.6008** on the L40S and **$0.2637** on the H200 -- **56% less** -- and the H200 also gives each developer slightly more output (11.9 tok/s against 10.4).

| developers | g6e.4xlarge fleet | H200 slices (shared p5en) | dedicated whole p5en |
|--:|---|---|---|
| 100 | 20 boxes, $60.08/hr | 4 slices, **$31.65/hr** (47% less) | 1 box, $63.30/hr (5% **more**) |
| 240 | 48 boxes, $144.19/hr | 8 slices, **$63.30/hr** (56% less) | 1 box, **$63.30/hr** (56% less) |
| 500 | 100 boxes, $300.40/hr | 17 slices, **$134.50/hr** (55% less) | 3 boxes, **$189.89/hr** (37% less) |

At 240 developers the annual gap is **$1.26M against $554k**.

> [!WARNING]
> **The p5en only wins if the box is full.** A p5en.48xlarge sells as one 8-GPU unit, so 100 developers use 42% of it and pay for all of it, which puts it 5% behind a 20-box g6e fleet. A dedicated p5en breaks even at **106 developers** -- the point where the g6e fleet has to buy its 22nd box. Below that, either share the spare GPUs with other work (the slice column assumes you can) or stay on g6e.

## Every model lands the same way

The 27B dense (FP8) comparison above pits one H200 against **one** L40S. The harder test gives the L40S box every advantage: three more models, each already benchmarked on a **whole g6e.12xlarge** (4x L40S, TP=4, $10.493/hr), re-run on **one** H200 at $7.912/hr. The single GPU is the *cheaper* option here -- **0.754x the hourly price** -- so it only has to reach three quarters of the four-GPU box's throughput to win on cost.

One H200 beat the whole box on all three models, on throughput and on cost per token together. Each headline below quotes both sides at that model's **usable knee**, not at its cheapest level; [the note below](#why-the-knee-and-not-the-cheapest-level) gives the rule.

**35B dense -- one H200 replaces the four-L40S box outright.** It sustains **28,467 tok/s across 20 concurrent sessions at 2.4 s to first token**, against **13,366 tok/s across 2 sessions at 4.5 s** on the whole g6e.12xlarge: 2.13x the throughput, serving 10x the sessions, at **3.01x lower cost per token** ($0.077 against $0.232 per 1M) and $0.119 against $0.324 per task.

**31B dense -- one H200 beats four L40S by two thirds.** **4,169 tok/s across 7 sessions** against **2,491 tok/s across 2**, at **2.26x lower cost per token** ($0.523 against $1.181 per 1M) and $0.750 against $1.667 per task. This is the one arm where the H200's usable point carries the *higher* first-token wait (10.7 s against 5.0 s): a dense 31B model is prefill-bound on either card, so the H200 spends its advantage on batch depth instead of latency.

**30B dense (coder) -- near-parity throughput, spread across 7 sessions instead of 1.** **13,088 tok/s across 7 sessions** against **12,000 tok/s across a single session**, at **1.37x lower cost per token** ($0.169 against $0.232 per 1M) and $0.464 against $0.671 per task. The 1.09x throughput margin is thin, so read this arm with the [caveat below](#what-this-means-for-hosting).

**27B dense (FP8) -- the worked example in the rest of this doc, against one L40S rather than four.** **6,870 tok/s across 30 sessions** against **1,547 tok/s across 5** on a g6e.4xlarge: 4.44x the throughput, 41% cheaper per token, 56% cheaper per developer-hour.

Full curves, latency percentiles and per-task costs sit in the [source data](#source-data). Each model repeats the mechanism the 27B dense (FP8) sections below take apart, so this doc measures it once.

### The g6e knee falls at one or two sessions

**Where each box's knee falls** decides the hosting question. Every g6e arm knees at **c=1 or c=2**. One level past it, queueing takes 35-39% of first-token latency: 30B dense (coder) at c=2, 31B dense and 35B dense at c=5. One H200 holds **7 to 20** sessions inside the same latency envelope.

The four-GPU box fails for a different reason than the single L40S in the 27B dense (FP8) arm below. There, KV cache pinned at 1.000 and vLLM capped admission at 7 sessions. Here KV peaks at just 23-43% at each arm's knee, so memory is not the limit: the box runs out of prefill compute. Agentic coding sends about 1.4M prompt tokens per task against 21K generated, so prefill does nearly all the work, and requests queue for a compute slot while half the KV pool sits idle. KV saturates only much later, at c=20, where 30B dense (coder) peaks at 0.995 with 81% of its TTFT already spent queueing.

The g6e's published throughput peaks therefore overstate what it can serve. 30B dense (coder) reports 20,045 tok/s at c=10 on the g6e box, its best figure anywhere. Queueing accounts for 59% of first-token latency at that level, so the figure measures a growing backlog. Its largest usable figure is 12,000 tok/s, at a single session. The [L40S peak section](#the-l40s-peak-comes-with-a-67-second-wait) takes the same failure apart for 27B dense (FP8), where the throughput peak costs a 67-second wait.

A one-session ceiling cannot host an endpoint a team shares. Serving 100 developers from g6e takes 20 boxes for the 27B dense (FP8) shape and 100 for 30B dense (coder) at its g6e knee, the arithmetic in [fleet consolidation](#fleet-consolidation). One H200 slice serves 7 to 20 sessions and costs less per token.

### Why the knee and not the cheapest level

`build_performance_summary` reports the cheapest $/1M across a whole sweep, which for 31B dense lands at c=20 -- where mean TTFT is 72 s and **88% of it is queue time**. Prefill time per request stays flat at 7-9 s across every concurrency in that arm, so the GPU's work per request never changes. All TTFT growth past the knee comes from requests parked waiting for admission, so the extra concurrency buys a backlog.

The knee used above is the highest concurrency where **queue time is under 30% of TTFT and mean TTFT is under 15 s**. Quoting it costs almost nothing: pushing 31B dense from its knee to its cheapest level saves 15% per token for **6.7x the TTFT**. For 35B dense and 30B dense (coder) the knee *is* the cheapest level, so there is no trade to make at all.

The same rule picks the knee on both instance families. The g6e loses more from it, because its cheapest levels are its most queued.

### What this means for hosting

Host on **p5en.48xlarge, one H200 per model replica (TP=1)**. Reserve g6e for cases where a p5en cannot be filled, since a p5en sells as one indivisible 8-GPU unit and only wins when the whole box is used -- the [break-even is 106 developers](#what-it-costs-to-serve-n-developers) for the 27B dense (FP8) shape. Below that, share the spare GPUs with other work or stay on g6e and accept the single-session ceiling.

35B dense makes the strongest case for consolidation: 28,467 tok/s at 2.4 s TTFT on one card, TTFT between 1.9 s and 2.6 s from c=1 all the way to c=20, and a curve still climbing at the top of the sweep. Eight such replicas fit in one p5en.

> [!WARNING]
> **Treat 30B dense (coder) as provisional, the weakest of the three.** A rate change on either box could flip its 1.09x throughput gain, its H200 knee sits at 25% queue share against a 30% threshold, and its g6e baseline knee of c=1 looks suspect: that arm was already 38% queued at c=2, which points at a noisy baseline. The cost conclusion holds. The throughput margin carries little weight. Re-running its g6e arm would settle it.

## Config

| | value |
|---|---|
| Model | `<YOUR_MODEL_HF_ID>`, served as `<your-model-slug>` |
| Tensor parallelism | TP=1. The H200 arm sets `CUDA_VISIBLE_DEVICES=0` so vLLM cannot recruit the other seven cards |
| Context window | 65,536 |
| GPU memory utilisation | 0.90 |
| KV cache dtype | fp8 |
| Scheduler caps | `--max-num-seqs 32`, `--max-num-batched-tokens 8192` |
| Sweep | concurrency 1, 2, 5, 7, 10, 15, 20 (H200 adds 30), 600 s per level |
| Dataset | [`multi-repo-throughput.yaml`](../benchmarks/dataset/multi-repo-throughput.yaml), 25 tasks across 25 public repos |

Holding the flags fixed has two side effects worth naming, both deliberate:

- **`GPU_MEM_UTIL=0.90` hands the H200 far more KV cache** (0.90 x 141 GB against 0.90 x 48 GB, less about 29 GB of weights on both). Equalising it would hide the thing an H200 sells. The memory comes with the card.
- **`MAX_MODEL_LEN` stays at 65,536** though an H200 could serve much more. Raising it would change a second variable.

The H200 arm added c=30, one level past the baseline's c=20, and found the H200 still climbing where the L40S had peaked and stalled. So every H200 figure in the next three sections is a floor, and [the ceiling section](#the-h200s-ceiling-measured) reports a second arm that goes looking for the wall.

## Why the gap exists

Three mechanisms, each measured on its own.

### Memory capacity sets a hard admission ceiling

`requests_running` counts the sequences vLLM held resident. Both boxes ran `--max-num-seqs 32`, so no number below is a config cap.

| offered c | L40S running (peak / mean) | L40S waiting (mean) | L40S KV peak | H200 running (peak / mean) | H200 waiting (mean) | H200 KV peak |
|--:|--:|--:|--:|--:|--:|--:|
| 5 | 6 / 4.25 | 0.29 | 0.994 | 5 / 4.11 | 0.02 | 0.086 |
| 7 | **7** / 4.61 | 0.43 | **1.000** | 7 / 5.71 | 0.04 | 0.111 |
| 10 | **7** / 5.10 | 2.85 | **1.000** | 10 / 8.13 | 0.12 | 0.164 |
| 15 | **7** / 5.13 | 6.98 | **1.000** | 15 / 12.19 | 0.21 | 0.232 |
| 20 | **7** / 5.33 | 10.64 | **1.000** | 20 / 16.15 | 0.39 | 0.301 |
| 30 | | | | 30 / 23.86 | 0.92 | 0.408 |

The L40S ran **7 sequences and no more** at every offered load from 7 to 20, with KV pinned at 100%. An exhausted KV pool was refusing admission. Offering it 20 sessions parked 10.6 of them in a queue.

The H200 admitted every session offered at every level, held `waiting` under 1, and finished c=30 using 41% of its KV pool.

### Memory bandwidth sets per-token decode speed

At c=1 the KV pool sits at 3%, so capacity cannot explain anything. The decode latency gap there:

| | TPOT mean @ c=1 | HBM bandwidth |
|---|--:|--:|
| L40S | 61.6 ms | 864 GB/s |
| H200 | **11.9 ms** | 4.8 TB/s |
| ratio | **5.18x** | 5.56x |

Decode streams the active weights out of HBM for every token, so it runs at the speed of memory. The measured 5.18x tracks the 5.56x bandwidth ratio. This advantage is independent of capacity, and it is the one most people mean by "faster GPU".

### The H200 spends its speed on batch depth

The two advantages do not multiply. Compare each box at its own peak:

| | mean running | gen tok/s | per sequence |
|---|--:|--:|--:|
| L40S @ peak (c=10) | 5.10 | 71.03 | **13.9 tok/s** |
| H200 @ peak (c=30) | 23.86 | 356.72 | **14.9 tok/s** |

One session decodes at about the same rate on either GPU once both are loaded. The whole 5x aggregate gap comes from the H200 holding 4.7x more sessions. Its TPOT slides from 11.9 ms to 66.6 ms as it fills to 30 sequences, landing at the L40S's per-token latency while carrying 4.7x the users.

A fleet wants that trade. The same silicon buys lower latency for one developer or more developers at the same latency, and a shared endpoint takes the second.

## The L40S peak comes with a 67-second wait

The L40S hits 71.03 tok/s at c=10, a state where 2.9 of 10 sessions queue and KV stays pinned. The latency that buys:

| | TTFT mean | TTFT p50 | TTFT p99 | queue mean |
|---|--:|--:|--:|--:|
| L40S @ c=10, its peak | **67 s** | **80 s** | 640 s | 41 s |
| L40S @ c=20 | **101 s** | **160 s** | 640 s | 79 s |
| L40S @ c=5 | 10.5 s | 7.5 s | 80 s | 2.0 s |
| H200 @ c=30, its peak | **9.2 s** | **2.5 s** | 80 s | 3.9 s |

At its throughput peak the L40S makes half its requests wait 80 s or more for a first token, which rules out interactive use. Its usable concurrency is about 5, not the 10 where the headline number lives.

> vLLM records TTFT in a coarse histogram, so p50/p90/p99 are **bucket upper bounds** (2.5, 7.5, 20, 40, 80, 160, 640 s), not exact quantiles. Read `p99 = 640 s` as "somewhere between 160 s and 640 s". The means are exact.

Comparing the two boxes at matched latency instead of matched concurrency widens the gap:

| at TTFT mean about 10 s | operating point | gen tok/s |
|---|---|--:|
| L40S | c=5 | 51.87 |
| H200 | c=30 | **356.72** |
| ratio | | **6.88x** |

The H200 slice does 6.9x the work at the same wait, for 2.63x the price. Matching on means is the conservative choice: on medians the H200 at c=30 (2.5 s) beats the L40S at c=5 (7.5 s) outright, so a median-matched comparison would favour the H200 further.

## Cost per token

Each box is costed at its own cheapest operating point, both at public on-demand:

| | operating point | total tok/s | $/1M blended | $/task (8M in : 50K out) |
|---|---|--:|--:|--:|
| g6e.4xlarge | c=5 | 1,547 | $0.5393 | $4.3413 |
| p5en slice | c=30 | 6,870 | **$0.3199** | **$2.5754** |
| H200 advantage | | 4.44x | **41% cheaper** | **41% cheaper** |

The break-even test asks how much throughput the H200 slice needs to earn its price: 1,547 tok/s x 2.6338 = **4,075 tok/s**. It delivered 6,870, clearing the bar by **69%**. The margin is wide enough that a rate change on either box does not flip the sign.

> **Prefill picks the cost-optimal concurrency, not decode.** At 160:1 the prompt tokens dominate the blended figure. The L40S is cheapest at c=5 rather than at its c=10 throughput peak, because KV thrash cut its prompt throughput from 1,495 to 855 tok/s between the two levels. The H200's prompt throughput was still rising at c=30 (3,534 to 6,513 tok/s from c=20).

## Fleet consolidation

Per-task cost stays flat across replicas, so a per-slice figure is a fleet figure. Read a p5en as **eight independent TP=1 replicas** of this model:

| | one g6e.4xlarge | one p5en.48xlarge (8 slices) |
|---|--:|--:|
| Total tok/s | 1,547 | about 54,960 (8 x 6,870) |
| Sessions at once | 7 | 240 at matched config (8 x 30); 696 at `--max-num-seqs 128` (8 x 87) |
| $/hr, public on-demand | $3.004 | $63.30 |

Matching one p5en's throughput takes about **36 g6e.4xlarge instances**, which cost **$108/hr against $63/hr**. Matching its session capacity takes about 34, which agrees. One node to patch, watch and load-balance replaces about 35.

> Treat this as a **projection, not a measurement**. It assumes the eight TP=1 replicas scale in a straight line. Host contention (PCIe, CPU for eight servers, NIC) went unmeasured, and eight replicas need a load balancer in front. A full-box replica sweep would settle it.

## The H200's ceiling, measured

The arm above inherited `--max-num-seqs 32` and so never found a wall. A second arm raised that one flag to **128** and swept c=30/40/50/60/80/100, 600 s per level, everything else byte-identical.

| c | gen tok/s | prompt tok/s | total | running (peak / mean) | waiting mean | KV peak | TTFT mean | TPOT mean | $/1M |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 30 | 404.63 | 3,762 | 4,167 | 30 / 23.73 | 1.10 | 0.418 | 16.7 s | 55.5 ms | $0.527 |
| 40 | 368.22 | 5,255 | 5,623 | 40 / 31.41 | 1.66 | 0.522 | 17.8 s | 92.5 ms | $0.391 |
| 50 | 427.96 | 6,239 | 6,667 | 50 / 38.68 | 2.62 | 0.663 | 23.0 s | 91.5 ms | $0.330 |
| 60 | **453.69** | 7,943 | 8,396 | 60 / 45.68 | 3.83 | 0.772 | 26.1 s | 99.8 ms | $0.262 |
| 80 | 400.68 | 8,935 | **9,336** | 80 / 59.01 | 6.98 | **1.000** | 40.3 s | 140.2 ms | **$0.235** |
| 100 | 446.05 | 6,661 | 7,107 | **87** / 61.69 | **20.81** | **1.000** | 88.4 s | 167.7 ms | $0.309 |

**The admission ceiling is 87 sessions.** At c=100 the H200 held 87 and queued the other 13, with KV pinned at 1.000 and `waiting` mean jumping from 6.98 to 20.81. That is the same signature the L40S showed at 7: running plateaus below the offered load, waiting climbs, KV saturates. So one H200 admits **12x** the sessions of one L40S.

The three ceilings landed far apart, which is why a capacity plan needs to name which one it means:

1. **Max sustained generation: 454 tok/s at c=60.** Beyond that it falls (400.68 at c=80).
2. **Max total throughput: 9,336 tok/s at c=80**, driven by prefill, which keeps rising after generation peaks.
3. **Max sessions under a 10 s TTFT budget: none of these levels.** Even c=30 costs 16.7 s. The interactive operating point stays in the first arm, at `--max-num-seqs 32` and c=30 (9.2 s).

Raising `--max-num-seqs` buys aggregate throughput and cuts cost per token to **$0.235/1M**, 56% under the L40S's $0.539, and $1.90/task against $4.34, at a first-token wait no interactive user would accept. That trade suits batch and overnight work rather than a developer-facing endpoint.

> [!WARNING]
> **The c=30 replication control did not reproduce, so the two arms do not splice into one curve.** Run at the same offered concurrency, the `--max-num-seqs 128` arm returned 404.63 gen tok/s and 3,762 prompt tok/s against the first arm's 356.72 and 6,513: generation up 13%, prefill down 42%, total down 39%, TTFT mean up from 9.2 s to 16.7 s. Resident sequences matched closely (mean 23.73 against 23.86, KV peak 0.418 against 0.408), so the scheduler is dividing the same 8,192-token batch budget differently, favouring decode steps over prefill. Read each arm on its own terms. Every headline figure in this doc comes from the first arm, where the only difference from the L40S is the GPU.

## Caveats

- **One sweep per level, 600 s windows.** Levels wobble: the H200's c=5 (230.19) and c=7 (228.06) sit inside each other's noise, because each level draws tasks at random and the agent's client-side duty cycle varies. Trust the curve, not a single level delta under about 5%.
- **The matched-config arm never reached the H200's wall.** At c=30 it held 41% KV with every session admitted, capped by the inherited `--max-num-seqs 32`, so read its 356.72 tok/s and 30 sessions as floors. A second arm raised that flag and found the wall at 87 sessions: see [the H200's ceiling, measured](#the-h200s-ceiling-measured).
- **Levels above c=25 repeat tasks.** The dataset holds 25, so c=30 runs 5 duplicates and c=100 runs about 4 copies of each task, always with distinct clones and slot ids but the same task text. [`vllm-serve.sh`](../self-hosted/vllm/scripts/vllm-serve.sh) passes `--enable-prefix-caching` on every arm, so shared prompt prefixes can flatter those levels. Each session explores its own clone and diverges early, so the effect should be small at c=30 and grows with concurrency, which is one more reason to read the c=80 and c=100 figures as approximate. No level at or below c=25 is affected.
- **The L40S baseline kept only its summary.** Its directory holds `performance-summary.json` and the dashboard, with no per-level session JSONs or DuckDB, so its $/task figures come from its measured blended $/1M against the same nominal 8M-in / 50K-out shape the H200 arm used. That keeps the cost comparison fair and leaves the session-level checks below one-sided.
- **Many sessions end in `error`, and the throughput figures survive it.** The matched H200 arm logged 28% errors at c=30 (16 of 57) and 57-75% at c=1 to c=7; the ceiling arm fell from 23% at c=30 to 0% at c=100. The rate drops as concurrency rises because the 600 s window cuts sessions off before they run long enough to hit an agent turn or token cap, which points at client-side caps rather than server failures. Errored sessions still generated tokens (median 3,143 output tokens for the matched arm's c=30 errors), and throughput comes from vLLM's own `generation_tokens_total` counter rather than from completed sessions, so the tok/s figures count that work. Read every arm as a measurement of **server load**, not of task success. Quality belongs to [the `/swe` benchmark](agentic-coding-throughput-comparison.md), and the L40S baseline kept no session records to compare error rates against.
- **No discount is modelled, on either box.** Every cost above is public on-demand, us-east-1, as of 2026-08-11. A real fleet would sit under a commitment and pay roughly 43% of these rates. Applying the same discount to both boxes rescales every dollar figure and changes no ratio; applying different discounts to each is the case that could move the conclusion, and it is out of scope here.

## How to reproduce

```bash
# 1. Serve on ONE GPU. On a p5en, pin it so vLLM cannot recruit idle cards.
cd self-hosted/vllm/scripts
CUDA_VISIBLE_DEVICES=0 MODEL="<YOUR_MODEL_HF_ID>" SERVED_NAME="<your-model-slug>" \
  TP=1 MAX_MODEL_LEN=65536 GPU_MEM_UTIL=0.90 TOOL_PARSER="<tool_parser_for_your_model>" \
  EXTRA_ARGS="--max-num-seqs 32 --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
              --chat-template <path/to/chat-template.jinja>" \
  ./vllm-serve.sh

# 2. Sweep. The default out-dir is throughput/{model}, which IS this arm (the
#    bare slug names the canonical p5en sweep), so no --out-dir is needed here.
cd ..
./scripts/run-throughput-sweep.sh --model <your-model-slug> \
  --out-dir benchmark-output/throughput/<your-model-slug> \
  --concurrencies "1 2 5 7 10 15 20 30" --duration-seconds 600 --context-window 65536

# 3. Summarise. --tp 1 prorates a partial box to a single-GPU price.
uv run python -m clients.build_performance_summary --model <your-model-slug> \
  --db benchmark-output/throughput/<your-model-slug>/throughput-metrics.duckdb \
  --instance-type p5en.48xlarge --tp 1 \
  --input-tokens-per-task 8000000 --output-tokens-per-task 50000 \
  --out benchmark-output/throughput/<your-model-slug>/performance-summary.json
uv run python -m clients.build_performance_dashboard \
  --summary benchmark-output/throughput/<your-model-slug>/performance-summary.json
```

For the ceiling hunt, change one flag and one directory: `--max-num-seqs 128`, `--concurrencies "30 40 50 60 80 100"`, and `--out-dir .../your-model-slug-seqs128`. Keep `--model <your-model-slug>` to match the served name, and pass `--out-dir` for every arm that is *not* the canonical one: the default is `throughput/{model}`, which holds the p5en single-H200 baseline, so an unsuffixed re-run overwrites it. The L40S baseline lives in the `-g6e` sibling for the same reason.

## Source data

### The three-model confirmation

Each row is a whole 4-GPU g6e.12xlarge against one H200 of a p5en, same model and same serving flags on both sides -- only the GPU and the GPU count change.

| Model | g6e.12xlarge (4x L40S, TP=4) | p5en slice (1x H200, TP=1) |
|---|---|---|
| 35B dense | run locally to generate | run locally to generate |
| 31B dense | run locally to generate | run locally to generate |
| 30B dense (coder) | run locally to generate | run locally to generate |

> The three H200 arms ran on a p5en.48xlarge (8x H200 DLAMI) with `CUDA_VISIBLE_DEVICES=0`, at a 200,000-token context window, `GPU_MEM_UTIL=0.90`, no `--max-num-seqs` override, 600 s per level, concurrency 1/2/5/7/10/15/20. Each g6e side carries over its own committed sweep and its own per-task token shape, so `$/task` is comparable arm to arm but not model to model. 30B dense (coder) stopped at c=15 because its throughput had gone flat for two consecutive levels. 35B dense was still climbing at c=20, so its figure is a floor. Costs here are rescaled from each summary's configured rate to public on-demand ($10.493/hr for g6e.12xlarge, $7.912/hr for one of eight H200s), which is why they differ from the cost fields inside the JSON.
>
> **Concurrency counts sessions, not requests.** An agentic session issues more than one API call at a time, so `requests_running` runs ahead of the offered concurrency: 35B dense peaked at 43 running requests at c=20 and at 11 at c=1. Every "N sessions" figure above is the number of `/swe` sessions the harness held in flight, which is the number that maps to developers.

### The 27B dense (FP8) illustration

Every number in the rest of this doc comes from these three summaries.

| Arm | Instance | Config | Summary | Dashboard |
|---|---|---|---|---|
| L40S baseline | g6e.4xlarge, TP=1 | `--max-num-seqs 32` | run locally to generate | run locally to generate |
| H200 slice, matched | p5en.48xlarge, TP=1 | `--max-num-seqs 32` | run locally to generate | run locally to generate |
| H200 slice, ceiling hunt | p5en.48xlarge, TP=1 | `--max-num-seqs 128` | run locally to generate | run locally to generate |

> Both H200 arms ran on p5en.48xlarge (8x H200 DLAMI, GPU 0 only), 600 s per level, 65,536 context: the matched arm at c=1 to 30, the ceiling arm at c=30 to 100. Rates are public on-demand, us-east-1: p5en.48xlarge $63.296/hr for the full box, prorated to $7.912/hr for one of eight GPUs, and g6e.4xlarge $3.004/hr. The committed summaries and dashboards use the configured `dollars_per_hour` instead -- the 3-year commitment rate for both families -- so their cost fields are ~43% of the ones here. Re-running either sweep regenerates that arm's summary and dashboard. Update the tables here when the runs change.
