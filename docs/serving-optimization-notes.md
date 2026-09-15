# vLLM serving optimization notes

A running log of serving-configuration findings for the self-hosted (vLLM) path. Append new entries at the top with a date and the model/hardware they came from. The goal is a small set of **portable, common-sense defaults** we do not have to re-tune per model, plus a record of *why* — so the next model does not repeat the investigation.

> Companion doc: [cost-per-task-methodology.md](cost-per-task-methodology.md) — how the fixed instance price becomes a cost per token / per task, the two cost lenses, and why the prefill-bound shape found below points to horizontal scaling.

## TL;DR — the portable defaults (do not tune per model)

| Knob | Setting | Why it generalizes |
|---|---|---|
| `--enable-chunked-prefill` | **on** (vLLM V1 default) | Interleaves large prefills with decode so one giant prompt cannot freeze generation. Universal win; never turn off. |
| `--max-num-batched-tokens` | **unset** (V1 auto) | V1 derives it from KV-cache size and the model. It self-scales per hardware/model — hand-setting it is what *creates* a per-model chore. |
| `--max-num-seqs` | **unset** | Only set it for the specific boot-time "can't fit N sequences in KV cache" error (`vllm-serve.sh` documents this). That is a correctness fix, not a throughput tune. |
| `--enable-prefix-caching` | **on** | Free KV reuse when prompts share a prefix. Helps a lot in single-repo/single-user workloads; helps little across diverse repos (see below) — but never hurts. |

**We leave `vllm-serve.sh` on these defaults.** Full config resolved at boot confirms the two prefill knobs are already `enable_chunked_prefill=True` and `max_num_batched_tokens` auto.

---

## 2026-07-26 — CORRECTION: the "prefill-saturated" run below was a backed-up server, not the model

A clean re-run of the same sweep (same model, hardware, dataset, windows) — this time with a **c=1 baseline** and **TTFT reported as p50/p90 with a queue-vs-prefill split** — showed the server is **healthy**, not saturated:

- TTFT p50 **~1-2s uncontended, ~5-8s at c=20** (not minutes); prefill mean a flat ~2-4s; queue-wait p50 ~0s until c=7, ~2s by c=20.
- Generation throughput ~90-145 tok/s, prompt ~9-13K tok/s, KV never saturated, zero preemptions.
- At c=10 the clean run completed **171 requests** in the window vs **23** in the run below — the earlier run's requests were stuck in a **stale scheduler backlog** (queue mean 135s vs 5.5s).

So the entry below correctly ruled out KV pressure, but its "prefill saturation / 2-4 minute TTFT" conclusion was measuring a **transient backed-up server state**, not the workload's true behavior. **Lesson baked into the tooling:** always run a **c=1 baseline** and watch the **queue-vs-prefill decomposition** — if c=1 TTFT is not small, or queue-wait dominates prefill at low concurrency, the server is not clean and the run must be discarded. See [cost-per-task-methodology.md](cost-per-task-methodology.md) for the corrected curve. The portable serving defaults (top of this file) are unchanged and remain correct.

---

## 2026-07-25 — Agentic coding is prefill-bound, not KV-bound (qwen3.6-35b on g6e.12xlarge, 4xL40S)

> **Superseded — see the 2026-07-26 correction above.** This investigation's KV-vs-prefill reasoning is sound, but its headline numbers came from a backed-up server and overstate the problem. Kept for the reasoning and the metric-artifact lesson.

**Context.** Running the throughput skill against qwen3.6-35b. Switched the load dataset from `mcp-gateway-registry` (5 tasks, ONE repo) to `multi-repo-throughput` (25 tasks, 25 DIFFERENT repos) and saw generation throughput drop sharply and *fall* with concurrency. Investigated whether it was KV-cache pressure. **It is not** — it is prefill (prompt-processing) saturation, and it is the true cost of this workload, not a mistuning.

### What the data showed

Single-repo run (RUN1) vs multi-repo run (RUN2), server-side counters from the DuckDB collector:

| | RUN1 single-repo | RUN2 multi-repo |
|---|---|---|
| gen tok/s @ c2 | 53.5 | 44.5 |
| gen tok/s @ c5 | 79.8 (rising) | 19.6 (collapsing) |
| prefix-cache hit % | 32-60% | ~14% |
| KV% peak @ c2 (same 2 sessions) | 28.7% | 62.6% |

Multi-repo, per concurrency level (RUN2):

| c | gen t/s | prompt t/s | prefill:decode | KV% peak | preempt | wait peak |
|--:|--:|--:|--:|--:|--:|--:|
| 2 | 44.5 | 5936 | 133:1 | 62.6 | 0 | 28 |
| 5 | 19.6 | 5568 | 283:1 | 48.0 | 0 | 35 |
| 7 | 16.4 | 4544 | 277:1 | 28.8 | 0 | 40 |

### Why it is NOT KV-cache pressure

- KV usage **falls** as concurrency rises (62 -> 48 -> 29%). Under KV pressure it would pin near 100%.
- **Zero preemptions, zero swaps** at every level. KV pressure shows up as preemptions.
- Requests wait for `reason="capacity"` (the scheduler's per-step prefill token budget), not for KV blocks.

### Why it IS prefill saturation

- prefill:decode token ratio is **133:1 at c2, rising to ~280:1** — the GPU spends ~99.5% of its time processing prompts (~5-6K prompt tok/s) and almost none generating (16-44 gen tok/s).
- Agentic coding has an extreme request shape: ~100K+ token read-heavy prompts, tiny outputs. Measured per-task ratio on qwen3.6-35b was **~50:1 input:output** (1.51M in : 30K out across 5 real /swe2 runs).
- The prefix cache had been *masking* this: in the single-repo run, 32-60% of each prompt's KV was already computed and reused, so the fixed prefill budget covered far more effective prompt and left GPU time for decode — gen throughput even *rose* with concurrency. Across 25 distinct repos the hit rate drops to ~14%, nearly every prompt token must be prefilled from scratch, and more concurrency just means more giant prompts contending for the same prefill budget -> gen tok/s *falls*.

### The takeaways

1. **The single-repo number was an optimistic best case.** N developers all in one repo with a warm shared cache is the friendliest possible workload, not a representative one. The **multi-repo number is the honest cost** of N developers on N different projects — which is what the cost model should price.
2. **No serving knob fixes this**, because it is not broken. No setting makes 25 distinct 100K-token prefills free. `max_num_batched_tokens` only changes how the fixed prefill work is packed per step; it cannot create FLOPs. Raising it trades latency for throughput but does not change the prefill-bound regime. This is a property of the **workload shape**, not the config.
3. **Do not headline the cost on generation tok/s.** At a ~50:1 input:output ratio it is a misleading denominator that makes a healthy, 99%-busy server look broken. **Headline the blended (per-processed-token) cost** — cost per (prompt + generation) token. That number was stable at ~5-8K processed tok/s across the whole concurrency sweep in *both* runs, because it measures the work the machine actually does. It is model- and hardware-portable by construction and needs zero per-model tuning. The lab-style split lens (input priced at `w` x output) understates cost for this workload and should stay a secondary view.

### Constraints honored

- **200K context window is fixed** (`MAX_MODEL_LEN=200000`) — not reduced. The bottleneck is prefill compute, not the window or KV size anyway, so lowering it would not have helped.
- No changes made to `vllm-serve.sh`; defaults are the recommended values.
