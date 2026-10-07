# Does the swe-router pay for itself?

> Across the 21 tasks the router selected **3 different models**: **Open dense 27B** 13x, **Closed frontier (gen 3)** 3x, **Open MoE 2.8T / 104B active** 1x. On 4 further task(s) nothing cleared the floor, so the skill's answer was to stay on the baseline.
>
> Against running the **Closed frontier (gen 3)** model on everything, that cost **45.4% less** ($137.16 against $251.04) for a quality change of **-4.6%** (78.99 against 82.83 mean task score, -3.84 points).
>
> The saving is total-over-total, which is what lands on a bill. The mean of the per-task percentages is 60.6%, higher because it weights a cheap task the same as an expensive one.

Replays the `swe-router` skill over all 21 tasks of `mcp-gateway-registry-v2`, then looks up what the model it picked ACTUALLY scored and cost on that task, against running the Closed frontier (gen 3) model on everything.

- **Sampling.** Leave-one-out: each task routes from tier means recomputed with that task excluded, so no pick knows the run it is scored against.
- **Floor.** Judged per task by omp running the skill's step 1 against the cloned repo, driven by the same Closed frontier (gen 3) model used as the baseline -- the real judgment the skill asks for, not a policy constant. The floor and tier each task was judged to need are in the table below.
- **Tier.** Classified per task by the same judged run, NOT read from the dataset. Each row carries the dataset's own `complexity` label beside it so disagreement is visible.
- **Candidates.** 18 models the developer could select, with the organisational allow-list ignored (`--no-allow-list`). The full candidate set, with the measured score and cost behind each one, is the committed [`vend/swe-router/models.json`](../vend/swe-router/models.json).
- **Cost basis.** Metered provider bills for Bedrock models; hardware-derived ($/token from the throughput sweep x tokens the server processed) for self-hosted ones. Mixing the two on one axis is directional -- see [cost-per-task-methodology.md](cost-per-task-methodology.md).
- **Scoring.** `task_score` from the repo-grounded `openai.gpt-5.6-sol` judge. One run per task, so a per-task gap under ~3 points is noise.
- **Runs.** omp harness, /swe3, measured 2026-09-08.

A ⚠ marks a task where the model the router picked landed below the floor it was chosen to clear. That is the router getting it wrong, and the totals count it.

## Judged floors and tiers

| Task | Tier | Floor | Router pick | Predicted | Actual | Baseline | Δ score | Cost | Baseline cost | Saving |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| default-create-in-idp-checkbox-unchecked | low | 80 | Open dense 27B | 80.70 | 81.8 | 90.6 | -8.8 | $0.54 | $5.53 | +90% |
| build-docker-images-from-uv-lock | medium | 75 | Open dense 27B | 81.32 | 65.6 ⚠ | 83.2 | -17.6 | $0.50 | $12.00 | +96% |
| honor-cloud-provider-override-in-ui | low | 70 | Open dense 27B | 80.05 | 84.4 | 88.4 | -4.0 | $0.58 | $4.81 | +88% |
| fix-reserved-groups-var-in-service-account-script | low | 80 | Open dense 27B | 82.15 | 76.0 ⚠ | 87.6 | -11.6 | $0.26 | $5.35 | +95% |
| cli-custom-egress-oauth-provider-flags | low | 80 | Open MoE 2.8T / 104B active | 83.50 | 87.2 | 91.2 | -4.0 | $1.96 | $8.57 | +77% |
| derive-repo-url-from-skill-md | medium | 75 | Open dense 27B | 78.80 | 78.2 | 78.2 | +0.0 | $0.71 | $11.29 | +94% |
| consistent-csrf-across-toggle-endpoints | high | 80 | _stay on the baseline_ | -- | 86.4 | 86.4 | +0.0 | $10.94 | $10.94 | +0% |
| configurable-ui-title | medium | 75 | Open dense 27B | 77.72 | 83.6 | 83.2 | +0.4 | $0.84 | $10.35 | +92% |
| configurable-mcp-proxy-upstream-timeout | medium | 70 | Open dense 27B | 77.76 | 83.4 | 79.2 | +4.2 | $0.74 | $10.12 | +93% |
| nginx-location-trailing-slash-route-hijack | medium | 80 | Closed frontier (gen 3) | 82.04 | 80.4 | 80.4 | +0.0 | $9.22 | $9.22 | -0% |
| registration-admission-control-gate | high | 80 | _stay on the baseline_ | -- | 81.8 | 81.8 | +0.0 | $13.29 | $13.29 | +0% |
| server-side-oauth-token-storage | high | 75 | Open dense 27B | 77.27 | 54.0 ⚠ | 81.2 | -27.2 | $2.54 | $24.32 | +90% |
| lifecycle-workflow-webhooks | high | 80 | Closed frontier (gen 3) | 80.25 | 77.8 ⚠ | 77.8 | +0.0 | $31.96 | $31.96 | -0% |
| per-caller-per-target-rate-limits-and-quarantine | high | 80 | _stay on the baseline_ | -- | 81.0 | 81.0 | +0.0 | $31.69 | $31.69 | +0% |
| idp-authenticated-embedding-endpoint | high | 80 | Closed frontier (gen 3) | 80.45 | 77.0 ⚠ | 77.0 | +0.0 | $18.10 | $18.10 | -0% |
| index-demo-videos-in-one-page | low | 70 | Open dense 27B | 80.92 | 83.0 | 87.0 | -4.0 | $0.36 | $6.60 | +94% |
| hide-register-button-on-virtual-and-skills-tabs | trivial | 70 | Open dense 27B | 81.05 | 82.8 | 86.4 | -3.6 | $0.44 | $3.98 | +89% |
| pass-ssrf-allowlist-env-to-registry-container | trivial | 75 | Open dense 27B | 79.20 | 90.2 | 89.6 | +0.6 | $0.61 | $6.35 | +90% |
| macos-setup-python-version-precheck | low | 70 | Open dense 27B | 80.92 | 74.0 | 80.2 | -6.2 | $0.45 | $4.08 | +89% |
| portable-env-secret-generation-in-build-script | low | 80 | Open dense 27B | 80.92 | 77.0 ⚠ | 75.8 | +1.2 | $0.46 | $11.52 | +96% |
| logout-id-token-hint-out-of-browser-url | high | 80 | _stay on the baseline_ | -- | 73.2 ⚠ | 73.2 | +0.0 | $10.97 | $10.97 | +0% |

**Totals over 21 tasks** (14 switched away from the baseline)

| | Router | Baseline | Difference |
|---|---:|---:|---:|
| Total cost | $137.16 | $251.04 | **-$113.88 (45.4%)** |
| Mean score (21 tasks scored in both arms) | 78.99 | 82.83 | **-3.84** |
| Tasks under floor | 7 | 4 | +3 |
| Tasks failed outright | 0 | 0 | +0 |

Models the router used: Closed frontier (gen 3), Open MoE 2.8T / 104B active, Open dense 27B.
