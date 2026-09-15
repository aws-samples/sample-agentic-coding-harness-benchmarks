# End-to-end test with a self-hosted vLLM model

This is the full run-book for benchmarking a **self-hosted open-weight model** (Path 3) from a cold start to scored results: serve the model, capture a live GPU metrics time series into DuckDB, run the SWE benchmark against the `mcp-gateway-registry` dataset, and score the artifacts with the codex judge.

It stitches together three components, each documented on its own elsewhere; this page is the ordered checklist that runs them together:

- [Path 3 - self-hosted vLLM](path-self-hosted-vllm.md) and [self-hosted/vllm/README.md](../../self-hosted/vllm/README.md) - serving the model.
- [harness reference](harness-reference.md) - the runner config, the benchmark harness, and the judge.

The example serves **Qwen3.6-35B-A3B**; swap in any model from [self-hosted/vllm/models/](../../self-hosted/vllm/models/) that fits your GPU. All commands assume the reference node (g6e.12xlarge, 4x L40S, 184 GB) and are run from the repo root unless noted.

> **One command instead of the manual steps.** Once the vLLM server is up (Step 1) and, optionally, the metrics collector is running (Step 2), the pre-flight checks, the benchmark run, and the judge (Steps 0, 3, 4) are wrapped by a single orchestrator that fails loudly at the first problem and prints the tail/status command for each long-running step:
> ```bash
> cd benchmarks
> ./scripts/run-e2e-benchmark.sh --provider vllm --model qwen3.6-35b \
>     --dataset dataset/mcp-gateway-registry.yaml --yes
> ```
> The `/benchmark` skill drives the same script interactively. The manual steps below are the run-book that script automates -- read them to understand each stage, or to run a stage on its own. The orchestrator does **not** start the vLLM server or the collector (Steps 1-2); those are long-lived services you bring up first.

## Prerequisites

- The vLLM server dependencies installed (`~/vllm-env` with the `vllm` CLI) - see [self-hosted/vllm/README.md](../../self-hosted/vllm/README.md) or run the `/vllm-setup` skill.
- The benchmark harness environment set up once:
  ```bash
  cd benchmarks
  uv sync
  cp config/runner.example.yaml config/runner.yaml   # first time only
  ```

## Step 0 - Pre-flight checks

Run these before serving anything. The most important one is the **artifact-folder check**: the harness drives the `/swe` skill non-interactively, and the skill **stops and asks what to do if the target `{model}/` folder already contains any of the four artifacts** (see [SKILL.md](../../.claude/skills/swe/SKILL.md), "Handle an existing benchmark folder"). In a headless run there is nobody to answer that prompt, so a pre-existing folder makes the run stall or the model improvise. Clear (or move) any prior run for this exact `{model}` before starting.

**Check whether target folders already exist.** The harness writes to `swe-benchmark-data/{model-slug}/{harness}/mcp-gateway-registry/{task}/`, one folder per task. For the example (`--model qwen3.6-35b`, so the slug is `qwen3.6-35b`):

```bash
cd benchmarks
MODEL_SLUG=qwen3.6-35b
DATASET_REPO=mcp-gateway-registry
found=0
for task in remove-faiss remove-efs-from-terraform-aws-ecs \
            ssrf-hardening-outbound-url-validation \
            migrate-ecs-env-vars-to-secrets-manager \
            replace-keycloak-db-password-with-rds-iam; do
    dir="swe-benchmark-data/$MODEL_SLUG/$DATASET_REPO/$task"
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "EXISTS (needs clearing): $dir"
        found=1
    fi
done
[ "$found" -eq 0 ] && echo "OK: no existing $MODEL_SLUG artifact folders; safe to run"
```

> The `{model-slug}` is the folder name the skill uses, which is **not** always the same string you pass to `--model`. For a Bedrock inference profile the harness strips the vendor/region prefix and any `[...]` suffix (e.g. `us.anthropic.claude-opus-4-8` -> `claude-opus-4-8`); for a self-hosted served name like `qwen3.6-35b` the slug is identical. See the `model` row in the [runner-config table](harness-reference.md#the-runner-config).

**Clear them if the check reported any** (removes only this model's folders for these tasks; sibling model folders and other tasks are left untouched):

```bash
cd benchmarks
MODEL_SLUG=qwen3.6-35b
DATASET_REPO=mcp-gateway-registry
for task in remove-faiss remove-efs-from-terraform-aws-ecs \
            ssrf-hardening-outbound-url-validation \
            migrate-ecs-env-vars-to-secrets-manager \
            replace-keycloak-db-password-with-rds-iam; do
    rm -rf "swe-benchmark-data/$MODEL_SLUG/$DATASET_REPO/$task"
done
echo "cleared any prior $MODEL_SLUG folders"
```

If instead you want to **keep** a prior run, rename its folder (e.g. `qwen3.6-35b` -> `qwen3.6-35b-run1`) so the fresh pass writes to a clean `qwen3.6-35b`.

**Other pre-flight checks:**

- **GPUs are free** (or only holding a server you intend to replace): `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv`.
- **Port 8000 is free** (nothing already bound): `curl -s -m 2 http://127.0.0.1:8000/health && echo "  <- something is already serving :8000"`. If a server is already up with the model you want, you can skip Step 1.
- **The harness config exists**: `test -f config/runner.yaml && echo "runner.yaml present" || echo "run: cp config/runner.example.yaml config/runner.yaml"`.
- **The judge can actually reach Bedrock** (Step 4 runs codex against it). `aws sts get-caller-identity` proves the box can reach AWS but NOT that codex is pointed at Bedrock -- an unconfigured codex ignores those credentials and 401s against `api.openai.com`. Prove it end to end:
  `codex exec --skip-git-repo-check "Reply with exactly: JUDGE OK"`
  If that fails, see [agent-cli-bedrock-setup.md](agent-cli-bedrock-setup.md).

## Step 1 - Start vLLM serving the model of interest

Serve the model on `127.0.0.1:8000` with tensor parallelism across all four GPUs, at the **maximum context window this node can serve**. Per [its model guide](../../self-hosted/vllm/models/qwen3.6-35b-a3b.md), that is `MAX_MODEL_LEN=200000` (200K): the model is 256K-native, but the guide warns the full 256K "would consume so much KV cache it may not boot at useful concurrency" on 4x L40S, and extending past 256K with YaRN is "academic on this node - 4x L40S has nowhere near the VRAM." 200K sits just under native, so no rope scaling is needed, while leaving a little KV-cache headroom. It is a hard ceiling you must set explicitly (it does not auto-expand to native, and it is far above the script's own 32768 default):

```bash
cd self-hosted/vllm/scripts
MODEL="Qwen/Qwen3.6-35B-A3B" \
SERVED_NAME="qwen3.6-35b" \
TP=4 \
PORT=8000 \
MAX_MODEL_LEN=200000 \
GPU_MEM_UTIL=0.90 \
TOOL_PARSER="qwen3_coder" \
  ./vllm-serve.sh
```

The first boot downloads the weights (~72 GB for this model) and can take several minutes; subsequent boots with the weights cached are much faster. Wait until the server reports it is listening on `127.0.0.1:8000`.

> **If it OOMs or logs `Maximum concurrency ... 1x` at boot,** the KV cache for 200K did not fit at useful concurrency on your VRAM -- lower the window (e.g. `MAX_MODEL_LEN=131072` or `65536`) and re-serve. VRAM, not the model's native window, is the real ceiling on 4x L40S.

Confirm it is up and can do a tool-call-capable chat completion (the `/swe` run depends on tool calls working):

```bash
# health
curl -s http://127.0.0.1:8000/health && echo "  <- healthy"

# which model is served
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool

# quick inference smoke test
curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.6-35b","max_tokens":16,
         "messages":[{"role":"user","content":"Reply with exactly: OK"}]}' \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

> **To serve a different model,** pick its guide from [self-hosted/vllm/models/](../../self-hosted/vllm/models/) (each states whether it fits 4x L40S and its correct `TOOL_PARSER`), set `MODEL`/`SERVED_NAME` accordingly, and use the `served-model-name` you chose as `--model` in Step 3. Note the parser differs by family: the Qwen *Coder*/3.6 MoE models use `qwen3_coder`; the dense `Qwen3-32B` uses `hermes`.

## Step 2 - Clear the DuckDB database and start the metrics collector

The collector scrapes vLLM's Prometheus `/metrics` endpoint on a fixed interval and stores every `vllm:*` sample in `benchmark-output/vllm-metrics.duckdb`, giving an independent, continuous GPU time series that stays active for the whole benchmark (independent of the harness's own per-run snapshots). Start each end-to-end test from an empty database so the time series covers only this run.

**Clear the database.** The collector recreates its tables on start (`CREATE TABLE IF NOT EXISTS`), so emptying the base tables and reclaiming space is safe; the schema and views survive:

```bash
cd self-hosted/vllm
uv run python - <<'PY'
import duckdb, os
db = "benchmark-output/vllm-metrics.duckdb"
if os.path.exists(db):
    con = duckdb.connect(db)
    for t in ("metric_samples", "metric_scrapes", "collector_sessions"):
        con.execute(f'DELETE FROM "{t}"')
    con.execute("DROP SEQUENCE IF EXISTS metric_scrape_id_seq")
    con.execute("CREATE SEQUENCE metric_scrape_id_seq START 1")
    con.execute("CHECKPOINT")
    con.execute("VACUUM")
    con.close()
    print(f"cleared {db} ({os.path.getsize(db):,} bytes)")
else:
    print("no database yet; the collector will create it on start")
PY
```

(Alternatively, just delete the file: `rm -f benchmark-output/vllm-metrics.duckdb` - the collector recreates it from scratch. Clearing in place preserves the file's location and permissions.)

**Start the collector** (backgrounded; default one-second interval, default DB path):

```bash
cd self-hosted/vllm/scripts
./vllm-metrics.sh start
./vllm-metrics.sh status   # confirm it is running
```

**Confirm the database is growing.** Row counts should climb every second while the collector runs:

```bash
cd self-hosted/vllm
uv run python - <<'PY'
import duckdb
con = duckdb.connect("benchmark-output/vllm-metrics.duckdb", read_only=True)
print("scrapes:", con.execute("SELECT count(*) FROM metric_scrapes").fetchone()[0])
print("samples:", con.execute("SELECT count(*) FROM metric_samples").fetchone()[0])
con.close()
PY
sleep 5
# run the same snippet again and confirm both numbers have increased
```

The collector keeps running in the background across the whole benchmark. Leave it up until Step 3 finishes, then stop it (Step 5).

## Step 3 - Run the SWE benchmark against mcp-gateway-registry

Drive the harness through the `endpoint` provider, pointed at the local vLLM server, using the `served-model-name` from Step 1 as `--model`. Start with a single task to confirm the whole path (serve -> tool calls -> artifacts) before committing to the full dataset:

```bash
cd benchmarks

# One-task confirmation, with a live trace
uv run scripts/run-swe-headless.py --config config/runner.yaml \
    --provider endpoint --endpoint http://127.0.0.1:8000 \
    --model qwen3.6-35b \
    --dataset dataset/mcp-gateway-registry.yaml --count 1 --stream

# Full reference dataset (all 5 tasks)
uv run scripts/run-swe-headless.py --config config/runner.yaml \
    --provider endpoint --endpoint http://127.0.0.1:8000 \
    --model qwen3.6-35b \
    --dataset dataset/mcp-gateway-registry.yaml
```

Each task lands its four artifacts plus `metrics.json` under `swe-benchmark-data/qwen3.6-35b/mcp-gateway-registry/{task}/`. Because this is the `endpoint` path against vLLM, `metrics.json` also carries the populated `vllm_prometheus` block (prefix-cache hit rate, per-run token/latency deltas, in-flight KV-cache peak). Keep `concurrency: 1` (the default) if you want trustworthy per-run vLLM cache numbers; see [Running tasks concurrently](harness-reference.md#running-tasks-concurrently) for the trade-off. Full flag reference: [Common invocations](harness-reference.md#common-invocations).

## Step 4 - Score the artifacts with the codex judge (LLM-as-judge)

The judge reads the four artifacts a run produced, checks their factual claims against the actual repository (checked out read-only), scores them against the [rubric](harness-reference.md#the-rubric), and writes `eval.json` beside them (mirrored into `metrics.json` under `evaluation`). Score every folder the benchmark just produced in one batch:

```bash
cd benchmarks/scripts

# Score every task this model just produced; skip any that already have an eval.json
uv run python codex_judge.py --recursive --no-overwrite \
    --folder ../swe-benchmark-data/qwen3.6-35b
```

To score a single task folder instead:

```bash
cd benchmarks/scripts
uv run python codex_judge.py \
    --folder ../swe-benchmark-data/qwen3.6-35b/mcp-gateway-registry/remove-faiss
```

`codex exec` buffers and prints only its final message, so a multi-minute run at the default `high` reasoning effort looks idle while it is really working - give it a few minutes per folder. Flags, model overrides, and the eval schema are documented in [Running the codex judge](harness-reference.md#running-the-codex-judge).

## Step 5 - Wrap up

Stop the metrics collector once the benchmark and judging are done:

```bash
cd self-hosted/vllm/scripts
./vllm-metrics.sh stop
```

Optionally render the collected GPU time series to a self-contained HTML dashboard, and leave the vLLM server running (or stop it) as you like:

```bash
cd self-hosted/vllm
uv run python -m clients.build_dashboard \
    --db benchmark-output/vllm-metrics.duckdb \
    --output benchmark-output/dashboard.html
```

At this point each `swe-benchmark-data/qwen3.6-35b/mcp-gateway-registry/{task}/` folder holds the artifacts, a `metrics.json` (cost + vLLM server metrics + the mirrored evaluation), and an `eval.json` (quality scores) - directly comparable to the same task run by any other model on any of the three paths.
