---
name: vllm-setup
description: "Stand up a vLLM inference server for an open-weight coding model on a multi-GPU EC2 node (reference: g6e.12xlarge, 4xL40S). Drives the full flow end to end — verify the GPU node, install vLLM and its OS/Python dependencies (including the two Deep Learning AMI-specific fixes), serve a model with tensor parallelism and tool calling, confirm inference works, and optionally install opencode to drive it as a coding agent. Use when the user wants to self-host a model with vLLM, get vLLM inference running on EC2, set up opencode against a local model, or reproduce the hosting-strategy throughput benchmark. Wraps the scripts in self-hosted/vllm/scripts/."
license: Apache-2.0
metadata:
  author: Amit Arora
  version: "1.0"
---

# vLLM Setup Skill

Use this skill to bring up a vLLM inference server for an open-weight coding model on a multi-GPU EC2 GPU node, and confirm it actually serves tokens. The install is heavy (apt packages, a multi-GB wheel, a ~57 GB model download, and two environment fixes specific to the Deep Learning AMI), so this skill drives the vetted scripts rather than having the user paste commands by hand.

**This skill runs ON the GPU instance**, not the user's laptop. The very first thing it must do is confirm it is on a GPU node (Step 1). If there is no GPU, stop and tell the user to run this on the EC2 instance instead.

All the underlying logic lives in [`self-hosted/vllm/scripts/`](../../../self-hosted/vllm/scripts/): `vllm-install.sh`, `vllm-serve.sh`, `vllm-verify.sh`, `opencode-setup.sh`. This skill orchestrates them, reports each step, confirms inference works, and optionally sets up opencode. The full architecture and the *why* behind every dependency is documented in [`self-hosted/vllm/README.md`](../../../self-hosted/vllm/README.md) — read it if the user asks what is being installed or why a step exists. Every script also takes `--help`.

> **Running on a p5en.48xlarge (8x H200 / NVSwitch)?** The scripts were verified on the reference g6e.12xlarge (4x L40S). An 8-GPU NVSwitch box needs extra one-time fixes the reference node never hits — NVIDIA driver alignment + reboot, putting the venv/weights on the large NVMe (the root disk is tiny), and two CUDA-JIT fixes at serve time (`ninja` on PATH, and unversioned `libcudart.so`/`libcuda.so` for FlashInfer's mnnvl allreduce link step). All of them, with copy-pasteable commands and a symptom→fix table, are in [`p5en-h200-cuda-fixes.md`](p5en-h200-cuda-fixes.md). Read it before Step 3/4 on that hardware; skip it entirely on the 4x L40S reference node.

## Workflow

1. **Confirm the node** — verify a GPU + capture specs; abort if not on a GPU box
2. **Confirm the model** — announce the default, let the user override
3. **Install** — run `vllm-install.sh`, report each layer
4. **Serve** — run `vllm-serve.sh` (tool calling ON) with the chosen model
5. **Verify** — run `vllm-verify.sh`, show the real inference round-trip
6. **Drive Claude Code** — verify Claude Code against the vLLM endpoint
7. **Drive another agent** — optionally offer opencode via `opencode-setup.sh` (idempotent)
8. **Report** — summarize, show how to monitor and stop

Keep the SSH-tunnel / client-connection steps for after inference is confirmed; the goal of this skill is a working local endpoint on the instance.

---

## Step 1 — Confirm this is the GPU node

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nproc && free -h | head -2 && df -h / | tail -1
```

- **No `nvidia-smi` / no GPU:** stop. Tell the user this skill must run on the EC2 GPU instance (e.g. `g6e.12xlarge`), not their laptop, and offer to help launch one.
- **GPU present:** report what was found — number and type of GPUs, total VRAM, vCPU, RAM, free disk. Confirm there is enough free disk for the model (a 30B model is ~57 GB on disk; warn if free space is under ~80 GB).

Also detect the instance type and AMI if metadata is reachable (best-effort):

```bash
TOKEN=$(curl -sf -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
curl -sf -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type
```

## Step 2 — Confirm the model

Announce the default and let the user override before installing:

> I'll serve **`Qwen/Qwen3-Coder-30B-A3B-Instruct`** (a 3B-active MoE coder model, ~61 GB in BF16) across all detected GPUs with tensor parallelism.
>
> Alternatives that fit a 4×L40S (184 GB) node: `Qwen/Qwen3-32B` (dense), `Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3-Coder-Next` (80B MoE — use a smaller `MAX_MODEL_LEN`). Want the default, or a different model?

Lock in `MODEL` and a short `SERVED_NAME` (e.g. `qwen3-coder-30b`) from the answer.

## Step 3 — Install vLLM and dependencies

From the repo's script directory:

```bash
cd self-hosted/vllm/scripts
./vllm-install.sh
```

`vllm-install.sh` is idempotent — it skips anything already present. As it runs, tell the user what each layer is for (the script prints headers; summarize them):

- **build-essential + python3.12-dev** — vLLM's Triton backend JIT-compiles a CUDA helper at startup that needs `gcc` and `<Python.h>`; the DLAMI lacks the Python headers by default. This is the #1 thing that breaks a naive install.
- **uv → `~/vllm-env` → vLLM** — an isolated venv so the install is disposable and does not touch the AMI's `/opt/pytorch` env.
- **nvtop + gpustat** — live GPU monitoring for watching the benchmark.

If the install fails, read the error, cross-reference the "two DLAMI-specific fixes" section of the README, and fix before proceeding. Do NOT continue to serving on a failed install.

## Step 4 — Serve the model

```bash
cd self-hosted/vllm/scripts
MODEL="<chosen>" SERVED_NAME="<chosen>" ./vllm-serve.sh
```

- Note the reference-node fixes the serve script applies automatically: `VLLM_USE_FLASHINFER_SAMPLER=0` (native sampler — avoids FlashInfer's runtime nvcc requirement against a `/usr/local/cuda` that does not exist on the DLAMI) and a `CUDA_HOME` fallback pointing at `/opt/pytorch/cuda`.
- **On a p5en.48xlarge (8x H200 / NVSwitch):** the serve script's automatic fixes are not enough — export the PATH and CUDA-link environment from [`p5en-h200-cuda-fixes.md`](p5en-h200-cuda-fixes.md) *before* running `vllm-serve.sh`, or startup fails at KV-cache init (`No such file or directory: 'ninja'`) and again after graph capture (`ld: cannot find -lcudart`). That file has the exact exports and a symptom→fix table.
- The script tees the full server log to `self-hosted/vllm/logs/vllm-serve.log` (gitignored) and polls until ready. First serve of a model downloads the weights (~57 GB for 30B) — this can take several minutes. Reassure the user; tail the log if they want to watch: `tail -f self-hosted/vllm/logs/vllm-serve.log`.
- If the process exits early, read the tail of that log for the root cause.

Once ready, surface the useful runtime numbers vLLM printed — especially the KV cache size and **"Maximum concurrency for N tokens per request"** line, since that concurrency figure is what the throughput/cost benchmark builds on.

Tool calling is ON by default (`--enable-auto-tool-choice --tool-call-parser qwen3_coder`), which agentic clients like opencode require. For a non-coder Qwen3 model use `TOOL_PARSER=hermes`; for a plain completion server use `TOOL_PARSER=none`.

## Step 5 — Verify inference

```bash
cd self-hosted/vllm/scripts
./vllm-verify.sh
```

Show the user the model's actual reply and the prompt/completion token counts. This proves the endpoint serves real tokens. Be explicit that the single-request tokens/sec here is **not** the throughput number — batched concurrency is much higher (cite the concurrency figure from Step 4).

## Step 6 — Drive Claude Code

Now that raw inference works, verify Claude Code itself against the vLLM endpoint:

```bash
cd self-hosted/vllm/scripts
./claude-local.sh -p "Reply with exactly: claude-vllm-ok"
```

The launcher uses a temporary settings file and sets `CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096`. This cap is required for the default 32K served context; Claude Code's default 32K output request can exceed vLLM's context limit after the prompt is included.

If this fails with a context-length error, lower `CLAUDE_MAX_OUTPUT_TOKENS` or restart vLLM with a larger `MAX_MODEL_LEN` plus the model-appropriate rope scaling.

## Step 7 — Drive a coding agent with opencode

Now that raw inference works, offer to wire up opencode so the user can run a real agentic session against the self-hosted model. Ask first:

> Inference works. Want me to set up **opencode** to drive this model as a coding agent?

If yes, run the idempotent setup script:

```bash
cd self-hosted/vllm/scripts
./opencode-setup.sh            # or --launch to drop straight into a session
```

The script:

- **Checks whether opencode is already installed and skips the install if so** (binary at `~/.opencode/bin/opencode`) — only installs when missing.
- Writes `~/.config/opencode/opencode.json` registering a custom OpenAI-compatible provider `vllm` → `http://localhost:8000/v1`, backing up any existing config.
- Confirms the vLLM server is reachable.

Then verify end to end with a real agentic call and show the user the reply:

```bash
export PATH="$HOME/.opencode/bin:$PATH"
opencode run "Reply with only: opencode-vllm-ok"
```

If it errors with *"auto tool choice requires --enable-auto-tool-choice"*, the server was started with `TOOL_PARSER=none` — restart it with the default parser (Step 4).

## Step 7 — Report

Summarize:

- node (instance type, GPUs, VRAM), model served, `served-model-name`, port
- that the OpenAI-compatible API is live at `http://127.0.0.1:8000/v1`
- Claude Code status (verified with `claude-local.sh`?)
- opencode status (installed?, provider wired, verified)
- how to monitor: `nvtop`, `~/vllm-env/bin/gpustat -i 1`
- how to stop: `./vllm-serve.sh --stop`
- how to reach it from a laptop: SSH tunnel (`scripts/tunnel.sh` with `LOCAL_MODEL_PORT=8000`)

---

## Notes for the operator

- **Idempotent / resumable:** every script skips work already done. Re-running the skill after a fix is safe.
- **One model at a time by default:** the default 30B fits with room to spare; a single replica with a large KV cache maximizes concurrency. Only serve a second model on a second port if VRAM allows.
- **Logs are never committed:** `self-hosted/vllm/logs/` is gitignored.
- **Precision is BF16 (unquantized)** by default — deliberate, to keep the benchmark an apples-to-apples quality comparison with full-precision APIs.
