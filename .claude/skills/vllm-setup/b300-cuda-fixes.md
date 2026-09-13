# Serving on a p6-b300.48xlarge (8x B300) - extra node fixes

> Companion note to [SKILL.md](SKILL.md), the B300 analog of [p5en-h200-cuda-fixes.md](p5en-h200-cuda-fixes.md). The vLLM scripts were verified on the reference **g6e.12xlarge (4x L40S)**. This repo's **p6-b300.48xlarge (8x B300 SXM6, NVSwitch, ~2.15 TiB VRAM)** node is an 8-GPU NVSwitch box like the p5en, so it hits the same *class* of multi-GPU startup issues - but B300 is `sm_100` (Blackwell Ultra), newer than the p5en's Hopper H200, and the first failure it hits is different. This file records every extra change needed here. If you are on the 4x L40S reference node you need none of this.

## Node baseline (what was already fine, 2026-09)

Unlike the p5en, this node needed **no driver realignment** and **no reboot**:

- `nvidia-smi` clean at driver **595.91.07**, 8x `NVIDIA B300 SXM6`, ~268 GiB each.
- `nvidia-fabricmanager` **active**, NVSwitch fabric **`State: Completed`**.
- vLLM **0.28.0** already installed at `~/vllm-env` and it registers `KimiK3ForConditionalGeneration`, `KimiLinearForCausalLM`, etc. (nccl 2.29.7, compressed-tensors 0.17.0).
- Root disk 484 GB (460 GB free) - big enough for the venv; the **27 TB ephemeral NVMe at `/opt/dlami/nvme`** holds HF weights (`vllm-serve.sh` defaults `HF_HOME=/opt/dlami/nvme/hf-cache`). The venv sits on root here (fits), unlike the p5en where it had to move to the NVMe.

So the p5en's Section 0 (driver realign) and the "move the venv to NVMe" step do **not** apply here. What does apply is the NCCL fix below.

## Fix 1 (root fix) - reset the GPUs to restore NVLS multicast

**Symptom.** Any TP>1 serve dies at engine-core init, *before* weights download, with all ranks reporting:

```
RuntimeError: NCCL error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)
```

at `ncclCommInitRank` (`pynccl_wrapper.py` -> `parallel_state.py` -> `cuda_communicator.py`). The generic message hides the cause; `NCCL_DEBUG=WARN` reveals it:

```
transport/nvls.cc:287 NCCL WARN Failed to bind NVLink SHARP (NVLS) Multicast memory of size 2097152 :
  CUDA error 401 'the operation cannot be performed in the present state'.
This is usually caused by a system or configuration error in the Fabric Manager or NVSwitches.
Disable NVLS (NCCL_NVLS_ENABLE=0) if you wish to avoid this error in the future.
```

**Cause (verified from the fabricmanager log).** The fabric is up (`State: Completed`), but the partition's **multicast team state is stuck** and needs a GPU reset to recover. `/var/log/fabricmanager.log` shows the multicast team setup failing outright:

```
[ERROR] cannot find exporter GPU in partition Id 57082 gpuHandle 0x0 to send Multicast Team Setup Response ...
[ERROR] All GPUs in the partition need to be reset to recover
[ERROR] failed to add multicast team with request ID ... in partition 57082.
```

A crashed NVLS-multicast attempt (or a prior tenant) left the multicast allocation corrupted in GPU/NVSwitch hardware; NCCL's `ncclCommInitRank` then can't bind multicast (`CUDA error 401`). This is **not** a missing capability - `nvidia-imex-595` (matching the driver) is installable, `FABRIC_MODE=0` with a default partition is correct, and multicast works fine once the state is cleared. A `systemctl restart nvidia-fabricmanager` alone does **not** clear it (the state is in hardware, not the FM daemon).

**Fix (verified).** Reset all GPUs to clear the stuck multicast state. GPUs must be idle (no processes) first:

```bash
sudo systemctl stop nvidia-fabricmanager
sudo nvidia-smi -r                         # resets all 8 GPUs; prints "was successfully reset" per GPU
sudo systemctl start nvidia-fabricmanager  # rebuilds the fabric; wait ~10 s for state "3 (configured)"
```

After this, NVLS multicast binds and Kimi-K3 (which *requires* multicast) loads. **Do NOT set `NCCL_NVLS_ENABLE=0` for K3** - that disables the very feature K3's kernel needs; the reset is the correct fix, and NVLS stays *enabled*.

> **`NCCL_NVLS_ENABLE=0` is only a fallback for models that do NOT require multicast** (Qwen, GLM, Kimi-K2.7, ...). If you cannot reset the GPUs and only need such a model, disabling NVLS lets NCCL fall back to standard NVLink all-reduce and boots. It does not help K3.

**Verify (cheap, no model, ~10 s) - a bare 8-GPU NCCL all-reduce with NVLS ENABLED:**

```bash
cat > /tmp/nccl_probe.py <<'PY'
import os, torch, torch.distributed as dist
rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world)
x=torch.ones(1024, device=f"cuda:{rank}"); dist.all_reduce(x); torch.cuda.synchronize()
if rank==0: print(f"NCCL OK: all_reduce sum={x[0].item()} across {world} GPUs")
dist.destroy_process_group()
PY
~/vllm-env/bin/torchrun --nproc_per_node=8 --nnodes=1 /tmp/nccl_probe.py
# expect: NCCL OK: all_reduce sum=8.0 across 8 GPUs   (with NVLS ENABLED, post-reset)
```

Before the reset this probe fails with the NVLS/CUDA-401 warning; after the reset it prints `NCCL OK` with NVLS left on. (The `NET/OFI` / `aws-ofi-nccl` RDMA warnings printed alongside are benign for single-node NVLink and are not the fatal error.)

## Why the reset matters for Kimi-K3 specifically (and why `NCCL_NVLS_ENABLE=0` is not enough)

Some models' vLLM kernels **mandate NVLS multicast** - you cannot work around a stuck fabric by disabling NVLS, you must reset (Fix 1). **Kimi-K3** is the example. Before the reset, with `NCCL_NVLS_ENABLE=0` set, NCCL comm init passed but model load then died with

```
File ".../vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py", line 892
RuntimeError: routed NVLS multicast mapping is unavailable
```

**Why K3 specifically.** K3 is a *latent-MoE* model: `use_latent_moe = routed_expert_hidden_size is not None`, and K3's config sets `routed_expert_hidden_size: 3584`, so the latent-MoE path is forced on with **no toggle**. Its `LatentMoERunner` uses a cute-dsl collective (`allreduce_rmsnorm_reduce_scatter`, `fused_add_multicast_gemm`) that requires a routed/mailbox **NVLS multicast** mapping and raises if the multicast pointer is null. None of `VLLM_KIMI_K3_GEMM_RS`, `VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT`, `VLLM_DISABLE_SHARED_EXPERTS_STREAM`, etc. disable the multicast requirement.

**Status: RESOLVED by Fix 1 (the GPU reset).** Initially this looked like a missing-capability / IMEX-provisioning problem (`nvidia-imex` binary absent, no `/dev/nvidia-caps-imex-channels`), but that was a red herring: the fabricmanager log's own guidance - *"All GPUs in the partition need to be reset to recover"* - was literal. After `sudo nvidia-smi -r` (with FM stopped, then restarted), the NVLS probe passes **with NVLS enabled** and Kimi-K3 loads its multicast collective. No `nvidia-imex` install and no reboot were needed. If the stuck state recurs (e.g. after another mid-multicast crash), repeat Fix 1.

Takeaway: on this node, treat `routed NVLS multicast mapping is unavailable` (or NCCL `CUDA error 401` at `ncclCommInitRank`) as **"reset the GPUs"**, not "disable NVLS" - especially for K3, which cannot run without multicast.

## Fix 3 (confirmed on B300) - the p5en CUDA-JIT link fixes ARE needed

Once past NCCL (Fix 1), Kimi-K3 MXFP4 hit **both** p5en CUDA-JIT failures, in order, exactly as predicted:

1. During CUDA-graph capture: `RuntimeError: Worker failed with error '[Errno 2] No such file or directory: 'ninja''`. FlashInfer's JIT shells out to `ninja` by name; `vllm-serve.sh` runs vLLM by absolute path so the venv `bin/` is not on PATH. **Fix:** `export PATH="$HOME/vllm-env/bin:$CUDA_HOME/bin:$PATH"` (ninja lives at `~/vllm-env/bin/ninja`; nvcc at `/opt/pytorch/cuda/bin`).
2. Next, the JIT link step: `/usr/bin/x86_64-linux-gnu-ld.bfd: cannot find -lcudart`. The linker wants unversioned `libcudart.so`/`libnvrtc.so`; only versioned `*.so.13` exist (in the venv's `nvidia/cu13/lib`). FlashInfer also hardcodes `-L$CUDA_HOME/lib64`, which does not exist on this DLAMI (`/opt/pytorch/cuda` has only `lib`). **Fix:** create a link dir of unversioned symlinks AND populate `$CUDA_HOME/lib64` (+`stubs`), then export `LIBRARY_PATH`/`LD_LIBRARY_PATH`. Exact commands are in the verified serve block of [kimi-k3.md](../../../self-hosted/vllm/models/kimi-k3.md). (`libcuda.so` already has an unversioned symlink in `/usr/lib/x86_64-linux-gnu`; only `libcudart`/`libnvrtc` need creating.)

`CUDA_HOME` here is `/opt/pytorch/cuda` (CUDA 13.0), not `/usr/local/cuda` (which does not exist). These are the same class of fixes as p5en Fixes 1+2/3, so applying that whole block is the reliable path.

## First-boot JIT is very long (one-time, cached) - sm_103a

Kimi-K3 MXFP4 verified serving on 8×B300 (TP=8, vLLM 0.28.0), but the **first** boot after weights are cached still took ~60-75 min, dominated by JIT for the brand-new B300 arch (`sm_103a`; note B200 is `sm_100`, B300/GB300 is `sm_103a`):

- **~30 min** single-threaded `ptxas` assembling a ~116 MB PTX for the trtllm fused-MoE kernel (14 objects). Box looks ~99% idle because ptxas is single-threaded.
- **~24 min** FlashInfer autotuning `trtllm_fp4_block_scale_moe` (22 configs at ~65 s each; `[AutoTuner]:` progress bar).
- **~5 min** CUDA-graph capture (83 sizes).

All of it caches (`~/.cache/flashinfer`, etc.), so later boots are far shorter. If you only need to confirm the endpoint quickly, `--enforce-eager` skips graph capture (not the autotune). Verified runtime once up: `DEEPGEMM_MXFP4` MoE, `FLASHINFER_MLA` + `FlashKDA` attention, `mnnvl` fused allreduce, ~1.35M-token KV cache, 10.3× concurrency at 131K. **Parsers: `kimi_k3` for both tool and reasoning** (`kimi_k2` mis-parses K3).

## Model fit on this node (important)

~2.15 TiB total VRAM. A model's **total** weight bytes must fit the **sum** of all 8 GPUs (TP shards the model but the total is still the ceiling):

| Precision | Bytes/param | e.g. Kimi-K3 (2.78 T params) | Fits? |
|---|---|---|---|
| BF16 | 2 | ~5.56 TB | No |
| FP8 | 1 | ~2.82 TB (`RedHatAI/Kimi-K3-FP8-BLOCK`) | No - exceeds total VRAM before KV cache |
| 4-bit (MXFP4 / NVFP4) | 0.5 | ~1.56 TB (`moonshotai/Kimi-K3`, native MXFP4) | Yes - ~370 GB left for KV at 0.90 util |

See [self-hosted/vllm/models/kimi-k3.md](../../../self-hosted/vllm/models/kimi-k3.md).

## Quick failure -> fix reference

| Symptom in the log | Cause | Fix |
|---|---|---|
| `NCCL error: unhandled cuda error` at `ncclCommInitRank`, before download; `NCCL_DEBUG=WARN` shows `nvls.cc ... CUDA error 401` | stuck NVLS multicast team state in GPU/NVSwitch hardware | Fix 1: reset GPUs (stop FM -> `nvidia-smi -r` -> start FM). For a model that does NOT need multicast, `NCCL_NVLS_ENABLE=0` is a fallback. |
| `routed NVLS multicast mapping is unavailable` at model load (Kimi-K3) | K3's latent-MoE kernel needs multicast, which is stuck | Fix 1: reset GPUs (do NOT disable NVLS - K3 requires it) |
| CUDA OOM during weight load at TP=8 | model's total weights exceed ~2.15 TiB (e.g. FP8/BF16 Kimi-K3) | serve a 4-bit variant that fits (see table) |
| `No such file or directory: 'ninja'` or `ld: cannot find -l...` after weight load | CUDA-JIT link path (not yet observed on B300) | p5en Fixes 1+2 (see above) |
