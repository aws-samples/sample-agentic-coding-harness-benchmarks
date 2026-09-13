# Serving on a p5en.48xlarge (8x H200) - extra CUDA fixes

> Companion note to [SKILL.md](SKILL.md). The vLLM scripts and the model guides under [`self-hosted/vllm/`](../../../self-hosted/vllm/) were verified on the **reference node**: g6e.12xlarge (4x L40S). On a **p5en.48xlarge (8x H200 141GB, NVSwitch)** several additional issues surface that the reference node never hits, because 8-GPU tensor parallelism over NVSwitch activates code paths (DeepGemm JIT, FlashInfer mnnvl allreduce) that JIT-compile CUDA at server startup. This file records every extra change needed there. If you are on the reference 4x L40S node you do **not** need any of this; if you are on a p5en (or any 8x H200 / NVSwitch box), you do.

All paths below assume the venv and caches live on the large ephemeral NVMe (`/opt/dlami/nvme`) rather than the small root disk - see "Disk layout" first.

## 0. Driver alignment (prerequisite, one-time)

The DLAMI shipped a **mismatched NVIDIA driver**: the loaded kernel module was `595.71.05` while the userspace libraries had been upgraded (likely by unattended-upgrades) to `610.43.02`, so `nvidia-smi` failed with `Failed to initialize NVML: Driver/library version mismatch`. `nvidia-fabricmanager` (mandatory on an NVSwitch box) was apt-held at 595.

Fix - align the whole stack to 610.43.02 and reboot:

```bash
sudo apt-get install -y --allow-change-held-packages \
  nvidia-dkms-610=610.43.02-0ubuntu0.24.04.1 \
  nvidia-fabricmanager=610.43.02-1ubuntu1 \
  nvidia-utils-610=610.43.02-0ubuntu0.24.04.1
sudo apt-mark hold nvidia-fabricmanager   # re-pin FM (preserve the original hold intent)
sudo reboot                                # required: unload 595, load the freshly built 610 module
```

After reboot, verify before doing anything else:

```bash
nvidia-smi --query-gpu=index,name,driver_version --format=csv   # expect 8x H200 @ 610.43.02
systemctl is-active nvidia-fabricmanager                        # expect: active
nvidia-smi -q | grep -A2 Fabric                                 # expect: State: Completed / Status: Success
```

The NVLink fabric State must be `Completed` - tensor parallelism across 8 GPUs depends on it.

## Disk layout: put the venv, caches, and weights on the NVMe

On this node the root disk `/` is tiny (~29 GB). The vLLM venv (torch + CUDA wheels, several GB) and especially the model weights (Kimi-K2.7-Code is ~555 GB on disk) must not go there. Use the 27 TB ephemeral NVMe at `/opt/dlami/nvme`:

```bash
mkdir -p /opt/dlami/nvme/{vllm-env,uv-cache,hf-cache,tmp,cuda-link}
export VLLM_ENV=/opt/dlami/nvme/vllm-env      # vllm-install.sh + vllm-serve.sh both honor this
export UV_CACHE_DIR=/opt/dlami/nvme/uv-cache
export TMPDIR=/opt/dlami/nvme/tmp
# HF_HOME auto-defaults to /opt/dlami/nvme/hf-cache in vllm-serve.sh when the volume exists.
```

`/opt/dlami/nvme` is **ephemeral** - wiped on instance stop/terminate. Weights re-download after a stop; that is the right trade for a serving box.

## The CUDA startup fixes (the core of this note)

On 8x H200 + NVSwitch, startup fails in four places the reference node never reaches. Fixes 1 to 3 are CUDA JIT problems: vLLM builds kernels at startup and cannot find the tools or libraries to link them, so they surface late, after the weights load. Fix 4 is different in kind and in timing: NCCL cannot set up the 8-GPU collectives at all, so it fails before a single weight is read. All four are fixed with environment variables passed to `vllm-serve.sh`, plus a handful of symlinks for Fixes 2 and 3.

Read Fix 4 first if the server dies early with no mention of a missing library: its top-level error names neither NCCL nor the real cause.

### Fix 1 - `ninja` must be on PATH (DeepGemm + FlashInfer JIT)

DeepGemm and FlashInfer shell out to the **`ninja` executable** by name to build CUDA kernels. `vllm-install.sh` pip-installs the `ninja` package into the venv (`$VLLM_ENV/bin/ninja`), but `vllm-serve.sh` invokes vLLM by absolute path without activating the venv, so `bin/` is not on the subprocess PATH. Symptom:

```
RuntimeError: Worker failed with error '[Errno 2] No such file or directory: 'ninja''
```

(occurs at KV-cache init, after weight loading). Fix - put the venv bin (and the CUDA toolkit bin, for `nvcc`) on PATH:

```bash
export CUDA_HOME=/opt/pytorch/cuda                 # the DLAMI's real toolkit (has nvcc); /usr/local/cuda does NOT exist
export PATH="$VLLM_ENV/bin:$CUDA_HOME/bin:$PATH"
```

### Fix 2 - CUDA libs must be linkable (`-lcudart` / `-lcuda`)

Once `ninja` runs, FlashInfer's **mnnvl allreduce** kernel (`trtllm_mnnvl_comm`, auto-selected on an NVSwitch box) JIT-compiles and fails at the **link** step:

```
/usr/bin/ld: cannot find -lcudart: No such file or directory
/usr/bin/ld: cannot find -lcuda:   No such file or directory
collect2: error: ld returned 1 exit status
RuntimeError: Ninja build failed. ... Engine core initialization failed.
```

(occurs late, after CUDA graph capture). The linker wants **unversioned** `libcudart.so` / `libcuda.so`, but this node only has versioned files:
- `libcudart.so.13` inside the venv at `nvidia/cu13/lib/`
- `libcuda.so.1 -> libcuda.so.610.43.02` in `/usr/lib/x86_64-linux-gnu/` (driver)

Fix - create a link dir of unversioned symlinks and expose it via `LIBRARY_PATH` (link time) and `LD_LIBRARY_PATH` (runtime):

```bash
LINKDIR=/opt/dlami/nvme/cuda-link
VENV_CU13="$VLLM_ENV/lib/python3.12/site-packages/nvidia/cu13/lib"
mkdir -p "$LINKDIR"
ln -sf "$VENV_CU13/libcudart.so.13"            "$LINKDIR/libcudart.so"
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1  "$LINKDIR/libcuda.so"

export LIBRARY_PATH="$LINKDIR:$VENV_CU13:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LINKDIR:$VENV_CU13:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
```

Verify the link resolves before launching (fast sanity check):

```bash
echo 'int main(){return 0;}' > /tmp/t.c
gcc /tmp/t.c -L"$LINKDIR" -lcudart -lcuda -o /tmp/t.out && echo "LINK OK"
```

> Alternative: set `VLLM_FLASHINFER_ALLREDUCE_BACKEND` away from `mnnvl`, or otherwise disable the FlashInfer fused allreduce, so it never JIT-compiles. vLLM falls back to the CUSTOM/PYNCCL allreduce backends (already available). Fix 2 above keeps the optimized allreduce path instead of dropping it; prefer it unless you want to skip the JIT entirely.

### Fix 3 - FP8 models (GLM-5.2) also need `libnvrtc.so`, in `$CUDA_HOME/lib64`

Kimi-K2.7-Code needs only Fix 1 + Fix 2. **GLM-5.2-FP8 needs one more.** Its FP8 path JIT-compiles an extra FlashInfer kernel, `fp8_blockscale_gemm_90` (a CUTLASS/DeepGemm blockscale GEMM), that links against **NVRTC** (NVIDIA Runtime Compilation) in addition to cudart/cuda. Symptom (late, after CUDA graph capture, same shape as Fix 2 but a different lib):

```
/usr/bin/ld: cannot find -lnvrtc: No such file or directory
collect2: error: ld returned 1 exit status
RuntimeError: Ninja build failed ... fp8_blockscale_gemm_90.so ... Engine core initialization failed.
```

Two things make this distinct from Fix 2:

1. **A different missing lib - `libnvrtc`.** Only `libnvrtc.so.13` exists (in the venv's `nvidia/cu13/lib`); the linker wants unversioned `libnvrtc.so`.
2. **FlashInfer's link command hardcodes its `-L` search dirs to `$CUDA_HOME/lib64` and `$CUDA_HOME/lib64/stubs`** (visible in the failing `c++ ... -shared -L/opt/pytorch/cuda/lib64 -L/opt/pytorch/cuda/lib64/stubs -lcudart -lcuda -lnvrtc ...` line). On the DLAMI, `/opt/pytorch/cuda` has a `lib` dir but **no `lib64`** - so even `-lcudart`/`-lcuda` are only found via `LIBRARY_PATH` from Fix 2, and `-lnvrtc` is not found at all. The robust fix is to create `$CUDA_HOME/lib64` (+ `stubs`) and populate it with the unversioned symlinks the build command expects.

Fix - add `libnvrtc.so` to the Fix-2 link dir AND materialize the symlinks in `$CUDA_HOME/lib64` where FlashInfer actually looks:

```bash
LINKDIR=/opt/dlami/nvme/cuda-link
VENV_CU13="$VLLM_ENV/lib/python3.12/site-packages/nvidia/cu13/lib"

# (a) extend the Fix-2 link dir with nvrtc
ln -sf "$VENV_CU13/libnvrtc.so.13" "$LINKDIR/libnvrtc.so"

# (b) create + populate $CUDA_HOME/lib64 and its stubs (the dirs FlashInfer's -L points at).
#     /opt/pytorch/cuda has only lib/, no lib64 - so we make it. $CUDA_HOME/lib is writable
#     without sudo on the DLAMI; if lib64 is not, prefix these with sudo.
mkdir -p "$CUDA_HOME/lib64/stubs"
ln -sf "$VENV_CU13/libcudart.so.13"            "$CUDA_HOME/lib64/libcudart.so"
ln -sf "$VENV_CU13/libnvrtc.so.13"             "$CUDA_HOME/lib64/libnvrtc.so"
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1  "$CUDA_HOME/lib64/libcuda.so"
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1  "$CUDA_HOME/lib64/stubs/libcuda.so"

# then add lib64 + stubs to LIBRARY_PATH (belt and suspenders):
export LIBRARY_PATH="$LINKDIR:$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$VENV_CU13:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LINKDIR:$CUDA_HOME/lib64:$VENV_CU13:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
```

Verify all three resolve before launching:

```bash
echo 'int main(){return 0;}' > /tmp/t.c
gcc /tmp/t.c -L"$LINKDIR" -lcudart -lcuda -lnvrtc -o /tmp/t.out && echo "LINK OK"
```

If a prior boot already failed on this, clear the stale JIT cache so it rebuilds cleanly: `rm -rf ~/.cache/flashinfer/*/*/cached_ops/fp8_blockscale_gemm_90`.

GLM-5.2 verified serving with this: `zai-org/GLM-5.2-FP8`, TP=8, `MAX_MODEL_LEN=300000` (passes the >=200K gate), `GPU_MEM_UTIL=0.95`, `glm47` tool+reasoning parsers. Runtime: GPU KV cache ~413,000 tokens, max concurrency 1.38x at 300K.

Note the pre-existing DLAMI fixes the serve script already applies automatically and that you should NOT undo: `VLLM_USE_FLASHINFER_SAMPLER=0` (native sampler, no runtime nvcc) and the `CUDA_HOME` fallback to `/opt/pytorch/cuda`.

### Fix 4 - NCCL: disable NVLink SHARP (NVLS), or nothing starts at all

Fixes 1-3 are link-time problems that surface late, after weight loading. This one lands **before any weights load**, and its top-level message names neither NCCL nor NVLS:

```
RuntimeError: NCCL error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)
RuntimeError: Engine core initialization failed. See root cause above.
```

Re-run with `NCCL_DEBUG=INFO` and the real cause appears:

```
transport/nvls.cc:287 NCCL WARN Failed to bind NVLink SHARP (NVLS) Multicast memory
  of size 2097152 : CUDA error 401 'the operation cannot be performed in the present state'.
This is usually caused by a system or configuration error in the Fabric Manager or NVSwitches.
Disable NVLS (NCCL_NVLS_ENABLE=0) if you wish to avoid this error in the future.
```

NCCL tries to bind NVLink SHARP multicast memory, the NVSwitch in-network reduction path, and the driver refuses with CUDA 401. NCCL treats that as fatal, so every rank dies at `ncclCommInitRank` and vLLM never reaches the model.

The fix is the switch NCCL itself names:

```bash
export NCCL_NVLS_ENABLE=0
```

**Prove it is the node, not vLLM, in one minute.** This reproduces with a bare 8-GPU all_reduce and no vLLM at all, which is the fastest way to tell a node fault from a serving bug:

```bash
cat > /tmp/nccl_test.py <<'EOF'
import torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r)
t = torch.ones(1024, device=f"cuda:{r}"); dist.all_reduce(t)
if r == 0: print(f"NCCL ALL_REDUCE OK, world={dist.get_world_size()}, sum={t[0].item()}")
EOF
$VLLM_ENV/bin/python -m torch.distributed.run --nproc_per_node=8 /tmp/nccl_test.py
```

Without the switch every rank raises `ncclUnhandledCudaError`; with it the test prints `NCCL ALL_REDUCE OK, world=8, sum=8.0`. Measured on a p5en.48xlarge with driver 595.91.07, NCCL 2.29.7, torch 2.13.0+cu130, vLLM 0.28.0.

Three things worth knowing:

1. **A healthy fabric does not rule this out.** `nvidia-smi -q` reported `State: Completed / Status: Success` on all 8 GPUs, `nvidia-fabricmanager` was active, and NVLink showed 26.562 GB/s per link. Section 0 was already satisfied. The NVLS bind still failed, so do not read a clean fabric as proof that collectives work.
2. **The `aws-ofi-nccl` warnings in the same log are a red herring.** `NET/OFI Failed to initialize rdma protocol` appears right before the failure and is unrelated: `NCCL_NET_PLUGIN=none` does not fix it, and a single-node TP job never uses the network plugin.
3. **It costs performance.** Disabling NVLS drops in-network reduction, so TP=8 collectives are slower than on a node where the bind succeeds. Wall-clock-derived cost per task from a run with NVLS off is therefore not strictly comparable to one with it on. Say so in any results doc built from such a run.

## Putting it together - the full p5en launch

`vllm-serve.sh` inherits the caller's environment, so export everything above first, then run it with the model guide's parameters ([kimi-k2.7-code.md](../../../self-hosted/vllm/models/kimi-k2.7-code.md), TP=8):

```bash
cd self-hosted/vllm/scripts

export VLLM_ENV=/opt/dlami/nvme/vllm-env
export CUDA_HOME=/opt/pytorch/cuda
export TMPDIR=/opt/dlami/nvme/tmp
export PATH="$VLLM_ENV/bin:$CUDA_HOME/bin:$PATH"                        # Fix 1

LINKDIR=/opt/dlami/nvme/cuda-link                                       # Fix 2 + Fix 3
VENV_CU13="$VLLM_ENV/lib/python3.12/site-packages/nvidia/cu13/lib"
mkdir -p "$LINKDIR"
ln -sf "$VENV_CU13/libcudart.so.13"           "$LINKDIR/libcudart.so"
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$LINKDIR/libcuda.so"
ln -sf "$VENV_CU13/libnvrtc.so.13"            "$LINKDIR/libnvrtc.so"    # Fix 3 (FP8 models)

# Fix 3: FlashInfer's FP8 kernel link cmd hardcodes -L$CUDA_HOME/lib64[/stubs], which
# does not exist on the DLAMI (only $CUDA_HOME/lib). Create + populate it.
mkdir -p "$CUDA_HOME/lib64/stubs"
ln -sf "$VENV_CU13/libcudart.so.13"           "$CUDA_HOME/lib64/libcudart.so"
ln -sf "$VENV_CU13/libnvrtc.so.13"            "$CUDA_HOME/lib64/libnvrtc.so"
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_HOME/lib64/libcuda.so"
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_HOME/lib64/stubs/libcuda.so"

export LIBRARY_PATH="$LINKDIR:$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$VENV_CU13:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LINKDIR:$CUDA_HOME/lib64:$VENV_CU13:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

export NCCL_NVLS_ENABLE=0    # Fix 4 - without this NCCL dies before any weights load

# --- Kimi-K2.7-Code (needs Fix 1 + Fix 2; Fix 3 is harmless to leave in) ---
MODEL="moonshotai/Kimi-K2.7-Code" \
SERVED_NAME="kimi-k2.7-code" \
TP=8 PORT=8000 MAX_MODEL_LEN=131072 GPU_MEM_UTIL=0.90 \
TOOL_PARSER="kimi_k2" REASONING_PARSER="kimi_k2" \
EXTRA_ARGS="--trust-remote-code" \
  ./vllm-serve.sh

# --- OR GLM-5.2-FP8 (needs Fix 1 + Fix 2 + Fix 3; passes the >=200K gate at 300K) ---
MODEL="zai-org/GLM-5.2-FP8" \
SERVED_NAME="glm-5.2" \
TP=8 PORT=8000 MAX_MODEL_LEN=300000 GPU_MEM_UTIL=0.95 \
TOOL_PARSER="glm47" REASONING_PARSER="glm47" \
EXTRA_ARGS="--trust-remote-code" \
  ./vllm-serve.sh
```

## Startup timing and what "healthy" looks like

First boot: ~10-15 min weight download (~555 GB at ~1 GB/s with an HF token) + ~4 min weight load (64 shards) + torch.compile + CUDA graph capture (51 graphs). Subsequent boots skip the download. Healthy-boot signposts in `self-hosted/vllm/logs/vllm-serve.log`:

- `Using CompressedTensorsWNA16MarlinMoEMethod` / `MARLIN WNA16 MoE backend` (correct quantization)
- `Using FLASH_ATTN_MLA attention backend` (correct for the DeepSeek-V3/MLA arch)
- `Loading safetensors checkpoint shards: 100%`
- `Capturing CUDA graphs ... 51/51`
- `GPU KV cache size: ~740,000 tokens` / `Maximum concurrency for 131072 tokens per request: ~5.6x`
- `Server ready at http://127.0.0.1:8000/v1`

## Quick failure -> fix reference

| Symptom in the log | Cause | Fix |
|---|---|---|
| `NVML: Driver/library version mismatch` | kernel module vs userspace driver skew | Section 0 (align to 610 + reboot) |
| `No space left` during download, or root disk fills | weights/venv on the small root disk | Disk layout (use `/opt/dlami/nvme`) |
| `No such file or directory: 'ninja'` | ninja binary not on subprocess PATH | Fix 1 (PATH) |
| `ld: cannot find -lcudart / -lcuda` -> `Ninja build failed` -> `Engine core initialization failed` | FlashInfer mnnvl allreduce can't link CUDA | Fix 2 (cuda-link + LIBRARY_PATH) |
| `NCCL error: unhandled cuda error` -> `Engine core initialization failed`, BEFORE weight loading (with `NCCL_DEBUG=INFO`: `nvls.cc:287 Failed to bind NVLink SHARP (NVLS) Multicast memory ... CUDA error 401`) | NCCL cannot bind NVSwitch multicast memory; a healthy Fabric Manager does not rule this out | Fix 4 (`NCCL_NVLS_ENABLE=0`) |
| `ld: cannot find -lnvrtc` -> `Ninja build failed` on `fp8_blockscale_gemm_90` -> `Engine core initialization failed` | FP8 model (GLM-5.2) FlashInfer kernel needs NVRTC, and FlashInfer's `-L` points at a non-existent `$CUDA_HOME/lib64` | Fix 3 (libnvrtc.so + populate `$CUDA_HOME/lib64`) |

**Model -> which fixes:** Kimi-K2.7-Code = Fix 1 + Fix 2. GLM-5.2-FP8 = Fix 1 + Fix 2 + Fix 3. **Fix 4 applies to every multi-GPU model on this node**, whatever the precision, because it fails at distributed init before a model is even loaded. Applying all four is harmless for any model, so the combined launch block above is a safe default on this node.

## Verified serve configs (what we actually ran on this p5en)

These are the exact `vllm-serve.sh` settings used for each model benchmarked on this node (all with the CUDA env exports above, and the venv/caches on `/opt/dlami/nvme`). **`TP` is not always 8** - block-FP8 MoE models impose sharding constraints (see the note under the table).

| Model | HF repo | TP | MAX_MODEL_LEN | GPU_MEM_UTIL | TOOL_PARSER | REASONING_PARSER | EXTRA_ARGS | Weights |
|---|---|---:|---:|---:|---|---|---|---|
| kimi-k2.7-code | `moonshotai/Kimi-K2.7-Code` | 8 | 131072 | 0.90 | `kimi_k2` | `kimi_k2` | `--trust-remote-code` | ~555 GB |
| glm-5.2 | `zai-org/GLM-5.2-FP8` | 8 | 300000 | 0.95 | `glm47` | `glm47` | `--trust-remote-code` | ~750 GB FP8 |
| minimax-m2.5 | `MiniMaxAI/MiniMax-M2.5` | **4** | 196608 | 0.92 | `minimax_m2` | `minimax_m2` | `--trust-remote-code` | ~466 GB FP8 |
| qwen3-coder-480b | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` | **4** | 200000 | 0.95 | `qwen3_coder` | (none) | `--trust-remote-code` | ~482 GB FP8 |

Runtime observed: kimi 128K window / 740K-tok KV / 5.65x concurrency; glm-5.2 300K / 413K KV / 1.38x; minimax-m2.5 192K / 1.18M KV / 6.0x; qwen3-coder-480b 200K / 255K KV / 1.28x.

### Why TP is 4 for MiniMax-M2.5 and Qwen3-Coder-480B (not 8)

Block-quantized FP8 MoE models must satisfy **two** tensor-parallel divisibility constraints at once, and TP=8 fails both of these models:

1. **KV-head constraint:** `num_key_value_heads % TP == 0`. Both models have `num_key_value_heads = 8`, so TP must divide 8 -> {1,2,4,8}. (Failing this aborts with `assert self.total_num_kv_heads % tp_size == 0`.)
2. **FP8 MoE block constraint:** the per-GPU MoE shard `moe_intermediate_size / TP` must be divisible by the FP8 weight block size (128). MiniMax `moe_intermediate_size=1536` -> TP in {1,2,4,6}; Qwen-480B `moe_intermediate_size=2560` -> TP in {1,2,4}. (Failing this aborts with `ValueError: The output_size of gate's and up's weight = N is not divisible by weight quantization block_n = 128`.)

The intersection for both models is **TP=4** (TP=8 fails constraint 2: MiniMax 1536/8=192, Qwen 2560/8=320, neither divisible by 128). Consequence worth knowing: since these run on 4 GPUs, **two replicas fit on one 8x H200 node** (GPUs 0-3 and 4-7 via `CUDA_VISIBLE_DEVICES`, different `PORT`) - useful for throughput, though the benchmark harness drives one endpoint at a time. Kimi and GLM-5.2 genuinely need all 8 GPUs (TP=8) and cannot be doubled here.

To check a new model's viable TP before serving, read `num_key_value_heads` and `moe_intermediate_size` from its HF `config.json` and apply both rules above. BF16 (unquantized) models drop constraint 2 entirely (no weight blocks), so they follow only the KV-head rule - e.g. Qwen3-Coder-480B in BF16 (~960 GB) would run at TP=8, but does not fit at TP=4.
