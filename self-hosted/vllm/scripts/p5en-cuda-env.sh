#!/usr/bin/env bash
# p5en.48xlarge (8xH200) CUDA + cache environment. SOURCE this, do not execute it.
#
#   source "<repo>/self-hosted/vllm/scripts/p5en-cuda-env.sh"
#
# No-op on any other node: keyed on 8 GPUs reporting as H200, and every step is
# idempotent, so sourcing it repeatedly costs nothing. Sets P5EN_CUDA_ENV_APPLIED
# to 1 or 0 so a caller can log what happened.
#
# WHY THIS EXISTS
#
# Two independent defects on this node, both of which have already killed runs:
#
# 1. JIT link failures. The FP8 models compile DeepGemm/FlashInfer kernels at
#    engine init, and that link step fails on the stock DLAMI: there is no
#    /usr/local/cuda (nvcc lives in /opt/pytorch/cuda), ninja is only inside the
#    vLLM venv, and FlashInfer's link command hardcodes
#    -L$CUDA_HOME/lib64[/stubs], which the DLAMI does not ship. Full rationale:
#    .claude/skills/vllm-setup/p5en-h200-cuda-fixes.md (Fixes 1-3).
#
# 2. The 29 GB root disk. torch.compile and Triton default their caches to
#    ~/.cache, which is on /. One cache is written per model config, so a
#    multi-model run accumulates several GB and fills the disk. On 2026-08-30 it
#    hit 100% mid-run: the active model died with "OSError: [Errno 28] No space
#    left on device" writing its event stream, and the NEXT model's vLLM then
#    failed inside Triton compilation because its cache had nowhere to go -- one
#    root cause presenting as two unrelated-looking symptoms, with nothing
#    warning first. The 28 TB ephemeral NVMe has the room.
#
# This lives in its own sourceable file because BOTH entry points need it and
# only one of them had it. run-multi-model-benchmark.sh (the quality path)
# exported these inline, so vllm-serve.sh worked when launched through the
# orchestrator and failed when launched directly -- which is how the throughput
# sweep runs. Keeping one copy is what stops the two paths drifting again.
#
# Everything is set with ${VAR:-default} so an explicit caller value always wins.

p5en_cuda_env_apply() {
  local gpus h200
  gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
  h200="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c H200 || true)"
  if [[ "$gpus" -ne 8 || "$h200" -ne 8 ]]; then
    export P5EN_CUDA_ENV_APPLIED=0
    return 0
  fi

  # The venv and caches live on the big NVMe: the 29 GB root disk cannot hold
  # torch, the CUDA wheels and the weights.
  local nvme=/opt/dlami/nvme
  export VLLM_ENV="${VLLM_ENV:-$nvme/vllm-env}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$nvme/uv-cache}"
  export TMPDIR="${TMPDIR:-$nvme/tmp}"
  export CUDA_HOME="${CUDA_HOME:-/opt/pytorch/cuda}"
  export PATH="$VLLM_ENV/bin:$CUDA_HOME/bin:$PATH"                     # Fix 1: ninja + nvcc

  # Defect 2: keep the compile caches off /. These live under $VLLM_ENV rather
  # than directly under $nvme for the same ownership reason as the link dir
  # below -- and a torch.compile cache matters more than most, because its
  # contents are loaded as code.
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$VLLM_ENV/cache/vllm}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$VLLM_ENV/cache/triton}"
  # Warn rather than abort: these are created lazily by vLLM/Triton anyway, and a
  # caller who overrode one of these to an unwritable path should get a clear
  # message, not a bare mkdir error that kills a `set -e` caller mid-source.
  local d
  for d in "$TMPDIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"; do
    mkdir -p "$d" 2>/dev/null || echo "[warn]  cannot create $d -- check the override that set it" >&2
  done

  # Fixes 2 + 3. The link dir lives INSIDE $VLLM_ENV, not directly under $nvme:
  # /opt/dlami/nvme is mode 1777 (world-writable, like /tmp) and is wiped on every
  # instance stop, so a path created there does not pre-exist on a fresh boot --
  # another local user could create it first, or as a symlink to a directory they
  # own, and our ln -sf would then populate a directory they control. That
  # directory is prepended to LD_LIBRARY_PATH for every vLLM process, which turns
  # it into arbitrary shared-library injection into the serving process. $VLLM_ENV
  # is owned by us and not world-writable, so it closes that window.
  local linkdir="$VLLM_ENV/cuda-link"
  local venv_cu13="$VLLM_ENV/lib/python3.12/site-packages/nvidia/cu13/lib"
  mkdir -p "$linkdir"

  # VERIFY before trusting, do not just assume. The paragraph above is only true
  # once our install has created $VLLM_ENV; on a fresh boot the NVMe is empty, so
  # $VLLM_ENV does not exist and the race is still open at that moment. Check that
  # neither path is a symlink and that both are owned by us -- if not, something
  # else created them and we must not populate a directory we are about to put on
  # LD_LIBRARY_PATH. Fail closed: skip the fixes and say so. FP8 JIT will then
  # fail loudly at engine init, which is the correct outcome; silently loading
  # libraries from a directory another local user controls is not.
  local uid owner p
  uid="$(id -u)"
  for p in "$VLLM_ENV" "$linkdir"; do
    owner="$(stat -c %u "$p" 2>/dev/null || echo -1)"
    if [[ -L "$p" || "$owner" != "$uid" ]]; then
      echo "[warn]  $p is a symlink or not owned by uid $uid (owner=$owner):" >&2
      echo "[warn]  refusing to populate it or add it to LD_LIBRARY_PATH." >&2
      echo "[warn]  Remove it and re-run the vLLM install, or set VLLM_ENV to a path you own." >&2
      export P5EN_CUDA_ENV_APPLIED=0
      return 0
    fi
  done

  ln -sf "$venv_cu13/libcudart.so.13"           "$linkdir/libcudart.so"
  ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$linkdir/libcuda.so"
  ln -sf "$venv_cu13/libnvrtc.so.13"            "$linkdir/libnvrtc.so"

  # Fix 3: create and populate the lib64[/stubs] that FlashInfer's -L expects.
  if mkdir -p "$CUDA_HOME/lib64/stubs" 2>/dev/null; then
    ln -sf "$venv_cu13/libcudart.so.13"           "$CUDA_HOME/lib64/libcudart.so"
    ln -sf "$venv_cu13/libnvrtc.so.13"            "$CUDA_HOME/lib64/libnvrtc.so"
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_HOME/lib64/libcuda.so"
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_HOME/lib64/stubs/libcuda.so"
  else
    echo "[warn]  could not write $CUDA_HOME/lib64 (needs sudo); FP8 JIT may fail with 'cannot find -lnvrtc'" >&2
  fi

  export LIBRARY_PATH="$linkdir:$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$venv_cu13:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$linkdir:$CUDA_HOME/lib64:$venv_cu13:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  export P5EN_CUDA_ENV_APPLIED=1
}

p5en_cuda_env_apply
