#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# vllm-serve.sh — Serve an open-weight model with vLLM tensor-parallel on a
#                 multi-GPU EC2 node (reference: g6e.12xlarge, 4x L40S).
#
# vLLM shards the model across all GPUs with tensor parallelism
# (--tensor-parallel-size), so a 30B–80B model that will not fit on one L40S
# (46 GB) serves comfortably across four. Tensor parallelism keeps every GPU
# busy on every token and sustains high
# throughput under concurrent load — the regime the cost model in the strategy
# doc depends on.
#
# Usage:
#   ./vllm-serve.sh                         # default: qwen3-coder-30b, TP=4
#   MODEL=Qwen/Qwen3-32B ./vllm-serve.sh    # a different HF model
#   TP=2 ./vllm-serve.sh                    # fewer GPUs
#   ROPE_SCALING=4 MAX_MODEL_LEN=131072 ./vllm-serve.sh   # extend context to 128K (YaRN)
#   ./vllm-serve.sh --foreground            # run in the foreground (see logs live)
#
# Environment variables (all optional — sensible defaults for a 4x L40S node):
#   MODEL              HF repo id to serve            (default: Qwen/Qwen3-Coder-30B-A3B-Instruct)
#   SERVED_NAME        name clients pass as --model   (default: qwen3-coder-30b)
#   TP                 tensor-parallel size / #GPUs   (default: 4)
#   PORT               OpenAI-compatible API port     (default: 8000)
#                      Server state is scoped to PORT, so several models can be
#                      served side by side on one host (one per GPU, each with its
#                      own CUDA_VISIBLE_DEVICES). Port 8000 keeps the historical
#                      paths /tmp/vllm-serve.pid and logs/vllm-serve.log; any other
#                      port gets the -$PORT suffix. `--stop` then stops only that
#                      port's instance; use `--stop-all` to stop every instance.
#   MAX_MODEL_LEN      context window to serve        (default: 32768)
#   ROPE_SCALING       extend context past the model's native window with YaRN.
#                      Two forms:
#                        - a bare number  → YaRN factor, e.g. ROPE_SCALING=4
#                          (serves 4x the native 32768 = 131072 tokens / 128K)
#                        - a full JSON object nested under `rope_scaling` in
#                          --hf-overrides, e.g. '{"rope_type":"yarn","factor":4.0,...}'
#                      Leave unset (default) to serve at the native window. Set
#                      MAX_MODEL_LEN to the extended length alongside it.
#                      (default: unset — no rope scaling)
#   MAX_NUM_SEQS       cap on concurrent sequences. Usually leave unset (vLLM
#                      defaults to 256). REQUIRED for hybrid Mamba models on a
#                      VRAM-tight node (e.g. Qwen3-Coder-Next): if boot fails with
#                      "max_num_seqs (256) exceeds available Mamba cache blocks (N)",
#                      set this to N or lower. (default: unset)
#   GPU_MEM_UTIL       fraction of VRAM vLLM may use  (default: 0.90)
#   TOOL_PARSER        vLLM tool-call parser          (default: qwen3_coder)
#                      set to "" / "none" to disable tool calling
#   REASONING_PARSER   vLLM reasoning parser — separates thinking tokens into a
#                      dedicated content block instead of leaking into response text.
#                      Examples: glm47, qwen3, deepseek_r1. Leave unset to disable.
#                      (default: unset)
#   EXTRA_ARGS         extra flags passed through to vllm serve verbatim.
#                      e.g. EXTRA_ARGS="--trust-remote-code --enforce-eager"
#                      (default: unset)
#   HF_HOME            where HF downloads/caches weights. Defaults to the DLAMI's
#                      large NVMe scratch (/opt/dlami/nvme/hf-cache) when present,
#                      because the root disk (~193 GB) is too small for the 80B
#                      (~160 GB). Set HF_HOME=/path to override, or "" for the
#                      default ~/.cache/huggingface. NOTE: the NVMe scratch is
#                      EPHEMERAL — wiped on instance stop; weights re-download after.
#   VLLM_ENV           path to the vLLM virtualenv    (default: ~/vllm-env)
#   HF_TOKEN           HuggingFace token for faster, un-rate-limited downloads.
#                      If unset, the script also reads a gitignored .hf_token file
#                      (repo root, this vllm dir, or ~). Strongly recommended — see
#                      the loud warning at startup if no token is found. (optional)
#
# Tool calling is ON by default. Agentic clients (opencode, Claude Code) send
# `tool_choice: "auto"`, which vLLM rejects unless the server was started with
# --enable-auto-tool-choice and a matching --tool-call-parser. The default
# parser `qwen3_coder` is correct for the Qwen3-Coder models; use `hermes` for
# other Qwen3 chat models, or set TOOL_PARSER=none for a plain completion
# server. Run `vllm serve --help` for the full parser list.
#
# The server binds to 127.0.0.1 only. Reach it from your laptop with an SSH
# tunnel (see tunnel.sh) — no public ingress.
# ---------------------------------------------------------------------------

MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
SERVED_NAME="${SERVED_NAME:-qwen3-coder-30b}"
TP="${TP:-4}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
ROPE_SCALING="${ROPE_SCALING:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
VLLM_ENV="${VLLM_ENV:-$HOME/vllm-env}"

# Where HuggingFace downloads and caches weights. The default HF location is
# ~/.cache/huggingface on the ROOT disk, which on this node is only ~193 GB — the
# 80B model (~160 GB) will not fit there alongside anything else. The DLAMI ships a
# large local-NVMe scratch volume at /opt/dlami/nvme (3.5 TB here); if it exists and
# is writable we default HF_HOME there so big models have room. Override with
# HF_HOME=/some/path, or set it to "" to force the default ~/.cache location.
# NOTE: /opt/dlami/nvme is EPHEMERAL — wiped on instance stop/terminate, so weights
# cached there must be re-downloaded after a stop. That is usually the right trade
# for a serving box (huge, fast, no EBS cost), but the cache is not durable.
if [[ -z "${HF_HOME+x}" ]]; then
  if [[ -d /opt/dlami/nvme && -w /opt/dlami/nvme ]]; then
    HF_HOME="/opt/dlami/nvme/hf-cache"
  fi
fi

# Logs are written under the repo's gitignored logs dir (self-hosted/vllm/logs/)
# and simultaneously streamed to the console via tee. Override with LOG_DIR=...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/logs}"

# On an 8xH200 (p5en) DLAMI, FP8 models cannot JIT their kernels without CUDA
# path fixes, and torch.compile/Triton default their caches to the 29 GB root
# disk, which a multi-model run fills. Applied here so serving works the same
# whether this script is launched directly (the throughput sweep) or through
# run-multi-model-benchmark.sh (the quality path) -- previously only the latter
# exported them. No-op on any other node; explicit caller values always win.
# shellcheck source=./p5en-cuda-env.sh
. "$SCRIPT_DIR/p5en-cuda-env.sh"

FOREGROUND=0
[[ "${1:-}" == "--foreground" || "${1:-}" == "-f" ]] && FOREGROUND=1

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[0;33m'; BOLD='\033[1m'; RESET='\033[0m'
info()   { echo -e "${BLUE}[info]${RESET}  $1"; }
ok()     { echo -e "${GREEN}[ok]${RESET}    $1"; }
warn()   { echo -e "${YELLOW}[warn]${RESET}  $1"; }
fail()   { echo -e "${RED}[fail]${RESET}  $1"; exit 1; }

# --help: print the header comment block (env vars + options) and exit.
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '5,65p' "$0" | sed 's/^# \{0,1\}//; s/^#$//'
  echo "Options: --foreground|-f  (run in foreground)   --stop  (stop this PORT + free its GPUs)   --stop-all  (stop every vLLM instance)"
  exit 0
fi

# PORT is now used to BUILD FILE PATHS (the pid file and the log name) and those
# paths are interpolated into the `bash -c` string that launches the server, so it
# has to be validated before either use. Unvalidated, PORT=../../etc/x would place
# the log and pid file outside their directories, and a PORT containing a quote
# would break out of the launch string. It is only ever a TCP port, so require
# exactly that.
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  fail "PORT must be an integer in 1..65535, got: '$PORT'"
fi

# Server state is scoped to PORT so several instances can run side by side on one
# host -- one per GPU (or GPU pair), each serving a different model, which is how a
# parallel throughput sweep gets six arms done in two passes instead of six. Port
# 8000 deliberately keeps the historical unsuffixed paths, so every existing doc,
# `tail -f logs/vllm-serve.log` habit and bare `--stop` behaves exactly as before.
if [[ "$PORT" == "8000" ]]; then
  PID_FILE="/tmp/vllm-serve.pid"
  LOG_NAME="vllm-serve.log"
else
  PID_FILE="/tmp/vllm-serve-$PORT.pid"
  LOG_NAME="vllm-serve-$PORT.log"
fi

# --stop: kill this PORT's server (launcher + tee + vLLM workers) and free its GPUs.
# --stop-all: kill EVERY vLLM instance on the host, by name (the old blunt behaviour).
if [[ "${1:-}" == "--stop" || "${1:-}" == "--stop-all" ]]; then
  STOPPED=0
  # The engine and TP workers are children of the launcher, but they RENAME
  # themselves to "VLLM::Worker_TP<N>" / "VLLM::EngineCore", so `pkill -f "vllm
  # serve"` misses them and they keep the GPUs pinned. Renaming does not change
  # the process GROUP, though, so signalling the launcher's group reaches all of
  # them -- and, unlike a name match, reaches ONLY this instance's workers.
  if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE" 2>/dev/null || true)
    # /tmp is world-writable, so the pid file is UNTRUSTED input, and this block
    # now signals a whole process GROUP rather than a single pid -- so without
    # these checks `--stop` would be a primitive for SIGKILLing an arbitrary
    # process group, with whoever created /tmp/vllm-serve-*.pid choosing the
    # target. Require a plain pid, and require that pid to actually be one of our
    # vLLM launchers before signalling its group. (The kill itself still runs as
    # this user, so this bounds a local nuisance, not a privilege escalation.)
    if [[ ! "$PID" =~ ^[1-9][0-9]*$ ]]; then
      warn "Ignoring $PID_FILE: contents are not a pid."
      PID=""
    elif ! ps -p "$PID" -o args= 2>/dev/null | grep -q vllm; then
      warn "Ignoring $PID_FILE: pid $PID is not a vLLM process."
      PID=""
    fi
    if [[ -n "$PID" ]]; then
      # `|| true`: under `set -e -o pipefail` a dead pid makes `ps` fail, and an
      # assignment from a failing pipeline would abort the script before it could
      # fall back to the name-based rescue below.
      PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ' || true)
      if [[ -n "$PGID" ]]; then
        kill -TERM -- "-$PGID" 2>/dev/null && STOPPED=1 || true
      else
        kill "$PID" 2>/dev/null && STOPPED=1 || true
      fi
    fi
    rm -f "$PID_FILE" 2>/dev/null || true
  fi
  # Fall back to the blunt name-based sweep when asked for explicitly, or when
  # there is no pid file to work from AND no other instance is running that it
  # could collaterally kill. That keeps the old "free the GPUs no matter what"
  # rescue for the single-server case without endangering sibling arms.
  # `|| true` for the same pipefail reason: `ls` fails when no pid file matches.
  OTHER_PIDS=$(ls /tmp/vllm-serve-*.pid /tmp/vllm-serve.pid 2>/dev/null | wc -l || true)
  if [[ "${1:-}" == "--stop-all" || ( "$STOPPED" -eq 0 && "$OTHER_PIDS" -eq 0 ) ]]; then
    for pat in "vllm serve" "VLLM::Worker" "VLLM::EngineCore"; do
      pkill -f "$pat" 2>/dev/null && STOPPED=1 || true
    done
  fi
  # Give them a moment to release VRAM, then escalate to SIGKILL for any straggler
  # (CUDA teardown occasionally wedges a worker in an uninterruptible state).
  if [[ "$STOPPED" -eq 1 ]]; then
    sleep 3
    if [[ -n "${PGID:-}" ]]; then
      kill -KILL -- "-$PGID" 2>/dev/null || true
    elif [[ "${1:-}" == "--stop-all" || "$OTHER_PIDS" -eq 0 ]]; then
      # Only escalate by name when there is no sibling instance to hit.
      for pat in "vllm serve" "VLLM::Worker" "VLLM::EngineCore"; do
        pkill -9 -f "$pat" 2>/dev/null || true
      done
    fi
    ok "Stopped vLLM server on port $PORT. GPUs free once the workers exit (check: nvidia-smi)."
  else
    warn "No running vLLM server found."
  fi
  exit 0
fi

VLLM_BIN="$VLLM_ENV/bin/vllm"
[[ -x "$VLLM_BIN" ]] || fail "vLLM not found at $VLLM_BIN. Run ./vllm-install.sh first (or set VLLM_ENV)."

# Fail fast on a bad tool-call parser name BEFORE the (multi-GB, multi-minute)
# model download and engine load. vLLM only validates --tool-call-parser after
# it starts loading, so a typo or a filename/registry mismatch (e.g. the Gemma4
# parser file is gemma4_engine_tool_parser.py but the REGISTERED name is
# "gemma4") otherwise wastes a full download. Query the installed vLLM's actual
# registered names and check the requested one against them.
if [[ -n "$TOOL_PARSER" && "$TOOL_PARSER" != "none" ]]; then
  VALID_PARSERS="$("$VLLM_ENV/bin/python" - <<'PY' 2>/dev/null
# Mirror vLLM's own check in validate_api_server_args: the registry is lazily
# populated, so list_registered() is what actually triggers registration and
# returns the real names (plain .keys() reads empty before that).
names = []
try:
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
    names = list(ToolParserManager.list_registered())
except Exception:
    try:  # older vLLM layout
        from vllm.entrypoints.openai.tool_parsers import ToolParserManager
        names = list(ToolParserManager.list_registered())
    except Exception:
        names = []
print(" ".join(sorted(names)))
PY
)"
  if [[ -n "$VALID_PARSERS" && " $VALID_PARSERS " != *" $TOOL_PARSER "* ]]; then
    fail "invalid TOOL_PARSER '$TOOL_PARSER' for the installed vLLM.
       Valid parsers: $VALID_PARSERS
       (Note: the registered name can differ from the parser's source filename --
       e.g. use 'gemma4', not 'gemma4_engine'. Set TOOL_PARSER=none for a plain
       completion server.)"
  fi
  [[ -z "$VALID_PARSERS" ]] && info "Could not enumerate tool parsers to pre-validate '$TOOL_PARSER'; vLLM will validate at load time."
fi

# Sanity: enough GPUs for the requested tensor-parallel size?
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | xargs)
[[ "$GPU_COUNT" -ge "$TP" ]] || fail "Requested TP=$TP but only $GPU_COUNT GPU(s) visible."

# Point HuggingFace at the chosen cache and report where weights will land + free
# space there, so a too-small disk is obvious BEFORE a multi-hour download stalls.
if [[ -n "${HF_HOME:-}" ]]; then
  mkdir -p "$HF_HOME" 2>/dev/null || true
  export HF_HOME
  export HF_HUB_CACHE="$HF_HOME/hub"
fi
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
CACHE_FREE=$(df -h "$CACHE_DIR" 2>/dev/null | awk 'NR==2{print $4}')

info "Model:        $MODEL"
info "Served as:    $SERVED_NAME  (clients pass --model $SERVED_NAME)"
info "GPUs:         $TP of $GPU_COUNT (tensor parallelism)"
info "Context:      $MAX_MODEL_LEN tokens"
info "Weights cache: $CACHE_DIR  (${CACHE_FREE:-?} free)"
info "API:          http://127.0.0.1:$PORT/v1  (OpenAI-compatible)"
echo ""

# Telemetry off: vLLM's usage stats never leave the box (strategy doc §6 egress).
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

# HuggingFace token resolution. A token is NOT strictly required (these models are
# public), but downloading without one hits HF's stricter anonymous rate limits and
# is dramatically slower — a 60-160 GB model can crawl or stall. We look for a token
# in this order: the HF_TOKEN env var, then a .hf_token file (repo root, this repo's
# vllm dir, or $HOME). .hf_token is gitignored — see the repo .gitignore.
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." 2>/dev/null && pwd)"
VLLM_DIR="$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd)"
if [[ -z "${HF_TOKEN:-}" ]]; then
  for tf in "$REPO_ROOT/.hf_token" "$VLLM_DIR/.hf_token" "$HOME/.hf_token"; do
    if [[ -s "$tf" ]]; then
      # First non-empty, non-comment line; trim whitespace. Never echo the value.
      HF_TOKEN="$(grep -v '^[[:space:]]*#' "$tf" | grep -m1 . | tr -d '[:space:]')"
      [[ -n "$HF_TOKEN" ]] && info "HF token:     loaded from ${tf/#$HOME/~} (value hidden)" && break
    fi
  done
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  # vLLM/huggingface_hub read HF_TOKEN, but some paths still look at these — set all.
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
else
  # No token anywhere — say so LOUDLY. Downloads still work, just slowly.
  echo ""
  warn "════════════════════════════════════════════════════════════════════════════"
  warn "  NO HUGGINGFACE TOKEN FOUND."
  warn ""
  warn "  These models are large (Qwen3-Coder-30B ≈ 61 GB, the 80B ≈ 160 GB). Without"
  warn "  a token you hit HuggingFace's anonymous rate limits, and the first download"
  warn "  will be EXTREMELY SLOW — it can crawl for hours or stall entirely."
  warn ""
  warn "  We strongly recommend getting a FREE token and configuring it here:"
  warn "    1. Create one (read scope is enough): https://huggingface.co/settings/tokens"
  warn "    2. Save it to a gitignored file at the repo root:"
  warn "         echo 'hf_xxxxxxxxxxxxxxxxxxxx' > \"$REPO_ROOT/.hf_token\""
  warn "       (or export HF_TOKEN=hf_xxxx before running this script)"
  warn "    3. Re-run this script — it will pick the token up automatically."
  warn "════════════════════════════════════════════════════════════════════════════"
  echo ""
fi

# Use vLLM's native Torch top-k/top-p sampler instead of FlashInfer's. FlashInfer
# JIT-compiles CUDA sampling kernels at startup and hardcodes CUDA_HOME=/usr/local/cuda,
# which does NOT exist on the Deep Learning AMI (its toolkit lives at
# /opt/pytorch/cuda). The native sampler needs no runtime nvcc, so the server
# boots reliably and faster. (Verified fix on the Ubuntu 24.04 DLAMI, 2026-07.)
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Belt-and-suspenders: if anything else needs to JIT-compile against CUDA, point
# it at the toolkit the DLAMI actually ships rather than the missing default.
if [[ -z "${CUDA_HOME:-}" ]]; then
  for c in /opt/pytorch/cuda /usr/local/cuda; do
    [[ -x "$c/bin/nvcc" ]] && export CUDA_HOME="$c" && break
  done
fi

ARGS=(
  serve "$MODEL"
  --tensor-parallel-size "$TP"
  --host 127.0.0.1
  --port "$PORT"
  --served-model-name "$SERVED_NAME"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --enable-prefix-caching
)

# Extend the context past the model's native window with YaRN rope scaling.
# vLLM 0.25 removed the old --rope-scaling CLI flag, so patch the Hugging Face
# model config through --hf-overrides instead. ROPE_SCALING accepts either a
# bare YaRN factor (e.g. 4) or a full JSON object.
if [[ -n "$ROPE_SCALING" ]]; then
  if [[ "$ROPE_SCALING" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    # Bare number → YaRN factor. Derive the native window from MAX_MODEL_LEN / factor
    # (integer) so the two stay consistent: factor 4 + len 131072 ⇒ original 32768.
    ORIG_CTX=$(awk "BEGIN{printf \"%d\", $MAX_MODEL_LEN / $ROPE_SCALING}")
    ROPE_JSON="{\"rope_type\":\"yarn\",\"factor\":$ROPE_SCALING,\"original_max_position_embeddings\":$ORIG_CTX}"
    info "Rope scaling: YaRN factor $ROPE_SCALING (native ${ORIG_CTX} → ${MAX_MODEL_LEN} tokens)"
  else
    # Anything else is assumed to be a full JSON object; pass it through verbatim.
    ROPE_JSON="$ROPE_SCALING"
    info "Rope scaling: $ROPE_JSON"
  fi
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
  ARGS+=( --hf-overrides "{\"rope_scaling\":$ROPE_JSON}" )
fi

# Cap the number of concurrent sequences. Mostly you leave this unset (vLLM defaults
# to 256), but HYBRID models with Mamba/linear-attention layers (e.g.
# Qwen3-Coder-Next) allocate one Mamba state-cache block per in-flight sequence, and
# on a VRAM-tight node there may be fewer blocks than the default 256 — vLLM then
# aborts at CUDA-graph capture with "max_num_seqs (256) exceeds available Mamba cache
# blocks (N)". Setting MAX_NUM_SEQS at or below that N (the error prints it) fixes it.
if [[ -n "$MAX_NUM_SEQS" ]]; then
  ARGS+=( --max-num-seqs "$MAX_NUM_SEQS" )
  info "Max seqs:     $MAX_NUM_SEQS (concurrent sequences cap)"
fi

# Enable tool calling unless explicitly disabled — agentic clients need it.
if [[ -n "$TOOL_PARSER" && "$TOOL_PARSER" != "none" ]]; then
  ARGS+=( --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" )
  info "Tools:        enabled (parser: $TOOL_PARSER)"
else
  info "Tools:        disabled (plain completion server)"
fi

# Reasoning parser — separates thinking into a dedicated content block.
if [[ -n "$REASONING_PARSER" ]]; then
  ARGS+=( --reasoning-parser "$REASONING_PARSER" )
  info "Reasoning:    enabled (parser: $REASONING_PARSER)"
fi

# Extra args — pass-through for model-specific flags (e.g. --trust-remote-code).
if [[ -n "$EXTRA_ARGS" ]]; then
  read -ra EXTRA_ARRAY <<< "$EXTRA_ARGS"
  ARGS+=( "${EXTRA_ARRAY[@]}" )
  info "Extra args:   $EXTRA_ARGS"
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$LOG_NAME"

if [[ "$FOREGROUND" -eq 1 ]]; then
  info "Starting vLLM in the foreground (Ctrl-C to stop). First run downloads weights."
  info "Full log streams to the console AND is tee'd to: $LOG"
  info "(logs/ is gitignored — the log is never committed)"
  # tee: everything vLLM prints goes to the terminal and to the log file at once.
  exec "$VLLM_BIN" "${ARGS[@]}" 2>&1 | tee "$LOG"
fi

info "Starting vLLM in the background. Full log tee'd to: $LOG"
info "(logs/ is gitignored — the log is never committed)"
info "First run downloads the weights (30B ≈ 61 GB) — allow several minutes."
# tee inside the background job so the file captures everything; the foreground
# shell stays free to poll for readiness below.
#
# setsid puts the launcher in its own session, which is what makes `--stop` able
# to signal the whole process GROUP. That matters because the TP workers rename
# themselves and so cannot be matched by name -- but they do inherit the group.
# Without setsid, a background job in a non-interactive script inherits the
# SCRIPT's process group, so a group kill would also kill the calling driver.
# The pid is written from inside the child, whose $$ is the new session leader
# (and therefore the group id), rather than from $! out here, which is the same
# pid only when setsid happens not to fork.
# `|| true`: in a sticky /tmp this unlink fails if the file belongs to another
# user, and `set -e` would otherwise abort the launch instead of falling back.
rm -f "$PID_FILE" 2>/dev/null || true
nohup setsid bash -c "echo \$\$ > '$PID_FILE'; '$VLLM_BIN' $(printf '%q ' "${ARGS[@]}") 2>&1 | tee '$LOG'" >/dev/null 2>&1 &
for _ in $(seq 1 50); do [[ -s "$PID_FILE" ]] && break; sleep 0.1; done
SERVE_PID="$(cat "$PID_FILE" 2>/dev/null || echo $!)"
info "Launcher PID $SERVE_PID (saved to $PID_FILE)"
info "Tail the log live with:  tail -f $LOG"
echo ""

# Poll for readiness. Weight download can be slow, so wait generously.
info "Waiting for the server to become ready (up to 30 min for first download)..."
for i in $(seq 1 360); do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo ""; fail "vLLM process exited early. Check: tail -50 $LOG"
  fi
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo ""
    ok "Server ready at http://127.0.0.1:$PORT/v1"
    curl -s "http://127.0.0.1:$PORT/v1/models" | \
      "$VLLM_ENV/bin/python" -c "import json,sys; [print('       served model:', m['id']) for m in json.load(sys.stdin).get('data',[])]" 2>/dev/null || true
    echo ""
    echo "Next steps:"
    echo "  1. Verify inference:   ./vllm-verify.sh"
    echo "  2. Tunnel from laptop: LOCAL_MODEL_PORT=$PORT G6E_IP=<ip> ./tunnel.sh start"
    echo "  3. Stop the server:    PORT=$PORT ./vllm-serve.sh --stop   (or: kill \$(cat $PID_FILE))"
    exit 0
  fi
  sleep 5
done
echo ""
fail "Server did not become ready in time. Check: tail -50 $LOG"
