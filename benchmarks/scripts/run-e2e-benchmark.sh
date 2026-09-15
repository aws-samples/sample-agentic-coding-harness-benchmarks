#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# run-e2e-benchmark.sh -- one end-to-end SWE benchmark run, from pre-flight
# checks through scoring, for a model on any of the three hosting paths.
#
# Three inputs (flags or positional):
#   --provider   bedrock | litellm | vllm
#                  bedrock  = Anthropic models on Bedrock (provider=bedrock)
#                  litellm  = open-weight models on Bedrock via the LiteLLM
#                             mantle proxy (provider=endpoint at the proxy)
#                  vllm     = self-hosted model on a local vLLM server
#                             (provider=endpoint at :8000)
#   --model      the model id / served-model-name (e.g. qwen3.6-35b,
#                us.anthropic.claude-opus-4-8, moonshotai.kimi-k2-thinking)
#   --dataset    dataset YAML, relative to benchmarks/ (e.g.
#                dataset/mcp-gateway-registry.yaml)
#
# What it does, failing LOUDLY at the first problem:
#   0. Pre-flight: tools present, endpoint reachable, model served/valid,
#      AWS creds if needed, and -- key -- no pre-existing artifact folders that
#      would stall the headless /swe2 overwrite prompt.
#   1. Run the SWE benchmark harness over every task in the dataset.
#   2. Score the produced artifacts with the codex judge.
# At every step it prints the exact command to watch progress (tail / status).
#
# This script does NOT start vLLM or the LiteLLM proxy for you -- those are
# separate long-lived services with their own lifecycle scripts. It checks they
# are up and tells you how to start them if not. See:
#   benchmarks/docs/end-to-end-self-hosted-run.md   (the full run-book)
#   benchmarks/docs/path-*.md                        (per-path setup)
#
# Usage:
#   ./scripts/run-e2e-benchmark.sh --provider vllm --model qwen3.6-35b \
#       --dataset dataset/mcp-gateway-registry.yaml
#   ./scripts/run-e2e-benchmark.sh vllm qwen3.6-35b dataset/mcp-gateway-registry.yaml
#   ./scripts/run-e2e-benchmark.sh --provider vllm --model qwen3.6-35b \
#       --dataset dataset/hello-world.yaml --count 1 --yes
#
# Optional flags:
#   --agent NAME           coding agent that runs the task: claude (Claude Code,
#                          default), pi, omp (oh-my-pi), codex (OpenAI Codex) or
#                          kiro. Same task either way. claude, pi, omp and codex
#                          support every --provider: an OpenAI-compatible
#                          endpoint (vllm/litellm) or native Amazon Bedrock.
#                          codex speaks only the Responses API, so a vllm run
#                          needs a tool-call parser that accepts
#                          Responses-shaped tools (qwen3_coder, hermes -- NOT
#                          minicpm5xml, dots, hy_v3, hy_v4, rust, step3,
#                          step3p5). See issue #183.
#   --skill NAME           SWE skill: swe3 (default, single-agent, no subagents)
#                          or swe2 (multi-agent fan-out). Same six artifacts. The
#                          default maps to the canonical harness folder; the
#                          non-default swe2 lands under <harness>-swe2 so the two
#                          never overwrite each other.
#   --count N              run only the first N tasks (0 = all, default)
#   --tasks a,b,c          run only these task ids (comma-separated); scopes the
#                          folder-clear and the judge to the same set. Useful to
#                          re-run a single task that failed. Mutually informative
#                          with --count; --tasks selects by id, --count by position.
#   --max-output-tokens N  override the per-response output cap for this run
#                          (e.g. 4096 on a small-window model so the prompt has
#                          usable input room; usable input ~= window - this)
#   --max-retries N        override retries for a TRANSIENT task failure (a task
#                          that ran out of turns is never retried); 0 = none
#   --timeout-seconds N    override the per-task wall-clock timeout (raise it for
#                          a slow/dense model that produces artifacts but does not
#                          return before the default cutoff)
#   --tensor-parallel-size N  record the vLLM tensor-parallel size in the metrics
#                          serving block (provenance only; does not change serving)
#   --precision NAME       record the served weight precision (e.g. BF16, FP8) in
#                          the metrics serving block
#   --skip-judge           run the harness only; score later
#   --yes                  clear pre-existing artifact folders without asking
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$BENCHMARKS_DIR")"
VLLM_DIR="$REPO_ROOT/self-hosted/vllm"

# Defaults
AGENT="claude"            # coding agent: claude (Claude Code) or pi (pi agent)
SKILL="swe3"              # SWE skill: swe3 (single-agent, default) or swe2 (multi-agent)
PROVIDER=""
MODEL=""
DATASET=""
COUNT="0"                 # 0 = all tasks
TASKS=""                  # comma-separated task ids; empty = all
MAX_OUTPUT_TOKENS=""      # empty = use the config value
TIMEOUT_SECONDS=""        # empty = use the config value
MAX_RETRIES=""            # empty = use the config value
TENSOR_PARALLEL_SIZE=""   # empty = use the config value
PRECISION=""              # empty = use the config value
ENDPOINT=""               # derived from provider unless overridden
AWS_REGION_ARG="${AWS_REGION:-us-east-1}"
ASSUME_YES=0
SKIP_JUDGE=0
CONFIG="$BENCHMARKS_DIR/config/runner.yaml"

# Endpoints per path
VLLM_ENDPOINT="http://127.0.0.1:8000"
LITELLM_ENDPOINT="http://127.0.0.1:4000"

# --- pretty output -----------------------------------------------------------
info()  { printf '\033[0;36m[info]\033[0m  %s\n' "$1"; }
ok()    { printf '\033[0;32m[ok]\033[0m    %s\n' "$1"; }
warn()  { printf '\033[0;33m[warn]\033[0m  %s\n' "$1"; }
step()  { printf '\n\033[1;35m=== %s ===\033[0m\n' "$1"; }
die()   { printf '\033[0;31m[FAIL]\033[0m  %s\n' "$1" >&2; exit 1; }

usage() {
    sed -n '3,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# --- arg parsing -------------------------------------------------------------
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)    AGENT="${2:?--agent needs a value}"; shift 2 ;;
        --skill)    SKILL="${2:?--skill needs a value}"; shift 2 ;;
        --provider) PROVIDER="${2:?--provider needs a value}"; shift 2 ;;
        --model)    MODEL="${2:?--model needs a value}"; shift 2 ;;
        --dataset)  DATASET="${2:?--dataset needs a value}"; shift 2 ;;
        --count)    COUNT="${2:?--count needs a value}"; shift 2 ;;
        --tasks)    TASKS="${2:?--tasks needs a value}"; shift 2 ;;
        --max-output-tokens) MAX_OUTPUT_TOKENS="${2:?--max-output-tokens needs a value}"; shift 2 ;;
        --max-retries) MAX_RETRIES="${2:?--max-retries needs a value}"; shift 2 ;;
        --timeout-seconds) TIMEOUT_SECONDS="${2:?--timeout-seconds needs a value}"; shift 2 ;;
        --tensor-parallel-size) TENSOR_PARALLEL_SIZE="${2:?--tensor-parallel-size needs a value}"; shift 2 ;;
        --precision) PRECISION="${2:?--precision needs a value}"; shift 2 ;;
        --endpoint) ENDPOINT="${2:?--endpoint needs a value}"; shift 2 ;;
        --aws-region) AWS_REGION_ARG="${2:?--aws-region needs a value}"; shift 2 ;;
        --config)   CONFIG="${2:?--config needs a value}"; shift 2 ;;
        --yes|-y)   ASSUME_YES=1; shift ;;
        --skip-judge) SKIP_JUDGE=1; shift ;;
        -h|--help)  usage 0 ;;
        --*)        die "unknown flag: $1 (see --help)" ;;
        *)          POSITIONAL+=("$1"); shift ;;
    esac
done

# Positional fallback: provider model dataset
[[ -z "$PROVIDER" && ${#POSITIONAL[@]} -ge 1 ]] && PROVIDER="${POSITIONAL[0]}"
[[ -z "$MODEL"    && ${#POSITIONAL[@]} -ge 2 ]] && MODEL="${POSITIONAL[1]}"
[[ -z "$DATASET"  && ${#POSITIONAL[@]} -ge 3 ]] && DATASET="${POSITIONAL[2]}"

# kiro-cli only drives Kiro's own managed models (no vllm/litellm/bedrock
# routing), so agent=kiro forces provider=kiro -- the sole valid pairing. Do this
# BEFORE the provider checks so `--agent kiro` works with or without --provider.
[[ "$AGENT" == "kiro" ]] && PROVIDER="kiro"

# --- validate the three inputs ----------------------------------------------
[[ -n "$PROVIDER" ]] || die "provider is required (bedrock | litellm | vllm | kiro). See --help."
[[ -n "$MODEL"    ]] || die "model is required. See --help."
[[ -n "$DATASET"  ]] || die "dataset is required (e.g. dataset/mcp-gateway-registry.yaml). See --help."

case "$PROVIDER" in
    bedrock|litellm|vllm|kiro) ;;
    *) die "invalid provider '$PROVIDER'. Must be one of: bedrock, litellm, vllm, kiro." ;;
esac

case "$AGENT" in
    claude|pi|omp|kiro|codex) ;;
    *) die "invalid agent '$AGENT'. Must be one of: claude, pi, omp, kiro, codex." ;;
esac
case "$SKILL" in
    swe2|swe3) ;;
    *) die "invalid skill '$SKILL'. Must be one of: swe2, swe3." ;;
esac
# pi supports both an OpenAI-compatible endpoint (vllm/litellm) and native Amazon
# Bedrock (it bundles the AWS SDK bedrock-runtime client), so no agent/provider
# combination is rejected here.

# Resolve the dataset path relative to benchmarks/ and confirm it exists.
DATASET_PATH="$DATASET"
[[ "$DATASET_PATH" = /* ]] || DATASET_PATH="$BENCHMARKS_DIR/$DATASET"
[[ -f "$DATASET_PATH" ]] || die "dataset file not found: $DATASET_PATH"

# Map provider -> harness --provider and default endpoint.
case "$PROVIDER" in
    bedrock)  HARNESS_PROVIDER="bedrock";  DEFAULT_ENDPOINT="" ;;
    litellm)  HARNESS_PROVIDER="endpoint"; DEFAULT_ENDPOINT="$LITELLM_ENDPOINT" ;;
    vllm)     HARNESS_PROVIDER="endpoint"; DEFAULT_ENDPOINT="$VLLM_ENDPOINT" ;;
    kiro)     HARNESS_PROVIDER="kiro";     DEFAULT_ENDPOINT="" ;;
    *) die "invalid provider '$PROVIDER'. Must be one of: bedrock, litellm, vllm, kiro." ;;
esac
[[ -n "$ENDPOINT" ]] || ENDPOINT="$DEFAULT_ENDPOINT"

cd "$BENCHMARKS_DIR"

# =============================================================================
step "Step 0 - Pre-flight checks"
# =============================================================================

# uv is the entry point for everything below.
command -v uv >/dev/null 2>&1 || die "uv is not installed or not on PATH. Install: https://docs.astral.sh/uv/"
ok "uv found: $(command -v uv)"

# The harness env must be synced.
[[ -d "$BENCHMARKS_DIR/.venv" ]] || die "benchmarks venv missing. Run: (cd $BENCHMARKS_DIR && uv sync)"
ok "benchmarks venv present"

# The runner config must exist (harness reads it).
if [[ ! -f "$CONFIG" ]]; then
    die "runner config not found: $CONFIG
       Create it once: (cd $BENCHMARKS_DIR && cp config/runner.example.yaml config/runner.yaml)"
fi
ok "runner config: $CONFIG"

# The coding-agent CLIs this benchmark drives must be installed:
#   - the chosen --agent : produces the artifacts ('claude -p', 'pi -p',
#     'omp -p', 'kiro-cli chat' or 'codex exec').
#   - codex : the judge runs 'codex exec' to score them (unless --skip-judge).
# Check both here, up front, so a missing codex fails fast instead of after the
# entire (long) harness run has already completed.
if [[ "$AGENT" == "pi" ]]; then
    command -v pi >/dev/null 2>&1 || die "pi CLI not found on PATH (--agent pi runs 'pi -p'). Install the pi coding agent (needs Node >=22)."
    ok "pi CLI found: $(command -v pi)"
elif [[ "$AGENT" == "omp" ]]; then
    command -v omp >/dev/null 2>&1 || die "omp CLI not found on PATH (--agent omp runs 'omp -p'). Install it: curl -fsSL https://omp.sh/install | sh"
    ok "omp CLI found: $(command -v omp)"
elif [[ "$AGENT" == "kiro" ]]; then
    command -v kiro-cli >/dev/null 2>&1 || die "kiro-cli not found on PATH (--agent kiro runs 'kiro-cli chat'). Install it: curl -fsSL https://cli.kiro.dev/install | bash (see docs/kiro-cli-setup.md), then sign in with 'kiro-cli login'."
    ok "kiro-cli found: $(command -v kiro-cli)"
elif [[ "$AGENT" == "codex" ]]; then
    command -v codex >/dev/null 2>&1 || die "codex CLI not found on PATH (--agent codex runs 'codex exec'). Install codex."
    ok "codex CLI found: $(command -v codex) ($(codex --version 2>/dev/null || echo 'version unknown'))"
else
    command -v claude >/dev/null 2>&1 || die "claude CLI not found on PATH (the harness runs 'claude -p'). Install Claude Code."
    ok "claude CLI found: $(command -v claude)"
fi
if [[ "$SKIP_JUDGE" -eq 1 ]]; then
    command -v codex >/dev/null 2>&1 && ok "codex CLI found: $(command -v codex)" \
        || warn "codex CLI not found, but --skip-judge is set, so scoring is skipped."
else
    command -v codex >/dev/null 2>&1 || die "codex CLI not found on PATH (the judge runs 'codex exec'). Install codex, or re-run with --skip-judge."
    ok "codex CLI found: $(command -v codex)"
fi

# LOUD EXPECTATION: both CLIs must be wired to Amazon Bedrock. The harness points
# claude at the model under test (Bedrock, the LiteLLM proxy, or local vLLM), but
# the codex JUDGE always calls Amazon Bedrock for its scoring model, and on the
# bedrock path claude does too. This benchmark assumes both 'claude' and 'codex'
# are already configured to reach Amazon Bedrock (credentials/region/base URL) on
# THIS machine. It does not configure them for you -- if either is pointed
# elsewhere or unauthenticated, the run or the scoring will fail.
warn "EXPECTATION: 'claude' and 'codex' on this machine are assumed to be wired to Amazon Bedrock (the judge always calls Bedrock; on --provider bedrock, claude does too). This script does not configure them."

# Per-path readiness.
case "$PROVIDER" in
    bedrock)
        info "Path: Anthropic models directly on Amazon Bedrock (provider=bedrock)."
        command -v aws >/dev/null 2>&1 || die "aws CLI not found; needed for Bedrock credentials."
        aws sts get-caller-identity >/dev/null 2>&1 \
            || die "AWS credentials not usable (aws sts get-caller-identity failed). Configure creds for region $AWS_REGION_ARG."
        ok "AWS credentials OK (region $AWS_REGION_ARG)"
        case "$MODEL" in
            *anthropic*|*claude*) ;;
            *) warn "provider=bedrock is Anthropic-only; '$MODEL' does not look like an Anthropic id. Non-Anthropic Bedrock models need --provider litellm." ;;
        esac
        ;;
    litellm)
        info "Path: open-weight models on Amazon Bedrock via the LiteLLM proxy (provider=endpoint at $ENDPOINT)."
        if ! curl -s -m 5 "$ENDPOINT/health/liveliness" >/dev/null 2>&1; then
            die "LiteLLM proxy not reachable at $ENDPOINT.
       Start it: (cd $BENCHMARKS_DIR && ./scripts/bedrock-mantle-proxy.sh)
       Status:   (cd $BENCHMARKS_DIR && ./scripts/bedrock-mantle-proxy.sh --status)
       Log:      tail -f $BENCHMARKS_DIR/.litellm.log"
        fi
        ok "LiteLLM proxy healthy at $ENDPOINT"
        ;;
    vllm)
        info "Path: self-hosted model on a local vLLM server (provider=endpoint at $ENDPOINT)."
        if ! curl -s -m 5 "$ENDPOINT/health" >/dev/null 2>&1; then
            die "vLLM server not reachable at $ENDPOINT.
       Start it: (cd $VLLM_DIR/scripts && MODEL=... SERVED_NAME=$MODEL MAX_MODEL_LEN=200000 ./vllm-serve.sh)
       Log:      tail -f $VLLM_DIR/logs/vllm-serve.log"
        fi
        # Confirm the requested served-model-name is actually the one loaded.
        SERVED="$(curl -s -m 5 "$ENDPOINT/v1/models" 2>/dev/null \
            | uv run python -c 'import sys,json; d=json.load(sys.stdin); print(",".join(m["id"] for m in d.get("data",[])))' 2>/dev/null || true)"
        if [[ -z "$SERVED" ]]; then
            warn "vLLM is up but /v1/models did not return a model list; continuing."
        elif [[ ",$SERVED," != *",$MODEL,"* ]]; then
            die "vLLM is serving [$SERVED], not '$MODEL'. Pass --model matching the served-model-name, or restart vLLM with SERVED_NAME=$MODEL."
        else
            ok "vLLM serving '$MODEL' at $ENDPOINT"
        fi
        # Read the live server's context window (max_model_len) so we can
        # calibrate Claude Code's auto-compaction to it. Claude Code cannot
        # detect a custom model's window; without this the conversation grows
        # until vLLM rejects the request (500) and the client retries forever.
        VLLM_CONTEXT_WINDOW="$(curl -s -m 5 "$ENDPOINT/v1/models" 2>/dev/null \
            | uv run python -c 'import sys,json
d=json.load(sys.stdin).get("data",[])
w=next((m.get("max_model_len") for m in d if m.get("max_model_len")), None)
print(w if w else "")' 2>/dev/null || true)"
        if [[ -n "$VLLM_CONTEXT_WINDOW" ]]; then
            ok "vLLM context window (max_model_len): $VLLM_CONTEXT_WINDOW -- auto-compaction will be calibrated to it"
        else
            warn "Could not read vLLM max_model_len from /v1/models; falling back to the config's context_window. Long tasks may overflow the window."
        fi
        # The DuckDB metrics collector is optional but recommended on this path.
        if uv run --project "$VLLM_DIR" python -c 'import sys' >/dev/null 2>&1; then :; fi
        if pgrep -f "collect_metrics" >/dev/null 2>&1; then
            ok "vLLM metrics collector is running (DuckDB time series is being captured)"
        else
            warn "vLLM metrics collector is NOT running. To capture a GPU time series:
       Clear + start: see benchmarks/docs/end-to-end-self-hosted-run.md (Step 2)
       Start:         (cd $VLLM_DIR/scripts && ./vllm-metrics.sh start)
       Status:        (cd $VLLM_DIR/scripts && ./vllm-metrics.sh status)"
        fi
        ;;
esac

# The critical check: pre-existing artifact folders stall the headless /swe2 run.
# Scope the check/clear to --tasks when given, so a single-task re-run does not
# touch (or clear) the other tasks' existing folders.
TASKS_ARG=()
[[ -n "$TASKS" ]] && TASKS_ARG=(--tasks "$TASKS")
info "Checking for pre-existing artifact folders (these stall the headless /swe2 overwrite prompt)..."
set +e
uv run python scripts/preflight_check.py --dataset "$DATASET" --model "$MODEL" --agent "$AGENT" --skill "$SKILL" "${TASKS_ARG[@]}" --check
PREFLIGHT_RC=$?
set -e
if [[ "$PREFLIGHT_RC" -eq 2 ]]; then
    if [[ "$ASSUME_YES" -eq 1 ]]; then
        warn "Clearing pre-existing artifact folders (--yes given)..."
        uv run python scripts/preflight_check.py --dataset "$DATASET" --model "$MODEL" --agent "$AGENT" --skill "$SKILL" "${TASKS_ARG[@]}" --clear
    else
        die "Pre-existing artifact folders would stall the run. Re-run with --yes to clear them automatically, or clear manually:
       uv run python scripts/preflight_check.py --dataset $DATASET --model $MODEL --agent $AGENT --skill $SKILL ${TASKS_ARG[*]} --clear"
    fi
elif [[ "$PREFLIGHT_RC" -ne 0 ]]; then
    die "pre-flight folder check failed (exit $PREFLIGHT_RC)."
fi
ok "No blocking artifact folders."

ok "Pre-flight complete."

# The harness clones each task's repo into <clone_dir>/swe-clone-<task-id>/ and
# removes it in its own finally block after each task. Register a trap as a
# backstop so a killed or crashed run does not leave clones behind: it removes
# exactly this dataset's task dirs (never a broad glob, which would hit
# swe-judge-repos or the swe-benchmark-data output dir).
CLONE_DIRS="$(uv run python -c "import sys; sys.path.insert(0,'scripts')
from dataset_loader import load_dataset
from runner_config import load_runner_config
import importlib.util
s=importlib.util.spec_from_file_location('h','scripts/run-swe-headless.py')
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
cfg=load_runner_config('$CONFIG', {'model':'$MODEL','dataset':'$DATASET'})
d=load_dataset('$DATASET_PATH')
for t in d.tasks:
    print(f'{cfg.clone_dir}/swe-clone-{m._safe_task_slug(t.id)}')" 2>/dev/null || true)"

_cleanup_clones() {
    [[ -z "$CLONE_DIRS" ]] && return 0
    while IFS= read -r dir; do
        [[ -n "$dir" && -d "$dir" ]] && rm -rf -- "$dir"
    done <<< "$CLONE_DIRS"
    # Always succeed: this is a best-effort backstop on EXIT, and its return
    # value becomes the script's exit code. The harness already removes each
    # clone after its task, so the -d test above is normally false (nothing to
    # remove) -- without this, that falsy test would report a spurious failure.
    return 0
}
trap _cleanup_clones EXIT

# =============================================================================
step "Step 1 - Run the SWE benchmark"
# =============================================================================
BENCH_ARGS=(--config "$CONFIG" --agent "$AGENT" --skill "$SKILL" --provider "$HARNESS_PROVIDER" --model "$MODEL" --dataset "$DATASET")
# --stream/--verbose is the Claude Code live-trace mode; pi emits its own JSON
# event stream and kiro-cli streams plain text -- neither has an equivalent, so
# only add it for the claude agent.
[[ "$AGENT" == "claude" ]] && BENCH_ARGS+=(--stream --verbose)
[[ "$COUNT" != "0" ]] && BENCH_ARGS+=(--count "$COUNT")
[[ -n "$TASKS" ]] && BENCH_ARGS+=(--tasks "$TASKS")
[[ "$HARNESS_PROVIDER" == "endpoint" ]] && BENCH_ARGS+=(--endpoint "$ENDPOINT")
[[ "$PROVIDER" == "bedrock" ]] && BENCH_ARGS+=(--aws-region "$AWS_REGION_ARG")
# On the vllm path, calibrate auto-compaction to the live server's window.
[[ -n "${VLLM_CONTEXT_WINDOW:-}" ]] && BENCH_ARGS+=(--context-window "$VLLM_CONTEXT_WINDOW")
[[ -n "$MAX_OUTPUT_TOKENS" ]] && BENCH_ARGS+=(--max-output-tokens "$MAX_OUTPUT_TOKENS")
[[ -n "$MAX_RETRIES" ]] && BENCH_ARGS+=(--max-retries "$MAX_RETRIES")
[[ -n "$TIMEOUT_SECONDS" ]] && BENCH_ARGS+=(--timeout-seconds "$TIMEOUT_SECONDS")
[[ -n "$TENSOR_PARALLEL_SIZE" ]] && BENCH_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
[[ -n "$PRECISION" ]] && BENCH_ARGS+=(--precision "$PRECISION")

SLUG="$(uv run python -c "import sys; sys.path.insert(0,'scripts'); from runner_config import model_to_slug; print(model_to_slug('$MODEL', normalize_dots='$AGENT'=='kiro'))")"
# Harness folder level (claude -> claude-code, pi -> pi), from the single source of
# truth. Skill (swe2/swe3) is its OWN path level, so artifacts live under
# <model>/<harness>/<skill>/<repo>/<task> and the judge/summary target that tree.
HARNESS_SLUG="$(uv run python -c "import sys; sys.path.insert(0,'scripts'); from runner_config import HARNESS_SLUGS; print(HARNESS_SLUGS['$AGENT'])")"
info "Command:"
info "  uv run scripts/run-swe-headless.py ${BENCH_ARGS[*]}"
info "Artifacts will land under: swe-benchmark-data/$SLUG/$HARNESS_SLUG/$SKILL/<repo>/<task>/"
info "Watch GPU metrics (vllm path):  cd $VLLM_DIR && uv run python -m clients.build_dashboard && open benchmark-output/dashboard.html"
echo
uv run scripts/run-swe-headless.py "${BENCH_ARGS[@]}" \
    || die "benchmark run failed. Inspect the trace above; per-task errors are also recorded in each task's metrics.json."
ok "Benchmark run complete."

# =============================================================================
step "Step 2 - Score the artifacts (codex judge)"
# =============================================================================
if [[ "$SKIP_JUDGE" -eq 1 ]]; then
    warn "Skipping the judge (--skip-judge). Score later with:"
    warn "  (cd $BENCHMARKS_DIR/scripts && uv run python codex_judge.py --recursive --no-overwrite --folder ../swe-benchmark-data)"
else
    command -v codex >/dev/null 2>&1 || die "codex CLI not found on PATH (the judge runs 'codex exec'). Install codex, or re-run with --skip-judge."
    # Judge only the folders this model+dataset just produced: point at the
    # <model-slug>/<scope> subtree and let --recursive + --no-overwrite handle it.
    # The scope is the dataset's output_scope when it sets one, else the repo
    # name -- the same rule _artifact_dir uses, so this must not be re-derived
    # here from the repo alone.
    SCOPE_SUBDIR="$(uv run python -c "import sys; sys.path.insert(0,'scripts'); from dataset_loader import load_dataset; d=load_dataset('$DATASET_PATH'); import importlib.util,pathlib; s=importlib.util.spec_from_file_location('h','scripts/run-swe-headless.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(d.scope_for(m._repo_name(d.tasks[0].repo)))")"
    JUDGE_TARGET="swe-benchmark-data/$SLUG/$HARNESS_SLUG/$SKILL/$SCOPE_SUBDIR"
    info "Command:"
    info "  (cd scripts && uv run python codex_judge.py --recursive --no-overwrite --folder ../$JUDGE_TARGET)"
    info "codex exec buffers output and prints only its final message per folder -- a few minutes each at high effort is normal."
    echo
    ( cd scripts && uv run python codex_judge.py --recursive --no-overwrite --folder "../$JUDGE_TARGET" ) \
        || die "judge run failed. See the log above; re-run just the judge with the command shown."
    ok "Scoring complete."

    # Write the machine-readable run-summary.json + human-readable run-summary.md
    # from the scored artifacts, so the run is summarized on disk for later
    # charting without re-parsing every task folder. Best-effort: a summary
    # failure must not fail an otherwise-good scored run.
    if uv run python scripts/summarize_run.py --folder "$JUDGE_TARGET" \
        --run-date "$(date -u +%Y-%m-%d)"; then
        ok "Run summary written: $JUDGE_TARGET/run-summary.{json,md}"
    else
        warn "Could not write run-summary (scores are still in each task's eval.json)."
    fi
fi

# =============================================================================
step "Done"
# =============================================================================
ok "End-to-end benchmark finished for provider=$PROVIDER model=$MODEL dataset=$DATASET"
info "Per-task results (metrics.json cost + eval.json quality) are under:"
info "  $BENCHMARKS_DIR/swe-benchmark-data/$SLUG/$HARNESS_SLUG/$SKILL/*/*/"
