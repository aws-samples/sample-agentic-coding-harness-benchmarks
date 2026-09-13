#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# setup-machine.sh -- inspect this instance and install everything this repo
# needs in order to run, including the GPU stack when a GPU is present.
#
# Two modes:
#   --check    (default) detect only. Prints the machine facts, what is already
#              present, and the exact install plan. Changes nothing.
#   --install  run that plan: install every missing component, echoing each one
#              as it completes, then print a summary table and next steps.
#
# Installed on EVERY instance (the core the harness needs):
#   git (+ a configured user.name/user.email), the GitHub CLI (gh), python3
#   >= 3.10, Node.js >= 22, uv, the claude / codex / pi CLIs, the AWS CLI,
#   pre-commit plus the repo hook, and a synced virtualenv for each of the two
#   uv projects (benchmarks/ and self-hosted/vllm/).
#
# Installed ONLY when an NVIDIA GPU is present (skipped loudly otherwise):
#   vLLM (via self-hosted/vllm/scripts/vllm-install.sh), nvtop, nvitop.
#
# Usage:
#   ./setup-machine.sh                       # detect and print the plan
#   ./setup-machine.sh --install             # install the plan (prompts once)
#   ./setup-machine.sh --install --yes       # install without prompting
#   ./setup-machine.sh --install --skip-gpu  # core only, even on a GPU box
#   ./setup-machine.sh --install --with-omp --with-kiro
#   ./setup-machine.sh --help
#
# Flags:
#   --check       detect only (default)
#   --install     perform the installs
#   --yes         do not prompt for confirmation
#   --skip-gpu    never install the GPU stack, even on a GPU instance
#   --with-omp    also install the omp agent (third-party 'curl | sh' installer)
#   --with-kiro   also install kiro-cli (third-party 'curl | bash' installer;
#                 needs an interactive 'kiro-cli login' afterwards)
#   --git-name    global git user.name  (or the GIT_USER_NAME env var)
#   --git-email   global git user.email (or the GIT_USER_EMAIL env var)
#
# git's global user.name/user.email are what gh and every commit attribute work
# to. This script never invents them: supply both, or it reports the identity as
# still needing to be set and tells you the two commands.
#
# Idempotent: anything already present is reported and left alone, so re-running
# after a partial failure only does what is left.
#
# Installing a CLI is NOT the same as wiring it to Amazon Bedrock. See
# benchmarks/docs/agent-cli-bedrock-setup.md for that step; this script prints
# the reminder at the end and never edits your CLI config.
# ---------------------------------------------------------------------------

# uv and everything `uv tool install` places go in ~/.local/bin, which a shell
# started before the install does not have on PATH. Without this, a second run
# re-detects those as missing and reinstalls them.
#
# APPENDED, not prepended, on purpose: this script invokes curl, dpkg, tee and
# apt-get under sudo, and a user-writable directory ahead of the system paths
# would let anything dropped in ~/.local/bin shadow those. Appending still finds
# uv and the uv-installed tools, which exist nowhere else.
export PATH="$PATH:$HOME/.local/bin"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BENCHMARKS_DIR="$REPO_ROOT/benchmarks"
VLLM_DIR="$REPO_ROOT/self-hosted/vllm"
VLLM_ENV_EXPLICIT="no"
[[ -n "${VLLM_ENV:-}" ]] && VLLM_ENV_EXPLICIT="yes"
VLLM_ENV="${VLLM_ENV:-$HOME/vllm-env}"

# vLLM plus its torch/CUDA wheels need roughly this much room before any model
# weights are downloaded. The p5en root volume is small, which is exactly the
# case p5en-h200-cuda-fixes.md tells you to move onto the NVMe.
VLLM_MIN_FREE_GB=40

# The Deep Learning AMI mounts its large ephemeral NVMe here.
DLAMI_NVME="/opt/dlami/nvme"

MIN_PYTHON="3.10"
MIN_NODE="22"

MODE="check"
ASSUME_YES="no"
SKIP_GPU="no"
WITH_OMP="no"
WITH_KIRO="no"
GIT_USER_NAME="${GIT_USER_NAME:-}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'
BOLD='\033[1m'; RESET='\033[0m'
info()   { echo -e "${BLUE}[info]${RESET}  $1"; }
ok()     { echo -e "${GREEN}[ok]${RESET}    $1"; }
done_()  { echo -e "${GREEN}${BOLD}[done]${RESET}  $1"; }
warn()   { echo -e "${YELLOW}[warn]${RESET}  $1"; }
die()    { echo -e "${RED}[fail]${RESET}  $1" >&2; exit 1; }
# Report a failure without exiting, so _run_installs can move on to the next
# component. Returns 1 for callers that propagate it as a component failure.
fail()   { echo -e "${RED}[fail]${RESET}  $1" >&2; return 1; }
header() { echo -e "\n${BOLD}=== $1 ===${RESET}"; }
rule()   { echo -e "${BOLD}=======================================================================${RESET}"; }

# Component registry. Every component has a label, a one-line reason it is
# needed, and a state filled in by detect(): present | plan | skip | manual.
COMP_ORDER=()
declare -A COMP_LABEL=()
declare -A COMP_WHY=()
declare -A COMP_STATE=()
declare -A COMP_DETAIL=()

GPU_PRESENT="no"
GPU_SUMMARY="none detected"
GPU_COUNT=0
INSTANCE_TYPE=""
BIG_VOLUME=""
VLLM_ENV_NOTE=""
NEEDS_SUDO="no"
INSTALLED_COUNT=0
FAILED_COUNT=0


_register() {
    local key="$1" label="$2" why="$3"
    COMP_ORDER+=("$key")
    COMP_LABEL["$key"]="$label"
    COMP_WHY["$key"]="$why"
    COMP_STATE["$key"]="plan"
    COMP_DETAIL["$key"]=""
}


_mark() {
    local key="$1" state="$2" detail="${3:-}"
    COMP_STATE["$key"]="$state"
    COMP_DETAIL["$key"]="$detail"
}


# True when $1 (found version) is >= $2 (minimum version).
_version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}


_have() {
    command -v "$1" >/dev/null 2>&1
}


_is_apt_system() {
    _have apt-get
}


_sudo_available() {
    _have sudo && sudo -n true 2>/dev/null
}


# Install a global npm package, falling back to sudo when the global prefix is
# root-owned (the usual case for a NodeSource install).
_npm_global() {
    local pkg="$1"
    if npm install -g "$pkg"; then
        return 0
    fi
    warn "global npm install failed unprivileged; retrying with sudo"
    sudo npm install -g "$pkg"
}


_apt_install() {
    local pkg="$1"
    _is_apt_system || die "$pkg needs an apt-based distribution; install it manually."
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg"
}


# Die unless an option that takes a value actually got one. Called directly (not
# in a command substitution) so that `die` exits the script rather than a subshell.
_require_value() {
    local flag="$1" value="$2"
    if [[ -z "$value" || "$value" == -* ]]; then
        die "$flag requires a value (e.g. $flag 'Your Name')"
    fi
}


_parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --check)     MODE="check" ;;
            --install)   MODE="install" ;;
            --yes|-y)    ASSUME_YES="yes" ;;
            --skip-gpu)  SKIP_GPU="yes" ;;
            --with-omp)  WITH_OMP="yes" ;;
            --with-kiro) WITH_KIRO="yes" ;;
            # Reject a missing or flag-shaped value rather than silently storing
            # "--install" as somebody's name, and rather than letting a bare
            # `shift` past the end abort the script with a naked exit 1.
            --git-name)  _require_value "$1" "${2:-}"; GIT_USER_NAME="$2";  shift ;;
            --git-email) _require_value "$1" "${2:-}"; GIT_USER_EMAIL="$2"; shift ;;
            -h|--help)   sed -n '5,45p' "$0" | sed 's/^# \{0,1\}//; s/^#$//'; exit 0 ;;
            *)           die "unknown argument '$1' (try --help)" ;;
        esac
        shift
    done
}


_report_machine() {
    header "Step 1 - What machine is this"
    local kernel distro cpus mem disk instance
    kernel="$(uname -sr)"
    distro="$( { grep -m1 '^PRETTY_NAME=' /etc/os-release 2>/dev/null || echo 'PRETTY_NAME=unknown'; } | cut -d'"' -f2)"
    cpus="$(nproc 2>/dev/null || echo '?')"
    mem="$(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo '?')"
    disk="$(df -h "$REPO_ROOT" 2>/dev/null | tail -1 | awk '{print $4" free of "$2}')"
    echo "  OS          : $distro ($kernel)"
    echo "  CPU / RAM   : ${cpus} vCPU, ${mem} RAM"
    echo "  Disk        : ${disk:-unknown} (at $REPO_ROOT)"

    # EC2 instance type, best effort via IMDSv2. A non-EC2 box just prints n/a.
    local token=""
    token="$(curl -sf -m 2 -X PUT 'http://169.254.169.254/latest/api/token' \
        -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true)"
    if [[ -n "$token" ]]; then
        instance="$(curl -sf -m 2 -H "X-aws-ec2-metadata-token: $token" \
            'http://169.254.169.254/latest/meta-data/instance-type' 2>/dev/null || true)"
    fi
    INSTANCE_TYPE="${instance:-}"
    echo "  EC2 instance: ${INSTANCE_TYPE:-n/a (not an EC2 instance, or IMDS unreachable)}"
    _detect_gpu
    echo "  GPU         : $GPU_SUMMARY"
}


_detect_gpu() {
    if _have nvidia-smi && nvidia-smi -L >/dev/null 2>&1 && [[ -n "$(nvidia-smi -L 2>/dev/null)" ]]; then
        GPU_PRESENT="yes"
        local count name driver
        count="$(nvidia-smi -L | wc -l | xargs)"
        GPU_COUNT="$count"
        name="$(nvidia-smi --query-gpu=name --format=csv,noheader | awk 'NR==1' | xargs)"
        driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk 'NR==1' | xargs)"
        GPU_SUMMARY="${count}x ${name} (driver ${driver})"
        return
    fi
    if _have lspci && lspci 2>/dev/null | grep -qi 'nvidia'; then
        GPU_SUMMARY="NVIDIA hardware present but no working nvidia-smi -- driver not installed"
        return
    fi
    GPU_SUMMARY="none detected (CPU-only instance)"
}


_detect_core() {
    _register git "git" "clone the target repos each benchmark task runs against"
    if _have git; then
        _mark git present "$(git --version | awk '{print $3}')"
    fi

    _register python "python3 >= $MIN_PYTHON" "the harness, judge and plot scripts are Python 3.10+"
    if _have python3; then
        local pyver
        pyver="$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo 0)"
        if _version_ge "$pyver" "$MIN_PYTHON"; then
            _mark python present "$pyver"
        else
            _mark python plan "found $pyver, need >= $MIN_PYTHON"
        fi
    fi

    _register gitconfig "git identity" "gh and every commit need a global user.name/user.email"
    local git_name git_email
    git_name="$(git config --global user.name 2>/dev/null || true)"
    git_email="$(git config --global user.email 2>/dev/null || true)"
    if [[ -n "$git_name" && -n "$git_email" ]]; then
        _mark gitconfig present "$git_name <$git_email>"
    elif [[ -n "$GIT_USER_NAME" && -n "$GIT_USER_EMAIL" ]]; then
        _mark gitconfig plan "will set: $GIT_USER_NAME <$GIT_USER_EMAIL>"
    else
        _mark gitconfig plan "NEEDS --git-name and --git-email (ask the user; never invent them)"
    fi

    _register gh "gh (GitHub CLI)" "opening the PRs this repo's workflow requires, and gh label list"
    if _have gh; then
        _mark gh present "$(gh --version 2>/dev/null | awk 'NR==1{print $3}')"
    fi

    _register uv "uv" "the repo's package manager -- AGENTS.md forbids using pip directly"
    if _have uv; then
        _mark uv present "$(uv --version 2>/dev/null | awk '{print $2}')"
    fi

    _register node "Node.js >= $MIN_NODE" "the claude, codex and pi CLIs are npm packages (pi needs Node >= 22)"
    if _have node; then
        local nodever nodemajor
        nodever="$(node -v 2>/dev/null | sed 's/^v//')"
        nodemajor="${nodever%%.*}"
        if [[ -n "$nodemajor" ]] && (( nodemajor >= MIN_NODE )); then
            _mark node present "$nodever"
        else
            _mark node plan "found $nodever, need >= $MIN_NODE"
        fi
    fi

    _register claude "claude (Claude Code)" "the default harness: the runner shells out to 'claude -p'"
    if _have claude; then _mark claude present "$(claude --version 2>/dev/null | awk 'NR==1')"; fi

    _register codex "codex" "the judge: codex_judge.py scores artifacts with 'codex exec'"
    if _have codex; then _mark codex present "$(codex --version 2>/dev/null | awk 'NR==1')"; fi

    _register pi "pi (pi coding agent)" "the second harness (--agent pi), used for most published results"
    if _have pi; then _mark pi present "$(pi --version 2>/dev/null | awk 'NR==1')"; fi

    _register aws "AWS CLI" "Bedrock credentials and the pre-flight identity check"
    if _have aws; then _mark aws present "$(aws --version 2>&1 | awk '{print $1}')"; fi

    _register precommit "pre-commit + repo hook" "CI fails on unformatted Python; the hook formats on every commit"
    if _have pre-commit && [[ -f "$REPO_ROOT/.git/hooks/pre-commit" ]]; then
        _mark precommit present "$(pre-commit --version 2>/dev/null | awk '{print $2}')"
    fi

    _register venv_benchmarks "benchmarks/ venv (uv sync)" "the harness project's dependencies (pydantic, matplotlib, ...)"
    if [[ -x "$BENCHMARKS_DIR/.venv/bin/python" ]]; then _mark venv_benchmarks present "$BENCHMARKS_DIR/.venv"; fi

    _register venv_vllm "self-hosted/vllm/ venv (uv sync)" "the client project's dependencies (openai, duckdb, ...)"
    if [[ -x "$VLLM_DIR/.venv/bin/python" ]]; then _mark venv_vllm present "$VLLM_DIR/.venv"; fi

    if [[ "$WITH_OMP" == "yes" ]]; then
        _register omp "omp (oh-my-pi)" "optional third harness (--agent omp), requested with --with-omp"
        if _have omp; then _mark omp present "$(omp --version 2>/dev/null | awk 'NR==1')"; fi
    fi
    if [[ "$WITH_KIRO" == "yes" ]]; then
        _register kiro "kiro-cli" "optional fourth harness (--agent kiro), requested with --with-kiro"
        if _have kiro-cli; then _mark kiro present "$(kiro-cli --version 2>/dev/null | awk 'NR==1')"; fi
    fi
}


_detect_gpu_stack() {
    _register nvtop "nvtop" "live per-GPU utilization TUI while a benchmark runs"
    _register nvitop "nvitop" "per-process GPU view: which PID holds which slice of VRAM"
    _register vllm "vLLM (in $VLLM_ENV)" "serves the open-weight model for the self-hosted path (Path 3)"

    if [[ "$GPU_PRESENT" != "yes" || "$SKIP_GPU" == "yes" ]]; then
        local reason="no NVIDIA GPU on this instance"
        if [[ "$SKIP_GPU" == "yes" ]]; then reason="--skip-gpu requested"; fi
        _mark nvtop skip "$reason"
        _mark nvitop skip "$reason"
        _mark vllm skip "$reason"
        return
    fi

    if _have nvtop; then _mark nvtop present "$(nvtop --version 2>/dev/null | awk '{print $NF}')"; fi
    if _have nvitop; then _mark nvitop present "$(nvitop --version 2>/dev/null | awk '{print $NF}')"; fi
    if [[ -x "$VLLM_ENV/bin/python" ]] && "$VLLM_ENV/bin/python" -c 'import vllm' 2>/dev/null; then
        _mark vllm present "$("$VLLM_ENV/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
    fi
}


# Free GB on the filesystem holding $1 (the nearest existing ancestor).
_free_gb() {
    local target="$1"
    while [[ -n "$target" && ! -d "$target" ]]; do
        target="$(dirname "$target")"
    done
    df -BG --output=avail "$target" 2>/dev/null | tail -1 | tr -dc '0-9'
}


# Largest writable real volume with room for vLLM, or empty if there is none.
# tmpfs is excluded on purpose: /dev/shm can advertise a terabyte, but it is RAM,
# and a multi-GB venv written there is a memory leak with a mount point.
_pick_big_volume() {
    local avail target best="" best_gb=0
    while read -r avail target; do
        [[ -z "${target:-}" ]] && continue
        avail="${avail%G}"
        [[ "$avail" =~ ^[0-9]+$ ]] || continue
        (( avail < VLLM_MIN_FREE_GB )) && continue
        [[ -w "$target" ]] || continue
        # The Deep Learning AMI's ephemeral NVMe is what the p5en run-book names,
        # so prefer it outright rather than by size.
        if [[ "$target" == "$DLAMI_NVME" ]]; then
            echo "$target"
            return 0
        fi
        if (( avail > best_gb )); then
            best_gb="$avail"
            best="$target"
        fi
    done < <(df -BG --output=avail,target -x tmpfs -x devtmpfs -x efivarfs \
                -x squashfs -x overlay -x iso9660 2>/dev/null | tail -n +2)
    echo "$best"
}


# The root volume on a GPU AMI is routinely too small for torch + CUDA wheels,
# let alone weights, while the box has a multi-TB ephemeral NVMe mounted. Move
# the venv and caches there automatically -- this is the disk layout the p5en
# run-book prescribes, not an invention.
_resolve_vllm_env() {
    [[ "$GPU_PRESENT" == "yes" && "$SKIP_GPU" != "yes" ]] || return 0
    if [[ "$VLLM_ENV_EXPLICIT" == "yes" ]]; then
        VLLM_ENV_NOTE="VLLM_ENV was set explicitly; using $VLLM_ENV as given."
        return 0
    fi
    local home_gb
    home_gb="$(_free_gb "$VLLM_ENV")"
    [[ -n "$home_gb" ]] || return 0
    (( home_gb >= VLLM_MIN_FREE_GB )) && return 0

    BIG_VOLUME="$(_pick_big_volume)"
    if [[ -z "$BIG_VOLUME" ]]; then
        VLLM_ENV_NOTE="NO_ROOM:${home_gb}"
        return 0
    fi
    local big_gb
    big_gb="$(_free_gb "$BIG_VOLUME")"
    VLLM_ENV="$BIG_VOLUME/vllm-env"
    VLLM_ENV_NOTE="RELOCATED:${home_gb}:${big_gb}"
}


# Hardware-specific warnings the vLLM install would otherwise hit hours later.
_gpu_advisories() {
    case "$VLLM_ENV_NOTE" in
        RELOCATED:*)
            local home_gb big_gb
            home_gb="$(echo "$VLLM_ENV_NOTE" | cut -d: -f2)"
            big_gb="$(echo "$VLLM_ENV_NOTE" | cut -d: -f3)"
            echo
            info "DISK: the volume holding \$HOME has only ${home_gb}G free, but this box has a large"
            info "      volume at $BIG_VOLUME (${big_gb}G free). The vLLM venv and its caches will go"
            info "      there instead of the root disk:"
            info "        VLLM_ENV=$VLLM_ENV"
            info "        UV_CACHE_DIR=$BIG_VOLUME/uv-cache"
            info "        TMPDIR=$BIG_VOLUME/tmp"
            info "      This is the disk layout p5en-h200-cuda-fixes.md prescribes. Note the ephemeral"
            info "      NVMe is WIPED on instance stop/terminate -- weights re-download after a stop."
            info "      Override with: VLLM_ENV=/your/path $0 --install"
            ;;
        NO_ROOM:*)
            local free_gb
            free_gb="$(echo "$VLLM_ENV_NOTE" | cut -d: -f2)"
            echo
            warn "DISK: only ${free_gb}G free on the volume holding $VLLM_ENV, vLLM plus its torch/CUDA"
            warn "      wheels want ~${VLLM_MIN_FREE_GB}G before any weights land (a 30B model is another ~57G),"
            warn "      and no larger volume was found. Attach one, then re-run with:"
            warn "        VLLM_ENV=/your/large/volume/vllm-env $0 --install"
            ;;
    esac
    if [[ "$INSTANCE_TYPE" == p5* ]] || (( GPU_COUNT >= 8 )); then
        echo
        warn "HARDWARE: this looks like an 8-GPU NVSwitch node (${GPU_COUNT}x GPU, ${INSTANCE_TYPE:-unknown type})."
        warn "      The install scripts were verified on the 4x L40S reference node; this box needs"
        warn "      extra one-time driver/NVMe/CUDA-JIT fixes documented in:"
        warn "        $REPO_ROOT/.claude/skills/vllm-setup/p5en-h200-cuda-fixes.md"
        warn "      Read it before serving a model, or startup fails at KV-cache init."
    fi
}


_planned_keys() {
    local key
    for key in "${COMP_ORDER[@]}"; do
        if [[ "${COMP_STATE[$key]}" == "plan" ]]; then echo "$key"; fi
    done
}


_announce_plan() {
    local planned=() skipped=() present=() key
    while IFS= read -r key; do [[ -n "$key" ]] && planned+=("$key"); done < <(_planned_keys)
    for key in "${COMP_ORDER[@]}"; do
        case "${COMP_STATE[$key]}" in
            skip)    skipped+=("$key") ;;
            present) present+=("$key") ;;
        esac
    done

    header "Step 2 - Already present"
    if (( ${#present[@]} == 0 )); then
        echo "  (nothing -- this is a bare machine)"
    else
        for key in "${present[@]}"; do
            printf '  %-28s %s\n' "${COMP_LABEL[$key]}" "${COMP_DETAIL[$key]}"
        done
    fi

    echo
    rule
    if (( ${#planned[@]} == 0 )); then
        echo -e "${BOLD} INSTALL PLAN -- nothing to install, this machine is ready${RESET}"
    else
        echo -e "${BOLD} INSTALL PLAN -- ${#planned[@]} component(s) WILL BE INSTALLED${RESET}"
    fi
    rule
    local i=1
    for key in "${planned[@]}"; do
        printf ' %2d. %-28s %s\n' "$i" "${COMP_LABEL[$key]}" "${COMP_WHY[$key]}"
        if [[ -n "${COMP_DETAIL[$key]}" ]]; then printf '     %-28s (%s)\n' "" "${COMP_DETAIL[$key]}"; fi
        i=$((i + 1))
    done
    if (( ${#skipped[@]} > 0 )); then
        echo
        echo -e "${BOLD} SKIPPED (${COMP_DETAIL[${skipped[0]}]}):${RESET}"
        for key in "${skipped[@]}"; do
            printf '     %-28s %s\n' "${COMP_LABEL[$key]}" "${COMP_WHY[$key]}"
        done
    fi
    rule

    if [[ "$GPU_PRESENT" == "yes" && "$SKIP_GPU" != "yes" ]]; then
        _gpu_advisories
    fi

    if [[ "$WITH_OMP" != "yes" || "$WITH_KIRO" != "yes" ]]; then
        echo
        info "Optional harnesses NOT in this plan (both ship third-party 'curl | sh' installers,"
        info "so they are opt-in): re-run with --with-omp and/or --with-kiro to include them."
        info "On a shared node, review an installer before running it (docs/kiro-cli-setup.md):"
        info "  curl -fsSL https://cli.kiro.dev/install -o /tmp/kiro-install.sh && less /tmp/kiro-install.sh"
    fi

    # Warn about sudo up front rather than stalling on a password prompt mid-run.
    for key in "${planned[@]}"; do
        case "$key" in
            node|nvtop|aws|vllm|gh) NEEDS_SUDO="yes" ;;
        esac
    done
    if [[ "$NEEDS_SUDO" == "yes" ]] && ! _sudo_available; then
        echo
        warn "Part of this plan needs sudo (apt packages) and passwordless sudo is NOT available."
        warn "You will be prompted for a password, so do not run this unattended."
    fi
}


_install_git()      { _apt_install git; }
_install_aws()      { _apt_install awscli; }
_install_nvtop()    { _apt_install nvtop; }


_install_python() {
    # uv can supply an interpreter without touching the system Python.
    _have uv || _install_uv
    uv python install "3.12"
}


_install_uv() {
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Appended for the same reason as the export at the top of this file.
    export PATH="$PATH:$HOME/.local/bin"
    _have uv || die "uv installed but not on PATH; add \$HOME/.local/bin to PATH and re-run."
    uv tool update-shell >/dev/null 2>&1 || true
}


_install_node() {
    _is_apt_system || die "Node >= $MIN_NODE must be installed manually on this distribution."
    # Canonical vendor domain over HTTPS, as a literal: never build the URL of a
    # script that is about to run as root. No 'sudo -E' either -- the installer
    # does not need the caller's environment, so do not hand it to a root shell.
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash -
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs
}


# GitHub's own apt repository, verified by their signing key. Preferred over a
# pipe-to-shell installer: apt checks the signature on every future upgrade too.
_install_gh() {
    _is_apt_system || die "gh must be installed manually on this distribution: https://github.com/cli/cli#installation"
    sudo mkdir -p -m 755 /etc/apt/keyrings
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
    sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y gh
}


# Never guess an identity: a wrong name/email is baked into every commit the
# machine makes, and rewriting that later is a history rewrite.
_install_gitconfig() {
    if [[ -z "$GIT_USER_NAME" || -z "$GIT_USER_EMAIL" ]]; then
        fail "git identity not supplied. Re-run with: --git-name 'Your Name' --git-email you@example.com"
        return 1
    fi
    git config --global user.name "$GIT_USER_NAME"
    git config --global user.email "$GIT_USER_EMAIL"
    ok "git identity set: $(git config --global user.name) <$(git config --global user.email)>"
}


_install_claude()   { _npm_global "@anthropic-ai/claude-code"; }
_install_codex()    { _npm_global "@openai/codex"; }
_install_pi()       { _npm_global "@earendil-works/pi-coding-agent"; }
_install_omp()      { curl -fsSL https://omp.sh/install | sh; }
_install_kiro()     { curl -fsSL https://cli.kiro.dev/install | bash; }
_install_nvitop()   { uv tool install nvitop; }


_install_precommit() {
    _have pre-commit || uv tool install pre-commit
    ( cd "$REPO_ROOT" && "$(command -v pre-commit || echo "$HOME/.local/bin/pre-commit")" install )
}


_install_venv_benchmarks() { ( cd "$BENCHMARKS_DIR" && uv sync ); }
_install_venv_vllm()       { ( cd "$VLLM_DIR" && uv sync ); }


_install_vllm() {
    local installer="$VLLM_DIR/scripts/vllm-install.sh"
    [[ -x "$installer" ]] || die "vLLM installer not found or not executable: $installer"
    if [[ -n "$BIG_VOLUME" ]]; then
        mkdir -p "$BIG_VOLUME/uv-cache" "$BIG_VOLUME/tmp"
        export UV_CACHE_DIR="$BIG_VOLUME/uv-cache"
        export TMPDIR="$BIG_VOLUME/tmp"
        info "Caches redirected to $BIG_VOLUME (uv-cache, tmp) to keep the root disk clear."
    fi
    export VLLM_ENV
    info "Delegating to $installer (the vetted installer: build deps, venv, vLLM, gpustat)."
    info "VLLM_ENV=$VLLM_ENV"
    "$installer"
}


_install_one() {
    local key="$1"
    case "$key" in
        git)              _install_git ;;
        gitconfig)        _install_gitconfig ;;
        gh)               _install_gh ;;
        python)           _install_python ;;
        uv)               _install_uv ;;
        node)             _install_node ;;
        claude)           _install_claude ;;
        codex)            _install_codex ;;
        pi)               _install_pi ;;
        aws)              _install_aws ;;
        precommit)        _install_precommit ;;
        venv_benchmarks)  _install_venv_benchmarks ;;
        venv_vllm)        _install_venv_vllm ;;
        omp)              _install_omp ;;
        kiro)             _install_kiro ;;
        nvtop)            _install_nvtop ;;
        nvitop)           _install_nvitop ;;
        vllm)             _install_vllm ;;
        *)                die "no installer defined for '$key'" ;;
    esac
}


_confirm() {
    if [[ "$ASSUME_YES" == "yes" ]]; then return 0; fi
    echo
    read -r -p "Proceed with the install plan above? [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { info "Nothing installed. Re-run with --install --yes to skip this prompt."; exit 0; }
}


_run_installs() {
    local planned=() key
    while IFS= read -r key; do [[ -n "$key" ]] && planned+=("$key"); done < <(_planned_keys)
    if (( ${#planned[@]} == 0 )); then
        header "Step 3 - Install"
        ok "Nothing to install; every component is already present."
        return
    fi

    _confirm
    header "Step 3 - Install (${#planned[@]} component(s))"
    local total="${#planned[@]}" i=1
    for key in "${planned[@]}"; do
        echo
        echo -e "${BOLD}--- (${i}/${total}) installing ${COMP_LABEL[$key]} ---${RESET}"
        # Keep going after a failure so one broken component does not block the
        # rest; the summary reports exactly which ones failed.
        if _install_one "$key"; then
            _mark "$key" installed "installed by this run"
            done_ "(${i}/${total}) ${COMP_LABEL[$key]} -- INSTALLED"
            INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
        else
            _mark "$key" failed "install failed -- see the output above"
            warn "(${i}/${total}) ${COMP_LABEL[$key]} -- FAILED (continuing)"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
        i=$((i + 1))
    done
}


_state_plain() {
    case "$1" in
        present)   echo "ALREADY PRESENT" ;;
        installed) echo "INSTALLED" ;;
        plan)      echo "TO INSTALL" ;;
        skip)      echo "SKIPPED" ;;
        failed)    echo "FAILED" ;;
        *)         echo "$1" ;;
    esac
}


_state_color() {
    case "$1" in
        present|installed) echo "$GREEN" ;;
        plan)              echo "$YELLOW" ;;
        skip)              echo "$BLUE" ;;
        failed)            echo "$RED" ;;
        *)                 echo "$RESET" ;;
    esac
}


_summarize() {
    echo
    rule
    if [[ "$MODE" == "check" ]]; then
        echo -e "${BOLD} SUMMARY -- detection only, nothing was changed${RESET}"
    else
        echo -e "${BOLD} SUMMARY -- ${INSTALLED_COUNT} installed, ${FAILED_COUNT} failed${RESET}"
    fi
    rule
    local key
    for key in "${COMP_ORDER[@]}"; do
        printf ' %-30s %b%-16s%b %s\n' \
            "${COMP_LABEL[$key]}" "$(_state_color "${COMP_STATE[$key]}")" \
            "$(_state_plain "${COMP_STATE[$key]}")" "$RESET" "${COMP_DETAIL[$key]}"
    done
    rule
    echo "  GPU stack (vLLM, nvtop, nvitop): $( [[ "$GPU_PRESENT" == "yes" && "$SKIP_GPU" != "yes" ]] && echo 'INCLUDED -- GPU detected' || echo 'SKIPPED -- no GPU on this instance' )"
    echo "  Machine: $GPU_SUMMARY"

    header "Next steps"
    if [[ "$MODE" == "check" ]]; then
        echo "  1. Re-run with --install to apply the plan above."
    else
        if (( FAILED_COUNT > 0 )); then echo "  0. Re-run this script to retry the FAILED component(s); it is idempotent."; fi
        echo "  1. Open a new shell (or 'export PATH=\$HOME/.local/bin:\$PATH') so newly installed CLIs resolve."
    fi
    echo "  2. Wire 'claude' and 'codex' to Amazon Bedrock (codex must be >= 0.144 for the native"
    echo "     amazon-bedrock provider; 'codex --version' to confirm)"
    echo "     -- installing them does NOT configure them, and an unconfigured codex silently"
    echo "     calls api.openai.com and 401s mid-run:"
    echo "       $REPO_ROOT/benchmarks/docs/agent-cli-bedrock-setup.md"
    echo "  3. Authenticate the GitHub CLI (interactive, cannot be scripted): gh auth login"
    echo "  4. Create your runner config:  cp $BENCHMARKS_DIR/config/runner.example.yaml $BENCHMARKS_DIR/config/runner.yaml"
    echo "  5. Smoke-test the harness:     cd $BENCHMARKS_DIR && uv run python -m unittest discover -s tests"
    echo "     (stdlib unittest, which is what CI runs -- pytest is not a declared dependency)"
    if [[ "$GPU_PRESENT" == "yes" && "$SKIP_GPU" != "yes" ]]; then
        echo "  6. Serve a model: read $VLLM_DIR/models/<model>.md first, then self-hosted/vllm/scripts/vllm-serve.sh"
    fi
    if [[ "$WITH_KIRO" == "yes" ]]; then
        echo "  7. kiro-cli needs an interactive sign-in: 'kiro-cli login' (see docs/kiro-cli-setup.md)"
    fi
    echo
}


main() {
    _parse_args "$@"
    echo -e "${BOLD}setup-machine.sh -- dependency check for $REPO_ROOT${RESET}"
    echo "Mode: $MODE"
    _report_machine
    _resolve_vllm_env
    _detect_core
    _detect_gpu_stack
    _announce_plan
    if [[ "$MODE" == "install" ]]; then _run_installs; fi
    _summarize
    if (( FAILED_COUNT > 0 )); then exit 1; fi
    exit 0
}

main "$@"
