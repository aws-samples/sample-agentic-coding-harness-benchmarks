---
name: setup-machine
description: "Examine the instance this repo is checked out on, report exactly which dependencies are missing, and install them. Detects the machine (OS, vCPU/RAM, disk, EC2 instance type, GPU), then installs the core stack every path needs -- git, a global git identity, the GitHub CLI (gh), Python 3.10+, Node.js 22+, uv, the claude / codex / pi CLIs, the AWS CLI, pre-commit, and a synced venv for each of the two uv projects. When an NVIDIA GPU is present it ALSO installs vLLM, nvtop and nvitop; with no GPU it skips those three and says so. Use when the user asks what needs installing, wants the machine bootstrapped or set up, hits a missing-command error (uv/claude/codex/pi/gh not found), or is preparing a fresh EC2 box to run benchmarks. Wraps .claude/skills/setup-machine/setup-machine.sh."
license: Apache-2.0
metadata:
  author: Amit Arora
  version: "1.0"
---

# Setup Machine Skill

Use this skill to take a machine from a bare checkout to one that can actually run this repository, without the user guessing which of the four agent CLIs, two uv projects, and (optionally) the GPU serving stack they are missing.

The skill is a thin driver over [`setup-machine.sh`](setup-machine.sh), which holds all the detection and install logic. Run the script; do not re-derive its checks by hand, and do not install anything it does not plan.

**The GPU stack is conditional.** If `nvidia-smi` reports at least one GPU, the plan includes **vLLM, nvtop and nvitop**. If it does not, those three are skipped and reported as skipped -- never silently dropped, and never installed on a CPU-only box where they are useless.

## Workflow

1. **Examine** -- run the script in its default `--check` mode: machine facts, what is present, the full install plan. Nothing is changed.
2. **Announce** -- relay the plan to the user loudly and in full, including what is being skipped and why.
3. **Install** -- re-run with `--install`, echoing each component as it completes.
4. **Summarize** -- report the final table and the next steps the script prints.

---

## Step 1 -- Examine the instance

```bash
cd <repo root>
./.claude/skills/setup-machine/setup-machine.sh
```

Default mode is `--check`: it detects and prints, and installs nothing. It reports:

- **The machine** -- OS, kernel, vCPU, RAM, free disk, EC2 instance type (via IMDSv2, best effort), and the GPU line: count, model, and driver version, or `none detected`.
- **Already present** -- every component found, with its version.
- **The install plan** -- a numbered list of what is missing, each with the reason this repo needs it.
- **Skipped** -- the GPU stack when there is no GPU (or `--skip-gpu` was passed).
- **Advisories** -- where the vLLM venv will live (see below), and an 8-GPU NVSwitch warning pointing at [`p5en-h200-cuda-fixes.md`](../vllm-setup/p5en-h200-cuda-fixes.md) on a p5-class node.

### Where the vLLM venv lands

A GPU AMI's root disk is routinely ~29 GB, which is not enough for torch and the CUDA wheels, let alone model weights -- while the same box has a multi-TB ephemeral NVMe mounted. The script resolves this itself rather than failing partway through a download:

1. An explicitly set `VLLM_ENV` always wins, and is reported as such.
2. Otherwise, if `$HOME`'s volume has less than 40 GB free, it picks a large volume -- preferring the Deep Learning AMI's `/opt/dlami/nvme` by name, else the largest writable real filesystem -- and puts the venv, `UV_CACHE_DIR` and `TMPDIR` there. This is the disk layout [`p5en-h200-cuda-fixes.md`](../vllm-setup/p5en-h200-cuda-fixes.md) prescribes, not an invention of this skill.
3. If the root disk is tight and no large volume exists, it warns and tells the user to attach one; it does not install vLLM into a volume that cannot hold it.

`tmpfs` is deliberately excluded from that search: `/dev/shm` can advertise a terabyte, but it is RAM. **Tell the user when the venv gets relocated, and that the ephemeral NVMe is wiped on instance stop** -- weights re-download after a restart, which is the right trade for a serving box but a surprise if unannounced.

Two GPU cases the script distinguishes, and you should repeat the distinction to the user:

- **No NVIDIA hardware** -- CPU-only instance. The GPU stack is skipped; everything else still applies. This is a perfectly good machine for driving Bedrock-hosted models (Paths 1 and 2); it just cannot self-host (Path 3).
- **NVIDIA hardware present but `nvidia-smi` does not work** -- the driver is missing. Do *not* try to install vLLM: it will fail. Tell the user the driver needs installing first (the Deep Learning AMI ships one), then re-run.

## Step 2 -- Announce the plan, loudly

Before installing anything, tell the user in your own message:

- **how many components** will be installed and **what each one is for** (the script prints the reason per line -- do not drop it),
- **what is being skipped and why** ("no GPU on this instance, so vLLM, nvtop and nvitop are skipped"),
- **anything that needs sudo** -- the script warns up front when apt packages are planned and passwordless sudo is unavailable, because that turns an unattended run into one that stalls on a password prompt,
- **any advisory it printed** (disk, NVSwitch hardware).

Do not skip this because the script already printed it. The user asked to be told what is about to be installed, and the script's output may be scrolled off or buffered.

## Step 3 -- Install

```bash
./.claude/skills/setup-machine/setup-machine.sh --install --yes
```

Use `--yes` when you have already relayed the plan and the user has agreed; drop it to make the script prompt for itself. Other flags:

| Flag | Effect |
| --- | --- |
| `--git-name "Full Name"` | the `user.name` for the global git identity |
| `--git-email "you@example.com"` | the `user.email` for the global git identity |
| `--skip-gpu` | never install the GPU stack, even on a GPU box (useful when the box only drives Bedrock) |
| `--with-omp` | also install the `omp` agent (third-party `curl \| sh` installer) |
| `--with-kiro` | also install `kiro-cli` (third-party `curl \| bash` installer; needs an interactive `kiro-cli login` afterwards) |
| `VLLM_ENV=/path/vllm-env` | pin the vLLM venv location; overrides the automatic volume choice below |

`omp` and `kiro-cli` are **opt-in** because both ship third-party pipe-to-shell installers and kiro additionally requires an interactive sign-in. Offer them; do not add them unasked.

### Ask the user for their git identity

If `git config --global user.name` / `user.email` are unset, the script plans a `git identity` component -- and **refuses to install it without both `--git-name` and `--git-email`**. That is deliberate: a guessed identity is baked into every commit the machine makes and rewriting history to fix it is far worse than one question. So **ask the user for their name and email** before the install step, and pass them through:

```bash
./.claude/skills/setup-machine/setup-machine.sh --install --yes \
    --git-name "Full Name" --git-email "you@example.com"
```

Never infer these from the shell user, the hostname, an existing commit, or a `~/.gitconfig` in another checkout. If the user declines to give them, run the install anyway -- every other component proceeds and `git identity` is simply reported as failed, which is honest.

The script echoes `[done] (n/total) <component> -- INSTALLED` as each one lands, and **keeps going after a failure** so one broken component does not block the rest. It is idempotent: re-running after a partial failure only does what remains.

What it installs, and why the repo needs it:

| Component | Why |
| --- | --- |
| `git` | clones the target repo each benchmark task runs against |
| git identity | `user.name` / `user.email`, without which any commit the machine makes fails outright; **must be supplied by the user**, never guessed |
| `gh` (GitHub CLI) | AGENTS.md's workflow is branch + PR, which `gh pr create` drives; also how issues and CI runs are read. Installed from GitHub's own apt repo and keyring so future `apt upgrade`s stay signature-verified |
| Python 3.10+ | the harness, judge and plot scripts are 3.10+ (installed via `uv python` -- the system Python is left alone) |
| `uv` | the repo's package manager; AGENTS.md forbids using `pip` directly |
| Node.js 22+ | `claude`, `codex` and `pi` are npm packages, and pi needs >= 22 |
| `claude` | the default harness -- the runner shells out to `claude -p` |
| `codex` | the judge -- `codex_judge.py` scores artifacts with `codex exec` |
| `pi` | the second harness (`--agent pi`), which produced most published results |
| AWS CLI | Bedrock credentials and the pre-flight identity check |
| pre-commit + hook | CI fails on unformatted Python; the hook makes formatting mechanical |
| `benchmarks/.venv` | `uv sync` in the harness project |
| `self-hosted/vllm/.venv` | `uv sync` in the client project (openai, duckdb, ...) |
| **vLLM** *(GPU only)* | serves the open-weight model for the self-hosted path (Path 3) |
| **nvtop** *(GPU only)* | live per-GPU utilization TUI while a benchmark runs |
| **nvitop** *(GPU only)* | per-process GPU view: which PID holds which slice of VRAM |

vLLM is **not** installed inline -- the script delegates to [`self-hosted/vllm/scripts/vllm-install.sh`](../../../self-hosted/vllm/scripts/vllm-install.sh), the vetted installer that also handles the two Deep Learning AMI fixes (`python3.12-dev` + `build-essential` for the Triton JIT). Never reimplement that here.

## Step 4 -- Summarize

Relay the script's final summary: one row per component with `ALREADY PRESENT` / `INSTALLED` / `SKIPPED` / `FAILED`, the GPU verdict, and then the next steps. Call out four things explicitly, because they are the ones that bite later:

1. **A new shell is needed** for freshly installed CLIs to resolve (or `export PATH=$HOME/.local/bin:$PATH`).
2. **Installing `claude` and `codex` does not wire them to Amazon Bedrock.** An unconfigured `codex` ignores perfectly healthy AWS credentials and 401s against `api.openai.com` partway into a run. Point at [benchmarks/docs/agent-cli-bedrock-setup.md](../../../benchmarks/docs/agent-cli-bedrock-setup.md) and offer to walk it.
3. **Installing `gh` does not authenticate it.** `gh auth login` is interactive (browser or token) and cannot be scripted from here, so `gh pr create` fails until the user runs it.
4. **The runner config is not created by this skill.** `cp benchmarks/config/runner.example.yaml benchmarks/config/runner.yaml` and edit it.

If anything reported `FAILED`, say which component and what the error was, and offer to re-run (idempotent) or fix it by hand. Do not report the machine as ready while a component is failed.

Verify the result rather than assuming it:

```bash
cd benchmarks && uv run python -m unittest discover -s tests
cd self-hosted/vllm && uv run python -m unittest discover -s tests
```

A green suite means the project is genuinely usable, not just installed. Use `unittest`, not `pytest`: the suites are stdlib `unittest` and that is what [.github/workflows/test.yml](../../../.github/workflows/test.yml) runs. `pytest` is not in either project's dev dependencies, so `uv run pytest` fails on a correctly set-up machine -- do not report that as a setup failure.

## What this skill does NOT do

- It does not configure any CLI's credentials or model routing (see [agent-cli-bedrock-setup.md](../../../benchmarks/docs/agent-cli-bedrock-setup.md) and [kiro-cli-setup.md](../../../docs/kiro-cli-setup.md)).
- It does not serve a model -- that is the [`vllm-setup`](../vllm-setup/SKILL.md) skill, which this skill's GPU step only prepares the ground for.
- It does not run a benchmark -- that is [`benchmark`](../benchmark/SKILL.md).
- It does not install NVIDIA drivers. If the hardware is there but `nvidia-smi` is not, say so and stop.
