# Agent guidelines

Instructions for terminal coding agents (opencode, etc.) working in this repo.

## Do not explore proactively

Do **not** scan, index, or "read the whole repo" to build context before acting. This repo is large (~270 MB, thousands of files) and most of it is not source you should touch. Read a file only when the current task specifically requires it, and prefer targeted reads (a named path) over broad `glob`/`grep`/`list` sweeps across the tree.

## Never read or search these paths

They are large, generated, or irrelevant to code changes — walking them wastes context and time:

- `tmp/` — scratch output (~119 MB)
- `benchmarks/swe-benchmark-data/*/repo/` and `**/swe-benchmark-data/*/repo/` — cloned target repos, not this project's source
- `benchmarks/swe-benchmark-data/**` — large generated benchmark artifacts; read a specific file only if the task names it
- `.venv/`, `**/.venv/` — Python virtualenvs
- `.git/`, `.scratchpad/`, `results/`, `docs-local/`
- `self-hosted/vllm/logs/` — server logs (can be huge)
- `*.log`, `*.pem`, `*.key`, `.hf_token` — logs and secrets; never read secrets

## Where the code actually is

When a task is unscoped, the source worth reading lives under:

- `self-hosted/vllm/` — the vLLM self-hosting path (scripts, model guides, clients)
- `benchmarks/` — harness scripts, datasets, runner config, and docs
- `vend/swe-router/` — the vendored model-selection skill
- top-level `README.md` and each subdirectory's `README.md`

## Conventions

- **Markdown:** do not hard-wrap prose. Write each paragraph as a single line and let the renderer soft-wrap. Tables, code fences, and lists are unaffected.
- Match the style of surrounding code; keep changes minimal and scoped to the task.

## Git rules

- **Never commit directly to `main`.** Always create a feature branch and open a PR.
- **Never merge PRs directly to `main`.** Use a PR review workflow with an approved merge.
