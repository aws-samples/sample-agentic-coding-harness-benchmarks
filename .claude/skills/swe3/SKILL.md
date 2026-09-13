---
name: swe3
description: "Single-agent, end-to-end SWE benchmark: takes a coding problem from idea to a working code change, producing six artifacts (GitHub issue spec, low-level design, expert review, testing plan, patch.diff, implementation.md under benchmarks/swe-benchmark-data/{model-name}/{harness}/{repo-name}/{problem-name}/). Does ALL work in the main agent loop with NO subagent fan-out, so it measures single-agent harness cost for an apples-to-apples comparison against harnesses that cannot spawn subagents (e.g. pi). Design, implement, and capture the change as a reviewable patch - the orchestration is single-agent (inline, not parallel Task subagents)."
license: Apache-2.0
metadata:
  author: Amit Arora
  version: "1.0"
---

# Software Engineering with Implementation (SWE3) Skill

Use this skill when the user wants to evaluate how a particular LLM performs on a software-engineering problem **end to end, including writing the code**. Every run is treated as a benchmark: the problem is the input, the model is the contestant, and the artifacts (`github-issue.md`, `lld.md`, `review.md`, `testing.md`, `patch.diff`, `implementation.md`) are the output that can be compared across models.

**This skill goes all the way to a code change.** It produces four design artifacts AND then implements the design by editing the cloned repository in place, capturing every edit as a `git diff` (`patch.diff`) plus an `implementation.md` summary. It still does NOT run the target repo's tests, commit, push, or open a PR - the deliverable is the six artifact files described below, with the actual code changes captured as a reviewable, comparable patch.

## Non-Interactive Mode (Headless)

When ALL required parameters are provided upfront in a single message, skip all confirmation prompts and run end-to-end without asking questions. This enables fully automated benchmark runs via `claude -p`.

**Required parameters for non-interactive mode:**

```
/swe3 repo: <local-path-to-repo> problem: <problem-slug> model: <model-name> tag: <git-tag> [artifacts_dir: <absolute-artifact-dir>] answers: "<answers-block>"
```

Example:
```
/swe3 repo: benchmarks/swe-benchmark-data/mcp-gateway-registry/repo problem: ssrf-hardening-outbound-url-validation model: kimi-k2.7-code tag: 1.24.4 answers: "1. Security audit finding — the registry fetches user-supplied URLs with no SSRF guard. 2. Operators and downstream teams. 3. Python/FastAPI, ECS, no deadline, backwards-compatible. 4. Medium."
```

The `repo:` path may be a checkout the caller already cloned (including a temporary directory such as one under `/tmp`) or a path that does not exist yet. If it does not exist, clone it at `tag:` per Step 1.4 without asking. Either way, `{repo-name}` is the basename of the `repo:` path. **If the caller passed `repo:` as an existing checkout, it is ALREADY cloned — do NOT run `git clone`, and do NOT `cd` into any other repository; treat that path as the sole code source.**

**When non-interactive mode is triggered:**
- Do NOT ask for model confirmation — use the `model:` parameter directly
- Do NOT ask for the GitHub URL — derive `{repo-name}` from the `repo:` path basename; if the path is missing, clone at `tag:` (Step 1.4) without prompting
- Do NOT ask for tag confirmation — use the `tag:` parameter
- Do NOT ask for task confirmation — use the `problem:` parameter as-is
- Do NOT ask clarifying questions (1.5) — parse answers from the `answers:` parameter
- Do NOT ask before creating folders or overwriting — proceed directly
- Do NOT ask before editing the cloned repo in Step 8.5 — implement directly (the clone is a disposable temp checkout; edits are captured as patch.diff, never committed)
- Do NOT present summary or seek guidance at the end — just write all 6 artifacts (including patch.diff and implementation.md) and exit
- **If `artifacts_dir:` is provided, use it VERBATIM as `$ART_DIR` (see Step 3). Do NOT recompute the artifact directory from `git rev-parse --show-toplevel` — that returns the wrong root if you have `cd`-ed into another repo, sending artifacts into a phantom tree.**

**Detection rule:** If the user message contains ALL of `repo:`, `problem:`, `model:`, AND `answers:` — enter non-interactive mode. If any are missing, fall back to the normal interactive flow below.

---

## Workflow

1. **Gather Requirements** - Detect the active model and confirm it; ask for the GitHub URL; ask for tag-vs-main; confirm the task; locate or clone the target repo with user approval
2. **Quick Codebase Review** - Explore the codebase to understand structure
3. **Create Benchmark Folder** - If the harness passed an `artifacts_dir:`, that path IS the folder: use it VERBATIM and derive nothing. Only when no `artifacts_dir:` was given, build `benchmarks/swe-benchmark-data/{model-name}/{harness-name}/{repo-name}/{problem-name}/` yourself, where `{harness-name}` is the coding agent producing the run (`claude-code` when you run this skill, `pi` for a pi run).
4. **Write GitHub Issue** - Create `github-issue.md` with the issue specification
5. **Deep Codebase Analysis** - Thoroughly explore relevant code
6. **Write Low-Level Design** - Create `lld.md` with technical details
7. **Expert Review** - Create `review.md` with multi-persona feedback
7.5. **Address Review Findings in the LLD** - if any reviewer raised a critical / must-fix / should-fix item, loop back and revise `lld.md` to resolve it BEFORE implementing; record the resolution in `review.md`
8. **Write Testing Plan** - Create `testing.md` with functional, backwards-compat, UX, deployment, and E2E tests
8.5. **Implement the Change** - Edit the cloned repo in place to realize the (revised) LLD, then capture `patch.diff` and `implementation.md`
9. **Present Summary & Seek Guidance** - Present the six artifacts and ask for direction

---

## Single-agent execution: do NOT use subagents

**This is the single-agent variant. Do ALL work yourself, in the main agent loop. Do NOT use the `Task` tool or spawn subagents of any kind, for any step.** A multi-agent run would fan out to parallel `Task` subagents (codebase analysis in Step 5, the five expert reviews in Step 7, edit drafting in Step 8); here you instead do that same work **inline** with your own Read/Grep/Glob/Bash/Edit calls, one facet after another.

Why this variant exists: subagent fan-out re-pays the full agent context (system prompt, tool catalog, skill text, task context) for every subagent and re-reads it each turn, which multiplies token usage several-fold. This variant removes that so the run measures **single-agent harness cost** -- an apples-to-apples comparison against harnesses that have no subagent mechanism (e.g. pi). The deliverables and the depth of work are unchanged; only the orchestration differs.

Rules of thumb:

- **Never call the `Task` tool.** No `subagent_type=Explore`, no parallel fan-out, no delegated edits. If you catch yourself about to launch a subagent, do that work directly instead.
- **Investigate sequentially in the main loop.** Cover the same facets a fan-out would (data structures, config, call sites, tests, deployment) with your own Read/Grep/Glob calls, one after another. The ~50-file-read cap and "finish the artifacts over exhaustive research" discipline below still apply -- they matter more here since there is no parallelism to hide latency.
- **You author everything.** You read, you decide, you write every artifact, you apply and reconcile every edit. There is no gather-and-synthesize hand-off because there are no subagents.
- **Expect this to be slower in wall-clock** (no parallelism) but to consume far fewer tokens. That token reduction is the point of the measurement; do not reintroduce subagents to speed it up.

### Pace and budget (finish all six artifacts)

The failure mode to avoid is spending the whole turn/context budget researching and never writing the artifacts. Bias toward completion:

- **Start with a short plan and a todo list** (max ~10 steps), and check items off as you go. This keeps the run on track across long sessions and compaction boundaries.
- **Time-box exploration.** After you have read enough to understand the scope -- aim for **under ~50 file reads** -- start writing artifacts immediately. Depth beyond that rarely raises the score and risks running out of turns before `patch.diff` exists.
- **Prioritize completing all six artifacts over exhaustive research.** A complete, serviceable set beats a perfect `lld.md` with no implementation.
- **Persist context if you need to.** If you must track findings across a compaction boundary, write scratch notes to a temporary `.md` in the artifact directory rather than holding everything in context.

---

## Step 1: Gather Requirements

**NEVER guess the repo URL, the tag, or the task description.** All of them must come from the user. Do not infer them from session context, the current working directory, recent files, or memory.

The skill MUST do four things in this exact order at the very start of every run, even when some values were passed in as parameters. Confirm what was passed; ask for what is missing. Do not move past Step 1 until all four have an explicit user-confirmed answer.

### 1.0 Announce and confirm the model first

Before anything else, the skill must figure out which model is currently driving this session and tell the user, then ask the user to confirm or override.

How to figure it out:

- Look at the system context for the active model id (e.g. a line like "You are powered by the model named Opus 4.7 (1M context). The exact model ID is us.anthropic.claude-opus-4-7[1m].").
- Pick the canonical model name from the ID. Strip vendor/region prefixes (`us.anthropic.`), drop bracketed context-window suffixes (`[1m]`), and use kebab-case. Examples: `us.anthropic.claude-opus-4-7[1m]` -> `claude-opus-4-7`; `claude-sonnet-4-6` -> `claude-sonnet-4-6`; `claude-haiku-4-5-20251001` -> `claude-haiku-4-5`.
- If you cannot determine the model from system context, do not invent one - tell the user you could not detect it and ask.

Then announce and confirm:

> I am using **`{detected-model-name}`** for this run. This will be used as the `{model-name}` folder under the benchmark directory.
>
> Is that correct, or would you like to use a different name? (Reply with the name in kebab-case, e.g. `claude-opus-4-7`, `claude-sonnet-4-6`, `gpt-5`.)

Wait for confirmation. Only after the user confirms (or supplies an override) lock in `{model-name}` and continue. Do **not** ask for the model again later in Step 1.5; remove that question from the remaining clarifications since it has already been settled here.

### 1.1 Question 1 - GitHub repo URL

**Always ask first.** This is the canonical identifier of the target repository; everything else (folder names, clone commands, README rows) is derived from it.

> **Q1.** What is the GitHub URL of the repository you want to benchmark?
> Example: `https://github.com/agentic-community/mcp-gateway-registry`

If the user provided `--repo <url>` (or any equivalent param), echo the URL back and ask for confirmation rather than skipping the question.

From the URL, derive:
- `{repo-name}` = the basename of the URL with `.git` stripped (kebab-case as-is, e.g. `mcp-gateway-registry`).
- `{owner}` = the path segment before the repo name (used only in messages, never inferred for anything else).

State the derived `{repo-name}` back to the user before continuing.

### 1.2 Question 2 - Git tag or main

**Ask second, only after Q1 is answered.**

> **Q2.** Which version should I check out?
> 1. A specific git tag (e.g. `1.24.4`) - recommended for reproducible benchmarks.
> 2. `main` - latest commit on the default branch.

Record the answer as `{ref}`. If the user picked a tag, `{ref}` is the tag name. If the user picked main, `{ref}` is `main`.

### 1.3 Question 3 - Confirm the task

**Ask third, only after Q1 and Q2 are answered.**

> **Q3.** Is this your task?
> "{the task description the user originally provided, or a placeholder if none was provided}"

If the user passed `--task <description>`, repeat it verbatim and ask for yes/no confirmation. If they did not provide a task at all, ask them to describe it now and then confirm:

> What task should the model attempt against this repo? Example: "remove FAISS from the codebase and documentation".

Once the user confirms the task wording, derive a kebab-case `{problem-name}` from it (e.g. "remove FAISS from the codebase" -> `remove-faiss`) and **confirm the derived name with the user one more time** before creating any folders.

### 1.4 Locate or Clone the Target Repository

Now that `{repo-name}`, `{ref}`, and `{problem-name}` are settled, resolve the source checkout. Artifacts always land under the fixed benchmark output path (see Step 3); the *source clone* location, however, is flexible - the target repo is read-only input and may live either inside the benchmark tree or in a temporary directory. All benchmark-tree paths are expressed relative to the repository root via `git rev-parse --show-toplevel` - never hardcode absolute paths under it. Let:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
BENCH_DIR="$REPO_ROOT/benchmarks/swe-benchmark-data"
```

1. **Check for an existing local checkout first.** If the user already pointed you at a checkout (e.g. a `/tmp` clone the harness made, recorded as `{repo-path}`), confirm the ref:

   ```bash
   git -C "{repo-path}" describe --tags --exact-match  # for tags
   # or
   git -C "{repo-path}" rev-parse --abbrev-ref HEAD     # for main
   ```

   If the local checkout is at the wrong ref, tell the user and ask whether to re-clone at `{ref}` or keep the existing checkout.

2. **If no checkout exists, clone the repo yourself at `{ref}`.** The clone is a **writable workspace** (Step 8.5 edits it and diffs against the pinned ref), so clone it wherever is convenient - you do NOT need to place it inside the benchmark tree, and a temporary directory such as `/tmp` is ideal. A shallow clone at the exact ref is sufficient and still lets `git diff` work:

   ```bash
   CLONE_DIR="$(mktemp -d /tmp/swe3-{repo-name}-XXXXXX)"
   git clone --branch {ref} --depth 1 {url} "$CLONE_DIR"
   ```

   In **interactive** mode, announce the clone command and wait for approval before running it. In **non-interactive mode** (see "Non-Interactive Mode (Headless)" above), clone directly without asking. Record the resulting checkout path as `{repo-path}` and use it as the sole code source for the rest of the run.

   A `/tmp` clone is disposable and never committed; you do not need to clean it up, but you may. **Read it cleanly through Step 8 (do not edit it during design), then edit it freely in Step 8.5** - the edits are captured as `patch.diff` and never committed anywhere.

   > **Pin the baseline for the diff.** Immediately after cloning (and before any edit), record the exact starting commit so Step 8.5 can diff against it even in a shallow clone:
   >
   > ```bash
   > BASE_SHA="$(git -C "$CLONE_DIR" rev-parse HEAD)"
   > ```
   >
   > If the caller handed you an existing checkout, it may already contain unrelated local edits; note this in `implementation.md` and diff against `BASE_SHA` regardless so the patch reflects only your work relative to the pinned ref.

### 1.5 Remaining Clarifying Questions

Once the model, URL, ref, task, and the local checkout are settled, gather the rest. Do **not** re-ask which model is being benchmarked - that was already settled in Step 1.0.

1. What problem does this solve?
2. Who are the users/consumers?
3. Are there any constraints (language, framework, environment, deadlines)?
4. What is the expected scope (small/medium/large)?

## Step 2: Quick Codebase Review

Before creating any design documents, perform a quick exploration of the codebase to understand:

1. **Project Structure** - top-level layout, source roots, config files
2. **Related Components** - existing features similar to the one being designed
3. **Entry Points** - main scripts, CLIs, or app entrypoints

This quick review takes 5-10 minutes and helps you ask better clarifying questions and avoid proposing designs that conflict with existing architecture.

## Step 3: Create Benchmark Folder

All artifacts live under a top-level `benchmarks/` directory. Within it, every run gets its own `{model-name}/{harness-name}/{scope}/{problem-name}/` subfolder. Grouping by model first keeps each model's full set of results together; the `{harness-name}` level (e.g. `claude-code`, `pi`) then keeps runs of the same model by different coding agents from overwriting each other, while still letting models be compared on the same `{scope}/{problem-name}` across sibling folders.

> **`{scope}` is NOT always the repository name, which is why a derived path is unsafe.** It defaults to the repo name, but a dataset that sets `output_scope` overrides it, so that two datasets over the *same* repository do not share a results folder (for example `mcp-gateway-registry-v2` rather than `mcp-gateway-registry`). You cannot infer that value from the repo URL -- only the caller knows it. If the harness passed `artifacts_dir:` and you build a path from `{repo-name}` instead, the six artifacts land in a sibling folder the harness never looks at: it scores the task **0/6 despite every file being written successfully**, throws the work away, and re-runs the whole task. Use `artifacts_dir:` verbatim whenever it is given.

> **CRITICAL - write to the ABSOLUTE artifact path, never the bare relative string.** The `benchmarks/swe-benchmark-data/...` paths shown throughout this skill are written relative to the repository root for readability. Your working directory is often already inside `benchmarks/` (the harness runs there), so if you pass a bare relative path like `benchmarks/swe-benchmark-data/{model}/...` to Write/Edit it resolves against the cwd and doubles to `benchmarks/benchmarks/swe-benchmark-data/...` -- the files land in the wrong place, the run scores 0/4 artifacts despite "File created successfully", and the work is lost.
>
> **If the caller passed `artifacts_dir:`, that IS `$ART_DIR` -- use it verbatim and skip the `git rev-parse` derivation below entirely:**
>
> ```bash
> ART_DIR="<the absolute path from the artifacts_dir: parameter>"
> mkdir -p "$ART_DIR"
> ```
>
> Otherwise (no `artifacts_dir:` given), resolve the repo root and build the directory yourself -- but ONLY if your working directory is still inside the harness's repository. If you have `cd`-ed into the cloned target repo (or any other git repo), `git rev-parse --show-toplevel` returns THAT repo's root, not the harness root, and every artifact lands in a phantom tree:
>
> ```bash
> REPO_ROOT="$(git rev-parse --show-toplevel)"
> ART_DIR="$REPO_ROOT/benchmarks/swe-benchmark-data/{model-name}/{harness-name}/{repo-name}/{problem-name}"
> mkdir -p "$ART_DIR"
> ```
>
> Every `Write`/`Edit` of an **artifact** in Steps 4-8.5 must use an **absolute** `file_path` of the form `$ART_DIR/github-issue.md` (i.e. the fully-expanded `/.../benchmarks/swe-benchmark-data/{model}/{repo}/{problem}/github-issue.md`), not a path beginning with `benchmarks/`. This applies to `patch.diff` and `implementation.md` too. After writing each file, confirm it exists at the absolute path (e.g. `test -f "$ART_DIR/github-issue.md"`). (Note: the in-place code edits of Step 8.5 are the exception - those are written into `{repo-path}`, the clone, not under `$ART_DIR`; only the captured `patch.diff` and `implementation.md` land in `$ART_DIR`.)

The target repository's source code is **not** stored here. It is cloned locally at a specific tag by each contributor following the instructions in `benchmarks/swe-benchmark-data/README.md`, into a `repo/` subdirectory, or into a temporary clone (e.g. under `/tmp`, as the harness does). The `repo/` checkout is gitignored so it is never committed.

### Folder Structure

```
benchmarks/
└── swe-benchmark-data/
    ├── README.md                       # Lists target repos, tags, and tasks to benchmark
    └── {model-name}/
        └── {harness-name}/              # coding agent: claude-code, pi, ...
            └── {repo-name}/
                ├── {problem-name}/
                │   ├── github-issue.md      # GitHub issue specification
                │   ├── lld.md               # Low-level design document
                │   ├── review.md            # Expert review document
                │   ├── testing.md           # Testing plan (functional, backwards-compat, UX, deployment, E2E)
                │   ├── patch.diff           # git diff of the implemented change (Step 8.5)
                │   └── implementation.md    # Summary of the code change: files touched, how to apply, deviations
                └── {next-problem-name}/
                    └── ...
```

Conventions:

- Use kebab-case for `{model-name}` and prefer the canonical model id (e.g. `claude-opus-4-8`, `claude-sonnet-5`, `qwen3.6-35b`, `gpt-5`).
- Use kebab-case for `{repo-name}` and match the upstream repository name (e.g. `mcp-gateway-registry`). The list of supported `{repo-name}` values, their upstream URLs, and the tag to clone are all defined in `benchmarks/swe-benchmark-data/README.md`.
- Source code for the target lives at the checkout resolved in Step 1.4 (`{repo-path}`): a temporary clone such as one under `/tmp`. If no checkout exists yet, clone it at `{ref}` per Step 1.4 rather than stopping.
- Use kebab-case for `{problem-name}` (e.g. `remove-faiss`, `remove-efs-from-terraform-aws-ecs`). Prefer the exact name listed in the benchmark README's task table.
- The same `{repo-name}/{problem-name}` under different `{model-name}/` folders lets models be compared on one problem - do not delete sibling model folders.

### Pre-existing Artifacts: Confirm Before Overwriting

Before writing any artifact, check whether the target `{model-name}/` folder already contains any of `github-issue.md`, `lld.md`, `review.md`, `testing.md`, `patch.diff`, or `implementation.md`. If **one or more** of them exist, **stop and ask the user** what to do. Never silently overwrite.

Concretely:

1. List which of the six files already exist (with size and last-modified time, so the user can see they're real prior work).
2. Present the choices clearly and wait for the user's answer:

   > The following artifacts already exist at `benchmarks/swe-benchmark-data/{model}/{repo}/{problem}/`:
   > - `lld.md` (12.6 KB, modified 2026-06-04)
   > - `review.md` (4.1 KB, modified 2026-06-04)
   >
   > How would you like to proceed?
   > 1. **Delete all six files first**, then run a clean `/swe3` pass (recommended for a fresh benchmark).
   > 2. **Overwrite in place** as each artifact is regenerated (existing files get replaced one by one).
   > 3. **Append a suffix** to the model folder (e.g. `claude-opus-4-7-run2/`) and write the new run there, leaving the prior run intact.
   > 4. **Abort** - keep everything as-is and exit the skill.

3. Only proceed once the user picks an option:
   - **Option 1 (delete first):** remove the six files (and only those six; do not touch sibling folders or the cloned `repo/`). Print the `rm` commands you ran.
   - **Option 2 (overwrite in place):** continue, overwriting each artifact when its step writes the file.
   - **Option 3 (append suffix):** ask the user to confirm the suffix, then create `{model-name}-{suffix}/` and treat that as the new target folder for the rest of the run.
   - **Option 4 (abort):** stop the skill cleanly, do not modify anything, do not create empty folders.

Even if all six files are present and the user picks option 2, do not "merge" with prior content - each new step writes the new artifact end-to-end. The prior file is replaced, not edited in place.

If `benchmarks/swe-benchmark-data/{model-name}/{repo-name}/{problem-name}/` exists but is **empty**, no confirmation is needed; proceed normally.

Example for the same problem solved by two models (each under its own model folder):

```
benchmarks/swe-benchmark-data/
├── claude-opus-4-8/
│   └── mcp-gateway-registry/
│       └── remove-faiss/
│           ├── github-issue.md
│           ├── lld.md
│           ├── review.md
│           ├── testing.md
│           ├── patch.diff
│           └── implementation.md
└── claude-sonnet-5/
    └── mcp-gateway-registry/
        └── remove-faiss/
            ├── github-issue.md
            ├── lld.md
            ├── review.md
            ├── testing.md
            ├── patch.diff
            └── implementation.md
```

## Step 4: Write GitHub Issue (github-issue.md)

Create a comprehensive GitHub issue specification. This is the artifact that would be filed against the upstream repo to track the task.

### Template

```markdown
# GitHub Issue: {Feature / Task Title}

## Title
{concise title for the issue}

## Labels
- {appropriate labels: enhancement, bug, refactor, infra, docs, etc.}

## Description

### Problem Statement
{What problem does this solve? Why is it needed?}

### Proposed Solution
{High-level description of the solution}

### User Stories
- As a {user type}, I want to {action} so that {benefit}

### Acceptance Criteria
- [ ] {Criterion 1}
- [ ] {Criterion 2}

### Out of Scope
- {What is explicitly NOT included}

### Dependencies
- {Any dependent issues or external dependencies}

### Related Issues
- #{issue numbers if any}
```

## Step 5: Deep Codebase Analysis

**CRITICAL:** Before writing the LLD, you MUST thoroughly understand all relevant code in the cloned `repo/`. A design that ignores existing patterns will fail when an implementer picks it up.

### What to Analyze

1. **Existing Models and Data Structures** - Pydantic models, dataclasses, schemas
2. **Service / Business Logic Patterns** - how logic is organized, error handling, logging, caching
3. **Route / CLI / Entrypoint Patterns** - request/response shapes, argparse layouts
4. **Storage / IO Layer** - persistence, file IO, network calls
5. **Configuration and Constants** - env vars, settings classes, feature flags
6. **Existing Tests** - testing patterns, fixtures, mocking conventions

### How to Analyze

**Cover all six areas yourself, in the main loop, one facet at a time. Do NOT spawn subagents.** Work through the facets sequentially with your own Read/Grep/Glob calls - for example:

- map the models/data structures and configuration/constants (areas 1 and 5),
- map the service/business-logic and storage/IO patterns (areas 2 and 4),
- map the route/CLI/entrypoint patterns and how the feature under design is reached today (area 3),
- map the existing tests, fixtures, and mocking conventions (area 6).

Read actual code (not just file names); record the key files, patterns, and integration points, and note TODOs and known issues as you go. You own the LLD and do this investigation directly - there is no fan-out and no synthesis hand-off. Stay within the file-read budget below; the goal is grounded, non-redundant investigation, not an exhaustive sweep.

### Document Your Findings

Capture in your LLD:
- Key files reviewed
- Patterns identified
- Integration points for the new change
- Constraints or limitations discovered

## Step 6: Write Low-Level Design (lld.md)

Create a detailed technical design document. This is the most critical document - it should contain enough detail for an entry-level developer to implement the change later.

```markdown
# Low-Level Design: {Feature Name}

*Created: {date}*
*Author: Claude*
*Status: Draft*

## Table of Contents
1. [Overview](#overview)
2. [Codebase Analysis](#codebase-analysis)
3. [Architecture](#architecture)
4. [Data Models](#data-models)
5. [API / CLI Design](#api--cli-design)
6. [Configuration Parameters](#configuration-parameters)
7. [New Dependencies](#new-dependencies)
8. [Implementation Details](#implementation-details)
9. [Observability](#observability)
10. [Scaling Considerations](#scaling-considerations)
11. [File Changes](#file-changes)
12. [Testing Strategy](#testing-strategy)
13. [Alternatives Considered](#alternatives-considered)
14. [Rollout Plan](#rollout-plan)

## Overview
### Problem Statement
{Detailed problem description}

### Goals
- {Goal 1}

### Non-Goals
- {What this design explicitly does NOT address}

## Codebase Analysis

### Key Files Reviewed

| File/Directory | Purpose | Relevance to This Change |
|----------------|---------|--------------------------|
| `{path}` | {Description} | {How it relates} |

### Existing Patterns Identified
1. **Pattern Name**: {Description}
   - Files: `{file1}`, `{file2}`
   - How a future implementer should follow this: {How}

### Integration Points

| Component | Integration Type | Details |
|-----------|------------------|---------|
| {Existing component} | {Extends/Uses/Depends on} | {Specific details} |

### Constraints and Limitations Discovered
- {Constraint}: {How it affects the design}

## Architecture

### System Context Diagram
{ASCII diagram showing how this fits into the overall system}

### Sequence Diagram
{Show the flow of requests/data}

### Component Diagram
{Show internal components and their relationships}

## Data Models

### New Models
```python
class NewModel(BaseModel):
    """Description."""

    field_name: str = Field(
        ...,
        description="What this field represents",
        min_length=1,
        max_length=100
    )
```

### Model Changes
{Changes to existing models}

## API / CLI Design

### New Endpoints / Commands
**Description:** {What it does}

**Request / Invocation:**
```bash
uv run python -m {module} --param value
```

**Expected Response / Output:**
```json
{ "id": "123", "status": "success" }
```

**Error Cases:**
- 400 / nonzero exit: {when}

## Configuration Parameters

### New Environment Variables

| Variable Name | Type | Default | Required | Description |
|---------------|------|---------|----------|-------------|
| `FEATURE_ENABLED` | bool | `true` | No | Enable/disable the feature |

### Settings / Config Class Updates
```python
feature_enabled: bool = Field(
    default=True,
    description="Enable/disable feature X"
)
```

### Deployment Surface Checklist
List every surface where this parameter must appear (`.env.example`, `docker-compose.yml`, Terraform vars, Helm values, etc.) so an implementer can tick them off later.

## New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `package-name` | `latest` | {Why needed} |

If no new dependencies are required, explicitly state: "This change uses only existing dependencies."

## Implementation Details

### Step-by-Step Plan (for a future implementer)

#### Step 1: {First Step}
**File:** `path/to/file.py`
**Lines:** {approximate line numbers or "new file"}

```python
def new_function(
    param1: str,
    param2: int
) -> dict:
    """Description."""
    if not param1:
        raise ValueError("param1 is required")
    return {"status": "success", "data": process(param1, param2)}
```

### Error Handling
{How errors should be handled}

### Logging
{What should be logged and at what level}

## Observability
### Tracing / Metrics / Logging Points
{Spans, metrics, key log events}

## Scaling Considerations
- Current load assumptions
- Horizontal scaling
- Bottlenecks
- Caching strategy

## File Changes

### New Files

| File Path | Description |
|-----------|-------------|
| `src/feature/new_module.py` | {What it does} |

### Modified Files

| File Path | Lines | Change Description |
|-----------|-------|--------------------|
| `src/main.py` | ~50 | {What changes} |

### Estimated Lines of Code

| Category | Lines |
|----------|-------|
| New code | ~{X} |
| New tests | ~{X} |
| Modified code | ~{X} |
| **Total** | **~{X}** |

## Testing Strategy
{Pointer to testing.md - the full plan lives there}

## Alternatives Considered

### Alternative 1: {Name}
**Description:** ...
**Pros / Cons:** ...
**Why Rejected:** ...

### Comparison Matrix

| Criteria | Chosen | Alt 1 | Alt 2 |
|----------|--------|-------|-------|
| Complexity | Low | Med | High |

## Rollout Plan
- Phase 1: Implementation (out of scope for this skill)
- Phase 2: Testing
- Phase 3: Deployment

## Open Questions
- {Unresolved}

## References
- {Docs / similar implementations}
```

## Step 7: Expert Review (review.md)

Create a review document with feedback from multiple expert personas:

| Role | Reviewer | Focus |
|------|----------|-------|
| Frontend Engineer | Pixel | UI/UX, components, state, API integration |
| Backend Engineer | Byte | API design, data models, business logic, performance |
| SRE/DevOps Engineer | Circuit | Deployment, monitoring, scaling, infrastructure |
| Security Engineer | Cipher | AuthN/AuthZ, validation, OWASP, data protection |
| SMTS (Overall) | Sage | Architecture, code quality, maintainability |

**Write all five persona reviews yourself, in the main loop, one after another. Do NOT spawn subagents.** For each persona in the table, adopt that persona's perspective (the focus column above), review the issue spec and the LLD strictly from that angle, and write its section in the structure below - then move to the next persona. Assemble the five sections into `review.md` and write the Review Summary table. Keep each persona's review realistic and critical - identify actual issues, not just praise. The five perspectives are still all covered; you simply produce them sequentially instead of via parallel `Task` subagents.

For each reviewer, capture:
- **Strengths** observed in the design
- **Concerns** identified
- **New libraries / infra dependencies** required (with justification)
- **Better alternatives considered**
- **Recommendations**
- **Questions for author**
- **Verdict:** APPROVED / APPROVED WITH CHANGES / NEEDS REVISION

End with a Review Summary table and Next Steps. Reviews must be realistic, identifying actual issues rather than just praise.

## Step 7.5: Address Review Findings in the LLD (MANDATORY feedback loop)

The expert review is not decorative - it must change the design when it finds real problems. After assembling `review.md`, triage every finding by severity and **loop back into `lld.md` before writing any code**:

1. **Collect the blockers.** Gather every finding the reviewers marked as **critical**, **must-fix**, **NEEDS REVISION**, or an explicit "should fix" (including any concrete vulnerability, correctness bug, or bypass, regardless of the label a persona used). A single reviewer raising a critical/should-fix item is enough to trigger this loop - do not require consensus.
2. **If there are none** (every verdict is APPROVED with only optional/nice-to-have suggestions), record that in `review.md` ("No blocking findings; proceeding to implementation") and continue to Step 8.
3. **If there are any, revise `lld.md` to resolve each one before proceeding.** For every blocker: update the affected LLD sections (architecture, implementation steps, file changes, data models, non-goals) so the design actually addresses it, not just acknowledges it. Where a finding is a genuine limitation you are deliberately deferring, move it to the LLD's Non-Goals / Open Questions with an explicit, honest justification rather than silently dropping it - a deferred risk must be documented as unmitigated, never implied to be fixed.
4. **Re-validate the revised design against the codebase.** If a fix touches new files or call sites, confirm them in the cloned repo with a targeted re-read in the main loop (no subagents) so the revised LLD stays grounded, exactly as in Step 5.
5. **Reflect the resolutions in `review.md`.** Under each blocking finding, add a short "Resolution:" line stating how the revised LLD now addresses it (or why it is deferred). The review document should show the design responding to the critique, not a static snapshot.
6. **Only then proceed to Step 8 (testing plan) and Step 8.5 (implementation).** The code you implement in Step 8.5 must realize the *revised* LLD, so the patch already incorporates the review's must-fix items instead of shipping the flaws the reviewers caught.

This loop is what makes the review meaningful: the implementation reflects a design that has already survived critique. Do not skip it, and do not implement against the pre-review LLD when blocking findings exist. (If, while implementing in Step 8.5, you discover a further problem the review missed, the same rule applies: fix the LLD, note it, then continue - the design and the code must not diverge.)

## Step 8: Write Testing Plan (testing.md)

Create a comprehensive testing plan with **executable, copy-pasteable tests** covering every externally observable change. A future implementer should be able to walk through this document and verify the change works end-to-end without inventing test cases.

### When Each Test Category Applies

| Category | Include When |
|----------|--------------|
| Functional Tests (CLI / curl) | Change adds/modifies any HTTP endpoint or CLI command |
| Backwards Compatibility Tests | Change touches an existing endpoint, schema, CLI command, default, or model |
| UX Tests | Change adds/modifies any UI surface (web UI, CLI output, error messages) |
| Deployment Surface Tests (Docker, ECS, Helm) | Change adds/modifies any config parameter on any surface |
| E2E Tests | Change adds a workflow that spans multiple endpoints or services |

Always include the heading for each category. If a category does not apply, replace the body with: `**Not Applicable** - {one-line justification}`.

### Testing Plan Template (high level)

```markdown
# Testing Plan: {Feature Name}

*Created: {date}*
*Related LLD: `./lld.md`*
*Related Issue: `./github-issue.md`*

## Overview
### Scope of Testing
{1-2 sentences describing what is being tested and why}

### Prerequisites
- [ ] {Service running}
- [ ] {Auth tokens / fixtures available}

### Shared Variables
```bash
export REGISTRY_URL="http://localhost"
export ACCESS_TOKEN=$(jq -r '.access_token' .oauth-tokens/ingress.json)
```

## 1. Functional Tests
### 1.1 curl / HTTP Tests
{One subsection per new or modified endpoint with command, expected status, expected response, assertions, and a negative case}

### 1.2 CLI Tests
{One subsection per new or modified CLI command with exact invocation and expected output}

## 2. Backwards Compatibility Tests
{Pre-change request shapes still accepted; CLI without new flags behaves as before; defaults preserve prior behavior}

## 3. UX Tests
{Web UI flows; CLI output / error message clarity}

## 4. Deployment Surface Tests
### 4.1 Docker wiring
### 4.2 Terraform / ECS wiring
### 4.3 Helm / EKS wiring
### 4.4 Deploy and verify
### 4.5 Rollback verification

## 5. End-to-End API Tests
{Multi-step scenarios that exercise full business workflows}

## 6. Test Execution Checklist
- [ ] Section 1 (Functional) passes
- [ ] Section 2 (Backwards Compat) verified or marked Not Applicable
- [ ] Section 3 (UX) verified or marked Not Applicable
- [ ] Section 4 (Deployment) verified or marked Not Applicable
- [ ] Section 5 (E2E) verified or marked Not Applicable
- [ ] Unit tests added under `tests/unit/`
- [ ] Integration tests added under `tests/integration/`
- [ ] `uv run pytest tests/` passes with no regressions
```

### Guidance for Generating testing.md

1. Make tests copy-pasteable. Match the env var conventions used by existing scripts.
2. Cover every new endpoint and every new CLI command described in the LLD.
3. Anchor deployment tests on concrete files - reference exact Terraform/Helm/Docker file paths.
4. Mark Not Applicable explicitly. Do not silently omit sections.
5. Align with backwards-compat rules. Pre-change shapes must still be tested.
6. Do not invent endpoints or flags. Every URL, flag, and Terraform variable must exist in the LLD or codebase.

## Step 8.5: Implement the Change

This is the implementation step. Having designed the change (LLD) and planned its tests (testing.md), now **actually make the code change in the cloned repo**, then capture it as a reviewable, comparable patch. This is the benchmark's implementation phase - it measures whether the model can turn its own design into working code.

**Why edit the clone and capture a diff (rather than edit in place only, or copy files):** the clone at `{repo-path}` is a disposable temp checkout, so edits made there would be lost and would not be comparable across models. Editing in place lets the model work against real, resolvable code (imports resolve, it can re-read its own edits, patterns are visible), and capturing a `git diff` against the pinned baseline produces a durable, minimal, reviewable artifact - exactly the SWE-bench shape (the model's output IS a patch). The patch, not the mutated clone, is the artifact.

### 8.5.1 Implement against the LLD

> **CRITICAL - the code lives in `{repo-path}`, NOT your working directory. Anchor every edit to it or you will loop and run out of turns.** The cloned repo you are editing is a **separate folder** (a temp checkout, e.g. `/tmp/swe-clone-<task>/<repo-name>`), *not* the directory the harness runs you from. This is the single most common way this step fails: the model edits a guessed or working-dir-relative path (e.g. `src/registry/api/server_routes.py`), the file does not exist there, the `Edit`/`Write` silently no-ops or errors, the model re-describes the same edit, gets told "the file still contains the original code," tries again, and burns the entire turn budget without landing a single change - producing 4/6 artifacts (design only, no `patch.diff`). Avoid it with this discipline:
>
> 1. **Record the absolute repo root once, up front, and treat it as fixed for the whole step.** At the start of 8.5, resolve and remember it:
>    ```bash
>    REPO="{repo-path}"            # the exact clone path from Step 1.4, e.g. /tmp/swe-clone-remove-faiss/mcp-gateway-registry
>    git -C "$REPO" rev-parse --show-toplevel   # confirm it IS the clone root; use this value as $REPO
>    ```
>    Keep `$REPO` in mind for every file operation below. Never let it drift to your cwd, to `benchmarks/...`, or to a path you guessed from the LLD.
> 2. **Every file path you Read/Edit/Write in the target repo MUST be the absolute `$REPO/<path-relative-to-repo-root>`.** The LLD may write paths relative to the repo root (e.g. `registry/api/server_routes.py`); prepend `$REPO/` to get the real location (`$REPO/registry/api/server_routes.py`). Do **not** invent prefixes like `src/` that the LLD did not state.
> 3. **VERIFY THE PATH EXISTS BEFORE EDITING IT.** Before the first edit to any file, confirm it is really there - `ls "$REPO/<path>"` or `Read` it. If it is missing, do not guess variants; `find "$REPO" -name '<basename>'` (allowed) to locate the true path, then edit that. A `git -C "$REPO" ls-files | grep <name>` also finds the canonical path fast.
> 4. **If an edit does not land, STOP and re-locate - never retry the same path.** If `Edit`/`Write` errors, or a follow-up `Read`/`git diff` shows the file unchanged, the path or the `old_string` is wrong. Do **not** re-issue the same edit or reword it in a loop (that is exactly what exhausts the turn budget). Instead: re-find the file under `$REPO`, re-read its current contents, and issue the edit against the confirmed path/text. Two failed attempts on one file = re-locate, do not try a third blind.
> 5. **Periodically re-ground.** After a batch of edits, run `git -C "$REPO" diff --stat` to confirm your changes are actually accumulating in the clone. If it is empty when you believe you have edited files, your paths are wrong - fix the anchor before continuing, do not keep editing into the void.

Edit the files in `{repo-path}` to realize the design in `lld.md`. Treat the LLD's "Implementation Details" and "File Changes" sections as the plan of record:

- **Follow the LLD.** Make the edits the LLD calls out - new files, modified files, dependency and config changes. If while implementing you discover the LLD is wrong or incomplete, make the smallest correct change AND record the deviation in `implementation.md` (8.5.4); do not silently diverge.
- **Match the repo's conventions**, not this repository's `CLAUDE.md` - you are writing code in the *target* repo (its language, style, formatting, import order, test layout). `CLAUDE.md` guides the *design*; the *implementation* must look native to the repo being changed.
- **Keep the diff minimal and on-topic.** Change only what the task requires. Do not reformat unrelated files, bump unrelated dependencies, or leave debug prints. A tight diff is what makes cross-model comparison meaningful.
- **Make all edits yourself in the main loop.** Do NOT dispatch subagents to draft edits. Apply each file change directly and reconcile as you go so the final tree stays coherent.
- **Do NOT run the target repo's tests, linters, builds, or any command against it.** This skill reads and edits code but does not execute the target project (no `pytest`, `ruff`, `mypy`, `npm`, `cargo`, `go build`, `terraform`, `docker`, ...). The `testing.md` plan describes how a human would verify; running it is out of scope and keeps every model on an equal, execution-free footing. (You MAY run `git` on the clone - `status`, `diff`, `add` - since that inspects your own edits, not the project.)
- **Know the tool sandbox, and do not fight it.** This run executes with a fixed allowlist; only these tools work without a prompt: `Read`, `Glob`, `Grep`, `Write`, `Edit`, `Task`, and Bash **only** in the forms `git*`, `cd*`, `ls*`, `cat*`, `find*`, `head*`, `tail*`, `wc*`, `mktemp*`. Every other Bash command - `sed`, `awk`, `echo >`, `python`/`python -m py_compile`, `bash -n`, `uv`, `pip`, `pytest`, package managers - is **denied by design**, not by accident. Work within it:
  - **Make all file changes with the `Edit` and `Write` tools, never with shell.** Do not reach for `sed -i`, `echo >`, or a heredoc to edit or create files - those are `Bash` and will be denied. `Edit` (use `replace_all: true` when a string occurs more than once) and `Write` are always available and are the intended way to change code.
  - **Do not try to compile, syntax-check, or lint your edits.** `python -m py_compile`, `bash -n`, `ruff`, `mypy` are execution and are out of scope. Verify by re-reading your own edit and by `git -C "{repo-path}" diff`; that is the only check this benchmark expects.
  - **A denial is final - adapt or move on, do not retry the same command.** If a tool call is denied, do not re-issue it, reword it slightly, or loop trying variants (that just burns turns). Switch to an allowed tool (usually `Edit`/`Write` instead of shell), or accept the step cannot be done here and continue. Only stop and surface it if it is truly essential and has no allowed-tool equivalent.
  - **`.claude/` paths may be blocked by a built-in guard even though `Edit`/`Write` are allowed elsewhere.** If an edit to a `.claude/...` file in the clone is denied, do not retry it - note it as skipped in `implementation.md` and move on. Editing a repo's own skill/config files is rarely load-bearing for the task.

### 8.5.2 Capture the patch

After the edits are complete, capture everything changed relative to the pinned baseline (`BASE_SHA` from Step 1.4) as a single unified diff, written to the artifact folder as `patch.diff`:

```bash
# Includes new (untracked) files as well as modifications. Run from the clone.
git -C "{repo-path}" add -A
git -C "{repo-path}" diff --staged "$BASE_SHA" > "$ART_DIR/patch.diff"
# Sanity-check it is non-empty and re-readable, then confirm the artifact exists.
git -C "{repo-path}" apply --check "$ART_DIR/patch.diff" && echo "patch applies cleanly"
test -s "$ART_DIR/patch.diff" && echo "patch.diff written"
```

If `git apply --check` fails or the patch is empty, the implementation did not actually land - fix the edits and re-capture before moving on. The patch must apply cleanly onto a fresh checkout at `{ref}` so any reviewer (or an automated grader) can reconstruct the change.

### 8.5.3 Do not commit or push

`git add` is only used to stage the diff for capture. **Do NOT `git commit`, `git push`, create a branch in the clone, or open a PR.** The clone is disposable; the durable output is `patch.diff` under `$ART_DIR`. (Committing inside a `--depth 1` temp clone would also make the `$BASE_SHA` diff harder to reproduce.)

### 8.5.4 Write implementation.md

Write a concise `implementation.md` next to the other artifacts summarizing the code change so a reviewer understands it without reading the whole diff:

```markdown
# Implementation Summary: {Feature Name}

*Created: {date}*
*Baseline ref: `{ref}` (commit `{BASE_SHA}`)*
*Patch: `./patch.diff`*
*Related LLD: `./lld.md`*

## What Changed
{2-4 sentences: what the code change does, at a glance.}

## Files Touched

| File | Change | Lines +/- | Notes |
|------|--------|-----------|-------|
| `path/to/file.py` | modified | +42 / -8 | {what and why} |
| `path/to/new_module.py` | added | +90 / -0 | {what it provides} |

## How to Apply

```bash
git clone --branch {ref} --depth 1 {url} repo && cd repo
git apply /path/to/patch.diff
```

## Deviations from the LLD
{List any place the implementation differs from lld.md and why. "None - implemented exactly as designed." if it matches.}

## Not Implemented / Follow-ups
{Anything the LLD called out that was intentionally left out of this patch, with a one-line reason. "None." if complete.}

## Verification (not executed)
{Point to testing.md. State explicitly that tests were designed but not run, per skill constraints.}
```

Populate the +/- line counts from `git -C "{repo-path}" diff --staged "$BASE_SHA" --numstat`. Keep `implementation.md` honest: if the patch only partially realizes the LLD, say so here rather than overstating completeness.

## Step 9: Present Summary & Seek Guidance

**Completion check first -- all six artifacts are mandatory.** Before you present anything or stop, verify that every one of the six files exists in the artifact directory (`$ART_DIR`): `github-issue.md`, `lld.md`, `review.md`, `testing.md`, `patch.diff`, `implementation.md`. A run that produces only some of them is a failure that scores 0, no matter how good the ones you wrote are. If any is missing, write it now before finishing -- do not stop with an incomplete set. A quick check:

```bash
for f in github-issue.md lld.md review.md testing.md patch.diff implementation.md; do
  test -f "$ART_DIR/$f" && echo "OK   $f" || echo "MISSING $f"
done
```

If anything is `MISSING`, complete it (for `patch.diff`/`implementation.md`, that means implementing the change per `lld.md` in the clone and capturing the diff -- Step 8.5) before proceeding.

After producing the six artifacts, present a clear summary to the user. **Do not run the target repo's tests, push, commit, or open a PR** - the implementation lives only as `patch.diff`. This skill ends at delivery of the design-plus-implementation package.

```markdown
## Delivery Summary

### Documents Created

Locations below are shown relative to the repo root; write them to the **absolute** `$ART_DIR/<file>` path from Step 3 (never a bare `benchmarks/...` path -- see the Step 3 warning).

| Document | Location | Description |
|----------|----------|-------------|
| GitHub Issue | `$ART_DIR/github-issue.md` | Issue specification |
| Low-Level Design | `$ART_DIR/lld.md` | Technical design |
| Expert Review | `$ART_DIR/review.md` | Multi-persona review |
| Testing Plan | `$ART_DIR/testing.md` | All test categories |
| Implemented Patch | `$ART_DIR/patch.diff` | git diff of the code change (applies onto `{ref}`) |
| Implementation Summary | `$ART_DIR/implementation.md` | Files touched, how to apply, deviations |

### Implemented Change

| Metric | Value |
|--------|-------|
| Files changed | {N} |
| Lines added / removed | +{X} / -{Y} |
| Patch applies onto `{ref}` | yes (git apply --check passed) |
| Fully realizes the LLD | yes / partial (see implementation.md) |

### Review Verdicts

| Reviewer | Verdict | Blockers | Key Recommendations |
|----------|---------|----------|---------------------|
| Frontend (Pixel) | {verdict} | {count} | {summary} |
| Backend (Byte) | {verdict} | {count} | {summary} |
| SRE (Circuit) | {verdict} | {count} | {summary} |
| Security (Cipher) | {verdict} | {count} | {summary} |
| SMTS (Sage) | {verdict} | {count} | {summary} |

### Configuration Parameters Proposed

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `PARAM_NAME` | type | value | {description} |

### New Dependencies Proposed

| Package | Type | Required By |
|---------|------|-------------|
| `package-name` | Python | Backend |

### Implemented Effort (from patch.diff)

| Category | Lines of Code |
|----------|---------------|
| New code | ~{X} |
| Tests (if the patch adds them) | ~{X} |
| Modified | ~{X} |
| **Total** | **~{X}** |
```

### Seeking Guidance

After the summary, ask the user:

1. Are there any blockers from the expert review you want me to address by revising the design or the patch?
2. Would you like me to refine any artifact (e.g. expand a specific LLD section, add more test cases, tighten the patch)?
3. Should I open the GitHub issue against the upstream repo using `github-issue.md`?

Do not run the target repo's tests, push, commit, or open a PR until the user explicitly authorizes it as a separate request. The code change is delivered as `patch.diff`, not applied to any real branch.

---

## Important Guidelines

### Design Principles
- Favor simple designs over unnecessary complexity
- Prefer straightforward code over clever solutions
- Design for maintainability by entry-level developers
- Add observability from the start, not as an afterthought

### Documentation Quality
1. **Be Thorough**: The LLD should be detailed enough that someone unfamiliar with the codebase can implement it
2. **Use Diagrams**: ASCII diagrams help visualize the design
3. **Include Code**: Show actual or pseudo-code for key functions
4. **Specify Files**: Always mention which files to create/modify and approximate line numbers
5. **Consider All Aspects**: Think about error handling, logging, testing, and deployment
6. **Expert Reviews**: Make the reviews realistic - identify actual issues, not just praise

### Hard Stops
1. **Implement the change (Step 8.5), but only in the disposable clone.** Edit `{repo-path}` to realize the LLD; the change is delivered as `patch.diff`, never as a commit on any real branch.
2. **Do not run the target repo's tests, linters, or builds.** Read and edit the code; do not execute the project (no `pytest`/`ruff`/`mypy`/`npm`/`cargo`/`go`/`terraform`/`docker`). `git status`/`diff`/`add` on the clone is allowed (it inspects your own edits).
3. **Edit only via `Edit`/`Write`; the Bash allowlist is `git*`/`cd*`/`ls*`/`cat*`/`find*`/`head*`/`tail*`/`wc*`/`mktemp*` and nothing else.** `sed`/`awk`/`echo >`/`python`/`py_compile`/`bash -n`/`uv`/`pip` are denied by design. A denial is final - do not retry the same command; switch to an allowed tool or move on (see Step 8.5.1).
4. **Do not commit, push, create a branch in the clone, or open a PR.** `git add` is used only to stage the diff for capture.
5. **Do not touch anything outside the clone and `$ART_DIR`.** The six artifacts and the in-place edits to `{repo-path}` are the only writes.

## Example Usage

User: "Run task 1 for mcp-gateway-registry with claude-opus-4-7."

1. Look up task 1 in `benchmarks/swe-benchmark-data/README.md` (`remove-faiss`). Confirm `repo-name = mcp-gateway-registry`, `problem-name = remove-faiss`, `model-name = claude-opus-4-8`. Resolve a local checkout of the target repo at the listed tag (a temp clone under `/tmp` is fine); if none exists, ask the user for the GitHub URL and tag, announce the clone command, and wait for approval.
2. Quick codebase review of the checkout to find every FAISS reference (imports, dependencies, configs, docs)
3. Create `benchmarks/swe-benchmark-data/claude-opus-4-8/mcp-gateway-registry/remove-faiss/`
4. Write `github-issue.md` describing the FAISS removal task (problem, scope, acceptance criteria, out-of-scope)
5. Deep code analysis of FAISS usage and the maintained replacement
6. Write `lld.md` covering: files to edit, dependency removals, doc updates, fallback path
7. Write `review.md` with backend, SRE, security, and SMTS verdicts
8. Write `testing.md` with import-removal greps, backwards-compat tests, and a build/test pass plan
8.5. Implement the removal in the clone (delete FAISS imports/deps/config/docs, wire the replacement), then capture `patch.diff` and write `implementation.md`
9. Present the six-artifact summary (including the patch stats) and ask whether to refine anything or open the GitHub issue upstream

When the same problem is later run with a different model (e.g. `claude-sonnet-5`), repeat the workflow into `benchmarks/swe-benchmark-data/claude-sonnet-5/mcp-gateway-registry/remove-faiss/`. The two model folders make per-model artifacts directly comparable on the same problem.

---

## Constraints

- **Implementation is in scope; execution is not.** Produce the four design artifacts AND the implemented `patch.diff` + `implementation.md` (six total). Edit the clone to do it, but do NOT run pytest, ruff, mypy, terraform, docker, or any build/test command against it. Reading and editing the code is fine; executing the project is out of scope (keeps every model on equal footing).
- **No commit/push/PR.** The change is delivered as a patch only.
- **No emojis, clever code, or em-dashes** in any output.
- **Naming**: always "Amazon Bedrock" (never "AWS Bedrock").
- **Best Practices**: the *design* recommendations should follow `CLAUDE.md` (logging, Pydantic, modularity); the *implementation* must match the target repo's own conventions so the patch looks native to that codebase.

### Benchmark Isolation (CRITICAL)

This skill is a benchmark. Each model run must be completely independent so artifacts are directly comparable. Read the cloned source repo only; do not read sibling model artifacts or communicate with other sessions. Specifically:

- **Do NOT read any files under `benchmarks/swe-benchmark-data/`** other than the model's own target folder (`{model-name}/{repo-name}/{problem-name}/`). Sibling model folders (e.g. `claude-opus-4-8/`, `kimi-k2-thinking/`, etc.) contain artifacts from other benchmark runs — reading them contaminates the benchmark.
- **Do NOT read `benchmarks/swe-benchmark-data/README.md`** during analysis. The task description in this `/swe3` invocation is the only allowed input from the benchmark directory.
- **Do NOT use the `claude-peers` MCP tool** (`mcp__claude-peers__*`) to message, list, or coordinate with other Claude Code sessions during a `/swe3` run. Each session must produce its design and implementation independently.
- **The only allowed code source** is the cloned target repo at `{repo-path}` (a temp clone such as one under `/tmp`, as resolved in Step 1.4). Read that thoroughly; ignore everything else under `benchmarks/`.

If the user explicitly asks you to compare with prior runs after artifacts are written, that is a separate request — done after the six artifacts are saved, not during their production.

### Self-Review (CRITICAL)

After writing all six artifact files (`github-issue.md`, `lld.md`, `review.md`, `testing.md`, `patch.diff`, `implementation.md`), go back and re-read each one against the original task description and clarifying answers. Verify you have not missed any requirement, acceptance criterion, or constraint that was stated earlier. In particular, cross-check that `patch.diff` actually implements what `lld.md` designed and what `implementation.md` claims - re-read the diff, do not trust memory. Confirm the patch still passes `git apply --check`. If you find a gap, fix the artifact or the edits and re-capture the patch immediately before presenting the summary in Step 9.
