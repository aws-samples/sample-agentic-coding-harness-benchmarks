---
name: release-notes
description: "Create release notes for a new version tag of this benchmark repository. Decides the correct semver bump from the change set (new dataset or new model is a minor; a methodology change or completely new functionality is a major; everything else is a patch), gathers all commits, PRs and closed issues since the previous release, writes the release notes markdown following the project format, then opens a PR and tags the merge commit. Use when the user wants to cut a release, write release notes, decide the next version number, or tag a release."
license: Apache-2.0
metadata:
  author: Amit Arora
  version: "1.0"
---

# Release Notes Skill

Use this skill when the user wants to cut a release of `agentic-coding-harness-benchmarks`. It decides the correct next version, gathers every change since the previous release, writes structured release notes in the project format, opens a PR (this repo never commits to `main` directly), and tags the merge commit after the PR is approved and merged.

This repo is a benchmark harness plus its published results, docs and slides. It is **not** a deployed service, so there is no Docker Compose / Helm / Terraform upgrade path and no live smoke test. What a release captures instead is: new models benchmarked, new datasets, methodology changes, new harness functionality, and the results / frontier / doc updates that follow.

## Versioning scheme (read this first)

Releases follow [Semantic Versioning 2.0.0](https://semver.org): `MAJOR.MINOR.PATCH`, **bare, with no `v` prefix** (e.g. `0.2.0`, not `v0.2.0`). The one existing non-version tag, `media-assets`, hosts video/asset release attachments and is not a release: ignore it when finding the latest version.

### What each bump means in this repo

The "public API" of a benchmark repo is its dataset, its measured models, and its measurement methodology. Map the change set to a bump with these rules, in priority order (if a change set matches more than one, take the highest):

| Bump | Trigger | Examples |
|------|---------|----------|
| **MAJOR** | A methodology change, or completely new functionality | The judge or scoring changes so old and new scores are no longer comparable; the cost-per-task derivation changes; the cost/quality frontier is recomputed on a new basis; a new benchmark type or hosting path is added; the config schema changes in a backward-incompatible way. |
| **MINOR** | A new dataset, or a new model added to the frontier | A new SWE dataset (e.g. a new `sweN` task set); a new model benchmarked and folded into results, `models.json`, charts and docs; a new skill or a new backward-compatible config knob. |
| **PATCH** | Everything else, backward compatible | Doc / slide fixes, chart or frontier regeneration for an existing model, bug fixes in a script, dependency bumps, typo and link fixes. |

Because a methodology change makes previously published numbers no longer directly comparable, it is the clearest MAJOR signal: **when scores from before and after the change cannot sit in the same table, bump MAJOR.** New datasets and new models are additive, so they are MINOR even though they add a lot of data.

### Choosing the first version and the 1.0.0 line

There are no version tags yet, so the first release sets the convention:

- In `0.y.z` the methodology and results are still settling and nothing is promised to stay comparable across releases. This is the honest place to start: **use `0.1.0` for the first release** unless the user says the harness and its published frontier are already something people depend on.
- Go to `1.0.0` when you are ready to stand behind a stable methodology and results contract, and to bump MAJOR whenever a methodology change breaks it. `1.0.0` is a commitment, not a maturity trophy earned after `0.9.0` -- there is no rule that the first release must be `0.9.0`, and you never need to march through `0.9.0` before `1.0.0`.

**Always confirm the computed version with the user** before writing anything. State the bump you chose and the one-line reason (e.g. "MINOR: adds the minimax-m3 model to the omp/swe3 frontier"), and let them override.

## Input

- An optional version, `{major}.{minor}.{patch}` (e.g. `0.2.0`), **no `v` prefix**. If the user gives a `v`-prefixed version, strip the `v` and confirm.
- If no version is given, compute the recommended bump from the change set (see above) and confirm it with the user.

## Output

- `docs/release-notes/{version}.md` -- the release notes file (e.g. `docs/release-notes/0.2.0.md`).
- On the first release only: `docs/release-notes/README.md` -- a small index of releases, and one new row in the root `README.md` "Documentation map" table pointing at it.
- A git tag `{version}` on the merge commit, created **after** the release-notes PR is merged to `main`.

## Workflow

### Step 0: Verification gate

A release must not ship code or results that fail the repo's own checks. Before writing notes, confirm the checks pass for every `uv` project touched since the base version (`benchmarks/` and/or `self-hosted/vllm/`). CI gates on the unittest suites, so those are the hard block.

1. **Ask the user, using AskUserQuestion**, whether the checks have already been run and passed. Offer:
   - "Yes, they passed" (proceed to Step 1)
   - "No, run them now" (Recommended -- run them, see below)
   - "Skip them" (proceed only after the warning below)

2. **If the user asks you to run them**, run the AGENTS.md check line from inside each touched `uv` project. At minimum the test suites must pass:
   ```bash
   # From benchmarks/ (and/or self-hosted/vllm/) -- whichever owns the changed files
   uv run ruff check --fix . && uv run ruff format .
   uv run python -m unittest discover -s tests
   ```
   - **If any test FAILS, STOP.** Do not cut the release. Report which tests failed and help investigate. A release must not be cut with a failing suite.
   - Run the fuller line (`bandit`, `mypy`) too when the change is security- or type-sensitive; treat a new failure as a blocker.

3. **If the user chooses to skip**, warn once that the release is being cut without verification, then proceed only if they confirm.

Only after this gate is resolved continue to Step 1.

### Step 1: Determine the new version tag

1. Compute the recommended bump from the change set using the [Versioning scheme](#versioning-scheme-read-this-first) rules, or take the version the user gave.
2. Normalize to bare semver (strip any leading `v`).
3. Verify the tag does not already exist: `git tag -l {version}`.
4. If it exists, ask the user whether to move it or pick a different version.
5. **Confirm the version and the one-line bump reason with the user before proceeding.**

### Step 2: Determine the base version (ask the user to confirm)

Release notes are incremental from the previous release.

1. List existing release notes and version tags (excluding the `media-assets` asset tag):
   ```bash
   ls docs/release-notes/*.md 2>/dev/null
   git tag --sort=-v:refname | grep -E '^v?[0-9]+\.[0-9]+\.[0-9]+$'
   ```
2. **First release (no version tags):** there is no base. Diff from the repository's first commit and summarize the harness as it stands, rather than an incremental changelog:
   ```bash
   FIRST_COMMIT=$(git rev-list --max-parents=0 HEAD | tail -1)
   ```
   Use `$FIRST_COMMIT` wherever the steps below say `{base_tag}`.
3. **Otherwise:** take the most recent version tag as the base and **confirm it with the user** using AskUserQuestion (present it as the recommended option, with the 2-3 previous tags as alternatives). The user may want to span multiple versions.

### Step 3: Gather all changes between base and HEAD

Run these in parallel to gather the change data:

```bash
# All commits (including merges) between base and HEAD
git log {base_tag}..HEAD --oneline

# Non-merge commits (for detailed change analysis)
git log {base_tag}..HEAD --oneline --no-merges

# Merge commits (to extract PR numbers)
git log {base_tag}..HEAD --oneline --grep="Merge pull request"

# Contributors -- direct authors of commits on main
# WARNING: this misses co-authors of squash-merged PRs (Step 4 #9 explains).
git log {base_tag}..HEAD --format="%aN" | sort | uniq -c | sort -rn

# Contributors -- per-PR commit authors (catches squash-merge co-authors)
# Squash merges collapse a branch's commits into one commit on main authored by
# the merger, so `git log` above will not show the real code authors.
# `gh pr view --json commits` returns the original branch commits with authors intact.
for pr in $(git log {base_tag}..HEAD --oneline --grep="Merge pull request\|(#[0-9]\+)$" | grep -oE "#[0-9]+" | tr -d '#' | sort -u); do
  gh pr view $pr --json number,author,commits \
    --jq '"PR #\(.number) | opener: \(.author.login) | commit_authors: \([.commits[].authors[].name] | unique | join(", "))"' 2>/dev/null
done

# Config-schema changes (this repo's analog of an env-var diff): the RunnerConfig
# model, the example config, and the pricing table. Any change here is a config
# change users may need to act on; a backward-incompatible one is a MAJOR signal.
git diff {base_tag}..HEAD --stat -- \
  benchmarks/scripts/runner_config.py \
  benchmarks/config/runner.example.yaml \
  self-hosted/vllm/pricing.json

# New models benchmarked (per-model serving guides and the frontier data)
git diff {base_tag}..HEAD --stat -- self-hosted/vllm/models/ docs/metrics/

# New or changed skills
git diff {base_tag}..HEAD --stat -- .claude/skills/

# Dataset changes (new sweN task sets)
git diff {base_tag}..HEAD --stat -- benchmarks/dataset/

# Closed issues since the base tag was cut (floor by the base tag's commit date)
BASE_TAG_DATE=$(git log -1 --format=%cI {base_tag})
gh issue list --state closed --limit 200 --json number,title,closedAt,labels \
  --jq ".[] | select(.closedAt >= \"$BASE_TAG_DATE\") | \"\(.number) | \(.title) | \(.closedAt)\""

# Closed issues referenced by merged PRs (most reliable mapping)
for pr in $(git log {base_tag}..HEAD --oneline --grep="Merge pull request" | grep -oE "#[0-9]+" | tr -d '#' | sort -u); do
  gh pr view $pr --json number,title,closingIssuesReferences \
    --jq '"\(.number) | \(.title) | closes: \(.closingIssuesReferences | map("#\(.number)") | join(","))"' 2>/dev/null
done
```

### Step 4: Categorize changes

Analyze the commits and PRs and sort them. The bump you chose in Step 1 should be justified by what lands in the first few categories.

1. **New models benchmarked** (MINOR): a model added to the frontier -- new `models.json` / `docs/metrics/` frontier entries, a new per-model guide under `self-hosted/vllm/models/`, new charts and results docs. Name the model, the harness/skill and dataset it was run on, and the headline number.
2. **New datasets** (MINOR): a new `sweN` task set or a new set of tasks under `benchmarks/dataset/`. Say how many tasks and how it differs from existing datasets (never merge scores across datasets).
3. **Methodology changes** (MAJOR): anything that changes what a number means -- the judge, scoring, cost-per-task derivation, frontier basis, or a backward-incompatible config-schema change. Spell out why old and new numbers are no longer comparable.
4. **New functionality** (MAJOR): a new benchmark type, hosting path, or major harness capability.
5. **Skills** (MINOR for a new skill; PATCH for a fix): new or changed entries under `.claude/skills/`.
6. **Results and frontier updates** (PATCH when just regenerated for existing models): recomputed charts, frontier JSON, decks.
7. **Config changes**: new or changed `RunnerConfig` knobs, `runner.example.yaml`, `pricing.json` (from the diff above).
8. **Bug fixes** (PATCH): `fix:`-prefixed commits or PRs labeled `bug`.
9. **Documentation / slides** (PATCH): `docs:` commits, changes under `docs/`, deck HTML/PDF regen.
10. **Dependency updates**: `pyproject.toml` / lockfile bumps in either `uv` project.
11. **Closed issues**: build from the `closingIssuesReferences` of every merged PR (GitHub auto-closes `Closes #N` / `Fixes #N`), supplemented by manually-closed issues whose `closedAt` falls between the base-tag commit date and HEAD. De-duplicate by issue number.
12. **Contributors**: build the union of TWO sources, because neither alone is complete:
    - **Direct authors on main**: `git log {base_tag}..HEAD --format="%aN"`.
    - **Per-PR commit authors**: for every merged PR, `gh pr view <num> --json commits --jq '[.commits[].authors[].name] | unique'`, unioned. This catches co-authors of squash-merged PRs whose branch commits collapse into one commit authored by the merger. Without it, every contributor on a squash-merged branch except the merger is silently dropped.

    Resolve each unique name to a GitHub login from a real PR: `.author.login` if they opened one, or `.commits[].authors[].login` if they were a co-author only. Verify uncertain guesses with `gh api users/<candidate>` (404 means wrong). **Never synthesize a username from a display name** -- "Amit Arora" is `aarora79`, not `amitarora`.

### Step 5: Write the release notes

Write `docs/release-notes/{version}.md` following this structure. Keep every prose section to the writing skill ([.claude/skills/writing/SKILL.md](../writing/SKILL.md)): plain words, no em-dashes, no emojis, and every number cited to the file it came from.

```markdown
# Release {version} - {short title summarizing the headline change}

**{Month} {Year}**

Version bump: **{MAJOR|MINOR|PATCH}** -- {one-line reason, e.g. "adds the minimax-m3 model to the omp/swe3 frontier"}.

---

## Upgrading from {base_version}

This repo is cloned and run, not deployed, so upgrading is a pull plus a dependency sync.

```bash
git pull origin main
git checkout {version}

# Sync the uv project(s) you run. If dependencies changed, sync both:
cd benchmarks && uv sync && cd ..
cd self-hosted/vllm && uv sync && cd ../..
```

### Methodology changes

{For a MAJOR release, state exactly what changed in how a number is produced and
why results from {base_version} and {version} must not be compared or merged in
the same table. For MINOR / PATCH, write: "None. Numbers remain comparable with
{base_version}."}

### Config changes

| File | Change | Action needed |
|------|--------|---------------|
| {runner_config.py / runner.example.yaml / pricing.json} | {what changed} | {what a user must update, or "none -- has a default"} |

{If no config changed, write: "No configuration changes in this release."}

---

## New models

### {Model name} on {harness} / {dataset}

{What it scored and where it sits on the frontier, cited to the frontier JSON /
results doc. Link the per-model serving guide.}

{Repeat per model. Omit the whole section if no model was added.}

---

## New datasets

- **{dataset name}** -- {N tasks, how it was sourced, how it differs from existing sets, and the reminder that its scores never merge into another dataset's table}.

{Omit if no dataset was added.}

---

## New functionality

### {Feature name}

{What it does and why it matters. Link the PR.}

[PR #{number}](https://github.com/aws-samples/sample-agentic-coding-harness-benchmarks/pull/{number})

{Omit if none.}

---

## What's changed

{Group remaining changes by category. Only include categories that have changes.}

### Skills
- {change} (#{pr})

### Results and frontier
- {change} (#{pr})

### Documentation and slides
- {change} (#{pr})

---

## Bug fixes

- {fix} (#{pr})

{Omit if none.}

---

## Closed issues

| Issue | Title | Closed by |
|-------|-------|-----------|
| #{n} | {title} | {PR #{pr} or "manual"} |

{Sorted by issue number descending. If none: "No issues were closed in this release window."}

---

## Pull requests included

| PR | Title |
|----|-------|
| #{n} | {title} |

{All merged PRs between base and HEAD, sorted by PR number descending.}

---

## Dependency updates

| Package | Previous | Updated | Project |
|---------|----------|---------|---------|
| {package} | {old} | {new} | {benchmarks / self-hosted/vllm} |

{Omit if none.}

---

## Contributors

Thank you to everyone who contributed to this release:

- **{Full Name}** ([@{login}](https://github.com/{login}))

{Union of direct authors and per-PR commit authors (Step 4 #12). Every login
resolved from a real PR. Sorted by commit count descending.}

---

**Full Changelog:** [{base_version}...{version}](https://github.com/aws-samples/sample-agentic-coding-harness-benchmarks/compare/{base_version}...{version})
```

For the **first release** there is no `{base_version}` to diff against: drop the "Upgrading from" section's diff framing, describe the harness and its published frontier as it stands at this version, and omit the "Full Changelog" compare link (or point it at the first commit).

### Step 6: Present the draft for review

After writing the file:

1. Tell the user the file is at `docs/release-notes/{version}.md`.
2. Summarize: the chosen bump and why, new models, new datasets, methodology changes, PR count, bug-fix count, closed-issue count, contributor count.
3. Ask the user to review and confirm, or request changes.

### Step 7: Security gate, PR, and tag

This repo **never commits or merges directly to `main`.** Cut the release through a PR, and tag only after it merges.

1. **Run the `security-check` skill** over the pending diff (mandatory gate before any commit or PR). Resolve every blocker before continuing; do not commit while the verdict is NEEDS REVISION.

2. **On the first release only**, also create `docs/release-notes/README.md` as a short index (a table of `| Version | Date | Highlights |` rows, newest first) and add one row to the root `README.md` "Documentation map" table pointing at it. On later releases, prepend a row to that index instead.

3. **Commit on the feature branch** (you should already be on one; if on `main`, create `release/{version}` first):
   ```bash
   git add docs/release-notes/{version}.md
   # first release also:
   git add docs/release-notes/README.md README.md
   git commit -m "docs: add {version} release notes"
   git push -u origin HEAD
   ```

4. **Open a PR** with a professional description that summarizes the release (no assistant attribution anywhere):
   ```bash
   gh pr create --title "Release {version}" --body-file <(printf '...') --base main
   ```

5. **After the PR is approved and merged to `main`**, tag the merge commit and push the tag:
   ```bash
   git checkout main && git pull origin main
   git tag {version}          # bare semver, no v prefix
   git push origin {version}
   ```

6. **Optionally publish a GitHub Release** from the tag using the notes file:
   ```bash
   gh release create {version} --title "Release {version}" --notes-file docs/release-notes/{version}.md
   ```

7. **Verify:**
   ```bash
   git tag -l {version} --format="%(refname:short) -> %(objectname:short)"
   ```
   Tell the user the notes are merged and the tag is created and pushed.

## Important rules

- **Pick the bump by the repo's rules, not by feel:** new dataset or new model is a MINOR; a methodology change or completely new functionality is a MAJOR; everything else is a PATCH. A methodology change is the clearest MAJOR because it breaks comparability of published numbers. Always confirm the computed version with the user.
- **Bare semver, no `v` prefix.** Ignore the `media-assets` tag when finding the latest version.
- **Never commit or merge to `main` directly.** Cut every release through a feature branch and a PR; tag the merge commit only after it lands.
- **Always run the `security-check` gate** before committing and before opening the PR.
- **Run the Step 0 verification gate first.** If any test suite fails, STOP and do not cut the release. Only the user's explicit skip may bypass it, warned once.
- **Never skip the base-version confirmation** in Step 2; the user may span multiple versions.
- **Never include emojis** in the release notes or commits, and run the [writing skill](../writing/SKILL.md) revision pass over the prose before committing.
- **Never name the coding assistant** anywhere -- not in the notes, commit messages, PR title or body.
- **Check every number against its source** (a `performance-summary.json`, a frontier JSON, a results doc) and cite it, per AGENTS.md.
- **Always credit squash-merge co-authors** via `gh pr view <num> --json commits`, and **never synthesize a GitHub username from a display name** ("Amit Arora" is `aarora79`).

## Example usage

```
User: /release-notes
Assistant: (computes the bump from the change set, e.g. "MINOR -- adds the
minimax-m3 model to the omp/swe3 frontier", confirms 0.2.0 with the user,
runs the verification gate, gathers changes since the base tag, writes
docs/release-notes/0.2.0.md, runs security-check, opens a PR, and tags after merge)
```
