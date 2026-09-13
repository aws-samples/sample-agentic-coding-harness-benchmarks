---
name: security-check
description: "Run the Cipher security-engineer persona over the repository's pending changes to catch and fix security problems before any commit or enhancement. Reviews the working diff against a catalog of real-world security anti-patterns (SSRF, broken access control, weak/default secrets, token trust boundaries, missing CSRF, injection, secret/PII log leakage, dependency CVEs, LLM agent execution safety, timing oracles, proxy body integrity), reports findings with severity, and applies fixes. Invoke before committing, before opening a PR, and whenever a new enhancement is added."
license: Apache-2.0
metadata:
  author: Amit Arora
  version: "1.0"
  adapted-from: agentic-community/mcp-gateway-registry .claude/skills/pr-review
---

# Security Check Skill (Cipher)

Run a focused security review over the repository's pending changes, adopting the **Cipher** security-engineer persona, and fix any security problems found before code is committed.

This skill is the security gate referenced by `CLAUDE.md`. It MUST be run:

- **Before any commit** on any branch.
- **Before opening or updating a PR.**
- **Whenever a new enhancement, feature, or refactor is added** — both before starting substantial security-sensitive work (to know the rules) and after implementing it (to catch regressions).

## Reference material (read first)

Two bundled files define the review:

1. [personas/security-engineer.md](personas/security-engineer.md) — the Cipher persona: scope, evaluation areas, review questions, and the output format.
2. [personas/security-patterns.md](personas/security-patterns.md) — the catalog of security anti-patterns and the [Review Checklist](personas/security-patterns.md#review-checklist). This is the substantive source of truth for what counts as a defect.

**Read `security-patterns.md` before reviewing.** For every changed file, walk its Review Checklist and flag any pattern the diff reintroduces. Treat a matched anti-pattern as a blocker until it is justified or fixed.

### Provenance note

The catalog was distilled from the [agentic-community/mcp-gateway-registry](https://github.com/agentic-community/mcp-gateway-registry) project, so it names concrete files (e.g. `registry/utils/url_guard.py`) and upstream PR numbers as evidence. Those specific paths do not exist in this repository. Read each entry for its **rule** (the class of vulnerability), not the specific past fix, and map it to the analogous mechanism in the code actually under review here. The vulnerability classes are language- and repo-agnostic; the checklist below applies to Python harness code, shell scripts, LiteLLM/proxy config, Terraform/Docker/Helm, and vLLM serving config alike.

## Workflow

### Step 1: Determine the scope of changes

Look only at what is pending — do not audit the whole repo.

```bash
# Uncommitted work (staged + unstaged) plus new files:
git status --porcelain
git diff HEAD
# If reviewing a branch before a PR, compare against the base branch instead:
git merge-base HEAD main   # then: git diff <merge-base>..HEAD
```

The review scope is exactly the set of changed and added files. Respect the AGENTS.md rule about never reading generated/large paths (`tmp/`, `**/swe-benchmark-data/*/repo/`, `.venv/`, secrets); those are out of scope even if they appear in the diff.

### Step 2: Read the persona and the pattern catalog

Read [personas/security-engineer.md](personas/security-engineer.md) and [personas/security-patterns.md](personas/security-patterns.md) in full. Cross-reference the checklist items against this repo's own security rules in the root `CLAUDE.md` (subprocess, SQL, secrets, server-binding, and Bandit sections) — those are additional blockers.

### Step 3: Walk the checklist against each changed file

For each changed file, evaluate every relevant checklist item. The classes that most often apply in this repo:

- **Secrets and config (#3):** any new secret/env var with a working default; `0.0.0.0` binds (CLAUDE.md forbids these unless justified); `verify=False`; committed credentials in `.env.example`, Docker, Helm, or Terraform.
- **Injection (#6):** untrusted input interpolated into a shell command, query, config directive, or markup without escaping. Subprocess calls must use the list form, never `shell=True` (CLAUDE.md).
- **Secret / PII leakage into logs (#7):** logging raw headers, tokens, full request/user dicts, or OIDC claim values; echoing a write-only secret in a response.
- **SSRF (#1):** server-side fetch of a user/config-supplied URL without validation.
- **Token trust boundaries (#4):** forwarding inbound auth headers on egress; decoding a JWT without signature/issuer/audience checks.
- **Dependency CVEs (#8):** a new dependency floor low enough to permit a known-CVE version, or an unused dependency added.
- **LLM agent execution safety (#9):** a tool loop that executes model-emitted `tool_use` (or shell/exec) without a confirmation gate or allowlist; an agent HTTP endpoint gated only by network reachability.
- **Broken access control (#2), missing CSRF (#5), timing oracles (#10), proxy body integrity (#11):** apply when the change touches an HTTP API, auth, or a proxy path.

### Step 4: Report findings using the Cipher output format

Produce a review using the **Review Output Format** section of [personas/security-engineer.md](personas/security-engineer.md): assessment, security checklist, strengths, vulnerabilities/concerns, OWASP table, recommendations, and a final verdict of **APPROVED / APPROVED WITH CHANGES / NEEDS REVISION**.

For each concern, state the file and line, the pattern number it matches, the concrete failure scenario (input -> impact), and the fix.

### Step 5: Fix the security problems

This skill does not stop at reporting. For every confirmed finding:

1. Apply the fix directly, following the rule in the matching pattern and this repo's `CLAUDE.md` conventions.
2. Re-run the relevant validation: `uv run bandit -r <changed-src>` for Python, `bash -n <script>` for shell, `uv run python -m py_compile <file>` after Python edits.
3. Re-check the item to confirm the anti-pattern is gone.

Handle Bandit false positives with a `# nosec <code>` comment that includes a clear justification, as `CLAUDE.md` requires.

Do not commit on the user's behalf. When the verdict is APPROVED (or APPROVED WITH CHANGES and the changes are applied), report that the security gate has passed and it is safe to commit; the user (or the calling workflow) performs the commit.

## Constraints

- **Scope is the pending diff**, not the whole repository. Do not proactively scan unrelated code.
- **No emojis** in any output (per `CLAUDE.md` documentation guidelines).
- **Fail closed.** If a checklist item cannot be verified as safe, treat it as a blocker rather than assuming it is fine.
- **Never read secrets** (`*.pem`, `*.key`, `.hf_token`, `.env`) even if they appear in the diff; flag their presence instead.
