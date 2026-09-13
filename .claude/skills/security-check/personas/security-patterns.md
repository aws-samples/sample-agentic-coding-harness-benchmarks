# Security Patterns and Anti-Patterns

A catalog of recurring security defect classes, distilled into reusable rules, so that new code and code reviews do not reintroduce the same classes of vulnerability.

These patterns were originally distilled from defects that shipped and were fixed in the [agentic-community/mcp-gateway-registry](https://github.com/agentic-community/mcp-gateway-registry) project (a FastAPI/OAuth2 MCP gateway), then generalized. The vulnerability classes are language- and repo-agnostic; the specific mechanisms named as examples (nginx, MongoDB, JWKS, FastAPI routers) are illustrative, not a claim about this repository's stack. Read each entry for its **rule** and map it to the analogous mechanism in the code actually under review.

**How to use this document:**

- **Writing a feature:** read the patterns that touch your surface (new API route? read #2 and #5. Outbound fetch? read #1. New secret/env var? read #3). Apply the rule to whatever your code does.
- **Reviewing a change:** the [Review Checklist](#review-checklist) at the bottom maps each pattern to a yes/no question. The Security Engineer persona references this file.

Each pattern is written as: **the mistake** -> **the rule** -> **how to enforce it** -> **what to check**.

---

## 1. Server-side fetch of a user/config-controlled URL (SSRF)

**The mistake.** Taking a URL that a user, peer, or external config supplied (a proxy target, webhook, OAuth `token_url`, agent-card URL, model/artifact source) and fetching it with a plain HTTP client. An attacker points it at `169.254.169.254` (cloud metadata), a loopback admin port, or an internal host and either exfiltrates the secret being sent (an OAuth `client_secret`, refresh token, API key) or reaches an internal service. DNS rebinding defeats a naive "resolve then check" because the hostname re-resolves to a private IP between the check and the connect.

**The rule.** Every server-side fetch of a non-first-party URL is validated, fails closed, and is re-validated on every redirect hop:

- Only `http`/`https` schemes.
- Host must resolve **exclusively** to public IPs (block private, loopback, link-local, reserved, multicast, unspecified, and IPv4-mapped IPv6).
- The cloud metadata address is **never** allowlistable.
- Pin the connection to the validated IP so there is no rebinding window; re-validate redirects.
- Any resolution failure or ambiguity -> reject, never fall through to a permissive path.

**How to enforce it.** Route all outbound fetches of non-first-party URLs through a single shared guard/helper rather than a bare client, so the validation cannot be forgotten per-call site. Make the guard the only sanctioned way to build an outbound client for such URLs.

**What to check.** Does any new code build an HTTP call (`httpx`/`aiohttp`/`requests`/`urllib`) from a stored or request-supplied URL without validation? Is a secret being POSTed to that URL (making a guard rejection mandatory *before* the send)? Are redirects followed without re-validation (`follow_redirects=True` on an unguarded client)?

---

## 2. Broken access control and info disclosure on API endpoints

**The mistake.** An external endpoint that returns more than the caller is entitled to: internal/backend URLs in a list/search response, all records to an anonymous caller, or a mutation keyed on the URL path instead of the stored resource identity. A recurring root cause is a new API endpoint skipping the authorization check its older counterpart enforced.

**The rule.**

- Every read endpoint that can surface sensitive fields gates disclosure behind a fail-closed visibility check; unprivileged callers get the minimal public view only.
- Every endpoint resolves the resource, returns 404 if absent, and runs the per-user access check (403) **before** returning data or mutating.
- Authorization keys on the **stored** resource identity, never on the attacker-controlled URL path.
- New external API endpoints must mirror the authorization of any existing counterpart -- do not assume network placement protects them.

**How to enforce it.** Centralize the visibility/redaction decision in one fail-closed helper and reuse it across every read path. Apply the resolve -> 404 -> authorize -> act ordering uniformly.

**What to check.** New GET returning records -- does it strip sensitive fields for unprivileged callers? New mutation -- does it 404-then-403 on the resolved resource? Does the check use the stored identity or the path param? Is there an existing route doing this authorization that the new route forgot?

---

## 3. Weak, committed, or default-permissive secrets and config

**The mistake.** Shipping a usable default: a password like `admin` in `.env.example`, a hardcoded vendor default, an unauthenticated data store, dev-mode ports bound to `0.0.0.0`, a signing key with no length floor, TLS disabled by default. These make a fresh deploy insecure by default and often get copied into production.

**The rule.**

- Secrets have **no working default**. Required secrets fail closed (Compose `${VAR:?}`, Helm `{{ required }}`, Pydantic validation) rather than falling back to a placeholder.
- Ship weak-value **denylists** and reject known-bad values (`admin`, `changeme`, shipped placeholders) case-insensitively.
- Enforce minimum entropy/length on signing keys (>= 32 bytes).
- Bind non-front-door ports to loopback (`127.0.0.1`) by default; publicly expose only the intended front door. Require an explicit opt-in for LAN/public binding.
- Data-store auth on by default; TLS enabled, never `none`.

**How to enforce it.** A preflight/validation step (startup check, config-model validator, or CI gate) rejects weak, reserved, or placeholder values before the service runs. Keep the denylist and length floors in one place.

**What to check.** Any new secret/env var -- does it have a working default that would ship insecure? Is it added to the reserved-name lists and the weak-secret denylist? New service port -- is it loopback-bound by default? Any `verify=False`, TLS disabled, or `0.0.0.0` bind? (This repo's `CLAUDE.md` forbids `0.0.0.0` binds unless justified.)

---

## 4. Token trust boundaries

**The mistake.** Treating all tokens as interchangeable: reusing a user's token as an internal service token, trusting a client-supplied session id without binding it to the authenticated user, accepting an id-token without verifying its signature, or relaying the caller's inbound `Authorization`/cookie headers onward to an untrusted upstream (leaking the caller's credential to a third party).

**The rule.**

- Internal service-to-service tokens are minted separately from user tokens and are not accepted where a user token is expected (and vice versa).
- Bind session identity to the authenticated principal; never trust a client-supplied session id as authorization.
- Verify id-token signatures (signature, issuer, expiry, audience via JWKS) before trusting claims.
- Treat inbound client auth headers as **ingress-only**: strip them on egress. Never forward a caller's credential to an upstream the caller does not control.

**How to enforce it.** Keep a distinct minting path for internal tokens; strip inbound auth headers at the egress boundary by default; verify signatures in one shared validator.

**What to check.** Does new code forward `request.headers["Authorization"]` or the session cookie to an outbound call? Is a session id or user id read from the request body/params and used for authorization without re-checking the authenticated principal? Is a JWT decoded with `verify_signature=False` or without audience/issuer checks?

---

## 5. Missing CSRF on state-changing endpoints

**The mistake.** A new router with mutating (POST/PUT/PATCH/DELETE) endpoints that does not apply the CSRF check. When there is no global CSRF middleware, a router that forgets it leaves high-privilege mutations reachable cross-origin using an authenticated operator's session cookie.

**The rule.** Every mutating endpoint reachable with a session cookie applies the CSRF check. Non-browser (Bearer-token) clients and read-only GETs are unaffected. Missing/invalid token -> 403.

**How to enforce it.** Apply the CSRF dependency uniformly on every mutating route, matching the pattern used across existing routers; when a new router is added, it must adopt the same pattern.

**What to check.** Does every new POST/PUT/PATCH/DELETE carry the CSRF check? When a new router is added, does it match the CSRF pattern of the existing routers?

---

## 6. Injection through unescaped interpolation

**The mistake.** Building a downstream string (a config directive, a database query, an HTML page, a shell command, an href) by interpolating untrusted input without escaping. Concretely: a URL with config metacharacters breaking out of a directive; `{"$regex": f"^{path}:"}` letting a crafted path inject a regex; a `javascript:`/`data:` URL rendered as a link; a value interpolated into an HTML response.

**The rule.**

- Never interpolate untrusted input into a config/query/markup/shell context without a context-appropriate escape.
- Config generators: reject metacharacters (`\r`, `\n`, `;`, spaces, quotes, braces) in any value that lands in a directive.
- Regex from input: escape the interpolated fragment (e.g. `re.escape()`).
- Frontend: render untrusted hrefs only through a scheme allowlist that blocks `javascript:`/`data:`.
- HTML responses: escape interpolated values.
- Shell: use the list form of subprocess, never `shell=True`; pass user data as arguments, never interpolated into the command (per `CLAUDE.md`).

**How to enforce it.** Prefer parameterized queries and allowlist-validated identifiers; centralize escaping/validation in helpers rather than inline at each call site.

**What to check.** Any f-string or `+` that puts request/external data into a query, config, shell, or markup string? Any new href/redirect that renders a stored URL without a scheme allowlist? Any regex built from input without escaping?

---

## 7. Secret and PII leakage into logs and responses

**The mistake.** Logging raw request headers, a full user-context dict, decoded token/OIDC claims, or a token-bearing payload; returning a write-only secret in a read/list response; writing a freshly minted credential to stdout or a world-readable file.

**The rule.**

- Route header dumps, body/user-context dicts, and claim sets through a redaction layer before logging. Log claim **names** and masked identifiers, never claim values or tokens.
- Secrets are **write-only in the API**: accept on create/update, never echo on read/list (use a response schema that excludes them).
- Machine-minted credentials go to owner-only (`0600`) files, never stdout/logs.

**How to enforce it.** Provide redaction helpers (`redact_headers`, `redact_mapping`, masked identity summaries) and route all sensitive logging through them; use response schemas that omit secret fields.

**What to check.** Does new logging print headers, a full request/user dict, a token, or claim values? Does a response model include a secret field that should be write-only? Does a CLI/script command print a minted secret?

---

## 8. Dependency CVE exposure

**The mistake.** A dependency floor low enough to permit a version with a known CVE, or carrying an unused dependency that drags in CVE-bearing transitive deps, relying on the lockfile alone to mitigate (off-lock/manifest installs stay exposed).

**The rule.**

- Raise manifest floors above the fixed version, not just the lockfile -- off-lock installs must be safe too.
- Remove dependencies that are declared but never imported.
- Apply the floor consistently across the root and every sub-project manifest/lock.

**How to enforce it.** Pin floors in the manifest (not only the lock); add a guard test asserting removed/banned packages stay out of every manifest and lock.

**What to check.** Does a new dependency's floor sit above known CVE fixes? Is a newly added dependency actually imported? Was the change applied to all relevant manifests, not one?

---

## 9. LLM agent and tool-execution safety

**The mistake.** An LLM tool loop that executes any `tool_use` the model emits without a human gate (the model output is untrusted and steerable by prompt injection), and an agent HTTP endpoint that accepts messages based on network reachability alone (any process that can reach the port drives the agent).

**The rule** (see also root `CLAUDE.md`, subprocess and LLM-agent sections):

- Gate every mutating/destructive tool call behind mandatory human confirmation; classify read vs. mutate at execution; fail closed (deny) when no confirmation channel exists.
- Deny-by-default allowlist for any shell/exec tool; reject unknown executables and shell metacharacters; scrub the environment.
- Authenticate the agent endpoint with an inbound bearer JWT (signature/issuer/expiry/audience) on every request before the message reaches the model; bind loopback by default; auth fails closed if the key source is unconfigured.

**How to enforce it.** Put the confirmation gate and allowlist at the execution boundary, not in the system prompt (prompt text is not an enforcement control). Authenticate the endpoint before the model sees the request.

**What to check.** Does a new agent tool execute a mutation without a confirmation gate? Does a shell/exec tool run arbitrary executables? Does a new agent HTTP endpoint validate a JWT before invoking the model, or rely on being "internal"?

---

## 10. Authenticating with a shared or guessable key; timing oracles

**The mistake.** Comparing an API key or signature with `==` (timing oracle), hashing it without a per-deployment secret (rainbow-tableable), or signing with a key that is a public constant in the source (anyone with the repo forges valid payloads). Also: a rate-limit response that leaks whether a key was valid.

**The rule.**

- Hash authentication keys with a **required per-deployment pepper**; compare in constant time.
- Do not authenticate on a secret that ships in the source tree.
- Rate-limit responses must not distinguish valid from invalid credentials.

**How to enforce it.** Use `hmac.compare_digest` (or an equivalent constant-time compare) for all secret/token/signature comparisons; require a per-deployment pepper that fails closed when unset.

**What to check.** Any `==` on a secret/token/signature (use `hmac.compare_digest`)? Any key hashed without a per-deployment secret? Any auth based on a constant baked into the repo? Does an error/rate-limit path reveal credential validity?

---

## 11. Proxy body integrity

**The mistake.** Authorizing a request based on metadata (path/headers) but forwarding a body that was not the one inspected, or forwarding a body that could not be inspected -- letting a caller smuggle an unauthorized operation past the authorization check.

**The rule.** Re-authorize the **exact** forwarded body; fail closed if the body cannot be inspected. Strip internal capture headers before forwarding upstream.

**How to enforce it.** Authorize the same bytes that are forwarded; reject uninspectable or oversized bodies; strip internal `X-*` capture headers on egress.

**What to check.** Does the proxy path authorize the same bytes it forwards? Does it fail closed on an uninspectable/oversized body? Are internal capture headers stripped on egress?

---

## Repository-Specific Patterns

The patterns above are generic. The ones below are specific to **this** repository -- an LLM benchmarking and self-hosting harness with three paths: Amazon Bedrock (direct), a LiteLLM proxy (`benchmarks/`), and self-hosted vLLM on EC2 (`self-hosted/`). The primary security surface here is not a web API; it is **spawning coding agents and inference servers, over untrusted model output and cloned third-party repos, while handling API tokens.** Apply these alongside the generic catalog.

### R1. Spawning subprocesses (`claude`, `codex`, `git`, `vllm`)

**The mistake.** Building a subprocess call with `shell=True`, interpolating a dataset value or model name into a command string, omitting a timeout, or not handling `TimeoutExpired`/`CalledProcessError`. The harness shells out to `claude`, `codex`, and `git` (`benchmarks/scripts/run-swe-headless.py`, `codex_judge.py`).

**The rule** (mirrors `CLAUDE.md` "Subprocess").

- List form only, never `shell=True`.
- The executable is a hardcoded constant (`claude`, `codex`, `git`) -- never built from user/dataset input. Dataset/config values are passed as list arguments, not interpolated into the command.
- Always set a `timeout`; always handle `TimeoutExpired` and `CalledProcessError`.
- Every `# nosec B603/B607/B404` carries a justification (hardcoded command, list args, no shell) -- the existing sites do this; keep it.

**What to check.** Any new `subprocess.run`/`Popen` with `shell=True`? Any executable name built from a variable? A missing `timeout=`? A new `nosec` without a justification comment?

### R2. Running coding agents against untrusted cloned repos (prompt-injection surface)

**The mistake.** The benchmark clones third-party target repos and runs an autonomous coding agent (`claude -p`, `codex exec`, opencode) inside them. A cloned repo's file contents and issue text are **untrusted model input**; a malicious repo can prompt-inject the agent into exfiltrating tokens or running destructive commands. Granting the agent broad tool permissions (`--dangerously-skip-permissions`, an over-broad `--allowedTools`, or a permission mode that auto-approves writes/exec) on that untrusted content turns injection into code execution on the host.

**The rule.**

- Keep the agent's permission mode and `--allowedTools` as narrow as the benchmark needs; do not widen to skip-all-permissions to make a run pass.
- Treat cloned `repo/` content and issue/problem text as untrusted; never feed a secret into a prompt run over it.
- The `/swe` skill's benchmark-isolation rules (do not read sibling artifacts, do not use cross-session MCP) are a security boundary, not just methodology -- preserve them.
- Prefer running agents in a disposable/gitignored clone (`/tmp` or `swe-benchmark-data/*/repo/`), never against this repo's own working tree.

**What to check.** Does a change add `--dangerously-skip-permissions` or broaden `--allowedTools`/`permission_mode` in `run-swe-headless.py` or a launch script? Does any prompt built for a cloned repo include a token or secret? Does a new agent run point at a non-disposable directory?

### R3. Model-provider tokens and API keys (Bedrock, Mantle, HF, vLLM)

**The mistake.** Committing or logging a real credential: the 12h Mantle bearer token (`MANTLE_API_KEY` from the AWS Bedrock token generator), a Hugging Face token (`.hf_token`, needed for gated model weights), Bedrock/AWS credentials, or a vLLM `--api-key`. Also: printing the resolved token in a proxy/serve log, or checking a populated `.env` into git.

**The rule.**

- All provider credentials come from env vars or files referenced as `os.environ/VAR` (as `litellm-mantle.yaml` already does) -- never inline literals.
- The client-facing key may be a throwaway (`"local"`); the **real** upstream token stays server-side in the proxy and is never echoed to clients or logs.
- `.env`, `*.pem`, `*.key`, and `.hf_token` are gitignored and never read/printed by tooling (AGENTS.md). Provide `.env.example` with placeholders, never real values.
- A minted short-lived token is a secret: do not write it to a world-readable file or stdout.

**What to check.** Any hardcoded key/token/`aws_access_key` literal? Does a new log line print `MANTLE_API_KEY`, an HF token, or an `Authorization` header value? Is a real secret added to a committed file (`runner.yaml`, a config, an artifact)? Is `.hf_token`/`.env` newly read and echoed?

### R4. Loopback-by-default for every served port

**The mistake.** Binding the LiteLLM proxy, the vLLM server, or a tunnel to `0.0.0.0` by default, exposing an unauthenticated inference/proxy endpoint (which holds the real upstream token) to the LAN. Today all three default to `127.0.0.1` (`vllm-serve.sh --host 127.0.0.1`, `bedrock-mantle-proxy.sh DEFAULT_HOST=127.0.0.1`, tunnel over SSH) -- keep it that way.

**The rule** (mirrors `CLAUDE.md` "Server Binding").

- Default bind is `127.0.0.1`. Non-loopback exposure is an explicit, documented opt-in flag, never the default.
- Reach a remote server via the SSH tunnel (`tunnel.sh`), not by binding it publicly.
- If a port must be exposed, it needs authentication in front of it; an open inference endpoint is an abuse and cost risk.

**What to check.** Does a new or changed script/config default a bind to `0.0.0.0`? Does a change remove the loopback default or the SSH-tunnel path? Is a newly exposed port left unauthenticated?

### R5. Supply-chain trust in setup scripts (`curl | bash`, `pip/uv install`)

**The mistake.** Piping a remote installer straight into a shell from a mutable URL (`curl -LsSf https://astral.sh/uv/install.sh | sh`, `curl -fsSL https://opencode.ai/install | bash`) or installing packages without a pinned floor, on a build/EC2 host. A compromised or MITM'd URL runs arbitrary code as the setup user.

**The rule.**

- Fetch installers over HTTPS from the canonical vendor domain only; do not add new `curl | bash` from an unofficial mirror.
- Prefer pinned versions for anything installed into the serving environment; apply CVE floors (generic pattern #8) to `pyproject.toml`.
- Do not add a setup step that downloads and executes a script from a URL built from a variable.

**What to check.** Does a new setup step pipe a remote script to a shell? Is the URL a canonical vendor domain over HTTPS? Are newly installed packages pinned/floored?

### R6. Secret and PII leakage into benchmark artifacts, logs, and metrics

**The mistake.** The harness writes session transcripts, token-count metrics, and design artifacts under `benchmarks/swe-benchmark-data/`. Committing a captured `Authorization` header, an API token echoed in a session JSONL, or PII from a cloned repo into these artifacts leaks it into git history. Also: server logs under `self-hosted/vllm/logs/` capturing request bodies with tokens.

**The rule** (specializes generic pattern #7).

- Metrics and reports record counts/latency/model ids, never raw credentials or full request bodies.
- Session logs and `logs/` are gitignored; artifacts committed to the repo must be scrubbed of tokens and PII first.
- When logging a request/response for debugging, redact `Authorization`/`x-api-key` headers and token fields.

**What to check.** Does a new artifact/report/metric field include a raw token, header, or request body? Is a session-log path newly committed instead of gitignored? Does new logging print an API key or bearer token?

---

## Review Checklist

Fast pass for a change touching the relevant surface. Any "no" is a blocker until justified.

**Outbound requests**
- [ ] Every fetch of a stored/request-supplied URL goes through the SSRF guard (#1)
- [ ] Secrets are never POSTed to an unvalidated URL; redirects are re-validated (#1)

**API endpoints**
- [ ] New GET strips sensitive fields for unprivileged callers via a fail-closed visibility check (#2)
- [ ] Mutations 404-then-403 on the resolved resource; authorization keys on stored identity, not the path (#2)
- [ ] New external API route mirrors any existing counterpart's authorization (#2)
- [ ] Every mutating endpoint carries the CSRF check (#5)

**Secrets and config**
- [ ] No new secret ships with a working default; required secrets fail closed (#3)
- [ ] New secret/env var added to reserved-name lists + weak-secret denylist (#3)
- [ ] New ports loopback-bound by default; no `0.0.0.0`, `verify=False`, or TLS disabled (#3)
- [ ] Signing keys enforce a minimum length (#3)

**Tokens and identity**
- [ ] Inbound auth headers/cookies are stripped on egress, never forwarded upstream (#4)
- [ ] JWTs verified (signature/issuer/expiry/audience); no client-supplied session id trusted for authorization (#4)

**Injection**
- [ ] No untrusted input interpolated into config/query/HTML/href/shell without escaping (#6)
- [ ] Regex fragments use `re.escape`; frontend hrefs use a scheme allowlist; subprocess uses the list form (#6)

**Logging and responses**
- [ ] No headers/user-context/tokens/claim values logged; use the redaction layer (#7)
- [ ] Secret fields are write-only in response schemas; machine-minted secrets go to `0600` files (#7)

**Dependencies**
- [ ] New dependency floors sit above known CVE fixes across all manifests; unused deps removed (#8)

**Agents / proxy / key auth**
- [ ] Mutating agent tools gated behind confirmation; agent HTTP endpoints validate a JWT (#9)
- [ ] Secrets compared with `hmac.compare_digest`; keys peppered per-deployment; no source-constant auth (#10)
- [ ] Proxy re-authorizes the exact forwarded body and fails closed on an uninspectable one (#11)

**This repo (benchmark / self-hosting harness)**
- [ ] Subprocess calls use the list form with a hardcoded executable, a timeout, and `TimeoutExpired`/`CalledProcessError` handling; `nosec` comments justified (R1)
- [ ] No agent permission widening (`--dangerously-skip-permissions`, over-broad `--allowedTools`) over untrusted cloned repos; no secret in a prompt run against cloned content (R2)
- [ ] No hardcoded provider credential; real upstream token stays server-side, never logged; `.env`/`.hf_token`/`*.key` never read or echoed (R3)
- [ ] Served ports (proxy, vLLM, tunnel) default to `127.0.0.1`; non-loopback exposure is an explicit opt-in, never unauthenticated (R4)
- [ ] No new `curl | bash` from a non-canonical URL; installed packages pinned/floored (R5)
- [ ] No token, header, or full request body written into a committed artifact, report, metric, or non-gitignored log (R6)

---

*Maintained alongside `CLAUDE.md` (subprocess/SQL/secrets/server-binding sections). When a new class of security defect is found, add the pattern here so reviews catch the next instance.*
