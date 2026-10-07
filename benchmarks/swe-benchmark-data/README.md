# SWE Benchmark Data

This directory holds the inputs and outputs of an LLM software-engineering benchmark. Each run takes a real-world problem inside a real repository and asks a specific model (driven by the `/swe` skill) to produce a GitHub issue spec, a low-level design, an expert review, and a testing plan. Multiple models can attempt the same problem so their artifacts can be compared side-by-side. **The skill stops at design and review - it does not implement the change.**

## Directory Layout

```
benchmarks/swe-benchmark-data/
├── README.md                       # This file
└── {repo-name}/
    ├── repo/                       # Cloned source (gitignored - cloned by contributor, never committed)
    ├── {problem-name}/
    │   ├── {model-name-A}/         # Artifacts produced by model A on this problem
    │   │   ├── github-issue.md
    │   │   ├── lld.md
    │   │   ├── review.md
    │   │   └── testing.md
    │   └── {model-name-B}/         # Artifacts produced by model B on the same problem
    │       └── ...
    └── {next-problem-name}/
        └── ...
```

The `repo/` checkout under each `{repo-name}/` is **not** stored in this repository. It is added to `.gitignore` so contributors clone their own copy at the right tag before invoking `/swe`. This avoids carrying large third-party trees and keeps the per-tag history pinned by the contributor, not by this repo.

## How to Set Up a Benchmark Repository Locally

Clone each target repository at the documented tag inside its `{repo-name}/` folder. Clone into a `repo/` subdirectory so artifacts and source never collide:

```bash
cd benchmarks/swe-benchmark-data/{repo-name}
git clone --branch <tag> --depth 1 https://github.com/<owner>/<name>.git repo
```

Use `--depth 1` to keep the checkout small. If you later need full history, run `git fetch --unshallow` from inside `repo/`.

---

## Benchmark Repositories

Each section below documents one target repository. To benchmark a model on one of its tasks, clone the repo at the listed tag and run `/swe` against the task description.

### 1. mcp-gateway-registry

| Field | Value |
|-------|-------|
| Source | https://github.com/agentic-community/mcp-gateway-registry |
| Tag | `1.24.4` |
| Local path | `benchmarks/swe-benchmark-data/mcp-gateway-registry/repo/` |
| Artifact path | `benchmarks/swe-benchmark-data/mcp-gateway-registry/{problem-name}/{model-name}/` |

#### Setup

```bash
cd benchmarks/swe-benchmark-data/mcp-gateway-registry
git clone --branch 1.24.4 --depth 1 https://github.com/agentic-community/mcp-gateway-registry.git repo
```

#### Tasks

The tasks below are run with multiple models via the `/swe` skill. For each `{model-name}`, the resulting artifacts land at `benchmarks/swe-benchmark-data/mcp-gateway-registry/{problem-name}/{model-name}/`.

| # | Problem name (folder) | Issue | Difficulty | Description |
|---|-----------------------|-------|-----------|-------------|
| 1 | `remove-faiss` | [#1285](https://github.com/agentic-community/mcp-gateway-registry/issues/1285) / [#452](https://github.com/agentic-community/mcp-gateway-registry/issues/452) | Medium | Remove FAISS from the codebase and documentation. FAISS is obsolete in this repo. Delete all FAISS imports, dependencies, configuration, and references in docs. Replace any remaining vector-search needs with the maintained DocumentDB hybrid search alternative already used elsewhere in the repo. |
| 2 | `remove-efs-from-terraform-aws-ecs` | [#1286](https://github.com/agentic-community/mcp-gateway-registry/issues/1286) | Medium | Remove EFS from `terraform/aws-ecs/`. EFS is obsolete in this deployment. Delete the EFS file system, mount targets, security groups, and any task-definition volume mounts that reference it. Update `variables.tf`, `terraform.tfvars.example`, and module wiring. Verify `terraform validate` and `terraform plan` still succeed. |
| 3 | `ssrf-hardening-outbound-url-validation` | [#1282](https://github.com/agentic-community/mcp-gateway-registry/issues/1282) | Medium | SSRF hardening: validate outbound URLs on agent card fetch (health check + pull-card endpoints). The model must identify vulnerable endpoints that make outbound HTTP requests based on user-supplied URLs, propose URL validation (deny internal/private IPs, allowlists), and design input sanitization to prevent SSRF attacks. |
| 4 | `migrate-ecs-env-vars-to-secrets-manager` | [#1134](https://github.com/agentic-community/mcp-gateway-registry/issues/1134) | High | Migrate sensitive ECS environment variables to AWS Secrets Manager. Identify which env vars in the ECS task definitions contain secrets (DB passwords, API keys, OAuth client secrets, admin passwords), create Secrets Manager resources in Terraform, update ECS task definitions to pull from Secrets Manager via the `secrets` block instead of passing plaintext via `environment`, and update the IAM task execution role to allow reading those secrets. |
| 5 | `replace-keycloak-db-password-with-rds-iam` | [#1303](https://github.com/agentic-community/mcp-gateway-registry/issues/1303) | High | Replace the Keycloak database password with RDS IAM authentication. The repo uses an Aurora MySQL cluster for Keycloak; remove static DB credentials from Terraform and ECS config, enable IAM database authentication on the Aurora MySQL cluster, update the Keycloak ECS task to generate short-lived IAM auth tokens via `rds:GenerateDBAuthToken`, and update IAM roles/policies accordingly. |

#### How to Run a Task with `/swe`

```
/swe

# When prompted by the skill:
# - repo-name   : mcp-gateway-registry
# - problem-name: remove-faiss              (use the kebab-case name from the table)
# - model-name  : claude-opus-4-8           (or whichever model is being benchmarked)
```

The skill will create `benchmarks/swe-benchmark-data/mcp-gateway-registry/remove-faiss/claude-opus-4-8/` and populate it with `github-issue.md`, `lld.md`, `review.md`, and `testing.md`. Re-run with a different `model-name` to add a sibling folder for direct comparison. The skill does not implement the change - that is a separate step the user can take with the design package as input.

#### Scoring

Each of the 4 artifacts is scored 0–100 by an independent ChatGPT session
(cross-lineage judge). Within each artifact, the judge applies the same
4-criterion rubric — each criterion worth 25 points, summing to 100 per
artifact:

| Criterion | 0–25 each | What the judge evaluates |
|-----------|-----------|--------------------------|
| **Completeness** | 25 | Did the artifact identify all affected files, deps, and components? |
| **Correctness** | 25 | Are the proposed changes technically right? Would the design actually work? |
| **Specificity** | 25 | Concrete file paths, code snippets, resource names — not hand-waving? |
| **Risk awareness** | 25 | Rollback plan, backwards-compat, deployment cutover, edge cases? |

**Artifact total = sum of 4 criteria (0–100).**
**Task score = mean of the 4 artifact totals.**

Results are reported in a 5×6 matrix (rows = tasks, columns = models). Per-cell
JSON with criterion breakdowns and judge notes lives at
`{task}/{model}/judge-gpt.json`. The aggregated matrix and synthesis are written
to `{repo}/JUDGE_RESULTS.md` once a run has been scored; this directory ships
empty, so that file appears after your first scored run.
