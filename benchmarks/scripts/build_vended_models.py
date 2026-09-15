#!/usr/bin/env python3
"""Generate the vended ``models.json`` the swe-router skill reads.

The skill runs in someone else's repository, inside whatever coding assistant
they use. It cannot import anything from this one. So the measurements it needs
are copied into ``vend/swe-router/models.json``, which is committed and served
raw, and this script is the only thing that writes it.

Two differences from the internal ``docs/metrics/pareto-frontier-*.json`` it
reads:

* **Every measured model is included, not just the frontier.** The skill ranks
  over the models the user's assistant actually offers, at the complexity tier
  the task sits in, so it needs the full set: a published frontier is derived
  from whole-dataset means and may contain none of the models a given user has.
  Frontier membership still travels, as ``on_combined_frontier`` and
  ``on_hosting_frontier``, because "nothing beats this on both axes" is useful
  context to show a reader. It is not a selection key, and the payload says so.
* **Provenance travels with the data.** The internal file assumes a reader who
  knows this repo. A vended file has no such reader, so it carries the schema
  version, when it was measured, the harness, the skill, the dataset and the
  judge. A consumer holding a stale copy can see that it is stale.

Usage:
    uv run scripts/build_vended_models.py
    uv run scripts/build_vended_models.py --check    # CI: fail if out of date
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess  # nosec B404 - used with list args, no shell, hardcoded 'git'
from datetime import date
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
DEFAULT_SOURCE = _REPO_ROOT / "docs" / "metrics" / "pareto-frontier-omp-swe3.json"
# Per-task scores live in the committed run summaries, not the frontier JSON,
# which only carries whole-dataset means. The vended file needs both: a mean
# hides that qwen3.8-27b scores 80.9 on low-complexity work and 57.2 on high.
DEFAULT_RUNS_DIR = _REPO_ROOT / "benchmarks" / "swe-benchmark-data"
TIERS = ("trivial", "low", "medium", "high")
DEFAULT_OUT = _REPO_ROOT / "vend" / "swe-router" / "models.json"

# Bumped when the shape of models.json changes in a way a consumer would notice.
# Consumers pin this; the skill refuses a major it does not know.
SCHEMA_VERSION = "1.0"

# The judge is not recorded in the frontier JSON, so it is stated here. Keep in
# step with codex_judge.py's default (JUDGE_MODEL overrides it per run).
JUDGE = {
    "model": "openai.gpt-5.6-sol",
    "reasoning_effort": "high",
    "repo_grounded": True,
}

# What each hosting label means for the dollar figures, in the consumer's terms.
# Without this a reader ranks a metered bill against a GPU-hour derivation.
COST_BASIS = {
    "Bedrock": ("Metered Amazon Bedrock token pricing -- what the invoice says."),
    "self-hosted": (
        "The server's hourly price divided by throughput measured at a stated "
        "concurrency -- a shared server under load, which is how a platform "
        "team runs one for a group of developers. A cost per task on the same "
        "footing as a metered bill."
    ),
}


def _git_commit(source: Path) -> str | None:
    """Return the short sha of the commit that last changed ``source``.

    Deliberately not HEAD. HEAD moves with every commit to the repository, so
    stamping it would make the generated file differ after any unrelated change
    and turn the --check guard into noise. The commit that last touched the
    source is what actually identifies this version of the data.

    Args:
        source: The frontier JSON.

    Returns:
        The abbreviated commit, or None outside a git checkout or for a file
        that has never been committed.
    """
    try:
        out = subprocess.run(  # nosec B603 B607 - hardcoded 'git', list args, no shell
            [
                "git",
                "-C",
                str(source.parent),
                "log",
                "-1",
                "--format=%h",
                "--",
                source.name,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _measured_on(source: Path) -> str:
    """Return the date the source frontier was last modified, ISO format.

    Uses the file's last git commit date when available, so regenerating without
    changing the data does not advance the measurement date. Falls back to the
    filesystem mtime.

    Args:
        source: The frontier JSON.

    Returns:
        An ISO date string.
    """
    try:
        out = subprocess.run(  # nosec B603 B607 - hardcoded 'git', list args, no shell
            [
                "git",
                "-C",
                str(source.parent),
                "log",
                "-1",
                "--format=%cs",
                "--",
                source.name,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return date.fromtimestamp(source.stat().st_mtime).isoformat()


def _relative_or_name(source: Path, repo_root: Path) -> str:
    """Return the source path relative to the repo, or its bare name.

    A source outside the repository has no meaningful relative path, and
    recording an absolute one would leak a local directory layout into a file
    published to strangers.

    Args:
        source: The frontier JSON.
        repo_root: Repository root.

    Returns:
        A repo-relative path, or the filename when the source is outside it.
    """
    try:
        return str(source.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return source.name


def per_tier_stats(
    runs_dir: Path, harness: str, skill: str, dataset: str
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return model -> {"score": tier->mean, "completion": tier->"n/m"}.

    Two numbers, because difficulty affects two different things.

    **Score.** The whole-dataset mean is the wrong number to compare a quality
    floor against when the task is hard. Models do not degrade equally: on this
    data ``claude-opus-5`` loses 3.1 points on high-complexity tasks while
    ``qwen3.8-27b`` loses 17.6. A model clearing a floor of 70 on its overall
    mean can sit at 57 on the tier that actually matters.

    **Completion.** A model that fails a task outright is worse than one that
    scores badly, and failures concentrate somewhere: ``devstral-2-123b``
    finished 3 of 6 medium tasks. A mean over the ones it survived says nothing
    about that.

    Failed tasks are excluded from the mean rather than averaged in as zero,
    matching how the frontier JSON computes ``mean_score``. Including them
    would make the tier means disagree with the overall score they sit beside.

    Args:
        runs_dir: The swe-benchmark-data root.
        harness: Harness folder, e.g. "omp".
        skill: Skill folder, e.g. "swe3".
        dataset: Dataset scope folder.

    Returns:
        Model slug -> {"score": {tier: mean}, "completion": {tier: "n/m"}}.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    pattern = f"*/{harness}/{skill}/{dataset}/run-summary.json"
    for path in sorted(runs_dir.glob(pattern)):
        summary = json.loads(path.read_text(encoding="utf-8"))
        model = summary.get("model_slug")
        scored: dict[str, list[float]] = {}
        attempted: dict[str, int] = {}
        for row in summary.get("tasks", []):
            tier = row.get("complexity")
            if not tier:
                continue
            attempted[tier] = attempted.get(tier, 0) + 1
            score = row.get("task_score")
            # A failed task scores 0 and is excluded, not averaged in -- the
            # completion counter is where that failure shows up instead.
            if score and not row.get("failed"):
                scored.setdefault(tier, []).append(score)
        if not model or not attempted:
            continue
        out[model] = {
            "score": {
                tier: round(sum(v) / len(v), 2)
                for tier in TIERS
                if (v := scored.get(tier))
            },
            "completion": {
                tier: f"{len(scored.get(tier, []))}/{attempted[tier]}"
                for tier in TIERS
                if tier in attempted
            },
        }
    return out


def build(
    source: Path, repo_root: Path, runs_dir: Path | None = None
) -> dict[str, Any]:
    """Build the vended payload from a frontier JSON.

    Args:
        source: Path to a ``pareto-frontier-*.json``.
        repo_root: Repository root, used for the provenance commit.

    Returns:
        The payload to write as models.json.

    Raises:
        SystemExit: If the source is missing or carries no models.
    """
    if not source.is_file():
        raise SystemExit(f"no frontier JSON at {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    models = data.get("all_models") or []
    if not models:
        raise SystemExit(f"{source} carries no all_models list")

    combined = {
        m["model"] for m in data.get("combined_frontier_cross_hosting_directional", [])
    }
    bedrock = {m["model"] for m in data.get("bedrock_frontier", [])}
    self_hosted = {m["model"] for m in data.get("self_hosted_frontier", [])}

    by_tier = per_tier_stats(
        runs_dir if runs_dir is not None else DEFAULT_RUNS_DIR,
        data.get("harness", ""),
        data.get("skill", ""),
        data.get("repo", ""),
    )
    out_models = []
    for m in sorted(models, key=lambda x: -x["mean_score"]):
        name = m["model"]
        out_models.append(
            {
                "model": name,
                "score": m["mean_score"],
                "cost_per_task_usd": m["mean_cost_per_task"],
                "hosting": m.get("hosting"),
                # A mean over fewer tasks deserves to be visible rather than
                # averaged into silence: three of the current models did not
                # finish every task.
                "tasks_completed": m.get("n_scored"),
                "tasks_total": m.get("n_tasks"),
                "excluded_tasks": m.get("excluded_tasks") or [],
                # Compare a quality floor against the tier the task actually
                # sits in. Models degrade at very different rates, and some
                # stop finishing hard tasks at all.
                # Frontier membership as published, so a consumer can see which
                # models are non-dominated across the whole dataset. Computed
                # from overall means, so it can disagree with the per-tier
                # ranking below: qwen3.8-27b is on the combined frontier and
                # still trails claude-sonnet-5 on high-complexity work. Report
                # it as context; select on score_by_complexity.
                "on_combined_frontier": name in combined,
                "on_hosting_frontier": name in bedrock or name in self_hosted,
                "score_by_complexity": by_tier.get(name, {}).get("score", {}),
                "completion_by_complexity": by_tier.get(name, {}).get("completion", {}),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "benchmarks/scripts/build_vended_models.py",
        "provenance": {
            "measured_on": _measured_on(source),
            "source_file": _relative_or_name(source, repo_root),
            "source_commit": _git_commit(source),
            "harness": data.get("harness"),
            "skill": data.get("skill"),
            "dataset": data.get("repo"),
            "judge": JUDGE,
            "repository": "https://github.com/aws-samples/sample-agentic-coding-harness-benchmarks",
        },
        "measurement_basis": {
            "what_the_score_is": (
                "Mean 0-100 score over the dataset's tasks. Each task is judged on "
                "six artifacts: a GitHub issue spec, a low-level design, an expert "
                "review, a testing plan, a patch, and an implementation summary."
            ),
            "single_repository_warning": (
                "Every task comes from one repository -- a Python/FastAPI and "
                "React service with nginx, Terraform, Helm and bash around it. "
                "Rankings are more portable than absolute scores, but applying "
                "these to a very different codebase is extrapolation."
            ),
            "runs_per_task": 1,
            "frontier_flags": (
                "on_combined_frontier and on_hosting_frontier mark models that "
                "nothing else beats on both score and cost across the WHOLE "
                "dataset. They are context for a reader, not a selection key: "
                "they come from overall means, so they can disagree with the "
                "per-tier ranking. Select on score_by_complexity at the task's "
                "tier. on_hosting_frontier is computed within one hosting basis "
                "(Bedrock or self-hosted), which is the apples-to-apples "
                "comparison; on_combined_frontier mixes the two and is "
                "directional."
            ),
            "complexity_tiers": (
                "trivial / low / medium / high, assigned by the scope of the "
                "change. The hardest tasks measured are bounded single-repo "
                "changes -- a rate-limiting subsystem, server-side OAuth token "
                "storage. Nothing here approaches a rewrite or a language port, "
                "so a task far beyond that range is outside what these numbers "
                "cover."
            ),
            "cost_basis": COST_BASIS,
        },
        "models": out_models,
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Generate the vended models.json.")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit non-zero if the committed file is out of date.",
    )
    return p.parse_args()


def main() -> None:
    """Generate models.json, or verify the committed copy matches."""
    args = _parse_args()
    payload = build(args.source, _REPO_ROOT)
    text = json.dumps(payload, indent=2) + "\n"

    if args.check:
        if not args.out.is_file():
            raise SystemExit(f"{args.out} does not exist; run without --check")
        if args.out.read_text(encoding="utf-8") != text:
            raise SystemExit(
                f"{args.out} is out of date. Regenerate with:\n"
                f"  uv run scripts/build_vended_models.py"
            )
        logger.info("%s is up to date (%d models)", args.out, len(payload["models"]))
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    logger.info(
        "wrote %s -- %d models, measured %s on %s/%s/%s",
        args.out,
        len(payload["models"]),
        payload["provenance"]["measured_on"],
        payload["provenance"]["harness"],
        payload["provenance"]["skill"],
        payload["provenance"]["dataset"],
    )


if __name__ == "__main__":
    main()
