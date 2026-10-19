#!/usr/bin/env python3
"""Generate the vended ``models.json`` the swe-router skill reads.

The skill runs in someone else's repository, inside whatever coding assistant
they use. It cannot import anything from this one. So the measurements it needs
are copied into ``vend/swe-router/models.json``, which is committed and served
raw, and this script is the only thing that writes it.

Three differences from the internal ``docs/metrics/pareto-frontier-*.json`` it
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
* **Scores are grouped by difficulty bucket.** The skill compares a task's
  required quality against each model's mean score over the benchmark tasks
  in the same bucket (easy, moderate, hard, very hard). The bucket of each
  benchmark task comes from the dataset's rubric scores file
  (``docs/metrics/swe-router-rubric-<dataset>-<judge>.json``), recomputed here
  from its six stored scores with the shipped ``routing-policy.json``, so a
  policy edit never leaves a stale bucket behind.

Usage:
    uv run scripts/build_vended_models.py
    uv run scripts/build_vended_models.py --check    # CI: fail if out of date
"""

from __future__ import annotations

import argparse
import json
import logging
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import swe_router_rubric as rubric

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
# The rubric pass whose buckets build the columns. One judge per dataset, and a
# strong one: on a cross-repository check the frontier judge agreed with itself
# across three repeats on 20 of 21 tasks, where a cheap judge managed 13. A
# bucket that moves between repeats moves the model the router picks.
RUBRIC_JUDGE = "claude-opus-5-5"
RUBRIC_SCORES_TEMPLATE = "swe-router-rubric-{dataset}-{judge}.json"
DEFAULT_RUBRIC_DIR = _REPO_ROOT / "docs" / "metrics"

# Bumped when the shape of models.json changes in a way a consumer would notice.
# Consumers pin this; the skill refuses a major it does not know. 1.1 replaced
# provenance.source_commit (a git sha) with source_sha256 (a content hash) and
# took measured_on from the run summaries instead of a commit date. 2.0 added
# score_by_difficulty and completion_by_difficulty, which the skill now selects
# on, and the rubric provenance behind them.
SCHEMA_VERSION = "2.0"
# provenance.source_sha256 length: enough to tell versions apart, short to read.
SOURCE_HASH_CHARS = 12

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


def _source_sha256(source: Path) -> str:
    """Return the first 12 hex digits of the SHA-256 of ``source``'s bytes.

    Identifies the exact version of the data from the file itself. An earlier
    version stamped the commit that last touched the file, which a squash merge
    replaces: the stamp made on the branch then no longer matched ``main``, and
    the committed-artifact check failed with every score still correct (after
    #184, #210 and #219). A content hash is the same on the branch and after the
    merge, and anyone can check it with ``sha256sum <file> | cut -c1-12``.

    Args:
        source: The frontier JSON.

    Returns:
        A 12-character lowercase hex string.
    """
    return hashlib.sha256(source.read_bytes()).hexdigest()[:SOURCE_HASH_CHARS]


def _latest_run_date(
    runs_dir: Path, harness: str, skill: str, dataset: str, models: set[str]
) -> str | None:
    """Return the latest ``run_date`` among the run summaries behind the frontier.

    That is when the newest measurement in the file was taken, which is what a
    consumer means by "measured on". It comes from committed data, so it does
    not move when a commit is rewritten, and it does not depend on anyone's
    timezone.

    Args:
        runs_dir: The swe-benchmark-data root.
        harness: Harness folder, e.g. "omp".
        skill: Skill folder, e.g. "swe3".
        dataset: Dataset scope folder.
        models: The model slugs the frontier JSON lists; other runs are ignored.

    Returns:
        An ISO date, or None when no matching summary records a ``run_date``.
    """
    dates = []
    pattern = f"*/{harness}/{skill}/{dataset}/run-summary.json"
    for path in runs_dir.glob(pattern):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("model_slug") in models and summary.get("run_date"):
            dates.append(summary["run_date"])
    if not dates:
        logger.warning(
            "no run_date in any %s summary for the frontier's models; "
            "measured_on will be null",
            pattern,
        )
        return None
    return max(dates)


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


def _per_level_stats(
    runs_dir: Path,
    harness: str,
    skill: str,
    dataset: str,
    level_of: Callable[[dict[str, Any]], str | None],
    levels: tuple[str, ...],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Group every model's task scores by a level and summarise each level.

    Args:
        runs_dir: The swe-benchmark-data root.
        harness: Harness folder, e.g. "omp".
        skill: Skill folder, e.g. "swe3".
        dataset: Dataset scope folder.
        level_of: Maps a run-summary task row to its level, or None to skip it.
        levels: Every level, in output order.

    Returns:
        Model slug -> {"score": {level: mean}, "completion": {level: "n/m"}}.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    pattern = f"*/{harness}/{skill}/{dataset}/run-summary.json"
    for path in sorted(runs_dir.glob(pattern)):
        summary = json.loads(path.read_text(encoding="utf-8"))
        model = summary.get("model_slug")
        scored: dict[str, list[float]] = {}
        attempted: dict[str, int] = {}
        for row in summary.get("tasks", []):
            level = level_of(row)
            if not level:
                continue
            attempted[level] = attempted.get(level, 0) + 1
            score = row.get("task_score")
            # A failed task scores 0 and is excluded, not averaged in -- the
            # completion counter is where that failure shows up instead.
            if score and not row.get("failed"):
                scored.setdefault(level, []).append(score)
        if not model or not attempted:
            continue
        out[model] = {
            "score": {
                level: round(sum(v) / len(v), 2)
                for level in levels
                if (v := scored.get(level))
            },
            "completion": {
                level: f"{len(scored.get(level, []))}/{attempted[level]}"
                for level in levels
                if level in attempted
            },
        }
    return out


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
    return _per_level_stats(
        runs_dir, harness, skill, dataset, lambda row: row.get("complexity"), TIERS
    )


def task_buckets(rubric_scores: Path, policy_path: Path) -> dict[str, str]:
    """Each benchmark task's difficulty bucket, from its stored rubric scores.

    The bucket is recomputed from the six scores with the current policy rather
    than read from the file's ``bucket`` field, so moving a bucket edge in
    ``routing-policy.json`` takes effect at the next build without a new pass.

    Args:
        rubric_scores: A rubric pass (``run-swe-router-headless.py --mode rubric``).
        policy_path: The routing policy that sets the bucket ranges.

    Returns:
        Task id -> bucket.

    Raises:
        SystemExit: If the file is not a rubric pass.
    """
    data = json.loads(rubric_scores.read_text(encoding="utf-8"))
    if data.get("mode") != "rubric":
        raise SystemExit(f"{rubric_scores} is not a rubric pass")
    policy = rubric.load_policy(policy_path)
    return {
        task_id: rubric.bucket(entry["scores"], policy)
        for task_id, entry in (data.get("tasks") or {}).items()
    }


def per_bucket_stats(
    runs_dir: Path, harness: str, skill: str, dataset: str, buckets: dict[str, str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return model -> {"score": bucket->mean, "completion": bucket->"n/m"}.

    The same arithmetic as ``per_tier_stats``, grouped by each task's rubric
    bucket instead of the tier written in the dataset.

    Raises:
        SystemExit: If a benchmark task has no bucket, which would drop its
            scores from every column without saying so.
    """

    def level_of(row: dict[str, Any]) -> str | None:
        task = row.get("task")
        if task not in buckets:
            raise SystemExit(
                f"task {task!r} has no rubric scores; re-run the rubric pass for "
                f"{dataset} (run-swe-router-headless.py --mode rubric)"
            )
        return buckets[task]

    return _per_level_stats(
        runs_dir, harness, skill, dataset, level_of, rubric.BUCKET_ORDER
    )


def _rubric_provenance(
    rubric_scores: Path | None, repo_root: Path
) -> dict[str, Any] | None:
    """Where the buckets came from, so a consumer can trace a column."""
    if rubric_scores is None:
        return None
    data = json.loads(rubric_scores.read_text(encoding="utf-8"))
    judged = data.get("judged_by") or {}
    return {
        "file": _relative_or_name(rubric_scores, repo_root),
        "sha256": _source_sha256(rubric_scores),
        "judge": judged.get("model"),
        "repeats": judged.get("repeats"),
        "buckets_recomputed_with": "vend/swe-router/routing-policy.json",
    }


def default_rubric_scores(dataset: str, rubric_dir: Path = DEFAULT_RUBRIC_DIR) -> Path:
    """The rubric pass a dataset's columns are built from."""
    return rubric_dir / RUBRIC_SCORES_TEMPLATE.format(
        dataset=dataset, judge=RUBRIC_JUDGE
    )


def build(
    source: Path,
    repo_root: Path,
    runs_dir: Path | None = None,
    rubric_scores: Path | None = None,
    policy_path: Path = rubric.DEFAULT_POLICY,
) -> dict[str, Any]:
    """Build the vended payload from a frontier JSON.

    Args:
        source: Path to a ``pareto-frontier-*.json``.
        repo_root: Repository root, used to record the source path.
        runs_dir: The swe-benchmark-data root; defaults to the committed one.
        rubric_scores: The dataset's rubric pass. None takes the default path
            for the frontier's dataset, and builds no bucket columns when that
            file does not exist.
        policy_path: The routing policy that sets the bucket ranges.

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

    runs_dir = runs_dir if runs_dir is not None else DEFAULT_RUNS_DIR
    by_tier = per_tier_stats(
        runs_dir,
        data.get("harness", ""),
        data.get("skill", ""),
        data.get("repo", ""),
    )
    if rubric_scores is None:
        candidate = default_rubric_scores(data.get("repo", ""))
        rubric_scores = candidate if candidate.is_file() else None
    if rubric_scores is None:
        logger.warning("no rubric scores for %s; no bucket columns", data.get("repo"))
        by_bucket: dict[str, dict[str, dict[str, Any]]] = {}
    else:
        by_bucket = per_bucket_stats(
            runs_dir,
            data.get("harness", ""),
            data.get("skill", ""),
            data.get("repo", ""),
            task_buckets(rubric_scores, policy_path),
        )
    measured_on = _latest_run_date(
        runs_dir,
        data.get("harness", ""),
        data.get("skill", ""),
        data.get("repo", ""),
        {m["model"] for m in models},
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
                # What the skill selects on: the mean score over the benchmark
                # tasks in each difficulty bucket, and how many it finished.
                "score_by_difficulty": by_bucket.get(name, {}).get("score", {}),
                "completion_by_difficulty": by_bucket.get(name, {}).get(
                    "completion", {}
                ),
                # The dataset's hand-set tiers, kept for the legacy --tier path.
                "score_by_complexity": by_tier.get(name, {}).get("score", {}),
                "completion_by_complexity": by_tier.get(name, {}).get("completion", {}),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "benchmarks/scripts/build_vended_models.py",
        "provenance": {
            "measured_on": measured_on,
            "source_file": _relative_or_name(source, repo_root),
            "source_sha256": _source_sha256(source),
            "harness": data.get("harness"),
            "skill": data.get("skill"),
            "dataset": data.get("repo"),
            "judge": JUDGE,
            "rubric_scores": _rubric_provenance(rubric_scores, repo_root),
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
                "per-bucket ranking. Select on score_by_difficulty at the task's "
                "bucket. on_hosting_frontier is computed within one hosting basis "
                "(Bedrock or self-hosted), which is the apples-to-apples "
                "comparison; on_combined_frontier mixes the two and is "
                "directional."
            ),
            "difficulty_buckets": (
                "easy / moderate / hard / very_hard: each benchmark task's "
                "difficulty level, from four rubric scores (hidden contract, "
                "silent invariant, coordinated surfaces, ambiguity) summed and "
                "bucketed by routing-policy.json. score_by_difficulty is a "
                "model's mean over the tasks in a bucket, failures excluded; "
                "completion_by_difficulty is where failures show. A bucket with "
                "no benchmark task has no entry."
            ),
            "complexity_tiers": (
                "trivial / low / medium / high, set by hand in the dataset from "
                "the scope of the change; score_by_complexity serves the legacy "
                "--tier path only. The hardest tasks measured are bounded single-repo "
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
        "--rubric-scores",
        type=Path,
        default=None,
        help="The dataset's rubric pass. Default: "
        f"docs/metrics/{RUBRIC_SCORES_TEMPLATE.format(dataset='<dataset>', judge=RUBRIC_JUDGE)}.",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit non-zero if the committed file is out of date.",
    )
    return p.parse_args()


def main() -> None:
    """Generate models.json, or verify the committed copy matches."""
    args = _parse_args()
    payload = build(args.source, _REPO_ROOT, rubric_scores=args.rubric_scores)
    if payload["provenance"]["rubric_scores"] is None:
        raise SystemExit(
            "no rubric scores for this dataset, so models.json would have no "
            "bucket columns. Run run-swe-router-headless.py --mode rubric, or "
            "pass --rubric-scores."
        )
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
