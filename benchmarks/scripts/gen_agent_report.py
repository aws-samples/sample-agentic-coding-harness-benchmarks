#!/usr/bin/env python3
"""Generate a per-agent (per-harness) results document from committed summaries.

For one coding agent (``claude-code``, ``pi``, ...) this walks every model's
committed ``run-summary.json`` under that harness and renders a single Markdown
file with:

  * a per-model results table (mean score, completion, tokens, wall-clock,
    hardware-derived cost), and
  * the two headline charts for that harness (cost-vs-quality Pareto and the
    quality radar), which the chart scripts render with harness-suffixed names.

The doc is regenerated from data, so it never drifts from the run-summaries.
Charts are NOT rendered here -- run ``plot_cost_quality.py --harness <h> --skill
<s>`` and ``plot_quality_radar.py --harness <h> --skill <s>`` first; this only
embeds them.

Usage:
    uv run scripts/gen_agent_report.py --harness pi --skill swe3
    uv run scripts/gen_agent_report.py --harness claude-code --skill swe2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from token_accounting import cache_partition_for_agent, compute_total_tokens_processed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
_BENCHMARKS_DIR = _SCRIPTS_DIR.parent
_REPO_ROOT = _BENCHMARKS_DIR.parent
DEFAULT_DATA_DIR = _BENCHMARKS_DIR / "swe-benchmark-data"
DEFAULT_OUT_DIR = _REPO_ROOT / "docs"
# The dataset scope a report covers unless --repo says otherwise.
DEFAULT_REPO = "mcp-gateway-registry"
RUN_SUMMARY_FILENAME = "run-summary.json"

# Throughput sweeps: each model's performance-summary.json carries a hardware-
# derived, per-token cost (blended lens) at its REAL instance rate (g6e vs p5en),
# precomputed by clients/build_performance_summary.py. A self-hosted run's cost is
# that per-token rate x the tokens it processed -- NOT a flat wall-clock x $/hr,
# which both charges idle agent-thinking time and applies one instance's price to
# models served on another. See docs/cost-per-task-methodology.md.
_THROUGHPUT_DIR = (
    _REPO_ROOT / "self-hosted" / "vllm" / "benchmark-output" / "throughput"
)

# Human labels for the harness slug used in the doc title and prose.
HARNESS_LABELS = {
    "claude-code": "Claude Code",
    "pi": "pi",
    "opencode": "opencode",
    "kiro-cli": "kiro-cli",
    # omp's own name is oh-my-pi; "omp" is the binary. Spell both out so the
    # generated page names the project a reader can go and find.
    "omp": "oh-my-pi (omp)",
}

# Harnesses with an install/configuration page in docs/. Linked from the report
# so a reader who wants to reproduce the run knows where the setup lives.
HARNESS_SETUP_DOCS = {
    "omp": ("omp setup", "omp-setup.md"),
    "kiro-cli": ("kiro-cli setup", "kiro-cli-setup.md"),
}

# Short per-harness code that (with the skill) suffixes chart filenames
# (cost-quality-cc-swe2.png, quality-radar-pi-swe3.png). Must match the codes the
# plot scripts use so the doc links resolve to the files they write.
HARNESS_CODES = {"claude-code": "cc", "pi": "pi", "opencode": "oc", "kiro-cli": "kiro"}


def _read_json(path: Path) -> dict[str, Any] | None:
    """Return the parsed JSON object at ``path``, or None if absent/invalid."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _blended_rate(model_slug: str) -> tuple[float, str] | None:
    """Return (blended $/token, instance_type) for a self-hosted model, or None.

    Uses the CHEAPEST blended cost-per-token across the model's throughput-sweep
    concurrency levels -- the best sustainable per-token cost on its benchmarked
    instance (a saturated box, not one idle request). The rate is precomputed at
    the model's true instance price (g6e vs p5en) by build_performance_summary.py.
    This mirrors plot_cost_quality._blended_cost_per_token so the doc's cost column
    and the cost-quality chart agree.

    Args:
        model_slug: The model's clean slug (e.g. "glm-5.2").

    Returns:
        ``(min blended_cost_per_token_usd, instance_type)``, or None when no
        throughput summary exists for the model.
    """
    summary = _read_json(_THROUGHPUT_DIR / model_slug / "performance-summary.json")
    if summary is None:
        return None
    rates = [
        r["blended_cost_per_token_usd"]
        for r in summary.get("levels", [])
        if isinstance(r.get("blended_cost_per_token_usd"), (int, float))
    ]
    if not rates:
        return None
    return (min(rates), summary.get("instance_type") or "self-hosted")


def _run_totals(summary: dict[str, Any]) -> dict[str, Any]:
    """Sum a run's per-task tokens, wall-clock, and metered cost into totals.

    ``total_tokens`` is the total tokens processed once each. It is the SUM OF
    THE PER-TASK ``total_tokens`` that ``summarize_run.py`` already wrote via
    ``compute_total_tokens_processed`` (issue #136) -- NOT a fresh classification
    of the summed fields.

    That distinction is the whole point. The helper decides per task whether the
    cache fields are a PARTITION of ``input_tokens`` (self-hosted vLLM: ``input``
    already contains the cached prompt, so ``total = input + output``) or ADDITIVE
    (Bedrock prompt caching: a task reports ``input: 2`` while processing ~180K
    cached tokens, so the cache must be added). Classifying the AGGREGATE instead
    lets a single anomalous task flip the verdict for every other task in the run.

    That is not hypothetical. A task's ``vllm_prometheus`` block is a window delta
    of SERVER-WIDE counters, accurate only when the run was the sole traffic on the
    server; when a window catches traffic that is not its own, that task's cache
    sum explodes (one glm-5.3 task reported ``input`` 479,697 against a cache sum
    of 47,795,047 -- a 99.6x ratio). Twenty of that run's twenty-one tasks matched
    the partition signature to within 0.02%, but the outlier dragged the summed
    ratio to 1.24, outside the 5% band, so the aggregate was classified ADDITIVE
    and the cache was re-added to all 21 tasks: 442,295,665 tokens reported against
    245,606,604 actually processed, inflating that row's cost 1.80x ($258.93 vs
    $143.78). Six of eleven self-hosted models were overstated 1.5-1.9x this way.

    Summing the per-task totals is also what ``plot_cost_quality.py`` and
    ``plot_cost_accuracy_bubble.py`` already do (they call the helper inside a
    per-task loop), which is why the charts and the Pareto-frontier JSON were
    correct while this table was not -- the report contradicted its own chart.

    Args:
        summary: One model's run-summary dict.

    Returns:
        Dict with total input/output tokens, total tokens processed, total
        latency seconds, and total metered cost (sum of per-task
        ``total_cost_usd``; None on the self-hosted path with no per-token price).
    """
    tasks = summary.get("tasks", []) or []
    tin = sum((t.get("input_tokens") or 0) for t in tasks)
    tout = sum((t.get("output_tokens") or 0) for t in tasks)
    tcr = sum((t.get("cache_read_tokens") or 0) for t in tasks)
    tcw = sum(
        (t.get("cache_write_tokens") or t.get("cache_creation_tokens") or 0)
        for t in tasks
    )
    tsec = sum((t.get("latency_seconds") or 0) for t in tasks)
    costs = [t.get("total_cost_usd") for t in tasks if t.get("total_cost_usd")]
    metered_cost = sum(costs) if costs else None
    context = (
        f"gen_agent_report:{summary.get('model_slug')}/"
        f"{summary.get('agent')}/{summary.get('skill')}"
    )
    # Prefer the per-task totals the summarizer already classified one task at a
    # time. Fall back to classifying the aggregate only for a legacy summary whose
    # tasks predate the per-task field, where there is nothing better to use.
    per_task = [t.get("total_tokens") for t in tasks]
    if per_task and all(isinstance(v, int) for v in per_task):
        total_tokens = sum(per_task)
    else:
        logger.warning(
            "%s: %d/%d tasks lack a per-task total_tokens; falling back to "
            "classifying the aggregate, which one anomalous task can skew",
            context,
            sum(1 for v in per_task if not isinstance(v, int)),
            len(per_task),
        )
        total_tokens = compute_total_tokens_processed(
            tin,
            tout,
            tcr,
            tcw,
            context=context,
            cache_partition=cache_partition_for_agent(summary.get("agent")),
        )
    return {
        "input_tokens": tin,
        "output_tokens": tout,
        "cache_read_tokens": tcr,
        "cache_write_tokens": tcw,
        "total_tokens": total_tokens,
        "latency_seconds": tsec,
        "metered_cost": metered_cost,
    }


def _collect(
    data_dir: Path, harness: str, skill: str, repo: str
) -> list[dict[str, Any]]:
    """Return one row per model that has a run-summary under this harness+skill.

    Reads ``<data-dir>/<model>/<harness>/<skill>/<repo>/run-summary.json``. Rows
    are sorted by mean score (a None mean -- a full harness collapse -- sorts last).
    """
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        summary = _read_json(model_dir / harness / skill / repo / RUN_SUMMARY_FILENAME)
        if summary is None:
            continue
        totals = _run_totals(summary)
        rows.append(
            {
                # Prefer the clean slug (e.g. "claude-opus-5") over the raw id
                # ("us.anthropic.claude-opus-5[1m]") for a readable table.
                "model": summary.get("model_slug") or model_dir.name,
                "provider": summary.get("provider"),
                "mean": summary.get("mean_task_score_excl_failed"),
                "num_scored": summary.get("num_scored"),
                "num_tasks": summary.get("num_tasks"),
                "failed_tasks": summary.get("failed_tasks") or [],
                "run_date": summary.get("run_date"),
                **totals,
            }
        )
    rows.sort(key=lambda r: (r["mean"] is None, -(r["mean"] or 0.0)))
    return rows


def _row_cost(row: dict[str, Any]) -> tuple[str, str]:
    """Return (cost string, basis label) for a model row.

    Two cost bases, never mixed on one number:
      * **metered (Bedrock)** -- a hosted API reports a real per-token bill; use
        the summed ``total_cost_usd``.
      * **hardware-derived (throughput)** -- a self-hosted model has no per-token
        price, so cost is the model's blended cost-per-token (measured by the
        p5en.48xlarge throughput sweep at peak concurrency -- one basis for the
        whole fleet, whatever box a model was served on) times the
        tokens this run processed: ``blended_$/token x total_tokens``. This
        replaces the old ``$/hr x wall-clock`` estimate, which charged idle
        agent-thinking time and applied one instance's price to every model.

    Args:
        row: A collected model row (provider, metered_cost, model, total_tokens).

    Returns:
        A ``(cost, basis)`` pair, e.g. ``("$0.63", "metered (Bedrock)")``.
    """
    if row.get("provider") == "bedrock":
        cost = row.get("metered_cost")
        return (f"${cost:.2f}" if cost else "--", "metered (Bedrock)")
    if row.get("provider") == "kiro":
        # kiro-cli reports no tokens; its cost is credits x $/credit, already
        # summed into total_cost_usd. A third basis -- not GPU-derived.
        cost = row.get("metered_cost")
        return (f"${cost:.2f}" if cost else "--", "Kiro credits ($0.04/credit)")
    # Self-hosted: price the tokens processed at the throughput-derived blended rate.
    rate = _blended_rate(row.get("model", ""))
    total_tokens = row.get("total_tokens") or 0
    if rate is None or not total_tokens:
        return ("--", "hardware-derived")
    cost_per_token, instance = rate
    return (f"${cost_per_token * total_tokens:.2f}", f"hardware-derived ({instance})")


def _render(
    rows: list[dict[str, Any]],
    *,
    harness: str,
    skill: str,
    repo: str,
    out_dir: Path,
) -> str:
    """Render the per-agent Markdown doc from the collected rows."""
    label = HARNESS_LABELS.get(harness, harness)
    # Charts live in docs/images, suffixed by the harness code (cc, pi, ...) and
    # skill (swe2, swe3) so each agent+skill's charts are self-identifying -- must
    # match the names the plot scripts write. Link relative to the doc's out_dir.
    code = HARNESS_CODES.get(harness, harness)
    img = (out_dir / "images").resolve()
    cq = img / f"cost-quality-{code}-{skill}.png"
    radar = img / f"quality-radar-{code}-{skill}.png"
    bubble = img / f"cost-accuracy-bubble-{code}-{skill}.png"

    def _rel(p: Path) -> str:
        try:
            return p.relative_to(out_dir.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    lines = [
        f"# Results: {label} harness ({skill})",
        "",
        f"Benchmark results for every model run under the **{label}** coding agent "
        f"with the **{skill}** skill on `{repo}`, generated from the committed "
        "`run-summary.json` files. Regenerate with `uv run "
        f"scripts/gen_agent_report.py --harness {harness} --skill {skill}"
        # The doc path carries no repo, so a non-default dataset must be named
        # here or the printed command silently regenerates a different report.
        f"{'' if repo == DEFAULT_REPO else f' --repo {repo}'}`. "
        + (
            f"See [{HARNESS_SETUP_DOCS[harness][0]}]({HARNESS_SETUP_DOCS[harness][1]}) "
            "for install and configuration. "
            if harness in HARNESS_SETUP_DOCS
            else ""
        )
        + "Companion to the cross-harness comparison "
        f"[agentic-coding-swe-comparison-{skill}.md](agentic-coding-swe-comparison-{skill}.md).",
        "",
        "## Results by model",
        "",
        "| Model | Mean score | Completed | Input | Output | Cache read | Cache write | Tokens processed† | Wall-clock | Run cost | Cost basis* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    any_hardware = False
    any_metered = False
    any_kiro = False
    for r in rows:
        mean = "-- (0 scored)" if r["mean"] is None else f"{r['mean']:.2f}"
        completed = f"{r['num_scored']}/{r['num_tasks']}"
        tin = f"{r.get('input_tokens', 0):,}"
        tout = f"{r.get('output_tokens', 0):,}"
        tcr = f"{r.get('cache_read_tokens', 0):,}"
        tcw = f"{r.get('cache_write_tokens', 0):,}"
        tok = f"{r.get('total_tokens', r['input_tokens'] + r['output_tokens']):,}"
        mins = (r["latency_seconds"] or 0) / 60.0
        wall = f"{mins:.1f}m" if mins else "--"
        cost, basis = _row_cost(r)
        any_hardware = any_hardware or basis.startswith("hardware-derived")
        any_metered = any_metered or basis.startswith("metered")
        any_kiro = any_kiro or basis.startswith("Kiro credits")
        lines.append(
            f"| {r['model']} | {mean} | {completed} | {tin} | {tout} | {tcr} | {tcw} "
            f"| {tok} | {wall} | {cost} | {basis} |"
        )
    # The cost column mixes two bases that are NOT comparable as raw dollars: a
    # metered API bill vs a GPU-time estimate. Spell that out so no one reads the
    # column as a single apples-to-apples number.
    note = [
        "\\* **Cost basis differs by row and the dollars are NOT directly comparable.**"
    ]
    if any_hardware:
        note.append(
            " _hardware-derived (throughput)_ (self-hosted vLLM): a rented GPU has no "
            "per-token bill, so cost is the model's blended cost-per-token -- the "
            "cheapest concurrency level of ITS OWN throughput sweep -- times the "
            "tokens this run processed. Each row is priced at the rate of the "
            "instance that model was actually served on, named in this column, at "
            "that instance's rate in self-hosted/vllm/pricing.json. **The instance "
            "differs by row, so a self-hosted dollar figure is the cost of that "
            "model's work on ITS OWN hardware, not a common basis**; comparing two "
            "self-hosted rows compares two model-plus-hardware pairings rather than "
            "the models alone. This prices the real work done, unlike a wall-clock "
            "estimate that would also charge idle agent-thinking time."
        )
    if any_metered:
        note.append(
            " _metered (Bedrock)_: a hosted API's real per-token bill, summed over "
            "the run. It is a metered invoice, not a hardware estimate, and (unlike "
            "the self-hosted rows) it benefits from Bedrock prompt caching."
        )
    if any_kiro:
        note.append(
            " _Kiro credits_ (kiro-cli): kiro-cli reports no tokens, only credits "
            "consumed; cost is credits x $0.04/credit (configurable), summed over the "
            "run. Credits already embed the model's rate multiplier. This is a third "
            "basis -- neither a metered token bill nor a GPU estimate. NOTE: Kiro is a "
            "per-developer monthly subscription (kiro.dev/pricing) with credits "
            "included in the seat; $0.04/credit is the OVERAGE rate, so this treats "
            "every credit as add-on overage (worst case). pi/Claude Code on Bedrock "
            "are pure usage-based per-token billing with no seat -- a fair comparison "
            "models kiro's seat cost + volume, not just this per-task figure."
        )
    note.append(" See [cost-per-task-methodology.md](cost-per-task-methodology.md).")
    lines += [
        "",
        "".join(note),
        "",
        "† **Tokens processed** counts input + output + cache-read + cache-write "
        "-- all tokens the model actually processed, not just fresh input+output. On "
        "the Bedrock path a task often reports only ~2 fresh input tokens with the "
        "rest served from prompt cache, so counting input+output alone would "
        "understate the real work ~100x. (Self-hosted rows report their cache reuse "
        "via server-side Prometheus counters, folded in here where present.)",
        "",
        "A task scoring 0 (missing/empty artifacts) is a model failure, excluded "
        "from the mean but counted in `Completed`. A model with 0 scored tasks "
        "did not complete any task under this harness.",
        "",
        "## Charts",
        "",
        "### Cost vs. quality (Pareto frontier)",
        "",
        f"![Cost vs quality, {label} harness]({_rel(cq)})",
        "",
        "### Quality by dimension (radar)",
        "",
        f"![Quality radar, {label} harness]({_rel(radar)})",
    ]
    # The cost/accuracy bubble sizes each bubble by tokens processed; a harness
    # that reports no token counts (e.g. kiro-cli, which bills in credits) has no
    # meaningful bubble area, so omit that chart for it.
    has_tokens = any((r.get("total_tokens") or 0) > 0 for r in rows)
    if has_tokens:
        lines += [
            "",
            "### Cost vs. accuracy (bubble area = tokens)",
            "",
            "x = cost per task, y = mean score, bubble area = total tokens processed, "
            "color = hosting basis (metered Bedrock vs hardware-derived self-hosted -- "
            "NOT directly comparable as raw dollars; see the cost note above).",
            "",
            f"![Cost vs accuracy, {label} harness]({_rel(bubble)})",
        ]
    # Exactly one trailing newline (the end-of-file-fixer hook strips extras).
    return "\n".join(lines).rstrip("\n") + "\n"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a per-agent results doc from committed run-summaries.",
        epilog="Example: uv run scripts/gen_agent_report.py --harness pi",
    )
    parser.add_argument(
        "--harness",
        required=True,
        help="Harness slug: claude-code, pi, opencode, kiro-cli.",
    )
    parser.add_argument(
        "--skill",
        default="swe3",
        help="SWE skill folder to read: 'swe3' (default) or 'swe2'.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Dataset scope.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write harness-<slug>.md into (default: docs/).",
    )
    return parser.parse_args()


def main() -> None:
    """Collect one harness's run-summaries and write its results doc."""
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    rows = _collect(data_dir, args.harness, args.skill, args.repo)
    if not rows:
        raise SystemExit(
            f"no run-summary.json found under "
            f"{data_dir}/*/{args.harness}/{args.skill}/{args.repo}"
        )
    doc = _render(
        rows,
        harness=args.harness,
        skill=args.skill,
        repo=args.repo,
        out_dir=args.out_dir.expanduser().resolve(),
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"harness-{args.harness}-{args.skill}.md"
    out_path.write_text(doc, encoding="utf-8")
    logger.info("wrote %s (%d models)", out_path, len(rows))


if __name__ == "__main__":
    main()
