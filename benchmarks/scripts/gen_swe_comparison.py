#!/usr/bin/env python3
"""Generate the cross-model, cross-harness /swe3 comparison doc.

The swe counterpart to agentic-coding-throughput-comparison.md (which is throughput/
serving-economics from the synthetic sweep). This one is built from the REAL
/swe3 benchmark runs and combines the three axes a buyer trades off -- quality,
tokens, and cost -- for every model under BOTH harnesses (Claude Code and pi),
plus wall-clock latency.

Numbers come from gen_agent_report (_collect + _row_cost), so this doc, the
per-harness docs, and the charts all agree. The doc embeds, for each harness,
a cost-vs-accuracy bubble chart (x=cost/task, y=score, bubble area=tokens)
rendered by plot_cost_accuracy_bubble.py.

Usage:
    uv run scripts/gen_swe_comparison.py                 # -> docs/agentic-coding-swe-comparison.md
    uv run scripts/gen_swe_comparison.py --skill swe3 --out-dir ../docs
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path
from typing import Any

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

_GEN_PATH = _SCRIPTS_DIR / "gen_agent_report.py"
_spec = importlib.util.spec_from_file_location("gen_agent_report", _GEN_PATH)
assert _spec is not None and _spec.loader is not None  # nosec B101 - import-by-path guard, not runtime validation
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

HARNESSES = ("claude-code", "pi")
HARNESS_LABELS = {"claude-code": "Claude Code", "pi": "pi"}
HARNESS_CODES = {"claude-code": "cc", "pi": "pi"}

# The harness-comparison chart needs cross-metric REASONING that code cannot
# produce (e.g. "Claude Code is marginally cheaper here but pi is far more
# accurate on the same model, so still pick pi"). That commentary is authored
# by hand and lives between these sentinels. The generator SEEDS it once, then
# PRESERVES whatever the author has written on every subsequent regen -- so it
# is the author's job to update it when the chart is regenerated.
MANUAL_BEGIN = "<!-- MANUAL:harness-reading BEGIN -- author-maintained; preserved across regens. Update when the chart changes. -->"
MANUAL_END = "<!-- MANUAL:harness-reading END -->"


def _extract_manual_block(text: str) -> str | None:
    """Return the author's content between the sentinels in an existing doc.

    Returns None when the sentinels are absent or the block is empty/whitespace
    (so a first run, or a wiped block, falls back to the seeded default).
    """
    start = text.find(MANUAL_BEGIN)
    end = text.find(MANUAL_END)
    if start == -1 or end == -1 or end <= start:
        return None
    inner = text[start + len(MANUAL_BEGIN) : end].strip("\n")
    return inner if inner.strip() else None


def _human_tokens(value: float) -> str:
    """Compact token count (e.g. 82.7M)."""
    if value >= 1e9:
        return f"{value / 1e9:.1f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.0f}"


def _rows(data_dir: Path, harness: str, skill: str, repo: str) -> list[dict[str, Any]]:
    """Collect display rows (score, tokens, cost, cost/task, cost/point, mins)."""
    out: list[dict[str, Any]] = []
    for r in gen._collect(data_dir, harness, skill, repo):
        cost_str, basis = gen._row_cost(r)
        scored = r.get("num_scored") or 0
        cost = None if cost_str == "--" else float(cost_str.lstrip("$"))
        mean = r.get("mean")
        out.append(
            {
                "model": r["model"],
                "mean": mean,
                "completed": f"{scored}/{r.get('num_tasks')}",
                "total_tokens": r.get("total_tokens") or 0,
                "cost": cost,
                "cost_str": cost_str,
                "basis": basis,
                "cost_per_task": (cost / scored) if (cost and scored) else None,
                "cost_per_point": (cost / mean) if (cost and mean) else None,
                "minutes": (r.get("latency_seconds") or 0) / 60.0,
                "bedrock": basis.startswith("metered"),
            }
        )
    return out


def _table(rows: list[dict[str, Any]], label: str) -> list[str]:
    """Render one harness's results table (sorted by score)."""
    ordered = sorted(rows, key=lambda r: (r["mean"] is None, -(r["mean"] or 0.0)))
    lines = [
        f"### {label}",
        "",
        "| Model | Hosting | Mean score | Completed | Tokens processed | "
        "Run cost | Cost/task | Cost/point | Wall-clock |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ordered:
        host = "Bedrock" if r["bedrock"] else "self-hosted"
        mean = "-- (0 scored)" if r["mean"] is None else f"{r['mean']:.2f}"
        cpt = f"${r['cost_per_task']:.2f}" if r["cost_per_task"] else "--"
        cpp = f"${r['cost_per_point']:.2f}" if r["cost_per_point"] else "--"
        wall = f"{r['minutes']:.0f}m" if r["minutes"] else "--"
        lines.append(
            f"| {r['model']} | {host} | {mean} | {r['completed']} | "
            f"{_human_tokens(r['total_tokens'])} | {r['cost_str']} | {cpt} | "
            f"{cpp} | {wall} |"
        )
    lines.append("")
    return lines


def _seed_manual_block(per: dict[str, list[dict[str, Any]]], skill: str) -> str:
    """Seed the author-maintained harness-reading prose with a real, data-backed
    first draft. This is only used the FIRST time (no prior block to preserve);
    the author is expected to rewrite/extend it. It deliberately makes the
    cross-metric argument -- cost alone does not decide; quality gates the choice.
    """
    cc = {r["model"]: r for r in per["claude-code"]}
    pi = {r["model"]: r for r in per["pi"]}

    def _line(model: str) -> str | None:
        """A one-model 'cheaper-but-worse' sentence, only if the data supports it."""
        a, b = cc.get(model), pi.get(model)
        if not a or not b or a["mean"] is None or b["mean"] is None:
            return None
        if not (a["cost_per_task"] and b["cost_per_task"]):
            return None
        cc_cheaper = a["cost_per_task"] < b["cost_per_task"]
        pi_better = b["mean"] > a["mean"]
        if not (cc_cheaper and pi_better):
            return None
        cost_gap = (b["cost_per_task"] - a["cost_per_task"]) / b["cost_per_task"] * 100
        score_gap = b["mean"] - a["mean"]
        return (
            f"Take **{model}**: Claude Code is ~{cost_gap:.0f}% cheaper per task "
            f"(${a['cost_per_task']:.2f} vs ${b['cost_per_task']:.2f}), but pi scores "
            f"{score_gap:.0f} points higher ({b['mean']:.0f} vs {a['mean']:.0f}/100). "
            "A few cents does not buy back that much quality -- so you still run it "
            "under pi."
        )

    example = _line("qwen3.6-35b") or next(
        (s for m in pi if (s := _line(m)) is not None), None
    )
    parts = [
        "Read the chart across metrics, not one panel at a time. Claude Code winning "
        "the **cost** panel for a model rarely settles the choice: on the models "
        "where it is cheaper, either the absolute gap is a few cents, or the model's "
        "accuracy is too low to pick regardless of price. What decides a model is "
        "**quality first, then cost among the models that clear your quality bar.**",
    ]
    if example:
        parts.append(example)
    parts.append(
        "The one metric where the harness choice is lopsided is **wall-clock**: pi's "
        "single-agent loop finishes faster on nearly every model (no sub-agent "
        "fan-out), so unless a model scores clearly higher under Claude Code and the "
        "task is worth the extra time and tokens, pi is the default at the terminal."
    )
    return "\n\n".join(parts)


def _render(
    data_dir: Path,
    skill: str,
    repo: str,
    out_dir: Path,
    manual_block: str | None = None,
) -> str:
    """Render the full comparison document.

    ``manual_block`` is the author-maintained harness-reading prose preserved
    from a prior version of the doc; when None, a data-seeded default is used.
    """
    per = {h: _rows(data_dir, h, skill, repo) for h in HARNESSES}
    img = "images"

    def _cheapest_per_point(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        vals = [r for r in rows if r["cost_per_point"]]
        return min(vals, key=lambda r: r["cost_per_point"]) if vals else None

    def _best_score(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        vals = [r for r in rows if r["mean"] is not None]
        return max(vals, key=lambda r: r["mean"]) if vals else None

    def _is_full(row: dict[str, Any]) -> bool:
        """True when the run completed every task (e.g. '5/5')."""
        done, total = row["completed"].split("/")
        return done == total

    def _best_open(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Highest-scoring self-hosted (open-weight) model."""
        vals = [r for r in rows if r["mean"] is not None and not r["bedrock"]]
        return max(vals, key=lambda r: r["mean"]) if vals else None

    def _best_value(
        rows: list[dict[str, Any]], min_score: float
    ) -> dict[str, Any] | None:
        """Cheapest $/task among models scoring at least ``min_score``."""
        vals = [
            r
            for r in rows
            if r["mean"] and r["mean"] >= min_score and r["cost_per_task"]
        ]
        return min(vals, key=lambda r: r["cost_per_task"]) if vals else None

    def _cheapest_full(
        rows: list[dict[str, Any]], *, self_hosted_only: bool = False
    ) -> dict[str, Any] | None:
        """Cheapest $/task among runs that completed every task."""
        vals = [
            r
            for r in rows
            if r["cost_per_task"]
            and _is_full(r)
            and (not self_hosted_only or not r["bedrock"])
        ]
        return min(vals, key=lambda r: r["cost_per_task"]) if vals else None

    def _unreliable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Runs that did NOT complete every task (a reliability flag)."""
        return [r for r in rows if not _is_full(r)]

    def _harness_tallies(
        per_h: dict[str, list[dict[str, Any]]],
    ) -> dict[str, tuple[int, int, int, int]]:
        """For models run under BOTH harnesses, tally (pi_wins, cc_wins, ties, n)
        per metric, using the same 2%-tie rule as the harness-delta chart."""
        cc = {r["model"]: r for r in per_h["claude-code"]}
        pi = {r["model"]: r for r in per_h["pi"]}
        common = [m for m in cc if m in pi]
        metrics = [
            ("mean", True),
            ("cost_per_task", False),
            ("total_tokens", False),
            ("minutes", False),
        ]
        tally: dict[str, tuple[int, int, int, int]] = {}
        for key, higher_is_better in metrics:
            pi_w = cc_w = tie = n = 0
            for m in common:
                a, b = cc[m].get(key), pi[m].get(key)
                if a is None or b is None:
                    continue
                n += 1
                if abs(a - b) < 1e-9 or (a and abs(a - b) / max(abs(a), abs(b)) < 0.02):
                    tie += 1
                    continue
                better_pi = b > a if higher_is_better else b < a
                pi_w, cc_w = (pi_w + 1, cc_w) if better_pi else (pi_w, cc_w + 1)
            tally[key] = (pi_w, cc_w, tie, n)
        return tally

    lines = [
        f"# Agentic coding: model comparison on /{skill} (quality, tokens, cost)",
        "",
        "How every benchmarked model compares as an **agentic coding** engine on "
        f"real `/{skill}` tasks against `{repo}`, under **both harnesses** (Claude "
        "Code and pi). Unlike the serving-economics view in "
        "[agentic-coding-throughput-comparison.md](agentic-coding-throughput-comparison.md) "
        "(synthetic throughput sweep), this doc is built from the actual benchmark "
        "runs and combines the three axes a buyer trades off -- **quality, tokens, "
        "and cost** -- plus wall-clock latency.",
        "",
        "Generated from the committed `run-summary.json` files; regenerate with "
        f"`uv run scripts/gen_swe_comparison.py --skill {skill}`. Numbers match the "
        "per-harness docs ([Claude Code]("
        f"harness-claude-code-{skill}.md), [pi](harness-pi-{skill}.md)) and the "
        "charts below exactly.",
        "",
        "## Cost basis (read this first)",
        "",
        "Two non-comparable cost bases share the cost columns; each row states which:",
        "",
        "- **metered (Bedrock)** -- a hosted API's real per-token bill, summed over "
        "the run. Benefits from Bedrock prompt caching.",
        "- **hardware-derived (self-hosted)** -- a rented GPU has no per-token bill, "
        "so cost is the model's blended $/token (measured by the p5en.48xlarge "
        "throughput sweep) times the tokens the run processed. Every self-hosted "
        "row uses that one sweep, including models served on a smaller "
        "g6e.12xlarge box, so the fleet shares a single basis -- a row is the cost "
        "of that model's work on p5en, not a quote for the box it ran on. See "
        "[cost-per-task-methodology.md](cost-per-task-methodology.md).",
        "",
        "`Cost/task` = run cost / scored tasks. `Cost/point` = run cost / mean score "
        "-- a value-efficiency figure (lower is more quality per dollar).",
        "",
        "## Does the harness matter?",
        "",
        "For every model run under both harnesses, this compares Claude Code vs pi "
        "on each metric. Each row is one model; the connector points to the better "
        "harness (higher score / lower cost, tokens, latency), and each panel title "
        "tallies how often pi wins. Comparing one model's two harnesses is fair even "
        "for cost -- its hosting basis is identical under both.",
        "",
        f"![Harness comparison, {skill}]({img}/harness-delta-{skill}.png)",
        "",
        "### Reading the chart (author-maintained)",
        "",
        "> The win-tallies above are mechanical. The prose below is **hand-written "
        "reasoning** about what the chart means for a model choice -- the kind of "
        "cross-metric judgement code cannot produce. It is written from the "
        "machine-readable data behind the charts: "
        f"[`metrics/harness-delta-{skill}.json`](metrics/harness-delta-{skill}.json) "
        "(every model x harness x metric, per-metric winner, win tallies) and "
        f"[`metrics/pareto-frontier-pi-{skill}.json`](metrics/pareto-frontier-pi-"
        f"{skill}.json) (the score-vs-cost frontier, split by hosting). It is "
        "preserved across regens. **When you regenerate the charts, re-read those "
        "JSONs and update this text to match.**",
        "",
        MANUAL_BEGIN,
        manual_block if manual_block is not None else _seed_manual_block(per, skill),
        MANUAL_END,
        "",
        "## Results by harness",
        "",
        "For each harness: a results table (quality, tokens, run cost + the two "
        "normalized cost lenses, wall-clock; sorted by score) followed by a "
        "cost-vs-accuracy bubble chart -- x = cost/task, y = mean score, bubble "
        "area = tokens processed, color = hosting basis.",
        "",
    ]
    for h in HARNESSES:
        lines += _table(per[h], HARNESS_LABELS[h])
        code = HARNESS_CODES[h]
        lines += [
            f"Cost vs. accuracy ({HARNESS_LABELS[h]}) -- bubble area = tokens "
            "processed, color = hosting (Bedrock vs self-hosted):",
            "",
            f"![{HARNESS_LABELS[h]} cost vs accuracy]"
            f"({img}/cost-accuracy-bubble-{code}-{skill}.png)",
            "",
        ]

    # Data-derived takeaways (so the prose never drifts from the tables).
    # Anchor on the pi harness for the model-picking guidance: it is the shape
    # a developer at the terminal actually sees (single agent, no fan-out).
    pi_rows = per["pi"]
    top = _best_score(pi_rows)
    open_top = _best_open(pi_rows)
    budget = _cheapest_full(pi_rows)
    open_budget = _cheapest_full(pi_rows, self_hosted_only=True)
    # A "competent" bar at 80% of the top score -- cheapest full run clearing it.
    bar = (top["mean"] * 0.8) if top else 0.0
    value = _best_value(pi_rows, bar)
    flaky = _unreliable(pi_rows)

    lines += [
        "## Guidance: which model for which task, and what it costs",
        "",
        "A practical way to read the tables: pick the cheapest model whose quality "
        "clears the bar your task needs. Costs below are **per task** (one real "
        f"`/{skill}` problem; a run is 5 tasks). Numbers are from the **pi** column "
        "-- the single-agent shape a developer drives at the terminal. Remember the "
        "two cost bases are not comparable as raw dollars (Bedrock is a metered "
        "bill; self-hosted is hardware-derived) -- see the methodology doc.",
        "",
    ]
    if top:
        lines.append(
            f"- **Top-quality tier (hard / high-stakes changes): `{top['model']}`** "
            f"-- highest score ({top['mean']:.0f}/100) at ${top['cost_per_task']:.2f}/task. "
            "Reach for it on security-sensitive, cross-cutting, or "
            "get-it-right-the-first-time work where a wrong design is expensive. "
            "You pay the most, but accuracy is the most."
        )
    if open_top:
        lines.append(
            f"- **Open-weight workhorse (bulk of day-to-day coding): `{open_top['model']}`** "
            f"-- best self-hosted quality ({open_top['mean']:.0f}/100) at "
            f"${open_top['cost_per_task']:.2f}/task. Strong on real refactors and "
            "features; the model to standardize on if you self-host and route most "
            "tickets to one engine."
        )
    if value and (not top or value["model"] != top["model"]):
        lines.append(
            f"- **Best value (most quality per dollar): `{value['model']}`** -- "
            f"clears ~{bar:.0f}/100 (80% of the top score) at just "
            f"${value['cost_per_task']:.2f}/task. The sweet spot for well-scoped "
            "tasks: most of the quality, a fraction of the cost."
        )
    if budget:
        lines.append(
            f"- **Budget tier (routine / high-volume edits): `{budget['model']}`** "
            f"-- cheapest full 5/5 run at ${budget['cost_per_task']:.2f}/task "
            f"(score {budget['mean']:.0f}/100). Good for boilerplate, small fixes, "
            "and throwaway scaffolding where you will review the output anyway."
            + (
                f" Cheapest self-hosted equivalent: `{open_budget['model']}` at "
                f"${open_budget['cost_per_task']:.2f}/task."
                if open_budget and open_budget["model"] != budget["model"]
                else ""
            )
        )
    if flaky:
        names = ", ".join(f"`{r['model']}` ({r['completed']})" for r in flaky)
        lines.append(
            f"- **Reliability flag:** {names} did **not** finish every task under pi "
            "-- cheap per task, but a non-completion is a failure, not a discount. "
            "Do not route unattended work to a model that does not reliably finish."
        )
    lines += [
        "",
        "## Does the harness change the answer? (pi vs Claude Code)",
        "",
        "For the models run under both harnesses, tallying each metric with the "
        "chart's 2%-tie rule (a model's hosting basis is identical under both, so "
        "even cost is a fair within-model comparison):",
        "",
    ]
    tally = _harness_tallies(per)
    label = {
        "mean": "Quality (mean score)",
        "cost_per_task": "Cost per task",
        "total_tokens": "Total tokens processed",
        "minutes": "Wall-clock latency",
    }
    for key in ("mean", "cost_per_task", "total_tokens", "minutes"):
        pi_w, cc_w, tie, n = tally[key]
        lines.append(
            f"- **{label[key]}:** pi wins {pi_w}/{n}, Claude Code wins {cc_w}/{n}"
            + (f", {tie} tie{'s' if tie != 1 else ''}" if tie else "")
            + "."
        )
    lines += [
        "",
        "- **Practical read:** pi's single-agent loop is consistently **faster in "
        "wall-clock** (no sub-agent fan-out to coordinate) and often cheaper, while "
        "Claude Code's multi-agent orchestration can lift quality on some models at "
        "the price of more tokens, dollars, and time. For a developer at the "
        "terminal, pi is the better default on latency and cost; switch to Claude "
        "Code when a specific model scores meaningfully higher there and the task "
        "justifies the extra spend. Pick the harness per model, not globally -- the "
        "same model can sit very differently under the two (compare its row across "
        "the tables and its bubble in each chart).",
        "",
        "## How to reproduce",
        "",
        "```bash",
        "cd benchmarks",
        f"uv run python scripts/gen_swe_comparison.py --skill {skill}",
        "# charts:",
        f"uv run python scripts/plot_cost_accuracy_bubble.py --harness pi --skill {skill}",
        f"uv run python scripts/plot_cost_accuracy_bubble.py --harness claude-code --skill {skill}",
        "```",
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate the cross-harness /swe comparison doc from run-summaries.",
        epilog="Example: uv run scripts/gen_swe_comparison.py --skill swe3",
    )
    parser.add_argument("--skill", default="swe3", help="SWE skill (default: swe3).")
    parser.add_argument("--repo", default="mcp-gateway-registry", help="Dataset scope.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Render the comparison doc and write it under out-dir."""
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_path = out_dir / f"agentic-coding-swe-comparison-{args.skill}.md"

    # Preserve the author-maintained harness-reading block from any prior doc.
    manual_block = None
    if out_path.exists():
        manual_block = _extract_manual_block(out_path.read_text(encoding="utf-8"))
        if manual_block is not None:
            logger.info("preserved author-maintained harness-reading block")
        else:
            logger.info("no prior manual block found; seeding a data-backed default")

    doc = _render(data_dir, args.skill, args.repo, out_dir, manual_block=manual_block)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
