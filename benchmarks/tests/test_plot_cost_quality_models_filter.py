"""Tests for restricting a cost/quality chart to a named subset of models.

A Pareto chart makes a claim by omission: a model that is not drawn looks like a
model that was beaten. So the ``--models`` filter has to be loud in exactly two
places -- a slug that produced nothing must be an error rather than a quiet gap,
and a filtered frontier must not be written to the fleet-wide path where it would
be read as the whole field. Both are pinned here.

Also pinned: the cost-basis footnote survives matplotlib's MathText handling.
Any dollar figure in that note must be escaped, or matplotlib silently deletes
the very amounts the note exists to state.

And pinned: the frontier writer refuses to rebuild a committed file from a
different dataset scope. ``--repo`` still defaults to v1 while the headline
results are v2, so the guard is what stands between a documented command and
a silently replaced chart.

And pinned: no two point labels overlap. With the white label plates removed,
an overlap is unreadable rather than merely ugly, so the placer's fixed-point
iteration is checked against the real fleet layout.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_SPEC = importlib.util.spec_from_file_location(
    "plot_cost_quality", _SCRIPTS_DIR / "plot_cost_quality.py"
)
cq = importlib.util.module_from_spec(_SPEC)
# Register before exec: @dataclass resolves its own module out of sys.modules, so
# an unregistered module makes ModelPoint's definition raise on import.
sys.modules[_SPEC.name] = cq
_SPEC.loader.exec_module(cq)


def _run_summary(score: float, cost: float) -> dict:
    """A minimal run-summary.json the aggregator will accept.

    ``mean_task_score_excl_failed`` is the field ``_point_from_summary`` gates
    on, and ``mean_cost_usd_excl_failed`` is the token-priced fallback used when
    the model has no throughput sweep -- which is the case for these fixtures.
    """
    return {
        "mean_task_score_excl_failed": score,
        "mean_cost_usd_excl_failed": cost,
        "num_tasks": 1,
        "num_scored": 1,
        "failed_tasks": [],
        "tasks": [
            {
                "task": "t1",
                "judge_score": score,
                "total_cost_usd": cost,
                "input_tokens": 1000,
                "output_tokens": 100,
            }
        ],
    }


class TestModelsFilter(unittest.TestCase):
    """The filter selects models, and refuses to silently drop a named one."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data = Path(self._tmp.name)
        for model, score, cost in (("alpha", 80.0, 5.0), ("beta", 60.0, 1.0)):
            run_dir = self.data / model / "omp" / "swe3" / "repo-x"
            run_dir.mkdir(parents=True)
            (run_dir / cq.RUN_SUMMARY_FILENAME).write_text(
                json.dumps(_run_summary(score, cost)), encoding="utf-8"
            )

    def _collect(self, models: list[str] | None) -> list[str]:
        points = cq._collect_points(self.data, "repo-x", "omp", "swe3", models)
        return [p.model for p in points]

    def test_none_plots_every_model(self) -> None:
        self.assertEqual(self._collect(None), ["alpha", "beta"])

    def test_filter_selects_only_named_models(self) -> None:
        self.assertEqual(self._collect(["beta"]), ["beta"])

    def test_unknown_slug_is_an_error_not_a_silent_omission(self) -> None:
        """A typo'd slug would otherwise quietly yield a chart missing a model."""
        with self.assertRaises(SystemExit) as ctx:
            self._collect(["alpha", "gamma"])
        message = str(ctx.exception)
        self.assertIn("gamma", message)
        self.assertIn("no scorable", message)

    def test_a_model_with_no_runs_for_this_repo_is_reported(self) -> None:
        with self.assertRaises(SystemExit):
            cq._collect_points(self.data, "other-repo", "omp", "swe3", ["alpha"])


class TestFrontierJsonPath(unittest.TestCase):
    """A filtered frontier must land on its own path, and say it is filtered."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out_dir = Path(self._tmp.name)
        self.points = [
            cq.ModelPoint(
                model="alpha",
                mean_cost=5.0,
                mean_score=80.0,
                n_tasks=1,
                n_scored=1,
                excluded=[],
                hosting="self-hosted",
            )
        ]

    def test_default_stem_is_the_fleet_wide_name(self) -> None:
        path = cq._write_frontier_json(
            self.points, harness="omp", skill="swe3", repo="r", out_dir=self.out_dir
        )
        self.assertEqual(path.name, "pareto-frontier-omp-swe3.json")

    def test_custom_stem_does_not_clobber_the_fleet_wide_file(self) -> None:
        fleet = self.out_dir / "pareto-frontier-omp-swe3.json"
        fleet.write_text('{"sentinel": true}', encoding="utf-8")
        path = cq._write_frontier_json(
            self.points,
            harness="omp",
            skill="swe3",
            repo="r",
            out_dir=self.out_dir,
            stem="pareto-frontier-subset",
            models_filter=["alpha"],
        )
        self.assertEqual(path.name, "pareto-frontier-subset.json")
        self.assertEqual(
            json.loads(fleet.read_text(encoding="utf-8")), {"sentinel": True}
        )

    def test_filter_is_recorded_in_the_payload(self) -> None:
        """Without this the file implies it covers every model that has runs."""
        path = cq._write_frontier_json(
            self.points,
            harness="omp",
            skill="swe3",
            repo="r",
            out_dir=self.out_dir,
            stem="s",
            models_filter=["beta", "alpha"],
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["models_filter"], ["alpha", "beta"])

    def test_unfiltered_payload_records_no_filter(self) -> None:
        path = cq._write_frontier_json(
            self.points, harness="omp", skill="swe3", repo="r", out_dir=self.out_dir
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(payload["models_filter"])


class TestEscapeDollars(unittest.TestCase):
    """Dollar figures in the footnote must print, not become italic MathText."""

    def test_pair_of_rates_is_escaped(self) -> None:
        out = cq._escape_dollars("p5en $27.72/hr, g6e $4.533/hr")
        self.assertEqual(out, r"p5en \$27.72/hr, g6e \$4.533/hr")

    def test_already_escaped_is_left_alone(self) -> None:
        """Double-escaping would print a literal backslash next to the amount."""
        self.assertEqual(cq._escape_dollars(r"costs \$5"), r"costs \$5")

    def test_text_without_dollars_is_unchanged(self) -> None:
        self.assertEqual(cq._escape_dollars("Kiro credits"), "Kiro credits")

    def test_default_note_is_safe_to_render(self) -> None:
        """No bare ``$`` may reach MathText, whatever the shipped default says.

        The note used to quote one fleet-wide rate and this asserted it. It no
        longer can: the canonical throughput arms sit on different instances
        (p5en, p6-b300, p5e, g6e.4xlarge) at rates from $1.298 to $94.91, so
        naming a single one would be false. The escape invariant is what this
        test is actually for, and it holds whether or not the note names money.
        """
        escaped = cq._escape_dollars(cq._DEFAULT_COST_BASIS_NOTE)
        self.assertNotIn("$", escaped.replace(r"\$", ""))


class TestLabelOffsets(unittest.TestCase):
    """Labels must not overlap once spread, and isolated labels must not move.

    The layout used is the real omp/swe3 fleet on the p5en 3-year-SP basis, which
    is where the bug showed: ``minimax-m2.5`` was pushed down out of its own
    cluster and landed on top of ``qwen3-coder-480b*``. The overlap check below is
    written independently of the placer's internals -- it rebuilds each label's
    box from the returned offset and the measured text width -- so it fails on the
    symptom rather than on how the algorithm happens to be structured.

    The box model here mirrors what ``_plot`` draws: an unmoved label sits right
    of its dot, and a moved one is centred over it so its leader line is vertical.
    Centring is the part that bit -- it widens a label leftward by half its text.
    """

    # (model, mean cost $/task, mean judge score)
    FLEET = [
        ("claude-opus-5", 7.63, 82.83),
        ("glm-5.3", 6.85, 81.27),
        ("qwen3.8-27b", 1.79, 78.48),
        ("claude-sonnet-5", 3.21, 76.97),
        ("glm-5.2", 4.65, 74.36),
        ("kimi-k2.7-code", 3.60, 69.98),
        ("deepseek-v3.2", 2.28, 60.99),
        ("gemma-4-31b", 0.87, 59.74),
        ("qwen3.6-35b", 0.29, 59.24),
        ("claude-haiku-4-5", 0.69, 56.18),
        ("minimax-m2.5", 0.48, 53.29),
        ("qwen3-coder-480b", 1.68, 50.83),
        ("devstral-2-123b", 0.80, 47.64),
        ("qwen3-coder-30b", 0.47, 42.58),
    ]

    def setUp(self) -> None:
        self.points = [
            cq.ModelPoint(
                model=model,
                mean_cost=cost,
                mean_score=score,
                n_tasks=5,
                n_scored=5,
                excluded=[],
                hosting="self-hosted",
            )
            for model, cost, score in self.FLEET
        ]
        # Same figure geometry and axis padding as _plot, so the pixel distances
        # the placer reasons about match the shipped chart.
        self.fig, self.ax = cq.plt.subplots(figsize=(16, 10), dpi=150)
        self.addCleanup(cq.plt.close, self.fig)
        xs = [p.mean_cost for p in self.points]
        ys = [p.mean_score for p in self.points]
        xpad = max((max(xs) - min(xs)) * 0.12, 1.0)
        ypad = max((max(ys) - min(ys)) * 0.12, 3.0)
        self.ax.set_xlim(max(0.0, min(xs) - xpad), max(xs) + xpad * 2.2)
        self.ax.set_ylim(max(0.0, min(ys) - ypad), min(100.0, max(ys) + ypad))
        self.fig.canvas.draw()
        self.offsets = cq._label_offsets(self.ax, self.fig, self.points)

    def _boxes(self) -> dict[str, tuple[float, float, float, float]]:
        """Each label's (x0, x1, y0, y1) in pixels, keyed by model slug."""
        widths = cq._text_widths_px(self.ax, self.fig, self.points, "normal")
        line_px = cq.POINT_LABEL_FONTSIZE * 1.35 * self.fig.dpi / 72.0
        left_edge = self.ax.transAxes.transform((0.0, 0.0))[0]
        right_edge = self.ax.transAxes.transform((1.0, 0.0))[0]
        boxes = {}
        for point in self.points:
            x_px, y_px = self.ax.transData.transform(
                (point.mean_cost, point.mean_score)
            )
            dy = self.offsets[id(point)]
            y = y_px + dy * self.fig.dpi / 72.0
            half = widths[id(point)] / 2
            if abs(dy) > 1e-6 and x_px - half > left_edge and x_px + half < right_edge:
                x0, x1 = x_px - half, x_px + half  # centred over the dot
            else:
                x0, x1 = x_px + 12, x_px + 12 + widths[id(point)]
            boxes[point.model] = (x0, x1, y - line_px / 2, y + line_px / 2)
        return boxes

    def test_no_two_labels_overlap(self) -> None:
        boxes = self._boxes()
        for (na, a), (nb, b) in itertools.combinations(boxes.items(), 2):
            overlaps = a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]
            self.assertFalse(overlaps, msg=f"{na} overlaps {nb}: {a} vs {b}")

    def test_the_regression_pair_is_separated(self) -> None:
        """minimax-m2.5 landing on qwen3-coder-480b is the case that regressed.

        Named explicitly as well as covered by the sweep above, because this pair
        is the one whose boxes only meet once both are centred -- the geometry the
        placer used to get wrong.
        """
        mini, coder = self._boxes()["minimax-m2.5"], self._boxes()["qwen3-coder-480b"]
        self.assertFalse(
            mini[0] < coder[1]
            and coder[0] < mini[1]
            and mini[2] < coder[3]
            and coder[2] < mini[3]
        )

    def test_an_isolated_label_does_not_move(self) -> None:
        """A 0 offset is what suppresses the leader line, so it must stay 0."""
        lonely = [
            cq.ModelPoint(
                model=model,
                mean_cost=cost,
                mean_score=score,
                n_tasks=1,
                n_scored=1,
                excluded=[],
                hosting="bedrock",
            )
            for model, cost, score in (("low", 0.5, 20.0), ("high", 7.0, 90.0))
        ]
        fig, ax = cq.plt.subplots(figsize=(16, 10), dpi=150)
        self.addCleanup(cq.plt.close, fig)
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 100)
        fig.canvas.draw()
        offsets = cq._label_offsets(ax, fig, lonely)
        self.assertEqual(set(offsets.values()), {0.0})

    def test_measured_widths_differ_by_label_length(self) -> None:
        """A single average width is what let the long labels collide undetected."""
        widths = cq._text_widths_px(self.ax, self.fig, self.points, "normal")
        by_model = {p.model: widths[id(p)] for p in self.points}
        self.assertLess(by_model["glm-5.3"], by_model["claude-haiku-4-5"])

    def test_probe_artists_are_removed(self) -> None:
        """The width probes must not be left behind as invisible chart text."""
        before = len(self.ax.texts)
        cq._text_widths_px(self.ax, self.fig, self.points, "normal")
        self.assertEqual(len(self.ax.texts), before)


if __name__ == "__main__":
    unittest.main()


class TestScopeChangeGuard(unittest.TestCase):
    """A frontier JSON must not be silently rebuilt from another dataset.

    ``--repo`` defaults to the v1 dataset while the headline results are v2, so
    running a documented command without the flag rebuilt a 19-model v2 chart
    from whatever v1 runs existed and overwrote the good file. The scope is
    already in the payload, so the writer compares it and refuses.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "pareto-frontier-omp-swe3.json"
        self.addCleanup(self._tmp.cleanup)

    def _write(self, **scope: object) -> None:
        self.path.write_text(json.dumps(scope), encoding="utf-8")

    def test_same_scope_is_allowed(self) -> None:
        self._write(harness="omp", skill="swe3", repo="mcp-gateway-registry-v2")
        cq._guard_scope_change(
            self.path, harness="omp", skill="swe3", repo="mcp-gateway-registry-v2"
        )

    def test_different_repo_is_refused(self) -> None:
        """The exact mistake: the v1 default overwriting the v2 frontier."""
        self._write(harness="omp", skill="swe3", repo="mcp-gateway-registry-v2")
        with self.assertRaisesRegex(cq.ScopeMismatchError, "mcp-gateway-registry-v2"):
            cq._guard_scope_change(
                self.path, harness="omp", skill="swe3", repo="mcp-gateway-registry"
            )

    def test_reverse_mismatch_is_refused(self) -> None:
        """And the same bug the other way: v2 scope over the v1 combined file."""
        self._write(
            harnesses=["claude-code", "pi"], skill="swe3", repo="mcp-gateway-registry"
        )
        with self.assertRaises(cq.ScopeMismatchError):
            cq._guard_scope_change(
                self.path,
                harness="claude-code+pi",
                skill="swe3",
                repo="mcp-gateway-registry-v2",
            )

    def test_combined_shape_matching_scope_is_allowed(self) -> None:
        """The combined file keys on ``harnesses`` (a list), not ``harness``."""
        self._write(
            harnesses=["claude-code", "pi"], skill="swe3", repo="mcp-gateway-registry"
        )
        cq._guard_scope_change(
            self.path,
            harness="claude-code+pi",
            skill="swe3",
            repo="mcp-gateway-registry",
        )

    def test_force_overrides_a_mismatch(self) -> None:
        self._write(harness="omp", skill="swe3", repo="mcp-gateway-registry-v2")
        cq._guard_scope_change(
            self.path,
            harness="omp",
            skill="swe3",
            repo="mcp-gateway-registry",
            force=True,
        )

    def test_missing_file_is_allowed(self) -> None:
        """A first run has nothing to clobber."""
        cq._guard_scope_change(
            self.path, harness="omp", skill="swe3", repo="mcp-gateway-registry-v2"
        )

    def test_unreadable_file_is_allowed(self) -> None:
        """Writing a fresh file is the repair for a corrupt one."""
        self.path.write_text("{not json", encoding="utf-8")
        cq._guard_scope_change(
            self.path, harness="omp", skill="swe3", repo="mcp-gateway-registry-v2"
        )

    def test_payload_without_scope_keys_is_allowed(self) -> None:
        """A pre-guard file that never recorded a scope must not hard-fail."""
        self._write(note="an older payload")
        cq._guard_scope_change(
            self.path, harness="omp", skill="swe3", repo="mcp-gateway-registry-v2"
        )
