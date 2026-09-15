"""Tests for the throughput harness's filesystem safety.

Two mechanisms, both of which delete files, so both are pinned here.

The slot dir is built from the model slug and then ``shutil.rmtree``d when the
session ends, so a slug that escapes ``clone_dir`` would silently delete a tree
outside it. ``model_to_slug`` does not sanitize path separators (it only strips a
Bedrock prefix and a bracketed suffix), so the harness has to.

``_sweep_stray_root_writes`` moves files out of the user's working tree -- the ones
a load session drops in the repo root by writing a bare relative path instead of its
absolute ``artifacts_dir``. Its guards (plain files only, mtime inside the window,
git-untracked, fail closed when trackedness is unknown) are what keep that from
touching real work, and it quarantines rather than deletes so a misattributed file is
recoverable.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess  # nosec B404 - list-form git only, to build a temp repo fixture
import sys
import tempfile
import time
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_HARNESS_PATH = _SCRIPTS_DIR / "run-throughput-harness.py"
_spec = importlib.util.spec_from_file_location("run_throughput_harness", _HARNESS_PATH)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

_CLONE_DIR = "/opt/dlami/nvme/tmp/swe-clones"


class TestSafePathComponent(unittest.TestCase):
    """``_safe_path_component`` must yield exactly one child-naming component."""

    def test_real_model_slugs_pass_through_unchanged(self) -> None:
        """Dots and dashes are legitimate in slugs and must survive verbatim.

        The committed artifact folders use dotted names (``qwen3.6-35b``), so
        rewriting them would point the harness at a different directory.
        """
        for slug in ("qwen3.6-35b", "gemma-4-31b", "qwen3-coder-30b", "glm-5.2"):
            with self.subTest(slug=slug):
                self.assertEqual(harness._safe_path_component(slug, "model"), slug)

    def test_traversal_cannot_escape_the_clone_dir(self) -> None:
        """A slug with ``..`` or ``/`` must not resolve outside ``clone_dir``."""
        hostile = (
            "../../../../etc/evil",
            "..",
            "../",
            "a/b/c",
            "/absolute/path",
            "....//....//tmp",
        )
        for slug in hostile:
            with self.subTest(slug=slug):
                safe = harness._safe_path_component(slug, "model")
                self.assertNotIn("/", safe)
                slot_dir = Path(_CLONE_DIR) / f"swe-thru-{safe}-c1-1"
                resolved = os.path.normpath(str(slot_dir))
                self.assertTrue(
                    resolved.startswith(_CLONE_DIR + os.sep),
                    f"{slug!r} escaped to {resolved}",
                )

    def test_empty_and_dot_only_slugs_fall_back(self) -> None:
        """A slug that reduces to nothing must not produce a bare or hidden dir."""
        for slug in ("", ".", "..", "..."):
            with self.subTest(slug=slug):
                self.assertEqual(harness._safe_path_component(slug, "model"), "model")

    def test_result_is_never_hidden(self) -> None:
        """Leading dots are stripped so the slot dir is visible to cleanup tooling."""
        self.assertEqual(harness._safe_path_component(".hidden", "model"), "hidden")


class TestSweepStrayRootWrites(unittest.TestCase):
    """The stray sweep must quarantine leaked load artifacts and nothing else."""

    def setUp(self) -> None:
        """Create a throwaway git repo with one tracked, committed file."""
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self._quarantine_tmp = tempfile.TemporaryDirectory()
        self.quarantine = Path(self._quarantine_tmp.name)
        self.addCleanup(self._quarantine_tmp.cleanup)
        env = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        }
        for args in (
            ["init", "-q"],
            ["config", "user.email", "test@example.invalid"],
            ["config", "user.name", "test"],
        ):
            subprocess.run(  # nosec B603 B607 - hardcoded 'git', list args, no shell
                ["git", *args], cwd=self.root, env=env, check=True, capture_output=True
            )
        (self.root / "tracked.md").write_text("real work\n", encoding="utf-8")
        for args in (["add", "tracked.md"], ["commit", "-qm", "seed"]):
            subprocess.run(  # nosec B603 B607 - hardcoded 'git', list args, no shell
                ["git", *args], cwd=self.root, env=env, check=True, capture_output=True
            )

    def _sweep(self, before: set[str], since: float) -> list[str]:
        """Run the sweep against the fixture repo and quarantine dir."""
        return harness._sweep_stray_root_writes(
            self.root, before, since, self.quarantine
        )

    def _quarantined(self) -> list[str]:
        """Names of files sitting in the quarantine tree, at any depth."""
        return sorted(p.name for p in self.quarantine.rglob("*") if p.is_file())

    def test_untracked_file_written_during_the_level_is_quarantined(self) -> None:
        """The github-issue.md case: a new untracked root file is moved out."""
        before = harness._root_entry_names(self.root)
        since = time.time()
        stray = self.root / "github-issue.md"
        stray.write_text("# GitHub Issue\n", encoding="utf-8")

        moved = self._sweep(before, since)

        self.assertEqual(moved, ["github-issue.md"])
        self.assertFalse(stray.exists())
        self.assertEqual(self._quarantined(), ["github-issue.md"])

    def test_quarantined_content_is_preserved(self) -> None:
        """Moved, not deleted: a misattributed file must be recoverable."""
        before = harness._root_entry_names(self.root)
        (self.root / "notes-from-a-session.md").write_text(
            "recoverable\n", encoding="utf-8"
        )

        self._sweep(before, time.time() - 5)

        survivors = [p for p in self.quarantine.rglob("*") if p.is_file()]
        self.assertEqual(len(survivors), 1)
        self.assertEqual(survivors[0].read_text(encoding="utf-8"), "recoverable\n")

    def test_tracked_file_is_never_touched(self) -> None:
        """A git-tracked path stays even when it looks new and was just written."""
        tracked = self.root / "tracked.md"
        tracked.write_text("edited during the window\n", encoding="utf-8")

        # before=set() pretends nothing was there at level start, the worst case.
        moved = self._sweep(set(), time.time() - 5)

        self.assertEqual(moved, [])
        self.assertTrue(tracked.exists())
        self.assertEqual(self._quarantined(), [])

    def test_preexisting_file_is_never_touched(self) -> None:
        """A file present at level start is not a stray, however it looks."""
        (self.root / "notes.md").write_text("mine\n", encoding="utf-8")
        before = harness._root_entry_names(self.root)

        moved = self._sweep(before, time.time() - 5)

        self.assertEqual(moved, [])
        self.assertTrue((self.root / "notes.md").exists())

    def test_file_older_than_the_window_is_never_touched(self) -> None:
        """Only files modified inside the window are attributed to this level."""
        old = self.root / "appeared-but-old.md"
        old.write_text("written earlier\n", encoding="utf-8")
        os.utime(old, (time.time() - 3600, time.time() - 3600))

        moved = self._sweep(set(), time.time())

        self.assertEqual(moved, [])
        self.assertTrue(old.exists())

    def test_directories_and_symlinks_are_reported_not_moved(self) -> None:
        """A new dir or symlink is left for a human; moving either is unsafe."""
        outside = Path(self._tmp.name).parent / f"sweep-target-{os.getpid()}.md"
        outside.write_text("must survive\n", encoding="utf-8")
        self.addCleanup(outside.unlink)
        (self.root / "strayfolder").mkdir()
        (self.root / "straylink.md").symlink_to(outside)

        moved = self._sweep(set(), time.time() - 5)

        self.assertEqual(moved, [])
        self.assertTrue((self.root / "strayfolder").is_dir())
        self.assertTrue((self.root / "straylink.md").is_symlink())
        self.assertTrue(outside.exists())
        self.assertEqual(self._quarantined(), [])

    def test_non_git_root_fails_closed(self) -> None:
        """Without a git repo, trackedness is unknown, so nothing is moved."""
        with tempfile.TemporaryDirectory() as plain:
            root = Path(plain)
            stray = root / "github-issue.md"
            stray.write_text("# GitHub Issue\n", encoding="utf-8")

            moved = harness._sweep_stray_root_writes(
                root, set(), time.time() - 5, self.quarantine
            )

            self.assertEqual(moved, [])
            self.assertTrue(stray.exists())
            self.assertEqual(self._quarantined(), [])

    def test_no_quarantine_dir_is_created_when_the_root_stays_clean(self) -> None:
        """A clean level must not litter clone_dir with empty quarantine dirs."""
        before = harness._root_entry_names(self.root)

        moved = self._sweep(before, time.time())

        self.assertEqual(moved, [])
        self.assertEqual(list(self.quarantine.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
