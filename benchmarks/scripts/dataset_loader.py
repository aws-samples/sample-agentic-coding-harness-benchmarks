#!/usr/bin/env python3
"""Load and validate SWE benchmark dataset YAML files.

A dataset file (see ``benchmarks/dataset/*.yaml``) is a metadata header plus a
list of software-engineering tasks. This module parses one into typed Pydantic
models and validates the schema, so every consumer (the run harness, the
reviewer, the report generators) reads the same enforced shape instead of
poking at raw dicts.

Run it from the ``benchmarks/`` directory with its own venv:

    uv run scripts/dataset_loader.py dataset/mcp-gateway-registry.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSIONS = {"1.0"}


class DatasetError(Exception):
    """Raised when a dataset file is missing, unparseable, or invalid."""


class GroundTruth(BaseModel):
    """Reviewer-facing notes on the intended solution (never shown to the agent)."""

    model_config = ConfigDict(extra="forbid")

    approach: str | None = None
    expectations: list[str] = Field(default_factory=list)
    reference_url: str | None = None


class Task(BaseModel):
    """One software-engineering task in a dataset."""

    model_config = ConfigDict(extra="forbid")

    id: str
    repo: str
    ref: str | None = None
    complexity: str
    tags: list[str] = Field(default_factory=list)
    problem_statement: str | None = None
    problem_issue_url: str | None = None
    clarifying_answers: str | None = None
    ground_truth: GroundTruth | None = None

    @model_validator(mode="after")
    def _require_problem_source(self) -> Task:
        """At least one of problem_statement / problem_issue_url must be set."""
        if not self.problem_statement and not self.problem_issue_url:
            raise ValueError(
                f"task '{self.id}': needs at least one of 'problem_statement' or "
                "'problem_issue_url'"
            )
        return self


class Dataset(BaseModel):
    """A parsed, validated benchmark dataset."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    name: str
    title: str
    description: str
    default_ref: str
    output_scope: str | None = None
    metrics: list[str] = Field(min_length=1)
    complexity_levels: list[str] = Field(min_length=1)
    tasks: list[Task] = Field(min_length=1)
    created: str | None = None

    @field_validator("created", mode="before")
    @classmethod
    def _coerce_created(cls, value: Any) -> Any:
        """YAML parses bare `2026-07-22` as a date; keep it as an ISO string."""
        if value is None:
            return None
        return str(value)

    @field_validator("output_scope")
    @classmethod
    def _validate_output_scope(cls, value: str | None) -> str | None:
        """Keep output_scope usable as a single folder name."""
        if value is None:
            return None
        if not value or value != value.strip() or "/" in value or value in {".", ".."}:
            raise ValueError(
                f"output_scope '{value}' must be a single folder name "
                "(no slashes, no surrounding whitespace)"
            )
        return value

    @model_validator(mode="after")
    def _validate_cross_field(self) -> Dataset:
        """Enforce schema version, complexity enum, unique ids, and default refs."""
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported schema_version '{self.schema_version}' "
                f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
            )

        levels = set(self.complexity_levels)
        seen: set[str] = set()
        for task in self.tasks:
            if task.complexity not in levels:
                raise ValueError(
                    f"task '{task.id}': complexity '{task.complexity}' not in "
                    f"{sorted(levels)}"
                )
            if task.id in seen:
                raise ValueError(f"duplicate task id '{task.id}'")
            seen.add(task.id)
            # Resolve each task's clone ref to the dataset default when unset, so
            # every consumer sees a concrete, reproducible ref.
            if task.ref is None:
                task.ref = self.default_ref
        return self

    def task_by_id(self, task_id: str) -> Task:
        """Return the task with the given id.

        Args:
            task_id: The task slug to look up.

        Returns:
            The matching task.

        Raises:
            KeyError: If no task has that id.
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)

    def resolved_ref(self, task: Task) -> str:
        """Return the git ref to clone for a task (its own ref or the default)."""
        return task.ref or self.default_ref

    def scope_for(self, repo_name: str) -> str:
        """Return the folder name results are grouped under for ``repo_name``.

        Results live at ``<model>/<harness>/<skill>/<scope>/<task>/``. The scope
        defaults to the repository name, which is right until two datasets target
        the *same* repository: they would then share a folder, and the
        folder-level ``run-summary.json`` of one would be rebuilt over the other's
        tasks. ``output_scope`` gives such a dataset its own folder.

        Args:
            repo_name: The repository basename the task clones, used as the
                default when the dataset sets no ``output_scope``.

        Returns:
            The scope folder name.
        """
        return self.output_scope or repo_name


def load_dataset(path: str | Path) -> Dataset:
    """Load and validate a benchmark dataset from a YAML file.

    Args:
        path: Path to the dataset YAML file.

    Returns:
        The parsed, validated Dataset.

    Raises:
        DatasetError: If the file is missing, unparseable, or fails validation.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DatasetError(f"Dataset file not found: {file_path}")

    try:
        raw: Any = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DatasetError(f"Failed to parse YAML in {file_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise DatasetError(f"{file_path}: top level must be a mapping")

    try:
        return Dataset.model_validate(raw)
    except ValidationError as exc:
        raise DatasetError(f"Invalid dataset {file_path}:\n{exc}") from exc


def _summarize(dataset: Dataset) -> None:
    """Log a short human-readable summary of a loaded dataset."""
    logger.info("Dataset: %s (%s)", dataset.title, dataset.name)
    logger.info("  schema_version: %s", dataset.schema_version)
    logger.info("  default_ref: %s", dataset.default_ref)
    logger.info("  metrics: %s", ", ".join(dataset.metrics))
    logger.info("  tasks: %s", len(dataset.tasks))
    by_level = {
        lvl: sum(1 for t in dataset.tasks if t.complexity == lvl)
        for lvl in dataset.complexity_levels
    }
    logger.info("  by complexity: %s", by_level)
    for task in dataset.tasks:
        if task.problem_statement and task.problem_issue_url:
            source = "issue+text"
        elif task.problem_statement:
            source = "text"
        else:
            source = "issue"
        logger.info(
            "    - %s [%s] ref=%s (%s)",
            task.id,
            task.complexity,
            dataset.resolved_ref(task),
            source,
        )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate and summarize a SWE benchmark dataset YAML file.",
        epilog="Example:\n  uv run scripts/dataset_loader.py "
        "dataset/mcp-gateway-registry.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", help="Path to the dataset YAML file")
    return parser.parse_args()


def main() -> None:
    """Validate the given dataset file and print a summary."""
    args = _parse_args()
    try:
        dataset = load_dataset(args.dataset)
    except DatasetError as exc:
        logger.error("Invalid dataset: %s", exc)
        sys.exit(1)
    _summarize(dataset)
    logger.info("Dataset is valid.")


if __name__ == "__main__":
    main()
