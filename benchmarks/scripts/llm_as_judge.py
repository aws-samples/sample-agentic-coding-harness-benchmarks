#!/usr/bin/env python3
"""Evaluate one folder of SWE design artifacts with an Amazon Bedrock judge.

Makes one Bedrock Mantle Responses API request, writes ``eval.json``, and adds
the same object to ``metrics.json["evaluation"]`` when metrics exist.

The score schema, prompt rendering, reply validation, and atomic write are
shared with the agentic ``codex_judge.py`` backend via ``judge_common.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from judge_common import (
    DEFAULT_TEMPLATE_PATH,
    EvaluationResult,
    JudgeError,
    atomic_write_json,
    identify_folder,
    missing_artifacts,
    optional_file,
    parse_and_validate_result,
    render_judge_prompt,
    zero_score_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
DEFAULT_MAX_OUTPUT_TOKENS = 8_000
DEFAULT_TIMEOUT_SECONDS = 300


def _response_text(payload: dict[str, Any]) -> str:
    if payload.get("status") == "incomplete":
        details = payload.get("incomplete_details")
        raise JudgeError(f"judge response is incomplete: {details}")
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            parts.extend(
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            )
        text = "".join(parts).strip()
        if text:
            return text
    raise JudgeError("judge response has no output_text content")


def evaluate_artifact_folder(
    folder: str | Path,
    model: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
    task_context: str | None = None,
    repository_context: str | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    reasoning_effort: str | None = None,
    use_json_response_format: bool = True,
    overwrite: bool = True,
    write_outputs: bool = True,
    session: Any | None = None,
) -> dict[str, Any]:
    """Evaluate one folder with exactly one Bedrock Mantle Responses request.

    Args:
        folder: Directory containing the four required Markdown artifacts, plus
            optionally the /swe2 implementation artifact (implementation.md +
            patch.diff); when absent, the implementation is judged 0.
        model: Raw model ID accepted by the configured Bedrock endpoint.
        base_url: OpenAI-compatible API base URL ending at ``/openai/v1``.
        api_key: Bearer token; defaults to ``MANTLE_API_KEY``.
        template_path: Judge prompt template path.
        task_context: Optional independent task requirements.
        repository_context: Optional independent repository evidence.
        max_output_tokens: Maximum completion tokens.
        timeout_seconds: HTTP timeout.
        reasoning_effort: Optional GPT-5-family reasoning effort.
        use_json_response_format: Request strict JSON Schema output when true.
        overwrite: Allow replacing an existing ``eval.json``.
        write_outputs: Write output files when true.
        session: Optional requests-compatible client for reuse or tests.

    Returns:
        The validated evaluation with attached judge metadata.
    """
    if not isinstance(model, str) or not model.strip():
        raise JudgeError("judge model must be a non-empty string")
    if max_output_tokens < 1 or timeout_seconds < 1:
        raise JudgeError("max_output_tokens and timeout_seconds must be positive")
    if not base_url.startswith(("http://", "https://")):
        raise JudgeError("base_url must start with http:// or https://")

    artifact_dir = Path(folder).expanduser().resolve()
    eval_path = artifact_dir / "eval.json"
    if eval_path.exists() and not overwrite:
        raise JudgeError(f"eval.json exists and overwrite is disabled: {eval_path}")

    # Missing/empty required artifacts are a model failure, not a judging error:
    # score 0 with an explicit verdict instead of erroring out (parity with the
    # codex judge).
    missing = missing_artifacts(artifact_dir)
    if missing:
        task_id, candidate_id = identify_folder(artifact_dir)
        logger.warning(
            "%s: missing artifact(s) %s -- scoring 0 (model failure)",
            artifact_dir,
            ", ".join(missing),
        )
        result = zero_score_result(
            task_id=task_id, candidate_id=candidate_id, missing=missing
        )
        result["judge"] = {
            "model": model,
            "provider": "bedrock-mantle",
            "repo_grounded": False,
            "scored_zero_missing_artifacts": missing,
            "evaluated_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        if write_outputs:
            atomic_write_json(eval_path, result)
            metrics_path = artifact_dir / "metrics.json"
            if metrics_path.exists():
                existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                existing["evaluation"] = result
                atomic_write_json(metrics_path, existing)
        return result

    prompt, task_id, candidate_id, metrics = render_judge_prompt(
        artifact_dir,
        template_path=template_path,
        task_context=task_context,
        repository_context=repository_context,
    )
    token = api_key or os.environ.get("MANTLE_API_KEY")
    if not token:
        raise JudgeError("set MANTLE_API_KEY or pass api_key")

    request_body: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if use_json_response_format:
        request_body["text"] = {
            "format": {
                "type": "json_schema",
                "name": "artifact_evaluation",
                "strict": True,
                "schema": EvaluationResult.model_json_schema(),
            }
        }
    if reasoning_effort is not None:
        request_body["reasoning"] = {"effort": reasoning_effort}

    endpoint = base_url.rstrip("/") + "/responses"
    requester = session or requests
    try:
        response = requester.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        response_payload = response.json()
    except requests.RequestException as exc:
        detail = getattr(getattr(exc, "response", None), "text", "")[:1_000]
        raise JudgeError(
            f"Bedrock judge request failed{f': {detail}' if detail else ''}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise JudgeError(f"Bedrock judge returned invalid HTTP JSON: {exc}") from exc
    if not isinstance(response_payload, dict):
        raise JudgeError("Bedrock judge HTTP response must be a JSON object")

    result = parse_and_validate_result(
        _response_text(response_payload),
        task_id=task_id,
        candidate_id=candidate_id,
    )
    judge: dict[str, Any] = {
        "model": model,
        "provider": "amazon-bedrock-mantle",
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if isinstance(response_payload.get("id"), str):
        judge["response_id"] = response_payload["id"]
    if isinstance(response_payload.get("usage"), dict):
        judge["usage"] = response_payload["usage"]
    result["judge"] = judge

    if write_outputs:
        atomic_write_json(eval_path, result)
        if metrics is not None:
            metrics["evaluation"] = result
            atomic_write_json(artifact_dir / "metrics.json", metrics)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score five SWE artifacts with one Bedrock Mantle Responses request."
    )
    parser.add_argument("--folder", required=True, help="Artifact folder")
    parser.add_argument("--model", required=True, help="Raw Bedrock judge model ID")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BEDROCK_MANTLE_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible Bedrock API base URL",
    )
    parser.add_argument(
        "--api-key",
        help="Bearer token; prefer the MANTLE_API_KEY environment variable",
    )
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE_PATH))
    parser.add_argument(
        "--task-context-file", help="File containing independent task requirements"
    )
    parser.add_argument(
        "--repository-context-file",
        help="File containing independent repository evidence",
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        help="Optional GPT-5-family reasoning effort",
    )
    parser.add_argument(
        "--no-json-response-format",
        action="store_true",
        help="Disable strict JSON Schema output",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail instead of replacing an existing eval.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = evaluate_artifact_folder(
            folder=args.folder,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            template_path=args.template,
            task_context=optional_file(args.task_context_file, "task context"),
            repository_context=optional_file(
                args.repository_context_file, "repository context"
            ),
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds,
            reasoning_effort=args.reasoning_effort,
            use_json_response_format=not args.no_json_response_format,
            overwrite=not args.no_overwrite,
        )
    except JudgeError as exc:
        logger.error("%s", exc)
        return 1

    eval_path = Path(args.folder).expanduser().resolve() / "eval.json"
    logger.info("wrote %s (task_score=%.2f)", eval_path, result["task_score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
