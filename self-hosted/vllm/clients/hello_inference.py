#!/usr/bin/env python3
"""Minimal inference client for a self-hosted vLLM endpoint.

Sends one chat completion to the OpenAI-compatible API vLLM exposes and prints
the reply plus token usage. This is the "does inference work end to end from
Python" smoke test that pairs with scripts/vllm-verify.sh (which uses curl).

Usage:
    # after `uv sync` in self-hosted/vllm/ (see README):
    uv run clients/hello_inference.py
    uv run clients/hello_inference.py --prompt "Write a bubble sort in Rust."
    BASE_URL=http://localhost:8000/v1 MODEL=qwen3-coder-30b uv run clients/hello_inference.py

Environment variables (CLI flags take precedence):
    BASE_URL   vLLM endpoint (default: http://127.0.0.1:8000/v1)
    MODEL      served-model-name to target (default: qwen3-coder-30b)
    API_KEY    ignored by vLLM, but the SDK requires a non-empty value
"""

import argparse
import logging
import os
import sys
import time

from openai import OpenAI, OpenAIError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "qwen3-coder-30b"
DEFAULT_API_KEY = "not-needed"  # vLLM ignores it; the SDK just needs a non-empty value
DEFAULT_PROMPT = "Write a Python function that returns the nth Fibonacci number."
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments. CLI flags override environment variables."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run clients/hello_inference.py\n"
            '  uv run clients/hello_inference.py --stream --prompt "Explain the GIL"\n'
            "  BASE_URL=http://localhost:8000/v1 MODEL=qwen3-coder-30b \\\n"
            "      uv run clients/hello_inference.py\n"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BASE_URL", DEFAULT_BASE_URL),
        help="vLLM OpenAI-compatible endpoint (env: BASE_URL)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", DEFAULT_MODEL),
        help="served-model-name to target (env: MODEL)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="user prompt to send",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="sampling temperature",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="stream tokens as they arrive",
    )
    return parser.parse_args()


def _stream_reply(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> None:
    """Stream a chat completion, printing chunks to stdout as they arrive.

    Args:
        client: Configured OpenAI-compatible client.
        model: served-model-name to target.
        prompt: User prompt to send.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
    """
    start = time.time()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    chunks = 0
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        chunks += 1 if delta else 0
    elapsed = time.time() - start
    rate = chunks / elapsed if elapsed > 0 else 0.0
    print(f"\n\n[streamed ~{chunks} chunks in {elapsed:.2f}s ~ {rate:.1f} chunks/sec]")


def _request_reply(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> None:
    """Send a single (non-streamed) chat completion and print the reply and usage.

    Args:
        client: Configured OpenAI-compatible client.
        model: served-model-name to target.
        prompt: User prompt to send.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
    """
    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = time.time() - start

    print("--- reply ------------------------------------")
    print(resp.choices[0].message.content)
    print("----------------------------------------------")

    usage = resp.usage
    if usage is None:
        logger.warning("Response contained no usage block; skipping token report.")
        return

    print(f"prompt tokens:     {usage.prompt_tokens}")
    print(f"completion tokens: {usage.completion_tokens}")
    print(f"wall clock:        {elapsed:.2f}s")
    if usage.completion_tokens and elapsed > 0:
        rate = usage.completion_tokens / elapsed
        print(
            f"single-request:    {rate:.1f} tokens/sec (not the batched throughput number)"
        )


def main() -> None:
    """Parse arguments, build the client, and run one chat completion."""
    args = _parse_args()
    api_key = os.environ.get("API_KEY", DEFAULT_API_KEY)

    logger.info("endpoint: %s", args.base_url)
    logger.info("model:    %s", args.model)
    logger.info("prompt:   %s", args.prompt)

    client = OpenAI(base_url=args.base_url, api_key=api_key)

    try:
        if args.stream:
            _stream_reply(
                client, args.model, args.prompt, args.max_tokens, args.temperature
            )
        else:
            _request_reply(
                client, args.model, args.prompt, args.max_tokens, args.temperature
            )
    except OpenAIError as exc:
        logger.error(
            "Inference request to %s failed: %s. Is the vLLM server up and serving "
            "model '%s'? Start it with scripts/vllm-serve.sh.",
            args.base_url,
            exc,
            args.model,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
