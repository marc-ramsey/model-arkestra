#!/usr/bin/env python3
"""CLI client for ModelArkestra — interactive or one-shot prompt mode."""

import argparse
import asyncio
import sys
import os

from model_arkestra.arkestra import ModelArkestra


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chat with a local LLM via ModelArkestra."
    )
    p.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "sample-config.yaml"),
        help="Path to the config file (default: ./sample-config.yaml)",
    )
    p.add_argument(
        "--model", "-m",
        default="qwen3.5-4b",
        help="Model name to use from config (default: qwen3.5-4b)",
    )
    p.add_argument(
        "--backend", "-b",
        default=None,
        help="Backend id override (e.g. rocm, radv)",
    )
    p.add_argument(
        "--container", "-c",
        choices=["process", "podman", "docker"],
        default=None,
        help="Runner type override",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="Explicit port (bypasses auto-allocation)",
    )
    p.add_argument(
        "--prompt", "-p",
        default=None,
        help="One-shot prompt — exits after response",
    )
    args, unknown = p.parse_known_args()
    # Everything unrecognized is passed through as model params
    args.model_kwargs = unknown
    return args


def parse_extra_kwargs(raw: list[str]) -> dict:
    """Convert ['temp', '0.7', 'top-p', '0.95'] to {'temp': 0.7, 'top-p': 0.95}."""
    kwargs: dict = {}
    i = 0
    while i + 1 < len(raw):
        key = raw[i]
        val = raw[i + 1]
        # Try numeric conversion
        try:
            if "." in val:
                val = float(val)
            else:
                val = int(val)
        except ValueError:
            pass  # keep as string
        kwargs[key] = val
        i += 2
    return kwargs


async def chat(
    arkestra: ModelArkestra,
    model_name: str,
    prompt: str,
    extra_kwargs: dict | None = None,
) -> None:
    """Run a single prompt through the model using streaming."""
    args = extra_kwargs or {}
    payload = {"prompt": prompt}
    payload.update(args)

    print(f"[{model_name}] ", end="", flush=True)
    full_text = ""
    total_tokens = 0
    async for chunk in arkestra.astream(model_name, payload):
        if "token" in chunk:
            token = chunk["token"]
            print(token, end="", flush=True)
            full_text += token
        elif "usage" in chunk:
            usage = chunk["usage"]
            total_tokens = usage.get("completion_tokens", 0)
    print(f"\n[{total_tokens} tokens]")


async def repl(arkestra: ModelArkestra, model_name: str, extra_kwargs: dict) -> None:
    """Interactive REPL loop."""
    print(f"Connected to {model_name}. Type 'quit' or Ctrl+D to exit.\n")
    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            continue
        if prompt.strip().lower() in ("quit", "exit"):
            break
        await chat(arkestra, model_name, prompt, extra_kwargs)


async def main() -> None:
    args = parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    arkestra = ModelArkestra(config_path)

    extra_kwargs = parse_extra_kwargs(args.model_kwargs) if args.model_kwargs else {}

    await arkestra.start(args.model, backend=args.backend, container=args.container, port=args.port)

    try:
        if args.prompt is not None:
            # One-shot mode
            await chat(arkestra, args.model, args.prompt, extra_kwargs)
        else:
            # Interactive REPL
            await repl(arkestra, args.model, extra_kwargs)
    finally:
        await arkestra.stop_all()


if __name__ == "__main__":
    asyncio.run(main())
