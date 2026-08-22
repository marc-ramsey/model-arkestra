"""Arkestra CLI — chat client and init scaffolding.

Usage:
    arkestra-cli --config config.yaml --model qwen3-4b          # direct (connects to model ports)
    arkestra-cli --server http://localhost:8080 --model gpt-4     # via server
    arkestra-cli init                                           # scaffold default config files

Chat client types ``/help`` for commands, ``/quit`` or ``Ctrl+D`` to exit.

Supports streaming output and full multi-turn conversation history.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

try:
    from importlib.resources import files as resources_files
except ImportError:
    from importlib_resources import files as resources_files

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp is required. Install with: pip install \"model-arkestra[proxy]\"", file=sys.stderr)
    sys.exit(1)


DEFAULT_SYS_PROMPT = "You are a helpful assistant."


# ── Direct mode ─────────────────────────────────────────────────────

async def chat_direct(config_path: str, model_name: str, broadcast_addr: str) -> None:
    """Chat directly against ModelArkestra — no server needed."""
    from model_arkestra.arkestra import ModelArkestra

    async with ModelArkestra(config_path, broadcast_addr=broadcast_addr) as runner:
        await runner.start(model_name)

        # Grab the allocated port for display
        ctx = runner._get_model_contexts()[0] if runner._get_model_contexts() else None
        port_str = f" (port {ctx.port})" if ctx else ""
        print(f"\nChatting with {model_name}{port_str}")
        print("Type /help for commands, /quit to exit.\n")

        history: list[dict] = [{"role": "system", "content": DEFAULT_SYS_PROMPT}]
        await _chat_loop(runner, model_name, history)


# ── Proxy mode ───────────────────────────────────────────────────────

async def chat_server(server_url: str, model: str) -> None:
    """Chat via an ArkestraServer server."""
    url = f"{server_url.rstrip('/')}/v1/chat/completions"

    print(f"\nConnected to server at {url}")
    print("Type /help for commands, /quit to exit.\n")

    history: list[dict] = [{"role": "system", "content": DEFAULT_SYS_PROMPT}]
    async with aiohttp.ClientSession() as session:
        await _chat_loop(session, url, model, history)


# ── Shared chat loop ─────────────────────────────────────────────────

async def _chat_loop(runner_or_session, target: str | str, history: list[dict]) -> None:
    """Common interactive chat loop — parameterised for direct vs server."""
    while True:
        user_input = await _prompt()
        if not user_input:
            break

        cmd = _parse_command(user_input)
        if cmd == "quit":
            print("Goodbye.")
            return
        if cmd == "help":
            _print_help()
            continue
        if cmd == "clear":
            history.clear()
            history.append({"role": "system", "content": DEFAULT_SYS_PROMPT})
            print("(history cleared)\n")
            continue
        if cmd == "history":
            for m in history:
                role = m["role"].upper()
                preview = m["content"][:80] + ("..." if len(m["content"]) > 80 else "")
                print(f"  [{role}] {preview}")
            print()
            continue

        new_system = cmd[9:].strip() if cmd.startswith("/system ") else None
        if new_system:
            history[0] = {"role": "system", "content": new_system}
            print("System prompt updated.\n")
            continue

        # ── Normal chat message ───────────────────────────────────────
        is_direct = hasattr(runner_or_session, 'ainvoke')
        history.append({"role": "user", "content": user_input})

        if is_direct:
            await _send_direct(runner_or_session, target, history)
        else:
            await _send_server(runner_or_session, target, model_name=target, history=history)


async def _send_direct(runner, model_name: str, history: list[dict]) -> None:
    """Send message via direct ModelArkestra runner."""
    try:
        result = await runner.ainvoke(model_name, messages=history)
        print(f"\n[{model_name}] {result}\n")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        history.pop()  # remove failed user message


async def _send_server(session, url: str, model_name: str, history: list[dict]) -> None:
    """Send streaming request via server."""
    print(f"[{model_name}] ", end="", flush=True)
    try:
        async with session.post(url, json={
            "model": model_name,
            "messages": history,
            "stream": True,
        }) as resp:
            if resp.status != 200:
                err = await resp.text()
                print(f"\nError {resp.status}: {err}\n")
                history.pop()
                return

            async for line in resp.content:
                line = line.strip()
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    token = chunk["choices"][0]["delta"].get("content", "")
                    print(token, end="", flush=True)
                except (json.JSONDecodeError, KeyError):
                    pass

        print("\n")  # newline after response
    except aiohttp.ClientError as e:
        print(f"\nConnection error: {e}", file=sys.stderr)
        history.pop()


# ── Helpers ───────────────────────────────────────────────────────────

async def _prompt() -> str | None:
    """Read user input. Returns None on EOF."""
    try:
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: input("\n> ")
        )
    except EOFError:
        return None


def _parse_command(text: str) -> str | None:
    """Return command string if input is a slash-command, else None."""
    stripped = text.strip()
    if stripped in ("/quit", "/exit"):
        return "quit"
    if stripped == "/help":
        return "help"
    if stripped == "/clear":
        return "clear"
    if stripped == "/history":
        return "history"
    if stripped.startswith("/system "):
        return stripped  # caller handles the rest
    return None


def _print_help() -> None:
    print("""
Commands:
  /help          Show this message
  /quit          Exit the chat
  /clear         Clear conversation history
  /history       Show full conversation history
  /system <txt>  Change the system prompt

Chat preserves full multi-turn history — every turn is sent to the model.
""")


# ═══════════════════════════════════════════════════════════
# Init command — scaffold default config files
# ═══════════════════════════════════════════════════════════
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "arkestra"
TEMPLATE_FILES = ["config.yaml.j2", "sources.yaml.j2"]


def cmd_init(force: bool = False) -> int:
    """Scaffold default config files in ~/.config/arkestra/.

    Returns 0 on success, 1 on error.
    """
    config_dir = DEFAULT_CONFIG_DIR
    config_file = config_dir / "config.yaml"
    sources_file = config_dir / "sources.yaml"

    # Check for existing files
    if not force and (config_file.exists() or sources_file.exists()):
        print(f"Config directory already exists: {config_dir}", file=sys.stderr)
        print("Files present:  ", end="")
        existing = [f.name for f in config_dir.iterdir() if f.is_file()]
        print(", ".join(existing) or "(none)")
        print("\nUse --force to overwrite existing files.", file=sys.stderr)
        return 1

    # Create directory and copy templates
    config_dir.mkdir(parents=True, exist_ok=True)
    src_pkg = resources_files("model_arkestra.templates")
    
    for template_name in TEMPLATE_FILES:
        dest = config_dir / template_name.replace(".j2", "")
        try:
            # Read from package resource
            content = (src_pkg / template_name).read_text()
            dest.write_text(content)
            print(f"Created {dest}")
        except Exception as exc:
            print(f"Error creating {template_name}: {exc}", file=sys.stderr)
            return 1

    print(f"\nConfig files scaffolded to {config_dir}")
    print("Edit config.yaml to add models and backends.")
    print("See sample-config.yaml in the repository for a complete reference.")
    return 0


def cmd_init_wrapper() -> None:
    """Convenience entry point: arkestra-init [--force]."""
    import argparse
    parser = argparse.ArgumentParser(prog="arkestra-init")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite existing config files")
    args, _ = parser.parse_known_args()
    sys.exit(cmd_init(force=args.force))


# ── Entry point ───────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Entry point for arkestra-cli console script.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
              Useful for testing or embedding.
    """
    parser = argparse.ArgumentParser(
        prog="arkestra-cli",
        description="ModelArkestra — chat client, init, and more.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── init subcommand ───────────────────────────────────────────────
    init_parser = subparsers.add_parser(
        "init",
        help="Scaffold default config files in ~/.config/arkestra/",
    )
    init_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing config files",
    )

    # ── chat subcommand (default) ─────────────────────────────────────
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive chat client for ModelArkestra models",
    )
    chat_group = chat_parser.add_mutually_exclusive_group(required=True)
    chat_group.add_argument("--config", "-c", help="YAML config file path (direct mode)")
    chat_group.add_argument("--server", "-x", default=None, help="Server URL")
    chat_parser.add_argument("--model", "-m", required=True, help="Model name to chat with")
    chat_parser.add_argument("--broadcast-addr", default=None,
                             help='Broadcast address for models (default: 0.0.0.0)')

    args = parser.parse_args(argv)

    # ── Route to subcommand handler ───────────────────────────────────
    if args.command == "init":
        sys.exit(cmd_init(force=args.force))

    # Default: chat subcommand (for backwards compat with no subcommand)
    if hasattr(args, 'config'):
        broadcast_addr = args.broadcast_addr or "0.0.0.0"
        if args.server:
            asyncio.run(chat_server(args.server, args.model))
        else:
            asyncio.run(chat_direct(args.config, args.model, broadcast_addr))


