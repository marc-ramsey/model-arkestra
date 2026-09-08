"""Arkestra CLI — thin RPC client for ModelArkestra.

Usage:
    arkestra models                # list cached models (name, size)
    arkestra start -m qwen3-4b     # start a model
    arkestra stop -m qwen3-4b      # stop a running model
    arkestra restart -m qwen3-4b   # restart with config-as-is
    arkestra chat -m qwen3-4b      # interactive chat

All endpoints require Authorization: Bearer <key> if api_key or admin_key is configured.
Both keys are accepted interchangeably — either grants access to /api/* routes.

Chat client types ``/help`` for commands, ``/quit`` or ``Ctrl+D`` to exit.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp is required. Install with: pip install \"model-arkestra[proxy]\"", file=sys.stderr)
    sys.exit(1)

DEFAULT_SYS_PROMPT = "You are a helpful assistant."
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "arkestra"


# ── Auth helpers ──────────────────────────────────────────────────────

def _read_api_key(config_path: str | None = None) -> str | None:
    """Read api_key or admin_key from config. Returns the first non-empty one."""
    path = config_path or os.environ.get("ARKESTRA_CONFIG") or str(DEFAULT_CONFIG_DIR / "config.yaml")
    try:
        import yaml
        data = yaml.safe_load(Path(path).read_text()) or {}
    except Exception:
        return None

    default = data.get("default") or {}
    key = default.get("api_key") or data.get("api_key") or \
          default.get("admin_key") or data.get("admin_key")
    return str(key) if key else None


# ── HTTP helpers ──────────────────────────────────────────────────────

async def _request(server_url: str, path: str, method: str = "GET",
                   json_body: dict | None = None, api_key: str | None = None) -> dict:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = server_url.rstrip("/") + path
    try:
        async with aiohttp.ClientSession() as session:
            kwargs: dict = {"headers": headers}
            if json_body is not None:
                kwargs["json"] = json_body
            async with session.request(method, url, **kwargs) as resp:
                return await resp.json()
    except Exception as exc:
        print(f"Server error: {exc}", file=sys.stderr)
        sys.exit(1)


def _resolve_server_url() -> str | None:
    """Try config then env, returns server URL or None."""
    config_path = os.environ.get("ARKESTRA_CONFIG") or \
                  str(DEFAULT_CONFIG_DIR / "config.yaml")
    try:
        import yaml
        data = yaml.safe_load(Path(config_path).read_text()) or {}
    except Exception:
        return None

    default = data.get("default") or data
    port = default.get("admin-port") or data.get("admin-port", 8080)
    host = os.environ.get("ARKESTRA_HOST") or "127.0.0.1"
    return f"http://{host}:{port}"


# ── Server-mode commands ─────────────────────────────────────────────

async def cmd_models(args):
    server = _resolve_server_url()
    if not server:
        print("No server URL found. Set ARKESTRA_CONFIG or config.yaml default.admin-port", file=sys.stderr)
        sys.exit(1)

    key = _read_api_key(args.config if hasattr(args, 'config') and args.config else None)
    data = await _request(server, "/api/models", api_key=key)
    models = data.get("models", [])
    if not models:
        print("No cached models.")
        return

    # Filter by -m if given
    name_filter = getattr(args, "model", None)
    models = [m for m in models if not name_filter or m["name"] == name_filter]

    header = f"{'NAME':<30} {'MODEL':<40} {'SIZE':>6}"
    print(header)
    print("-" * len(header))
    for m in models:
        model_ref = m.get("model", "-") or "-"
        size = f"{m.get('size', 0):.1f} GB" if m.get("size") else "-"
        print(f"{m['name']:<30} {model_ref:<40} {size:>6}")


async def cmd_start(args):
    server = _resolve_server_url()
    if not server:
        print("No server URL found.", file=sys.stderr)
        sys.exit(1)

    key = _read_api_key(getattr(args, 'config', None))
    result = await _request(server, f"/api/start/{args.model}", "POST", api_key=key)
    if result.get("ok"):
        port = result.get("port")
        print(f"Model '{args.model}' started (port {port})")
    else:
        print(result.get("detail", "Failed to start"), file=sys.stderr)


async def cmd_stop(args):
    server = _resolve_server_url()
    if not server:
        print("No server URL found.", file=sys.stderr)
        sys.exit(1)

    key = _read_api_key(getattr(args, 'config', None))
    result = await _request(server, f"/api/stop/{args.model}", "POST", api_key=key)
    if result.get("ok"):
        prev = result.get("previous_state", "unknown")
        print(f"Model '{args.model}' stopped (was {prev})")
    else:
        print(result.get("detail", "Failed to stop"), file=sys.stderr)


async def cmd_restart(args):
    server = _resolve_server_url()
    if not server:
        print("No server URL found.", file=sys.stderr)
        sys.exit(1)

    key = _read_api_key(getattr(args, 'config', None))
    result = await _request(server, f"/api/restart/{args.model}", "POST", api_key=key)
    if result.get("ok"):
        port = result.get("port")
        print(f"Model '{args.model}' restarted (port {port})")
    else:
        print(result.get("detail", "Failed to restart"), file=sys.stderr)


# ── Chat command ──────────────────────────────────────────────────────

async def chat_server(server_url: str, model: str, api_key: str | None = None) -> None:
    """Chat via an ArkestraServer server."""
    url = f"{server_url.rstrip('/')}/v1/chat/completions"

    print(f"\nConnected to server at {url}")
    print("Type /help for commands, /quit to exit.\n")

    history: list[dict] = [{"role": "system", "content": DEFAULT_SYS_PROMPT}]
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with aiohttp.ClientSession(headers=headers) as session:
        await _chat_loop(session, url, model, history)


async def _chat_loop(session, target: str, history: list[dict]) -> None:
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

        history.append({"role": "user", "content": user_input})
        await _send_server(session, target, history)


async def _send_server(session, model_name: str, history: list[dict]) -> None:
    """Send streaming request via server."""
    print(f"[{model_name}] ", end="", flush=True)
    try:
        async with session.post("/v1/chat/completions", json={
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

        print("\n")
    except aiohttp.ClientError as e:
        print(f"\nConnection error: {e}", file=sys.stderr)
        history.pop()


# ── Helpers ───────────────────────────────────────────────────────────

async def _prompt() -> str | None:
    try:
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: input("\n> ")
        )
    except EOFError:
        return None


def _parse_command(text: str) -> str | None:
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
        return stripped
    return None


def _print_help() -> None:
    print("""
Commands:
  /help          Show this message
  /quit          Exit the chat
  /clear         Clear conversation history
  /history       Show full conversation history
  /system <txt>  Change the system prompt
""")


# ── Entry point ───────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arkestra",
        description="ModelArkestra — chat client and model management.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── models ────────────────────────────────────────────────────────
    m_parser = subparsers.add_parser("models", help="List cached models (name, size)")
    m_parser.add_argument("-m", "--model", default=None, help="Filter by model name")
    m_parser.add_argument("--config", "-c", default=None, help="Config file path")

    # ── start ─────────────────────────────────────────────────────────
    s_parser = subparsers.add_parser("start", help="Start a model")
    s_parser.add_argument("-m", "--model", required=True, help="Model name")
    s_parser.add_argument("--config", "-c", default=None, help="Config file path")

    # ── stop ──────────────────────────────────────────────────────────
    p_parser = subparsers.add_parser("stop", help="Stop a running model")
    p_parser.add_argument("-m", "--model", required=True, help="Model name")
    p_parser.add_argument("--config", "-c", default=None, help="Config file path")

    # ── restart ───────────────────────────────────────────────────────
    r_parser = subparsers.add_parser("restart", help="Restart a model (config-as-is)")
    r_parser.add_argument("-m", "--model", required=True, help="Model name")
    r_parser.add_argument("--config", "-c", default=None, help="Config file path")

    # ── chat ──────────────────────────────────────────────────────────
    c_parser = subparsers.add_parser("chat", help="Interactive chat client")
    chat_group = c_parser.add_mutually_exclusive_group(required=False)
    chat_group.add_argument("--config", "-c", default=None, help="YAML config file (direct mode)")
    chat_group.add_argument("--server", "-x", default=None, help="Server URL")
    c_parser.add_argument("-m", "--model", required=True, help="Model name")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        # No subcommand — default to chat with --model (backwards compat if --model is set)
        print("Error: subcommand required (models, start, stop, restart, chat)", file=sys.stderr)
        sys.exit(1)

    dispatch = {
        "models": cmd_models,
        "start": cmd_start,
        "stop": cmd_stop,
        "restart": cmd_restart,
        "chat": cmd_chat,
    }

    handler = dispatch.get(args.command)
    if handler:
        asyncio.run(handler(args))


def cmd_chat(args):
    """Route chat to direct or server mode."""
    server_url = args.server  # None = direct mode (not used in thin client)

    key = _read_api_key(args.config)

    if server_url:
        asyncio.run(chat_server(server_url, args.model, api_key=key))
    else:
        print("Error: chat requires --server URL. For direct mode use arkestra-server directly.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
