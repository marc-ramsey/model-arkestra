"""Arkestra CLI — user-facing client for ModelArkestra.

Usage:
    arkestra models                # list cached models (name, size)
    arkestra chat -m qwen3-4b      # interactive chat (auto-starts if stopped)
    arkestra init [--force]        # scaffold config.yaml + backends.yaml

Connection resolution is shared with arkestra-server via model_arkestra.conn:
    CLI flag  >  ARKESTRA_* env  >  config.yaml  >  default

Chat types ``/help`` for in-loop commands, ``/quit`` or ``Ctrl+D`` to exit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp is required. Install with: pip install \"model-arkestra[proxy]\"", file=sys.stderr)
    sys.exit(1)

from model_arkestra.conn import add_common_args, resolve_conn
from model_arkestra.common import config_dir, resolve_config_path

DEFAULT_SYS_PROMPT = "You are a helpful assistant."

# Inference params mutable per-request (flags + in-loop /-commands).
INFERENCE_PARAMS = (
    ("temperature", float),
    ("top_p", float),
    ("max_tokens", int),
    ("frequency_penalty", float),
    ("presence_penalty", float),
)


# ── HTTP helpers ──────────────────────────────────────────────────────

async def _request(conn, path: str, method: str = "GET",
                   json_body: dict | None = None) -> dict:
    headers = {"Accept": "application/json"}
    if conn.api_key:
        headers["Authorization"] = f"Bearer {conn.api_key}"

    url = conn.url_for(path)
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


def _load_config_data(config_path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(Path(config_path).read_text()) or {}
    except Exception:
        return {}


def _read_api_key(config_path: Path) -> str | None:
    """Read api_key or admin_key from config for /api/* auth."""
    data = _load_config_data(config_path)
    default = data.get("default") or {}
    default_env = data.get("default-env") or {}
    key = (default_env.get("api-key") or default_env.get("api_key")
           or default.get("api_key") or data.get("api_key")
           or default_env.get("admin-key") or default_env.get("admin_key")
           or default.get("admin_key") or data.get("admin_key"))
    return str(key) if key else None


def _make_conn(args) -> "object":
    """Resolve the shared connection, falling back to config keys for auth."""
    cfg = _load_config_data(resolve_config_path(getattr(args, "config", None)))

    def cfg_get(path: str, default=None):
        parts = path.split("/")
        node = cfg
        for p in parts:
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    conn = resolve_conn(args, server=False, cfg_get=cfg_get)
    api_key = conn.api_key or _read_api_key(conn.config_path)
    if api_key == conn.api_key:
        return conn
    return conn.__class__(host=conn.host, port=conn.port, config_path=conn.config_path,
                          base_path=conn.base_path, api_key=api_key)


# ── models ────────────────────────────────────────────────────────────

async def cmd_models(args):
    conn = _make_conn(args)
    data = await _request(conn, "/api/models")
    models = data.get("models", [])
    if not models:
        print("No cached models.")
        return

    name_filter = getattr(args, "model", None)
    models = [m for m in models if not name_filter or m["name"] == name_filter]

    header = f"{'NAME':<30} {'MODEL':<40} {'SIZE':>6}"
    print(header)
    print("-" * len(header))
    for m in models:
        model_ref = m.get("model", "-") or "-"
        size = f"{m.get('size', 0):.1f} GB" if m.get("size") else "-"
        print(f"{m['name']:<30} {model_ref:<40} {size:>6}")


# ── chat ──────────────────────────────────────────────────────────────

def _params_from_args(args) -> dict:
    """Collect inference params from chat flags (None values dropped)."""
    params = {}
    for name, _ in INFERENCE_PARAMS:
        val = getattr(args, name, None)
        if val is not None:
            params[name] = val
    stop = getattr(args, "stop", None)
    if stop:
        params["stop"] = stop
    return params


async def cmd_chat(args):
    conn = _make_conn(args)
    params = _params_from_args(args)
    url = conn.url_for("/v1/chat/completions")

    print(f"\nConnected to {url}")
    print("Type /help for commands, /quit to exit.\n")

    history: list[dict] = [{"role": "system", "content": DEFAULT_SYS_PROMPT}]
    headers = {"Accept": "application/json"}
    if conn.api_key:
        headers["Authorization"] = f"Bearer {conn.api_key}"

    async with aiohttp.ClientSession(headers=headers) as session:
        await _chat_loop(session, url, args.model, history, params)


async def _chat_loop(session, target: str, model: str,
                     history: list[dict], params: dict) -> None:
    while True:
        user_input = await _prompt()
        if not user_input:
            break

        handled = _handle_command(user_input, history, params)
        if handled is not None:
            if handled == "quit":
                print("Goodbye.")
                return
            continue

        history.append({"role": "user", "content": user_input})
        await _send(session, target, model, history, params)


def _handle_command(text: str, history: list[dict], params: dict) -> str | None:
    """Process a /-command. Returns 'quit' to exit, None to send as a message."""
    stripped = text.strip()

    if stripped in ("/quit", "/exit"):
        return "quit"
    if stripped == "/help":
        _print_help(params)
        return "handled"
    if stripped == "/clear":
        history.clear()
        history.append({"role": "system", "content": DEFAULT_SYS_PROMPT})
        print("(history cleared)\n")
        return "handled"
    if stripped == "/history":
        for m in history:
            role = m["role"].upper()
            preview = m["content"][:80] + ("..." if len(m["content"]) > 80 else "")
            print(f"  [{role}] {preview}")
        print()
        return "handled"
    if stripped.startswith("/system"):
        if not history:
            history.append({"role": "system", "content": stripped[7:].strip()})
        else:
            history[0] = {"role": "system", "content": stripped[7:].strip()}
        print("System prompt updated.\n")
        return "handled"

    # Inference param commands: /temperature 0.7, /top-p 0.9, ...
    if stripped.startswith("/"):
        parts = stripped[1:].strip().split(None, 1)
        name = parts[0].replace("-", "_")
        value = parts[1].strip() if len(parts) > 1 else ""
        for flag_name, _ in INFERENCE_PARAMS:
            if name == flag_name:
                if not value:
                    print(f"Current {flag_name}: {params.get(flag_name, 'default')}")
                    return "handled"
                try:
                    params[flag_name] = float(value)
                except ValueError:
                    print(f"Invalid number: {value}")
                    return "handled"
                print(f"{flag_name} = {params[flag_name]}\n")
                return "handled"
        if name == "stop":
            if not value:
                print(f"Current stop: {params.get('stop', 'default')}")
                return "handled"
            params["stop"] = value
            print(f"stop = {value!r}\n")
            return "handled"

    return None


async def _send(session, target: str, model: str,
                history: list[dict], params: dict) -> None:
    """Send a streaming request, printing tokens as they arrive."""
    payload = {"model": model, "messages": history, "stream": True, **params}
    print(f"[{model}] ", end="", flush=True)
    try:
        async with session.post(target, json=payload) as resp:
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


async def _prompt() -> str | None:
    try:
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: input("\n> ")
        )
    except EOFError:
        return None


def _print_help(params: dict) -> None:
    lines = [
        "Commands:",
        "  /help          Show this message",
        "  /quit          Exit the chat",
        "  /clear         Clear conversation history",
        "  /history       Show full conversation history",
        "  /system <txt>  Change the system prompt",
        "",
        "Inference params (no value shows current):",
    ]
    for name, _ in INFERENCE_PARAMS:
        current = params.get(name, "default")
        lines.append(f"  /{name.replace('_', '-')} <n>   (current: {current})")
    lines.append(f"  /stop <txt>     (current: {params.get('stop', 'default')})")
    print("\n".join(lines) + "\n")


# ── init ──────────────────────────────────────────────────────────────

def cmd_init(args) -> None:
    """Scaffold config.yaml + backends.yaml from bundled templates."""
    import jinja2
    from importlib.resources import files
    from model_arkestra.gpu_detect import detect_all

    templates = files("model_arkestra.templates")
    out_dir = config_dir()
    targets = {
        out_dir / "config.yaml": (templates / "config.yaml.j2").read_text(),
        out_dir / "backends.yaml": (templates / "backends.yaml.j2").read_text(),
    }

    if not args.force:
        existing = [str(p) for p in targets if p.exists()]
        if existing:
            print(f"Refusing to overwrite: {', '.join(existing)} (use --force)", file=sys.stderr)
            sys.exit(1)

    # Detect hardware and set the default backend from the recommendation.
    try:
        hw = detect_all()
    except Exception:
        hw = {}
    recommendation = hw.get("recommendation")

    out_dir.mkdir(parents=True, exist_ok=True)
    for path, text in targets.items():
        if path.name == "config.yaml" and recommendation:
            backend, reason = recommendation
            text = _set_default_backend(text, backend)
            text += f"\n# init: default backend '{backend}' — {reason}\n"
        path.write_text(text)
        print(f"Wrote {path}")

    if recommendation:
        print(f"Default backend: {recommendation[0]} ({recommendation[1]})")
    print("\nNext: edit config.yaml, then run arkestra-server.")


def _set_default_backend(text: str, backend: str) -> str:
    """Rewrite the ``backends: default:`` line in rendered config text."""
    lines = text.splitlines()
    in_backends = False
    for i, line in enumerate(lines):
        if line.startswith("backends:"):
            in_backends = True
            continue
        if in_backends:
            if line.startswith("  default:"):
                lines[i] = f"  default: {backend}"
                break
            if not line.startswith("  "):
                break
    return "\n".join(lines) + "\n"


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
    add_common_args(m_parser)

    # ── chat ──────────────────────────────────────────────────────────
    c_parser = subparsers.add_parser("chat", help="Interactive chat client")
    c_parser.add_argument("-m", "--model", required=True, help="Model name")
    c_parser.add_argument("-T", "--temperature", type=float, default=None,
                          help="Sampling temperature")
    c_parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling")
    c_parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate")
    c_parser.add_argument("--frequency-penalty", type=float, default=None,
                          help="Frequency penalty")
    c_parser.add_argument("--presence-penalty", type=float, default=None,
                          help="Presence penalty")
    c_parser.add_argument("--stop", default=None, help="Stop sequence")
    add_common_args(c_parser)

    # ── init ──────────────────────────────────────────────────────────
    i_parser = subparsers.add_parser("init", help="Scaffold config.yaml + backends.yaml")
    i_parser.add_argument("--force", action="store_true",
                          help="Overwrite existing files")
    add_common_args(i_parser)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "init":
        cmd_init(args)
        return

    dispatch = {"models": cmd_models, "chat": cmd_chat}
    asyncio.run(dispatch[args.command](args))


if __name__ == "__main__":
    main()
