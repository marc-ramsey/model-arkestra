#!/usr/bin/env python3
"""Arkestra Admin CLI — HTTP client for the ModelArkestra admin API.

Usage:
    arkestra-admin models -x http://localhost:8080
    arkestra-admin start qwen3-4b temp=0.7 --api-key secret
    arkestra-admin logs qwen3-4b --lines 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

try:
    from aiohttp import ClientSession, ClientTimeout
except ImportError:
    print("Error: aiohttp required. Install with: pip install \"model-arkestra[proxy]\"", file=sys.stderr)
    sys.exit(1)


# ── Config reading (lightweight, no ModelArkestra instance needed) ─────────

def _read_admin_key(config_path: str | None = None) -> str | None:
    """Read ADMIN_KEY from config.yaml env section or return None."""
    try:
        import yaml
    except ImportError:
        return None

    path = config_path or os.path.expanduser("~/.config/arkestra/config.yaml")
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        env = data.get("env") or {}
        return env.get("ADMIN_KEY")
    except (FileNotFoundError, OSError):
        return None


# ── HTTP helpers ───────────────────────────────────────────────────────

async def _request(
    method: str,
    server_url: str,
    path: str,
    *,
    api_key: str | None = None,
    json_body: dict | None = None,
) -> Dict[str, Any]:
    """Execute an admin API request and return parsed JSON."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-admin-key"] = api_key

    url = server_url.rstrip("/") + path
    timeout = ClientTimeout(total=30)

    async with ClientSession(timeout=timeout) as session:
        kwargs: Dict[str, Any] = {"headers": headers}
        if json_body is not None:
            kwargs["json"] = json_body

        try:
            async with session.request(method, url, **kwargs) as resp:
                data = await resp.json()
                if resp.status >= 500:
                    print(f"Server error {resp.status}: {data.get('detail', data)}", file=sys.stderr)
                    sys.exit(1)
                return data
        except Exception as exc:
            print(f"Request failed: {exc}", file=sys.stderr)
            sys.exit(1)


def _print_json(data: Any) -> None:
    """Print compact JSON to stdout."""
    print(json.dumps(data, indent=2, default=str))


# ── Subcommand handlers ───────────────────────────────────────────────

async def cmd_models(args: argparse.Namespace) -> None:
    data = await _request("GET", args.server, "/admin/models", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(data)
        return

    models = data.get("models", [])
    if not models:
        print("No models in config.")
        return

    header = f"{'NAME':<30} {'STATUS':<12} {'PORT':<8} {'BACKEND':<20} {'RUNNER'}"
    print(header)
    print("-" * len(header))
    for m in models:
        runner = m.get("runner_type") or ""
        if isinstance(runner, str):
            runner = runner.replace("runnerstate.", "").lower()
        print(f"{m['id']:<30} {m['status']:<12} {str(m['port']) or '-':<8} {str(m.get('backend_id') or '-'):<20} {runner}")


async def cmd_start(args: argparse.Namespace) -> None:
    body: Dict[str, Any] = {}
    if args.port:
        body["port"] = args.port
    if args.backend:
        body["backend"] = args.backend
    if args.runner:
        body["runner"] = args.runner
    for kv in args.args or []:
        key, _, value = kv.partition("=")
        # Try to coerce numeric types
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        body[key] = value

    result = await _request("POST", args.server, f"/admin/start/{args.name}", api_key=args.api_key, json_body=body)
    if getattr(args, "json", False):
        _print_json(result)
    else:
        port = result.get("port")
        status = "ok" if result.get("ok") else "failed"
        print(f"Model '{args.name}' {status}" + (f" on port {port}" if port else ""))


async def cmd_stop(args: argparse.Namespace) -> None:
    result = await _request("POST", args.server, f"/admin/stop/{args.name}", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(result)
    elif result.get("ok"):
        print(f"Model '{args.name}' stopped (was {result.get('previous_state', 'unknown')})")
    else:
        print(f"Model '{args.name}' not found or already stopped.", file=sys.stderr)


async def cmd_stop_all(args: argparse.Namespace) -> None:
    result = await _request("POST", args.server, "/admin/stop-all", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(result)
    else:
        stopped = result.get("stopped", [])
        msg = result.get("message", "")
        print(msg or f"Stopped {len(stopped)} model(s)")


async def cmd_config_list(args: argparse.Namespace) -> None:
    data = await _request("GET", args.server, "/admin/config", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(data)
        return
    models = data.get("models", [])
    if not models:
        print("No models in config.")
    else:
        for m in models:
            print(f"  {m}")


async def cmd_config_get(args: argparse.Namespace) -> None:
    result = await _request("GET", args.server, f"/admin/config/{args.name}", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(result)
        return
    print(f"Model: {result['model']}")
    print(f"Status: {result.get('status') or 'stopped'}")
    cfg = result.get("config", {})
    for k, v in cfg.items():
        print(f"  {k}: {v}")


async def cmd_config_set(args: argparse.Namespace) -> None:
    body: Dict[str, Any] = {}
    for kv in args.args:
        key, _, value = kv.partition("=")
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        body[key] = value
    result = await _request("PUT", args.server, f"/admin/config/{args.name}", api_key=args.api_key, json_body=body)
    if getattr(args, "json", False):
        _print_json(result)
    else:
        print(f"Model '{args.name}' updated with {len(body)} field(s)")


async def cmd_config_create(args: argparse.Namespace) -> None:
    body: Dict[str, Any] = {"name": args.name or None, "checkpoint": args.checkpoint}
    if args.backend:
        body["backend"] = args.backend
    for kv in (args.args or []):
        key, _, value = kv.partition("=")
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        body[key] = value
    result = await _request("POST", args.server, "/admin/config", api_key=args.api_key, json_body=body)
    if getattr(args, "json", False):
        _print_json(result)
    elif result.get("ok"):
        print(f"Model '{result['model']}' created")


async def cmd_config_rm(args: argparse.Namespace) -> None:
    # Delete the model entry from config via admin API
    # The server doesn't have a dedicated delete endpoint, so we use PUT with empty/None values
    result = await _request("DELETE", args.server, f"/admin/config/{args.name}", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(result)
    elif result.get("ok"):
        print(f"Model '{args.name}' removed from config")
    else:
        print(result.get("detail", "Failed to remove"), file=sys.stderr)


async def cmd_logs(args: argparse.Namespace) -> None:
    if args.name == "all":
        path = "/admin/logs"
    else:
        path = f"/admin/log/{args.name}"

    params = {}
    if getattr(args, "lines", None):
        params["lines"] = args.lines

    data = await _request("GET", args.server, path, api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(data)
        return

    lines = data.get("lines", [])
    missed = data.get("missed_lines", 0)
    seq = data.get("seq", 0)

    if missed > 0:
        print(f"[skipped {missed} entries]\n")

    if args.name == "all":
        prefix = ""
    else:
        prefix = f"[{args.name}] "

    for entry in lines:
        text = entry.get("text", "") if isinstance(entry, dict) else entry
        print(f"{prefix}{text}")


async def cmd_eject(args: argparse.Namespace) -> None:
    result = await _request("POST", args.server, f"/admin/eject/{args.name}", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(result)
    elif result.get("ok"):
        print(f"Model '{args.name}' ejected — cache deleted")
    else:
        print(result.get("detail", "Failed to eject"), file=sys.stderr)


async def cmd_images_list(args: argparse.Namespace) -> None:
    data = await _request("GET", args.server, "/admin/images", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(data)
        return

    if not data:
        print("No backend images configured.")
        return

    header = f"{'BACKEND':<20} {'RUNNER':<12} {'IMAGE':<45} {'AVAIL'}"
    print(header)
    print("-" * len(header))
    for img in data:
        avail = "yes" if img.get("available") else "no"
        image = img.get("image") or "-"
        print(f"{img['backend_id']:<20} {img.get('runner', '-'):<12} {image:<45} {avail}")


async def cmd_images_build(args: argparse.Namespace) -> None:
    body = {"backend": args.backend}
    if getattr(args, "tag", None):
        body["tag"] = args.tag
    result = await _request("POST", args.server, "/admin/images/build", api_key=args.api_key, json_body=body)
    if getattr(args, "json", False):
        _print_json(result)
    elif result.get("skipped"):
        print(f"Skipped: {result.get('reason')}")
    else:
        ok = result.get("ok", True)
        status = "built" if ok else "failed"
        print(f"Image '{result.get('image')}' for backend '{args.backend}' {status}")


async def cmd_images_rm(args: argparse.Namespace) -> None:
    # URL-encode the image tag since it may contain slashes/colons
    from urllib.parse import quote
    encoded = quote(args.tag, safe="")
    result = await _request("DELETE", args.server, f"/admin/images/{encoded}", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(result)
    elif result.get("removed"):
        print(f"Image '{args.tag}' removed")
    elif result.get("skipped"):
        print(f"Skipped: {result.get('reason')}")
    else:
        print(result.get("detail", "Failed to remove image"), file=sys.stderr)


async def cmd_shutdown(args: argparse.Namespace) -> None:
    # Don't exit with error if server is shutting down (returns 503 or drops connection)
    import signal
    headers = {"Accept": "application/json"}
    api_key = args.api_key
    if api_key:
        headers["x-admin-key"] = api_key

    try:
        async with ClientSession(timeout=ClientTimeout(total=10)) as session:
            async with session.post(args.server.rstrip("/") + "/admin/shutdown", headers=headers) as resp:
                data = await resp.json()
                print(data.get("message", "Server shutting down"))
    except Exception:
        print("Shutdown signal sent (server may have already stopped)")


# ── Argument parser ───────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arkestra-admin",
        description="ModelArkestra admin CLI — manage models via the HTTP server API.",
    )
    parser.add_argument("--server", "-x", default="http://127.0.0.1:8080", help="Server URL (default: 127.0.0.1:8080)")
    parser.add_argument("--api-key", default=None, help="Admin API key (overrides config env)")
    parser.add_argument("--config", "-c", default=None, help="Config path for auto-reading ADMIN_KEY")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── models ──
    sub.add_parser("models", help="List all configured models with status")

    # ── start ──
    p = sub.add_parser("start", help="Start a model")
    p.add_argument("name", help="Model name")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--backend", default=None)
    p.add_argument("--runner", default=None)
    p.add_argument("args", nargs="*", help="Extra args as key=value (e.g. temp=0.7)")

    # ── stop ──
    p = sub.add_parser("stop", help="Stop a running model")
    p.add_argument("name", help="Model name")

    # ── stop-all ──
    sub.add_parser("stop-all", help="Stop all running models")

    # ── config ──
    cp = sub.add_parser("config", help="Manage model configs (requires sub-command)")
    cps = cp.add_subparsers(dest="config_cmd")

    # config list
    cps.add_parser("list", help="List model names in config")

    # config get
    p = cps.add_parser("get", help="Show a model's config + runtime status")
    p.add_argument("name")

    # config set
    p = cps.add_parser("set", help="Update a model field")
    p.add_argument("name")
    p.add_argument("args", nargs="+", help="Fields as key=value (e.g. backend=rocm)")

    # config create
    p = cps.add_parser("create", help="Add a new model to config")
    p.add_argument("--name", default=None)
    p.add_argument("--checkpoint", "-c", required=True, help="HF checkpoint path")
    p.add_argument("--backend", default=None)
    p.add_argument("args", nargs="*", help="Extra fields as key=value")

    # config rm
    p = cps.add_parser("rm", help="Remove a model from config")
    p.add_argument("name")

    # ── logs ──
    p = sub.add_parser("logs", help="View server or model logs")
    p.add_argument("name", help="Model name, or 'all' for global log")
    p.add_argument("--lines", "-n", type=int, default=100)

    # ── eject ──
    p = sub.add_parser("eject", help="Stop model and delete its checkpoint cache")
    p.add_argument("name")

    # ── images ──
    ip = sub.add_parser("images", help="Manage OCI container images (requires sub-command)")
    ips = ip.add_subparsers(dest="image_cmd")

    ips.add_parser("list", help="Show image availability per backend")

    pb = ips.add_parser("build", help="Build an OCI image for a backend")
    pb.add_argument("backend")
    pb.add_argument("--tag", default=None, help="Override image tag")

    pr = ips.add_parser("rm", help="Remove an OCI image")
    pr.add_argument("tag", help="Full image tag (e.g. docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.14)")

    # ── shutdown ──
    sub.add_parser("shutdown", help="Stop the server gracefully")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # ── Resolve auth ──
    api_key = args.api_key or os.environ.get("ADMIN_KEY") or _read_admin_key(args.config)
    args.api_key = api_key
    if not api_key:
        print("Error: no API key. Provide --api-key, set ADMIN_KEY env, or define it in config.yaml's env:", file=sys.stderr)
        sys.exit(1)

    # Dispatch to the right handler
    coro = _dispatch(args)
    asyncio.run(coro)


async def _dispatch(args: argparse.Namespace) -> None:
    """Top-level dispatcher — single entry point, no nested event loops."""
    if args.command == "config":
        await _cmd_config_dispatch(args)
    elif args.command == "images":
        await _cmd_images_dispatch(args)
    else:
        dispatch = {
            "models": cmd_models,
            "start": cmd_start,
            "stop": cmd_stop,
            "stop-all": cmd_stop_all,
            "logs": cmd_logs,
            "eject": cmd_eject,
            "shutdown": cmd_shutdown,
        }
        handler = dispatch.get(args.command)
        if not handler:
            build_parser().print_help()
            sys.exit(1)
        await handler(args)


async def _cmd_config_dispatch(args: argparse.Namespace) -> None:
    """Route config sub-commands to their handlers."""
    handlers = {
        "list": cmd_config_list,
        "get": cmd_config_get,
        "set": cmd_config_set,
        "create": cmd_config_create,
        "rm": cmd_config_rm,
    }
    cmd = getattr(args, "config_cmd", None)
    if not cmd:
        print("Error: config requires a sub-command (list|get|set|create|rm)", file=sys.stderr)
        sys.exit(1)
    handler = handlers.get(cmd)
    if not handler:
        print(f"Error: unknown config sub-command '{cmd}'", file=sys.stderr)
        sys.exit(1)
    await handler(args)


async def _cmd_images_dispatch(args: argparse.Namespace) -> None:
    """Route images sub-commands to their handlers."""
    handlers = {
        "list": cmd_images_list,
        "build": cmd_images_build,
        "rm": cmd_images_rm,
    }
    cmd = getattr(args, "image_cmd", None)
    if not cmd:
        print("Error: images requires a sub-command (list|build|rm)", file=sys.stderr)
        sys.exit(1)
    handler = handlers.get(cmd)
    if not handler:
        print(f"Error: unknown images sub-command '{cmd}'", file=sys.stderr)
        sys.exit(1)
    await handler(args)


if __name__ == "__main__":
    main()
