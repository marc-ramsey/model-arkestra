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
from pathlib import Path
from typing import Any, Dict, List

from model_arkestra.conn import Conn, add_common_args, resolve_conn

try:
    from aiohttp import ClientSession, ClientTimeout
except ImportError:
    print("Error: aiohttp required. Install with: pip install \"model-arkestra[proxy]\"", file=sys.stderr)
    sys.exit(1)


# ── Config reading (lightweight, no ModelArkestra instance needed) ─────────

def _load_config(path: str | None = None) -> dict:
    """Load config.yaml and return parsed dict, or {} on failure."""
    try:
        import yaml
    except ImportError:
        return {}

    p = path or os.path.expanduser("~/.config/arkestra/config.yaml")
    try:
        with open(p) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, OSError):
        return {}


def _read_admin_key(config_path: str | None = None) -> str | None:
    """Read the admin API key from config.yaml.

    Checks, in order: default.admin-key, default-env.admin_key, env.ADMIN_KEY.
    Returns None if none are set.
    """
    data = _load_config(config_path)
    for section, key in (("default", "admin-key"), ("default-env", "admin_key")):
        val = (data.get(section) or {}).get(key)
        if val:
            return str(val)
    val = (data.get("env") or {}).get("ADMIN_KEY")
    return str(val) if val else None


# ── HTTP helpers ───────────────────────────────────────────────────────

async def _request(
    method: str,
    conn: "Conn",
    path: str,
    *,
    api_key: str | None = None,
    json_body: dict | None = None,
) -> Dict[str, Any]:
    """Execute an admin API request and return parsed JSON."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = conn.url_for(path)
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

async def cmd_status(args: argparse.Namespace) -> None:
    """Print server status similar to startup banner."""
    import importlib.metadata
    import platform
    import re as _re

    version = importlib.metadata.version("model-arkestra")
    print(f"ModelArkestra v{version}")

    # URL from resolved connection
    url = args.conn.url.rstrip("/")
    print(f"  URL       → {url}")
    print(f"  API docs  → {url}/docs")

    # Hardware (local detection, same as server startup)
    try:
        from model_arkestra.gpu_detect import detect_all
        hw = detect_all()
    except Exception:
        hw = {}
    primary = hw.get("primary_gpu")
    if primary:
        vendor_map = {"amd": "AMD", "nvidia": "NVIDIA", "intel": "Intel"}
        vendor_name = vendor_map.get(primary["vendor"], "GPU")
        raw = primary["name"]
        brackets = _re.findall(r'\[([^\]]+)\]', raw)
        skus = [s.strip() for b in brackets if "/ " in b for s in b.split(" / ")]
        short = skus[-1] if skus else (brackets[-1] if brackets else raw.split(": ", 1)[-1].strip())
        line = f"{vendor_name} {short} • {primary['backend']}"
        gfx = hw.get("gfx_family")
        if gfx:
            line += f" ({gfx})"
    else:
        cpu_info = hw.get("cpu", {})
        line = f"CPU ({cpu_info.get('arch', platform.machine())})"
    print(f"  Hardware  → {line}")

    # Cache path from config / env / default
    from model_arkestra.common import default_cache_root, resolve_config_path
    data = _load_config(args.config)
    hf_cache = None
    hc = (data.get("default-env") or {}).get("hf-hub-cache") or os.environ.get("HF_HUB_CACHE")
    if hc:
        hf_cache = str(Path(hc).expanduser())
    if not hf_cache:
        hf_cache = str(default_cache_root())
    print(f"  Cache     → {hf_cache}")

    # Config path
    from model_arkestra.common import resolve_config_path as _rcp
    config_path = _rcp(args.config)
    print(f"  Config    → {config_path}")

    # Live status from server /health
    try:
        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            headers = {"Accept": "application/json"}
            if args.conn.api_key:
                headers["Authorization"] = f"Bearer {args.conn.api_key}"
            async with session.get(args.conn.url_for("/health"), headers=headers) as resp:
                health = await resp.json()
        uptime = health.get("uptime_seconds", 0)
        running = health.get("models_running", 0)
        # Format uptime
        if uptime < 60:
            up_str = f"{uptime:.0f}s"
        elif uptime < 3600:
            up_str = f"{uptime / 60:.0f}m"
        else:
            up_str = f"{uptime / 3600:.1f}h"
        print(f"  Status    → ok • up {up_str} • {running} model(s) running")
    except Exception:
        print("  Status    → unreachable")

async def cmd_models(args: argparse.Namespace) -> None:
    data = await _request("GET", args.conn, "/admin/models", api_key=args.api_key)
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
        runner = m.get("runner") or ""
        if isinstance(runner, str):
            runner = runner.replace("runnerstate.", "").lower()
        status_val = m.get("status", {})
        if isinstance(status_val, dict):
            status_val = status_val.get("value", "stopped")
        port = m.get("port") or "-"
        print(f"{m['name']:<30} {str(status_val):<12} {str(port):<8} {str(m.get('backend') or '-'):<20} {runner}")


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

    result = await _request("POST", args.conn, f"/admin/start/{args.name}", api_key=args.api_key, json_body=body)
    if getattr(args, "json", False):
        _print_json(result)
    else:
        port = result.get("port")
        status = "ok" if result.get("ok") else "failed"
        print(f"Model '{args.name}' {status}" + (f" on port {port}" if port else ""))


async def cmd_stop(args: argparse.Namespace) -> None:
    result = await _request("POST", args.conn, f"/admin/stop/{args.name}", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(result)
    elif result.get("ok"):
        print(f"Model '{args.name}' stopped (was {result.get('previous_state', 'unknown')})")
    else:
        print(f"Model '{args.name}' not found or already stopped.", file=sys.stderr)


async def cmd_stop_all(args: argparse.Namespace) -> None:
    result = await _request("POST", args.conn, "/admin/stop-all", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(result)
    else:
        stopped = result.get("stopped", [])
        msg = result.get("message", "")
        print(msg or f"Stopped {len(stopped)} model(s)")


async def cmd_config_list(args: argparse.Namespace) -> None:
    data = await _request("GET", args.conn, "/admin/config", api_key=args.api_key)
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
    result = await _request("GET", args.conn, f"/admin/config/{args.name}", api_key=args.api_key)
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
    result = await _request("PUT", args.conn, f"/admin/config/{args.name}", api_key=args.api_key, json_body=body)
    if getattr(args, "json", False):
        _print_json(result)
    else:
        print(f"Model '{args.name}' updated with {len(body)} field(s)")


async def cmd_config_create(args: argparse.Namespace) -> None:
    body: Dict[str, Any] = {
        "name": args.name or None,
        "model": args.model,
    }
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
    result = await _request("POST", args.conn, "/admin/config", api_key=args.api_key, json_body=body)
    if getattr(args, "json", False):
        _print_json(result)
    elif result.get("ok"):
        print(f"Model '{result['model']}' created")


async def cmd_config_rm(args: argparse.Namespace) -> None:
    # Delete the model entry from config via admin API
    # The server doesn't have a dedicated delete endpoint, so we use PUT with empty/None values
    result = await _request("DELETE", args.conn, f"/admin/config/{args.name}", api_key=args.api_key)
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

    data = await _request("GET", args.conn, path, api_key=args.api_key)
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


async def cmd_pull(args: argparse.Namespace) -> None:
    """Pull a checkpoint, blocking until complete with a progress indicator.

    The server never blocks; this client polls /admin/pull-status until the
    download finishes (state leaves 'downloading').
    """
    result = await _request("POST", args.conn, f"/admin/pull/{args.name}", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(result)
        return
    if not (result.get("ok") or result.get("already_downloading") or result.get("already_loaded")):
        print(result.get("detail", "Failed to pull"), file=sys.stderr)
        sys.exit(1)
    if result.get("already_loaded"):
        print(f"'{args.name}' is already loaded")
        return

    # Block here (client-side) until the background download completes.
    last = ""
    while True:
        st = await _request("GET", args.conn, f"/admin/pull-status/{args.name}", api_key=args.api_key)
        state = st.get("state", "")
        if state == "stopped":  # download_ok -> STOPPED
            print(f"\n'{args.name}' downloaded.")
            return
        if state in ("error",):
            print(f"\nPull failed: {st.get('error')}", file=sys.stderr)
            sys.exit(1)
        pct = st.get("pct")
        cur = st.get("current") or ""
        line = f"downloading {args.name}"
        if cur:
            line += f"  [{cur}]"
        if pct is not None:
            line += f"  {pct:5.1f}%"
        speed = st.get("speed_mbps")
        if speed:
            line += f"  {speed:6.2f} MB/s"
        if line != last:
            print(f"\r{line}", end="", flush=True)
            last = line
        await asyncio.sleep(1)


async def cmd_restart(args: argparse.Namespace) -> None:
    body: Dict[str, Any] = {}
    if args.backend:
        body["backend"] = args.backend
    if args.runner:
        body["runner"] = args.runner
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
    result = await _request("POST", args.conn, f"/admin/restart/{args.name}", api_key=args.api_key, json_body=body)
    if getattr(args, "json", False):
        _print_json(result)
    elif result.get("ok"):
        port = result.get("port")
        print(f"Model '{args.name}' restarted" + (f" on port {port}" if port else ""))
    else:
        print(result.get("detail", "Failed to restart"), file=sys.stderr)


async def cmd_cancel_pull(args: argparse.Namespace) -> None:
    result = await _request("POST", args.conn, f"/admin/cancel-pull/{args.name}", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(result)
    elif result.get("ok"):
        print(f"Pull cancelled for '{args.name}'")
    else:
        print(result.get("detail", "Failed to cancel pull"), file=sys.stderr)


async def cmd_clusters(args: argparse.Namespace) -> None:
    result = await _request("GET", args.conn, "/admin/clusters", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(result)
        return
    clusters = result.get("clusters", [])
    if not clusters:
        print("No clusters configured.")
        return
    header = f"{'NAME':<20} {'URL':<45} {'HEALTHY'}"
    print(header)
    print("-" * len(header))
    for c in clusters:
        healthy = "yes" if c.get("healthy") else "no"
        print(f"{c['name']:<20} {c.get('url', '-'):<45} {healthy}")


async def cmd_cluster_add(args: argparse.Namespace) -> None:
    body = {"url": args.cluster_url}
    if getattr(args, "admin_key", None):
        body["admin-key"] = args.admin_key
    result = await _request("POST", args.conn, f"/admin/clusters/{args.name}", api_key=args.api_key, json_body=body)
    if getattr(args, "json", False):
        _print_json(result)
    elif result.get("ok"):
        print(f"Cluster '{args.name}' added")
    else:
        print(result.get("detail", "Failed to add cluster"), file=sys.stderr)


async def cmd_cluster_delete(args: argparse.Namespace) -> None:
    result = await _request("DELETE", args.conn, f"/admin/clusters/{args.name}", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(result)
    elif result.get("ok"):
        print(f"Cluster '{args.name}' removed")
    else:
        print(result.get("detail", "Failed to delete cluster"), file=sys.stderr)


async def cmd_eject(args: argparse.Namespace) -> None:
    result = await _request("POST", args.conn, f"/admin/eject/{args.name}", api_key=args.api_key)
    if getattr(args, "json", False):
        _print_json(result)
    elif result.get("ok"):
        print(f"Model '{args.name}' ejected — cache deleted")
    else:
        print(result.get("detail", "Failed to eject"), file=sys.stderr)


async def cmd_images_list(args: argparse.Namespace) -> None:
    data = await _request("GET", args.conn, "/admin/images", api_key=args.api_key)
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


async def cmd_images_rm(args: argparse.Namespace) -> None:
    # URL-encode the image tag since it may contain slashes/colons
    from urllib.parse import quote
    encoded = quote(args.tag, safe="")
    result = await _request("DELETE", args.conn, f"/admin/images/{encoded}", api_key=args.api_key)
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
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with ClientSession(timeout=ClientTimeout(total=10)) as session:
            async with session.post(args.conn.url_for("/admin/shutdown"), headers=headers) as resp:
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
    add_common_args(parser)

    sub = parser.add_subparsers(dest="command", help="Available commands")

    sub.add_parser("status", help="Show server status and environment info")

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
    p.add_argument("--model", "-m", required=True,
                   help="Model ref (e.g. unsloth/Qwen3.5:Q4_K_M or lcl:/path/file.gguf)")
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

    # ── pull ──
    p = sub.add_parser("pull", help="Download model checkpoint from HuggingFace")
    p.add_argument("name")

    # ── restart ──
    p = sub.add_parser("restart", help="Restart a model (with optional backend/runner overrides)")
    p.add_argument("name")
    p.add_argument("--backend", default=None)
    p.add_argument("--runner", default=None)
    p.add_argument("args", nargs="*", help="Extra fields as key=value")

    # ── cancel-pull ──
    cp = sub.add_parser("cancel-pull", help="Cancel in-flight download")
    cp.add_argument("name")

    # ── clusters ──
    clp = sub.add_parser("clusters", help="Manage remote cluster proxies (requires sub-command)")
    clsubs = clp.add_subparsers(dest="cluster_cmd")
    clsubs.add_parser("list", help="List all clusters")
    cla = clsubs.add_parser("add", help="Add a remote cluster")
    cla.add_argument("name")
    cla.add_argument("cluster_url", metavar="URL", help="Cluster public URL (scheme://host:port/prefix)")
    cla.add_argument("--admin-key", default=None, help="Cluster's admin key")
    cld = clsubs.add_parser("delete", help="Remove a remote cluster")
    cld.add_argument("name")

    # ── images ──
    ip = sub.add_parser("images", help="Manage OCI container images (requires sub-command)")
    ips = ip.add_subparsers(dest="image_cmd")

    ips.add_parser("list", help="Show image availability per backend")

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

    # ── Resolve connection: CLI > env > config > default (shared resolver) ─
    def _cfg_get(path, default=None):
        data = _load_config(args.config)
        node = data
        for part in path.split("/"):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    conn = resolve_conn(args, server=False, cfg_get=_cfg_get)
    # Auth: shared resolver covers --api-key / ARKESTRA_API_KEY; fall back to config.
    api_key = conn.api_key or _read_admin_key(args.config)
    if api_key != conn.api_key:
        conn = Conn(scheme=conn.scheme, host=conn.host, port=conn.port,
                    base_path=conn.base_path, config_path=conn.config_path, api_key=api_key)
    args.conn = conn
    if not api_key:
        print("Error: no API key. Provide --api-key, set ARKESTRA_API_KEY env, or define admin_key in config.yaml", file=sys.stderr)
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
    elif args.command == "clusters":
        await _cmd_clusters_dispatch(args)
    else:
        dispatch = {
            "status": cmd_status,
            "models": cmd_models,
            "start": cmd_start,
            "stop": cmd_stop,
            "stop-all": cmd_stop_all,
            "logs": cmd_logs,
            "eject": cmd_eject,
            "shutdown": cmd_shutdown,
            "pull": cmd_pull,
            "restart": cmd_restart,
            "cancel-pull": cmd_cancel_pull,
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
        "rm": cmd_images_rm,
    }
    cmd = getattr(args, "image_cmd", None)
    if not cmd:
        print("Error: images requires a sub-command (list|rm)", file=sys.stderr)
        sys.exit(1)
    handler = handlers.get(cmd)
    if not handler:
        print(f"Error: unknown images sub-command '{cmd}'", file=sys.stderr)
        sys.exit(1)
    await handler(args)


async def _cmd_clusters_dispatch(args: argparse.Namespace) -> None:
    """Route clusters sub-commands to their handlers."""
    handlers = {
        "list": cmd_clusters,
        "add": cmd_cluster_add,
        "delete": cmd_cluster_delete,
    }
    cmd = getattr(args, "cluster_cmd", None)
    if not cmd:
        print("Error: clusters requires a sub-command (list|add|delete)", file=sys.stderr)
        sys.exit(1)
    handler = handlers.get(cmd)
    if not handler:
        print(f"Error: unknown clusters sub-command '{cmd}'", file=sys.stderr)
        sys.exit(1)
    await handler(args)


if __name__ == "__main__":
    main()
