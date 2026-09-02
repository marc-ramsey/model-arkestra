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
import os
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

async def chat_direct(config_path: Optional[str], model_name: str, broadcast_addr: str) -> None:
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
TEMPLATE_FILES = ["config.yaml.j2", "backends.yaml.j2", "schemas.yaml.j2"]

# Map backend names to source names in sources.yaml
BACKEND_TO_SOURCE: dict[str, str] = {
    "rocm": "ggml-org-rocm",
    "vulkan-radv": "official-vulkan-radv",
    "cuda": "official-cuda",
    "cpu": "ggml-org-cpu",
}


def _set_backend_in_config(config_path: Path, backend: str) -> None:
    """Update the default backend in a rendered config.yaml.

    Replaces the line ``default: <current>`` under ``backends:`` with
    the detected backend.  Uses simple text replacement (no YAML parser).
    """
    content = config_path.read_text()
    # Match the backends.default line and replace its value
    lines = content.splitlines(True)
    new_lines: list[str] = []
    in_backends_section = False
    replaced = False

    for line in lines:
        stripped = line.strip()
        if stripped == "backends:" or (stripped.startswith("backends") and ":" in stripped):
            in_backends_section = True
            new_lines.append(line)
            continue
        if in_backends_section and not replaced:
            # Check for the 'default:' key under backends
            if stripped == "default:" or (stripped.startswith("default:") and ":" in stripped):
                indent = len(line) - len(line.lstrip())
                new_lines.append(" " * indent + f"default: {backend}\n")
                replaced = True
                continue
            # If we hit a non-indented, non-comment line that isn't 'default',
            # we've left the backends section
            if stripped and not stripped.startswith("#"):
                in_backends_section = False
        new_lines.append(line)

    if replaced:
        config_path.write_text("".join(new_lines))


def _set_rocm_source_in_backends(backends_path: Path, gfx_family: str) -> None:
    """Update the 'rocm' backend's source_ref to use the per-GFX-family source.

    e.g. 'gfx1151' → sets rocm.source_ref = rocm-gfx1151
    """
    try:
        import yaml as _yaml
        data = _yaml.safe_load(backends_path.read_text()) or {}
    except Exception:
        return

    backends = data.setdefault("backends", {})
    rocm_backend = backends.get("rocm")
    if not isinstance(rocm_backend, dict):
        return

    # Map gfx ID → source_ref (must match a source entry in backends.yaml)
    source_key = f"rocm-{gfx_family}"
    rocm_backend["source_ref"] = source_key

    # Re-serialize YAML preserving structure
    backends_path.write_text(_yaml.dump(data, default_flow_style=False, sort_keys=False))
    print(f"\nWrote rocm.source_ref: {source_key}  (GFX {gfx_family})")


# ── Download backend commands ───────────────────────────────────────

def _load_sources(config_dir: Path) -> tuple[dict, dict]:
    """Load sources.yaml from config directory.
    
    Returns (sources_dict, global_defaults) or empty dicts if file missing.
    """
    sources_file = config_dir / "sources.yaml"
    if not sources_file.exists():
        return {}, {}
    try:
        import yaml
        data = yaml.safe_load(sources_file.read_text()) or {}
        sources = data.get("sources", {}) or {}
        defaults = data.get("defaults", {}) or {}
        return sources, defaults
    except Exception:
        return {}, {}


def cmd_download_backend(backend_name: str, version: str = "latest") -> int:
    """Download a pre-built backend binary from backends.yaml sources.

    Resolves the backend to a source in backends.yaml, downloads using
    BinaryDownloader, and writes the resolved `binary_dir` + `binary` path
    into the backend entry so process.py can find it at runtime.
    
    Args:
        backend_name: Name of backend (e.g., 'rocm', 'vulkan-radv', 'cpu').
        version: Version tag (default 'latest', or a pinned version like '2.95').
    
    Returns:
        0 on success, 1 on error.
    """
    from model_arkestra.binary_downloader import (
    BinaryDownloader,
    BinaryDownloaderError,
    RuntimeCheckError,
)

    config_dir = DEFAULT_CONFIG_DIR
    backends_file = config_dir / "backends.yaml"
    
    # Load sources from backends.yaml (merged with backends section)
    if not backends_file.exists():
        print("backends.yaml not found. Run 'model-arkestra init' first.", file=sys.stderr)
        return 1
    
    try:
        raw = _yaml.safe_load(backends_file.read_text()) or {}
        sources = raw.get("sources", {}) or {}
        defaults = raw.get("defaults", {}) or {}
    except Exception as e:
        print(f"Failed to parse backends.yaml: {e}", file=sys.stderr)
        return 1

    # Resolve backend → source name (check BACKEND_TO_SOURCE, direct source lookup,
    # and the backends section's source_ref field)
    source_name = BACKEND_TO_SOURCE.get(backend_name)
    if source_name is None and backend_name in sources:
        source_name = backend_name  # direct source name
    
    if source_name is None:
        # Try resolving via the backends section's source_ref field
        be_section = raw.get("backends", {})
        if backend_name in be_section:
            source_ref = be_section[backend_name].get("source_ref")
            if source_ref and source_ref in sources:
                source_name = source_ref
    
    if source_name is None:
        print(f"Unknown backend: {backend_name}", file=sys.stderr)
        avail = [k for k, v in BACKEND_TO_SOURCE.items() if v is not None]
        extra = [k for k in sources if k not in BACKEND_TO_SOURCE or BACKEND_TO_SOURCE.get(k) == k]
        print(f"Known backends: {', '.join(avail)}", file=sys.stderr)
        if extra:
            print(f"Or any source name from backends.yaml: {', '.join(extra[:5])}", file=sys.stderr)
        return 1
    
    source_cfg = sources.get(source_name)
    if not source_cfg or not isinstance(source_cfg, dict):
        print(f"Source '{source_name}' not found in backends.yaml", file=sys.stderr)
        return 1
    
    # Create downloader and resolve
    cache_dir = Path.home() / ".local" / "share" / "model-arkestra" / "bin-cache"
    try:
        downloader = BinaryDownloader(
            backend_id=backend_name,
            source_cfg=source_cfg,
            cache_dir=cache_dir,
            global_defaults=defaults if defaults else None,
        )
        result = asyncio.run(downloader.resolve(version=version))
    except BinaryDownloaderError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1
    
    # Handle runtime-check (no binary to wire — just validation)
    if result == "runtime-ok":
        print(f"✓ Runtime check passed for {backend_name}")
        print(f"  Source:      {source_name}")
        print(f"  Status:      nvidia-smi + CUDA libs verified")
        print(f"  Note:        Build from source or use container mode for the binary.")
        return 0
    
    # Wire the resolved binary path into backends.yaml so process.py can find it
    binary_dir = str(Path(result).parent)
    binary_name = Path(result).name
    _set_binary_in_backend(backends_file, backend_name, binary_dir, binary_name)
    
    print(f"✓ Downloaded {backend_name} to: {result}")
    print(f"  Version:   {version}")
    print(f"  Source:    {source_name}")
    print(f"  Cached in: {binary_dir}")
    return 0


def _set_binary_in_backend(
    backends_file: Path,
    backend_name: str,
    binary_dir: str,
    binary_name: str,
) -> None:
    """Update a backend entry in backends.yaml with resolved binary path.

    Writes `binary_dir` and `binary` keys into the backend definition so
    process.py can locate it at runtime. Uses atomic write to prevent
    corruption if interrupted.
    """
    data: dict = {}
    if backends_file.exists():
        try:
            data = _yaml.safe_load(backends_file.read_text()) or {}
        except Exception:
            data = {}
    
    be_section: dict = data.get("backends", {})
    if backend_name not in be_section:
        print(f"  Warning: Backend '{backend_name}' not found in backends.yaml — skipping path update.", file=sys.stderr)
        return
    
    be_entry = be_section[backend_name]
    binary_dir_expanded = str(Path(binary_dir).expanduser())
    be_entry["binary_dir"] = binary_dir_expanded
    be_entry["binary"] = binary_name
    data["backends"] = be_section
    
    tmp_path = backends_file.with_suffix(".yaml.tmp")
    with open(tmp_path, "w") as f:
        _yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    tmp_path.rename(backends_file)


def cmd_download_all() -> int:
    """Download primary and fallback backends based on detected hardware.
    
    Runs GPU detection, then downloads the recommended backend plus one
    fallback (e.g., vulkan-radv for AMD even if rocm is preferred).
    """
    from model_arkestra.gpu_detect import detect_all

    result = detect_all()
    primary_backend, reason = result["recommendation"]
    
    # Choose fallback based on hardware
    fallback: str | None = None
    if result["primary_gpu"] and result["primary_gpu"]["vendor"] == "amd":
        primary_backend = "rocm"
        fallback = "vulkan-radv"
    elif result["primary_gpu"] and result["primary_gpu"]["vendor"] == "nvidia":
        primary_backend = "cuda"
        fallback = None
    else:
        primary_backend = "cpu"
        fallback = None

    print(f"Downloading backend for detected hardware:")
    print(f"  Primary: {primary_backend}")
    if fallback:
        print(f"  Fallback: {fallback}")
    print()
    
    # Download primary
    rc1 = cmd_download_backend(primary_backend, version="latest")
    if rc1 != 0:
        return rc1
    
    # Download fallback if available
    if fallback and BACKEND_TO_SOURCE.get(fallback):
        print()
        rc2 = cmd_download_backend(fallback, version="latest")
        if rc2 != 0:
            print(f"\n⚠ Fallback download failed (primary {primary_backend} is OK).", file=sys.stderr)
    
    return 0


def cmd_init(force: bool = False) -> int:
    """Scaffold default config files in ~/.config/arkestra/.

    After writing templates, detects available GPU/CPU hardware and sets
    the recommended backend as the default.  Prints warnings for special
    cases (multi-GPU, ROCm preference on Strix Halo, etc.).

    Returns 0 on success, 1 on error.
    """
    from model_arkestra.gpu_detect import detect_all

    config_dir = DEFAULT_CONFIG_DIR
    config_file = config_dir / "config.yaml"
    backends_file = config_dir / "backends.yaml"

    # Check for existing files
    if not force and (config_file.exists() or backends_file.exists()):
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
            content = (src_pkg / template_name).read_text()
            dest.write_text(content)
            print(f"Created {dest}")
        except Exception as exc:
            print(f"Error creating {template_name}: {exc}", file=sys.stderr)
            return 1

    # ── Hardware detection & backend recommendation ────────────────
    result = detect_all()
    backend, reason = result["recommendation"]

    print(f"\nDetected GPU hardware:")
    if result["primary_gpu"]:
        g = result["primary_gpu"]
        gpu_desc = g["name"].split("]", 1)[-1].strip() if "]" in g["name"] else g["name"]
        print(f"  ✓ {gpu_desc}")
    else:
        cpu = result["cpu"]
        feat_str = ", ".join(cpu.get("features", [])) or "default"
        print(f"  No GPU found")
        print(f"  CPU: {cpu['arch']} ({cpu['vendor']}, {cpu['cores']} cores, {feat_str})")

    # Multi-GPU warning
    if result["multi_gpu_warn"]:
        n = len(result["gpus"])
        print(f"\n  ⚠ {n} GPUs detected — using first ({backend}) as default.")
        print("  Secondary GPU(s) ignored. Edit config.yaml to change.")

    # Warnings from detection
    for w in result.get("warnings", []):
        print(f"\n  ⚠ {w}")

    # Patch config with detected backend
    _set_backend_in_config(config_file, backend)
    print(f"\nWrote backends.default: {backend}")
    print(f"(reason: {reason})")

    # ── ROCm gfx family → source_ref wiring ────────────────────
    if backend == "rocm":
        gfx = result.get("gfx_family")
        if gfx:
            _set_rocm_source_in_backends(backends_file, gfx)
        else:
            # Fallback to generic ggml-org source
            print(f"\n  ℹ Could not detect exact GFX version — using default ROCm source.")
            print("     To use the optimal per-GPU binary, ensure rocm-smi is available.")

    print(f"\nConfig files scaffolded to {config_dir}")
    print("  config.yaml      — model configs (edit freely)")
    print("  backends.yaml    — backend definitions + download sources")
    print()
    print("To add a custom local binary:")
    print("  model-arkestra add-backend --local /path/to/binary [--name my-llama]")
    print("To see available backends:")
    print("  model-arkestra list-backends")
    return 0


# ── Backend management commands ───────────────────────────────────

def _load_backends_cfg() -> tuple[Path, dict]:
    """Load backends.yaml and return (path, data_dict). Raises on error."""
    path = DEFAULT_CONFIG_DIR / "backends.yaml"
    if not path.exists():
        print("backends.yaml not found. Run 'model-arkestra init' first.", file=sys.stderr)
        sys.exit(1)
    try:
        import yaml as _yaml
        data = _yaml.safe_load(path.read_text()) or {}
        return path, data
    except Exception as e:
        print(f"Failed to parse backends.yaml: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list_backends() -> int:
    """List all configured backends with their status."""
    import yaml as _yaml
    
    backends_path = DEFAULT_CONFIG_DIR / "backends.yaml"
    if not backends_path.exists():
        print("backends.yaml not found. Run 'model-arkestra init' first.", file=sys.stderr)
        return 1
    
    try:
        data = _yaml.safe_load(backends_path.read_text()) or {}
    except Exception as e:
        print(f"Failed to parse backends.yaml: {e}", file=sys.stderr)
        return 1
    
    backends = data.get("backends", {})
    if not backends:
        print("No backends defined in backends.yaml.")
        return 0
    
    cache_dir = Path.home() / ".local" / "share" / "model-arkestra" / "bin-cache"
    configs_path = DEFAULT_CONFIG_DIR / "config.yaml"
    current_default = None
    if configs_path.exists():
        try:
            cfg_data = _yaml.safe_load(configs_path.read_text()) or {}
            be_section = cfg_data.get("backends", {})
            if isinstance(be_section, dict):
                current_default = be_section.get("default")
        except Exception:
            pass
    
    print(f"{'Name':<20} {'Type':<16} {'Description':<50} {'Binary':<10} {'Default'}")
    print("-" * 115)
    
    for name, be_cfg in sorted(backends.items()):
        if not isinstance(be_cfg, dict):
            continue
        be_type = be_cfg.get("type", "-")
        desc = (be_cfg.get("description") or "-")[:48]
        cached = False
        if cache_dir.exists():
            bin_files = list(cache_dir.rglob(f"*{name}*llama-server*") or [])
            if not bin_files:
                src_name = be_cfg.get("source_ref", "")
                bin_files = list(cache_dir.rglob(f"*{src_name}*") or [])
            cached = len(bin_files) > 0
        
        binary_status = "✓ cached" if cached else "not found"
        default_marker = " ← default" if name == current_default else ""
        print(f"{name:<20} {be_type:<16} {desc:<50} {binary_status:<10}{default_marker}")
    
    return 0


def cmd_add_backend(
    local_path: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> int:
    """Add a custom backend with a local binary."""
    import yaml as _yaml
    
    if not local_path:
        print("Error: --local /path/to/binary is required.", file=sys.stderr)
        return 1
    
    bin_path = Path(local_path).expanduser().resolve()
    if not bin_path.exists():
        print(f"Binary not found: {bin_path}", file=sys.stderr)
        return 1
    if not os.access(str(bin_path), os.X_OK):
        print(f"Binary is not executable: {bin_path}", file=sys.stderr)
        return 1
    
    if name is None:
        name = bin_path.stem.replace("llama-server", "local-llama")
        if "avx" in str(bin_path).lower():
            name = f"local-{bin_path.stem}"
    
    backends_file = DEFAULT_CONFIG_DIR / "backends.yaml"
    data: dict = {}
    if backends_file.exists():
        try:
            data = _yaml.safe_load(backends_file.read_text()) or {}
        except Exception:
            data = {}
    
    be_section: dict = data.get("backends", {})
    if name in be_section:
        print(f"Backend '{name}' already exists. Remove it first.", file=sys.stderr)
        return 1
    
    be_section[name] = {
        "type": "custom",
        "description": description or f"Custom llama-server from {bin_path.parent}",
        "runner": "process",
        "binary_path": str(bin_path),
        "args": {
            "ngl": 999,
            "ctx-size": "${ctx-size}",
        },
    }
    data["backends"] = be_section
    
    tmp_path = backends_file.with_suffix(".yaml.tmp")
    with open(tmp_path, "w") as f:
        _yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    tmp_path.rename(backends_file)
    
    print(f"✓ Added backend '{name}' from {bin_path}")
    return 0


def cmd_remove_backend(backend_name: str) -> int:
    """Remove a backend from backends.yaml."""
    import yaml as _yaml
    
    backends_file = DEFAULT_CONFIG_DIR / "backends.yaml"
    if not backends_file.exists():
        print("backends.yaml not found. Run 'model-arkestra init' first.", file=sys.stderr)
        return 1
    
    try:
        data = _yaml.safe_load(backends_file.read_text()) or {}
    except Exception as e:
        print(f"Failed to parse backends.yaml: {e}", file=sys.stderr)
        return 1
    
    be_section: dict = data.get("backends", {})
    if not isinstance(be_section, dict):
        be_section = {}
    
    if backend_name not in be_section:
        available = list(be_section.keys())
        print(f"Backend '{backend_name}' not found.", file=sys.stderr)
        if available:
            print(f"Available: {', '.join(available)}", file=sys.stderr)
        return 1
    
    del be_section[backend_name]
    data["backends"] = be_section
    
    tmp_path = backends_file.with_suffix(".yaml.tmp")
    with open(tmp_path, "w") as f:
        _yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    tmp_path.rename(backends_file)
    
    print(f"✓ Removed backend '{backend_name}'")
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
    chat_group = chat_parser.add_mutually_exclusive_group(required=False)
    chat_group.add_argument("--config", "-c", default=None, help="YAML config file path (direct mode, default: ~/.config/arkestra/config.yaml)")
    chat_group.add_argument("--server", "-x", default=None, help="Server URL")
    chat_parser.add_argument("--model", "-m", required=True, help="Model name to chat with")
    chat_parser.add_argument("--broadcast-addr", default=None,
                             help='Broadcast address for models (default: 0.0.0.0)')

    # ── list-backends subcommand ──────────────────────────────────────
    subparsers.add_parser(
        "list-backends",
        help="List all configured backends with status",
    )

    # ── add-backend subcommand ────────────────────────────────────────
    add_parser = subparsers.add_parser(
        "add-backend",
        help="Add a custom backend with a local binary",
    )
    add_parser.add_argument("--local", "-l", help="Path to local llama-server binary")
    add_parser.add_argument("--name", "-n", default=None, help="Backend name (derived from path if not given)")
    add_parser.add_argument("--description", "-d", default=None, help="Human-readable description")

    # ── remove-backend subcommand ─────────────────────────────────────
    rm_parser = subparsers.add_parser(
        "remove-backend",
        help="Remove a backend from backends.yaml",
    )
    rm_parser.add_argument("backend", help="Backend name to remove")

    # ── download-backend subcommand ───────────────────────────────────
    dl_parser = subparsers.add_parser(
        "download-backend",
        help="Download a pre-built backend binary from backends.yaml sources",
    )
    dl_parser.add_argument("backend", help="Backend name (rocm, vulkan-radv, cpu)")
    dl_parser.add_argument(
        "--version", "-V", default="latest",
        help='Version tag (default: latest)',
    )

    # ── download-all subcommand ───────────────────────────────────────
    subparsers.add_parser(
        "download-all",
        help="Download primary + fallback backends based on detected hardware",
    )

    args = parser.parse_args(argv)

    # ── Route to subcommand handler ───────────────────────────────────
    if args.command == "init":
        sys.exit(cmd_init(force=args.force))

    if args.command == "list-backends":
        sys.exit(cmd_list_backends())

    if args.command == "add-backend":
        sys.exit(cmd_add_backend(
            local_path=getattr(args, 'local', None),
            name=getattr(args, 'name', None),
            description=getattr(args, 'description', None),
        ))

    if args.command == "remove-backend":
        sys.exit(cmd_remove_backend(args.backend))

    if args.command == "download-backend":
        sys.exit(cmd_download_backend(backend_name=args.backend, version=args.version))

    if args.command == "download-all":
        sys.exit(cmd_download_all())

    # Default: chat subcommand (for backwards compat with no subcommand)
    if hasattr(args, 'config'):
        broadcast_addr = args.broadcast_addr or "0.0.0.0"
        if args.server:
            asyncio.run(chat_server(args.server, args.model))
        else:
            asyncio.run(chat_direct(args.config, args.model, broadcast_addr))


