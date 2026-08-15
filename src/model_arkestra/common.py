"""Shared constants and utilities for container-based runners."""
from __future__ import annotations
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple, Union

INSPECT_RE = re.compile(r"^(exited|dead|paused|removing)\s*$", re.IGNORECASE)

# ── Known ROCm build directories, keyed by version string ────────────────
_ROCM_BUILD_MAP: Dict[str, str] = {}

# ── Default build dirs (sensible fallbacks — override via config or env) ────
_DEFAULT_VULKAN_DIR = "/usr/local/llama.cpp/build-vulkan-radv/bin"
_DEFAULT_ROCM_DIR = "/usr/local/llama.cpp/build-rocm/bin"


def register_rocm_build(version: str, build_dir: str) -> None:
    """Register a known ROCm build directory.

    Call this once at import time (or from config initialization) to map
    backend ``version`` strings to host build directories.
    """
    _ROCM_BUILD_MAP[version] = build_dir


def get_rocm_build_dirs() -> Dict[str, str]:
    """Return the full ROCm version → directory mapping."""
    return dict(_ROCM_BUILD_MAP)


def set_default_dirs(vulkan_dir: Optional[str] = None, rocm_dir: Optional[str] = None) -> None:
    """Override the default build directories (e.g. from config)."""
    global _DEFAULT_VULKAN_DIR, _DEFAULT_ROCM_DIR
    if vulkan_dir:
        _DEFAULT_VULKAN_DIR = vulkan_dir
    if rocm_dir:
        _DEFAULT_ROCM_DIR = rocm_dir


def resolve_binary_from_backend(backend: Dict[str, Any]) -> Optional[tuple]:
    """Resolve host binary dir and devices from a backend dict.

    Resolution order:
    1. Explicit ``backend["binary_dir"]`` — absolute path inside container
    2. ``backend["version"]`` mapped to known ROCm builds → ``_ROCM_BUILD_MAP``
    3. Image name substring heuristics (vulkan / rocm) as last resort

    Returns ``(binary_path, devices)`` or **None** when nothing resolves.
    """
    import os

    devices: List[str] = []

    # 1. Explicit binary_dir takes priority over everything
    binary_dir = backend.get("binary_dir")
    if binary_dir and os.path.isdir(str(binary_dir)):
        d = str(binary_dir).lower()
        if "rocm" in d or "hip" in d or "vulkan" in d:
            devices = ["/dev/dri/card1:rwm", "/dev/dri/renderD128:rwm"]
        return (str(binary_dir) + "/llama-server", devices)

    version = str(backend.get("version", ""))
    image = str(backend.get("image", "")).lower()

    # 2. ROCm version → known build directory
    if version and version in _ROCM_BUILD_MAP:
        rocm_dir = _ROCM_BUILD_MAP[version]
        if os.path.isdir(rocm_dir):
            devices = ["/dev/dri/card1:rwm", "/dev/dri/renderD128:rwm"]
            return (rocm_dir + "/llama-server", devices)

    # 3. Image-substring fallbacks (legacy convenience)
    if "vulkan" in image:
        vulkan_dir = backend.get("vulkan_dir", _DEFAULT_VULKAN_DIR)
        if os.path.isdir(str(vulkan_dir)):
            return (str(vulkan_dir) + "/llama-server", devices)
    elif "rocm" in image or "hip" in image:
        rocm_dir = backend.get("rocm_dir", _DEFAULT_ROCM_DIR)
        if os.path.isdir(str(rocm_dir)):
            devices = ["/dev/dri/card1:rwm", "/dev/dri/renderD128:rwm"]
            return (str(rocm_dir) + "/llama-server", devices)

    return None


def default_image_for_backend(backend_id: Optional[str]) -> str:
    """Derive a default image tag from the backend identifier."""
    if not backend_id:
        return "llama-strix-halo:vulkan"
    b = backend_id.lower()
    if any(k in b for k in ("rocm", "hip", "opencl")):
        return "llama-strix-halo:rocm"
    return "llama-strix-halo:vulkan"


def safe_container_name(name: str, port: int) -> str:
    """Generate a safe container name from model name and port."""
    safe = name.replace("_", "-").replace(".", "-")
    return f"llm-{safe}-{port}"


def _dict_to_cli(args_dict: Dict[str, Any]) -> List[str]:
    """Convert a flat args dict to CLI flag list (snake_case → kebab-case).

    Each key-value pair becomes two subprocess args:
      `--snake-case-key` `value`
    """
    cli: List[str] = []
    for key, value in args_dict.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            cli.extend([flag, str(value).lower()])
        else:
            cli.extend([flag, str(value)])
    return cli


def build_model_args(
    cm: Any,
    model_name: str,
    env_vars: Optional[Dict[str, Any]] = None,
    override_backend: Optional[str] = None,
) -> Optional[Tuple[List[str], str]]:
    """Build command arguments for a model using its backend config.

    Resolves the effective backend, fills template placeholders
    (CHECKPOINT, PORT), and concatenates backend arguments with
    model-specific arguments.  The ConfigManager is used only for data
    access — no method delegation required.

    Args:
        cm:               ConfigManager instance.
        model_name:       Model name as defined in the config file.
        env_vars:         Runtime environment variables for resolving
                          PORT and other placeholders.
        override_backend: Optional backend ID that overrides whichever
                          backend the model would normally use.

    Returns:
        A tuple of (arg_list, cmd_str) where arg_list is a list of
        individual arguments ready for subprocess_exec,
        and cmd_str is the space-joined string representation.
    """
    models = cm.get_vector("models")
    if not models:
        return None
    model = models.get(model_name)
    if model is None:
        return None

    # 1. Resolve which backend to use
    backend_id = _resolve_backend(cm, model, model_name, override_backend)

    # 2. Get backend definition
    backend = cm.get_backend(backend_id)
    if backend is None:
        raise RuntimeError(f"Backend '{backend_id}' not found in backends config")

    # 3. Resolve CHECKPOINT and PORT
    checkpoint = model.get("checkpoint", "")
    port = env_vars.get("PORT") if env_vars is not None else None
    if port is None:
        port = str(cm.data.get("models-start-port", 18000))

    # Build a temporary macro map for this resolution pass
    resolve_macros = dict(cm.data.get("macros", {}))
    resolve_macros["CHECKPOINT"] = checkpoint
    resolve_macros["PORT"] = port

    # 4. Resolve backend args — handle both string template and flat dict
    backend_args_raw = backend.get("args")
    if isinstance(backend_args_raw, str):
        # Legacy: expand macros via _resolve_string
        backend_args_resolved = cm._resolve_string(backend_args_raw, resolve_macros, strict=False)  # noqa: SLF001
        backend_arg_list = shlex.split(backend_args_resolved) if backend_args_resolved.strip() else []
    elif isinstance(backend_args_raw, dict):
        # New format: flat dict → CLI flags
        backend_arg_list = _dict_to_cli(backend_args_raw)
    else:
        backend_arg_list = []

    # 5. Get model args — handle both string and flat dict
    model_args_raw = model.get("args")
    if isinstance(model_args_raw, str):
        model_args_text = " ".join(model_args_raw.split())  # normalize whitespace
        model_arg_list = shlex.split(model_args_text) if model_args_text.strip() else []
    elif isinstance(model_args_raw, dict):
        # New format: flat dict → CLI flags
        model_arg_list = _dict_to_cli(model_args_raw)
    else:
        model_arg_list = []

    # Return merged list
    combined = backend_arg_list + model_arg_list
    return (combined, " ".join(combined))


def _resolve_backend(
    cm: Any,
    model: Dict[str, Any],
    model_name: str,
    override_backend: Optional[str] = None,
) -> str:
    """Determine which backend ID to use for a given model."""
    if override_backend:
        return override_backend

    model_backend = model.get("backend")
    if model_backend:
        return str(model_backend)

    global_default = cm.data.get("backends", {})
    if isinstance(global_default, dict):
        default_id = global_default.get("default")
        if default_id:
            return str(default_id)

    raise RuntimeError(
        f"No backend specified for model '{model_name}' "
        "(no override, no per-model 'backend' key, no 'backends.default')"
    )
