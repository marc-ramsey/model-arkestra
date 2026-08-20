"""Shared constants and utilities for container-based runners."""
from __future__ import annotations
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple, Union


import os
from pathlib import Path
# Subprocess env — convert os.environ to plain dict for uvloop compatibility
SUBPROCESS_ENV: Dict[str, str] = dict(os.environ)


def default_cache_root() -> Path:
    """Return a sensible default HuggingFace / GGUF model cache directory.

    Resolution order (avoids filling the root filesystem):
      1. ``HF_HUB_CACHE`` or ``LLAMA_CACHE`` environment variable
      2. ``$XDG_CACHE_HOME/huggingface`` (typically on a large data partition)
      3. ``/tmp/huggingface`` (on tmpfs — RAM disk, won't bloat root)

    Users can override entirely by setting the env var before starting the server.
    """
    for key in ("HF_HUB_CACHE", "LLAMA_CACHE"):
        val = os.environ.get(key)
        if val:
            return Path(val).expanduser()

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "huggingface"

    return Path("/tmp/huggingface")



INSPECT_RE = re.compile(r"^(exited|dead|paused|removing)\s*$", re.IGNORECASE)

# ── Known ROCm build directories, keyed by version string ────────────────
_ROCM_BUILD_MAP: Dict[str, str] = {}

# ── Default build dirs (sensible fallbacks — override via config or env) ────
_DEFAULT_VULKAN_DIR = "/usr/local/llama.cpp/build-vulkan-radv/bin"
_DEFAULT_ROCM_DIR = "/usr/local/llama.cpp/build-rocm/bin"
_DEFAULT_IMAGE = "ark-llama:vulkan-radv"


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
    """Derive a default image tag from the backend identifier.

    Resolution order:
      1. backends.<id>.image              — explicit per-backend
      2. backends.<default>.image         — global default backend
      3. hardcoded fallback               — ark-llama:vulkan-radv
    """
    try:
        import yaml, os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for candidate in ["sample-config.yaml", "config.yaml"]:
            path = os.path.join(project_root, candidate)
            if os.path.isfile(path):
                with open(path) as f:
                    cfg = yaml.safe_load(f) or {}
                backends = cfg.get("backends") or {}

                # 1. Explicit per-backend image
                if backend_id and isinstance(backends, dict):
                    be = backends.get(backend_id)
                    if isinstance(be, dict) and "image" in be:
                        return str(be["image"])

                    # 2. Global default backend's image
                    default_be_id = backends.get("default")
                    if default_be_id:
                        default_be = backends.get(default_be_id)
                        if isinstance(default_be, dict) and "image" in default_be:
                            return str(default_be["image"])

                # 3. Hardcoded fallbacks (legacy / programmatic usage)
                if backend_id and any(k in backend_id.lower() for k in ("rocm", "hip", "opencl")):
                    return "ark-llama:rocm"
                return _DEFAULT_IMAGE
    except Exception:
        pass
    # Hardcoded fallbacks
    if backend_id and any(k in backend_id.lower() for k in ("rocm", "hip", "opencl")):
        return "ark-llama:rocm"
    return "ark-llama:vulkan-radv"


def containerfile_for_backend(backend_id: Optional[str]) -> Optional[str]:
    """Return the container build file path for a backend's image.

    Reads backends.<id>.container. Returns None if not found.
    The caller should resolve relative to the project root.
    """
    try:
        import yaml, os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for candidate in ["sample-config.yaml", "config.yaml"]:
            path = os.path.join(project_root, candidate)
            if os.path.isfile(path):
                with open(path) as f:
                    cfg = yaml.safe_load(f) or {}
                backends = cfg.get("backends") or {}
                if isinstance(backends, dict) and backend_id:
                    be = backends.get(backend_id)
                    if isinstance(be, dict):
                        cf = be.get("container")
                        if cf:
                            return os.path.join("tests", "files", cf)
    except Exception:
        pass
    return None


def image_and_runner_for_backend(cm_data, backend_id: str) -> tuple[str, str]:
    """Resolve image tag and runner type for a backend from config.

    Reads backends.<id>.image and backends.<id>.runner (or falls through to
    backends.default.image). Returns (image_tag, runner_type).
    Runner type is one of: "podman", "docker", or "process".
    """
    backends = cm_data.get("backends") or {}
    runners_section = cm_data.get("runners") or {}

    # Resolve backend entry (explicit ID or default backend)
    be_id = backend_id if backend_id in (backends or {}) else backends.get("default", backend_id)
    backend = backends.get(be_id, {}) if isinstance(backends, dict) else {}
    if not isinstance(backend, dict):
        backend = {}

    # Resolve image: backend.image → default backend's image → hardcoded fallback
    image_tag = backend.get("image")
    if not image_tag:
        default_be_id = backends.get("default")
        if default_be_id and isinstance(backends, dict):
            default_be = backends.get(default_be_id)
            if isinstance(default_be, dict) and "image" in default_be:
                image_tag = str(default_be["image"])
    image_tag = image_tag or _DEFAULT_IMAGE

    # Resolve runner: explicit runner → runners section → process fallback
    runner_type = backend.get("runner")
    if not runner_type:
        runner_type = runners_section.get("default", "ProcessModelRunner").lower()
        # Map class name → runtime string
        runner_map = {"processmodelrunner": "process", "podmanmodelrunner": "podman",
                      "dockermodelrunner": "docker"}
        runner_type = runner_map.get(runner_type, runner_type)
    return image_tag, runner_type


def safe_container_name(name: str, port: int) -> str:
    """Generate a safe container name from model name and port."""
    safe = name.replace("_", "-").replace(".", "-")
    return f"llm-{safe}-{port}"


def _dict_to_cli(args_dict: Dict[str, Any]) -> List[str]:
    """Convert a flat args dict to CLI flag list (snake_case → kebab-case).

    Boolean True → ``--flag`` (presence-only).  False → omitted.
    All other values → ``--flag value``. Special key ``hf`` maps to ``-hf``
    for HuggingFace repo specification in llama.cpp.
    """
    cli: List[str] = []
    for key, value in args_dict.items():
        if key == "hf":
            # llama.cpp unified binary: -hf <repo> for HF model loading
            cli.extend(["-hf", str(value)])
        elif isinstance(value, bool):
            flag = f"--{key.replace('_', '-')}"
            if value:
                cli.append(flag)
        else:
            flag = f"--{key.replace('_', '-')}"
            cli.extend([flag, str(value)])
    return cli


def build_model_args(
    cm: Any,
    model_name: str,
    env_vars: Optional[Dict[str, Any]] = None,
    override_backend: Optional[str] = None,
    inference_kwargs: Optional[Dict[str, Any]] = None,
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
        inference_kwargs: Optional dict of runtime inference parameters
                          to merge into model args (last-wins).
                          String values containing ``${...}`` are macro-resolved.

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
        # Macro-resolve string values in dict before CLI conversion.
        resolved_backend: Dict[str, Any] = {}
        for key, val in backend_args_raw.items():
            if isinstance(val, str) and "${" in val:
                resolved_backend[key] = cm._resolve_string(val, resolve_macros, strict=False)  # noqa: SLF001
            else:
                resolved_backend[key] = val
        backend_arg_list = _dict_to_cli(resolved_backend)
    else:
        backend_arg_list = []

    # 5. Resolve defaults — handle both string template and flat dict
    defaults_raw = cm.data.get("defaults")
    if isinstance(defaults_raw, str):
        defaults_resolved = cm._resolve_string(defaults_raw, resolve_macros, strict=False)  # noqa: SLF001
        default_arg_list = shlex.split(defaults_resolved) if defaults_resolved.strip() else []
    elif isinstance(defaults_raw, dict):
        # Macro-resolve string values in dict before CLI conversion.
        resolved_defaults: Dict[str, Any] = {}
        for key, val in defaults_raw.items():
            if isinstance(val, str) and "${" in val:
                resolved_defaults[key] = cm._resolve_string(val, resolve_macros, strict=False)  # noqa: SLF001
            else:
                resolved_defaults[key] = val
        default_arg_list = _dict_to_cli(resolved_defaults)
    else:
        default_arg_list = []

    # 6. Get model args — handle both string and flat dict
    #    Merge inference_kwargs into model args (last-wins).
    model_args_raw = model.get("args")
    if isinstance(model_args_raw, str):
        model_args_text = " ".join(model_args_raw.split())  # normalize whitespace
        model_arg_list = shlex.split(model_args_text) if model_args_text.strip() else []
    elif isinstance(model_args_raw, dict):
        # Merge inference kwargs on top (last-wins for overlapping keys).
        merged_model = dict(model_args_raw)
        if inference_kwargs:
            for k, v in inference_kwargs.items():
                if k not in ("backend", "checkpoint"):
                    merged_model[k] = v
        # Macro-resolve any string values before CLI conversion.
        resolved_dict: Dict[str, Any] = {}
        for key, val in merged_model.items():
            if isinstance(val, str) and "${" in val:
                resolved_dict[key] = cm._resolve_string(val, resolve_macros, strict=False)  # noqa: SLF001
            else:
                resolved_dict[key] = val
        model_arg_list = _dict_to_cli(resolved_dict)
    else:
        model_arg_list = []

    # Return merged list: defaults → backend → model
    combined = default_arg_list + backend_arg_list + model_arg_list
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
