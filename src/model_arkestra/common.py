"""Shared constants and utilities for container-based runners."""
from __future__ import annotations
import re
import shlex
import subprocess as _subprocess
from typing import Any, Dict, List, Optional, Tuple


import os
from pathlib import Path
import yaml
# Subprocess env — convert os.environ to plain dict for uvloop compatibility
SUBPROCESS_ENV: Dict[str, str] = dict(os.environ)


def default_cache_root() -> Path:
    """Return a sensible default HuggingFace / GGUF model cache directory.

    Resolution order (avoids filling the root filesystem):
      1. ``HF_HUB_CACHE`` environment variable
      2. ``$XDG_CACHE_HOME/huggingface`` (typically on a large data partition)
      3. ``~/.cache/huggingface/hub`` (standard user cache directory)

    Users can override entirely by setting HF_HUB_CACHE before starting the server.
    """
    val = os.environ.get("HF_HUB_CACHE")
    if val:
        return Path(val).expanduser()

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "huggingface"

    # Standard user cache directory fallback
    home = Path.home()
    return home / ".cache" / "huggingface" / "hub"



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


def _resolve_backend_config_field(backend_id: Optional[str], field: str) -> Any:
    """Read a single field from backends.<id> or backends.<default>.

    Walks known config files, tries explicit backend_id first,
    then falls back to the default backend's value.
    Returns None if nothing found.
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for candidate in ["sample-config.yaml", "config.yaml"]:
            path = os.path.join(project_root, candidate)
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
            backends = cfg.get("backends") or {}
            if not isinstance(backends, dict):
                continue

            # 1. Explicit backend_id
            if backend_id:
                be = backends.get(backend_id)
                if isinstance(be, dict) and field in be:
                    return be[field]

            # 2. Default backend
            default_be_id = backends.get("default")
            if default_be_id:
                default_be = backends.get(default_be_id)
                if isinstance(default_be, dict) and field in default_be:
                    return default_be[field]
    except Exception:
        pass
    return None


def default_image_for_backend(backend_id: Optional[str]) -> str:
    """Derive a default image tag from the backend identifier.

    Resolution order:
      1. backends.<id>.image              — explicit per-backend
      2. backends.<default>.image         — global default backend
      3. hardcoded fallback               — ark-llama:vulkan-radv
    """
    image = _resolve_backend_config_field(backend_id, "image")
    if image:
        return str(image)
    # Hardcoded fallbacks (legacy / programmatic usage)
    if backend_id and any(k in backend_id.lower() for k in ("rocm", "hip", "opencl")):
        return "ark-llama:rocm"
    return _DEFAULT_IMAGE


def containerfile_for_backend(backend_id: Optional[str]) -> Optional[str]:
    """Return the container build file path for a backend's image.

    Reads backends.<id>.container. Returns None if not found.
    The caller should resolve relative to the project root.
    """
    cf = _resolve_backend_config_field(backend_id, "container")
    if cf:
        return os.path.join("tests", "files", cf)
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


def _run_subprocess(cmd: list, timeout: int = 30) -> _subprocess.CompletedProcess:
    """Run a subprocess call via asyncio.to_thread.

    Uses DEVNULL for stdin and temp files for stdout/stderr to avoid pipe
    inheritance deadlocks with podman/buildah nested child processes.
    """
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as tmpdir:
        stdout_path = os.path.join(tmpdir, "stdout")
        stderr_path = os.path.join(tmpdir, "stderr")
        proc = _subprocess.Popen(
            cmd,
            stdin=_subprocess.DEVNULL,
            stdout=open(stdout_path, "w"),
            stderr=open(stderr_path, "w"),
        )
        try:
            proc.wait(timeout=timeout)
        except _subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        with open(stdout_path) as f:
            stdout = f.read()
        with open(stderr_path) as f:
            stderr = f.read()
    return _subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _runtime_binary(runner_type: str) -> Optional[str]:
    """Return the CLI binary name for a runner type, or None."""
    return {"podman": "podman", "docker": "docker"}.get(runner_type)


def image_exists(runner: str, tag: str) -> bool:
    """Check if a container image exists locally for the given runner.

    Uses runtime-native existence check (exit 0 = present).

    Args:
        runner: "podman" or "docker"
        tag: full image tag (e.g. "ark-llama:rocm")

    Returns:
        True if the image is present in the local store.
    """
    cmd = {"podman": ["podman", "image", "exists", tag],
           "docker": ["docker", "inspect", tag]}.get(runner, [])
    proc = _run_subprocess(cmd, timeout=10)
    return proc.returncode == 0


def build_image(
    runner: str,
    image_tag: str,
    containerfile: str,
    context_dir: str,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Build a container image for the given runner.

    Args:
        runner: "podman" or "docker"
        image_tag: target image tag (e.g. "ark-llama:rocm")
        containerfile: path to Containerfile/Dockerfile
        context_dir: build context directory
        timeout: max seconds to wait

    Returns:
        Dict with success status and combined stdout/stderr output.
    """
    cmd = f"{runner} build -t {image_tag} -f {containerfile} {context_dir}".split()
    proc = _run_subprocess(cmd, timeout=timeout)
    return {
        "success": proc.returncode == 0,
        "output": proc.stdout + proc.stderr,
        "error": None if proc.returncode == 0 else proc.stderr or proc.stdout or "non-zero exit",
    }


def remove_image(
    runner: str,
    tag: str,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Remove a container image for the given runner.

    Args:
        runner: "podman" or "docker"
        tag: image tag to remove (e.g. "ark-llama:rocm")
        timeout: max seconds to wait

    Returns:
        Dict with removed status and any error output.
    """
    cmd = f"{runner} rmi -f {tag}".split()
    proc = _run_subprocess(cmd, timeout=timeout)
    return {
        "removed": proc.returncode == 0,
        "error": None if proc.returncode == 0 else proc.stderr,
    }


def safe_container_name(name: str, port: int) -> str:
    """Generate a safe container name from model name and port."""
    safe = name.replace("_", "-").replace(".", "-")
    return f"llm-{safe}-{port}"


def _dict_to_cli(args_dict: Dict[str, Any]) -> List[str]:
    """Convert a flat args dict to CLI flag list (snake_case → kebab-case).

    Boolean True → ``-flag`` or ``--flag`` (presence-only).  False → omitted.
    All other values → ``-flag value`` or ``--flag value``.
    Keys in _LLAMA_SHORT_FLAGS get a single `-`; everything else gets `--`.
    """
    # Set of config keys whose llama.cpp flag is -x (single dash), not --x
    _LLAMA_SHORT_FLAGS = {
        'c', 't', 'tb', 'fa', 'e', 'kvo',
        'ctk', 'ctv', 'dt', 'dio', 'lm', 'dev',
        'ot', 'cmoe', 'ncmoe',
        'ngl', 'sm', 'ts', 'mg', 'fit', 'fitt',
        'fitc', 'b', 'ub', 'hf', 'hff', 'hft',
        'dr', 'mu', 'cl',
        'a', 'ag', 'mm', 'mmu',
        's', 'l', 'j', 'jf',
        'bs', 'lcs', 'lcd',
        'ctxcp', 'cms', 'cram',
        'kvu', 'r', 'sp',
        'np', 'cb', 'to',
        'rea', 'sps', 'v', 'lv', 'm',
    }

    cli: List[str] = []
    for key, value in args_dict.items():
        kebab = key.replace('_', '-')
        prefix = '-' if kebab in _LLAMA_SHORT_FLAGS else '--'
        if isinstance(value, bool):
            if value:
                cli.append(f"{prefix}{kebab}")
        else:
            cli.extend([f"{prefix}{kebab}", str(value)])
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
    resolve_macros["NPROC"] = str(os.cpu_count())

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

# ── Engine resolution helpers ───────────────────────────────────
def _resolve_engine(cm: Any, engine_name: Optional[str] = None) -> Dict[str, Any]:
    """Resolve an engine config dict by name from the ``engines:`` section.

    Falls back to the first (and typically only) engine registered when
    *engine_name* is not provided.
    """
    engines = (cm.data or {}).get("engines") or {}
    if isinstance(engines, dict):
        if engine_name and engine_name in engines:
            eng = engines[engine_name]
            return eng if isinstance(eng, dict) else {}
        # No name specified — fall back to the first registered engine
        for eid, ecfg in engines.items():
            if isinstance(ecfg, dict):
                return ecfg
    return {}

def _merge_engine_defaults(engine_cfg: Dict[str, Any], backend_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge engine-level defaults into backend config.

    Backend values override engine values at the top level.  The ``args``
    sub-dict is deep-merged so that nested keys from the engine provide
    fallbacks while backend-specific args take precedence.
    """
    merged: Dict[str, Any] = {}
    # Top-level scalar/string fields — engine fallback → backend override
    for key in (engine_cfg or {}):
        merged[key] = engine_cfg.get(key)  # start with engine value
    for key in (backend_cfg or {}):
        if key == "args" and isinstance(merged.get("args"), dict):
            # Deep-merge args dicts: engine values are fallbacks
            deep_args = {**merged["args"], **backend_cfg["args"]}
            merged[key] = deep_args
        else:
            merged[key] = backend_cfg[key]
    return merged
