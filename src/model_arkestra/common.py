"""Shared constants and utilities for container-based runners."""
from __future__ import annotations
import os
import re
import subprocess as _subprocess
from dataclasses import dataclass, field
from model_arkestra.gpu_detect import detect_all
from pathlib import Path
import yaml
from typing import Any, Dict, List, Optional, Tuple
INFRA_KEYS = frozenset({
    'backend', 'runner', 'tags', 'max-log-lines',
})


# ── Model resolution ────────────────────────────────────────────────────────

@dataclass
class ModelRef:
    """Fully resolved model reference.

    Attributes:
        ref:          Fully qualified path, e.g. ``unsloth/Qwen3.5:Q4_K_M``
                      or ``/local/path/file.gguf``.
        repo:         Source type — ``"hf"`` or ``"lcl"``.
        cache_path:   Repo portion for HF cache dir computation,
                      e.g. ``unsloth--Qwen3.5``, or empty string for lcl.
    """
    ref: str
    repo: str            # "hf" | "lcl"
    cache_path: str = ""  # owner--model for HF, empty for lcl


def resolve_model_ref(
    raw: Optional[str],
    default_section: Dict[str, Any],
    model_repos: Optional[Dict[str, Any]] = None,
    backend_repo: Optional[str] = None,
) -> ModelRef:
    """Resolve a model reference through the full resolution chain.

    Syntax: ``[<ark-path>:][<repo>:]<model>[:<quant>]``

    Resolution chain for repo type (hf vs lcl):
      1. Parse raw string — bare ``/path`` → lcl, else tentatively hf
      2. Model-level ``repo:`` override if present in model dict
      3. Backend-level ``repo:`` from backend args dict
      4. Default ``model-repo`` from default section
      5. Hardwired fallback: "hf"

    Args:
        raw:           Raw ref from config ``model:"` field.
        default_section: The ``default:"` section from config data.
        model_repos:   Alias registry (e.g. ``{"fb": {"name": "foo-bar"}}``).
        backend_repo:  Optional repo type from backend args dict.

    Returns:
        Fully resolved ``ModelRef`` with all fields populated.
    """
    if not raw or not raw.strip():
        return ModelRef(ref="", repo="hf", cache_path="")

    raw = raw.strip()

    # --- Step 1: Handle local paths immediately ---
    if raw.startswith("lcl:"):
        return ModelRef(ref=raw, repo="lcl", cache_path="")
    if raw.startswith("/"):
        return ModelRef(ref=f"lcl:{raw}", repo="lcl", cache_path="")

    # --- Step 2: Parse raw string into components ---
    colon_idx = raw.find(":")
    if colon_idx == -1:
        # No colon — "owner/model" or just "model"
        slash_idx = raw.find("/")
        if slash_idx != -1:
            owner, model_name = raw[:slash_idx], raw[slash_idx + 1:]
            quant = ""
        else:
            owner = None
            model_name = raw
            quant = ""
    else:
        prefix, rest = raw[:colon_idx], raw[colon_idx + 1:]

        # Check if prefix is a registered alias
        if model_repos and prefix in model_repos:
            owner = model_repos[prefix].get("name", prefix)
            model_name, quant = _split_quant(rest)
        elif "/" in rest:
            # "alias/owner:model" — split after first slash
            slash_idx = rest.find("/")
            owner, model_with_quant = rest[:slash_idx], rest[slash_idx + 1:]
            model_name, quant = _split_quant(model_with_quant)
        elif "/" in prefix:
            # "owner/model:quant" — split prefix at first slash
            slash_idx = prefix.find("/")
            owner = prefix[:slash_idx]
            model_name = prefix[slash_idx + 1:]
            quant = rest if rest else ""
        else:
            # No "/" in either part — this is "modelname:quant"
            owner = None
            model_name, quant = _split_quant(raw)

    # --- Step 3: Resolve defaults through chain ---
    default_repo = str(default_section.get("model-repo", "") or "")
    default_quant = str(default_section.get("model-quant", "") or "")
    effective_repo = owner if owner else (backend_repo or default_repo)
    final_quant = quant or default_quant

    # --- Step 4: Build fully qualified ref ---
    if effective_repo and model_name:
        if "/" in effective_repo:
            # Owner already has slash — use as-is, append model if needed
            ref = f"{effective_repo}"
        else:
            ref = f"{effective_repo}/{model_name}"
    elif effective_repo:
        ref = effective_repo
    else:
        ref = model_name or ""

    # Append quantifier
    if final_quant and not ref.endswith(f":{final_quant}"):
        ref = f"{ref}:{final_quant}"

    # --- Step 5: Compute cache path (for HF refs only) ---
    cache_path = ""
    repo_type = "hf" if effective_repo else "lcl"
    if repo_type == "hf":
        hf_ref = ref.rsplit(":", 1)[0]  # strip quant suffix
        parts = hf_ref.split("/", 1)
        cache_path = parts[0].replace("/", "--") + "--" + parts[1] if len(parts) == 2 else ""

    return ModelRef(ref=ref, repo=repo_type, cache_path=cache_path)


def _split_quant(s: str) -> Tuple[str, str]:
    """Split ``name:quant`` into (name, quant), or (s, "") if no colon."""
    idx = s.find(":")
    if idx != -1:
        return s[:idx], s[idx + 1:]
    return s, ""
# Subprocess env — convert os.environ to plain dict for uvloop compatibility
SUBPROCESS_ENV: Dict[str, str] = dict(os.environ)

# ── Default config directory ───────────────────────────────────────────────
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "arkestra"


def resolve_config_path(config_path: Optional[str] = None) -> Path:
    """Resolve config.yaml path from explicit arg or default location.

    Resolution order:
      1. Explicit ``config_path`` argument (absolute or relative)
      2. ``DEFAULT_CONFIG_DIR / 'config.yaml'``

    Does NOT create the directory — that's left to ConfigManager which will
    raise FileNotFoundError with a clear message pointing users to
    sample-config.yaml for scaffolding.
    """
    if config_path:
        return Path(config_path).expanduser()
    return DEFAULT_CONFIG_DIR / "config.yaml"


def resolve_backends_path(config_dir: Optional[str] = None) -> Optional[Path]:
    """Resolve backends.yaml path from explicit arg or default location.

    Resolution order:
      1. Explicit ``config_dir`` argument → ``{config_dir}/backends.yaml``
      2. ``DEFAULT_CONFIG_DIR / 'backends.yaml'``

    Returns None if neither resolves to an existing file (caller should
    treat as "no backends configured" and fall back to Containerfile builds).
    """
    base = Path(config_dir).expanduser() if config_dir else DEFAULT_CONFIG_DIR
    path = base / "backends.yaml"
    return path if path.exists() else None


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


def download_onnx_model(
    repo_id: str,
    pattern: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download an ONNX model from HuggingFace into the standard HF_HUB cache.

    Mirrors the llama.cpp GGUF caching convention — uses the same
    ``HF_HUB_CACHE`` / ``XDG_CACHE_HOME/huggingface`` directory tree so all
    models live in one place.

    Args:
        repo_id:      HuggingFace repo ID (e.g. "Xenova/bge-small-en-v1.5").
        pattern:      Single glob pattern to download (shorthand for allow_patterns).
        allow_patterns: List of glob patterns to include. Defaults to ``onnx/*.onnx``.
        cache_dir:    Override HF_HUB_CACHE location. Uses ``default_cache_root()``
                      if not specified.

    Returns:
        Path to the downloaded ONNX model file.

    Example::

        >>> download_onnx_model("Xenova/bge-small-en-v1.5")
        # Downloads into ~/.cache/huggingface/hub/models--Xenova--bge-small-en-v1.5/
        # Returns: Path('/.../onnx/model.onnx')
    """
    if cache_dir is None:
        cache_dir = default_cache_root()

    from huggingface_hub import snapshot_download

    # Default to ONNX artifacts only
    if not allow_patterns and not pattern:
        allow_patterns = ["onnx/*.onnx", "tokenization*", "tokenizer.json", "vocab*"]

    snap = snapshot_download(
        repo_id,
        allow_patterns=allow_patterns,
        cache_dir=str(cache_dir),
    )

    # Find the .onnx file inside the snapshot
    for p in Path(snap).rglob("*.onnx"):
        return p

    raise FileNotFoundError(
        f"No *.onnx found in '{repo_id}' snapshots at {snap}. "
        f"Use 'pattern' or 'allow_patterns' to match specific files."
    )


def resolve_onnx_model_path(
    model_ref: str,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Resolve an ONNX model reference to a concrete file path.

    Accepts either a local filesystem path or a HuggingFace repo ID.

    Args:
        model_ref: Either an absolute/relative path to ``*.onnx``, or a HF repo ID.
        cache_dir: Cache directory for downloads. Defaults to ``default_cache_root()``.

    Returns:
        Absolute path to the ONNX model file.
    """
    ref_path = Path(model_ref)
    if ref_path.exists():
        return ref_path.resolve()

    # Treat as repo ID — download into HF cache
    try:
        return download_onnx_model(model_ref, cache_dir=cache_dir)
    except ImportError:
        raise FileNotFoundError(
            f"'{model_ref}' does not exist on disk and huggingface_hub is not installed. "
            f"Install with: pip install 'model-arkestra[onnx]'"
        )



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
        runner_type = runners_section.get("default", "process")
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





def _resolve_arg(model_data: Dict, backend_cfg: Dict, default_section: Dict,
                 key: str):
    """Resolve one key through the unified chain.

    Resolution order:
      1. Model root field (``model_data[key]``)
      2. Backend args (``backend_args["key"]``)
      3. Default section (``default.key``)
      4. None — caller skips missing values
    """
    for v in (model_data.get(key), backend_cfg.get("args", {}).get(key),
              default_section.get(key)):
        if v is not None and v != "":
            return v
    return None


def build_model_args(
    cm: Any,
    model_name: str,
    inference_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Merge all config args through unified resolution chain.

    Collects keys from model root (excluding infra). Resolves each value from:
    model root → backend.args → default section. Runtime inference_kwargs override.

    Infra keys (``backend``, ``runner``, ``tags``, ``max-log-lines``) are skipped.

    Args:
        cm:               ConfigManager instance.
        model_name:       Model name as defined in the config file.
        inference_kwargs: Optional dict of runtime inference parameters
                          to merge into model args (last-wins).

    Returns:
        A flat dict of merged key→value pairs, or None if model not found.
    """
    models = cm.data.get("models")
    if not models:
        return None
    model = models.get(model_name)
    if model is None:
        return None

    result: Dict[str, Any] = {}
    default_section = cm.data.get("default") or {}
    backend_id = model.get("backend") or _resolve_backend(cm, model, model_name)
    backend_cfg = (cm.data.get("backends") or {}).get(backend_id, {})
    if not isinstance(backend_cfg, dict):
        backend_cfg = {}

    # ── Collect all unique keys from model root (skip infra) ─────────
    keys: set[str] = {k for k in model if k not in INFRA_KEYS}

    # ── Resolve each key through unified chain ───────────────────────
    for key in keys:
        val = _resolve_arg(model, backend_cfg, default_section, key)
        if val is not None:
            result[key] = val

    # ── Runtime kwargs override everything (skip infra) ──────────────
    if inference_kwargs:
        for k, v in inference_kwargs.items():
            if k not in INFRA_KEYS:
                result[k] = v
    return result


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

    # Check top-level flat config key: backend-default (legacy)
    flat_default = cm.data.get("backend-default")
    if flat_default:
        return str(flat_default)

    # Check new schema: default.backend
    default_section = cm.data.get("default", {}) or {}
    new_default = default_section.get("backend")
    if new_default:
        return str(new_default)

    # Nested backends.default key
    backends_section = cm.data.get("backends", {})
    if isinstance(backends_section, dict):
        default_id = backends_section.get("default")
        if default_id:
            return str(default_id)

    # Ultimate fallback — matches BaseModelRunner._DEFAULT_BACKEND
    return "cpu"

# ── Capability resolution helpers ───────────────────────
def resolve_tags(model_cfg: Dict | None, global_cfg: Dict,
                 backend_id: str | None = None) -> list[str]:
    """Resolve available capability tags using the normal chain:

    1. Per-model ``tags`` (explicit override)
    2. Backend-declared ``backends.<id>.capabilities``
    3. Engine-declared ``engines.<name>.capabilities``
    4. Hardcoded fallback ``["chat"]``
    """
    if model_cfg and model_cfg.get("tags"):
        return list(model_cfg["tags"])

    b = (global_cfg or {}).get("backends") or {}
    if isinstance(b, dict):
        bid = backend_id or (model_cfg or {}).get("backend") or b.get("default")
        if bid:
            caps = (b.get(str(bid)) or {}).get("capabilities")
            if isinstance(caps, list) and caps:
                return list(caps)

    be_id = backend_id or (model_cfg or {}).get("backend", "")
    bcfg = (global_cfg.get("backends") or {}).get(str(be_id), {})
    engine_name = (bcfg if isinstance(bcfg, dict) else {}).get("engine")
    if engine_name:
        engines = (global_cfg or {}).get("engines") or {}
        eng_caps = (engines.get(engine_name) or {}).get("capabilities")
        if isinstance(eng_caps, list) and eng_caps:
            return list(eng_caps)

    return ["chat"]

# ── Engine resolution helpers ───────────────────────────────────
def _get_device_profile_env(
    cm: Any,
) -> Dict[str, Any]:
    """Detect GPU and return matching device-profile env vars.

    Same detection logic as `_merge_device_profile_args`, but returns
    the ``env`` sub-dict instead of ``args``.
    """
    result = detect_all()
    primary = result.get("primary_gpu")
    if not primary:
        return {}

    profiles: Dict[str, Any] = {}
    for engine_cfg in (cm.data.get("engines") or {}).values():
        if isinstance(engine_cfg, dict) and "device-profiles" in engine_cfg:
            profiles.update(engine_cfg["device-profiles"])
    if not profiles:
        return {}

    vendor = primary.get("vendor", "")
    matched_key: Optional[str] = None

    if vendor == "amd":
        gfx = result.get("gfx_family")
        if gfx and gfx in profiles:
            matched_key = gfx
        elif "rocm" in profiles:
            matched_key = "rocm"
    elif vendor == "nvidia":
        gpu_name = primary.get("name", "").lower()
        for key in profiles:
            if any(part in gpu_name for part in key.replace('-', ' ').split()):
                matched_key = key
                break
        if not matched_key and "cuda" in profiles:
            matched_key = "cuda"
    elif vendor == "intel":
        if "vulkan" in profiles:
            matched_key = "vulkan"

    if matched_key is None:
        return {}

    prof = profiles.get(matched_key, {})
    return prof.get("env") or {}
