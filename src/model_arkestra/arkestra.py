"""Single entry point for all model operations — wraps ConfigManager + lazy runners."""
from __future__ import annotations
import asyncio
import logging
import os
import shutil
import yaml
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

from model_arkestra.config_manager import ModelConfigManager
from model_arkestra.gpu_detect import has_rocm, has_vulkan, has_nvidia, detect_all as _detect_all
from model_arkestra.base import BaseRunner
from model_arkestra.common import (
    _resolve_backend, _resolve_device_profile, default_cache_root,
    resolve_config_path, image_and_runner_for_backend, resolve_model_ref,
    resolve_tags as _resolve_model_tags,
)
from model_arkestra.docker import DockerRunner
from model_arkestra.onnx_runner import OnnxRunner
from model_arkestra.podman import PodmanRunner
from model_arkestra.process import ProcessRunner
from model_arkestra.remote import RemoteRunner
from model_arkestra.types import RunnerState, _Model
from model_arkestra.unicode_ringbuffer import UnicodeRingBuffer
from model_arkestra.http_proxy import model_status_for_ctx

logger = logging.getLogger(__name__)


class ModelArkestra:
    """Start/ainvoke/astream/stop all models through one object."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        start_port: int = 18000,
        backends_config: Optional[Dict[str, Any]] = None,
        local_url: str = "",
        **runner_kwargs: Any,
    ):
        # Resolve config path — defaults to ~/.config/arkestra/config.yaml
        self._config_path = resolve_config_path(config_path)
        self._cm = ModelConfigManager(str(self._config_path))

        # Single source of truth: backends.yaml (base) merged with config.yaml (overlay).
        # Nested dicts deep-merge; top-level keys coexist.
        backends_path = Path(self._config_path).parent / "backends.yaml"
        base = (backends_config if backends_config is not None
                else (yaml.safe_load(open(backends_path)) or {} if backends_path.exists() else {}))
        self._cm.merge(base)
        default_section = self._cm.get("default", {})

        # ── Registry: name→Model ownership, cluster routing, port pool ──
        from model_arkestra.models import Registry
        self._registry = Registry(self._cm, local_url=local_url)

        self._runners: Dict[str, BaseRunner] = {}
        self._runner_kwargs = runner_kwargs
        # ── Global log buffer (single ring for all server-level events) ─
        app_log_lines = int(self._cm.get("default/app-log-lines", 2000))
        self._global_log_buf = UnicodeRingBuffer(app_log_lines * _Model.AVG_LINE_BYTES)
        self._global_log_seq: int = 0
        # Extract sources section for binary_downloader compatibility
        self._sources: Dict[str, Any] = self._cm.get("sources", {})
        # ── Hardware detection (GPU/CPU, single init-time query) ───
        self._validate_backend_runtime()
        self._device_profile: Optional[Dict[str, Any]] = None
        self._hardware_detection: Optional[Dict[str, Any]] = None
        # ── Pre-create contexts for all configured models ──────────
        self._pre_create_model_contexts()

    # ── port allocation (global) ───────────────────────────────────────
    def log(self, text: str, level: str = "INFO") -> None:
        """Log a line — prints to terminal with ANSI colors (uvicorn style), writes plain text to ring buffer."""
        _COLORS = {"INFO": "36", "WARNING": "33", "ERROR": "31", "DEBUG": "90"}
        color = _COLORS.get(level, "37")

        # Pad level + colon to 8 chars (matches uvicorn: "INFO:   ", "WARNING: ", etc.)
        prefix = f"{level}:".ljust(10)
        colored_text = f"\033[{color}m{prefix}\033[0m{text}"  # noqa: PLR2004

        print(colored_text, flush=True)

        # Store plain text in ring buffer (no ANSI codes)
        self._global_log_seq += 1
        if not text.endswith("\n"):
            text = text + "\n"
        for _ in range(20):
            try:
                self._global_log_buf.write(self._global_log_seq, text)
                break
            except UnicodeRingBuffer.BufferFullError:
                if not self._global_log_buf:
                    return
                self._global_log_buf.read_entries(max_lines=1)

    # ── port allocation (global) ───────────────────────────────────────
    def worker_port(self, model_name: str) -> int:
        """Allocate a port for *model_name* (delegates to the registry pool).

        Reuses the port from an existing context (stop→restart), otherwise
        allocates a fresh port from the pool.
        """
        return self._registry.allocate_port(model_name)
    
    # ── backend runtime validation (warn + fallback) ────────────────
    def _validate_backend_runtime(self) -> None:
        """Ensure the configured default backend's runtime is available.

        If it is missing, warn and record a detected fallback (CPU-capable)
        in the config manager's runtime-only override. Downstream resolution
        reads ``effective_default_backend()`` so every reader picks a usable
        backend instead of hard-erroring at startup. The user's config file
        is left untouched.
        """
        backends = self._cm.get("backends", {})
        if not isinstance(backends, dict):
            return  # no backends section — skip validation
        backend_id = backends.get("default")
        if not backend_id:
            return  # no default backend set — skip validation

        runtime_checks = {
            "vulkan-radv": has_vulkan,
            "rocm": has_rocm,
            "cuda": has_nvidia,
        }
        checker = runtime_checks.get(backend_id)
        if checker and not checker():
            fallback, reason = (self.device_detection.get("recommendation") or ("cpu", ""))
            self._cm._effective_default_backend = fallback
            logger.warning(
                f"Backend '{backend_id}' runtime not detected — "
                f"falling back to '{fallback}' ({reason})."
            )



    # ── device profile resolution (single init-time query) ───────
    def _get_device_profile(self) -> Dict[str, Any]:
        """Lazy-cached GPU detection. Called once on first access."""
        if self._device_profile is None:
            self._device_profile = _resolve_device_profile(self.cm)
        return self._device_profile

    @property
    def device_profile(self) -> Dict[str, str]:
        """GPU device-profile env vars (empty dict if no GPU matched)."""
        return self._get_device_profile().get("env", {})

    @property
    def models(self) -> Dict[str, Any]:
        """Name → Model map. The single source of truth for model state.

        Includes alias names: several model names may map to the same shared
        context (models referencing one checkpoint).
        """
        result: Dict[str, Any] = {}
        for name in self._registry.model_names:
            ckpt_id = self._registry.get_checkpoint_id(name)
            if ckpt_id:
                ctx = self._registry.get_context_by_checkpoint(ckpt_id)
                if ctx:
                    result[name] = ctx
        return result

    @property
    def device_detection(self) -> Dict[str, Any]:
        """Raw GPU/CPU detection result — cached once at first access."""
        if self._hardware_detection is None:
            self._hardware_detection = _detect_all()
        return self._hardware_detection

    # ── hardware detection (public API) ────────────────────────
    @property
    def hardware(self) -> Dict[str, Any]:
        """Full GPU/CPU detection result — cached once at first access.

        Returns:
            Dict with keys: gpus, primary_gpu, primary_backend, gfx_family,
            multi_gpu_warn, cpu, has_runtime, recommendation, warnings.
        """
        return self.device_detection

    def _pre_create_model_contexts(self) -> None:
        """Create a _Model for every configured model.

        Ports are allocated, backends resolved, and state set based on
        cache existence.  This ensures model_obj() always returns a
        valid context for any configured model — no more None gaps.
        """
        hf_cache = self.resolve_config("hf_hub_cache") or str(default_cache_root())
        models_cfg = self._cm.get("models", {})
        if not isinstance(models_cfg, dict):
            return

        default_section = (self._cm.data.get("default") or {})

        # Group model names by the checkpoint they reference. Models sharing a
        # checkpoint become ONE process (one _Model, one set of weights/KV);
        # each model name is an alias for that shared context.
        ckpt_to_models: Dict[str, list] = {}
        standalone: list = []
        for model_name in models_cfg:
            ckpt_id = self._cm.checkpoint_for(model_name)
            if ckpt_id:
                ckpt_to_models.setdefault(ckpt_id, []).append(model_name)
            else:
                standalone.append(model_name)

        # Build the list of (context_name, [model aliases]) to create.
        groups: list = [(m, [m]) for m in standalone]
        groups += [(ckpt_id, names) for ckpt_id, names in ckpt_to_models.items()]

        for ctx_name, aliases in groups:
            # Use the first alias's config; all share the same checkpoint so the
            # merged load-profile (ref, backend, parallel, ...) is identical.
            primary = aliases[0]
            model_cfg = self.get_model(primary) or {}

            # Resolve backend and runner type
            backend_id = _resolve_backend(self._cm, model_cfg, primary)
            cm_data = self._cm.data
            _, runner_type = image_and_runner_for_backend(cm_data, backend_id)

            # Only pre-create contexts for runners that have a process-level
            # lifecycle (not remote/ONNX which manage their own context logic).
            if runner_type in ("remote", "onnx"):
                continue

            # Determine initial state based on cache existence
            resolved = resolve_model_ref(
                raw=model_cfg.get("model", ""),
                default_section=default_section,
                model_repos=self._cm.data.get("model-repos"),
            )
            is_cached = False
            if resolved.cache_path:
                cache_path = Path(hf_cache).expanduser() / f"models--{resolved.cache_path}"
                # A usable cache has at least one model blob (GGUF or ONNX)
                snapshots_dir = cache_path / "snapshots"
                blobs_dir = cache_path / "blobs"
                if snapshots_dir.exists():
                    is_cached = any(snapshots_dir.glob("*/*.gguf")) or any(snapshots_dir.glob("*/*.onnx"))
                if not is_cached and blobs_dir.exists():
                    is_cached = any(blobs_dir.glob("*.gguf")) or any(blobs_dir.glob("*.onnx"))

            # Create the shared context — named by checkpoint id (or model name
            # for standalone). Port assigned at first start only.
            from model_arkestra.types import _Model
            ctx = _Model(ctx_name, None, max_log_lines=500)
            ctx.backend_id = backend_id
            ctx.runner_type = runner_type
            if not is_cached:
                ctx._state = RunnerState.UNCACHED   # construction-time init
            if resolved.cache_path:
                cache_root = default_cache_root()
                ctx._cache_dir = cache_root / f"models--{resolved.cache_path}"

            # Register once (single owner) and alias under every model name that
            # shares this checkpoint, so existing per-name lookups keep working.
            self._registry.register(ctx_name, ctx, aliases)
            runner = self.get_runner_instance(runner_type, primary)
            runner._ctx = ctx


    # ── cluster topology (delegates to the registry) ───────────────
    @property
    def clusters(self) -> Dict[str, Dict[str, Any]]:
        return self._registry.clusters

    @property
    def local_cluster_key(self) -> str:
        return self._registry.local_cluster_key

    def reload_clusters(self) -> None:
        """Re-read cluster config from cm (after add/remove)."""
        self._registry.reload_clusters()

    def _parse_cluster_prefix(self, model_name: str) -> Tuple[str, str]:
        """Split ``<cluster>/<model-id>``; no prefix → local cluster."""
        return self._registry.parse_prefix(model_name)

    def resolve_model_cluster_addr(self, model_name: str) -> Tuple[str, Optional[str], str]:
        """Return ``(cluster_name, base_url|None, local_model_id)``.

        Local cluster → base_url None. Remote cluster → worker URL to proxy.
        """
        return self._registry.resolve(model_name)

    def local_model_name(self, model_name: str) -> str:
        """Strip a ``<cluster>/`` prefix; unknown prefixes pass through."""
        return self._registry.local_model_name(model_name)

    # ── ConfigManager delegation ───────────────────────────────────────
    @property
    def cm(self) -> ConfigManager:
        return self._cm

    def get_model(self, model_name: str, env_vars: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        return self._cm.get_model(model_name, env_vars)

    def get_models(self) -> list:
        return self._cm.get_models()

    def get_backend(self, backend_id: str) -> Optional[Dict[str, Any]]:
        # Check backends.yaml first (preferred), then config.yaml (legacy)
        be = self._cm.get(f"backends/{backend_id}") if backend_id else None
        if be and isinstance(be, dict):
            return be
        # Fall back to config.yaml for legacy inline backend definitions
        return self._cm.get_backend(backend_id)

    # ── model introspection (runtime state) ────────────────────────────

    def model_obj(self, model_name: str) -> Optional[_Model]:
        """Return the live _Model for *model_name* (cluster-prefix aware), or None.

        Distinct from ``get_model()`` which returns the static config dict.
        """
        local_name = self.local_model_name(model_name)
        return self._registry.get_context(local_name)

    def _provider_for(self, model_name: str):
        """Return the inference Provider for *model_name* (raises if unknown)."""
        ctx = self.model_obj(model_name)
        if ctx is None:
            from model_arkestra.types import ModelNotStarted
            raise ModelNotStarted(self.local_model_name(model_name))
        return ctx.provider

    def get_v1_models(self) -> Dict[str, Any]:
        """OpenAI-compatible ``/v1/models`` response with WebUI status fields."""
        from time import time

        contexts_by_name = {ctx.name: ctx for ctx in self.models.values()}
        data = []
        for model_name in self.get_models():
            # Skip remote-cluster models (not tracked locally)
            cluster_name, local_name = self._parse_cluster_prefix(model_name)
            if cluster_name != self.local_cluster_key:
                continue
            ctx = contexts_by_name.get(model_name)
            model_cfg = self.get_model(model_name) or {}
            owned_by = str(model_cfg.get("owned_by", "local")) if isinstance(model_cfg, dict) else "local"

            entry: Dict[str, Any] = {
                "name": model_name,
                "object": "model",
                "created": int(time()),
                "owned_by": owned_by,
                "status": model_status_for_ctx(ctx),
                "context_length": self._resolve_context_length(model_name),
            }

            data.append(entry)

        return {"object": "list", "data": data}

    def _resolve_context_length(self, model_name: str) -> Union[int, str]:
        """Resolved ``ctx-size`` for a model, or ``"n/a"`` when unresolvable.

        Uses the same config chain as process launch (model → backend →
        default). Unexpanded placeholders and non-llama models yield "n/a".
        """
        try:
            from model_arkestra.common import build_model_args
            merged = build_model_args(self._cm, model_name)
            val = merged.get("ctx-size") if merged else None
            if isinstance(val, int):
                return val
            if isinstance(val, str) and val.isdigit():
                return int(val)
        except Exception:
            pass
        return "n/a"

    # ── runner class map — one hop, no magic ─────────────────────────

    _RUNNER_CLASSES: Dict[str, type] = {
        "process": ProcessRunner,
        "podman": PodmanRunner,
        "docker": DockerRunner,
        "onnx": OnnxRunner,
        "remote": RemoteRunner,
    }

    def get_runner_instance(self, runner_type: str, model_name: Optional[str] = None) -> BaseRunner:
        """Instantiate a fresh runner per ``model_name`` (one runner per model)."""
        key = f"{runner_type}:{model_name}" if model_name else runner_type
        if key not in self._runners:
            cls = self._RUNNER_CLASSES.get(runner_type)
            if cls is None:
                raise ValueError(
                    f"Unknown runner type '{runner_type}'. "
                    f"Available: {list(self._RUNNER_CLASSES.keys())}"
                )
            self._runners[key] = cls(self._cm, arkestra=self, **self._runner_kwargs)
        return self._runners[key]

    # ── backward-compat shims (delegate to unified lazy factory) ─────────

    @property
    def process_runner(self) -> ProcessRunner:
        if "process" not in self._runners:
            self.get_runner_instance("process")
        return self._runners["process"]  # type: ignore[return-value]

    @property
    def podman_runner(self) -> PodmanRunner:
        if "podman" not in self._runners:
            self.get_runner_instance("podman")
        return self._runners["podman"]  # type: ignore[return-value]

    @property
    def docker_runner(self) -> DockerRunner:
        if "docker" not in self._runners:
            self.get_runner_instance("docker")
        return self._runners["docker"]  # type: ignore[return-value]

    # ── backend resolution ─────────────────────────────────────────────
    def resolve_backend_id(self, model_name: str, env_vars: Dict[str, Any], override: Optional[str] = None) -> str:
        if override:
            return override
        ctx = self.model_obj(model_name)
        ctx_backend = getattr(ctx, "backend_id", None) if ctx else None
        if ctx_backend:
            return ctx_backend
        model = self.get_model(model_name) or {}
        return _resolve_backend(self._cm, model, model_name, None)

    def _get_runner(self, model_name: str, env_vars: Dict[str, Any], backend: Optional[str] = None) -> BaseRunner:
        # Find the runner that owns this context
        local_name = self.local_model_name(model_name)
        ctx = self.model_obj(local_name)
        if ctx is not None and getattr(ctx, '_runner', None) is not None:
            return ctx._runner
        runner_type = self.resolve_runner_type(model_name, env_vars, backend)
        return self.get_runner_instance(runner_type, model_name)

    def resolve_runner_type(self, model_name: str, env_vars: Dict[str, Any], override_backend: Optional[str] = None) -> str:
        """Resolve runner type: model → backend.runner → default.container-type → runners.default → process."""
        model_cfg = self._cm.get("models", {}).get(model_name, {})
        cm = self._cm.data

        if runner := model_cfg.get("runner"):
            return self._normalize_container(runner)

        backend_id = self.resolve_backend_id(model_name, env_vars, override_backend)
        be = cm.get("backends", {}).get(backend_id, {}) or {}
        if runner := be.get("runner"):
            return self._normalize_container(runner)

        # Global container-type is a fallback, not an override of backend settings
        default_type = self._cm.get("default/container-type", None) or (
            cm.get("runners", {}) or {}).get("default", "process")
        return self._normalize_container(default_type)

    def _normalize_container(self, runner_type: str) -> str:
        """Normalize 'container' sentinel → default.container-type."""
        if runner_type == "container":
            return self._cm.get("default/container-type", "process")
        return runner_type

    # ── env resolution (computed at init, never persisted) ───────

    def _build_env(self) -> Dict[str, str]:
        """Merge default-env YAML with actual os.environ into _env.

        precedence: explicit constructor args > os.environ > default-env defaults.
        The _env dict is computed once at startup and never written to disk.
        Keys are normalized to kebab-case regardless of YAML convention used.
        Env var lookup uppercases the key and replaces '-' with '_'.
        """
        defaults = self._cm.get("default-env", {}) or {}
        result: Dict[str, str] = {}
        for raw_key in defaults:
            # Normalize any case/convention to kebab-case
            norm_key = raw_key.lower().replace("_", "-")
            env_name = norm_key.upper().replace("-", "_")
            os_val = os.environ.get(env_name)
            if os_val:
                result[norm_key] = os_val
            else:
                val = defaults[raw_key]
                if val is not None:
                    result[norm_key] = str(val) if not isinstance(val, str) else val
        return result

    def _ensure_env(self) -> Dict[str, str]:
        """Return the computed _env dict (computed lazily at first access)."""
        if not hasattr(self, "_env") or self._env is None:
            self._env = self._build_env()
        return self._env

    def resolve_config(self, key: str, explicit: Optional[str] = None) -> Optional[str]:
        """Resolve a config value with unified precedence.

        Keys use kebab-case (e.g. "hf-hub-cache", "admin-key", "api-key").
        Input is normalized to kebab-case before lookup, so underscore or
        camelCase inputs also work: ``resolve_config("admin_key")`` resolves
        the same as ``resolve_config("admin-key")``.
        Precedence: explicit arg → _env (default-env + os.environ merged).
        The _env section is never persisted to disk — it's always freshly
        computed from default-env config values plus the actual process env.
        """
        if explicit is not None and explicit != "":
            return explicit
        # Normalize any input convention to kebab-case for lookup
        norm_key = key.lower().replace("_", "-")
        env_cfg = self._ensure_env()
        return env_cfg.get(norm_key) or ""

    def _cache_root(self) -> Path:
        """Resolve hf_hub_cache to a root Path."""
        val = self.resolve_config("hf_hub_cache")
        if val:
            return Path(val).expanduser()
        return default_cache_root()

    def _cache_dir_for_checkpoint(self, repo: str) -> Path:
        """Return the cache directory path for a given HuggingFace repo string."""
        return self._cache_root() / f"models--{repo.replace('/', '--')}"

    def _cleanup_partial_cache(self, cache_path: Optional[str]) -> None:
        """Remove partial download artifacts from a cancelled pull."""
        if not cache_path:
            return
        try:
            cache_dir = self._cache_dir_for_checkpoint(cache_path)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        except Exception as e:
            self.log(f"[pull] cleanup error: {e}", level="WARNING")

    # ── context manager ────────────────────────────────────────

    async def __aenter__(self) -> "ModelArkestra":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    # ── lifecycle API ──────────────────────────────────────────────────
    async def start(
        self,
        model_name: str,
        **overrides: Any,
    ) -> None:
        """Start a model.

        Model names are resolved as ``<cluster>/<model-id>``.  If no cluster
        prefix is given the model belongs to the local cluster.

        Remote-cluster models proxy all requests to the target arkestra worker;
        no local port or subprocess is allocated.

        ``**overrides`` is a flat mix of infrastructure and inference kwargs:

        * Infra keys (handled by ModelArkestra): ``port``, ``backend``, ``runner``
        * Everything else → inference param, passed through to runner.start()
          as bare kwargs. Converted to ``--flag value`` CLI flags at subprocess
          boundary.

        Runner values: ``process``, ``podman``, ``docker``, or ``container``
        (resolves to ``container-type`` from top-level config).

        Example::

            await arkestra.start("qwen3-4b", temp=1.0, top_k=20)
            await arkestra.start("local/qwen3-4b", port=18000, backend="rocm-container", temp=0.9)
        """
        cluster_name, base_url, local_name = self.resolve_model_cluster_addr(model_name)

        # ── remote-cluster model: proxy everything through the worker ──
        if base_url is not None:
            return await self._start_remote_model(local_name, overrides, cluster_name, base_url)

        # ── local-cluster model ──────────────────────────────────────
        model = self.get_model(local_name)
        if not model:
            raise ValueError(f"Unknown model '{local_name}' in cluster '{cluster_name}'.")

        # Shared-checkpoint guard: if this model's context (shared by all models
        # of the same checkpoint) is already running, it's a no-op — the process
        # and its KV cache are already up for every alias.
        ctx = self.model_obj(local_name)
        if ctx is not None and ctx.state == RunnerState.RUNNING:
            return

        backends_cfg = self._cm.get("backends", {})
        runners_cfg = self._cm.get("runners", {})

        # Separate infra keys from inference kwargs
        infra_keys = {"port", "backend", "runner"}
        port = overrides.get("port")
        backend = overrides.get("backend")
        runner_type_override = overrides.get("runner")
        inference_kwargs = {k: v for k, v in overrides.items() if k not in infra_keys}

        # Validate explicit backend
        if backend and backend not in backends_cfg:
            raise ValueError(
                f"Unknown backend '{backend}'. Available: {list(backends_cfg.keys())}"
            )

        # Gate on eligible states
        if not self.can_start(local_name):
            raise ValueError("model not available")

        # Resolve runner type from model config — tag-driven routing
        be_id = self.resolve_backend_id(local_name, {}, backend)
        be_cfg = backends_cfg.get(be_id, {})
        resolved_runner = self._normalize_container(
            str(be_cfg.get("runner", "process"))
        )

        model_cfg = self.get_model(local_name) or {}
        tags = _resolve_model_tags(model_cfg, self._cm.data, backend_id=be_id)
        is_onnx = (resolved_runner == "onnx" or
                   any(t in tags for t in ("asr", "tts", "embed")))

        if is_onnx and resolved_runner == "onnx":
            return await self._start_onnx_model(local_name, inference_kwargs)

        # Allocate port if not explicitly provided.
        if port is None:
            port = self.worker_port(local_name)

        # runner= selects the transport layer
        if runner_type_override is not None:
            inst = self.get_runner_instance(self._normalize_container(runner_type_override), local_name)
            if inst._ctx is None and ctx is not None:
                inst._ctx = ctx
            await inst.start(local_name, port=port, backend=backend, **inference_kwargs)
            ctx = inst.ctx
            ctx.runner_type = runner_type_override
            ctx._runner = inst
            self.log(f"[action=start model={model_name} port={port}]")
            return

        # Resolve backend + runner type from config
        be_id = self.resolve_backend_id(local_name, {}, backend)
        be_cfg = backends_cfg.get(be_id, {})
        resolved_runner = self._normalize_container(str(be_cfg.get("runner", "process")))

        if resolved_runner not in runners_cfg and resolved_runner not in ("process", "podman", "docker", "onnx", "remote"):
            raise ValueError(
                f"Backend '{be_id}' resolves to unknown runner type '{resolved_runner}'. "
                f"Available runners: {list(runners_cfg.keys())}"
            )

        runner = self.get_runner_instance(resolved_runner, local_name)
        # Aliases of a shared checkpoint have no runner of their own — reuse
        # the owner's (runner._ctx is the shared context).
        if runner._ctx is None and ctx is not None:
            runner._ctx = ctx
        await runner.start(local_name, port=port, backend=backend, **inference_kwargs)
        ctx = runner.ctx
        ctx.runner_type = resolved_runner
        ctx._runner = runner
        if resolved_runner == "remote" and not ctx._remote_base_url:
            # Legacy remote backend (runner: remote + base_url, no cluster prefix)
            ctx._remote_base_url = str(be_cfg.get("base_url", "")).rstrip("/")
            ctx._admin_key = be_cfg.get("admin_key") or be_cfg.get("admin-key") or ""
        self.log(f"[action=start model={model_name} port={port}]")

    async def execute(self, model_name: str, capability: str, **kwargs) -> Any:
        """Dispatch any capability to the appropriate runner handler."""
        runner = self._get_runner(model_name, {}, None)
        handler = getattr(runner, capability, None)
        if not handler:
            raise RuntimeError(
                f"Runner for '{model_name}' does not implement capability '{capability}'. "
                f"Available: {', '.join(m for m in dir(runner) if not m.startswith('_') and callable(getattr(runner, m)))}"
            )
        return await handler(model_name, **kwargs)

    async def _start_onnx_model(
        self, model_name: str, inference_kwargs: Dict[str, Any],
    ) -> None:
        """Start an ONNX model — load into memory, no subprocess needed.

        Skips port allocation and HTTP health checks since ONNX models
        run in-process. The runner manages the InferenceSession lifecycle.
        """
        # Find or create the onnx runner for this model
        runner = self.get_runner_instance("onnx", model_name)

        # Create context manually — no port allocation needed
        ctx = self.model_obj(model_name)
        if ctx is None:
            eff_port = inference_kwargs.get("port") or 0  # dummy port for context compatibility
            log_size = inference_kwargs.get("max_log_lines", self._cm.get("default/log-buffer-size", 2000))
            model_data = self.get_model(model_name, env_vars={})
            model_path_str = str((model_data or {}).get("model_path", ""))
            if not model_path_str:
                default_section = self._cm.get("default", {})
                resolved = resolve_model_ref(
                    raw=(model_data or {}).get("model"),
                    default_section=default_section,
                    model_repos=self._cm.get("default/model-repos"),
                )
                if resolved.repo == "hf":
                    model_path_str = f"hf:{resolved.ref}"
                elif resolved.repo == "lcl":
                    model_path_str = resolved.ref.removeprefix("lcl:")

            from model_arkestra.types import _Model
            ctx = _Model(model_name, eff_port, max_log_lines=log_size)
            ctx.backend_id = "onnx"
            ctx._model_path = model_path_str  # store for runner to use
            ctx._runner = runner
            ctx.set_state("load")
            self._registry.register(model_name, ctx, [model_name])
            runner._ctx = ctx

        # Start the ONNX model (loads InferenceSession into memory)
        await runner.start(model_name, port=ctx.port, backend="onnx",
                           **{k: v for k, v in inference_kwargs.items()})
        ctx = runner.ctx
        ctx.runner_type = "onnx"

        logger.info("ONNX model '%s' loaded into memory", model_name)

    async def _start_remote_model(
        self, local_name: str, overrides: Dict[str, Any], cluster_name: str, base_url: str,
    ) -> None:
        """Start a remote-cluster model — proxy all traffic to the target arkestra worker.

        No local port is allocated.  All HTTP calls (start, stop, chat, embed)
        are forwarded to ``base_url`` from the cluster configuration.
        """
        inference_kwargs = {k: v for k, v in overrides.items() if k not in {"port", "backend", "runner"}}

        # Find or create the remote runner (shared per model instance)
        cluster_cfg = self.clusters.get(cluster_name) or {}
        ctx = self.model_obj(local_name)
        if ctx is None:
            log_size = inference_kwargs.get("max_log_lines", self._cm.get("default/log-buffer-size", 2000))
            from model_arkestra.types import _Model as MC
            ctx = MC(local_name, 0, max_log_lines=log_size)  # port=0 for remote models
            ctx.backend_id = "remote"
            ctx.cluster = cluster_name
            ctx._remote_base_url = base_url
            ctx._admin_key = cluster_cfg.get("admin-key") or cluster_cfg.get("admin_key") or ""
            self._registry.register(local_name, ctx, [local_name])
            runner = self.get_runner_instance("remote", local_name)
            runner._ctx = ctx
            ctx._runner = runner

        # Pass inference kwargs and start (proxies to worker)
        runner = ctx._runner
        runner._inference_kwargs[local_name] = inference_kwargs
        await runner.start(local_name, port=ctx.port, backend="remote", **inference_kwargs)
        ctx.runner_type = "remote"
        self.log(f"[action=start model={cluster_name}/{local_name} remote={base_url}]")

    async def embed(self, model_name: str, text: str) -> Dict[str, Any]:
        """Encode text → embedding vector. Returns OpenAI-compatible response."""
        return await self._provider_for(model_name).embed(text)

    async def transcribe(self, model_name: str, audio_bytes: bytes,
                         language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio → text via ONNX model."""
        return await self._provider_for(model_name).transcribe(audio_bytes, language or "")

    async def synthesize(self, model_name: str, text: str,
                         voice: Optional[str] = None,
                         speed: float = 1.0) -> bytes:
        """Generate speech from text via TTS ONNX model."""
        return await self._provider_for(model_name).synthesize(text, voice or "", speed)

    async def stream_asr(self, model_name: str, audio_bytes: bytes) -> Dict[str, Any]:
        """Streaming ASR with partial/final results (sherpa-ai)."""
        return await self._provider_for(model_name).stream_asr(audio_bytes)

    async def stream_tts(self, model_name: str, text: str) -> bytes:
        """Piper TTS — single WAV frame for WebSocket streaming."""
        return await self._provider_for(model_name).stream_tts(text)

    async def stop(self, model_name: str) -> None:
        """Stop the named model."""
        _, _, local_name = self.resolve_model_cluster_addr(model_name)
        ctx = self.model_obj(local_name)
        if ctx is not None and getattr(ctx, '_runner', None) is not None:
            self.log(f"[action=stop model={model_name}]")
            try:
                await ctx._runner.stop()
            except Exception:
                pass

    async def pull(self, model_name: str) -> Dict[str, Any]:
        """Begin pulling a model's checkpoint (non-blocking).

        Transitions the context to DOWNLOADING and spawns the download as a
        background task. Returns immediately; callers poll status for progress.
        Never blocks the event loop.
        """
        _, _, local_name = self.resolve_model_cluster_addr(model_name)
        if not self.get_model(local_name):
            raise ValueError(f"Unknown model '{local_name}'")

        ctx = self.model_obj(local_name)
        if ctx is None:
            # Create a context so the download has somewhere to report state.
            from model_arkestra.types import _Model as _M
            ctx = _M(local_name, 0)
            self._registry.register(local_name, ctx, [local_name])

        if ctx.state == RunnerState.RUNNING:
            return {"ok": True, "model": local_name, "already_loaded": True}
        if ctx.state == RunnerState.DOWNLOADING and ctx.download_task and not ctx.download_task.done():
            return {"ok": True, "model": local_name, "already_downloading": True}

        ctx.set_state("pull")
        task = asyncio.create_task(self.pull_model(ctx, local_name))
        ctx.download_task = task
        self.log(f"[action=pull model={local_name}]")
        return {"ok": True, "model": local_name}

    async def pull_model(self, ctx: _Model, model_name: str | None = None) -> None:
        """Background task: download the checkpoint's exact file set.

        Resolves ``repo:quant`` to the specific GGUF files (primary + any
        mmproj/mtp sidecars) via :mod:`model_arkestra.hf_gguf` and fetches only
        those into the HF cache. Runs in a worker thread; never blocks the
        event loop.

        ``model_name`` is the *model* name (not the checkpoint id) — required
        for grouped checkpoints where ``ctx.name`` is the shared checkpoint id
        and does not resolve via :meth:`get_model`.
        """
        model_name = model_name or ctx.name
        try:
            model_data = self.get_model(model_name) or {}
            raw = model_data.get("model", "")
            resolved = resolve_model_ref(
                raw,
                default_section=(self._cm.data.get("default") or {}),
                model_repos=self._cm.data.get("model-repos"),
            )
            if not resolved.cache_path:
                raise ValueError(f"No cacheable model ref: {raw}")

            from model_arkestra.hf_gguf import resolve_plan, download_plan, split_repo_tag
            from huggingface_hub import HfApi

            repo, _tag = split_repo_tag(resolved.ref)
            if not repo:
                raise ValueError(f"Cannot pull non-HF ref: {resolved.ref}")

            def log_progress(line: str) -> None:
                ctx._append_log_line(f"[pull] {model_name}: {line}")

            def _do_download() -> list[str]:
                files = HfApi().list_repo_files(repo)
                plan = resolve_plan(resolved.ref, files)
                total = len(plan.files)

                def _prog(filename: str, i: int) -> None:
                    ctx._append_log_line(f"[pull] {model_name}: ({i}/{total}) {filename}")
                    ctx.download_current = filename
                    if total:
                        ctx.download_pct = round(100.0 * (i - 1) / total, 1)

                log_progress(f"fetching {total} file(s): "
                             f"{', '.join(plan.files)}")
                return download_plan(plan, cache_dir=str(self._cache_root()),
                                     token=self._cm.get("default/hf-token", None),
                                     progress_cb=_prog)

            await asyncio.to_thread(_do_download)

            ctx.set_state("download_ok")
            self.log(f"[pull] model={model_name} complete")
        except asyncio.CancelledError:
            self._cleanup_partial_cache(resolved.cache_path)
            ctx.set_state("download_cancel")
            self.log(f"[pull] model={model_name} cancelled")
            raise
        except Exception as e:
            self._cleanup_partial_cache(resolved.cache_path)
            ctx.set_state("download_fail")
            ctx.last_error = str(e)
            self.log(f"[pull] model={model_name} FAILED: {e}", level="ERROR")

    def can_start(self, model_name: str) -> bool:
        """Check if model is eligible for a fresh start."""
        ctx = self.model_obj(model_name)
        if not ctx:
            # No context yet — allowed if model is defined in config (fresh start).
            return self.get_model(model_name) is not None
        return ctx.state in (RunnerState.STOPPED, RunnerState.ERROR, RunnerState.UNCACHED)

    def can_restart(self, model_name: str) -> bool:
        """Check if model is eligible for a restart."""
        ctx = self.model_obj(model_name)
        return ctx.state in (RunnerState.STOPPED, RunnerState.ERROR,
                             RunnerState.LOADING, RunnerState.RUNNING)

    def can_stop(self, model_name: str) -> bool:
        """Check if model is in a state that can be stopped."""
        ctx = self.model_obj(model_name)
        if not ctx:
            return False
        return ctx.state in (RunnerState.LOADING, RunnerState.RUNNING,
                             RunnerState.STOPPING, RunnerState.DOWNLOADING)

    async def eject(self, model_name: str) -> Dict[str, Any]:
        """Stop a model's checkpoint and delete its cached weight files.

        Ejecting any model name acts on the *checkpoint* it references — the
        shared process and cache that all models of that checkpoint use. Every
        model sharing the checkpoint is taken down together (that is intended;
        the UI is responsible for surfacing the blast radius).
        """
        cfg = self._cm.get("models", {})
        if model_name not in cfg:
            raise ValueError(f"Model '{model_name}' not in config")

        # Resolve the weight ref via the merged (checkpoint + model) config.
        model_cfg = self.get_model(model_name) or {}
        default_section = self._cm.get("default", {})
        raw = model_cfg.get("model")
        resolved = resolve_model_ref(
            raw=raw,
            default_section=default_section,
            model_repos=self._cm.get("default/model-repos"),
        )
        cache_path = resolved.cache_path
        result: Dict[str, Any] = {
            "ok": True,
            "model": model_name,
            "cache_deleted": False,
        }

        # Stop the shared process first (always).
        await self.stop(model_name)

        if not cache_path:
            # No cache to clear — context stays for visibility
            return result

        cache_root = self._cache_root()
        cache_dir = self._cache_dir_for_checkpoint(cache_path)

        # Get the (shared) context for this model.
        ctx = self.model_obj(model_name)

        # Delete cache, mark context as UNCACHED (cache gone).
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            result["cache_deleted"] = True
            result["cache_path"] = str(cache_dir)
        if ctx:
            ctx.set_state("eject")

        return result


    async def restart(
        self,
        model_name: str,
        **overrides: Any,
    ) -> None:
        """Stop a model and start a new instance on the same port.

        Optional kwargs override backend, runner, or pass inference params.
        All overrides are transient — they disappear on restart.
        """
        await self.stop(model_name)
        await self.start(model_name, **overrides)

    async def stop_all(self) -> None:
        """Stop all model processes across every runner, keeping entries alive."""
        for r in self._runners.values():
            await r.stop_all()

    async def shutdown(self) -> None:
        """Full teardown — stop models, clear runners, reset port allocator."""
        self.log(f"[action=shutdown]")
        # Cancel any active pull tasks
        for ctx in self._registry.all_contexts:
            if ctx.download_task and not ctx.download_task.done():
                ctx.download_task.cancel()
        for r in self._runners.values():
            await r.shutdown()
        self._runners.clear()
        self._registry.reset_ports()

    @property
    def running_models(self) -> Set[str]:
        result: Set[str] = set()
        for r in self._runners.values():
            result.update(r.running_models)
        return result

    # ── request API ────────────────────────────────────────────────────
    async def ainvoke(self, model_name: str, prompt: str = "", backend: Optional[str] = None,
                      messages: Optional[list] = None, **kwargs: Any) -> str:
        return (await self.ainvoke_full(model_name, prompt, backend=backend,
                                        messages=messages, **kwargs)).get("content", "")

    async def ainvoke_full(self, model_name: str, prompt: str = "", backend: Optional[str] = None,
                           messages: Optional[list] = None, **kwargs: Any) -> Dict[str, Any]:
        """Run inference, returning the full ``{"content", "usage"}`` dict."""
        prov = self._provider_for(model_name)
        if messages is not None:
            return await prov.invoke_full("", messages=messages, **kwargs)
        return await prov.invoke_full(prompt, **kwargs)

    async def astream(self, model_name: str, payload: Dict[str, Any], backend: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        prov = self._provider_for(model_name)
        async for chunk in prov.stream(payload):
            yield chunk

    async def request(self, model_name: str, path: str, **kwargs: Any) -> Any:
        return await self._provider_for(model_name).request(path, **kwargs)

    async def get_logs(self, model_name: str, lines: int = 100) -> List[str]:
        """Return the last N log lines for a model."""
        local_name = self.local_model_name(model_name)
        ctx = self.model_obj(local_name)
        if ctx is not None and getattr(ctx, '_runner', None) is not None:
            return await ctx._runner.get_logs(ctx, lines)
        return []
