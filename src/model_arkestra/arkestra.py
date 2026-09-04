"""Single entry point for all model operations — wraps ConfigManager + lazy runners."""
from __future__ import annotations
import logging
import os
import shutil
import yaml
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from model_arkestra.config_manager import ModelConfigManager
from model_arkestra.gpu_detect import detect_all, has_rocm, has_vulkan, has_nvidia
from model_arkestra.base import BaseModelRunner
from model_arkestra.common import (
    _resolve_backend, default_cache_root, resolve_config_path,
    resolve_model_ref, resolve_tags as _resolve_model_tags,
    download_hf_model,
)
from model_arkestra.docker import DockerModelRunner
from model_arkestra.onnx_runner import OnnxRunner
from model_arkestra.podman import PodmanModelRunner
from model_arkestra.process import ProcessModelRunner
from model_arkestra.remote import RemoteModelRunner
from model_arkestra.types import RunnerState, _ModelContext
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
        self._next_port = self._cm.get("default/model-start-port", start_port)

        self._runners: Dict[str, BaseModelRunner] = {}
        self._runner_kwargs = runner_kwargs
        # ── Cluster topology ───────────────────────────────────────
        self._load_clusters()
        # Extract sources section for binary_downloader compatibility
        self._sources: Dict[str, Any] = self._cm.get("sources", {})
        # ── Global log buffer (single ring for all server-level events) ─
        default_section = self._cm.get("default", {})
        app_log_lines = int(self._cm.get("default/app-log-lines", 2000))
        self._global_log_buf = UnicodeRingBuffer(app_log_lines * _ModelContext.AVG_LINE_BYTES)
        self._global_log_seq: int = 0
        # ── Backend runtime validation (hard error on mismatch) ───────
        self._validate_backend_runtime()
        # ── Device profile detection & matching ────────────────────
        self._matched_profile = self._detect_device_profiles()

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
        """Allocate a port for *model_name*.

        Reuses the port from an existing STOPPED context (stop→restart),
        otherwise allocates a fresh port from the pool.
        """
        # Check for existing stopped context — reuse its port to avoid
        # exhausting the port pool on repeated stop/start cycles (the same model).
        for runner in self._runners.values():
            ctx = runner._models.get(model_name)
            if ctx is not None and ctx.port is not None:
                return ctx.port

        default_section = self._cm.get("default", {})
        start_port = self._cm.get("default/model-start-port", 18000)
        max_ports = self._cm.get("default/model-ports", 32)
        end_port = start_port + max_ports - 1

        if self._next_port > end_port:
            raise RuntimeError(
                f"Port range exceeded: {start_port}–{end_port}"
            )

        port = self._next_port
        self._next_port += 1
        return port
    
    # ── backend runtime validation (hard error) ─────────────────────
    def _validate_backend_runtime(self) -> None:
        """Ensure the configured default backend's runtime is available.

        Raises RuntimeError if a non-CPU backend is configured but its
        runtime is not detected on the system. CPU backends always pass
        since the binary will be downloaded at model-start time.
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
        if checker:
            if not checker():
                suggestion = "Run 'model-arkestra init' to auto-detect your GPU."
                raise RuntimeError(
                    f"Backend '{backend_id}' configured but runtime not detected. {suggestion}"
                )
        # CPU backends and unknown IDs pass by default

    def _detect_device_profiles(self) -> Dict[str, Any]:
        """Detect GPU hardware and find matching engine device-profile.

        Returns {env: {...}, args: {...}} for the best-matching device profile,
        or empty dict if no match found (backend-specific settings still apply).

        Priority: exact key match → family fallback (rocm/cuda/vulkan) → none.
        """
        result = detect_all()
        primary = result.get("primary_gpu")
        if not primary:
            return {}

        # Collect device-profiles from all engines
        profiles: Dict[str, Dict] = {}
        for engine_cfg in (self.cm.get("engines", {}) or {}).values():
            if isinstance(engine_cfg, dict) and "device-profiles" in engine_cfg:
                profiles.update(engine_cfg["device-profiles"])
        if not profiles:
            return {}

        vendor = primary.get("vendor", "")
        matched_key: Optional[str] = None

        # ROCm: try exact gfx_family → family fallback
        if vendor == "amd":
            gfx = result.get("gfx_family")
            if gfx and gfx in profiles:
                matched_key = gfx
            elif "rocm" in profiles:
                matched_key = "rocm"
        # NVIDIA: try GPU name patterns → family fallback
        elif vendor == "nvidia":
            gpu_name = primary.get("name", "").lower()
            for key in profiles:
                if any(part in gpu_name for part in key.replace('-', ' ').split()):
                    matched_key = key
                    break
            if not matched_key and "cuda" in profiles:
                matched_key = "cuda"
        # Vulkan/Intel: family fallback
        elif vendor in ("intel",):
            if "vulkan" in profiles:
                matched_key = "vulkan"

        if matched_key is None:
            return {}

        prof = profiles[matched_key]
        return {
            "env": prof.get("env") or {},
            "args": prof.get("args") or {},
        }

    # ── cluster topology ───────────────────────────────────────────
    def _load_clusters(self) -> None:
        """Load managed arkestra clusters from config.

        Config structure::

            clusters:
              worker0:                    # remote managed
                base-url: http://worker0:8080
                admin-key: secret         # optional
        """
        self._clusters: Dict[str, Dict[str, Any]] = {}

        # Auto-create the local cluster
        host = self._cm.get("default/host", "0.0.0.0")
        port = self._cm.get("default/admin-port", 8080)
        self._local_cluster_key: str = self._cm.get("default/local-cluster-key", "local")
        self._clusters[self._local_cluster_key] = {
            "base-url": f"http://{host}:{port}",
            "admin-key": self._cm.get("env/ADMIN_KEY"),
        }

        # Parse remote clusters from YAML
        raw_clusters = self._cm.get("clusters", {})
        if not isinstance(raw_clusters, dict):
            return
        for name, cfg in raw_clusters.items():
            if not isinstance(cfg, dict):
                continue
            base_url = cfg.get("base-url", "")
            if not base_url:
                self.log(f"[config] skipping cluster '{name}': missing base-url", level="WARNING")
                continue
            # Strip trailing slash for consistency
            cfg = dict(cfg)
            if isinstance(base_url, str):
                cfg["base-url"] = base_url.rstrip("/")
            self._clusters[name] = cfg

    def _parse_cluster_prefix(self, model_name: str) -> Tuple[str, str]:
        """Split ``<cluster>/<model-id>`` into its components.

        Returns ``(cluster_name, model_id)``.  If no prefix is present,
        the model belongs to the local cluster.
        """
        if "/" in model_name:
            return model_name.split("/", 1)
        return self._local_cluster_key, model_name

    def resolve_model_cluster_addr(self, model_name: str) -> Tuple[str, Optional[str], str]:
        """Resolve cluster routing for a model name.

        Returns ``(cluster_name, base_url|None, local_model_id)``.  For the
        local cluster ``base_url`` is None (use port pool / direct runner).
        For remote clusters it returns the target URL to proxy through.

        Falls back to the legacy ``runner: remote`` backend config when a
        ``/<model-id>`` prefix does not match any declared cluster.
        """
        cluster_name, local_id = self._parse_cluster_prefix(model_name)
        cfg = self._clusters.get(cluster_name)
        if cfg is None:
            # Legacy fallback: check backends for runner=remote + base_url
            backends_cfg = self._cm.get("backends", {})
            be = backends_cfg.get(cluster_name, {})
            if isinstance(be, dict) and be.get("runner") == "remote" and be.get("base_url"):
                return cluster_name, str(be["base_url"]).rstrip("/"), local_id
            raise ValueError(f"Unknown cluster '{cluster_name}' for model '{model_name}'. "
                             f"Declare it in the 'clusters:' top-level key or add a backend "
                             f"entry with runner='remote' and base_url.")
        base_url = cfg.get("base-url") if cluster_name != self._local_cluster_key else None
        return cluster_name, base_url, local_id

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

    def _get_model_contexts(self) -> list[_ModelContext]:
        """Return all tracked _ModelContext objects across every runner."""
        contexts: list[_ModelContext] = []
        for r in self._runners.values():
            contexts.extend(r._models.values())  # noqa: SLF001
        return contexts

    def find_context(self, model_name: str) -> Optional[_ModelContext]:
        """Return the _ModelContext for *model_name*, or None."""
        _, _, local_name = self.resolve_model_cluster_addr(model_name)
        for ctx in self._get_model_contexts():
            if ctx.name == local_name:
                return ctx
        return None

    def get_v1_models(self) -> Dict[str, Any]:
        """OpenAI-compatible ``/v1/models`` response with WebUI status fields."""
        from time import time

        contexts_by_name = {ctx.name: ctx for ctx in self._get_model_contexts()}
        data = []
        for model_name in self.get_models():
            # Skip remote-cluster models (not tracked locally)
            cluster_name, local_name = self._parse_cluster_prefix(model_name)
            if cluster_name != self._local_cluster_key:
                continue
            ctx = contexts_by_name.get(model_name)
            model_cfg = self.get_model(model_name) or {}
            owned_by = str(model_cfg.get("owned_by", "local")) if isinstance(model_cfg, dict) else "local"

            entry: Dict[str, Any] = {
                "id": f"{cluster_name}/{model_name}",
                "object": "model",
                "created": int(time()),
                "owned_by": owned_by,
                "status": model_status_for_ctx(ctx),
            }

            data.append(entry)

        return {"object": "list", "data": data}

    # ── runner class map — one hop, no magic ─────────────────────────

    _RUNNER_CLASSES: Dict[str, type] = {
        "process": ProcessModelRunner,
        "podman": PodmanModelRunner,
        "docker": DockerModelRunner,
        "onnx": OnnxRunner,
        "remote": RemoteModelRunner,
    }

    def _get_runner_instance(self, runner_type: str, model_name: Optional[str] = None) -> BaseModelRunner:
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
    def process_runner(self) -> ProcessModelRunner:
        if "process" not in self._runners:
            self._get_runner_instance("process")
        return self._runners["process"]  # type: ignore[return-value]

    @property
    def podman_runner(self) -> PodmanModelRunner:
        if "podman" not in self._runners:
            self._get_runner_instance("podman")
        return self._runners["podman"]  # type: ignore[return-value]

    @property
    def docker_runner(self) -> DockerModelRunner:
        if "docker" not in self._runners:
            self._get_runner_instance("docker")
        return self._runners["docker"]  # type: ignore[return-value]

    # ── backend resolution ─────────────────────────────────────────────
    def _resolve_backend_id(self, model_name: str, env_vars: Dict[str, Any], override: Optional[str] = None) -> str:
        if override:
            return override
        ctx = self.find_context(model_name)
        ctx_backend = getattr(ctx, "backend_id", None) if ctx else None
        if ctx_backend:
            return ctx_backend
        model = self.get_model(model_name) or {}
        return _resolve_backend(self._cm, model, model_name, None)

    def _get_runner(self, model_name: str, env_vars: Dict[str, Any], backend: Optional[str] = None) -> BaseModelRunner:
        # Find the runner that has this model
        for r in self._runners.values():
            if model_name in r._models and r._models[model_name].state == RunnerState.RUNNING:
                return r
        runner_type = self._resolve_runner_type(model_name, env_vars, backend)
        return self._get_runner_instance(runner_type, model_name)

    def _resolve_runner_type(self, model_name: str, env_vars: Dict[str, Any], override_backend: Optional[str] = None) -> str:
        """Resolve runner type: model → backend.runner → default.container-type → runners.default → process."""
        model_cfg = self._cm.get("models", {}).get(model_name, {})
        cm = self._cm.data

        if runner := model_cfg.get("runner"):
            return self._normalize_container(runner)

        backend_id = self._resolve_backend_id(model_name, env_vars, override_backend)
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

    # ── cache helpers ────────────────────────────────────────────────

    def resolve_config(self, key: str, explicit: Optional[str] = None) -> Optional[str]:
        """Resolve a config value with unified precedence.

        Precedence: explicit arg → os.environ → config.env section → None.
        """
        if explicit is not None and explicit != "":
            return explicit
        val = os.environ.get(key)
        if val:
            return val
        env_cfg = self._cm.get("env", {})
        return env_cfg.get(key) if isinstance(env_cfg, dict) else ""

    def _cache_root(self) -> Path:
        """Resolve HF_HUB_CACHE to a root Path."""
        val = self.resolve_config("HF_HUB_CACHE")
        if val:
            return Path(val).expanduser()
        return default_cache_root()

    def _cache_dir_for_checkpoint(self, repo: str) -> Path:
        """Return the cache directory path for a given HuggingFace repo string."""
        return self._cache_root() / f"models--{repo.replace('/', '--')}"

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

        # Resolve runner type from model config — tag-driven routing
        be_id = self._resolve_backend_id(local_name, {}, backend)
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
            inst = self._get_runner_instance(self._normalize_container(runner_type_override), local_name)
            await inst.start(local_name, port=port, backend=backend, **inference_kwargs)
            ctx = inst._models[local_name]
            ctx.runner_type = runner_type_override
            ctx._runner = inst
            self.log(f"[action=start model={model_name} port={port}]")
            return

        # Resolve backend + runner type from config
        be_id = self._resolve_backend_id(local_name, {}, backend)
        be_cfg = backends_cfg.get(be_id, {})
        resolved_runner = self._normalize_container(str(be_cfg.get("runner", "process")))

        if resolved_runner not in runners_cfg and resolved_runner not in ("process", "podman", "docker", "onnx", "remote"):
            raise ValueError(
                f"Backend '{be_id}' resolves to unknown runner type '{resolved_runner}'. "
                f"Available runners: {list(runners_cfg.keys())}"
            )

        runner = self._get_runner_instance(resolved_runner, local_name)
        await runner.start(local_name, port=port, backend=backend, **inference_kwargs)
        ctx = runner._models[local_name]
        ctx.runner_type = resolved_runner
        ctx._runner = runner
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
        runner = self._get_runner_instance("onnx", model_name)

        # Create context manually — no port allocation needed
        ctx = self.find_context(model_name)
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

            from model_arkestra.types import _ModelContext
            ctx = _ModelContext(model_name, eff_port, max_log_lines=log_size)
            ctx.backend_id = "onnx"
            ctx._model_path = model_path_str  # store for runner to use
            ctx._runner = runner
            ctx.state = RunnerState.LOADING
            runner._models[model_name] = ctx  # noqa: SLF001

        # Start the ONNX model (loads InferenceSession into memory)
        await runner.start(model_name, port=ctx.port, backend="onnx",
                           **{k: v for k, v in inference_kwargs.items()})
        ctx = runner._models[model_name]  # noqa: SLF001
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
        ctx = self.find_context(local_name)
        if ctx is None:
            log_size = inference_kwargs.get("max_log_lines", self._cm.get("default/log-buffer-size", 2000))
            from model_arkestra.types import _ModelContext as MC
            ctx = MC(local_name, 0, max_log_lines=log_size)  # port=0 for remote models
            ctx.backend_id = "remote"
            ctx.cluster = cluster_name
            ctx._remote_base_url = base_url
            runner = self._get_runner_instance("remote")
            runner._models[local_name] = ctx  # noqa: SLF001
            ctx._runner = runner

        # Pass inference kwargs and start (proxies to worker)
        runner = ctx._runner
        runner._inference_kwargs[local_name] = inference_kwargs
        await runner.start(local_name, port=ctx.port, backend="remote", **inference_kwargs)
        ctx.runner_type = "remote"
        self.log(f"[action=start model={cluster_name}/{local_name} remote={base_url}]")

    async def embed(self, model_name: str, text: str) -> Dict[str, Any]:
        """Encode text → embedding vector via ONNX model.

        Returns OpenAI-compatible response with ``data[].embedding`` list.
        """
        runner = self._get_runner(model_name, {}, None)
        return await runner.embed(model_name, text)  # type: ignore[attr-defined]

    async def transcribe(self, model_name: str, audio_bytes: bytes,
                         language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio → text via ONNX model."""
        return await self.execute(model_name, "transcribe", audio_bytes=audio_bytes, language=language)

    async def synthesize(self, model_name: str, text: str,
                         voice: Optional[str] = None,
                         speed: float = 1.0) -> bytes:
        """Generate speech from text via TTS ONNX model."""
        return await self.execute(model_name, "synthesize", text=text, voice=voice, speed=speed)

    async def stop(self, model_name: str) -> None:
        """Stop the named model."""
        _, _, local_name = self.resolve_model_cluster_addr(model_name)
        for r in self._runners.values():
            if local_name in r._models:  # noqa: SLF001
                self.log(f"[action=stop model={model_name}]")
                try:
                    await r.stop()
                except Exception:
                    pass
                return

    async def _download_model(self, ctx: _ModelContext) -> None:
        """Background task: download model checkpoint from HuggingFace.

        Resolves the model reference, calls ``snapshot_download`` with
        progress callbacks, and transitions the context state on
        completion (UNCACHED) or failure (ERROR).
        """
        model_name = ctx.name
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

            cache_dir = self._cache_dir_for_checkpoint(resolved.cache_path)
            cache_dir.mkdir(parents=True, exist_ok=True)

            def log_progress(line: str) -> None:
                ctx._append_log_line(f"[download] {model_name}: {line}")

            download_hf_model(resolved.repo_id, cache_dir, log_progress)

            ctx.state = RunnerState.UNCACHED
            self.log(f"[download] model={model_name} complete")
        except asyncio.CancelledError:
            self.log(f"[download] model={model_name} cancelled")
            ctx.state = RunnerState.STOPPED
            raise
        except Exception as e:
            ctx.state = RunnerState.ERROR
            ctx.last_error = str(e)
            self.log(f"[download] model={model_name} FAILED: {e}", level="ERROR")

    def can_start(self, model_name: str) -> bool:
        """Check if model is eligible for a fresh start."""
        ctx = self.find_context(model_name)
        if not ctx:
            return False
        return ctx.state in (RunnerState.STOPPED, RunnerState.ERROR)

    def can_restart(self, model_name: str) -> bool:
        """Check if model is eligible for a restart."""
        ctx = self.find_context(model_name)
        if not ctx:
            return False
        return ctx.state in (RunnerState.STOPPED, RunnerState.ERROR,
                             RunnerState.LOADING, RunnerState.RUNNING)

    def can_stop(self, model_name: str) -> bool:
        """Check if model is in a state that can be stopped."""
        ctx = self.find_context(model_name)
        if not ctx:
            return False
        return ctx.state in (RunnerState.LOADING, RunnerState.RUNNING,
                             RunnerState.STOPPING, RunnerState.DOWNLOADING)

    async def eject(self, model_name: str) -> Dict[str, Any]:
        """Stop a model and delete its cached checkpoint files.

        Returns a dict with details about what was removed.  Raises ValueError
        if other running models share the same underlying cache directory.
        """
        cfg = self._cm.get("models", {})
        if model_name not in cfg:
            raise ValueError(f"Model '{model_name}' not in config")

        model_cfg = cfg[model_name]
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
            "contexts_cleared": 0,
        }

        # Stop the model first (always)
        await self.stop(model_name)

        if not cache_path:
            # No cache to clear — just record context cleanup and return
            for r in self._runners.values():
                if model_name in r._models:
                    del r._models[model_name]
                    result["contexts_cleared"] += 1
            return result

        cache_root = self._cache_root()
        cache_dir = self._cache_dir_for_checkpoint(cache_path)

        # Safety check: other running contexts sharing this cache?
        if cache_dir.exists():
            targets = []
            for ctx in self._get_model_contexts():
                if ctx.name == model_name or ctx.state != RunnerState.RUNNING:
                    continue
                other_cfg = cfg.get(ctx.name, {})
                other_raw = other_cfg.get("model")
                other_resolved = resolve_model_ref(
                    raw=other_raw,
                    default_section=default_section,
                    model_repos=self._cm.get("default/model-repos"),
                )
                if not other_resolved.cache_path:
                    continue
                if self._cache_dir_for_checkpoint(other_resolved.cache_path) == cache_dir:
                    targets.append(ctx.name)
            if targets:
                raise ValueError(
                    f"Model '{model_name}' is in use by other running runners: "
                    + ", ".join(targets)
                )

        # Delete cache, then clear contexts
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            result["cache_deleted"] = True
            result["cache_path"] = str(cache_dir)
        for r in self._runners.values():
            if model_name in r._models:
                del r._models[model_name]
                result["contexts_cleared"] += 1

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
        # Cancel any active download tasks
        for r in self._runners.values():
            models = getattr(r, '_models', {})
            for ctx in models.values():
                if ctx.download_task and not ctx.download_task.done():
                    ctx.download_task.cancel()
        for r in self._runners.values():
            await r.shutdown()
        self._runners.clear()
        self._next_port = self._cm.get("default/model-start-port", 18000)

    @property
    def running_models(self) -> Set[str]:
        result: Set[str] = set()
        for r in self._runners.values():
            result.update(r.running_models)
        return result

    # ── request API ────────────────────────────────────────────────────
    async def ainvoke(self, model_name: str, prompt: str = "", backend: Optional[str] = None,
                      messages: Optional[list] = None, **kwargs: Any) -> str:
        runner = self._get_runner(model_name, {}, backend)
        if messages is not None:
            return await runner.ainvoke(model_name, "", messages=messages, **kwargs)
        return await runner.ainvoke(model_name, prompt, **kwargs)

    async def astream(self, model_name: str, payload: Dict[str, Any], backend: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        runner = self._get_runner(model_name, {}, backend)
        if "messages" not in payload and "prompt" in payload:
            # Keep prompt-based flow (backward compat)
            pass
        # If messages are already in payload (list of dicts), they go through as-is
        async for chunk in runner.astream(model_name, payload):
            yield chunk

    async def request(self, model_name: str, path: str, **kwargs: Any) -> Any:
        runner = self._get_runner(model_name, {}, None)
        return await runner.request(model_name, path, **kwargs)

    async def get_logs(self, model_name: str, lines: int = 100) -> List[str]:
        """Return the last N log lines for a model."""
        for r in self._runners.values():
            if model_name in r._models:  # noqa: SLF001
                return await r.get_logs(model_name, lines)
        return []
