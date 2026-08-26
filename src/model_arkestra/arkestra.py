"""Single entry point for all model operations — wraps ConfigManager + lazy runners."""
from __future__ import annotations
import asyncio
import os
import shutil
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from llm_config_manager.config_manager import ConfigManager
from model_arkestra.base import BaseModelRunner
from model_arkestra.common import (
    _resolve_backend, default_cache_root, resolve_config_path,
)
from model_arkestra.docker import DockerModelRunner
from model_arkestra.onnx_runner import OnnxRunner
from model_arkestra.podman import PodmanModelRunner
from model_arkestra.process import ProcessModelRunner
from model_arkestra.remote import RemoteModelRunner
from model_arkestra.types import RunnerState, _ModelContext
from model_arkestra.unicode_ringbuffer import UnicodeRingBuffer
from model_arkestra.http_proxy import model_status_for_ctx


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
        self._cm = ConfigManager(str(self._config_path))
        self._next_port = self._cm.data.get('models-start-port', start_port)
        # Load backends.yaml from same directory as config (if present)
        parent = self._config_path.parent
        backends_path = parent / "backends.yaml"
        if backends_config is not None:
            self._backends_cfg = backends_config
        elif backends_path.exists():
            with open(backends_path) as f:
                self._backends_cfg = yaml.safe_load(f) or {}
        else:
            self._backends_cfg = {}
        # Merge backends.yaml into config data so cm.get_backend() finds both
        be_section = self._backends_cfg.get("backends", {})
        if be_section and isinstance(be_section, dict):
            existing_backends = self._cm.data.get("backends") or {}
            if not isinstance(existing_backends, dict):
                existing_backends = {}
            for k, v in be_section.items():
                if k == "default":
                    continue  # skip the 'default' key (just picks a backend name)
                existing_backends[k] = v
            self._cm.data["backends"] = existing_backends
        self._runners: Dict[str, BaseModelRunner] = {}
        self._runner_kwargs = runner_kwargs
        self._build_runner_class_map()
        # Extract sources section for binary_downloader compatibility
        self._sources: Dict[str, Any] = self._backends_cfg.get("sources", {})
        # ── Global log buffer (single ring for all server-level events) ─
        app_log_lines = int(self._cm.data.get('app-log-lines', 2000))
        self._global_log_buf = UnicodeRingBuffer(app_log_lines * _ModelContext.AVG_LINE_BYTES)
        self._global_log_seq: int = 0
        # ── Backend runtime validation (hard error on mismatch) ───────
        self._validate_backend_runtime()

    # ── port allocation (global) ───────────────────────────────────────
    async def _log(self, text: str) -> None:
        """Append a line to the global log buffer."""
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
    def _alloc(self) -> int:
        start_port = self._cm.data.get('models-start-port', 18000)
        max_ports = self._cm.data.get('model-ports', 32)
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
        backends = self._cm.data.get("backends") or {}
        if not isinstance(backends, dict):
            return  # no backends section — skip validation
        backend_id = backends.get("default")
        if not backend_id:
            return  # no default backend set — skip validation

        runtime_checks = {
            "vulkan-radv": self._check_vulkan,
            "rocm": self._check_rocm,
            "cuda": self._check_nvidia,
        }
        checker = runtime_checks.get(backend_id)
        if checker:
            if not checker():
                suggestion = "Run 'model-arkestra init' to auto-detect your GPU."
                raise RuntimeError(
                    f"Backend '{backend_id}' configured but runtime not detected. {suggestion}"
                )
        # CPU backends and unknown IDs pass by default

    @staticmethod
    def _check_vulkan() -> bool:
        try:
            result = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True, timeout=5, text=True,
            )
            return result.returncode == 0 and "Vulkan Instance Version" in result.stdout
        except (FileNotFoundError, OSError):
            return False

    @staticmethod
    def _check_rocm() -> bool:
        try:
            subprocess.run(
                ["rocm-smi", "--showconfig"],
                capture_output=True, timeout=5,
            )
            return True
        except (FileNotFoundError, OSError):
            pass
        # Fallback: check for ROCm lib directory
        return any(
            p.exists() for p in [Path("/opt/rocm/lib"), Path("/usr/lib64/rocm")]
        )

    @staticmethod
    def _check_nvidia() -> bool:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv"],
                capture_output=True, timeout=5, text=True,
            )
            return result.returncode == 0 and "NVIDIA" in result.stdout
        except (FileNotFoundError, OSError):
            return False
    
    def _get_existing_port(self, model_name: str) -> Optional[int]:
        """Find the port from a stopped context for *model_name*, if one exists.

        Returns the port number if the model has an existing STOPPED context,
        or None if the model is new (never started) or only has RUNNING contexts.
        This enables stop→restart to reuse the same port instead of allocating
        a new one from the pool — prevents port exhaustion on repeated cycles.
        """
        for runner in self._runners.values():
            ctx = runner._models.get(model_name)
            if ctx is not None and ctx.port is not None:
                return ctx.port
        return None

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
        be = self._backends_cfg.get("backends", {}).get(backend_id)
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
        for ctx in self._get_model_contexts():
            if ctx.name == model_name:
                return ctx
        return None

    def get_v1_models(self) -> Dict[str, Any]:
        """OpenAI-compatible ``/v1/models`` response with WebUI status fields."""
        from time import time

        contexts_by_name = {ctx.name: ctx for ctx in self._get_model_contexts()}
        data = []
        for model_name in self.get_models():
            ctx = contexts_by_name.get(model_name)
            model_cfg = self.get_model(model_name) or {}
            owned_by = str(model_cfg.get("owned_by", "local")) if isinstance(model_cfg, dict) else "local"

            entry: Dict[str, Any] = {
                "id": model_name,
                "object": "model",
                "created": int(time()),
                "owned_by": owned_by,
                "status": model_status_for_ctx(ctx),
            }

            data.append(entry)

        return {"object": "list", "data": data}

    # ── runner class registry (config-driven) ─────────────────────────

    def _build_runner_class_map(self) -> None:
        """Build the mapping from runner type strings to concrete classes.

        Built-in classes are registered first keyed by their lowercase name
        (e.g. ``ProcessModelRunner`` → ``"process"``).  The ``runners:`` section
        in config.yaml can then override or extend this map.
        """
        self._runner_classes: Dict[str, type] = {}

        # Built-in concrete runners keyed by their lowercase short name.
        for _cls in (ProcessModelRunner, PodmanModelRunner, DockerModelRunner, OnnxRunner, RemoteModelRunner):
            key = _cls.__name__.lower().replace("modelrunner", "")
            self._runner_classes[key] = _cls

        # Config can override built-ins or add entirely new ones.
        runner_cfg = self._cm.data.get("runners") or {}
        for key, class_name in runner_cfg.items():
            if key == "default":
                continue
            target = getattr(sys.modules[__name__], class_name, None)
            if target is not None:
                self._runner_classes[key] = target

    def _get_runner_instance(self, runner_type: str, model_name: Optional[str] = None) -> BaseModelRunner:
        """Instantiate a fresh runner per ``model_name`` (one runner per model)."""
        key = f"{runner_type}:{model_name}" if model_name else runner_type
        if key not in self._runners:
            cls = self._runner_classes.get(runner_type)
            if cls is None:
                raise ValueError(
                    f"Unknown runner type '{runner_type}'. "
                    f"Available: {list(self._runner_classes.keys())}"
                )
            self._runners[key] = cls(self._cm, **self._runner_kwargs)
            self._runners[key]._arkestra = self  # reference for resolve_config
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
        backends = self._cm.data.get("backends", {})
        if isinstance(backends, dict):
            backend_id = self._resolve_backend_id(
                model_name, env_vars, override_backend
            )
            be = backends.get(backend_id, {})
            if isinstance(be, dict) and "runner" in be:
                rtype = str(be["runner"])
                # Resolve special "container" runner to the top-level default
                if rtype == "container":
                    rtype = self._cm.data.get("container_type", "process") or "process"
                return rtype
        runners_cfg = self._cm.data.get("runners", {})
        if isinstance(runners_cfg, dict):
            default_type = str(runners_cfg.get("default", "process"))
            # If backend resolution gave us something, verify it exists in runners
            backends_section = self._cm.data.get("backends", {})
            if isinstance(backends_section, dict) and override_backend:
                be = backends_section.get(override_backend, {})
                if isinstance(be, dict) and "runner" in be:
                    rtype = str(be["runner"])
                    if rtype in runners_cfg:
                        return rtype
            return default_type
        return "process"

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
        return (self._cm.data.get("env") or {}).get(key)

    def _cache_root(self) -> Path:
        """Resolve HF_HUB_CACHE to a root Path."""
        val = self.resolve_config("HF_HUB_CACHE")
        if val:
            return Path(val).expanduser()
        return default_cache_root()

    def _cache_dir_for_checkpoint(self, checkpoint: str) -> Path:
        """Return the cache directory path for a given checkpoint string."""
        return self._cache_root() / f"models--{checkpoint.replace('/', '--')}"

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

        ``**overrides`` is a flat mix of infrastructure and inference kwargs:

        * Infra keys (handled by ModelArkestra): ``port``, ``backend``, ``runner``
        * Everything else → inference param, passed through to runner.start()
          as bare kwargs. Converted to ``--flag value`` CLI flags at subprocess
          boundary.

        Runner values: ``process``, ``podman``, ``docker``, or ``container``
        (resolves to ``container_type`` from top-level config).

        Example::

            await arkestra.start("qwen3-4b", temp=1.0, top_k=20)
            await arkestra.start("qwen3-4b", port=18000, backend="rocm-container", temp=0.9)
        """
        # Validate inputs before allocating anything
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Unknown model '{model_name}'.")

        backends_cfg = self._cm.data.get("backends", {})
        runners_cfg = self._cm.data.get("runners", {})

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

        # Detect ONNX models — they don't need ports or HTTP health checks
        resolved_backend = backend or self._resolve_backend_id(model_name, {})
        is_onnx = str(resolved_backend) == "onnx"

        if is_onnx:
            return await self._start_onnx_model(model_name, inference_kwargs)

        # Detect remote models — proxy all calls to target worker
        resolved_be_id = backend or self._resolve_backend_id(model_name, {})
        be_cfg = backends_cfg.get(str(resolved_be_id), {})
        is_remote = str(be_cfg.get("runner", "")) == "remote"

        if is_remote:
            return await self._start_remote_model(model_name, inference_kwargs, be_cfg)

        # Allocate port if not explicitly provided.
        # Check for existing stopped context first — reuse its port to avoid
        # exhausting the port pool on repeated stop/start cycles (the same model).
        if port is None:
            port = self._get_existing_port(model_name)
            if port is None:
                port = self._alloc()

        # runner= selects the transport layer
        if runner_type_override is not None:
            inst = self._get_runner_instance(runner_type_override, model_name)
            await inst.start(model_name, port=port, backend=backend, **inference_kwargs)
            ctx = inst._models[model_name]
            ctx.runner_type = runner_type_override
            ctx._runner = inst
            await self._log(f"[action=start model={model_name} port={port}]")
            return

        # Resolve backend + runner type from config
        be_id = self._resolve_backend_id(model_name, {}, backend)
        be_cfg = backends_cfg.get(be_id, {})
        resolved_runner = str(be_cfg.get("runner", "process"))
        if resolved_runner == "container":
            resolved_runner = self._cm.data.get("container_type", "process") or "process"

        if resolved_runner not in runners_cfg and resolved_runner not in ("process", "podman", "docker", "onnx", "remote"):
            raise ValueError(
                f"Backend '{be_id}' resolves to unknown runner type '{resolved_runner}'. "
                f"Available runners: {list(runners_cfg.keys())}"
            )

        runner = self._get_runner_instance(resolved_runner, model_name)
        await runner.start(model_name, port=port, backend=backend, **inference_kwargs)
        ctx = runner._models[model_name]
        ctx.runner_type = resolved_runner
        ctx._runner = runner
        await self._log(f"[action=start model={model_name} port={port}]")

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
            log_size = inference_kwargs.get("max_log_lines", self.cm.data.get('log-buffer-size') or 2000)
            model_data = self.get_model(model_name, env_vars={})
            model_path_str = str((model_data or {}).get("model_path", ""))
            if not model_path_str:
                checkpoint = (model_data or {}).get("checkpoint", "")
                model_path_str = checkpoint  # fall back to checkpoint field

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
        self, model_name: str, inference_kwargs: Dict[str, Any], backend_cfg: Dict[str, Any],
    ) -> None:
        """Start a remote model — proxy lifecycle to the target arkestra worker.

        No local port is allocated. All HTTP calls (start, stop, chat, embed)
        are forwarded to ``base_url`` from the backend configuration.
        """
        base_url = str(backend_cfg.get("base_url", "")).rstrip("/")
        if not base_url:
            raise ValueError(f"Backend '{model_name}' uses runner: remote but has no base_url.")

        # Find or create the remote runner instance (by backend_id)
        backends_section = self._cm.data.get("backends", {})
        runner = self._get_runner_instance("remote")  # single shared runner per base_url
        runner._backend_id = str(backend_cfg)  # tag for routing
        ctx = self.find_context(model_name)
        if ctx is None:
            log_size = inference_kwargs.get("max_log_lines", self.cm.data.get('log-buffer-size') or 2000)
            from model_arkestra.types import _ModelContext as MC
            ctx = MC(model_name, 0, max_log_lines=log_size)  # port=0 for remote models
            ctx.backend_id = backend_cfg.get("runner")
            ctx._remote_base_url = base_url
            runner._models[model_name] = ctx  # noqa: SLF001

        # Pass inference kwargs and start (proxies to worker)
        runner._inference_kwargs[model_name] = inference_kwargs
        await runner.start(model_name, port=ctx.port, backend="remote", **inference_kwargs)
        ctx.runner_type = "remote"
        ctx._runner = runner  # noqa: SLF001
        await self._log(f"[action=start model={model_name} remote={base_url}]")

    async def embed(self, model_name: str, text: str) -> Dict[str, Any]:
        """Encode text → embedding vector via ONNX model.

        Returns OpenAI-compatible response with ``data[].embedding`` list.
        """
        runner = self._get_runner("onnx", {}, None)
        return await runner.embed(model_name, text)  # type: ignore[attr-defined]

    async def transcribe(self, model_name: str, audio_bytes: bytes,
                         language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio → text via Whisper ONNX model.

        Expects raw WAV audio bytes. Returns {"text": "...", "language": "..."}.
        """
        runner = self._get_runner("onnx", {}, None)
        return await runner.transcribe(model_name, audio_bytes, language)  # type: ignore[attr-defined]

    async def synthesize(self, model_name: str, text: str) -> bytes:
        """Generate speech from text via TTS ONNX model.

        Returns raw WAV audio bytes.
        """
        runner = self._get_runner("onnx", {}, None)
        return await runner.synthesize(model_name, text)  # type: ignore[attr-defined]

    async def stop(self, model_name: str) -> None:
        """Stop the named model."""
        for r in self._runners.values():
            if model_name in r._models:  # noqa: SLF001
                await self._log(f"[action=stop model={model_name}]")
                try:
                    await r.stop()
                except Exception:
                    pass
                return

    async def eject(self, model_name: str) -> Dict[str, Any]:
        """Stop a model and delete its cached checkpoint files.

        Returns a dict with details about what was removed.  Raises ValueError
        if other running models share the same underlying cache directory.
        """
        cfg = self._cm.data.get("models") or {}
        if model_name not in cfg:
            raise ValueError(f"Model '{model_name}' not in config")

        checkpoint = cfg[model_name].get("checkpoint", "")
        result: Dict[str, Any] = {
            "ok": True,
            "model": model_name,
            "cache_deleted": False,
            "contexts_cleared": 0,
        }

        # Stop the model first (always)
        await self.stop(model_name)

        if not checkpoint:
            # No cache to clear — just record context cleanup and return
            for r in self._runners.values():
                if model_name in r._models:
                    del r._models[model_name]
                    result["contexts_cleared"] += 1
            return result

        cache_root = self._cache_root()
        cache_dir = cache_root / f"models--{checkpoint.replace('/', '--')}"

        # Safety check: other running contexts sharing this cache?
        if cache_dir.exists():
            targets = [
                ctx.name
                for ctx in self._get_model_contexts()
                if ctx.name != model_name
                and ctx.state == RunnerState.RUNNING
                and (
                    cfg.get(ctx.name, {}).get("checkpoint", "")
                    and self._cache_dir_for_checkpoint(cfg[ctx.name]["checkpoint"]).resolve()
                    == cache_dir.resolve()
                )
            ]
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
        await self._log(f"[action=shutdown]")
        for r in self._runners.values():
            await r.shutdown()
        self._runners.clear()
        self._next_port = self._cm.data.get('models-start-port', 18000)

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
