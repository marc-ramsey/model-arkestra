"""Single entry point for all model operations — wraps ConfigManager + lazy runners."""
from __future__ import annotations
import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from llm_config_manager.config_manager import ConfigManager
from model_arkestra.base import BaseModelRunner
from model_arkestra.common import _resolve_backend, default_cache_root
from model_arkestra.docker import DockerModelRunner
from model_arkestra.podman import PodmanModelRunner
from model_arkestra.process import ProcessModelRunner
from model_arkestra.types import RunnerState, _ModelContext


class ModelArkestra:
    """Start/ainvoke/astream/stop all models through one object."""

    def __init__(self, config_path: str, start_port: int = 18000, **runner_kwargs: Any):
        self._cm = ConfigManager(config_path)
        self._next_port = self._cm.data.get('models-start-port', start_port)
        self._runners: Dict[str, BaseModelRunner] = {}
        self._runner_kwargs = runner_kwargs
        self._build_runner_class_map()

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
    
    # ── ConfigManager delegation ───────────────────────────────────────
    @property
    def cm(self) -> ConfigManager:
        return self._cm

    def get_model(self, model_name: str, env_vars: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        return self._cm.get_model(model_name, env_vars)

    def get_models(self) -> list:
        return self._cm.get_models()

    def get_backend(self, backend_id: str) -> Optional[Dict[str, Any]]:
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
        """OpenAI-compatible ``/v1/models`` response."""
        from time import time

        contexts_by_name = {ctx.name: ctx for ctx in self._get_model_contexts()}
        data = []
        for model_name in self.get_models():
            ctx = contexts_by_name.get(model_name)
            model_cfg = self.get_model(model_name) or {}
            owned_by = str(model_cfg.get("owned_by", "local")) if isinstance(model_cfg, dict) else "local"
            state_label = str(ctx.state).lower().replace("runnerstate.", "") if ctx else "stopped"

            data.append({
                "id": model_name,
                "object": "model",
                "created": int(time()),
                "owned_by": owned_by,
                "status": state_label,
            })

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
        for _cls in (ProcessModelRunner, PodmanModelRunner, DockerModelRunner):
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
        model = self.get_model(model_name) or {}
        return _resolve_backend(self._cm, model, model_name, override)

    def _get_runner(self, model_name: str, env_vars: Dict[str, Any], backend: Optional[str] = None) -> BaseModelRunner:
        # Find the runner that has this model
        for r in self._runners.values():
            if model_name in r._models and r._models[model_name].state == RunnerState.RUNNING:
                return r
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
                return str(be["runner"])
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

    def _cache_root(self) -> Path:
        """Resolve HF_HUB_CACHE or LLAMA_CACHE env var to a root Path."""
        for key in ("HF_HUB_CACHE", "LLAMA_CACHE"):
            val = (self._cm.data.get("env") or {}).get(key)
            if val:
                return Path(val).expanduser()
            val = os.environ.get(key)
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

        Example::

            await arkestra.start("qwen3-4b", temp=1.0, top_k=20)
            await arkestra.start("qwen3-4b", port=18000, backend="rocm", temp=0.9)
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

        # Allocate port if not explicitly provided
        if port is None:
            port = self._alloc()

        # runner= selects the transport layer
        if runner_type_override is not None:
            inst = self._get_runner_instance(runner_type_override, model_name)
            await inst.start(model_name, port=port, backend=backend, **inference_kwargs)
            ctx = inst._models[model_name]
            ctx.runner_type = runner_type_override
            ctx._runner = inst
            return

        # Resolve backend + runner type from config
        be_id = self._resolve_backend_id(model_name, {}, backend)
        be_cfg = backends_cfg.get(be_id, {})
        resolved_runner = str(be_cfg.get("runner", "process"))

        if resolved_runner not in runners_cfg and resolved_runner not in ("process", "podman", "docker"):
            raise ValueError(
                f"Backend '{be_id}' resolves to unknown runner type '{resolved_runner}'. "
                f"Available runners: {list(runners_cfg.keys())}"
            )

        runner = self._get_runner_instance(resolved_runner, model_name)
        await runner.start(model_name, port=port, backend=backend, **inference_kwargs)
        ctx = runner._models[model_name]
        ctx.runner_type = resolved_runner
        ctx._runner = runner

    async def stop(self, model_name: str) -> None:
        """Stop the named model."""
        for r in self._runners.values():
            if model_name in r._models:  # noqa: SLF001
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

        cache_dir = self._cache_dir_for_checkpoint(checkpoint)

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
        runner = self._get_runner(model_name, {}, backend)
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
