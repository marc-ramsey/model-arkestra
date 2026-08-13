"""Single entry point for all model operations — wraps ConfigManager + lazy runners."""
from __future__ import annotations
import asyncio
import sys
from typing import Any, AsyncIterator, Dict, Optional, Set

from llm_config_manager.config_manager import ConfigManager
from model_arkestra.base import BaseModelRunner
from model_arkestra.common import _resolve_backend
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

    def get_model_list(self) -> list[str]:
        """List of model names tracked at runtime (may include stopped/errored models)."""
        return [ctx.name for ctx in self._get_model_contexts()]

    def get_v1_models(self) -> Dict[str, Any]:
        """OpenAI-compatible ``/v1/models`` response."""
        from time import time

        data = []
        for ctx in self._get_model_contexts():
            model_cfg = self.get_model(ctx.name) or {}
            backend_id = ctx.backend_id or model_cfg.get("backend")
            owned_by = str(model_cfg.get("owned_by", "local")) if isinstance(model_cfg, dict) else "local"
            state_label = str(ctx.state).lower().replace("runnerstate.", "")

            entry: Dict[str, Any] = {
                "id": ctx.name,
                "object": "model",
                "created": int(time()),
                "owned_by": owned_by,
                "status": state_label,
                "port": ctx.port,
                "runner_type": ctx.runner_type,
                "backend_id": backend_id,
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

    # ── context manager ────────────────────────────────────────────────

    async def __aenter__(self) -> "ModelArkestra":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    # ── lifecycle API ──────────────────────────────────────────────────
    async def start(
        self,
        model_name: str,
        port: Optional[int] = None,
        backend: Optional[str] = None,
        runner: Optional[str] = None,
    ) -> None:
        """Start a model.

        * ``backend`` — resolves from config.yaml (the backend_id to use).
        * ``runner`` — explicit override for which runner to use:
          ``None`` / ``"process"`` → ProcessModelRunner,
          ``"podman"`` → PodmanModelRunner, ``"docker"`` → DockerModelRunner.
        """
        # Validate inputs before allocating anything
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Unknown model '{model_name}'.")

        backends_cfg = self._cm.data.get("backends", {})
        runners_cfg = self._cm.data.get("runners", {})

        # Validate explicit backend
        if backend and backend not in backends_cfg:
            raise ValueError(
                f"Unknown backend '{backend}'. Available: {list(backends_cfg.keys())}"
            )

        # Allocate port if not explicitly provided
        if port is None:
            port = self._alloc()

        # runner= selects the transport layer
        if runner is not None:
            inst = self._get_runner_instance(runner, model_name)
            await inst.start(model_name, port=port, backend=backend)
            ctx = inst._models[model_name]
            ctx.runner_type = runner
            return
        else:
            be_id = self._resolve_backend_id(model_name, {}, backend)
            be_cfg = backends_cfg.get(be_id, {})
            runner_type = str(be_cfg.get("runner", "process"))
            if runner_type not in runners_cfg and runner_type not in ("process", "podman", "docker"):
                raise ValueError(
                    f"Backend '{be_id}' resolves to unknown runner type '{runner_type}'. "
                    f"Available runners: {list(runners_cfg.keys())}"
                )
            runner = self._get_runner_instance(runner_type, model_name)

        await runner.start(model_name, port=port, backend=backend)
        ctx = runner._models[model_name]
        ctx.runner_type = runner_type

    async def stop(self, model_name: str) -> None:
        """Stop the named model."""
        for r in self._runners.values():
            if model_name in r._models:  # noqa: SLF001
                try:
                    await r.stop()
                except Exception:
                    pass
                return

    async def restart(
        self,
        model_name: str,
        *,
        backend: Optional[str] = None,
        runner: Optional[str] = None,
    ) -> None:
        """Stop a model and start a new instance on the same port.

        Optional ``backend`` / ``runner`` kwargs override the runner for
        the restarted instance; either, both, or neither may be supplied.
        """
        await self.stop(model_name)
        await self.start(model_name, backend=backend, runner=runner)

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
