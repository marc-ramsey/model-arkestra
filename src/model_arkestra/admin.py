"""ArkestraAdmin — administrative routes for ModelArkestra.

This module defines ArkestraAdmin, a subcomponent of ArkestraServer that
installs admin endpoints (GET /, GET/POST /admin/*) onto the same FastAPI app.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
except ImportError:
    raise RuntimeError("model_arkestra.admin requires fastapi")


class ArkestraAdmin:
    """Admin subcomponent that installs routes on an ArkestraServer's app."""

    def __init__(self, server: "ArkestraServer", admin_key: Optional[str], app: FastAPI):
        self.server = server
        self.admin_key = admin_key
        self._app = app
        self._installed = False

    def install(self) -> "ArkestraAdmin":
        """Install all admin routes on the FastAPI app. Idempotent."""
        if self._installed:
            return self
        self._add_root_route()
        self._add_auth_middleware()
        self._add_models_route()
        self._add_stop_route()
        self._add_update_route()
        self._add_eject_route()
        self._installed = True
        return self

    # ── helpers ────────────────────────────────────────────────────────

    def _resolve_env(self, key: str, explicit: Optional[str] = None) -> str:
        """Resolve an env var: method arg > config env > OS env."""
        if explicit is not None:
            return explicit
        cfg_env = self.server._arkestra.cm.data.get("env", {})
        if key in cfg_env and cfg_env[key]:
            return str(cfg_env[key])
        import os
        return os.environ.get(key, "")

    def _add_root_route(self) -> None:
        from pathlib import Path

        from fastapi.responses import FileResponse

        html = Path(__file__).parent.parent.parent / "static" / "index.html"

        @self._app.get("/")
        async def root():
            return FileResponse(html, media_type="text/html")

        @self._app.get("/index.html")
        async def index_html():
            return FileResponse(html, media_type="text/html")

    def _add_auth_middleware(self) -> None:
        if not self.admin_key:
            return

        @self._app.middleware("http")
        async def admin_auth(request: Request, call_next):
            if request.url.path.startswith("/admin"):
                key = request.headers.get("X-Admin-Key", "")
                if not key or key != self.admin_key:
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Invalid or missing X-Admin-Key"},
                    )
            return await call_next(request)

    def _add_models_route(self) -> None:
        import os
        from pathlib import Path

        @self._app.get("/admin/models")
        async def admin_models():
            try:
                cfg = self.server._arkestra.cm.data.get("models") or {}
                contexts_by_name = {ctx.name: ctx for ctx in self.server._arkestra._get_model_contexts()}

                data = []
                for model_name in self.server._arkestra.get_models():
                    ctx = contexts_by_name.get(model_name)
                    model_cfg = self.server._arkestra.get_model(model_name) or {}

                    if ctx:
                        status = str(ctx.state).lower().replace("runnerstate.", "")
                        entry = {
                            "id": ctx.name,
                            "status": status,
                            "port": ctx.port,
                            "runner_type": ctx.runner_type,
                            "backend_id": ctx.backend_id or model_cfg.get("backend"),
                        }
                    else:
                        checkpoint = model_cfg.get("checkpoint", "")
                        hf_cache = self._resolve_env("HF_HUB_CACHE") or self._resolve_env("LLAMA_CACHE") or "~/.cache/huggingface/hub"
                        cache_path = Path(hf_cache).expanduser() / f"models--{checkpoint.replace('/', '--')}" if checkpoint else None
                        is_cached = cache_path.exists() if cache_path else False

                        entry = {
                            "id": model_name,
                            "status": "stopped" if is_cached else "uncached",
                            "port": None,
                            "runner_type": None,
                            "backend_id": model_cfg.get("backend"),
                        }
                    data.append(entry)

                return {"object": "list", "data": data}
            except Exception as e:
                raise HTTPException(status_code=503, detail=str(e))

    def _add_stop_route(self) -> None:
        from model_arkestra.types import RunnerState

        @self._app.post("/admin/stop/{model}")
        async def admin_stop(model: str):
            ctxs = list(self.server._arkestra._get_model_contexts())
            matching = [c for c in ctxs if c.name == model]
            if not matching:
                raise HTTPException(
                    status_code=404, detail=f"Model '{model}' not found in runners"
                )
            prev_state = matching[0].state
            if prev_state in (RunnerState.STOPPED, RunnerState.STOPPING):
                return {
                    "ok": True,
                    "model": model,
                    "previous_state": str(prev_state),
                }, 202
            await self.server._arkestra.stop(model)
            return {
                "ok": True,
                "model": model,
                "previous_state": str(prev_state),
            }

    def _add_update_route(self) -> None:
        import copy
        from fastapi.responses import JSONResponse

        @self._app.post("/admin/update/{model}")
        async def admin_update(
            model: str,
            request: Request,
        ):
            # Collect all query params except 'name' and 'port'
            raw_params = dict(request.query_params)
            override_params = {k: v for k, v in raw_params.items() if k not in ("name", "port")}

            # Check model exists in config
            cfg = self.server._arkestra.cm.data.get("models") or {}
            if model not in cfg:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not in config")

            # Snapshot current entry for atomic rollback
            old_key = model
            snapshot = copy.deepcopy(cfg[model])

            # Enforce name uniqueness (name is excluded from overrides but still checked)
            new_name = raw_params.get("name", "")
            if new_name and new_name != model and new_name in cfg:
                raise HTTPException(
                    status_code=409,
                    detail=f"Model name '{new_name}' already exists in config",
                )

            # Apply changes to config
            if new_name and new_name != model:
                cfg[new_name] = cfg.pop(model)
                model = new_name
            else:
                for k, v in override_params.items():
                    cfg[model][k] = v

            # Write updated config back to disk
            self.server._arkestra.cm.export(self.server._arkestra.cm.config_path)

            # Restart the model with new backend/runner if provided
            backend = override_params.get("backend")
            runner = override_params.get("runner")
            try:
                await self.server._arkestra.restart(model, backend=backend, runner=runner)
            except Exception as exc:
                # Rollback: restore snapshot under the old key
                cfg[old_key] = snapshot
                if new_name and new_name != old_key:
                    del cfg[new_name]
                self.server._arkestra.cm.export(self.server._arkestra.cm.config_path)
                raise HTTPException(status_code=500, detail=f"Restart failed: {exc}")

            return {"ok": True, "model": model}

    def _add_eject_route(self) -> None:
        import os
        from pathlib import Path

        @self._app.post("/admin/eject/{model}")
        async def admin_eject(model: str):
            # 0. Stop the model if it's running
            try:
                await self.server._arkestra.stop(model)
            except Exception:
                pass

            # 1. Look up checkpoint in config
            cfg = self.server._arkestra.cm.data.get("models") or {}
            if model not in cfg:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not in config")

            checkpoint = cfg[model].get("checkpoint", "")
            if not checkpoint:
                return {"ok": True, "model": model}

            # 2. Find and delete cache entry under HF_HUB_CACHE
            hf_cache = os.environ.get("HF_HUB_CACHE", os.environ.get("LLAMA_CACHE", "~/.cache/huggingface/hub"))
            cache_dir = Path(hf_cache).expanduser() / f"models--{checkpoint.replace('/', '--')}"
            if cache_dir and cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)

            # 3. Clear all runner context entries for this model
            for r in self.server._arkestra._runners.values():
                r._models.pop(model, None)  # noqa: SLF001

            return {"ok": True, "model": model}


# Type hints — resolved at runtime via string ref
from model_arkestra.server import ArkestraServer  # noqa: E402, F401 (imported for type hint only in __init__)
