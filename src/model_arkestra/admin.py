"""ArkestraAdmin — administrative routes for ModelArkestra.

This module defines ArkestraAdmin, a subcomponent of ArkestraServer that
installs admin endpoints (GET /, GET/POST /admin/*) onto the same FastAPI app.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, HTMLResponse
except ImportError:
    raise RuntimeError("model_arkestra.admin requires fastapi")

from model_arkestra.common import (
    _runtime_binary,
    build_image,
    containerfile_for_backend,
    default_cache_root,
    image_and_runner_for_backend,
    image_exists as _image_exists,
    remove_image,
)
from model_arkestra.types import RunnerState

# ── Model config field definitions (single source of truth) ─────────────
MODEL_CONFIG_FIELDS = frozenset({"args", "checkpoint", "backend", "capabilities", "runner", "tags", "max_log_lines"})
INFRA_KEYS = frozenset({"args", "checkpoint", "backend", "runner", "max_log_lines"})


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
        self._add_config_routes()
        self._add_stop_route()
        self._add_stop_all_route()
        self._add_shutdown_route()
        self._add_start_route()
        self._add_eject_route()
        self._add_log_route()
        self._add_images_route()
        self._installed = True
        return self

    # ── helpers ────────────────────────────────────────────────────────

    @property
    def _config_data(self) -> dict:
        return self.server._arkestra.cm.data

    @property
    def _models_cfg(self) -> Dict[str, Any]:
        return self.server._arkestra.cm.data.get("models") or {}

    @property
    def _backends_cfg(self) -> Dict[str, Any]:
        return self.server._arkestra.cm.data.get("backends") or {}

    def _resolve_env(self, key: str, explicit: Optional[str] = None) -> str:
        """Resolve an env var: method arg > config env > OS env."""
        if explicit is not None:
            return explicit
        cfg_env = self.server._arkestra.cm.data.get("env", {})
        if key in cfg_env and cfg_env[key]:
            return str(cfg_env[key])
        return os.environ.get(key, "")

    def _backend_for_image(self, image_tag: str) -> Optional[str]:
        """Return the backend_id whose ``image`` matches *image_tag*, or None."""
        backends = self._backends_cfg
        for bid, be_cfg in backends.items():
            if isinstance(be_cfg, dict) and str(be_cfg.get("image", "")) == image_tag:
                return bid
        return None

    def _add_root_route(self) -> None:
        html = Path(__file__).parent.parent.parent / "static" / "index.html"
        content = html.read_text().replace("{{ADMIN_KEY}}", self.admin_key or "")

        @self._app.get("/")
        @self._app.get("/index.html")
        async def root():
            return HTMLResponse(content, media_type="text/html",
                                headers={"Cache-Control": "no-store"})

    def _add_auth_middleware(self) -> None:
        # Always install the middleware (plumbing stays in place even when no key is set).
        # When admin_key is empty/None, it's a no-op pass-through.
        @self._app.middleware("http")
        async def admin_auth(request: Request, call_next):
            path = request.url.path
            is_admin = path == "/admin" or path.startswith("/admin/")
            if is_admin and self.admin_key:
                key = request.headers.get("x-admin-key", "")
                if key != self.admin_key:
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Invalid or missing admin_key header"},
                    )
            return await call_next(request)

    def _add_models_route(self) -> None:
        @self._app.get("/admin/models")
        async def admin_models():
            try:
                cfg = self._models_cfg
                contexts_by_name = {ctx.name: ctx for ctx in self.server._arkestra._get_model_contexts()}

                data = []

                # Resolve cache root once — it's a global config value, not per-model.
                hf_cache = self._resolve_env("HF_HUB_CACHE") or str(default_cache_root())

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
                            "args": model_cfg.get("args", ""),
                            "checkpoint": model_cfg.get("checkpoint", ""),
                            "capabilities": model_cfg.get("capabilities", []),
                        }
                    else:
                        checkpoint = model_cfg.get("checkpoint", "")
                        # Strip revision tag (e.g. :Q4_K_M) — HF Hub directories
                        # store models under the base name only.
                        base_checkpoint = checkpoint.split(":")[0] if ":" in checkpoint else checkpoint
                        cache_path = Path(hf_cache).expanduser() / f"models--{base_checkpoint.replace('/', '--')}" if base_checkpoint else None
                        is_cached = cache_path.exists() if cache_path else False

                        entry = {
                            "id": model_name,
                            "status": "stopped" if is_cached else "uncached",
                            "port": None,
                            "runner_type": None,
                            "backend_id": model_cfg.get("backend"),
                            "args": model_cfg.get("args", ""),
                            "checkpoint": checkpoint,
                            "capabilities": model_cfg.get("capabilities", []),
                        }
                    data.append(entry)

                # Top-level metadata for dropdown options (static per-server)
                backends = self._backends_cfg
                runner_types = list(self.server._arkestra._runner_classes.keys()) if hasattr(self.server._arkestra, "_runner_classes") else []

                return {
                    "models": data,
                    "backends": backends,
                    "runner_types": runner_types,
                }
            except Exception as e:
                raise HTTPException(status_code=503, detail=str(e))

    def _add_stop_route(self) -> None:
        @self._app.post("/admin/stop/{model}")
        async def admin_stop(model: str):
            ctx = self.server._arkestra.find_context(model)
            if not ctx:
                raise HTTPException(
                    status_code=404, detail=f"Model '{model}' not found in runners"
                )
            prev_state = ctx.state
            if prev_state.is_terminal:
                return JSONResponse(
                    status_code=202,
                    content={
                        "ok": True,
                        "model": model,
                        "previous_state": str(prev_state),
                    },
                )
            await self.server._arkestra.stop(model)
            return {
                "ok": True,
                "model": model,
                "previous_state": str(prev_state),
            }

    def _add_stop_all_route(self) -> None:
        @self._app.post("/admin/stop-all")
        async def admin_stop_all():
            ctxs = list(self.server._arkestra._get_model_contexts())
            running = [c.name for c in ctxs if not c.state.is_terminal]
            if not running:
                return JSONResponse(
                    status_code=200,
                    content={"ok": True, "message": "No models running — nothing to stop", "stopped": []},
                )
            await self.server._arkestra.stop_all()
            return JSONResponse(
                status_code=200,
                content={
                    "ok": True,
                    "message": f"Stopped {len(running)} model(s) — will restart implicitly on next request",
                    "stopped": running,
                },
            )

    def _add_shutdown_route(self) -> None:
        @self._app.post("/admin/shutdown")
        async def admin_shutdown():
            result = {"ok": True, "message": "Server shutting down"}

            async def do_shutdown():
                # Brief pause so uvicorn flushes the response to the client
                await asyncio.sleep(0.2)
                print("[SHUTDOWN] Stopping all models gracefully…", flush=True)
                try:
                    await self.server._arkestra.shutdown()
                    print("[SHUTDOWN] All models stopped.", flush=True)
                except Exception as e:
                    print(f"[SHUTDOWN] Error during model stop: {e}", flush=True)
                # _server is set via start() or when embedded
                # (as in live tests that attach server_obj to proxy._server).
                if self.server._server:
                    await self.server._server.shutdown()
                    print("[SHUTDOWN] uvicorn stopped.", flush=True)
                else:
                    # CLI mode — no Server object to tell, just exit
                    sys.exit(0)

            task = asyncio.create_task(do_shutdown())
            if hasattr(self._app, "_shutdown_task"):
                del self._app._shutdown_task
            self._app._shutdown_task = task  # type: ignore
            return JSONResponse(status_code=200, content=result)

    def _add_config_routes(self) -> None:
        @self._app.get("/admin/config")
        async def admin_config_list():
            """Return list of model names in config."""
            cfg = self._models_cfg
            return {"models": list(cfg.keys())}

        @self._app.post("/admin/config")
        async def admin_config_create(body: Dict[str, Any]):
            """Create a new model entry in config. Requires at least 'checkpoint'."""
            cfg = self._models_cfg

            # Validate required fields
            if not body.get("checkpoint"):
                raise HTTPException(
                    status_code=400, detail="'checkpoint' is required to create a model"
                )

            name = body.get("name") or (body.get("checkpoint", "").split(":")[0].rsplit("/", 1)[-1] if body.get("checkpoint") else "")
            if not name:
                raise HTTPException(
                    status_code=400, detail="Could not determine model name from checkpoint"
                )

            if name in cfg:
                raise HTTPException(
                    status_code=409, detail=f"Model '{name}' already exists in config"
                )

            # Build new model entry from body fields (with safe defaults)
            new_model: Dict[str, Any] = {
                "checkpoint": str(body["checkpoint"]),
            }
            for key in MODEL_CONFIG_FIELDS:
                if key in body and body[key] is not None:
                    new_model[key] = body[key]

            cfg[name] = new_model
            self.server._arkestra.cm.export(self.server._arkestra.cm.config_path)
            return JSONResponse(
                status_code=201,
                content={"ok": True, "model": name},
            )

        @self._app.get("/admin/config/{model}")
        async def admin_config_get(model: str):
            """Return a single model's configuration."""
            cfg = self._models_cfg
            if model not in cfg:
                raise HTTPException(
                    status_code=404, detail=f"Model '{model}' not in config"
                )
            # Also resolve current runtime status if the model is loaded
            contexts = {ctx.name: ctx for ctx in self.server._arkestra._get_model_contexts()}
            ctx = contexts.get(model)
            status = str(ctx.state).lower().replace("runnerstate.", "") if ctx else None

            return {"ok": True, "model": model, "config": copy.deepcopy(cfg[model]), "status": status}

        @self._app.put("/admin/config/{model}")
        async def admin_config_update(model: str, body: Dict[str, Any]):
            """Update an existing model's configuration."""
            cfg = self._models_cfg
            if model not in cfg:
                raise HTTPException(
                    status_code=404, detail=f"Model '{model}' not in config"
                )

            # Snapshot for rollback
            snapshot = copy.deepcopy(cfg[model])

            try:
                for key, value in body.items():
                    if key in MODEL_CONFIG_FIELDS:
                        cfg[model][key] = value
                self.server._arkestra.cm.export(self.server._arkestra.cm.config_path)
                return {"ok": True, "model": model}
            except Exception as exc:
                cfg[model].update(snapshot)
                raise HTTPException(status_code=500, detail=f"Save failed: {exc}")

    def _add_start_route(self) -> None:
        @self._app.post("/admin/start/{model}")
        async def admin_start(model: str, body: Dict[str, Any] | None = None):
            cfg = self._models_cfg
            if model not in cfg:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not in config")

            # Build raw kwargs — infra keys handled by ModelArkestra, rest are inference params
            kw = {}
            for key in INFRA_KEYS:
                if body and key in body and body[key] is not None:
                    val = body[key]
                    if key == "max_log_lines":
                        try:
                            val = int(val)
                        except (ValueError, TypeError):
                            continue
                    kw[key] = val
            # Any other keys in body are inference params — pass through as-is
            if body:
                for key, value in body.items():
                    if key not in INFRA_KEYS and value is not None:
                        kw[key] = value

            # If model is already running with overrides, stop first then start fresh
            if kw and self.server._arkestra.find_context(model):
                await self.server._arkestra.stop(model)

            try:
                await self.server._arkestra.start(model, **kw)
                ctx = self.server._arkestra.find_context(model)
                port = ctx.port if ctx else None
                return {"ok": True, "model": model, "port": port}
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Start failed: {exc}")

    def _add_log_route(self) -> None:
        @self._app.get("/admin/log/{model}")
        async def admin_log(
            model: str,
            since: int = 0,
            lines: int = 100,
        ):
            # Validate model exists in config first
            cfg = self._models_cfg
            if model not in cfg:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not in config")

            ctx = self.server._arkestra.find_context(model)
            if not ctx:
                # Model not yet started — return empty result
                return {
                    "since": 0,
                    "missed_lines": 0,
                    "lines": [],
                }

            # Read delta from ring buffer
            new_lines, oldest_seq = ctx._get_lines_since(since, lines)

            # Calculate missed lines (entries pruned from ring before the requested since point)
            if since > 0:
                # Oldest valid seq is _oldest_valid_seq. If since > current_log_seq,
                # the entire requested range has been evicted.
                total_missed = max(0, min(since, ctx._log_seq) - oldest_seq)
            else:
                total_missed = 0

            return {
                "since": ctx._log_seq,
                "missed_lines": total_missed,
                "lines": [{"seq": seq, "text": text} for seq, text in new_lines],
            }

    def _add_images_route(self) -> None:
        @self._app.get("/admin/images")
        async def admin_images():
            cm_data = self._config_data
            backends = self._backends_cfg
            if not isinstance(backends, dict):
                raise HTTPException(status_code=500,
                                    detail="No backends section in config")

            # First pass: build metadata list with defaults.
            entries: list[dict] = []
            for idx, (backend_id, be_cfg) in enumerate(backends.items()):
                if backend_id == "default" or not isinstance(be_cfg, dict):
                    continue
                image_tag = str(be_cfg.get("image", ""))
                container_path = be_cfg.get("container", "")
                _, runner_type = image_and_runner_for_backend(cm_data, backend_id)
                runtime_detected = _runtime_binary(runner_type) is not None
                entries.append({
                    "backend_id": backend_id,
                    "runner": runner_type,
                    "runtime_detected": runtime_detected,
                    "image": image_tag,
                    "containerfile": container_path,
                    "available": False,  # default; may be overwritten below
                })

            # Parallel check — all _image_exists calls run concurrently.
            checks: list = []  # (index,) -> coroutine
            coros: list = []
            for idx, e in enumerate(entries):
                if e["runtime_detected"] and e["image"]:
                    _, rt = image_and_runner_for_backend(cm_data, e["backend_id"])
                    checks.append(idx)
                    coros.append(asyncio.to_thread(_image_exists, rt, e["image"]))
            if coros:
                for idx, available in zip(checks, await asyncio.gather(*coros)):
                    entries[idx]["available"] = available

            return entries

        @self._app.post("/admin/images/build")
        async def admin_build_image(body: dict):
            backend_id = body.get("backend")
            if not backend_id:
                raise HTTPException(status_code=400, detail="Missing 'backend' in request body")

            cm_data = self._config_data
            image_tag, runner_type = image_and_runner_for_backend(cm_data, backend_id)
            containerfile_path = containerfile_for_backend(backend_id)

            if _runtime_binary(runner_type) is None:
                return {"skipped": True,
                        "reason": f"runner={runner_type} but no '{runner_type}' binary found on PATH",
                        "image": image_tag,
                        "runtime": runner_type}

            if not containerfile_path:
                raise HTTPException(status_code=404,
                                    detail=f"No containerfile found for backend '{backend_id}'")

            # Resolve containerfile path relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            full_cf_path = os.path.join(project_root, containerfile_path)
            if not os.path.isfile(full_cf_path):
                raise HTTPException(status_code=404,
                                    detail=f"Containerfile not found: {containerfile_path}")

            result = await asyncio.to_thread(build_image, runner_type, image_tag, full_cf_path, project_root)
            return {
                "backend": backend_id,
                "image": image_tag,
                "runtime": runner_type,
                **result,
            }

        @self._app.delete("/admin/images/{image_tag}")
        async def admin_remove_image(image_tag: str):
            found_backend = self._backend_for_image(image_tag)

            if not found_backend:
                raise HTTPException(status_code=404,
                                    detail=f"Image '{image_tag}' not configured in any backend")

            _, runner_type = image_and_runner_for_backend(
                self._config_data, found_backend
            )
            if _runtime_binary(runner_type) is None:
                return {"skipped": True,
                        "reason": f"runner={runner_type} but no '{runner_type}' binary found on PATH",
                        "image": image_tag}

            result = await asyncio.to_thread(remove_image, runner_type, image_tag)
            return {
                "removed": result["removed"],
                "image": image_tag,
                "error": result.get("error"),
            }

    def _add_eject_route(self) -> None:
        @self._app.post("/admin/eject/{model}")
        async def admin_eject(model: str):
            try:
                result = await self.server._arkestra.eject(model)
                return JSONResponse(status_code=200, content=result)
            except ValueError as exc:
                detail = str(exc)
                # Not-in-config is 404; shared-cache conflict is 409
                status = 404 if "not in config" in detail else 409
                raise HTTPException(status_code=status, detail=detail)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Eject failed: {exc}")


# Type hints — resolved at runtime via string ref
from model_arkestra.server import ArkestraServer  # noqa: E402, F401 (imported for type hint only in __init__)
