"""ArkestraAdmin — administrative routes for ModelArkestra.

This module defines ArkestraAdmin, a subcomponent of ArkestraServer that
installs admin endpoints (GET /, GET/POST /admin/*) onto the same FastAPI app.
"""
from __future__ import annotations

import asyncio
import aiohttp
import copy
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, HTMLResponse, Response
except ImportError:
    raise RuntimeError("model_arkestra.admin requires fastapi")

from model_arkestra.common import (
    _resolve_backend,
    _runtime_binary,
    build_image,
    containerfile_for_backend,
    default_cache_root,
    image_and_runner_for_backend,
    image_exists as _image_exists,
    remove_image,
    resolve_model_ref,
    resolve_tags as _resolve_tags,
)
from model_arkestra.http_proxy import model_status_for_ctx
from model_arkestra.types import RunnerState, _ModelContext

# ── Model config field definitions (single source of truth) ─────────────
MODEL_CONFIG_FIELDS = frozenset({"backend", "runner", "tags", "max_log_lines", "args"})
INFRA_KEYS = frozenset({"backend", "runner", "max_log_lines"})



class ArkestraAdmin:
    """Admin subcomponent that installs routes on an ArkestraServer's app."""

    def __init__(self, server: "ArkestraServer", admin_key: Optional[str], app: FastAPI):
        self.server = server
        self.admin_key = admin_key
        self._app = app
        self._installed = False
        # Load schema registry from schemas.yaml (same dir as config)
        self._schemas: Dict[str, Any] = {}
        self._load_schema_registry()

    def _load_schema_registry(self) -> None:
        """Load named schemas from schemas.yaml in the config directory.

        Falls back to bundled templates/schemas.yaml.j2 if none found.
        """
        import yaml
        try:
            parent = Path(self.server._arkestra._config_path).parent
            schema_path = parent / "schemas.yaml"
        except (AttributeError, TypeError):
            self._schemas = {}
            return

        if schema_path.exists():
            with open(schema_path) as f:
                self._schemas = yaml.safe_load(f) or {}
        else:
            # Bundled fallback: ship a sensible default in the package
            try:
                from importlib.resources import files
                bundled = (files("model_arkestra.templates") / "schemas.yaml.j2").read_text()
                self._schemas = yaml.safe_load(bundled) or {}
            except Exception:
                self._schemas = {}

    def install(self) -> "ArkestraAdmin":
        """Install all admin routes on the FastAPI app. Idempotent."""
        if self._installed:
            return self
        self._add_root_route()
        self._add_static_route()
        self._add_auth_middleware()
        self._add_clusters_route()
        self._add_models_route()
        self._add_config_routes()
        self._add_stop_route()
        self._add_stop_all_route()
        self._add_shutdown_route()
        self._add_start_route()
        self._add_restart_route()
        self._add_eject_route()
        self._add_log_route()
        self._add_global_log_route()
        self._add_images_route()
        self._add_download_route()
        self._add_download_stop_route()
        self._installed = True
        return self

    # ── helpers ────────────────────────────────────────────────────────

    @property
    def _config_data(self) -> dict:
        return self.server._arkestra.cm.data

    @property
    def _models_cfg(self) -> Dict[str, Any]:
        return self.server._arkestra.cm.data.get("models") or {}

    def _backend_for_image(self, image_tag: str) -> Optional[str]:
        """Return the backend_id whose ``image`` matches *image_tag*, or None."""
        for bid, be_cfg in (self.server._arkestra.cm.data.get("backends") or {}).items():
            if isinstance(be_cfg, dict) and str(be_cfg.get("image", "")) == image_tag:
                return bid
        return None

    def _args_schema(self, model_name: str) -> Dict[str, Any]:
        """Return arg schema for a model from the schema registry.

        Resolution:
          1. Explicit ``args_schema`` in model config (override)
          2. Backend.engine → schemas["model-args"][engine_name]
          3. engines.default-engine (from backends.yaml) fallback

        Returns the full schema — backend and model configs merge their
        values into these keys via `config.args`.
        """
        cfg = self._models_cfg.get(model_name, {})
        explicit = cfg.get("args_schema")
        if explicit:
            return explicit

        # Resolve engine name for this model's backend
        bid = _resolve_backend(self.server._arkestra.cm, cfg, model_name)
        cm_data = self.server._arkestra.cm.data
        bcfg = (cm_data.get("backends") or {}).get(bid, {})
        engine_name = bcfg.get("engine") if isinstance(bcfg, dict) else None
        if not engine_name:
            engine_name = (cm_data.get("default") or {}).get("engine")
        if not engine_name:
            engine_name = cm_data.get("engines", {}).get("default-engine")
        if not engine_name:
            return {}

        # Look up in schema registry
        model_args_schema = self._schemas.get("model-args", {})
        schema = model_args_schema.get(engine_name, {}) or {}

        # Always expose model/repo/mmproj for CLI builder integration.
        schema.setdefault("name", {"type": "string"})
        schema.setdefault("repo", {"type": "string", "options": ["hf", "lcl"]})
        schema.setdefault("model", {"type": "string"})
        schema.setdefault("mmproj", {"type": "string"})
        return schema

    def _resolve_model_backend(self, model_name: str, model_cfg: dict) -> str:
        """Resolve backend for a model using the normal chain."""
        return _resolve_backend(self.server._arkestra.cm, model_cfg, model_name)

    def _add_clusters_route(self) -> None:
        @self._app.get("/admin/clusters")
        async def admin_clusters():
            """Return managed cluster list with connectivity status."""
            clusters = self.server._arkestra._clusters
            result = []
            for name, cfg in clusters.items():
                base_url = str(cfg.get("base-url", ""))
                # Ping the health endpoint to check reachability
                healthy = False
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{base_url}/health", timeout=3) as resp:
                            healthy = (resp.status == 200)
                except Exception:
                    pass
                result.append({
                    "name": name,
                    "base-url": base_url,
                    "healthy": healthy,
                })
            return JSONResponse(status_code=200, content={"clusters": result})

    def _add_root_route(self) -> None:
        html = Path(__file__).parent.parent.parent / "static" / "index.html"
        content = html.read_text().replace("{{ADMIN_KEY}}", self.admin_key or "")

        @self._app.get("/")
        @self._app.get("/index.html")
        async def root():
            return HTMLResponse(content, media_type="text/html",
                                headers={"Cache-Control": "no-store"})

    def _add_static_route(self) -> None:
        static_dir = Path(__file__).parent.parent.parent / "static"

        @self._app.get("/static/{path:path}")
        async def serve_static(path: str):
            file_path = static_dir / path
            if not file_path.is_file():
                raise HTTPException(404, "Not found")
            mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            return Response(content=file_path.read_bytes(), media_type=mime,
                            headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

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

                # Resolve cache root — env var → config env → default.
                hf_cache = self.server._arkestra.resolve_config("HF_HUB_CACHE")
                if not hf_cache:
                    hf_cache = str(default_cache_root())

                for model_name in self.server._arkestra.get_models():
                    ctx = contexts_by_name.get(model_name)
                    model_cfg = self.server._arkestra.get_model(model_name) or {}

                    if ctx:
                        webui_status = model_status_for_ctx(ctx)
                        model_ref = model_cfg.get("model", "")
                        entry = {
                            "id": ctx.name,
                            "status": webui_status,
                            "port": ctx.port,
                            "runner_type": ctx.runner_type,
                            "backend_id": ctx.backend_id or self._resolve_model_backend(ctx.name, model_cfg),
                            "args": {},
                            "model": model_ref,
                            "downloading": ctx.state == RunnerState.DOWNLOADING,
                        }
                    else:
                        default_section = (self.server._arkestra.cm.data.get("default") or {})
                        raw_model = model_cfg.get("model", "")
                        resolved = resolve_model_ref(
                            raw=raw_model,
                            default_section=default_section,
                            model_repos=self.server._arkestra.cm.data.get("model-repos"),
                        )
                        cache_path = None
                        if resolved.cache_path:
                            cache_path = Path(hf_cache).expanduser() / f"models--{resolved.cache_path}"
                        is_cached = cache_path.exists() if cache_path else False
                        resolved_backend = self._resolve_model_backend(model_name, model_cfg)
                        _, runner_type = image_and_runner_for_backend(self.server._arkestra.cm.data, resolved_backend)
                        entry = {
                            "id": model_name,
                            "status": {"value": "cached"} if is_cached else {"value": "uncached"},
                            "port": None,
                            "runner_type": runner_type,
                            "backend_id": resolved_backend,
                            "args": {},
                            "model": model_cfg.get("model", ""),
                            "downloading": False,
                        }

                    # Resolve available capabilities per-model (normal chain)
                    global_cfg = self.server._arkestra.cm.data or {}
                    bcfg = (global_cfg.get("backends") or {}).get(str(entry.get("backend_id") or ""))
                    entry["tags"] = _resolve_tags(
                        model_cfg, global_cfg,
                        backend_id=str(entry.get("backend_id") or "") or None,
                    )
                    data.append(entry)

                # Top-level metadata for dropdown options (static per-server)
                backends = self.server._arkestra.cm.data.get("backends") or {}
                runner_types = list(self.server._arkestra._RUNNER_CLASSES.keys())

                return {
                    "models": data,
                    "backends": backends,
                    "runner_types": runner_types,
                }
            except Exception as e:
                raise HTTPException(status_code=503, detail=str(e))

    def _add_stop_route(self) -> None:
        @self._app.post("/admin/stop/{model:path}")
        async def admin_stop(model: str):
            if model not in self._models_cfg:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not configured")
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
                    try:
                        await asyncio.wait_for(
                            self.server._server.shutdown(),
                            timeout=10.0,
                        )
                        print("[SHUTDOWN] uvicorn stopped.", flush=True)
                    except (asyncio.TimeoutError, Exception):
                        print("[SHUTDOWN] uvicorn shutdown timed out/errored, forcing exit.", flush=True)
                        os._exit(1)
                else:
                    os._exit(0)

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
            """Create a new model entry in config."""
            cfg = self._models_cfg

            if not body.get("model"):
                raise HTTPException(
                    status_code=400, detail="Need 'model' to create a model"
                )

            model_str = str(body["model"])

            name = body.get("name") or model_str.split(":")[0].rsplit("/", 1)[-1]
            if not name:
                raise HTTPException(
                    status_code=400, detail="Could not determine model name from model field"
                )

            if name in cfg:
                raise HTTPException(
                    status_code=409, detail=f"Model '{name}' already exists in config"
                )

            # Build new model entry from body fields (with safe defaults)
            new_model: Dict[str, Any] = {}
            if model_str:
                new_model["model"] = model_str
            if "repo" in body and body["repo"] is not None:
                new_model["repo"] = body["repo"]
            for key in MODEL_CONFIG_FIELDS:
                if key in body and body[key] is not None:
                    new_model[key] = body[key]

            cfg[name] = new_model
            self.server._arkestra.cm.export(self.server._arkestra.cm.config_path)
            return JSONResponse(
                status_code=201,
                content={"ok": True, "model": name},
            )

        @self._app.get("/admin/config/{model:path}")
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
            status = model_status_for_ctx(ctx)

            # Resolve available capabilities for this model
            global_cfg = self.server._arkestra.cm.data or {}
            bid = cfg[model].get("backend")
            bcfg = (global_cfg.get("backends") or {}).get(str(bid) if bid else "")
            available_caps = _resolve_tags(
                cfg[model], global_cfg,
                backend_id=str(bid) if bid else None,
            )

            return {
                "ok": True, "model": model,
                "config": copy.deepcopy(cfg[model]),
                "args_schema": self._args_schema(model),
                "status": status,
                "tags": available_caps,
                "backends": global_cfg.get("backends") or {},
                "runner_types": list(self.server._arkestra._RUNNER_CLASSES.keys()),
                "default": global_cfg.get("default") or {},
            }

        @self._app.put("/admin/config/{model:path}")
        async def admin_config_update(model: str, body: Dict[str, Any]):
            """Update an existing model's configuration."""
            cfg = self._models_cfg
            if model not in cfg:
                raise HTTPException(
                    status_code=404, detail=f"Model '{model}' not in config"
                )

            # Check for name conflicts (another model already uses this name)
            new_name = body.get("name")
            if isinstance(new_name, str) and new_name:
                for other_id, other_cfg in cfg.items():
                    if other_id == model: continue
                    if str(other_cfg.get("name", "")) == new_name:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Name '{new_name}' already used by model '{other_id}'"
                        )

            # Snapshot for rollback
            snapshot = copy.deepcopy(cfg[model])

            try:
                arg_schema = self._args_schema(model) or {}
                for key, value in body.items():
                    if key in MODEL_CONFIG_FIELDS:
                        cfg[model][key] = value
                    elif key in arg_schema:
                        # Coerce to schema type so YAML stays typed (int/float not string)
                        s = arg_schema[key]
                        t = s.get("type", "string")
                        if t == "integer":
                            try: cfg[model][key] = int(value)
                            except (ValueError, TypeError): pass
                        elif t == "float":
                            try: cfg[model][key] = float(value)
                            except (ValueError, TypeError): pass
                        else:
                            cfg[model][key] = value
                self.server._arkestra.cm.export(self.server._arkestra.cm.config_path)
                return {"ok": True, "model": model}
            except Exception as exc:
                cfg[model].update(snapshot)
                raise HTTPException(status_code=500, detail=f"Save failed: {exc}")

    def _add_start_route(self) -> None:
        @self._app.post("/admin/start/{model:path}")
        async def admin_start(model: str, body: Dict[str, Any] | None = None):
            cfg = self._models_cfg
            if model not in cfg:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not in config")

            if not self.server._arkestra.can_start(model):
                raise HTTPException(status_code=409, detail="model not available")

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

            try:
                await self.server._arkestra.start(model, **kw)
                ctx = self.server._arkestra.find_context(model)
                port = ctx.port if ctx else None
                return {"ok": True, "model": model, "port": port}
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Start failed: {exc}")

    def _add_restart_route(self) -> None:
        @self._app.post("/admin/restart/{model:path}")
        async def admin_restart(model: str, body: Dict[str, Any] | None = None):
            cfg = self._models_cfg
            if model not in cfg:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not in config")

            if not self.server._arkestra.can_restart(model):
                raise HTTPException(status_code=409, detail="model not available")

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

            # Stop current instance if running/loading, then start fresh
            ctx = self.server._arkestra.find_context(model)
            if ctx and ctx.state in (RunnerState.RUNNING, RunnerState.LOADING):
                try:
                    await self.server._arkestra.stop(model)
                except Exception:
                    pass

            try:
                await self.server._arkestra.start(model, **kw)
                ctx = self.server._arkestra.find_context(model)
                port = ctx.port if ctx else None
                return {"ok": True, "model": model, "port": port}
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Restart failed: {exc}")

    def _add_log_route(self) -> None:
        @self._app.get("/admin/log/{model:path}")
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
            if not ctx or not hasattr(ctx, '_get_lines_since'):
                # Model not yet started — return empty result
                return JSONResponse(
                    status_code=200,
                    content={"since": 0, "missed_lines": 0, "lines": []},
                    headers={"X-Missed-Lines": "0", "X-Current-Max": "0"},
                )

            # Remote models — logs are on the worker server
            if getattr(ctx, '_remote_base_url', None):
                return JSONResponse(
                    status_code=200,
                    content={"since": 0, "missed_lines": 0, "lines": []},
                    headers={"X-Missed-Lines": "0", "X-Current-Max": "0", "X-Note": "remote-worker"},
                )

            # Read delta from ring buffer
            new_lines, oldest_seq = ctx._get_lines_since(since, lines)

            # Calculate missed lines (entries pruned from ring before the requested since point)
            total_missed = max(0, since - oldest_seq) if since > 0 else 0

            return JSONResponse(
                status_code=200,
                content={"since": ctx._log_seq, "missed_lines": total_missed, "lines": [{"seq": seq, "text": text} for seq, text in new_lines]},
                headers={"X-Missed-Lines": str(total_missed), "X-Current-Max": str(ctx._log_seq)},
            )

    def _add_global_log_route(self) -> None:
        @self._app.get("/admin/logs")
        async def admin_global_logs(
            since: int = 0,
            lines: int = 200,
        ):
            buf = self.server._arkestra._global_log_buf
            seq_max = self.server._arkestra._global_log_seq

            entries = buf.read_entries(max_lines=lines, next_line=since)

            # Calculate missed lines (entries evicted before requested since point)
            oldest = entries[0][0] if entries else 0
            total_missed = max(0, since - oldest) if since > 0 and oldest < since else 0

            return JSONResponse(
                status_code=200,
                content={"seq": seq_max, "missed_lines": total_missed,
                         "lines": [{"seq": s, "text": t} for s, t in entries]},
                headers={"X-Missed-Lines": str(total_missed), "X-Current-Max": str(seq_max)},
            )

    def _add_images_route(self) -> None:
        @self._app.get("/admin/images")
        async def admin_images():
            cm_data = self._config_data
            backends = self.server._arkestra.cm.data.get("backends") or {}
            if not isinstance(backends, dict):
                raise HTTPException(status_code=500,
                                    detail="No backends section in config")

            # First pass: build metadata list with defaults.
            entries: list[dict] = []
            for idx, (backend_id, be_cfg) in enumerate(backends.items()):
                if backend_id == "default" or not isinstance(be_cfg, dict):
                    continue
                image_tag = str(be_cfg.get("image", ""))
                source_ref = be_cfg.get("source_ref", "")
                sources = cm_data.get("sources", {}) or {}
                src_cfg = sources.get(source_ref) if source_ref else {}
                # For OCI-image sources, show the full image reference
                if not image_tag and src_cfg and src_cfg.get("type") == "oci-image":
                    repo = src_cfg.get("repo", "")
                    release = src_cfg.get("release_type", "")
                    image_tag = f"{repo}:{release}" if repo else source_ref
                container_path = be_cfg.get("container", "")
                _, runner_type = image_and_runner_for_backend(cm_data, backend_id)
                runtime_detected = _runtime_binary(runner_type) is not None
                entries.append({
                    "backend_id": backend_id,
                    "runner": runner_type,
                    "runtime_detected": runtime_detected,
                    "image": image_tag if image_tag else None,
                    "source_ref": source_ref if source_ref else None,
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
        @self._app.post("/admin/eject/{model:path}")
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

    def _add_download_route(self) -> None:
        @self._app.post("/admin/download/{model:path}")
        async def admin_download(model: str):
            """Start downloading a model's checkpoint from HuggingFace."""
            cfg = self._models_cfg
            if model not in cfg:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not in config")

            ctx = self.server._arkestra.find_context(model)

            # If already downloading, return existing task
            if ctx and ctx.state == RunnerState.DOWNLOADING and ctx.download_task:
                return {"ok": True, "model": model, "already_downloading": True}

            # If model is running or stopping, reject
            if ctx and ctx.state in (RunnerState.RUNNING, RunnerState.STOPPING):
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot download: model is {ctx.state.name.lower()}"
                )

            # If already uncached, nothing to download
            if ctx and ctx.state == RunnerState.UNCACHED:
                raise HTTPException(
                    status_code=409,
                    detail=f"Model '{model}' is already uncached (checkpoint present)"
                )

            # Create context if it doesn't exist yet
            if not ctx:
                model_cfg = cfg.get(model, {})
                be_id = self._resolve_model_backend(model, model_cfg)
                cm_data = self.server._arkestra.cm.data
                _, runner_type = image_and_runner_for_backend(cm_data, be_id)
                runner = self.server._arkestra._get_runner_instance(runner_type, model)
                ctx = _ModelContext(model, 0, max_log_lines=2000)
                ctx.backend_id = be_id
                ctx.state = RunnerState.DOWNLOADING
                runner._models[model] = ctx

            # Spawn download task
            task = asyncio.create_task(
                self.server._arkestra._download_model(ctx)
            )
            ctx.download_task = task
            return {"ok": True, "model": model}

    def _add_download_stop_route(self) -> None:
        @self._app.post("/admin/download/stop/{model:path}")
        async def admin_download_stop(model: str):
            """Cancel an in-progress model download."""
            cfg = self._models_cfg
            if model not in cfg:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not in config")

            ctx = self.server._arkestra.find_context(model)
            if not ctx or ctx.state != RunnerState.DOWNLOADING:
                raise HTTPException(
                    status_code=404,
                    detail=f"No active download for '{model}'"
                )

            task = ctx.download_task
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            ctx.download_task = None
            return {"ok": True, "model": model}


# Type hints — resolved at runtime via string ref
from model_arkestra.server import ArkestraServer  # noqa: E402, F401 (imported for type hint only in __init__)
