"""ArkestraAdmin — administrative routes for ModelArkestra.

This module defines ArkestraAdmin, a subcomponent of ArkestraServer that
installs admin endpoints (GET /, GET/POST /admin/*) onto the same FastAPI app.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, HTMLResponse
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
        self._add_config_routes()
        self._add_stop_route()
        self._add_restart_route()
        self._add_shutdown_route()
        self._add_start_route()
        self._add_eject_route()
        self._add_log_route()
        self._add_images_route()
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

        html = Path(__file__).parent.parent.parent / "static" / "index.html"

        @self._app.get("/")
        async def root():
            key = self.admin_key or ""
            content = html.read_text().replace("{{ADMIN_KEY}}", key)
            return HTMLResponse(content, media_type="text/html",
                                headers={"Cache-Control": "no-store"})

        @self._app.get("/index.html")
        async def index_html():
            key = self.admin_key or ""
            content = html.read_text().replace("{{ADMIN_KEY}}", key)
            return HTMLResponse(content, media_type="text/html",
                                headers={"Cache-Control": "no-store"})

    def _add_auth_middleware(self) -> None:
        # Always install the middleware (plumbing stays in place even when no key is set).
        # When admin_key is empty/None, it's a no-op pass-through.
        @self._app.middleware("http")
        async def admin_auth(request: Request, call_next):
            if request.url.path.startswith("/admin") and self.admin_key:
                key = request.headers.get("x-admin-key", "")
                if key != self.admin_key:
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Invalid or missing admin_key header"},
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
                            "args": model_cfg.get("args", ""),
                            "checkpoint": model_cfg.get("checkpoint", ""),
                            "capabilities": model_cfg.get("capabilities", []),
                        }
                    else:
                        checkpoint = model_cfg.get("checkpoint", "")
                        # Strip revision tag (e.g. :Q4_K_M) — HF Hub directories
                        # store models under the base name only.
                        base_checkpoint = checkpoint.split(":")[0] if ":" in checkpoint else checkpoint
                        hf_cache = self._resolve_env("HF_HUB_CACHE") or self._resolve_env("LLAMA_CACHE") or "~/.cache/huggingface/hub"
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
                backends = self.server._arkestra.cm.data.get("backends") or {}
                runner_types = list(self.server._arkestra._runner_classes.keys()) if hasattr(self.server._arkestra, "_runner_classes") else []

                return {
                    "models": data,
                    "backends": backends,
                    "runner_types": runner_types,
                }
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

    def _add_restart_route(self) -> None:
        from model_arkestra.types import RunnerState
        from fastapi.responses import JSONResponse
        import asyncio

        @self._app.post("/admin/restart")
        async def admin_restart():
            ctxs = list(self.server._arkestra._get_model_contexts())
            running = [c.name for c in ctxs if c.state not in (RunnerState.STOPPED, RunnerState.STOPPING)]
            if not running:
                return JSONResponse(
                    status_code=200,
                    content={"ok": True, "message": "No models running — nothing to restart", "stopped": []},
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
        from fastapi.responses import JSONResponse
        import asyncio
        import os

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
                    os._exit(0)

            task = asyncio.create_task(do_shutdown())
            if hasattr(self._app, "_shutdown_task"):
                del self._app._shutdown_task
            self._app._shutdown_task = task  # type: ignore
            return JSONResponse(status_code=200, content=result)

    def _add_config_routes(self) -> None:
        import copy

        KNOWN_KEYS = {"args", "checkpoint", "backend", "capabilities", "runner", "tags"}

        @self._app.get("/admin/config")
        async def admin_config_list():
            """Return list of model names in config."""
            cfg = self.server._arkestra.cm.data.get("models") or {}
            return {"models": list(cfg.keys())}

        @self._app.post("/admin/config")
        async def admin_config_create(body: Dict[str, Any]):
            """Create a new model entry in config. Requires at least 'checkpoint'."""
            cfg = self.server._arkestra.cm.data.get("models") or {}

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
            for key in ("args", "backend", "capabilities", "runner", "tags"):
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
            cfg = self.server._arkestra.cm.data.get("models") or {}
            if model not in cfg:
                raise HTTPException(
                    status_code=404, detail=f"Model '{model}' not in config"
                )
            # Also resolve current runtime status if the model is loaded
            status = None
            for ctx in self.server._arkestra._get_model_contexts():
                if ctx.name == model:
                    status = str(ctx.state).lower().replace("runnerstate.", "")
                    break

            return {"ok": True, "model": model, "config": copy.deepcopy(cfg[model]), "status": status}

        @self._app.put("/admin/config/{model}")
        async def admin_config_update(model: str, body: Dict[str, Any]):
            """Update an existing model's configuration."""
            cfg = self.server._arkestra.cm.data.get("models") or {}
            if model not in cfg:
                raise HTTPException(
                    status_code=404, detail=f"Model '{model}' not in config"
                )

            # Snapshot for rollback
            snapshot = copy.deepcopy(cfg[model])

            try:
                for key, value in body.items():
                    if key in KNOWN_KEYS or key == "max_log_lines":
                        cfg[model][key] = value
                self.server._arkestra.cm.export(self.server._arkestra.cm.config_path)
                return {"ok": True, "model": model}
            except Exception as exc:
                cfg[model].update(snapshot)
                raise HTTPException(status_code=500, detail=f"Save failed: {exc}")

    def _add_start_route(self) -> None:
        from model_arkestra.types import RunnerState
        from model_arkestra.base import BaseModelRunner

        @self._app.post("/admin/start/{model}")
        async def admin_start(model: str, body: Dict[str, Any] | None = None):
            cfg = self.server._arkestra.cm.data.get("models") or {}
            if model not in cfg:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not in config")

            # Build raw kwargs — infra keys handled by ModelArkestra, rest are inference params
            kw = {}
            for key in ("args", "checkpoint", "backend", "runner"):
                if body and key in body and body[key] is not None:
                    kw[key] = body[key]
            if body and "max_log_lines" in body and body["max_log_lines"] is not None:
                try:
                    kw["max_log_lines"] = int(body["max_log_lines"])
                except (ValueError, TypeError):
                    pass
            # Any other keys in body are inference params — pass through as-is
            if body:
                for key, value in body.items():
                    if key not in ("args", "checkpoint", "backend", "runner", "max_log_lines") and value is not None:
                        kw[key] = value

            # If model is already running with overrides, stop first then start fresh
            if kw:
                ctxs = [c for c in self.server._arkestra._get_model_contexts() if c.name == model]
                if ctxs and ctxs[0].state == RunnerState.RUNNING:
                    await self.server._arkestra.stop(model)

            try:
                await self.server._arkestra.start(model, **kw)
                ctxs = [c for c in self.server._arkestra._get_model_contexts() if c.name == model]
                port = ctxs[0].port if ctxs else None
                return {"ok": True, "model": model, "port": port}
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Start failed: {exc}")

    def _add_log_route(self) -> None:
        import asyncio
        import json

        @self._app.get("/admin/log/{model}")
        async def admin_log(
            model: str,
            lines: int = 100,
            follow: bool = False,
        ):
            # Validate model exists in config first
            cfg = self.server._arkestra.cm.data.get("models") or {}
            if model not in cfg:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not in config")

            if follow:
                from fastapi.responses import StreamingResponse

                async def log_stream():
                    """SSE stream of new log lines for a model."""
                    ctxs = [c for c in self.server._arkestra._get_model_contexts() if c.name == model]
                    if not ctxs:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'No context for this model'})}\n\n"
                        return
                    ctx = ctxs[0]
                    known_runner = None
                    buffer_size = ctx._log_buffer.maxlen if hasattr(ctx, '_log_buffer') else BaseModelRunner.LOG_BUFFER_DEFAULT

                    # Track position for new-line detection
                    last_count = len(ctx._log_buffer)

                    # Send current tail
                    buf = list(ctx._log_buffer)
                    yield f"data: {json.dumps({'type': 'snapshot', 'lines': buf[-lines:]})}\n\n"

                    while True:
                        await asyncio.sleep(0.1)
                        new_buf = list(ctx._log_buffer)
                        if len(new_buf) > last_count:
                            new_lines = new_buf[last_count:]
                            last_count = len(new_buf)
                            yield f"data: {json.dumps({'type': 'line', 'lines': new_lines})}\n\n"
                
                return StreamingResponse(
                    log_stream(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            else:
                # Snapshot mode
                lines_data = []
                for runner in self.server._arkestra._runners.values():
                    if hasattr(runner, 'get_logs'):
                        try:
                            result = await runner.get_logs(model, lines)
                            if result:
                                lines_data = result
                                break
                        except Exception:
                            pass
                return {"object": "log", "data": lines_data}

    def _add_images_route(self) -> None:
        import subprocess
        import shutil

        def _detect_runtime(runner_type: str) -> bool:
            """Return True if the container runtime for this runner type is on PATH."""
            binary = {"podman": "podman", "docker": "docker"}.get(runner_type)
            return binary is not None and shutil.which(binary) is not None

        def _image_exists(image_tag: str, runtime_type: str) -> bool:
            """Check if an image tag exists in the local container store."""
            cmd = {"podman": ["podman", "images", "--format", "{{.Repository}}:{{.Tag}}",
                              image_tag],
                   "docker": ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}",
                              image_tag]}
            proc = subprocess.run(cmd.get(runtime_type, []), capture_output=True, text=True,
                                  timeout=10)
            return any(line.strip() == image_tag for line in proc.stdout.splitlines())

        @self._app.get("/admin/images")
        async def admin_images():
            cm_data = self.server._arkestra.cm.data
            backends = cm_data.get("backends") or {}
            if not isinstance(backends, dict):
                raise HTTPException(status_code=500,
                                    detail="No backends section in config")

            results = []
            for backend_id, be_cfg in backends.items():
                # Skip 'default' — it's a key, not a backend definition
                if backend_id == "default" or not isinstance(be_cfg, dict):
                    continue
                image_tag = str(be_cfg.get("image", ""))
                container_path = be_cfg.get("container", "")
                from model_arkestra.common import image_and_runner_for_backend
                _, runner_type = image_and_runner_for_backend(cm_data, backend_id)
                runtime_detected = _detect_runtime(runner_type)
                available = _image_exists(image_tag, runner_type) if runtime_detected and image_tag else False
                results.append({
                    "backend_id": backend_id,
                    "runner": runner_type,
                    "runtime_detected": runtime_detected,
                    "image": image_tag,
                    "containerfile": container_path,
                    "available": available,
                })
            return results

        @self._app.post("/admin/images/build")
        async def admin_build_image(body: dict):
            backend_id = body.get("backend")
            if not backend_id:
                raise HTTPException(status_code=400, detail="Missing 'backend' in request body")

            cm_data = self.server._arkestra.cm.data
            from model_arkestra.common import image_and_runner_for_backend, containerfile_for_backend
            image_tag, runner_type = image_and_runner_for_backend(cm_data, backend_id)
            containerfile_path = containerfile_for_backend(backend_id)

            runtime_detected = _detect_runtime(runner_type)
            if not runtime_detected:
                return {"skipped": True, "reason": f"runner={runner_type} but no '{runner_type}' binary found on PATH", "image": image_tag}

            if not containerfile_path:
                raise HTTPException(status_code=404, detail=f"No containerfile found for backend '{backend_id}'")

            # Resolve containerfile path relative to project root
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            full_cf_path = os.path.join(project_root, containerfile_path)
            if not os.path.isfile(full_cf_path):
                raise HTTPException(status_code=404,
                                    detail=f"Containerfile not found: {containerfile_path}")

            cmd = {"podman": ["podman", "build", "-t", image_tag, "-f", full_cf_path, project_root],
                   "docker": ["docker", "build", "-t", image_tag, "-f", full_cf_path, project_root]}
            proc = subprocess.run(cmd[runner_type], capture_output=True, text=True, timeout=600)
            success = proc.returncode == 0
            return {
                "backend": backend_id,
                "image": image_tag,
                "success": success,
                "runtime": runner_type,
                "output": proc.stdout + proc.stderr,
                "error": None if success else proc.stderr or proc.stdout or "non-zero exit",
            }

        @self._app.delete("/admin/images/{image_tag}")
        async def admin_remove_image(image_tag: str):
            cm_data = self.server._arkestra.cm.data
            backends = cm_data.get("backends") or {}

            # Find which backend_id this image belongs to
            found_backend = None
            for bid, be_cfg in backends.items():
                if isinstance(be_cfg, dict) and str(be_cfg.get("image", "")) == image_tag:
                    found_backend = bid
                    break

            if not found_backend:
                raise HTTPException(status_code=404,
                                    detail=f"Image '{image_tag}' not configured in any backend")

            from model_arkestra.common import image_and_runner_for_backend
            _, runner_type = image_and_runner_for_backend(cm_data, found_backend)
            if not _detect_runtime(runner_type):
                return {"skipped": True, "reason": f"runner={runner_type} but no '{runner_type}' binary found on PATH", "image": image_tag}

            cmd = {"podman": ["podman", "rmi", "-f", image_tag],
                   "docker": ["docker", "rmi", "-f", image_tag]}
            proc = subprocess.run(cmd[runner_type], capture_output=True, text=True, timeout=30)
            return {
                "removed": proc.returncode == 0,
                "image": image_tag,
                "error": None if proc.returncode == 0 else proc.stderr,
            }

    def _add_eject_route(self) -> None:
        from fastapi.responses import JSONResponse

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
