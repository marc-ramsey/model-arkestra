"""Plugin loader for arkestra-server.

A plugin is a Python module listed under the top-level ``plugins:`` key of
config.yaml. Each entry is either a path to a ``.py`` file or a directory
containing a single ``plugin.py``::

    plugins:
      - ~/arkestra-plugins/stats/plugin.py   # name "stats"
      - /srv/arkestra/plugins/uptime         # dir, loads uptime/plugin.py

The host imports each module and then:

1. Mounts it at ``/api/plugins/<name>/...`` if it exposes a FastAPI
   ``router`` (an APIRouter). The fixed prefix gives free api_key gating via
   the existing middleware and makes route collisions with core paths
   impossible.
2. Calls its optional sync ``register(ctx)`` hook once — for non-route setup.

Plugins are trusted code but get a *narrow* surface: they never receive the
raw FastAPI app, so they cannot add top-level routes or middleware that would
bypass auth gating. Everything goes through :class:`PluginContext`.

Failure policy: warn + skip. A broken plugin must never take down model
serving; its error is recorded in the result list served by GET /admin/plugins.
"""
from __future__ import annotations

import copy
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, FastAPI
except ImportError:
    raise RuntimeError("model_arkestra.plugins requires fastapi")

log = logging.getLogger(__name__)

PLUGIN_API_PREFIX = "/api/plugins"


class PluginContext:
    """Narrow surface handed to a plugin's ``register(ctx)`` hook.

    Deliberately not the FastAPI app — plugins cannot attach ungated routes
    or middleware through it. Extend here when a new need appears; keep each
    member read-only from the plugin side where possible.
    """

    def __init__(self, server: Any):
        self._server = server

    @property
    def models(self) -> Dict[str, str]:
        """Model name → state string (e.g. ``"running"``, ``"stopped"``)."""
        return {name: ctx.state.value for name, ctx in self._arkestra().models.items()}

    @property
    def config(self) -> Any:
        """Full merged config data (deep copy — mutations stay local)."""
        return copy.deepcopy(self._server._arkestra.cm.data)

    def _arkestra(self):
        return self._server._arkestra


def _resolve_py_path(base: Path) -> Optional[Path]:
    """Config entry → the .py file to load (None when unresolvable)."""
    if base.is_file():
        return base
    candidate = base / "plugin.py"
    if base.is_dir() and candidate.is_file():
        return candidate
    return None


def _load_module(name: str, py_path: Path):
    """Import *py_path* as a top-level module (no sys.path pollution).

    Registered in ``sys.modules`` for the duration of exec so dataclasses /
    pydantic inside plugin code resolve their own types correctly. Removed on
    failure to avoid leaking half-initialised modules.
    """
    mod_name = f"arkestra_plugin_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create module spec for {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return mod


def _plugin_name(base: Path) -> str:
    """Dir → its basename; file → its stem."""
    return base.name if base.is_dir() else base.stem


def load_plugins(app: FastAPI, server: Any) -> List[Dict[str, Any]]:
    """Load every plugin listed in config onto *app*; return per-plugin state.

    Each result entry: ``{"name", "path", "state": "ok"|"error", "error"?}``.
    A failing plugin is skipped — remaining plugins still load and the server
    stays up (single-user LAN posture, no sandboxing by design).
    """
    entries = server._arkestra.cm.data.get("plugins") or []
    if not isinstance(entries, list):
        log.warning("config 'plugins' must be a list — ignoring %s",
                    type(entries).__name__)
        return []

    results: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}   # name → path that owns it (for dup errors)

    for entry in entries:
        base = Path(str(entry)).expanduser()
        name = _plugin_name(base)
        path_str = str(base)

        def fail(reason: str) -> None:
            log.warning("plugin '%s' not loaded: %s", name, reason)
            results.append({"name": name, "path": path_str,
                            "state": "error", "error": reason})

        if not name.isidentifier():
            fail(f"plugin name '{name}' is not a valid identifier")
            continue
        if name in seen:
            fail(f"duplicate plugin name (already loaded from {seen[name]})")
            continue

        py_path = _resolve_py_path(base)
        if py_path is None:
            fail("path is neither a .py file nor a directory with plugin.py")
            continue

        try:
            mod = _load_module(name, py_path)
        except Exception as exc:
            fail(f"import failed: {exc}")
            continue

        router = getattr(mod, "router", None)
        if router is not None and not isinstance(router, APIRouter):
            fail("'router' must be a FastAPI APIRouter")
            continue

        # register() first (may raise → plugin skipped entirely), then mount.
        try:
            ctx = PluginContext(server)
            register = getattr(mod, "register", None)
            if callable(register):
                register(ctx)
            if isinstance(router, APIRouter):
                app.include_router(
                    router, prefix=f"{PLUGIN_API_PREFIX}/{name}")
        except Exception as exc:
            fail(f"setup failed: {exc}")
            continue

        seen[name] = path_str
        results.append({"name": name, "path": path_str, "state": "ok"})

    return results
