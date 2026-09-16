"""Registry: name → Model, cluster resolution, and the port pool.

The single place that knows how a model *name* maps to a Model object and
which cluster it belongs to. ``<cluster>/<model-id>`` parsing lives here and
only here — every other call site does one ``resolve()`` lookup.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class Registry:
    def __init__(self, cm: Any, local_url: str = ""):
        self._cm = cm
        self._models: Dict[str, _Model] = {}   # type: ignore[name-defined]
        self._local_cluster_key: str = cm.get("default/local-cluster-key", "local")

        # ── port pool ───────────────────────────────────────────
        self._start_port = int(cm.get("default/model-start-port", 18000))
        self._pool_size = int(cm.get("default/model-ports", 32))
        self._next_port = self._start_port

        # ── cluster topology ────────────────────────────────────
        self._clusters: Dict[str, Dict[str, Any]] = {}
        self._load_clusters(local_url)

    # ── model storage ───────────────────────────────────────────
    def register(self, model: Any, aliases: Optional[List[str]] = None) -> None:
        """Register a context under its own name plus any alias names.

        A context may be shared by several model names (models that reference
        the same checkpoint). Each alias resolves to the same context object.
        ``aliases`` should include at least one *model* name (not the checkpoint
        id) so config/capability lookup works against the ``models:`` section.
        """
        self._attach_config(model, aliases)
        self._models[model.name] = model
        for alias in (aliases or []):
            if alias != model.name:
                self._models[alias] = model

    def _attach_config(self, model: Any, aliases: Optional[List[str]] = None) -> None:
        """Best-effort fill of ``_model_cfg`` / ``_backend_cfg`` from the registry's cm.

        Uses the first alias (a real model name) for config lookup, since the
        context itself may be named by checkpoint id rather than a model key.
        """
        try:
            lookup_name = aliases[0] if aliases else model.name
            cfg = self._cm.get_model(lookup_name) or {}
            model._model_cfg = dict(cfg)
            be_id = cfg.get("backend")
            if be_id:
                model._backend_cfg = self._cm.get_backend(be_id) or {}
        except Exception:
            # Capability derivation degrades to defaults; never block registration.
            pass

    def get(self, name: str) -> Optional[Any]:
        return self._models.get(name)

    @property
    def all(self) -> List[Any]:
        return list(self._models.values())

    def find_by_local_name(self, local_name: str) -> Optional[Any]:
        return self._models.get(local_name)

    # ── port pool ───────────────────────────────────────────────
    def allocate_port(self, model_name: str) -> int:
        """Reuse a stopped model's port, else take the next from the pool."""
        m = self._models.get(model_name)
        if m is not None and m.port is not None:
            return m.port
        end_port = self._start_port + self._pool_size - 1
        if self._next_port > end_port:
            raise RuntimeError(f"Port range exceeded: {self._start_port}–{end_port}")
        port = self._next_port
        self._next_port += 1
        return port

    def reset_ports(self) -> None:
        self._next_port = self._start_port

    # ── cluster topology ────────────────────────────────────────
    def _load_clusters(self, local_url: str) -> None:
        if not local_url:
            local_url = self._cm.get("default/url", "http://127.0.0.1:8080")
        self._clusters[self._local_cluster_key] = {
            "url": local_url.rstrip("/"),
            "admin-key": self._cm.get("env/ADMIN_KEY"),
        }
        raw = self._cm.get("clusters", {})
        if not isinstance(raw, dict):
            return
        for name, cfg in raw.items():
            if not isinstance(cfg, dict) or not cfg.get("url"):
                continue
            cfg = dict(cfg)
            cfg["url"] = str(cfg["url"]).rstrip("/")
            self._clusters[name] = cfg

    @property
    def local_cluster_key(self) -> str:
        return self._local_cluster_key

    @property
    def clusters(self) -> Dict[str, Dict[str, Any]]:
        return self._clusters

    def reload_clusters(self) -> None:
        """Re-read cluster config from cm (after add/remove)."""
        local_url = self._cm.get("default/url", "http://127.0.0.1:8080")
        self._clusters.clear()
        self._load_clusters(local_url)

    def parse_prefix(self, model_name: str) -> Tuple[str, str]:
        if "/" in model_name:
            return model_name.split("/", 1)
        return self._local_cluster_key, model_name

    def resolve(self, model_name: str) -> Tuple[str, Optional[str], str]:
        """Return ``(cluster_name, base_url|None, local_id)``.

        Local cluster → base_url None (direct runner / port pool).
        Remote cluster → the worker URL to proxy through.
        """
        cluster_name, local_id = self.parse_prefix(model_name)
        cfg = self._clusters.get(cluster_name)
        if cfg is None:
            # Legacy fallback: backend with runner=remote + base_url
            be = (self._cm.get("backends", {}) or {}).get(cluster_name, {})
            if isinstance(be, dict) and be.get("runner") == "remote" and be.get("base_url"):
                return cluster_name, str(be["base_url"]).rstrip("/"), local_id
            raise ValueError(
                f"Unknown cluster '{cluster_name}' for model '{model_name}'. "
                f"Declare it in the 'clusters:' top-level key.")
        base_url = cfg.get("url") if cluster_name != self._local_cluster_key else None
        return cluster_name, base_url, local_id

    def local_model_name(self, model_name: str) -> str:
        try:
            return self.resolve(model_name)[2]
        except ValueError:
            return model_name
