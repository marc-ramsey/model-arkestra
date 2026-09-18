"""Registry: checkpoint contexts, model-name resolution, port pool, clusters.

Two dicts separate concerns:
  _checkpoints: checkpoint_id → _Model   (one context per shared weight set)
  _models:      model_name    → checkpoint_id  (resolution table)

A model name always resolves to exactly one checkpoint. A checkpoint may
serve multiple model names (aliases). The context is owned by the checkpoint;
model names are pure lookup keys.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class Registry:
    def __init__(self, cm: Any, local_url: str = ""):
        self._cm = cm
        self._checkpoints: Dict[str, Any] = {}   # checkpoint_id → _Model
        self._models: Dict[str, str] = {}        # model_name → checkpoint_id
        self._local_cluster_key: str = cm.get("default/local-cluster-key", "local")

        # ── port pool ───────────────────────────────────────────
        self._start_port = int(cm.get("default/model-start-port", 18000))
        self._pool_size = int(cm.get("default/model-ports", 32))
        self._next_port = self._start_port

        # ── cluster topology ────────────────────────────────────
        self._clusters: Dict[str, Dict[str, Any]] = {}
        self._load_clusters(local_url)

    # ── registration ────────────────────────────────────────────
    def register(self, checkpoint_id: str, ctx: Any, model_names: List[str]) -> None:
        """Register a checkpoint context and its model names.

        *checkpoint_id* is the canonical name for the weight set (e.g.
        "qwen3.6-35B"). *model_names* are the config keys that resolve to it
        (e.g. ["qwen3.6-35B-instruct", "qwen3.6-35B-think"]).
        """
        self._checkpoints[checkpoint_id] = ctx
        for name in model_names:
            self._models[name] = checkpoint_id

        # Attach config for capability derivation (use first model name).
        try:
            lookup_name = model_names[0] if model_names else checkpoint_id
            cfg = self._cm.get_model(lookup_name) or {}
            ctx._model_cfg = dict(cfg)
            be_id = cfg.get("backend")
            if be_id:
                ctx._backend_cfg = self._cm.get_backend(be_id) or {}
        except Exception:
            pass

    # ── lookup ──────────────────────────────────────────────────
    def get_checkpoint_id(self, model_name: str) -> Optional[str]:
        """Resolve a model name to its checkpoint id."""
        return self._models.get(model_name)

    def get_context(self, model_name: str) -> Optional[Any]:
        """Resolve a model name to its context (via checkpoint)."""
        ckpt_id = self._models.get(model_name)
        if ckpt_id is None:
            return None
        return self._checkpoints.get(ckpt_id)

    def get_context_by_checkpoint(self, checkpoint_id: str) -> Optional[Any]:
        return self._checkpoints.get(checkpoint_id)

    @property
    def all_contexts(self) -> List[Any]:
        """All unique contexts (one per checkpoint)."""
        return list(self._checkpoints.values())

    @property
    def model_names(self) -> List[str]:
        """All configured model names."""
        return list(self._models.keys())

    # ── port pool ───────────────────────────────────────────────
    def allocate_port(self, model_name: str) -> int:
        """Reuse a stopped checkpoint's port, else take the next from the pool."""
        ckpt_id = self._models.get(model_name)
        ctx = self._checkpoints.get(ckpt_id) if ckpt_id else None
        if ctx is not None and ctx.port is not None:
            return ctx.port
        end_port = self._start_port + self._pool_size - 1
        if self._next_port > end_port:
            raise RuntimeError(f"Port range exceeded: {self._start_port}\u2013{end_port}")
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
