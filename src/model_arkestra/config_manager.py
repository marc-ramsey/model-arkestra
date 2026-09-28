"""Model-aware ConfigManager — extends base with model-specific accessors."""
from __future__ import annotations


import copy
import json
import logging
from typing import Any, Dict, Optional, Union

import yaml
from importlib.resources import files as _files

from llm_config_manager import ConfigManager


class ModelConfigManager(ConfigManager):
    """ConfigManager extended with model and backend lookups.

    Layered config: the package-shipped backends.yaml is the read-only base;
    the user's config.yaml overlays it (per-key, deep merge — overlay wins).
    ``self.data`` holds the merged view (macros expanded); ``self._config``
    holds the raw user file (placeholders intact) and is the only thing
    :meth:`export` writes. All user state lives in config.yaml; nothing ever
    writes backends.yaml.
    """

    def __init__(self, config_path: str, strict_expansion: Optional[bool] = None):
        # User file — raw, no macro expansion; export() target.
        self._config = ConfigManager(config_path, expand_macros=False)
        super().__init__(config_path, strict_expansion=strict_expansion)
        # Shipped base underneath the user overlay → merged view in self.data.
        self.data = self._load_base()
        self.merge(copy.deepcopy(self._config.get_dict()))
        # Expand macros on the merged view only; _config keeps placeholders.
        self._expand_macros()
        # Runtime-only effective default backend. Set at init after GPU detection
        # when the configured default's runtime is missing. Kept OUT of self.data so
        # a config save never persists the auto-detected fallback over the user's choice.
        self._effective_default_backend: Optional[str] = None
        self._warn_nested_args()

    def _warn_nested_args(self) -> None:
        """Warn about nested ``args:`` blocks on models/checkpoints.

        Model and checkpoint inference keys are resolved flat (at the entry's
        own level); a nested mapping is silently ignored by the arg merger,
        so flag it loudly instead.
        """
        log = logging.getLogger(__name__)
        for section in ("models", "checkpoints"):
            entries = self.data.get(section)
            if not isinstance(entries, dict):
                continue
            for name, entry in entries.items():
                if isinstance(entry, dict) and isinstance(entry.get("args"), dict):
                    log.warning("%s '%s' has a nested 'args' block — ignored. "
                                "Move its keys flat under '%s'", section, name, name)

    @staticmethod
    def _load_base() -> Dict[str, Any]:
        """Load the package-shipped backends.yaml (empty dict if unavailable)."""
        try:
            text = (_files("model_arkestra.data") / "backends.yaml").read_text()
        except Exception:
            return {}
        return yaml.safe_load(text) or {}

    # ── Mutators: apply to the merged view AND the user file ─────────

    def __setitem__(self, path: str, value: Any) -> None:
        super().__setitem__(path, value)
        self._config[path] = value

    def merge(self, update: dict) -> dict:
        result = super().merge(update)
        self._config.merge(update)
        return result

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        if key in self._config.data:
            del self._config[key]

    def export(self, output_path: str, fmt: str = 'yaml') -> None:
        """Write the user config file only — never the merged view."""
        if fmt not in ('json', 'yaml'):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'json' or 'yaml'.")
        with open(output_path, 'w') as f:
            if fmt == 'json':
                json.dump(self._config.data, f, indent=2)
            else:
                yaml.dump(self._config.data, f, default_flow_style=False, sort_keys=False)

    def effective_default_backend(self) -> Optional[str]:
        """Return the backend to use when a model has no explicit ``backend:``.

        Prefers the runtime-detected override (set at init), else the configured
        ``backends.default``. Never writes to disk.
        """
        if self._effective_default_backend:
            return self._effective_default_backend
        be = self.data.get("backends") or {}
        if isinstance(be, dict):
            return be.get("default")
        return None

    def get_models(self) -> list[str]:
        """Return a list of all model instance names (the ``models:`` keys).

        These are the invocable, user-facing names. Distinct from checkpoint
        ids (see :meth:`get_checkpoints`) which name the shared weight files.
        """
        models = self.data.get("models")
        return list(models.keys()) if isinstance(models, dict) else []

    def get_checkpoints(self) -> list[str]:
        """Return a list of checkpoint ids (the ``checkpoints:`` keys).

        Each names one weight file and the load-profile args shared by every
        model instance that references it.
        """
        ckpts = self.data.get("checkpoints")
        return list(ckpts.keys()) if isinstance(ckpts, dict) else []

    def get_checkpoint(self, checkpoint_id: str) -> Union[Dict[str, Any], None]:
        """Return the raw config dict for a checkpoint id (no merge)."""
        ckpts = self.data.get("checkpoints")
        if not isinstance(ckpts, dict):
            return None
        ckpt = ckpts.get(checkpoint_id)
        return dict(ckpt) if isinstance(ckpt, dict) else None

    def checkpoint_for(self, model_name: str) -> Optional[str]:
        """Return the checkpoint id a model instance references, or None."""
        models = self.data.get("models")
        if not isinstance(models, dict):
            return None
        model = models.get(model_name)
        if not isinstance(model, dict):
            return None
        return model.get("checkpoint")

    def get_model(
        self, model_name: str, env_vars: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], None]:
        """Return the *effective* config dict for a named model instance.

        A model instance references a shared checkpoint via its ``checkpoint:``
        key. The effective config is the checkpoint's load-profile args merged
        with the instance's own args (instance wins on conflict). The
        checkpoint's ``ref`` is surfaced as ``model`` so existing callers that
        read ``cfg["model"]`` work unchanged.

        If *env_vars* is provided the values are resolved against them
        (*strict=True*).  Unresolved placeholders survive when *env_vars* is not
        given so they can be resolved at runtime.
        """
        models = self.data.get("models")
        if not isinstance(models, dict):
            return None
        model = models.get(model_name)
        if model is None:
            return None
        if not isinstance(model, dict):
            return None

        # Merge the referenced checkpoint underneath the instance config.
        merged: Dict[str, Any] = {}
        ckpt_id = model.get("checkpoint")
        if ckpt_id is not None:
            ckpt = self.get_checkpoint(ckpt_id)
            if ckpt is None:
                raise ValueError(
                    f"Model '{model_name}' references unknown checkpoint "
                    f"'{ckpt_id}'. Declare it in the 'checkpoints:' section."
                )
            merged.update(ckpt)
        # Instance args override checkpoint args — including an explicit
        # 'model:' key, which then wins over the checkpoint's ref.
        instance = {k: v for k, v in model.items() if k != "checkpoint"}
        merged.update(instance)
        # The weight ref lives on the checkpoint; surface it as 'model' so
        # callers reading cfg["model"] work unchanged. An instance-level
        # 'model:' key already won the merge and is left untouched.
        if "ref" in merged and "model" not in merged:
            merged["model"] = merged.pop("ref")
        elif "ref" in merged:
            del merged["ref"]

        if env_vars is not None:
            # Runtime resolution is always strict: a missing env var must
            # surface as an error, never as a literal ${...} in the command.
            return self._traverse(merged, env_vars, strict=True)
        # No env vars: placeholders survive for later runtime resolution.
        return self._traverse(merged, {}, strict=False)

    def get_backend(self, backend_id: str) -> Union[Dict[str, Any], None]:
        """Return the backend dict for *backend_id*.

        Returns a copy so callers can inspect without mutating the original.
        A missing or null ``backends:`` section yields None, not an error.
        """
        backends = self.data.get("backends")
        if not isinstance(backends, dict):
            return None
        be = backends.get(backend_id)
        return dict(be) if isinstance(be, dict) else None
