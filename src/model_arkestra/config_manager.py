"""Model-aware ConfigManager — extends base with model-specific accessors."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from llm_config_manager import ConfigManager


class ModelConfigManager(ConfigManager):
    """ConfigManager extended with model and backend lookups."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Runtime-only effective default backend. Set at init after GPU detection
        # when the configured default's runtime is missing. Kept OUT of self.data so
        # a config save never persists the auto-detected fallback over the user's choice.
        self._effective_default_backend: Optional[str] = None

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
