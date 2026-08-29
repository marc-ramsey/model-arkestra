"""Model-aware ConfigManager — extends base with model-specific accessors."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from llm_config_manager import ConfigManager


class ModelConfigManager(ConfigManager):
    """ConfigManager extended with model and backend lookups."""

    def get_models(self) -> list[str]:
        """Return a list of all available model names."""
        models = self.data.get("models")
        return list(models.keys()) if isinstance(models, dict) else []

    def get_model(
        self, model_name: str, env_vars: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], None]:
        """Return the config dict for a named model.

        If *env_vars* is provided the model's string values are resolved
        against them (*strict=True*).  Unresolved placeholders (e.g. $PORT)
        survive when *env_vars* is not given so they can be resolved at runtime.
        """
        models = self.data.get("models")
        if not isinstance(models, dict):
            return None
        model = models.get(model_name)
        if model is None:
            return None
        if env_vars is not None:
            return self._traverse(model, env_vars, strict=self.strict_expansion)
        # Normalise whitespace without strict mode so unresolved keys survive.
        return self._traverse(model, {}, strict=False)

    def get_backend(self, backend_id: str) -> Union[Dict[str, Any], None]:
        """Return the backend dict for *backend_id*.

        Returns a copy so callers can inspect without mutating the original.
        """
        be = self.data.get("backends", {}).get(backend_id)
        if not isinstance(be, dict):
            return None
        return dict(be)
