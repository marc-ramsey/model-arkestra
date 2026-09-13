"""Capability derivation — computed from config, never stored.

A model's capabilities are a pure function of its config: the ONNX ``type:``
for auxiliary models, or the llama-cpp default (with one override) for LLMs.
Remote models mirror their worker and are empty locally until probed.

    modality  =  one client + one capability (+ optionally one backend)
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet


# ONNX ``type:`` → capability. Each auxiliary model is exactly one modality.
_ONNX_TYPE_TO_CAPABILITY = {
    "embedding": "embed",
    "whisper": "asr",
    "asr": "asr",
    "tts": "tts",
}

# Default capabilities for a llama-cpp model (chat + embed).
_LLAMA_DEFAULT: FrozenSet[str] = frozenset({"chat", "embed"})


def derive_capabilities(model_cfg: Dict[str, Any], backend_cfg: Dict[str, Any]) -> FrozenSet[str]:
    """Compute the capability set for one model from its config.

    ``model_cfg`` is the raw ``models.<name>`` mapping; ``backend_cfg`` the
    resolved ``backends.<id>`` mapping (may be empty).
    """
    # ONNX auxiliary models are routed by runner/type, not backend engine.
    if _is_onnx(model_cfg):
        cap = _ONNX_TYPE_TO_CAPABILITY.get(str(model_cfg.get("type", "")))
        return frozenset({cap}) if cap else frozenset()

    # Remote cluster models mirror the worker; nothing is known locally.
    if model_cfg.get("runner") == "remote" or (backend_cfg or {}).get("runner") == "remote":
        return frozenset()

    # llama-cpp default, with the single embedding-only override.
    override = model_cfg.get("capabilities")
    if isinstance(override, list) and len(override) == 1 and override[0] == "embed":
        return frozenset({"embed"})
    return _LLAMA_DEFAULT


def _is_onnx(model_cfg: Dict[str, Any]) -> bool:
    """True when the model is an in-process ONNX auxiliary model."""
    if model_cfg.get("runner") == "onnx":
        return True
    # Tags route to the ONNX runner even without an explicit runner key.
    tags = model_cfg.get("tags")
    if isinstance(tags, list) and any(t in ("asr", "tts", "embed") for t in tags):
        return True
    return bool(model_cfg.get("model_path"))
