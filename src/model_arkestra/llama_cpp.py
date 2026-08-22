"""Llama.cpp engine — arg sets and CLI building for llama-server / llama serve."""
from __future__ import annotations

from typing import Any, Dict


class LlamaCppEngine:
    """Llama.cpp inference-parameter filtering."""

    # Inference params users can safely override via POST body.
    # Anything not in this set is silently dropped to prevent CLI crashes.
    LLAMA_INFER_ARGS: set[str] = {
        'temp', 'top-k', 'top-p', 'min-p', 'typical', 'top-nsigma',
        'frequency-penalty', 'presence-penalty', 'repeat-penalty', 'repeat-last-n',
        'rope-freq-base', 'rope-freq-scale', 'rope-scaling', 'rope-scale',
        'reasoning-budget', 'keep', 'ignore-eos', 'grammar', 'chat-template',
    }

    def filter_infer_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Return only inference-safe keys from arbitrary kwargs."""
        return {k: v for k, v in kwargs.items() if k in self.LLAMA_INFER_ARGS}
