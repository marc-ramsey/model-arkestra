"""Llama.cpp engine — arg sets and CLI building for llama-server / llama serve."""
from __future__ import annotations

from typing import Any, Dict, List


class LlamaCppEngine:
    """Llama.cpp inference-parameter filtering and CLI generation."""

    # Inference params users can safely override via POST body.
    LLAMA_INFER_ARGS: set[str] = {
        'temp', 'top-k', 'top-p', 'min-p', 'typical', 'top-nsigma',
        'frequency-penalty', 'presence-penalty', 'repeat-penalty', 'repeat-last-n',
        'rope-freq-base', 'rope-freq-scale', 'rope-scaling', 'rope-scale',
        'reasoning-budget', 'keep', 'ignore-eos', 'grammar', 'chat-template',
    }

    # Metadata keys handled separately — not converted to CLI flags.
    _METADATA_KEYS = {'model', 'hf', 'port'}

    def filter_infer_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Return only inference-safe keys from arbitrary kwargs."""
        return {k: v for k, v in kwargs.items() if k in self.LLAMA_INFER_ARGS}

    @staticmethod
    def build_cli_args(merged: Dict[str, Any], port: int) -> List[str]:
        """Convert merged param dict + port into llama-server CLI tokens.

        ``merged`` comes from ``build_model_args()`` — a flat dict of model
        args plus runtime inference kwargs.  Port is injected as the final
        parameter.  Metadata keys (model, hf) are skipped here since they
        are handled by the caller/runner layer.
        """
        cli: List[str] = []
        for key, value in merged.items():
            if key in LlamaCppEngine._METADATA_KEYS:
                continue
            kebab = key.replace('_', '-')
            prefix = '-' if kebab in _LLAMA_SHORT_FLAGS else '--'
            if isinstance(value, bool):
                if value:
                    cli.append(f"{prefix}{kebab}")
            elif value is not None:
                cli.extend([f"{prefix}{kebab}", str(value)])
        cli.extend(['--port', str(port)])
        return cli


# Keys whose llama.cpp flag uses -x (single dash) instead of --x.
_LLAMA_SHORT_FLAGS = {
    'c', 't', 'tb', 'fa', 'e', 'kvo',
    'ctk', 'ctv', 'dt', 'dio', 'lm', 'dev',
    'ot', 'cmoe', 'ncmoe',
    'ngl', 'sm', 'ts', 'mg', 'fit', 'fitt',
    'fitc', 'b', 'ub', 'hf', 'hff', 'hft',
    'dr', 'mu', 'cl',
    'a', 'ag', 'mm', 'mmu',
    's', 'l', 'j', 'jf',
    'bs', 'lcs', 'lcd',
    'ctxcp', 'cms', 'cram',
    'kvu', 'r', 'sp',
    'np', 'cb', 'to',
    'rea', 'sps', 'v', 'lv', 'm',
}
