"""Llama.cpp engine — arg sets and CLI building for llama-server / llama serve."""
from __future__ import annotations

from typing import Any, Dict, List


class LlamaCppEngine:
    """Llama.cpp inference-parameter filtering and CLI generation."""

    _DEFAULT_REPO = "hf"

    # Keys that get single-dash prefix in CLI (e.g. -ngl 999).
    # Everything else gets --key.
    _SINGLE_DASH = frozenset({
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
    })

    # Map OpenAI/config key names → llama-server CLI flags.
    _ALIAS_MAP = {
        'max-tokens': 'n-predict',
    }

    @staticmethod
    def build_cli_args(merged: Dict[str, Any], port: int) -> List[str]:
        """Convert merged param dict + port into llama-server CLI tokens.

        All keys except model/repo/mmproj/port become flags:
          --key value  (default), or -key value if in _SINGLE_DASH.
        Boolean True → presence-only flag (-flag / --flag). False → skipped.
        """
        cli: List[str] = []

        # ── Resolve model-related args ────────────────────────────────────
        repo = merged.get('repo', LlamaCppEngine._DEFAULT_REPO) or LlamaCppEngine._DEFAULT_REPO
        model_ref = merged.get('model')
        mmproj_path = merged.get('mmproj')

        if model_ref:
            if repo == 'hf':
                cli.extend(['-hf', model_ref, '--alias', model_ref])
            else:
                cli.extend(['--model', model_ref])

        if mmproj_path:
            cli.extend(['--mmproj', mmproj_path])

        # ── Emit all remaining keys as CLI flags ──────────────────────────
        for key, value in merged.items():
            if key in ('model', 'repo', 'mmproj', 'port'):
                continue
            # Resolve aliased keys (e.g. max-tokens → n-predict)
            kebab = LlamaCppEngine._ALIAS_MAP.get(key, key).replace('_', '-')
            prefix = '-' if kebab in LlamaCppEngine._SINGLE_DASH else '--'
            if isinstance(value, bool):
                if value:
                    cli.append(f"{prefix}{kebab}")
            elif value is not None:
                cli.extend([f"{prefix}{kebab}", str(value)])

        cli.extend(['--port', str(port)])
        return cli



