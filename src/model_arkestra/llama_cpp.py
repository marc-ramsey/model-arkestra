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

    # Config keys → llama-server CLI flags.
    _CLI_MAP: Dict[str, str] = {
        'temp':           '--temp',
        'top-k':          '--top-k',
        'top-p':          '--top-p',
        'min-p':          '--min-p',
        'typical':        '--typical',
        'rope-freq-base': '--rope-freq-base',
        'rope-freq-scale':'--rope-freq-scale',
        'reasoning-budget':'--reasoning-budget',
        'keep':           '--keep',
        'top-nsigma':     '--top-nsigma',
        'frequency-penalty':'--frequency-penalty',
        'presence-penalty':'--presence-penalty',
        'repeat-penalty': '--repeat-penalty',
        'repeat-last-n':  '--repeat-last-n',
        'rope-scaling':   '--rope-scaling',
        'rope-scale':     '--rope-scale',
        'ignore-eos':     '--ignore-eos',
        'grammar':        '--grammar',
        'chat-template':  '--chat-template',
        'jinja':          '--jinja',
        'jinja2':         '--jinja',
        'flash-attn':     '--flash-attn',
        'fa':             '--flash-attn',
        'ctx-size':       '--ctx-size',
        'ngl':            '-ngl',
        'threads':        '--threads',
        'threads-batch':  '--threads-batch',
        'no-mmap':        '--no-mmap',
    }

    _DEFAULT_REPO = "hf"

    def filter_infer_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Return only inference-safe keys from arbitrary kwargs."""
        return {k: v for k, v in kwargs.items() if k in self.LLAMA_INFER_ARGS}

    @staticmethod
    def build_cli_args(merged: Dict[str, Any], port: int) -> List[str]:
        """Convert merged param dict + port into llama-server CLI tokens.

        Reads ``model``, ``repo``, ``mmproj`` from merged (or uses defaults),
        emits the corresponding flags. Remaining keys map through _CLI_MAP.
        Unknown keys raise ValueError.
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

        # ── Emit inference args via whitelist map ─────────────────────────
        for key, value in merged.items():
            if key in ('model', 'repo', 'mmproj', 'port'):
                continue  # handled above
            if key not in LlamaCppEngine._CLI_MAP:
                raise ValueError(f"Unknown arg: {key}")
            flag = LlamaCppEngine._CLI_MAP[key]
            if isinstance(value, bool):
                if value:
                    cli.append(flag)
            elif value is not None and key != 'port':
                cli.extend([flag, str(value)])

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
