"""Stable-diffusion.cpp engine — CLI building for sd-server (image-gen).

sd-server takes a small, fixed set of flags: the model path, optional
VAE/tokenizer/LLM companions, and listen address/port. Unlike llama-cpp,
its merged dict is mostly *file paths*, not sampling scalars.
"""
from __future__ import annotations

from typing import Any, Dict, List


class SdcppEngine:
    """Arg builder for stable-diffusion.cpp's sd-server."""

    # Keys consumed as companion files (may be absent).
    _COMPANION_KEYS = ("vae", "tokenizer", "llm")

    # Keys never emitted as flags — resolved into model/listen handling.
    _SKIP_KEYS = frozenset({"model", "repo", "port", *(_COMPANION_KEYS)})

    @staticmethod
    def build_cli_args(merged: Dict[str, Any], port: int) -> List[str]:
        """Turn the merged param dict + port into sd-server CLI tokens.

        -m <model> is required; --vae/--tokenizer/--llm are companion files;
        remaining scalar keys become --kebab value flags (bool True →
        presence-only, False/None → skipped).
        """
        cli: List[str] = []

        model = merged.get("model")
        if model:
            cli.extend(["-m", str(model)])

        for key in SdcppEngine._COMPANION_KEYS:
            value = merged.get(key)
            if value:
                cli.extend([f"--{key}", str(value)])

        for key, value in merged.items():
            if key in SdcppEngine._SKIP_KEYS:
                continue
            kebab = str(key).replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cli.append(f"--{kebab}")
            elif value is not None:
                cli.extend([f"--{kebab}", str(value)])

        cli.extend(["--listen-port", str(port)])
        return cli
