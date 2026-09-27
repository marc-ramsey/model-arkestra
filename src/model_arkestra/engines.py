"""Engine registry — CLI arg building per inference engine.

An engine knows how to turn a merged param dict + port into the CLI tokens
for its server binary. Runners stay engine-agnostic: they look up the engine
class by name. New engine = one module + one entry in ENGINES.
"""
from __future__ import annotations

from typing import Any, Dict, List

from model_arkestra.llama_cpp import LlamaCppEngine
from model_arkestra.sdcpp import SdcppEngine


class EngineError(Exception):
    """Unknown engine name — a config typo, fail at start."""


ENGINES: Dict[str, type] = {
    "llama-cpp": LlamaCppEngine,
    "sdcpp": SdcppEngine,
}


def build_args(engine_name: str, merged: Dict[str, Any], port: int) -> List[str]:
    """Build CLI tokens for the given engine; unknown names raise EngineError.

    (The old degenerate list(merged.values()) fallback is gone — an unknown
    engine is a config error, not something to guess at.)
    """
    cls = ENGINES.get(engine_name)
    if cls is None:
        known = ", ".join(sorted(ENGINES))
        raise EngineError(f"unknown engine: {engine_name!r} (known: {known})")
    return cls.build_cli_args(merged, port)
