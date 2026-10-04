"""Resolution chain for stream-sock-timeout:
model → checkpoint(merged) → backend.args → default.

The value bounds the gap between SSE chunks on an in-flight stream (aiohttp
sock_read). Nothing set anywhere yields None so the provider falls back to its
hardcoded floor. The walk is shared with build_model_args via resolve_key_chain()
in common.py; this key deliberately stays OUT of schemas.yaml — it configures a
provider, not inference.
"""
from __future__ import annotations

import os

CFG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "stream_sock_timeout_cfg.yaml"
)


def _arkestra(yaml_text: str):
    """Build a minimal ModelArkestra from inline YAML."""
    with open(CFG_PATH, "w") as f:
        f.write(yaml_text)
    from model_arkestra.arkestra import ModelArkestra
    return ModelArkestra(CFG_PATH, ready_timeout=10)


def _yaml(default_extra: str = "", models_block: str = "") -> str:
    """Assemble a config doc. *models_block* is pre-indented YAML for the
    ``models:`` mapping (entries at 2-space indent)."""
    return f"""\
default:
  model-start-port: 18020
  model-ports: 4{default_extra}

backends: {{}}
runners:
  process:
    class-name: ProcessRunner

models:{models_block}
"""


class TestStreamSockTimeoutChain:

    def test_nothing_set_returns_none(self):
        a = _arkestra(_yaml(models_block="\n  m1:\n    model: dummy/x:Q4_K_M\n"))
        assert a.resolve_stream_sock_timeout("m1") is None

    def test_default_only(self):
        y = _yaml(default_extra="\n  stream-sock-timeout: 450",
                  models_block="\n  m1:\n    model: dummy/x:Q4_K_M\n")
        assert _arkestra(y).resolve_stream_sock_timeout("m1") == 450.0

    def test_model_wins_over_default(self):
        y = _yaml(default_extra="\n  stream-sock-timeout: 450",
                  models_block=("\n  m1:\n"
                                "    model: dummy/x:Q4_K_M\n"
                                "    stream-sock-timeout: 900\n"))
        assert _arkestra(y).resolve_stream_sock_timeout("m1") == 900.0

    def test_checkpoint_wins_over_default(self):
        y = _yaml(default_extra="\n  stream-sock-timeout: 450",
                  models_block=("\n  m1:\n"
                                "    checkpoint: bigckpt\n")) + """checkpoints:
  bigckpt:
    ref: dummy/x:Q4_K_M
    stream-sock-timeout: 600
"""
        assert _arkestra(y).resolve_stream_sock_timeout("m1") == 600.0

    def test_model_wins_over_checkpoint(self):
        y = _yaml(models_block=("\n  m1:\n"
                                "    checkpoint: bigckpt\n"
                                "    stream-sock-timeout: 900\n")) + """checkpoints:
  bigckpt:
    ref: dummy/x:Q4_K_M
    stream-sock-timeout: 600
"""
        assert _arkestra(y).resolve_stream_sock_timeout("m1") == 900.0

    def test_unknown_model_returns_none(self):
        """No entry for the name — even with a default set, nothing to resolve."""
        y = _yaml(default_extra="\n  stream-sock-timeout: 450",
                  models_block="\n  m1:\n    model: dummy/x:Q4_K_M\n")
        assert _arkestra(y).resolve_stream_sock_timeout("ghost") is None

    def test_backend_args_wins_over_default(self):
        """backend.args level — the rung added by sharing resolve_key_chain()."""
        y = (f"default:\n  model-start-port: 18020\n  model-ports: 4\n"
             f"  stream-sock-timeout: 450\n\n"
             f"backends:\n  gpu:\n    runner: process\n"
             f"    args:\n      stream-sock-timeout: 600\n\n"
             f"runners:\n  process:\n    class-name: ProcessRunner\n\n"
             f"models:\n  m1:\n    model: dummy/x:Q4_K_M\n    backend: gpu\n")
        assert _arkestra(y).resolve_stream_sock_timeout("m1") == 600.0

    def test_model_wins_over_backend_args(self):
        y = (f"backends:\n  gpu:\n    runner: process\n"
             f"    args:\n      stream-sock-timeout: 600\n\n"
             f"runners:\n  process:\n    class-name: ProcessRunner\n\n"
             f"models:\n  m1:\n    model: dummy/x:Q4_K_M\n    backend: gpu\n"
             f"    stream-sock-timeout: 900\n")
        assert _arkestra(y).resolve_stream_sock_timeout("m1") == 900.0
