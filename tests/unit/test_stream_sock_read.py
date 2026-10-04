"""LlamaProvider stream sock-read timeout — silent-gap tolerance.

Regression: agentic clients (pi-agent) hold a reasoning session open while the
model is mid-tool-call or producing one very slow token; llama-server emits no
bytes during that window. The old 120 s ``sock_read`` default fired inside such
gaps, killing healthy streams at exactly ~120 s ("stream finish without reason"
in client JSON). A silent stream with a live TCP connection is not evidence of
a dead peer — reset/EOF on a truly-dead server fires immediately regardless.

Property under test: the *default* (no config override) tolerates >= 300 s of
socket silence; an explicit short override still bounds reads as before.
"""
from __future__ import annotations

import pytest

from model_arkestra.providers.llama import LlamaProvider, stream_timeout


# Minimum acceptable silent-gap tolerance for the default timeout (seconds).
MIN_DEFAULT_SILENCE_TOLERANCE = 300.0


def test_default_tolerates_long_silent_gaps():
    """No config override: sock_read must not fire inside long reasoning/tool gaps."""
    t = stream_timeout(None)
    assert t.sock_read is None or t.sock_read >= MIN_DEFAULT_SILENCE_TOLERANCE, (
        f"default sock_read={t.sock_read}s kills streams silent for that many "
        f"seconds — agentic tool-call and slow-token gaps exceed it")


def test_default_provider_uses_tolerant_timeout():
    """Provider built without stream_sock_timeout inherits the tolerant default."""
    prov = LlamaProvider("m", 18020)
    assert (prov._stream_timeout.sock_read is None
            or prov._stream_timeout.sock_read >= MIN_DEFAULT_SILENCE_TOLERANCE)


def test_explicit_override_still_bounded():
    """Config default/stream-sock-timeout remains a working knob."""
    t = stream_timeout(2.0)
    assert t.sock_read == 2.0
