"""Smoke tests for admin log route fixes."""

import json
from collections import deque
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from model_arkestra.admin import ArkestraAdmin


def _build_admin(find_ctx_result=None, runner_get_logs=None):
    ctx = MagicMock()
    if find_ctx_result is None:
        find_ctx_result = ctx

    runner = AsyncMock()
    if runner_get_logs is not None:
        runner.get_logs = AsyncMock(return_value=runner_get_logs)
    ctx.runner = runner

    arkestra = MagicMock()
    arkestra.find_context.return_value = find_ctx_result
    arkestra.cm.data = {"models": {"test-model": {}}}

    app = FastAPI()
    server = MagicMock()
    server._arkestra = arkestra

    admin = ArkestraAdmin(server, None, app)
    admin.install()
    return app, arkestra, runner


# ── Issue #3: Snapshot mode ───────────────────────────────────────────────────


def test_snapshot_uses_find_context():
    app, arkestra, runner = _build_admin(runner_get_logs=["line1", "line2"])
    client = TestClient(app)
    resp = client.get("/admin/log/test-model", params={"follow": False})

    assert resp.status_code == 200
    assert resp.json()["data"] == ["line1", "line2"]
    arkestra.find_context.assert_called_once_with("test-model")


def test_snapshot_noop_when_get_logs_missing():
    ctx = MagicMock(spec=[])
    ctx.runner = None
    app, arkestra, _ = _build_admin(find_ctx_result=ctx)
    client = TestClient(app)
    resp = client.get("/admin/log/test-model", params={"follow": False})

    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_snapshot_noop_when_get_logs_raises():
    app, _, runner = _build_admin()
    runner.get_logs = AsyncMock(side_effect=RuntimeError("gone"))
    client = TestClient(app)
    resp = client.get("/admin/log/test-model", params={"follow": False})

    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_snapshot_passes_lines_param():
    app, _, runner = _build_admin(runner_get_logs=["a", "b"])
    client = TestClient(app)
    client.get("/admin/log/test-model", params={"lines": 5})

    runner.get_logs.assert_called_once_with("test-model", 5)


# ── Issue #6: Follow-mode wrap detection ──────────────────────────────────────


def test_deque_comparison_detects_wrap():
    """Prove ``new_buf != buf`` catches changes when shared deque wraps.

    The actual code reads ``list(ctx._log_buffer)`` twice from the **same**
    deque — so wrapping is inherent and content comparison detects it where
    size comparison fails.
    """
    d = deque(maxlen=3)
    d.extend(["a", "b", "c"])

    buf = list(d)      # first read (snapshot), d still holds ["a","b","c"]
    d.append("d")       # wrap: drops oldest, new enters. d now ["b","c","d"]
    new_buf = list(d)  # second read from same deque

    assert buf == ["a", "b", "c"]
    assert new_buf == ["b", "c", "d"]
    assert len(buf) == len(new_buf)          # size identical — old check fails
    assert new_buf != buf                    # content differs — new check works


def test_deque_comparison_skips_no_change():
    """Identical reads → no change detected."""
    d = deque(maxlen=3)
    d.extend(["x", "y", "z"])

    a = list(d)
    b = list(d)  # nothing changed, same deque state

    assert a == b
    assert not (a != b)
