"""Quick smoke test verifying the admin log route fix.

Tests the two scenarios:
1. Snapshot mode returns logs from the correct runner (find_context)
2. Snapshot mode returns empty data when no context exists for model
"""
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from model_arkestra.admin import ArkestraAdmin


def test_snapshot_uses_find_context():
    """Verify snapshot mode calls find_context and gets logs from that runner."""
    # Minimal mock to exercise the route code path
    ctx = MagicMock()
    ctx.name = "test-model"

    runner = MagicMock()
    runner.get_logs = AsyncMock(return_value=["line1", "line2", "line3"])
    ctx.runner = runner

    arkestra = MagicMock()
    arkestra.find_context.return_value = ctx
    arkestra.cm.data = {"models": {"test-model": {}}}


    app = FastAPI()
    server = MagicMock()
    server._arkestra = arkestra

    admin = ArkestraAdmin(server, None, app)
    admin.install()

    client = TestClient(app)
    resp = client.get("/admin/log/test-model", params={"follow": False})

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "log"
    assert body["data"] == ["line1", "line2", "line3"]

    # Verify find_context was called (not iterating _runners)
    arkestra.find_context.assert_called_once_with("test-model")
    runner.get_logs.assert_called_once_with("test-model", 100)


def test_snapshot_returns_empty_when_no_context():
    """When model has no running context, return empty data — don't error."""
    ctx = None  # find_context returns None

    arkestra = MagicMock()
    arkestra.find_context.return_value = ctx
    arkestra.cm.data = {"models": {"ghost-model": {}}}

    app = FastAPI()
    server = MagicMock()
    server._arkestra = arkestra

    admin = ArkestraAdmin(server, None, app)
    admin.install()

    client = TestClient(app)
    resp = client.get("/admin/log/ghost-model", params={"follow": False})

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "log"
    assert body["data"] == []


def test_snapshot_handles_missing_get_logs():
    """If runner exists but lacks get_logs, return empty — don't crash."""
    ctx = MagicMock()
    ctx.name = "test-model"

    runner = MagicMock(spec=[])  # no get_logs attribute
    ctx.runner = runner

    arkestra = MagicMock()
    arkestra.find_context.return_value = ctx
    arkestra.cm.data = {"models": {"test-model": {}}}

    app = FastAPI()
    server = MagicMock()
    server._arkestra = arkestra

    admin = ArkestraAdmin(server, None, app)
    admin.install()

    client = TestClient(app)
    resp = client.get("/admin/log/test-model", params={"follow": False})

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "log"
    assert body["data"] == []


def test_snapshot_handles_get_logs_exception():
    """If get_logs raises, catch and return empty — don't 500."""
    ctx = MagicMock()
    ctx.name = "test-model"

    runner = MagicMock()
    runner.get_logs = AsyncMock(side_effect=RuntimeError("container gone"))
    ctx.runner = runner

    arkestra = MagicMock()
    arkestra.find_context.return_value = ctx
    arkestra.cm.data = {"models": {"test-model": {}}}

    app = FastAPI()
    server = MagicMock()
    server._arkestra = arkestra

    admin = ArkestraAdmin(server, None, app)
    admin.install()

    client = TestClient(app)
    resp = client.get("/admin/log/test-model", params={"follow": False})

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "log"
    assert body["data"] == []


def test_snapshot_custom_lines_param():
    """Verify the 'lines' param is passed through to get_logs."""
    ctx = MagicMock()

    runner = MagicMock()
    runner.get_logs = AsyncMock(return_value=["a", "b"])
    ctx.runner = runner

    arkestra = MagicMock()
    arkestra.find_context.return_value = ctx
    arkestra.cm.data = {"models": {"m": {}}}

    app = FastAPI()
    server = MagicMock()
    server._arkestra = arkestra

    admin = ArkestraAdmin(server, None, app)
    admin.install()

    client = TestClient(app)
    resp = client.get("/admin/log/m", params={"lines": 5})

    assert resp.status_code == 200
    runner.get_logs.assert_called_once_with("m", 5)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
