"""Integration tests for /admin/images endpoints.

Tests require a container runtime (podman or docker) on PATH.
If neither is available, all tests are skipped with an explanation.
No mocks — uses real subprocess calls to podman/docker where possible.
"""

from __future__ import annotations

import os
import shutil

import pytest
from fastapi.testclient import TestClient


def _has_runtime():
    return bool(shutil.which("podman") or shutil.which("docker"))


@pytest.fixture(scope="module", autouse=True)
def skip_if_no_runtime():
    """Skip entire module if no container runtime is available."""
    if not _has_runtime():
        pytest.skip("No container runtime (podman/docker) found on PATH — skipping admin images tests")


@pytest.fixture()
def app_client():
    """Return a TestClient for the ArkestraAdmin app.

    Uses tests/test-admin-config.yaml — isolated from sample-config.yaml.
    Reads ADMIN_KEY from config so it stays in sync.
    """
    from model_arkestra.server import ArkestraServer
    server = ArkestraServer(
        config_path="tests/test-admin-config.yaml",
        port=18005,
        ready_timeout=5,
    )
    # Don't start the server — we only need the FastAPI app
    client = TestClient(server.get_app())
    key = server._arkestra.cm.data.get("env", {}).get("ADMIN_KEY") or ""
    client.headers["X-Admin-Key"] = key
    yield client


# ── GET /admin/images ────────────────────────────────────────────────


class TestListImages:
    def test_returns_list_of_backend_entries(self, app_client):
        resp = app_client.get("/admin/images")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # test-admin-config.yaml defines backends for rocm and vulkan-radv
        assert len(data) >= 1
        ids = {entry["backend_id"] for entry in data}
        assert "rocm" in ids or "vulkan-radv" in ids

    def test_entry_has_required_keys(self, app_client):
        resp = app_client.get("/admin/images")
        data = resp.json()
        required_keys = {"backend_id", "runner", "runtime_detected", "image",
                         "containerfile", "available"}
        for entry in data:
            assert required_keys.issubset(entry.keys()), f"Missing keys: {required_keys - entry.keys()}"

    def test_rocm_backend_resolves_podman_runner(self, app_client):
        resp = app_client.get("/admin/images")
        rocm_entry = [e for e in resp.json() if e["backend_id"] == "rocm"]
        assert len(rocm_entry) == 1
        assert rocm_entry[0]["runner"] == "podman"

    def test_rocm_image_tag_matches_config(self, app_client):
        resp = app_client.get("/admin/images")
        rocm_entry = [e for e in resp.json() if e["backend_id"] == "rocm"][0]
        assert rocm_entry["image"] == "ark-llama:rocm"

    def test_rocm_image_available_false_when_not_built(self, app_client):
        """If the image hasn't been built locally, available should be False."""
        resp = app_client.get("/admin/images")
        rocm_entry = [e for e in resp.json() if e["backend_id"] == "rocm"][0]
        # This may fail if podman is not on PATH (would have been skipped)
        # But the image should definitely not exist locally unless user built it
        if not _has_runtime():
            pytest.skip("no runtime")
        if rocm_entry["runtime_detected"] and rocm_entry["runner"] == "podman":
            assert rocm_entry["available"] is False, (
                "Image should not be available by default — user must build it first"
            )


# ── POST /admin/images/build ─────────────────────────────────────────


class TestBuildImage:
    def test_missing_backend_returns_400(self, app_client):
        resp = app_client.post("/admin/images/build", json={})
        assert resp.status_code == 400
        assert "backend" in resp.json()["detail"].lower()

    def test_unknown_backend_returns_error(self, app_client):
        """Requesting a backend not in the images config should return an error."""
        resp = app_client.post("/admin/images/build", json={"backend": "nonexistent"})
        data = resp.json()
        # Unknown backend falls through to defaults — returns skipped (no runner binary)
        assert data.get("skipped") is True or data.get("error") is not None

    @pytest.mark.slow
    @pytest.mark.skipif(not shutil.which("podman"), reason="podman not available")
    def test_build_rocm_resolves_correct_files(self, app_client):
        """When podman IS available, build should resolve containerfile and image from config.

        The actual Containerfile may fail to build on some systems (missing packages in image),
        but the endpoint should still return structured output with the correct backend/image/runtime.
        """
        resp = app_client.post("/admin/images/build", json={"backend": "rocm"})
        data = resp.json()
        assert data["backend"] == "rocm"
        assert data["image"] == "ark-llama:rocm"
        assert data["runtime"] in ("podman", "docker")
        # success reflects whether podman build exited cleanly — may be False on some systems
        assert "output" in data  # always includes build stdout/stderr


# ── DELETE /admin/images/{image} ─────────────────────────────────────


class TestRemoveImage:
    def test_unknown_image_returns_404(self, app_client):
        resp = app_client.delete("/admin/images/never-built-image:latest")
        assert resp.status_code == 404

    def test_remove_nonexistent_tag(self, app_client):
        """Request to remove a configured but not-yet-built image."""
        # This tests the path: find backend → resolve runner → check runtime → attempt rmi
        if shutil.which("podman"):
            resp = app_client.delete("/admin/images/ark-llama:rocm")
            data = resp.json()
            assert data.get("removed") is False or "error" in data
