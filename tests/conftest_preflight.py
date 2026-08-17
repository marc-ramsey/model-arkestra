"""Pre-flight validation for all runner backends.

Checks infrastructure dependencies (binary existence, image availability)
before any tests run. Fails immediately if required resources are missing —
never allows a "silent" graceful failure.

This prevents the class of bugs where mocked config produces valid strings
but real execution fails because the underlying resources don't exist.
"""

from __future__ import annotations

import subprocess
import os

import pytest


def _podman_images() -> set[str]:
    """Return all locally available podman images (repo:tag)."""
    try:
        out = subprocess.check_output(
            ["podman", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            text=True, timeout=5,
        )
        return {line.strip() for line in out.strip().split("\n") if line.strip()}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()


def _check_binary_available(backend_id: str, backends_cfg: dict) -> bool:
    """Check that process runner can find its binary for this backend."""
    be = backends_cfg.get(backend_id, {})
    if not be or isinstance(be, str):
        # Alias to another backend — check the alias target
        target_be = backends_cfg.get(be) if be else {}
        return _check_binary_available(str(be), backends_cfg)

    binary_dir = be.get("binary_dir", "")
    binary_name = be.get("binary", "llama-server")
    binary_path = os.path.join(binary_dir, binary_name)

    if not binary_path:
        # No binary_dir configured — runner will try default path
        return True  # can't validate without more context

    return os.path.isfile(binary_path)


def _check_image_available(backend_id: str, backends_cfg: dict) -> bool:
    """Check that podman/docker runner can find its image."""
    be = backends_cfg.get(backend_id, {})
    if not be or isinstance(be, str):
        target_be = backends_cfg.get(be) if be else {}
        return _check_image_available(str(be), backends_cfg)

    image = be.get("image", "")
    if not image:
        return True  # No image specified — process runner or other

    local_images = _podman_images()
    return image in local_images


@pytest.fixture(scope="session")
def preflight_check():
    """Validate all backends have their required resources before any test runs.

    FAILS (pytest.fail) if:
    - A backend config references a binary that doesn't exist on disk
    - A backend config references an image that doesn't exist locally
    - No podman/docker is available for container-backed backends

    This is the FIRST line of defense against "mocked test passes but real code broken" bugs.
    """
    import yaml

    test_cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test-config.yaml"
    )

    with open(test_cfg_path) as f:
        cfg = yaml.safe_load(f)

    backends_cfg = cfg.get("backends", {}) or {}
    errors = []

    for be_id, be in backends_cfg.items():
        if not isinstance(be, dict):
            continue

        runner = be.get("runner", "process")

        if runner == "process":
            # Must have binary_dir pointing to an existing file
            if not _check_binary_available(be_id, backends_cfg):
                errors.append(
                    f"Backend '{be_id}' (runner=process): binary not found. "
                    f"This will cause ALL process-runner tests to fail silently."
                )

        elif runner in ("podman", "docker"):
            # Must have image that exists locally
            if not _check_image_available(be_id, backends_cfg):
                errors.append(
                    f"Backend '{be_id}' (runner={runner}): image not available locally. "
                    f"No registry is configured for pull."
                )

            # Check container runtime exists
            runtime = runner  # podman or docker
            try:
                subprocess.run(
                    [runtime, "--version"],
                    capture_output=True, timeout=5,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                errors.append(
                    f"Backend '{be_id}' (runner={runner}): {runtime} binary not found."
                )

    if errors:
        pytest.fail(
            "Pre-flight infrastructure check failed.\n\n" + "\n".join(f"  - {e}" for e in errors)
            + "\n\nFix the backends section or remove them from test-config.yaml."
        )
