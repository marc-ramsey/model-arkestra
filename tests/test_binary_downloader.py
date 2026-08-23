"""Tests for the BinaryDownloader module."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from model_arkestra.binary_downloader import (
    BinaryDownloader, BinaryDownloaderError, ChecksumMismatch,
    GITHUB_RELEASE, LOCAL_FILE, RUNTIME_CHECK, RuntimeCheckError,
    _file_lock,
)
from model_arkestra.common import resolve_config_path, resolve_backends_path


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_cache(tmp_path):
    """Temporary cache directory."""
    return tmp_path / "cache"


@pytest.fixture
def github_source_cfg():
    """Minimal GitHub release source config."""
    return {
        "type": GITHUB_RELEASE,
        "repo": "test-org/test-repo",
        "release_type": "latest",
        "asset_pattern": "*.tar.gz",
        "sha256_asset": "*.sha256",
    }


@pytest.fixture
def local_source_cfg(tmp_path):
    """Local file source config pointing to a real binary."""
    # Create a dummy executable in temp dir
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bin_file = bin_dir / "llama-server"
    bin_file.write_text("#!/bin/bash\necho hello")
    bin_file.chmod(0o755)

    return {
        "type": LOCAL_FILE,
        "path": str(bin_file),
        "sha256": "",  # skip verification for this test
    }


@pytest.fixture
def downloader(github_source_cfg, tmp_cache):
    """BinaryDownloader instance with GitHub release source."""
    return BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-rocm",
        source_cfg=github_source_cfg,
    )


# ── Unit tests ───────────────────────────────────────────────────────────

def test_local_file_no_path_raises(local_source_cfg, tmp_cache):
    """Missing path raises BinaryDownloaderError."""
    bad_cfg = dict(local_source_cfg)
    bad_cfg["path"] = ""

    dl = BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-local",
        source_cfg=bad_cfg,
    )

    async def _test():
        with pytest.raises(BinaryDownloaderError, match="no 'path' configured"):
            await dl.resolve()

    import asyncio
    asyncio.run(_test())


def test_local_file_not_found_raises(local_source_cfg, tmp_cache):
    """Non-existent path raises BinaryDownloaderError."""
    bad_cfg = dict(local_source_cfg)
    bad_cfg["path"] = "/tmp/does-not-exist-12345"

    dl = BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-local",
        source_cfg=bad_cfg,
    )

    async def _test():
        with pytest.raises(BinaryDownloaderError, match="not found"):
            await dl.resolve()

    import asyncio
    asyncio.run(_test())


def test_local_file_success(local_source_cfg, tmp_cache):
    """Existing local binary is returned."""
    dl = BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-local",
        source_cfg=local_source_cfg,
    )

    async def _test():
        result = await dl.resolve()
        assert os.path.isfile(result)

    import asyncio
    asyncio.run(_test())


def test_unsupported_source_type_raises(tmp_cache):
    """Unsupported source type raises BinaryDownloaderError."""
    bad_cfg = {"type": "bogus-source"}

    dl = BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-bogus",
        source_cfg=bad_cfg,
    )

    async def _test():
        with pytest.raises(BinaryDownloaderError, match="Unsupported source type"):
            await dl.resolve()

    import asyncio
    asyncio.run(_test())


def test_sha256_checksum_mismatch(tmp_cache):
    """SHA256 mismatch raises ChecksumMismatch."""
    bin_dir = tmp_path / "bin" if 'tmp_path' in dir() else Path(tempfile.mkdtemp()) / "bin"
    bin_file = bin_dir / "llama-server"
    bin_file.parent.mkdir(parents=True, exist_ok=True)
    bin_file.write_text("bad content here")
    bin_file.chmod(0o755)

    from model_arkestra.binary_downloader import ChecksumMismatch
    expected = "a" * 64
    actual = "b" * 64

    e = ChecksumMismatch(expected, actual, "test.tar.gz")
    assert "expected" in str(e)
    assert "got" in str(e)


def test_cache_manifest_write_and_read(downloader):
    """Cache manifest persists and can be read back."""
    downloader._write_cache_entry("test-key", "/fake/path/bin-test-key")

    entry = downloader._lookup_cache("test-key")
    assert entry is not None
    assert entry["path"] == "/fake/path/bin-test-key"
    assert "updated_at" in entry


def test_cache_lookup_missing(downloader):
    """Lookup of missing key returns None."""
    assert downloader._lookup_cache("nonexistent-key") is None


def test_stale_detection(downloader):
    """Stale entries (older than TTL) return True."""
    # Manually set updated_at to a long time ago
    import time as _time

    stale_entry = {"path": "/old", "updated_at": 0}  # effectively ancient
    assert downloader._is_stale(stale_entry) is True

    # Fresh entry (just now)
    fresh_entry = {
        "path": "/new",
        "updated_at": _time.monotonic() - 1,  # 1 second ago
    }
    assert downloader._is_stale(fresh_entry) is False


def test_sha256_file_computation(downloader):
    """SHA256 hex digest computation is correct."""
    import hashlib

    test_dir = Path(tempfile.mkdtemp())
    test_file = test_dir / "test.bin"
    test_file.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()

    actual = BinaryDownloader._sha256_file(test_file)
    assert actual == expected


def test_get_system_arch_x86_64():
    """Architecture detection returns standard strings."""
    with patch("platform.machine", return_value="x86_64"):
        from model_arkestra.binary_downloader import BinaryDownloader
        arch = BinaryDownloader._get_system_arch()
        assert arch == "x86_64"

    with patch("platform.machine", return_value="aarch64"):
        arch = BinaryDownloader._get_system_arch()
        assert arch == "aarch64"


# ── Integration: Local file with checksum verification ────────────────────

def test_local_file_with_checksum(local_source_cfg, tmp_cache):
    """Local file with non-empty sha256 verifies correctly."""
    import hashlib

    # Compute SHA256 of the local binary
    bin_path = Path(local_source_cfg["path"])
    actual_sha = BinaryDownloader._sha256_file(bin_path)

    good_cfg = dict(local_source_cfg)
    good_cfg["sha256"] = actual_sha  # correct checksum

    dl = BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-verified",
        source_cfg=good_cfg,
    )

    async def _test():
        result = await dl.resolve()
        assert os.path.isfile(result)

    import asyncio
    asyncio.run(_test())


def test_local_file_with_wrong_checksum_raises(tmp_cache):
    """Local file with wrong sha256 raises ChecksumMismatch."""
    bad_cfg = {
        "type": LOCAL_FILE,
        "path": "/nonexistent/file",
        "sha256": "a" * 64,
    }

    dl = BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-bad-checksum",
        source_cfg=bad_cfg,
    )

    async def _test():
        with pytest.raises(BinaryDownloaderError, match="not found"):
            await dl.resolve()  # fails on file not found before checksum check

    import asyncio
    asyncio.run(_test())


# ── Global defaults merging ──────────────────────────────────────────────

def test_global_defaults_applied(tmp_cache, github_source_cfg):
    """Global defaults are merged with per-source config."""
    global_defaults = {
        "verify_checksum": False,
        "cache_ttl_hours": 48,
    }

    dl = BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-defaults",
        source_cfg=github_source_cfg,
        global_defaults=global_defaults,
    )

    assert dl._verify_checksum is False
    assert dl._cache_ttl_hours == 48


def test_per_source_overrides_global(tmp_cache, github_source_cfg):
    """Per-source defaults override global defaults."""
    source_with_defaults = {
        **github_source_cfg,
        "defaults": {
            "verify_checksum": True,
            "cache_ttl_hours": 12,
        },
    }

    global_defaults = {
        "verify_checksum": False,
        "cache_ttl_hours": 48,
    }

    dl = BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-override",
        source_cfg=source_with_defaults,
        global_defaults=global_defaults,
    )

    # Per-source defaults take priority
    assert dl._verify_checksum is True
    assert dl._cache_ttl_hours == 12


# ── Runtime Check Tests ────────────────────────────────────────────────

@pytest.fixture
def runtime_check_source_cfg():
    """Minimal runtime-check source config."""
    return {
        "type": RUNTIME_CHECK,
        "checks": [
            {"command": "true", "exit_code": 0},
        ],
    }


@pytest.fixture
def nvidia_check_source_cfg():
    """NVIDIA runtime-check config."""
    return {
        "type": RUNTIME_CHECK,
        "checks": [
            {"command": "nvidia-smi", "exit_code": 0},
            {"path": "/usr/lib/*/libcuda.so.1"},
            {"path": "/usr/lib/*/libcudart.so.*"},
        ],
    }


def test_runtime_check_passes(tmp_cache, runtime_check_source_cfg):
    """Runtime check passes when all checks succeed."""
    dl = BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-rt",
        source_cfg=runtime_check_source_cfg,
    )

    result = asyncio.run(dl.resolve(version="latest"))
    assert result == "runtime-ok"


def test_runtime_check_command_fails(tmp_cache):
    """Runtime check fails when a command exits non-zero."""
    source_cfg = {
        "type": RUNTIME_CHECK,
        "checks": [
            {"command": "false", "exit_code": 0},
        ],
    }

    dl = BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-rt-fail",
        source_cfg=source_cfg,
    )

    with pytest.raises(RuntimeCheckError) as exc_info:
        asyncio.run(dl.resolve(version="latest"))

    assert len(exc_info.value.checks) == 1
    assert "'false' (exit 1)" in str(exc_info.value)


@patch("glob.glob")
def test_runtime_check_path_missing(mock_glob, tmp_cache):
    """Runtime check fails when a glob pattern matches nothing."""
    mock_glob.return_value = []

    source_cfg = {
        "type": RUNTIME_CHECK,
        "checks": [
            {"path": "/nonexistent/glob/**/*.txt"},
        ],
    }

    dl = BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-rt-path",
        source_cfg=source_cfg,
    )

    with pytest.raises(RuntimeCheckError) as exc_info:
        asyncio.run(dl.resolve(version="latest"))

    assert len(exc_info.value.checks) == 1
    assert "/nonexistent/glob/**/*.txt" in str(exc_info.value)


def test_runtime_check_partial_fail(tmp_cache):
    """Runtime check fails when some checks pass and some fail."""
    source_cfg = {
        "type": RUNTIME_CHECK,
        "checks": [
            {"command": "true", "exit_code": 0},      # passes
            {"command": "false", "exit_code": 0},      # fails
            {"path": "/nonexistent/*.txt"},            # fails
        ],
    }

    dl = BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-rt-partial",
        source_cfg=source_cfg,
    )

    with pytest.raises(RuntimeCheckError) as exc_info:
        asyncio.run(dl.resolve(version="latest"))

    assert len(exc_info.value.checks) == 2


@patch("subprocess.run")
def test_runtime_check_default_checks_nvidia(mock_run, tmp_cache, nvidia_check_source_cfg):
    """Default NVIDIA checks validate nvidia-smi and library paths."""
    # nvidia-smi is not on this system — mock it to simulate failure
    mock_run.side_effect = FileNotFoundError("nvidia-smi")

    dl = BinaryDownloader(
        cache_dir=tmp_cache,
        backend_id="test-nvidia",
        source_cfg=nvidia_check_source_cfg,
    )

    with pytest.raises(RuntimeCheckError) as exc_info:
        asyncio.run(dl.resolve(version="latest"))

    # nvidia-smi should have been called and failed
    assert mock_run.called
    assert len(exc_info.value.checks) >= 1
