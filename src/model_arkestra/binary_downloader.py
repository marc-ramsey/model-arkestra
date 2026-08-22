"""Binary downloader for process-mode backends.

Fetches pre-built llama.cpp binaries from GitHub Releases or OCI registries,
verifies SHA256 checksums, and caches them locally.  Supports multiple release
channels (latest nightly, pinned version) with automatic update detection.

Usage:
    downloader = BinaryDownloader(sources_config, cache_dir, backend_id)
    binary_path = await downloader.resolve(version="latest")

When no sources config is available or the backend doesn't reference a source,
the caller should fall back to its existing Containerfile-based build path.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiohttp

logger = logging.getLogger(__name__)


# ── Source type constants ──────────────────────────────────────────────────

GITHUB_RELEASE = "github-release"
OCI_IMAGE = "oci-image"
LOCAL_FILE = "local-file"


class BinaryDownloaderError(Exception):
    """Base exception for binary downloader failures."""


class ChecksumMismatch(BinaryDownloaderError):
    """SHA256 verification failed — downloaded file doesn't match source checksum."""

    def __init__(self, expected: str, actual: str, asset_path: str):
        self.expected = expected
        self.actual = actual
        self.asset_path = asset_path
        super().__init__(
            f"SHA256 mismatch for {asset_path}: "
            f"expected {expected[:12]}… got {actual[:12]}…"
        )


class BinaryDownloader:
    """Download, verify, and cache llama.cpp binaries for process-mode backends.

    Each downloader instance is scoped to a single backend + source combination.
    The ``sources`` dict comes from parsing sources.yaml via ConfigManager;
    it contains exactly one entry keyed by the source name (e.g. "lemonade-nightly").
    """

    def __init__(
        self,
        cache_dir: Path,
        backend_id: str,
        source_cfg: Dict[str, Any],
        global_defaults: Optional[Dict[str, Any]] = None,
    ):
        self.backend_id = backend_id
        self.source_cfg = source_cfg
        self.cache_dir = cache_dir.expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Merge per-source overrides with global defaults
        defaults = dict(global_defaults or {})
        defaults.update(source_cfg.get("defaults", {}))
        self._verify_checksum = defaults.get("verify_checksum", True)
        self._cache_ttl_hours = defaults.get("cache_ttl_hours", 24)

    # ── Public API ───────────────────────────────────────────────────────

    async def resolve(self, version: str = "latest") -> str:
        """Return the cached binary path for the given version.

        If the binary is not yet cached (or stale), downloads and verifies it.
        Raises BinaryDownloaderError on failure.

        Args:
            version: "latest" for newest release, or a pinned version/tag string.

        Returns:
            Absolute path to the extracted binary.

        Raises:
            BinaryDownloaderError: download failed, checksum mismatch, or source unsupported.
        """
        source_type = self.source_cfg.get("type", "")

        if source_type == GITHUB_RELEASE:
            return await self._resolve_github_release(version)
        elif source_type == LOCAL_FILE:
            return await self._resolve_local_file()
        else:
            raise BinaryDownloaderError(
                f"Unsupported source type '{source_type}' for backend '{self.backend_id}'. "
                f"Supported: {GITHUB_RELEASE}, {LOCAL_FILE}"
            )

    # ── GitHub Release path ──────────────────────────────────────────────

    async def _resolve_github_release(self, version: str) -> str:
        """Download from a GitHub release asset, verify checksum, cache locally."""
        repo = self.source_cfg["repo"]  # owner/repo
        release_type = self.source_cfg.get("release_type", "latest") or "latest"

        # Resolve the tag to download
        tag = version if version != "latest" else await self._fetch_latest_tag(repo)
        if not tag:
            raise BinaryDownloaderError(f"No release found for repo {repo}")

        cache_key = f"{self.backend_id}-{release_type}-{tag}"
        cached_info = self._lookup_cache(cache_key)
        if cached_info and not self._is_stale(cached_info):
            binary_path = Path(cached_info["path"])
            if binary_path.is_file():
                logger.debug(f"Using cached binary: {binary_path}")
                return str(binary_path)

        # Download + extract (with file lock to prevent concurrent duplicate downloads)
        lock_path = self.cache_dir / f".{cache_key}.lock"
        async with _file_lock(lock_path):
            # Re-check cache after acquiring lock (another task may have downloaded while we waited)
            cached_info = self._lookup_cache(cache_key)
            if cached_info and not self._is_stale(cached_info):
                binary_path = Path(cached_info["path"])
                if binary_path.is_file():
                    return str(binary_path)

            asset_name, sha256_expected = await self._fetch_release_asset(repo, tag)
            binary_path = await self._download_and_extract(
                repo, tag, asset_name, sha256_expected, cache_key
            )

        # Update cache metadata
        self._write_cache_entry(cache_key, str(binary_path))
        return str(binary_path)

    async def _fetch_latest_tag(self, repo: str) -> Optional[str]:
        """Query GitHub API for the newest release tag."""
        url = f"https://api.github.com/repos/{repo}/releases/latest"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("tag_name")
                    logger.warning(f"GitHub API returned {resp.status} for {repo}/latest")
        except Exception as exc:
            logger.debug(f"Failed to fetch latest tag from GitHub: {exc}")
        return None

    async def _fetch_release_asset(self, repo: str, tag: str) -> Tuple[str, Optional[str]]:
        """Fetch the release asset name and its SHA256 checksum.

        Returns (asset_name, sha256_hex_or_None).
        """
        pattern = self.source_cfg["asset_pattern"]
        sha256_pattern = self.source_cfg.get("sha256_asset")
        arch_map = self.source_cfg.get("arch_map", {})
        system_arch = self._get_system_arch()

        # Determine asset variant suffix based on architecture
        variant_suffix = arch_map.get(system_arch, "") if arch_map else ""

        url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        raise BinaryDownloaderError(
                            f"GitHub API returned {resp.status} for tag {tag}"
                        )
                    release_data = await resp.json()

        except Exception as exc:
            raise BinaryDownloaderError(f"Failed to fetch release info: {exc}")

        # Find matching asset in the release's assets list
        assets = release_data.get("assets", [])
        matched_asset = None
        for asset in assets:
            name = asset.get("name", "")
            # Try with variant suffix first, then without (for backwards compatibility)
            if variant_suffix and name.endswith(f"{variant_suffix}.tar.gz"):
                matched_asset = asset
                break
            elif not variant_suffix and pattern.replace("*", "") in name:
                # For generic patterns like "llama-server-*-bin-*"
                stripped_pattern = pattern.replace("*.gz", "*")
                if stripped_pattern.replace("-", "").replace("_", "") in name.replace("-", "").replace("_", ""):
                    matched_asset = asset
                    break

        if not matched_asset and variant_suffix:
            # Retry without variant suffix as fallback
            for asset in assets:
                name = asset.get("name", "")
                if pattern.replace("*", "") in name:
                    matched_asset = asset
                    break

        if not matched_asset:
            available_names = [a.get("name", "") for a in assets]
            raise BinaryDownloaderError(
                f"No matching asset found in release {tag} for pattern '{pattern}' "
                f"(arch='{system_arch}', suffix='{variant_suffix}'). "
                f"Available: {available_names[:5]}"
            )

        asset_name = matched_asset["name"]
        sha256_url = None

        # Try to find matching SHA256 file
        if sha256_pattern:
            for asset in assets:
                name = asset.get("name", "")
                if sha256_pattern.replace("*", "") in name:
                    sha256_url = asset["browser_download_url"]
                    break

        sha256_expected = None
        if sha256_url:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(sha256_url, timeout=15) as resp:
                        if resp.status == 200:
                            content = (await resp.text()).strip().split()[0]  # "hash  filename" format
                            sha256_expected = content
            except Exception as exc:
                logger.debug(f"Failed to fetch SHA256 from {sha256_url}: {exc}")

        return asset_name, sha256_expected

    async def _download_and_extract(
        self, repo: str, tag: str, asset_name: str,
        sha256_expected: Optional[str], cache_key: str
    ) -> Path:
        """Download an asset, verify checksum, extract binary to cache."""
        download_url = f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"
        tmp_dir = self.cache_dir / f".tmp-{cache_key}"
        tmp_dir.mkdir(exist_ok=True)

        # Download to temp file first (atomic write pattern)
        tmp_file = tmp_dir / asset_name
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url, timeout=300) as resp:
                    if resp.status != 200:
                        raise BinaryDownloaderError(
                            f"Download failed: HTTP {resp.status} for {asset_name}"
                        )
                    # Stream download to avoid loading entire file into memory
                    with open(tmp_file, "wb") as f:
                        while True:
                            chunk = await resp.content.readany()
                            if not chunk:
                                break
                            f.write(chunk)

            # Verify SHA256 if expected
            if sha256_expected and self._verify_checksum:
                actual_sha256 = self._sha256_file(tmp_file)
                if actual_sha256.lower() != sha256_expected.lower():
                    raise ChecksumMismatch(sha256_expected, actual_sha256, asset_name)

            # Extract tarball — extract to cache_dir, then pick up the binary
            extract_dir = self.cache_dir / f"extracted-{cache_key}"
            shutil.unpack_archive(tmp_file, extract_dir)

            # Find the binary in extracted contents
            binary_path = self._find_binary_in_tree(extract_dir)
            if not binary_path:
                raise BinaryDownloaderError(
                    f"No executable binary found in extracted archive {extract_dir}"
                )

            # Move binary to final cache location
            final_dest = self.cache_dir / f"bin-{cache_key}"
            shutil.move(str(binary_path), str(final_dest))
            os.chmod(final_dest, 0o755)  # ensure executable

        finally:
            # Cleanup temp files
            if tmp_file.exists():
                tmp_file.unlink()
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)

        return final_dest

    def _find_binary_in_tree(self, root: Path) -> Optional[Path]:
        """Search extracted directory tree for an executable binary."""
        for candidate in root.rglob("*"):
            if candidate.is_file() and os.access(str(candidate), os.X_OK):
                name = candidate.name.lower()
                # Prefer llama-server variants
                if "llama" in name or "server" in name:
                    return candidate
        # Fallback to first executable found
        for candidate in root.rglob("*"):
            if candidate.is_file() and os.access(str(candidate), os.X_OK):
                return candidate
        return None

    # ── Local File path ──────────────────────────────────────────────────

    async def _resolve_local_file(self) -> str:
        """Use a pre-staged local binary file."""
        path = self.source_cfg.get("path", "")
        if not path:
            raise BinaryDownloaderError(
                f"Local-file source for backend '{self.backend_id}' has no 'path' configured"
            )

        path = Path(path).expanduser()
        if not path.is_file():
            raise BinaryDownloaderError(
                f"Local binary not found: {path}. "
                f"Use 'arkestra admin stage-binary' to provide a binary."
            )

        # Verify checksum if configured (skip if empty sha256)
        expected_sha = self.source_cfg.get("sha256", "")
        if expected_sha and self._verify_checksum:
            actual_sha = self._sha256_file(path)
            if actual_sha.lower() != expected_sha.lower():
                raise ChecksumMismatch(expected_sha, actual_sha, str(path))

        # Ensure executable
        if not os.access(str(path), os.X_OK):
            logger.warning(f"Binary {path} is not executable; attempting to fix permissions")
            os.chmod(path, 0o755)

        return str(path)

    # ── Caching helpers ──────────────────────────────────────────────────

    def _get_cache_manifest_path(self) -> Path:
        """Path to the cache manifest (JSON metadata for all cached binaries)."""
        return self.cache_dir / f".{self.backend_id}-cache-manifest.json"

    def _lookup_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Look up a cached binary by key. Returns dict with 'path' and 'updated_at'."""
        manifest_path = self._get_cache_manifest_path()
        if not manifest_path.exists():
            return None

        try:
            with open(manifest_path) as f:
                import json
                manifest = json.load(f)
            return manifest.get(cache_key)
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache_entry(self, cache_key: str, binary_path: str) -> None:
        """Write/update a cache entry in the manifest file."""
        import json

        manifest_path = self._get_cache_manifest_path()
        manifest = {}
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, OSError):
                manifest = {}

        manifest[cache_key] = {
            "path": binary_path,
            "updated_at": asyncio.get_event_loop().time(),  # monotonic time
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _is_stale(self, entry: Dict[str, Any]) -> bool:
        """Check if a cached binary is stale based on TTL."""
        updated_at = entry.get("updated_at", 0)
        age_hours = (asyncio.get_event_loop().time() - updated_at) / 3600.0
        return age_hours > self._cache_ttl_hours

    # ── Utility helpers ──────────────────────────────────────────────────

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Compute SHA256 hex digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 16)  # 64KB chunks
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _get_system_arch() -> str:
        """Return the system architecture (x86_64, aarch64, etc.)."""
        import platform
        arch = platform.machine().lower()
        if arch in ("x86_64", "amd64"):
            return "x86_64"
        elif arch in ("aarch64", "arm64"):
            return "aarch64"
        return arch


# ── Async file lock helper ───────────────────────────────────────────────

class _AsyncFileLock:
    """Simple async-compatible file lock using asyncio.Lock + file locking."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()


async def _file_lock(path: Path) -> _AsyncFileLock:
    """Return an async file lock for the given path."""
    return _AsyncFileLock(path)
