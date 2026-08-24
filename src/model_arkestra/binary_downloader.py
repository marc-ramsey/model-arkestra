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
import fnmatch
import hashlib
import logging
import os
import shutil
import subprocess
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
RUNTIME_CHECK = "runtime-check"


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


class RuntimeCheckError(BinaryDownloaderError):
    """Runtime prerequisites not satisfied — backend cannot run in process mode."""

    def __init__(self, checks: List[str]):
        self.checks = checks
        super().__init__(
            f"Runtime check failed for backend. Missing: {', '.join(checks)}. "
            "Try container mode or install the required packages."
        )


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
        elif source_type == RUNTIME_CHECK:
            await self._resolve_runtime_check()
            return "runtime-ok"
        elif source_type == OCI_IMAGE:
            return await self._resolve_oci_image(version)
        else:
            raise BinaryDownloaderError(
                f"Unsupported source type '{source_type}' for backend '{self.backend_id}'. "
                f"Supported: {GITHUB_RELEASE}, {LOCAL_FILE}, {RUNTIME_CHECK}, {OCI_IMAGE}"
            )

    # ── GitHub Release path ──────────────────────────────────────────────

    async def _resolve_runtime_check(self) -> None:
        """Verify CUDA/ROCm system prerequisites for process-mode backends.

        This is a pre-flight check — no download or caching occurs. It validates
        that the required runtime libraries and tools are available on the system.
        If all checks pass, the backend can run in process mode (user must provide
        the binary separately or use container mode).

        Raises:
            RuntimeCheckError: one or more prerequisite checks failed.
        """
        checks = self.source_cfg.get("checks", [])
        if not checks:
            # Default: check for nvidia-smi + CUDA runtime libraries
            checks = [
                {"command": "nvidia-smi", "exit_code": 0},
                {"path": "/usr/lib/*/libcuda.so.1"},
                {"path": "/usr/lib/*/libcudart.so.*"},
            ]

        failed: List[str] = []

        for check in checks:
            if "command" in check:
                cmd = check["command"]
                expected_code = check.get("exit_code", 0)
                try:
                    result = subprocess.run(
                        [cmd], capture_output=True, timeout=10
                    )
                    if result.returncode != expected_code:
                        failed.append(f"'{cmd}' (exit {result.returncode})")
                except FileNotFoundError:
                    failed.append(f"'{cmd}' not found")
            elif "path" in check:
                import glob
                pattern = check["path"]
                matches = glob.glob(pattern)
                if not matches:
                    failed.append(f"{pattern}")

        if failed:
            raise RuntimeCheckError(failed)

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
        """Query GitHub API for the newest release tag with binary assets.

        First tries ``/releases/latest`` (stable). If that has no real binary
        assets, falls back to the latest pre-release with at least one .tar.gz
        or .zip asset.
        """
        _BINARY_EXTENSIONS = ('.tar.gz', '.tgz', '.tar.bz2', '.tar.xz', '.zip', '.exe')

        def has_binary_assets(assets: list) -> bool:
            """True if any asset is an archive/binary (not just a text file)."""
            for a in assets:
                name = a.get("name", "")
                if name.endswith(_BINARY_EXTENSIONS):
                    return True
            return False

        # 1. Try stable release first
        url = f"https://api.github.com/repos/{repo}/releases/latest"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tag = data.get("tag_name")
                        assets = data.get("assets", [])
                        if tag and has_binary_assets(assets):
                            return tag
        except Exception as exc:
            logger.debug(f"Failed to fetch latest stable release: {exc}")

        # 2. Fallback: paginate releases looking for one with binary assets
        try:
            url = f"https://api.github.com/repos/{repo}/releases?per_page=30"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status == 200:
                        releases = await resp.json()
                        for rel in releases:
                            if has_binary_assets(rel.get("assets", [])):
                                return rel.get("tag_name")
        except Exception as exc:
            logger.debug(f"Failed to fetch releases list: {exc}")
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
            # Try with variant suffix first (for backward compat)
            if variant_suffix and name.endswith(f"{variant_suffix}.tar.gz"):
                matched_asset = asset
                break
            # Use fnmatch for proper glob matching (handles * in patterns)
            elif fnmatch.fnmatch(name, pattern):
                matched_asset = asset
                break

        if not matched_asset and variant_suffix:
            # Retry without variant suffix as fallback
            for asset in assets:
                name = asset.get("name", "")
                if fnmatch.fnmatch(name, pattern):
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
                if fnmatch.fnmatch(name, sha256_pattern):
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
            # For multi-binary builds (llama.cpp), ALL files including .so
            # shared libraries must be kept together in the same directory.
            final_dest = self.cache_dir / f"bin-{cache_key}"
            shutil.copytree(str(extract_dir), str(final_dest), dirs_exist_ok=True)

            # Rename the found binary to match cache-key naming convention
            renamed = final_dest / f"bin-{cache_key}"
            os.rename(str(binary_path.resolve()), str(renamed))
            os.chmod(str(renamed), 0o755)  # ensure executable
            result_path = str(renamed)

        finally:
            # Cleanup temp files
            if tmp_file.exists():
                tmp_file.unlink()
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)

        return result_path

    def _find_binary_in_tree(self, root: Path) -> Optional[Path]:
        """Locate the 'llama-server' executable in the extracted tree.

        Searches for a file named exactly ``llama-server`` (including nested
        subdirectories) that is either an ELF binary or a script with shebang.
        Returns None if not found.
        """
        candidates = list(root.rglob("llama-server"))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            # Check shebang (scripts)
            try:
                with open(candidate, "rb") as f:
                    header = f.read(2)
                    if header == b"#!":
                        os.chmod(str(candidate), 0o755)
                        return candidate
            except OSError:
                pass
            # Check ELF magic (compiled binaries)
            if self._is_elf(candidate):
                os.chmod(str(candidate), 0o755)
                return candidate
        return None

    @staticmethod
    def _is_elf(path: Path) -> bool:
        """Check if a file is an ELF executable by reading its magic bytes."""
        try:
            with open(path, "rb") as f:
                return f.read(4) == b"\x7fELF"
        except OSError:
            return False

    # ── OCI Image path ────────────────────────────────────────────────

    async def _resolve_oci_image(self, tag: str) -> str:
        """Resolve an OCI container image via podman/docker.

        Pulls the image if not present locally; returns the resolved image ref.
        """
        repo = self.source_cfg.get("repo", "")  # e.g. "docker.io/kyuz0/amd-strix-halo-toolboxes"
        release_type = self.source_cfg.get("release_type", "latest") or "latest"

        if not repo:
            raise BinaryDownloaderError(
                f"OCI-image source for backend '{self.backend_id}' has no 'repo' configured"
            )

        # Determine tag to use
        image_tag = tag if tag != "latest" else release_type
        cache_key = f"{self.backend_id}-{release_type}-{image_tag}"

        # Check if already cached (image exists locally)
        cached_info = self._lookup_cache(cache_key)
        if cached_info and not self._is_stale(cached_info):
            logger.debug(f"Using cached OCI image: {cached_info.get('ref', repo)}:{image_tag}")
            return f"{repo}:{image_tag}"

        # Pre-pull the image (with file lock to prevent concurrent pulls)
        lock_path = self.cache_dir / f".{cache_key}-pull.lock"
        async with _file_lock(lock_path):
            # Re-check after acquiring lock
            cached_info = self._lookup_cache(cache_key)
            if cached_info and not self._is_stale(cached_info):
                return f"{repo}:{image_tag}"

            # Pull the image via podman (prefer podman, fall back to docker)
            pull_cmd = None
            for cmd_name in ("podman", "docker"):
                try:
                    result = subprocess.run(
                        [cmd_name, "--version"],
                        capture_output=True, timeout=5
                    )
                    if result.returncode == 0:
                        pull_cmd = cmd_name
                        break
                except FileNotFoundError:
                    pass

            if pull_cmd is None:
                raise BinaryDownloaderError(
                    f"Neither podman nor docker found on system for OCI image source."
                )

            full_image = f"{repo}:{image_tag}"
            logger.info(f"Pulling OCI image {full_image} via {pull_cmd}")
            proc = await asyncio.create_subprocess_exec(
                pull_cmd, "pull", full_image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                err_msg = stderr.decode().strip() or f"exit code {proc.returncode}"
                raise BinaryDownloaderError(
                    f"Failed to pull OCI image {full_image}: {err_msg}"
                )

            logger.info(f"OCI image pulled successfully: {full_image}")

        # Cache the reference
        self._write_cache_entry(cache_key, {
            "ref": full_image,
            "updated_at": asyncio.get_event_loop().time(),
        })
        return full_image

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
        """Look up a cached entry by key. Returns dict with 'path' (binaries) or
        'ref' (OCI images), and 'updated_at'.
        """
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
        """Check if a cached entry is stale based on TTL. Works with both
        binary entries (string path) and OCI entries (dict with 'ref')."""
        if isinstance(entry, str):
            # Legacy format: string binary path — use file mtime as fallback
            p = Path(entry)
            if p.is_file() and p.stat().st_mtime:
                age_hours = (asyncio.get_event_loop().time() - p.stat().st_mtime) / 3600.0
                return age_hours > self._cache_ttl_hours
            return False
        updated_at = entry.get("updated_at", 0)
        if not updated_at:
            return True  # no timestamp — assume stale (no valid cache)
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


def _file_lock(path: Path) -> _AsyncFileLock:
    """Return an async file lock for the given path."""
    return _AsyncFileLock(path)
