"""Backend binary slots: install llama.cpp binaries per backend ID.

Reads provisioning specs from the merged backend config
(shipped backends.yaml + config.yaml overlay) — every backend with a
``source:`` entry is a known binary:

  - ``source: {type: remote, repo: ..., asset: llama-{tag}-...}``
      fetched from GitHub releases into ``<bin_dir>/<backend-id>/``
  - ``source: {type: local, path: /build/dir}``
      the build dir itself — verified, sha recorded, never fetched

Swaps are atomic: extract to .new, mv over the live dir. A running
llama-server keeps its open inode; the next launch picks up the new binary.
State (installed tag, sha256, pin) lives in ~/.local/arkestra/bin-state/state.json
(override the directory with ARKESTRA_BIN_STATE, e.g. in tests).
keyed by backend ID.

CLI surface lives in ``cli.py`` (``arkestra bin ...``); this module is the
library: verify/fetch/pin/update primitives.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml
from importlib.resources import files as _files

from model_arkestra.common import resolve_config_path
from model_arkestra.config_manager import ModelConfigManager

def state_dir() -> Path:
    """State dir; ARKESTRA_BIN_STATE overrides (tests must never touch real state)."""
    return Path(os.path.expanduser(
        os.environ.get("ARKESTRA_BIN_STATE",
                       "~/.local/arkestra/bin-state")))


# Local sources are verified, never fetched — tag is a fixed marker.
LOCAL_TAG = "local"


class BinError(Exception):
    pass


# ── config & state ─────────────────────────────────────────────────────────

def merged_config() -> Dict[str, Any]:
    """Merged config (shipped backends.yaml base + config.yaml overlay)."""
    config_path = resolve_config_path(None)
    try:
        base_text = (_files("model_arkestra.data") / "backends.yaml").read_text()
    except Exception:
        base_text = ""
    if not config_path.is_file():
        return yaml.safe_load(base_text) or {}
    try:
        cm = ModelConfigManager(str(config_path))
        cm.data = yaml.safe_load(base_text) or {}
        cm.merge(yaml.safe_load(config_path.read_text()) or {})
        return cm.data or {}
    except Exception:
        # Malformed overlay (e.g. a placeholder config mid-init) — use the
        # shipped base alone rather than crashing the caller.
        return yaml.safe_load(base_text) or {}


def merged_backends() -> Dict[str, Any]:
    backends = merged_config().get("backends") or {}
    return backends if isinstance(backends, dict) else {}


def referenced_backend_ids() -> set:
    """Backend IDs actually in play: default + every checkpoint/model backend."""
    cfg = merged_config()
    ids: set = set()
    be = cfg.get("backends") or {}
    if isinstance(be, dict) and be.get("default"):
        ids.add(str(be["default"]))
    for section in ("checkpoints", "models"):
        for entry in (cfg.get(section) or {}).values():
            if isinstance(entry, dict) and entry.get("backend"):
                ids.add(str(entry["backend"]))
    return ids


def sources() -> Dict[str, Dict[str, Any]]:
    """backend-id -> source dict, for every backend declaring a ``source:``."""
    out = {}
    for bid, be in merged_backends().items():
        if bid == "default" or not isinstance(be, dict):
            continue
        src = be.get("source")
        if isinstance(src, dict) and src:
            out[str(bid)] = src
    return out


def load_state() -> dict:
    f = state_dir() / "state.json"
    if f.is_file():
        return json.loads(f.read_text() or "{}")
    return {}


def save_state(state: dict) -> None:
    d = state_dir()
    d.mkdir(parents=True, exist_ok=True)
    f = d / "state.json"
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(f)


def bin_root() -> Path:
    return Path(os.path.expanduser(os.environ.get("ARKESTRA_BIN_DIR", "~/.local/arkestra/bin")))


def slot_path(bid: str, src: Dict[str, Any]) -> Path:
    """Slot dir for a backend: local -> its build dir, remote -> bin_root/<id>."""
    if src.get("type") == "local":
        return Path(os.path.expanduser(str(src.get("path") or "")))
    return bin_root() / bid


def require_source(bid: str) -> Dict[str, Any]:
    srcs = sources()
    if bid not in srcs:
        known = ", ".join(sorted(srcs))
        raise BinError(f"unknown binary backend: {bid} (known: {known})")
    return srcs[bid]


# ── fetch logic ────────────────────────────────────────────────────────────

def latest_tag(repo: str, asset_tpl: str) -> str | None:
    """Newest release whose assets include the template rendered for its tag."""
    url = f"https://api.github.com/repos/{repo}/releases?per_page=30"
    req = urllib.request.Request(url, headers={"User-Agent": "arkestra-bin", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        releases = json.loads(resp.read())
    for rel in releases:
        tag = rel.get("tag_name") or ""
        wanted = asset_tpl.format(tag=tag)
        names = [a.get("name", "") for a in rel.get("assets", [])]
        if wanted in names:
            return tag
    return None


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> str:
    """Stream url to dest; return sha256 hex of the file."""
    req = urllib.request.Request(url, headers={"User-Agent": "arkestra-bin"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise BinError(f"download failed ({e}): {url}") from None
    return file_sha(dest)


def extract(archive: Path, dest: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest, filter="data")
    else:
        raise BinError(f"unsupported archive type: {archive.name}")


def flatten_if_nested(dest: Path) -> None:
    """Archives may nest one level (llama-<tag>-.../); hoist contents up."""
    if (dest / "llama-server").exists():
        return
    nested = next(dest.glob("*/llama-server"), None)
    if nested is None:
        raise BinError(f"no llama-server found in extracted archive under {dest}")
    for child in nested.parent.iterdir():
        shutil.move(str(child), str(dest / child.name))
    nested.parent.rmdir()


def fetch_one(bid: str, src: Dict[str, Any], state: dict,
              latest: bool = False, fetch: bool = True) -> str:
    """Ensure the slot for one backend. Returns 'ok' | 'fetched' | 'missing'.

    latest=True tracks the newest release; fetch=False only verifies —
    a missing slot comes back 'missing' without touching the network.
    """
    if src.get("type") == "local":
        return _verify_local(bid, src, state)

    repo = src.get("repo") or ""
    asset_tpl = src.get("asset") or ""
    if not repo or not asset_tpl:
        raise BinError(f"backend '{bid}' needs 'repo' and 'asset' in source")

    slot = slot_path(bid, src)
    entry = state.get(bid) or {}
    binary = slot / "llama-server"
    tag = entry.get("pinned")
    if not tag and not latest and binary.is_file():
        return "ok"
    if not tag:
        if not fetch:
            return "missing"
        tag = latest_tag(repo, asset_tpl)
        if not tag:
            raise BinError(f"backend '{bid}': no release found with matching asset")

    if entry.get("tag") == tag and binary.is_file():
        return "ok"
    if not fetch:
        return "missing"

    asset_name = asset_tpl.format(tag=tag)
    url = f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"
    print(f"  {bid}: fetching {tag}", flush=True)

    slot.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".zip" if asset_name.endswith(".zip") else ".tar.gz"
    tmpfile = slot.parent / f".{bid}.download.{os.getpid()}{suffix}"
    sha = download(url, tmpfile)

    new = slot.with_name(f"{slot.name}.new.{os.getpid()}")
    old = slot.with_name(f"{slot.name}.old.{os.getpid()}")
    try:
        shutil.rmtree(new, ignore_errors=True)
        new.mkdir(parents=True)
        extract(tmpfile, new)
        flatten_if_nested(new)
        os.chmod(new / "llama-server", 0o755)

        # Atomic swap over the live slot.
        if old.exists():
            shutil.rmtree(old)
        if slot.exists():
            slot.rename(old)
        new.rename(slot)
        shutil.rmtree(old, ignore_errors=True)
    finally:
        tmpfile.unlink(missing_ok=True)
        shutil.rmtree(new, ignore_errors=True)

    state[bid] = {
        "tag": tag,
        "sha256": sha,
        "pinned": (state.get(bid) or {}).get("pinned"),
        "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    save_state(state)
    print(f"  {bid}: installed {tag} -> {slot / 'llama-server'}")
    return "fetched"


def _verify_local(bid: str, src: Dict[str, Any], state: dict) -> str:
    """Local build: verify path, record sha. Never downloads."""
    slot = slot_path(bid, src)
    binary = slot / "llama-server"
    if not binary.is_file():
        return "missing"

    sha = file_sha(binary)
    entry = state.get(bid) or {}
    if entry.get("sha256") != sha or entry.get("tag") != LOCAL_TAG:
        state[bid] = {
            "tag": LOCAL_TAG,
            "sha256": sha,
            "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        save_state(state)
        return "ok"
    return "ok"


def ensure(ids: set | None = None, fetch: bool = False) -> int:
    """Verify binary slots for ids; fetch missing remote slots when fetch.

    ids defaults to the referenced backend ids — backends no model uses
    are never touched. Returns the number of unusable slots; problems
    are printed as [bin] WARNING lines.
    """
    state = load_state()
    srcs = sources()
    if ids is None:
        ids = referenced_backend_ids()
    bad = 0
    for bid in sorted(ids):
        src = srcs.get(bid)
        if not src:
            continue  # backend without a source (e.g. onnx) — nothing to do
        try:
            res = fetch_one(bid, src, state, fetch=fetch)
        except BinError as e:
            print(f"[bin] WARNING: {e}", flush=True)
            bad += 1
            continue
        if res == "missing":
            print(f"[bin] WARNING: backend '{bid}' binary missing — "
                  f"run: arkestra bin fetch {bid}", flush=True)
            bad += 1
    return bad


def update(all_sources: bool = False) -> int:
    """Upgrade backends to the newest release (cron-friendly).

    Referenced backends by default; all_sources covers every source.
    Returns the number of failures, printed as [bin] ERROR lines.
    """
    state = load_state()
    srcs = sources()
    if all_sources:
        ids = sorted(srcs)
    else:
        ids = sorted(set(srcs) & referenced_backend_ids())
    bad = 0
    for bid in ids:
        try:
            fetch_one(bid, srcs[bid], state, latest=True)
        except BinError as e:
            print(f"[bin] ERROR: {e}", file=sys.stderr, flush=True)
            bad += 1
    return bad


# ── query / pin helpers (used by `arkestra bin`) ───────────────────────────

def list_text() -> str:
    state = load_state()
    srcs = sources()
    rows = [f"{'BACKEND':<18} {'TYPE':<8} {'TAG':<10} {'PIN':<12} SHA256(12)"]
    for bid in sorted(srcs):
        entry = state.get(bid) or {}
        typ = srcs[bid].get("type", "remote")
        tag = entry.get("tag", "-")
        pin = entry.get("pinned") or "no"
        if typ == "local":
            pin = "n/a"
        sha = (entry.get("sha256") or "")[:12] or "-"
        rows.append(f"{bid:<18} {typ:<8} {tag:<10} {pin:<12} {sha}")
    return "\n".join(rows)


def pin(bid: str, tag: str) -> None:
    src = require_source(bid)
    if src.get("type") == "local":
        raise BinError(f"backend '{bid}' is local — nothing to pin")
    state = load_state()
    entry = state.get(bid) or {}
    entry["pinned"] = tag
    state[bid] = entry
    save_state(state)
    print(f"Pinned {bid} to {tag} (next fetch will install it)")


def unpin(bid: str) -> None:
    src = require_source(bid)
    if src.get("type") == "local":
        raise BinError(f"backend '{bid}' is local — nothing to unpin")
    state = load_state()
    entry = state.get(bid) or {}
    entry.pop("pinned", None)
    state[bid] = entry
    save_state(state)
    print(f"Unpinned {bid} (tracking latest again)")
