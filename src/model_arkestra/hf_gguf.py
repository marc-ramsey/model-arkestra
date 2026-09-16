"""Resolve and download the exact GGUF file set for an HF ``repo:quant`` ref.

Ports the file-selection rules from llama.cpp's ``common/download.cpp`` so a
pull fetches only what the engine will actually load — primary model, optional
mmproj (vision) and mtp (speculative) sidecars, plus split shards — instead of
the whole repository.

Pure and reusable: no arkestra state, no processes, no ports. Downloads land in
the standard HF cache layout that ``llama-server -hf repo:quant`` reads from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Filename markers that disqualify a GGUF from being the *primary* model file.
_NON_MODEL_MARKERS = ("mmproj", "imatrix", "mtp-", "eagle3-", "dflash-", "dspark-")

_SPLIT_RE = re.compile(r"^(.+)-([0-9]{5})-of-([0-9]{5})$", re.IGNORECASE)
_TAG_RE = re.compile(r"[-.]([A-Z0-9_]+)$", re.IGNORECASE)


@dataclass
class GgufPlan:
    """The resolved set of files to fetch for one ``repo:quant`` ref."""

    repo: str
    tag: str = ""
    primary: str = ""          # primary model filename (first shard if split)
    shards: list[str] = field(default_factory=list)  # all split parts (or [primary])
    mmproj: str = ""           # vision projector, if resolved
    mtp: str = ""              # speculative sidecar, if resolved

    @property
    def files(self) -> list[str]:
        """Every distinct file to download."""
        out = list(self.shards or ([self.primary] if self.primary else []))
        for extra in (self.mmproj, self.mtp):
            if extra and extra not in out:
                out.append(extra)
        return out


def split_repo_tag(ref: str) -> tuple[str, str]:
    """Split ``repo[:tag]`` into ``(repo, tag)``. No colon → empty tag."""
    if ":" in ref:
        repo, _, tag = ref.partition(":")
        return repo, tag
    return ref, ""


def _is_model_file(path: str) -> bool:
    """True if path is a primary-model GGUF (not a sidecar/imatrix)."""
    if not path.endswith(".gguf"):
        return False
    name = path.rsplit("/", 1)[-1]
    return all(m not in name for m in _NON_MODEL_MARKERS)


def _split_info(path: str) -> tuple[str, str, int, int]:
    """Return ``(prefix, tag, index, count)`` for a (possibly sharded) GGUF.

    Handles both shard/tag orders seen on HF:
      ``prefix-TAG-NNNNN-of-MMMMM.gguf``  and  ``prefix-NNNNN-of-MMMMM-TAG.gguf``
    """
    base = path[:-5] if path.endswith(".gguf") else path
    # Strip a trailing quant tag (e.g. -Q8_0) so the shard regex can anchor.
    tag_m = _TAG_RE.search(base)
    tag = tag_m.group(1).upper() if tag_m else ""
    core = base[:tag_m.start()] if tag_m else base
    index, count = 1, 1
    m = _SPLIT_RE.match(core)
    if m:
        core, index, count = m.group(1), int(m.group(2)), int(m.group(3))
    return core, tag, index, count


def _quant_bits(tag: str) -> int:
    """Extract the leading number of a quant tag: ``Q4_0``→4, ``F16``→16.

    Mirrors llama.cpp's ``extract_quant_bits`` (``stoi`` stops at the first
    non-digit), so ``UD-Q4_K_XL`` → 4 and ``Q4_0`` → 4 (not 40).
    """
    pos = next((i for i, c in enumerate(tag) if c.isdigit()), -1)
    if pos < 0:
        return 0
    # Only the first contiguous digit run: stop at the first non-digit.
    first = ""
    for c in tag[pos:]:
        if c.isdigit():
            first += c
        else:
            break
    return int(first) if first else 0


def _find_primary(files: list[str], tag: str) -> str:
    """First model GGUF matching ``tag[.-]`` (case-insensitive); skip non-first shards."""
    if tag:
        pattern = re.compile(re.escape(tag) + r"[.-]", re.IGNORECASE)
        for f in files:
            if _is_model_file(f) and pattern.search(f):
                _, _, index, count = _split_info(f)
                if not (count > 1 and index != 1):
                    return f
    # No tag (or no match): fall back to first available model file.
    for f in files:
        if _is_model_file(f):
            _, _, index, count = _split_info(f)
            if not (count > 1 and index != 1):
                return f
    return ""


def _shards_for(files: list[str], primary: str) -> list[str]:
    """All split parts sharing the primary's prefix, or just [primary]."""
    prefix, _, index, count = _split_info(primary)
    if count <= 1:
        return [primary]
    out = []
    for f in files:
        p, _, i, c = _split_info(f)
        if c == count and p == prefix:
            out.append(f)
    return sorted(out, key=lambda x: _split_info(x)[2]) or [primary]


def _find_sibling(files: list[str], primary: str, keyword: str, tag: str = "") -> str:
    """Best sibling GGUF containing ``keyword`` (mmproj/mtp), llama.cpp rules.

    Prefers deepest shared dir prefix with the model, then an exact ``-TAG.``
    match, then closest quant-bit count to the model's bits.
    """
    primary_parts = primary.split("/")
    model_bits = _quant_bits(tag.upper()) if tag else _quant_bits(_split_info(primary)[1])
    tag_upper = tag.upper()

    best, best_depth, best_diff, best_exact, found = "", 0, 0, False, False
    for f in files:
        if not f.endswith(".gguf") or keyword not in f:
            continue
        sib_parts = f.split("/")
        # Must share a directory prefix with the model (same folder lineage).
        common = 0
        for a, b in zip(primary_parts, sib_parts):
            if a != b:
                break
            common += 1
        depth = len(sib_parts) - 1  # dir depth of the sibling
        bits = _quant_bits(_split_info(f)[1])
        diff = abs(bits - model_bits)
        exact = bool(tag_upper) and (f"-{tag_upper}." in f.upper())
        if (not found or depth > best_depth
                or (depth == best_depth and exact and not best_exact)
                or (depth == best_depth and exact == best_exact and diff < best_diff)):
            best, best_depth, best_diff, best_exact, found = f, depth, diff, exact, True
    return best


def resolve_plan(ref: str, files: list[str]) -> GgufPlan:
    """Build the download plan for ``ref`` given the repo's file listing.

    ``files`` is the full filename list from ``list_repo_files(repo)``.
    """
    repo, tag = split_repo_tag(ref)
    primary = _find_primary(files, tag)
    if not primary:
        return GgufPlan(repo=repo, tag=tag)
    shards = _shards_for(files, primary)
    mmproj = _find_sibling(files, primary, "mmproj", tag)
    mtp = _find_sibling(files, primary, "mtp-", tag)
    return GgufPlan(repo=repo, tag=tag, primary=primary, shards=shards,
                    mmproj=mmproj, mtp=mtp)


def download_plan(plan: GgufPlan, *, cache_dir: str | None = None,
                  token: str | None = None, progress_cb=None) -> list[str]:
    """Fetch every file in ``plan`` into the HF cache; return local paths.

    Uses ``hf_hub_download`` per file so only the planned files are fetched.
    ``progress_cb(filename, index)`` is called before each file starts (1-based).
    """
    from huggingface_hub import hf_hub_download

    total = len(plan.files)
    paths = []
    for i, filename in enumerate(plan.files, start=1):
        if progress_cb:
            progress_cb(filename, i)
        path = hf_hub_download(
            repo_id=plan.repo,
            filename=filename,
            cache_dir=cache_dir,
            token=token,
            force_download=False,
        )
        paths.append(path)
    return paths
