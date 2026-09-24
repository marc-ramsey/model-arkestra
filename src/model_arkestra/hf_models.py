"""Plan and download the exact file set for an HF repo ref, any model format.

Successor to hf_gguf: the GGUF rules (ported from llama.cpp's
``common/download.cpp``) now live here alongside generic planners, so a pull
fetches only what a model will actually load for GGUF, ONNX, or an unknown
format (whole-repo snapshot).

    ref ──► detect_format ──► plan ──► HfPlan ──► download_plan ──► HF cache

Formats:
    gguf          primary model + mmproj/mtp sidecars + split shards
    onnx-whisper  every ``*.onnx`` + tokenizer files (encoder + decoder)
    onnx-embed    every ``*.onnx`` + tokenizer files
    onnx-tts      every ``*.onnx`` (+ voice ``*.bin`` sidecars)
    onnx          single ``*.onnx`` named in the ref (``repo:file.onnx``)
    snapshot      the whole repo minus scaffolding (unknown formats)

Pure and reusable: no arkestra state, no processes, no ports. Downloads land
in the standard HF cache layout the runners read from.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field

# ── File-selection constants ─────────────────────────────────────────────

#: Filenames never part of a model download (repo scaffolding).
_SNAPSHOT_SKIP = ("readme", "license", "license.md", "citations",
                  ".gitattributes", ".gitignore", ".gitkeep")

#: Files that carry a tokenizer (every match in the listing is planned).
_TOKENIZER_CANDIDATES = ("tokenizer.json", "tokenizer.model", "vocab.json",
                         "special_tokens.json", "tokenizer_config.json")

#: HF dataset directory — never part of a model download.
_DATA_DIR = "data/"

# ── GGUF selection (ported from llama.cpp common/download.cpp) ───────────

# Filename markers that disqualify a GGUF from being the *primary* model file.
_NON_MODEL_MARKERS = ("mmproj", "imatrix", "mtp-", "eagle3-", "dflash-", "dspark-")

_SPLIT_RE = re.compile(r"^(.+)-([0-9]{5})-of-([0-9]{5})$", re.IGNORECASE)
_TAG_RE = re.compile(r"[-.]([A-Z0-9_]+)$", re.IGNORECASE)


# ── Plan ─────────────────────────────────────────────────────────────────

@dataclass
class HfPlan:
    """The resolved set of files to fetch for one repo ref.

    ``files`` is always the complete fetch list. The GGUF fields
    (primary/shards/mmproj/mtp) carry the selection breakdown for logging
    and are empty for non-GGUF formats.
    """

    repo: str
    format: str = ""       # gguf | onnx-whisper | onnx-embed | onnx-tts | onnx | snapshot
    files: list[str] = field(default_factory=list)
    tag: str = ""          # quant tag from the ref (GGUF)

    # GGUF selection breakdown
    primary: str = ""
    shards: list[str] = field(default_factory=list)
    mmproj: str = ""
    mtp: str = ""

    # ONNX: tokenizer source repo ("" → same repo as the model)
    tokenizer_repo: str = ""
    # TTS: voice sidecar files (e.g. voices/*.bin)
    voices: list[str] = field(default_factory=list)

    @property
    def has_files(self) -> bool:
        return bool(self.files)


def split_repo_tag(ref: str) -> tuple[str, str]:
    """Split ``repo[:tag]`` into ``(repo, tag)``. No colon → empty tag."""
    if ":" in ref:
        repo, _, tag = ref.partition(":")
        return repo, tag
    return ref, ""


# ── Format detection ─────────────────────────────────────────────────────

def detect_format(ref: str, files: list[str]) -> str:
    """Classify a ref + its file listing.

    The ref's tag is authoritative when it names a file (``repo:model.onnx``)
    or a glob (``repo:*.onnx``); otherwise the listing decides: any
    ``*.gguf`` → gguf, else any ``*.onnx`` → onnx subtype, else snapshot.
    """
    repo, tag = split_repo_tag(ref)
    if tag:
        if "*" in tag:
            return "snapshot"
        if tag.lower().endswith(".onnx"):
            return "onnx"
        if tag.lower().endswith(".gguf"):
            return "gguf"
    # Tag is a quant or absent — the listing decides.
    if any(f.lower().endswith(".gguf") for f in files):
        return "gguf"
    if any(f.lower().endswith(".onnx") for f in files):
        return _onnx_subtype(repo, files)
    return "snapshot"


def _onnx_subtype(repo: str, files: list[str]) -> str:
    """whisper / tts / embed from repo id and filenames."""
    hay = (repo + " " + " ".join(files)).lower()
    if "whisper" in hay or "asr" in hay:
        return "onnx-whisper"
    if "kokoro" in hay or "tts" in hay or "piper" in hay:
        return "onnx-tts"
    return "onnx-embed"


# ── Planners ─────────────────────────────────────────────────────────────

def build_plan(ref: str, files: list[str]) -> HfPlan:
    """Detect the format of ``ref`` and return its download plan."""
    fmt = detect_format(ref, files)
    repo, tag = split_repo_tag(ref)
    if fmt == "gguf":
        return _plan_gguf(repo, tag, files)
    if fmt.startswith("onnx"):
        return _plan_onnx(repo, tag, files, fmt)
    return _plan_snapshot(repo, tag, files)


def _plan_gguf(repo: str, tag: str, files: list[str]) -> HfPlan:
    """GGUF set: primary + any mmproj/mtp sidecars + split shards."""
    primary = _find_primary(files, tag)
    if not primary:
        return HfPlan(repo=repo, format="gguf", tag=tag)
    shards = _shards_for(files, primary)
    mmproj = _find_sibling(files, primary, "mmproj", tag)
    mtp = _find_sibling(files, primary, "mtp-", tag)
    extra = [e for e in (mmproj, mtp) if e and e not in shards]
    return HfPlan(repo=repo, format="gguf", tag=tag, primary=primary,
                  shards=shards, mmproj=mmproj, mtp=mtp,
                  files=shards + extra)


def _plan_onnx(repo: str, tag: str, files: list[str], fmt: str) -> HfPlan:
    """ONNX set: every ``*.onnx`` (+ ``*.bin`` voice sidecars for tts) + tokenizer."""
    if fmt == "onnx":
        # ref named one exact file (repo:model.onnx)
        onnx = [f for f in files if f == tag]
    else:
        onnx = [f for f in files if f.lower().endswith(".onnx")]
        voices = []
        if fmt == "onnx-tts":
            voices = [f for f in files if f.lower().endswith(".bin")]
            onnx += [f for f in voices if f not in onnx]
    # Repo-level pulls include tokenizer files; an exact-file ref does not.
    if fmt == "onnx":
        return HfPlan(repo=repo, format=fmt, files=onnx)
    token = [f for f in files
             if f.rsplit("/", 1)[-1] in _TOKENIZER_CANDIDATES]
    token = [t for t in token if t not in onnx]
    return HfPlan(repo=repo, format=fmt, files=onnx + token, voices=voices)


def _is_scaffolding(name: str) -> bool:
    """True for repo-scaffolding filenames (case-insensitive, .md-tolerant)."""
    name = name.lower()
    return any(name == s or name.startswith(s + ".") for s in _SNAPSHOT_SKIP)


def _plan_snapshot(repo: str, tag: str, files: list[str]) -> HfPlan:
    """Unknown format: the whole repo minus scaffolding files.

    A glob tag (``repo:*.onnx``) narrows the fetch to matching files.
    """
    if tag and "*" in tag:
        out = [f for f in files if fnmatch.fnmatch(f.rsplit("/", 1)[-1], tag)]
    else:
        out = [f for f in files
               if not _is_scaffolding(f.rsplit("/", 1)[-1])
               and not f.startswith(_DATA_DIR)]
    return HfPlan(repo=repo, format="snapshot", files=out)


# ── Download ─────────────────────────────────────────────────────────────

def download_plan(plan: HfPlan, *, cache_dir: str | None = None,
                  token: str | None = None, progress_cb=None) -> list[str]:
    """Fetch every file in ``plan`` into the HF cache; return local paths.

    Uses ``hf_hub_download`` per file so only the planned files are fetched.
    ``progress_cb(filename, index)`` is called before each file starts (1-based).
    """
    from huggingface_hub import hf_hub_download

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


# ── Model naming / scaffolds ─────────────────────────────────────────────

def derived_name(repo: str) -> str:
    """Model key for a pulled repo: ``owner-basename``, lowercase.

    ``Xenova/whisper-tiny`` → ``xenova-whisper-tiny``.
    """
    owner, _, base = repo.partition("/")
    return f"{owner}-{base}".lower().replace("/", "-")


#: onnx subtype → ``type:`` field for a startable config entry.
_ONNX_TYPE_FIELD = {
    "onnx-whisper": "whisper",
    "onnx-embed": "embed",
    "onnx": "embed",
    "onnx-tts": "tts",
}


def scaffold_entry(repo: str, plan: HfPlan,
                   backend: str | None = None) -> dict | None:
    """A startable ``models.<name>`` dict for a freshly pulled repo.

    Returns None when the format has no known runner (snapshot) — such
    refs cannot be pulled as models.
    """
    if plan.format == "snapshot":
        return None
    model_ref = f"{repo}:{plan.tag}" if plan.tag else repo
    if plan.format in _ONNX_TYPE_FIELD:
        entry = {
            "backend": "onnx",
            "model": model_ref,
            "type": _ONNX_TYPE_FIELD[plan.format],
            "tokenizer": plan.tokenizer_repo or repo,
        }
        if plan.format == "onnx-tts" and plan.voices:
            # Kokoro loads voices from the repo's voices/ dir in the cache.
            entry["voices_path"] = model_ref
        return entry
    # gguf — plain llama model on the effective default backend.
    return {"backend": backend or "", "model": model_ref}


# ── GGUF selection internals ─────────────────────────────────────────────

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
