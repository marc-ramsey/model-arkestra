"""Tests: hf_models plans + formats (no network).

Covers the GGUF rules (ported from llama.cpp's common/download.cpp) plus the
new ONNX and snapshot planners, format detection, and scaffold naming.
"""

from __future__ import annotations

from model_arkestra.hf_models import (
    build_plan, detect_format, derived_name, scaffold_entry,
    split_repo_tag, _quant_bits, _split_info,
)


# ── ref parsing (unchanged contract from hf_gguf) ──────────────────────────

def test_split_repo_tag():
    assert split_repo_tag("a/b:Q4_0") == ("a/b", "Q4_0")
    assert split_repo_tag("a/b") == ("a/b", "")
    assert split_repo_tag("unsloth/x-GGUF:UD-Q4_K_XL") == ("unsloth/x-GGUF", "UD-Q4_K_XL")


def test_quant_bits_stops_at_first_run():
    # int() would choke on '4_K_XL'; we must take only the leading digits.
    assert _quant_bits("Q4_0") == 4
    assert _quant_bits("F16") == 16
    assert _quant_bits("UD-Q4_K_XL") == 4
    assert _quant_bits("BF16") == 16
    assert _quant_bits("no-digits") == 0


# ── GGUF planner (ported rules) ────────────────────────────────────────────

def test_gguf_primary_matches_full_quant_token():
    files = [
        "m-Q4_K_M.gguf",
        "m-UD-Q4_K_XL.gguf",   # full token — must win over plain Q4_K_XL
        "m-Q4_K_XL.gguf",
        "mmproj-F16.gguf",
    ]
    plan = build_plan("repo/m:Q4_K_M", files)
    assert plan.format == "gguf"
    assert plan.primary == "m-Q4_K_M.gguf"


def test_gguf_primary_falls_back_when_no_tag():
    files = ["m-Q4_K_M.gguf", "m-Q8_0.gguf", "mmproj-F16.gguf"]
    plan = build_plan("repo/m", files)
    assert plan.primary == "m-Q4_K_M.gguf"


def test_gguf_sidecars_resolved_as_siblings():
    files = [
        "m-UD-Q4_K_XL.gguf",
        "mmproj-BF16.gguf",
        "MTP/mtp-m-BF16.gguf",
    ]
    plan = build_plan("repo/m:UD-Q4_K_XL", files)
    assert plan.mmproj == "mmproj-BF16.gguf"
    assert plan.mtp == "MTP/mtp-m-BF16.gguf"
    assert set(plan.files) == {"m-UD-Q4_K_XL.gguf", "mmproj-BF16.gguf",
                               "MTP/mtp-m-BF16.gguf"}


def test_gguf_text_only_model_has_no_sidecars():
    files = ["m-Q4_0.gguf", "imatrix.dat"]
    plan = build_plan("repo/m:Q4_0", files)
    assert plan.files == ["m-Q4_0.gguf"]


def test_gguf_split_shards_collected():
    files = [
        "big-00001-of-00003-Q8_0.gguf",
        "big-00002-of-00003-Q8_0.gguf",
        "big-00003-of-00003-Q8_0.gguf",
    ]
    plan = build_plan("repo/big:Q8_0", files)
    assert plan.shards == files


def test_split_info_parses_shard_and_tag():
    prefix, tag, idx, count = _split_info("big-00002-of-00003-Q8_0.gguf")
    assert (idx, count) == (2, 3)
    assert tag == "Q8_0"


# ── format detection ───────────────────────────────────────────────────────

def test_detect_onnx_whisper():
    files = ["encoder_model.onnx", "decoder_model_merged.onnx", "tokenizer.json"]
    assert detect_format("x/whisper-tiny", files) == "onnx-whisper"


def test_detect_onnx_embedding():
    files = ["model.onnx", "tokenizer.json"]
    assert detect_format("x/bge-small", files) == "onnx-embed"


def test_detect_gguf():
    files = ["m-Q4_K_M.gguf"]
    assert detect_format("x/m:Q4_K_M", files) == "gguf"


def test_detect_snapshot_fallback():
    files = ["weights.safetensors", "config.json"]
    assert detect_format("x/m", files) == "snapshot"


def test_ref_suffix_wins():
    # Explicit .onnx suffix on the ref beats the listing scan.
    files = ["model.onnx"]
    assert detect_format("x/m:model.onnx", files) == "onnx"


# ── ONNX planner ───────────────────────────────────────────────────────────

def test_onnx_plan_includes_models_and_tokenizer():
    files = [
        "encoder_model.onnx",
        "decoder_model_merged.onnx",
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "config.json",          # not a model/tokenizer file
        "model.safetensors",    # not part of the ONNX plan
        "README.md",
    ]
    plan = build_plan("x/whisper-tiny", files)
    assert plan.format == "onnx-whisper"
    assert set(plan.files) == {
        "encoder_model.onnx", "decoder_model_merged.onnx",
        "tokenizer.json", "tokenizer.model", "vocab.json",
    }


def test_onnx_exact_ref_suffix():
    # repo:file — pull that one file only.
    files = ["a.onnx", "b.onnx", "tokenizer.json"]
    plan = build_plan("x/m:b.onnx", files)
    assert plan.files == ["b.onnx"]


# ── snapshot planner ───────────────────────────────────────────────────────

def test_snapshot_plan_fetches_useful_files_only():
    files = [
        "weights.safetensors", "config.json", ".gitattributes",
        "README.md", "dataset_infos.json", "data/train.parquet",
    ]
    plan = build_plan("x/m", files)
    assert plan.format == "snapshot"
    assert "weights.safetensors" in plan.files
    assert "config.json" in plan.files
    assert ".gitattributes" not in plan.files
    assert "README.md" not in plan.files
    assert "data/train.parquet" not in plan.files


def test_snapshot_ref_suffix_filters():
    files = ["a.onnx", "b.onnx", "tokenizer.json"]
    plan = build_plan("x/m:*.onnx", files)
    assert plan.files == ["a.onnx", "b.onnx"]


# ── derived name + scaffold ────────────────────────────────────────────────

def test_derived_name_owner_basename():
    assert derived_name("Xenova/whisper-tiny") == "xenova-whisper-tiny"
    assert derived_name("unsloth/Qwen3-4B-GGUF") == "unsloth-qwen3-4b-gguf"


def test_scaffold_whisper():
    plan = HfPlanLike("onnx-whisper",
                      ["encoder_model.onnx", "decoder_model_merged.onnx",
                       "tokenizer.json"])
    entry = scaffold_entry("Xenova/whisper-tiny", plan)
    assert entry == {
        "backend": "onnx",
        "model": "Xenova/whisper-tiny",
        "type": "whisper",
        "tokenizer": "Xenova/whisper-tiny",
    }


def test_scaffold_embedding():
    entry = scaffold_entry("Xenova/bge-small",
                           HfPlanLike("onnx-embed", ["model.onnx"]))
    assert entry["type"] == "embed"
    assert entry["tokenizer"] == "Xenova/bge-small"


def test_scaffold_gguf_chat():
    entry = scaffold_entry("unsloth/Qwen3-4B-GGUF",
                           HfPlanLike("gguf", ["m-Q4_K_M.gguf"]),
                           backend="rocm")
    assert entry["backend"] == "rocm"
    assert entry["model"] == "unsloth/Qwen3-4B-GGUF"
    assert "type" not in entry


def test_scaffold_tts_includes_voices_path():
    plan = HfPlanLike("onnx-tts", ["model.onnx", "voices/a.bin"])
    plan.voices = ["voices/a.bin"]
    entry = scaffold_entry("Xenova/kokoro", plan)
    assert entry["type"] == "tts"
    assert entry["voices_path"] == "Xenova/kokoro"


def test_scaffold_tts_without_voices_omits_voices_path():
    plan = HfPlanLike("onnx-tts", ["model.onnx"])
    entry = scaffold_entry("Xenova/kokoro", plan)
    assert "voices_path" not in entry


def test_scaffold_snapshot_has_no_entry():
    assert scaffold_entry("x/m", HfPlanLike("snapshot", ["a"])) is None


# Minimal stand-in so scaffold tests don't need a real plan.
class HfPlanLike:
    def __init__(self, fmt, files):
        self.format = fmt
        self.repo = ""
        self.tag = ""
        self.files = files
        self.tokenizer_repo = ""
        self.voices = []
        self.primary = files[0] if files else ""
        self.shards = files
        self.mmproj = ""
        self.mtp = ""
