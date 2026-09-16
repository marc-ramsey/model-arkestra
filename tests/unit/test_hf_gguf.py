"""Tests: hf_gguf resolves repo:quant to the exact GGUF file set (no network).

Mirrors llama.cpp's common/download.cpp selection rules: primary by tag regex,
mmproj/mtp as best sibling, split-shard collection.
"""

from __future__ import annotations

from model_arkestra.hf_gguf import (
    resolve_plan, split_repo_tag, _quant_bits, _split_info,
)


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


def test_primary_matches_full_quant_token():
    files = [
        "m-Q4_K_M.gguf",
        "m-UD-Q4_K_XL.gguf",   # full token — must win over plain Q4_K_XL
        "m-Q4_K_XL.gguf",
        "mmproj-F16.gguf",
    ]
    plan = resolve_plan("repo/m:UD-Q4_K_XL", files)
    assert plan.primary == "m-UD-Q4_K_XL.gguf"


def test_primary_falls_back_when_no_tag():
    files = ["m-Q4_K_M.gguf", "m-Q8_0.gguf", "mmproj-F16.gguf"]
    plan = resolve_plan("repo/m", files)
    # No tag → first model file in listing order.
    assert plan.primary == "m-Q4_K_M.gguf"


def test_sidecars_resolved_as_siblings():
    files = [
        "m-UD-Q4_K_XL.gguf",
        "mmproj-BF16.gguf",
        "MTP/mtp-m-BF16.gguf",
    ]
    plan = resolve_plan("repo/m:UD-Q4_K_XL", files)
    assert plan.mmproj == "mmproj-BF16.gguf"
    assert plan.mtp == "MTP/mtp-m-BF16.gguf"
    assert set(plan.files) == {"m-UD-Q4_K_XL.gguf", "mmproj-BF16.gguf",
                               "MTP/mtp-m-BF16.gguf"}


def test_text_only_model_has_no_sidecars():
    files = ["m-Q4_0.gguf", "imatrix.dat"]  # imatrix is not a gguf sidecar
    plan = resolve_plan("repo/m:Q4_0", files)
    assert plan.primary == "m-Q4_0.gguf"
    assert plan.mmproj == ""
    assert plan.mtp == ""
    assert plan.files == ["m-Q4_0.gguf"]


def test_split_shards_collected():
    files = [
        "big-00001-of-00003-Q8_0.gguf",
        "big-00002-of-00003-Q8_0.gguf",
        "big-00003-of-00003-Q8_0.gguf",
    ]
    plan = resolve_plan("repo/big:Q8_0", files)
    assert len(plan.shards) == 3
    assert plan.shards[0] == "big-00001-of-00003-Q8_0.gguf"
    assert set(plan.files) == set(files)


def test_split_info_parses_shard_and_tag():
    prefix, tag, idx, count = _split_info("big-00002-of-00003-Q8_0.gguf")
    assert (idx, count) == (2, 3)
    assert tag == "Q8_0"
