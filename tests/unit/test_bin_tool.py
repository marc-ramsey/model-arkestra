"""Tests for bin_tool asset-template rendering and binary-name resolution.

Covers the two generalizations needed for stable-diffusion.cpp backends:
  - ``{sha}`` template token (last dash-segment of the tag)
  - ``source.binary`` key (default ``llama-server``)
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from model_arkestra import bin_tool

SD_TAG = "master-920-2f88688"
SD_ASSET = "sd-master-{sha}-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.14.0.zip"


class TestRenderAssetTpl:
    """``{tag}`` renders the full tag; ``{sha}`` renders its last dash-segment."""

    def test_sha_token(self):
        out = bin_tool._render_asset_tpl(SD_ASSET, SD_TAG)
        assert out == "sd-master-2f88688-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.14.0.zip"

    def test_tag_token_unchanged(self):
        out = bin_tool._render_asset_tpl("llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz", "b11065")
        assert out == "llama-b11065-bin-ubuntu-vulkan-x64.tar.gz"

    def test_single_segment_tag(self):
        # No dash — {sha} falls back to the whole tag.
        assert bin_tool._render_asset_tpl("x-{sha}.zip", "abc") == "x-abc.zip"


class TestLatestTag:
    """latest_tag() finds releases via the {sha}-rendered template."""

    def _mock_releases(self, releases):
        import json as _json
        import io
        from unittest.mock import patch, MagicMock

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return _json.dumps(releases).encode()

        with patch.object(bin_tool.urllib.request, "urlopen",
                          return_value=_Resp()) as m:
            tag = bin_tool.latest_tag("leejet/stable-diffusion.cpp", SD_ASSET)
        return tag

    def test_sha_match(self):
        releases = [
            {"tag_name": "master-917-017cc8e",
             "assets": [{"name": "sd-master-017cc8e-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"}]},
            {"tag_name": "master-920-2f88688",
             "assets": [{"name": "sd-master-2f88688-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.14.0.zip"}]},
        ]
        assert self._mock_releases(releases) == "master-920-2f88688"

    def test_no_match_returns_none(self):
        releases = [
            {"tag_name": "master-917-017cc8e",
             "assets": [{"name": "sd-master-017cc8e-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"}]},
        ]
        assert self._mock_releases(releases) is None


class TestBinaryName:
    """source.binary overrides the default llama-server slot name."""

    def test_default(self):
        assert bin_tool._binary_name({"type": "remote", "repo": "x", "asset": "y"}) == "llama-server"

    def test_override(self):
        src = {"type": "remote", "repo": "leejet/stable-diffusion.cpp",
               "asset": SD_ASSET, "binary": "sd-server"}
        assert bin_tool._binary_name(src) == "sd-server"


class TestFlattenNested:
    """flatten_if_nested() hoists a nested archive dir, honoring the binary name."""

    def test_nested_sd_server(self, tmp_path: Path):
        nested = tmp_path / "stable-diffusion.cpp-master-920-2f88688"
        nested.mkdir()
        (nested / "sd-server").write_text("bin")
        (nested / "libfoo.so").write_text("so")
        bin_tool.flatten_if_nested(tmp_path, "sd-server")
        assert (tmp_path / "sd-server").is_file()
        assert (tmp_path / "libfoo.so").is_file()
        assert not nested.exists()

    def test_missing_binary_raises(self, tmp_path: Path):
        (tmp_path / "something").write_text("x")
        with pytest.raises(bin_tool.BinError):
            bin_tool.flatten_if_nested(tmp_path, "sd-server")

    def test_already_flat(self, tmp_path: Path):
        (tmp_path / "llama-server").write_text("bin")
        bin_tool.flatten_if_nested(tmp_path, "llama-server")
        assert (tmp_path / "llama-server").is_file()
