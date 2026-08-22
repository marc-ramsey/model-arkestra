"""Tests for arkestra-cli init scaffolding."""

import os
from pathlib import Path
from unittest.mock import patch

from model_arkestra.cli import cmd_init, DEFAULT_CONFIG_DIR


def test_init_creates_files(tmp_path):
    """Init creates config.yaml and sources.yaml in default directory."""
    with patch("model_arkestra.cli.DEFAULT_CONFIG_DIR", tmp_path):
        result = cmd_init(force=True)

    assert result == 0
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "sources.yaml").exists()


def test_init_content_contains_scaffold_markers(tmp_path):
    """Generated config.yaml contains expected scaffold content."""
    with patch("model_arkestra.cli.DEFAULT_CONFIG_DIR", tmp_path):
        cmd_init(force=True)

    config = (tmp_path / "config.yaml").read_text()
    assert "models-start-port" in config
    # Backend is auto-detected — just verify a valid backend was set
    assert "backends:" in config
    assert "default:" in config


def test_init_preserves_existing(tmp_path):
    """Init refuses to overwrite existing files without --force."""
    # Pre-create config file
    (tmp_path / "config.yaml").write_text("# existing\n")

    with patch("model_arkestra.cli.DEFAULT_CONFIG_DIR", tmp_path):
        result = cmd_init(force=False)

    assert result == 1  # should return error code
    assert (tmp_path / "config.yaml").read_text() == "# existing\n"


def test_init_overwrites_with_force(tmp_path):
    """Init overwrites existing files when --force is used."""
    (tmp_path / "config.yaml").write_text("# old content\n")

    with patch("model_arkestra.cli.DEFAULT_CONFIG_DIR", tmp_path):
        result = cmd_init(force=True)

    assert result == 0
    # Check that new content was written
    config = (tmp_path / "config.yaml").read_text()
    assert "models-start-port" in config


def test_sources_yaml_has_preconfigured_sources(tmp_path):
    """Generated sources.yaml has pre-configured download sources."""
    with patch("model_arkestra.cli.DEFAULT_CONFIG_DIR", tmp_path):
        cmd_init(force=True)

    sources = (tmp_path / "sources.yaml").read_text()
    # Should have actual source definitions, not empty dict
    assert "sources:" in sources
    assert "github-release" in sources


def test_default_config_dir_is_xdg_compliant():
    """DEFAULT_CONFIG_DIR follows XDG Base Directory specification."""
    assert DEFAULT_CONFIG_DIR == Path.home() / ".config" / "arkestra"
