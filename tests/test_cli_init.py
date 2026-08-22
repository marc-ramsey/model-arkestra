"""Tests for arkestra-cli init scaffolding."""

import os
from pathlib import Path
from unittest.mock import patch

from model_arkestra.cli import cmd_init, DEFAULT_CONFIG_DIR


def test_init_creates_files(tmp_path):
    """Init creates config.yaml and backends.yaml in default directory."""
    with patch("model_arkestra.cli.DEFAULT_CONFIG_DIR", tmp_path):
        result = cmd_init(force=True)

    assert result == 0
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "backends.yaml").exists()


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


def test_backends_yaml_has_preconfigured_backends(tmp_path):
    """Generated backends.yaml has pre-configured backend definitions and sources."""
    with patch("model_arkestra.cli.DEFAULT_CONFIG_DIR", tmp_path):
        cmd_init(force=True)

    backends = (tmp_path / "backends.yaml").read_text()
    # Should have both backends: section and sources: section
    assert "backends:" in backends
    assert "sources:" in backends
    assert "github-release" in backends


def test_backends_yaml_has_built_in_backends(tmp_path):
    """Generated backends.yaml includes all four built-in backend definitions."""
    with patch("model_arkestra.cli.DEFAULT_CONFIG_DIR", tmp_path):
        cmd_init(force=True)

    backends = (tmp_path / "backends.yaml").read_text()
    assert "vulkan-radv:" in backends
    assert "rocm:" in backends
    assert "nvidia-cuda:" in backends
    assert "cpu-optimized:" in backends


def test_default_config_dir_is_xdg_compliant():
    """DEFAULT_CONFIG_DIR follows XDG Base Directory specification."""
    assert DEFAULT_CONFIG_DIR == Path.home() / ".config" / "arkestra"
