# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise image source metadata checks on every CI Python version."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_manifest_parses_source_dependencies(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("runtime_manifest", ROOT / "docker/scripts/record_qwen35_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname = "jit-source"\ndependencies = ["example>=2"]\n')
    monkeypatch.setattr(module.md, "version", lambda name: "2.0")
    module.validate_source_dependencies(project)
    monkeypatch.setattr(module.md, "version", lambda name: "1.0")
    with pytest.raises(RuntimeError, match="requires example>=2, found 1.0"):
        module.validate_source_dependencies(project)
