# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check relocation, reuse, and failed-copy isolation for the Agent venv
cache."""

import shutil
import subprocess
import venv
from pathlib import Path

import pytest

from examples.osworld_agentic.scripts.cache_agent_env import cache_environment, prepare_archive


def test_agent_env_cache_relocates_python_and_reuses_complete_copy(tmp_path: Path) -> None:
    source, target = tmp_path / "shared", tmp_path / "local"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(source)
    archive = tmp_path / "env.tar"
    prepare_archive(source, archive)
    python = cache_environment(str(archive), str(target))
    result = subprocess.check_output([python, "-c", "import sys; print(sys.prefix)"], text=True)
    assert Path(result.strip()) == target
    # Reusing a completed cache needs no reads from the shared environment.
    shutil.rmtree(source)
    prepare_archive(source, archive)
    archive.unlink()
    assert cache_environment(str(archive), str(target)) == python


def test_agent_env_cache_failed_copy_does_not_publish_partial_env(tmp_path: Path, monkeypatch) -> None:
    source, target = tmp_path / "shared", tmp_path / "local"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(source)
    archive = tmp_path / "env.tar"
    prepare_archive(source, archive)
    run = subprocess.run

    def fail_copy(command, **kwargs):
        (Path(command[-1]) / "partial").touch()
        raise OSError("copy interrupted")

    monkeypatch.setattr(subprocess, "run", fail_copy)
    with pytest.raises(OSError, match="copy interrupted"):
        cache_environment(str(archive), str(target))
    assert not target.exists()
    assert not list(tmp_path.glob("local-*"))
    monkeypatch.setattr(subprocess, "run", run)
    assert Path(cache_environment(str(archive), str(target))).is_file()


def test_agent_env_cache_does_not_overwrite_unmarked_directory(tmp_path: Path) -> None:
    target = tmp_path / "local"
    target.mkdir()
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        cache_environment(str(tmp_path / "shared"), str(target))
    assert target.exists()
