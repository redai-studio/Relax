# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the actual integration patch in isolated Git repositories."""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

from examples.chunkflow.setup import TARGETS, prepare


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/chunkflow"


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


@pytest.fixture
def checkout(tmp_path):
    revision = (EXAMPLE / "relax.version").read_text().strip()
    for path in TARGETS:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git(ROOT, "show", f"{revision}:{path}"))
    (tmp_path / "relax/__init__.py").write_text("# pinned core\n")
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "add", "relax")
    git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "base",
    )
    example = tmp_path / "examples/chunkflow"
    example.mkdir(parents=True)
    shutil.copyfile(EXAMPLE / "relax.patch", example / "relax.patch")
    (example / "relax.version").write_bytes(git(tmp_path, "rev-parse", "HEAD"))
    return tmp_path


def test_patch_applies_idempotently_and_restores_pinned_sources(checkout):
    originals = {path: (checkout / path).read_bytes() for path in TARGETS}
    with pytest.raises(RuntimeError, match="Apply the example first"):
        prepare(checkout, check=True)
    prepare(checkout)
    prepare(checkout)
    prepare(checkout, check=True)
    for path in TARGETS:
        assert (checkout / path).read_bytes() != originals[path]
        ast.parse((checkout / path).read_text())
    prepare(checkout, reverse=True)
    prepare(checkout, reverse=True)
    assert {path: (checkout / path).read_bytes() for path in TARGETS} == originals


@pytest.mark.parametrize("path", [TARGETS[0], "relax/__init__.py"])
def test_incompatible_source_is_rejected_without_overwriting_changes(checkout, path):
    target = checkout / path
    target.write_text(target.read_text() + "\n# unrelated user change\n")
    before = {name: (checkout / name).read_bytes() for name in (*TARGETS, "relax/__init__.py")}
    with pytest.raises(RuntimeError):
        prepare(checkout)
    assert {name: (checkout / name).read_bytes() for name in before} == before
