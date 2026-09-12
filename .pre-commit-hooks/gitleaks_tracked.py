#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run gitleaks against a snapshot of all git-tracked files."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


def _run(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.PIPE, text=True)


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return Path(result.stdout.strip())


def _tracked_files(repo_root: Path) -> list[Path]:
    result = _run(["git", "ls-files", "-z"], cwd=repo_root)
    tracked = []

    for raw_path in result.stdout.split("\0"):
        if not raw_path:
            continue

        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe git path from git ls-files: {raw_path!r}")

        tracked.append(path)

    return tracked


def _copy_tracked_files(repo_root: Path, snapshot_root: Path, tracked_files: Sequence[Path]) -> None:
    for path in tracked_files:
        source = repo_root / path
        if not source.exists() and not source.is_symlink():
            continue
        if source.is_dir() and not source.is_symlink():
            continue

        destination = snapshot_root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=False)


def _relay(output: str, stream, snapshot_root: Path) -> None:
    if not output:
        return

    display_output = output.replace(f"{snapshot_root}{os.sep}", "").replace(str(snapshot_root), ".")
    stream.write(display_output)


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        print("gitleaks-tracked does not accept filename arguments", file=sys.stderr)
        return 2

    if shutil.which("gitleaks") is None:
        print("gitleaks executable was not found in PATH", file=sys.stderr)
        return 1

    repo_root = _repo_root()
    tracked_files = _tracked_files(repo_root)
    if not tracked_files:
        return 0

    with tempfile.TemporaryDirectory(prefix="gitleaks-tracked-") as tmpdir:
        snapshot_root = Path(tmpdir)
        _copy_tracked_files(repo_root, snapshot_root, tracked_files)

        result = subprocess.run(
            ["gitleaks", "dir", "--redact", "--verbose", str(snapshot_root)],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        _relay(result.stdout, sys.stdout, snapshot_root)
        _relay(result.stderr, sys.stderr, snapshot_root)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
