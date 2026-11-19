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
        snapshot_root = Path(tmpdir) / "contents"
        snapshot_root.mkdir()
        names_root = Path(tmpdir) / "filenames"
        names_root.mkdir()
        _copy_tracked_files(repo_root, snapshot_root, tracked_files)
        # Gitleaks can skip binary, empty and symlink files before path rules.
        # Give each path a one-line text file so its fingerprint is stable even
        # when other tracked files are added. Keep these separate from real files.
        for path in tracked_files:
            manifest = names_root / ".gitleaks-paths" / f"{path.as_posix()}.txt"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(f"{path.as_posix()}\n", encoding="utf-8")

        ignore_file = snapshot_root / ".gitleaksignore"
        if ignore_file.is_file():
            shutil.copy2(ignore_file, names_root / ".gitleaksignore")

        # Relative scan paths keep native .gitleaksignore fingerprints stable.
        for scan_root in (snapshot_root, names_root):
            result = subprocess.run(
                ["gitleaks", "dir", "--config", str(repo_root / ".gitleaks.toml"), "--redact", "--verbose", "."],
                cwd=scan_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            _relay(result.stdout, sys.stdout, scan_root)
            _relay(result.stderr, sys.stderr, scan_root)
            if result.returncode:
                return result.returncode
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
