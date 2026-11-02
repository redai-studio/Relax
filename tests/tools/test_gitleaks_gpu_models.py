# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check GPU naming policy through the tracked-file pre-commit entry point."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("gitleaks") is None, reason="gitleaks executable is required")


@pytest.fixture
def scan_repo(tmp_path):
    shutil.copy2(ROOT / ".gitleaks.toml", tmp_path / ".gitleaks.toml")
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    return tmp_path


def run_scan(repo: Path) -> subprocess.CompletedProcess[str]:
    subprocess.run(["git", "add", "--all"], cwd=repo, check=True)
    return subprocess.run(
        [sys.executable, str(ROOT / ".pre-commit-hooks/gitleaks_tracked.py")],
        cwd=repo,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("model", ["B100", "B200", "B300", "GB200", "GB300", "H100"])  # gitleaks:allow
@pytest.mark.parametrize("location", ["content", "empty_filename", "binary_filename", "symlink_filename"])
def test_restricted_gpu_models_fail_scan(scan_repo, model, location):
    if location == "content":
        (scan_repo / "sample.txt").write_text(f"NVIDIA {model}\n")
    else:
        path = scan_repo / f"run-8x{model.lower()}.png"
        if location == "symlink_filename":
            (scan_repo / "sample.txt").write_text("example\n")
            path.symlink_to("sample.txt")
        else:
            path.write_bytes(b"\x00\x89PNG\r\n" if location == "binary_filename" else b"")

    result = run_scan(scan_repo)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "nvidia-gpu-model" in result.stdout + result.stderr


def test_other_gpu_models_pass_scan(scan_repo):
    models = ["V100", "A100", "A800", "A40", "H800", "H20", "H200", "L20", "L20Y", "L40", "L40S", "RTX 3070 Ti"]
    (scan_repo / "sample.txt").write_text("".join(f"NVIDIA {model}\n" for model in models))
    for model in models:
        (scan_repo / f"run-8x{model.lower()}.png").write_bytes(b"\x00\x89PNG\r\n")

    result = run_scan(scan_repo)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("model", ["B100", "B200", "B300", "GB200", "GB300", "H100"])  # gitleaks:allow
def test_reviewed_gpu_compatibility_keys_pass_with_inline_allow(scan_repo, model):
    (scan_repo / "sample.py").write_text(f'DEVICE = "{model}"  # gitleaks:allow -- compatibility key\n')

    result = run_scan(scan_repo)

    assert result.returncode == 0, result.stdout + result.stderr
