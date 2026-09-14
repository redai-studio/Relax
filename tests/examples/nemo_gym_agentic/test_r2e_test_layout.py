# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execute the shipped R2E-Gym layout patch without its heavyweight imports."""

import re
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


PATCH = Path(__file__).resolve().parents[3] / "examples/nemo_gym_agentic/service/patches/r2egym_test_layout.patch"


@pytest.fixture
def layout(tmp_path: Path) -> SimpleNamespace:
    additions = [
        line[1:] for line in PATCH.read_text().splitlines() if line.startswith("+") and not line.startswith("+++")
    ]
    start = next(i for i, line in enumerate(additions) if line.startswith("    def _setup_r2e_tests("))
    namespace = {"shlex": shlex}
    exec("class PatchedRuntime:\n" + "\n".join(additions[start:]), namespace)
    runtime = namespace["PatchedRuntime"]()
    runtime.alt_path = str(tmp_path / "private files")
    runtime.repo_path = str(tmp_path / "repo files")
    Path(runtime.alt_path).mkdir()
    Path(runtime.repo_path).mkdir()
    source = tmp_path / "source"

    def run(command: str) -> tuple[str, str]:
        # Relocate only the sandbox's fixed /r2e_tests source into this fixture.
        command = re.sub(r"(?<![\w/])/r2e_tests(?![\w/])", str(source), command)
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=10)
        return result.stdout + result.stderr, str(result.returncode)

    runtime.run = run
    return SimpleNamespace(
        runtime=runtime,
        source=source,
        tests=Path(runtime.alt_path) / "r2e_tests",
        link=Path(runtime.repo_path) / "r2e_tests",
    )


@pytest.mark.parametrize("initial", ["sif", "root_source", "repo_source", "self_link"])
def test_test_layout_preserves_tests_across_repeated_setup(layout: SimpleNamespace, initial: str) -> None:
    directory = {"root_source": layout.source, "repo_source": layout.link}.get(initial, layout.tests)
    directory.mkdir()
    (directory / "test_case.py").write_text("def test_case(): pass\n")
    if initial in {"sif", "self_link"}:
        layout.link.symlink_to(layout.tests, target_is_directory=True)
    if initial == "self_link":
        (layout.tests / "r2e_tests").symlink_to(layout.tests, target_is_directory=True)

    for _ in range(3):
        layout.runtime._setup_r2e_tests()
        assert layout.link.is_symlink()
        assert layout.link.resolve() == layout.tests
        assert (layout.link / "test_case.py").read_text() == "def test_case(): pass\n"
        assert not (layout.tests / "r2e_tests").exists()
        assert not (layout.tests / "r2e_tests").is_symlink()


def test_test_layout_preserves_a_real_nested_directory(layout: SimpleNamespace) -> None:
    nested = layout.tests / "r2e_tests"
    nested.mkdir(parents=True)
    (nested / "test_nested.py").write_text("preserve me")
    layout.runtime._setup_r2e_tests()
    assert (nested / "test_nested.py").read_text() == "preserve me"


def test_test_layout_preserves_conflicting_real_repository_directory(layout: SimpleNamespace) -> None:
    layout.tests.mkdir()
    layout.link.mkdir()
    (layout.link / "test_existing.py").write_text("preserve me")
    with pytest.raises(RuntimeError, match="test directory setup failed"):
        layout.runtime._setup_r2e_tests()
    assert (layout.link / "test_existing.py").read_text() == "preserve me"
