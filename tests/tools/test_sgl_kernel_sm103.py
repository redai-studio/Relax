# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise kernel selection and rebuild decisions without a GPU or pip
install."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("kernel_build", ROOT / "docker/scripts/ensure_sgl_kernel_sm103.py")
kernel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kernel)


def make_source(root, layout, version="0.4.7"):
    source = root / layout
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text('set(FLAGS "-gencode=arch=compute_120a,code=sm_120a")\n')
    (source / "pyproject.toml").write_text(f'[project]\nname = "sglang-kernel"\nversion = "{version}"\n')
    return source


@pytest.fixture
def build(monkeypatch, tmp_path):
    libs = [Path("sgl_kernel/sm90/common_ops.abi3.so"), Path("sgl_kernel/sm100/common_ops.abi3.so")]
    dist = SimpleNamespace(files=libs, locate_file=lambda path: tmp_path / path)
    metadata = SimpleNamespace(version=Mock(return_value="0.4.7"), distribution=Mock(return_value=dist))
    monkeypatch.setattr(kernel, "md", metadata)
    monkeypatch.setattr(kernel, "SGLANG_SOURCE", tmp_path / "source")
    monkeypatch.setenv("GPU_ARCH", "blackwell")
    monkeypatch.setenv("PIP_CONSTRAINT", str(tmp_path / "constraints.txt"))
    commands = Mock()
    monkeypatch.setattr(kernel.subprocess, "run", commands)
    return SimpleNamespace(root=tmp_path, metadata=metadata, commands=commands)


@pytest.mark.parametrize("target", ["103", "103a", "103f", "100f"])
def test_kernel_accepts_blackwell_compatible_cubin_without_sources(build, monkeypatch, target):
    inspect = Mock(return_value=f"Fatbin elf code:\n================\narch = sm_{target}\n")
    monkeypatch.setattr(kernel.subprocess, "check_output", inspect)
    kernel.main()
    build.commands.assert_not_called()
    assert inspect.call_count == 1
    assert "/sm100/" in inspect.call_args.args[0][-1]
    assert (build.root / "constraints.txt").read_text() == "sglang-kernel==0.4.7\n"


@pytest.mark.parametrize("target", ["90a", "100a", "120a", "1030", "100foo"])
def test_kernel_rejects_noncompatible_targets(monkeypatch, target):
    monkeypatch.setattr(
        kernel.subprocess, "check_output", lambda *a, **kw: f"Fatbin elf code:\n====\narch = sm_{target}\n"
    )
    assert not kernel.has_sm103([Path("common_ops.so")])
    assert not kernel.has_sm103([])


def test_kernel_does_not_accept_hopper_variant_as_blackwell(build, monkeypatch):
    monkeypatch.setattr(
        kernel.subprocess,
        "check_output",
        lambda args, **kw: (
            "Fatbin elf code:\n====\narch = sm_103a\n"
            if "/sm90/" in args[-1]
            else "Fatbin elf code:\n====\narch = sm_100a\n"
        ),
    )
    assert not kernel.has_sm103(kernel.common_ops())


def test_kernel_checks_all_selected_libraries(monkeypatch):
    inspect = Mock(
        side_effect=["Fatbin elf code:\n====\narch = sm_103a\n", "Fatbin elf code:\n====\narch = sm_100a\n"]
    )
    monkeypatch.setattr(kernel.subprocess, "check_output", inspect)
    assert not kernel.has_sm103([Path("one.so"), Path("two.so")])


@pytest.mark.parametrize("layout", ["python/sglang/kernels/aot", "sgl-kernel"])
def test_kernel_rebuild_uses_matching_source_and_keeps_dependencies(build, monkeypatch, layout):
    source = make_source(kernel.SGLANG_SOURCE, layout)
    monkeypatch.setattr(
        kernel.subprocess,
        "check_output",
        Mock(side_effect=["Fatbin elf code:\n====\narch = sm_100a\n", "Fatbin elf code:\n====\narch = sm_103a\n"]),
    )
    kernel.main()
    assert build.commands.call_count == 2
    install = build.commands.call_args
    assert install.kwargs["cwd"] == source
    assert "--no-build-isolation" in install.args[0]
    assert "--no-deps" in install.args[0]
    assert install.kwargs["env"]["CUDA_HOME"] == "/usr/local/cuda"
    assert '"-gencode=arch=compute_103a,code=sm_103a"' in (source / "CMakeLists.txt").read_text()
    contents = (source / "CMakeLists.txt").read_text()
    kernel.add_sm103_target(source / "CMakeLists.txt")
    assert (source / "CMakeLists.txt").read_text() == contents


def test_kernel_rebuild_still_verifies_result(build, monkeypatch):
    make_source(kernel.SGLANG_SOURCE, "python/sglang/kernels/aot")
    monkeypatch.setattr(
        kernel.subprocess, "check_output", lambda *args, **kw: "Fatbin elf code:\n====\narch = sm_100a\n"
    )
    with pytest.raises(AssertionError, match="still lacks SM103"):
        kernel.main()


def test_kernel_rebuild_rejects_version_drift(build, monkeypatch):
    make_source(kernel.SGLANG_SOURCE, "python/sglang/kernels/aot")
    monkeypatch.setattr(
        kernel.subprocess, "check_output", lambda *args, **kw: "Fatbin elf code:\n====\narch = sm_100a\n"
    )
    build.metadata.version.side_effect = ["0.4.7", "0.4.7", "0.4.8"]
    with pytest.raises(AssertionError, match="version change"):
        kernel.main()


def test_kernel_missing_source_reports_both_layouts(build, monkeypatch):
    monkeypatch.setattr(
        kernel.subprocess, "check_output", lambda *args, **kw: "Fatbin elf code:\n====\narch = sm_100a\n"
    )
    with pytest.raises(FileNotFoundError, match="python/sglang/kernels/aot.*sgl-kernel"):
        kernel.main()
    build.commands.assert_not_called()


def test_kernel_mismatched_source_fails_before_install(build, monkeypatch):
    make_source(kernel.SGLANG_SOURCE, "python/sglang/kernels/aot", version="0.4.8")
    monkeypatch.setattr(
        kernel.subprocess, "check_output", lambda *args, **kw: "Fatbin elf code:\n====\narch = sm_100a\n"
    )
    with pytest.raises(RuntimeError, match="does not match installed"):
        kernel.main()
    build.commands.assert_not_called()


def test_kernel_hopper_skips_build_and_metadata(build, monkeypatch):
    monkeypatch.setenv("GPU_ARCH", "hopper")
    kernel.main()
    build.metadata.version.assert_not_called()
    build.commands.assert_not_called()
    assert not (build.root / "constraints.txt").exists()


def test_kernel_unknown_cmake_flags_fail_closed(tmp_path):
    cmake = tmp_path / "CMakeLists.txt"
    cmake.write_text("# upstream no longer has our build anchor\n")
    with pytest.raises(RuntimeError, match="build flags changed"):
        kernel.add_sm103_target(cmake)


def test_kernel_missing_blackwell_library_is_not_hopper_success(build):
    build.metadata.distribution.return_value.files = [Path("sgl_kernel/sm90/common_ops.abi3.so")]
    with pytest.raises(RuntimeError, match="Blackwell common_ops library missing"):
        kernel.common_ops()


def test_kernel_does_not_accept_ptx_only_family_target(monkeypatch):
    output = "Fatbin ptx code:\n====\narch = sm_100f\nFatbin elf code:\n====\narch = sm_100a\n"
    monkeypatch.setattr(kernel.subprocess, "check_output", lambda *args, **kw: output)
    assert not kernel.has_sm103([Path("common_ops.so")])
