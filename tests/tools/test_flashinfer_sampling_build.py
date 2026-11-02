# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check architecture gating and sampling cache placement without CUDA
builds."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SPEC = importlib.util.spec_from_file_location(
    "sampling_build", Path(__file__).resolve().parents[2] / "docker/scripts/build_flashinfer_sampling.py"
)
sampling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampling)


def test_sampling_hopper_skips_build_and_package_lookup(monkeypatch):
    monkeypatch.setenv("GPU_ARCH", "hopper")
    distribution = Mock(side_effect=AssertionError("Hopper must not rebuild sampling"))
    monkeypatch.setattr(sampling.md, "distribution", distribution)
    with pytest.raises(SystemExit) as result:
        sampling.main()
    assert result.value.code == 0
    distribution.assert_not_called()


@pytest.mark.parametrize(
    "layout", ["opt/sglang/lib/python3.12/site-packages", "usr/local/lib/python3.12/dist-packages"]
)
def test_sampling_installs_into_active_distribution(monkeypatch, tmp_path, layout):
    monkeypatch.setenv("GPU_ARCH", "blackwell")
    root = tmp_path / layout
    aot = root / "flashinfer_jit_cache/jit_cache/sampling/sampling.so"
    aot.parent.mkdir(parents=True)
    aot.write_bytes(b"old cubin")
    built = tmp_path / "sampling.so"
    built.write_bytes(b"sm103 cubin")
    build = Mock()
    spec = SimpleNamespace(build=build, jit_library_path=built)
    monkeypatch.setitem(
        sampling.sys.modules, "flashinfer.jit.sampling", SimpleNamespace(gen_sampling_module=lambda: spec)
    )
    distribution = Mock(return_value=SimpleNamespace(locate_file=lambda path: root / path))
    monkeypatch.setattr(sampling.md, "distribution", distribution)
    monkeypatch.setattr(sampling.subprocess, "check_output", lambda *args, **kwargs: "sampling.sm_103.cubin")
    sampling.main()
    distribution.assert_called_once_with("flashinfer-jit-cache")
    build.assert_called_once_with(verbose=True)
    assert aot.read_bytes() == b"sm103 cubin"
