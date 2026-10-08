# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Tests for ``relax.utils.device``.

Focuses on the ``_BACKEND`` single-source-of-truth table: data integrity,
derived lookups, and the table-driven detection order. Adding a new accelerator
should only require a new ``AcceleratorType`` member plus one ``_BACKEND`` row
— these tests lock that contract down.
"""

import types
from dataclasses import replace

import pytest
import torch

from relax.utils import device
from relax.utils.device import AcceleratorType, BackendSpec, device_module


_CPU_ONLY = device.get_device_name() == "cpu"


# ---------------------------------------------------------------------------
# Backend table integrity
# ---------------------------------------------------------------------------
def test_device_backend_table_covers_every_accelerator() -> None:
    assert set(device._BACKEND) == set(AcceleratorType)


def test_device_backend_table_rows_complete() -> None:
    for accel, spec in device._BACKEND.items():
        assert isinstance(spec, BackendSpec)
        assert spec.dist_backend, f"{accel}: empty dist_backend"
        assert spec.ray_resource, f"{accel}: empty ray_resource"
        if accel is AcceleratorType.CPU:
            assert spec.visible_devices_env == ""
        else:
            assert spec.visible_devices_env, f"{accel}: empty visible_devices_env"


def test_device_backend_table_torch_namespaces() -> None:
    valid = {"cuda", "npu", "xpu", "ppu", "cpu"}
    cuda_family = {AcceleratorType.CUDA, AcceleratorType.ROCM, AcceleratorType.KLX}
    for accel, spec in device._BACKEND.items():
        assert spec.torch_namespace in valid, f"{accel}: bad torch_namespace {spec.torch_namespace!r}"
        assert (spec.torch_namespace == "cuda") == (accel in cuda_family)


def test_device_backend_table_cuda_family_probes() -> None:
    for accel, spec in device._BACKEND.items():
        if spec.torch_namespace != "cuda":
            assert spec.cuda_family_probe is None, f"{accel}: probe outside the cuda family"
    assert device._BACKEND[AcceleratorType.CUDA].cuda_family_probe is None
    assert device._BACKEND[AcceleratorType.ROCM].cuda_family_probe is not None
    assert device._BACKEND[AcceleratorType.KLX].cuda_family_probe is not None


def test_device_backend_table_detection_order() -> None:
    keys = list(device._BACKEND)
    assert keys[0] == AcceleratorType.CUDA
    assert keys.index(AcceleratorType.ROCM) < keys.index(AcceleratorType.KLX)


def test_device_backend_table_capability_flags() -> None:
    for accel, spec in device._BACKEND.items():
        if accel is AcceleratorType.KLX:
            assert spec.allow_non_blocking_copy is False
            assert spec.allow_pinned_host_memory is False
        else:
            assert spec.allow_non_blocking_copy is True
            assert spec.allow_pinned_host_memory is True


# ---------------------------------------------------------------------------
# Derived lookups
# ---------------------------------------------------------------------------
def test_device_derived_ray_reverse_map() -> None:
    assert device._RAY_RESOURCE_TO_ACCEL == {
        "NPU": AcceleratorType.NPU,
        "XPU": AcceleratorType.XPU,
        "PPU": AcceleratorType.PPU,
        "GPU": AcceleratorType.CUDA,
    }


def test_device_derived_ray_probe_order() -> None:
    assert device._RAY_PROBE_ORDER == ("NPU", "XPU", "PPU", "GPU")


# ---------------------------------------------------------------------------
# Detection: RELAX_DEVICE_TYPE override & probe ordering
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("accel", list(AcceleratorType))
def test_device_override_env_selects_accelerator(accel: AcceleratorType, monkeypatch) -> None:
    monkeypatch.setenv("RELAX_DEVICE_TYPE", accel.value)
    device._detect_accelerator.cache_clear()
    try:
        assert device._detect_accelerator() is accel
    finally:
        device._detect_accelerator.cache_clear()


def test_device_override_env_unknown_value_autodetects(monkeypatch) -> None:
    monkeypatch.setenv("RELAX_DEVICE_TYPE", "bogus")
    device._detect_accelerator.cache_clear()
    try:
        accel = device._detect_accelerator()
        assert isinstance(accel, AcceleratorType)
        assert accel.value != "bogus"
    finally:
        device._detect_accelerator.cache_clear()


def _patch_family_probe(monkeypatch, accel: AcceleratorType, probe) -> None:
    """Swap a table row's cuda-family probe (frozen rows need a row
    replace)."""
    monkeypatch.setitem(device._BACKEND, accel, replace(device._BACKEND[accel], cuda_family_probe=probe))


def test_device_detection_rocm_before_klx(monkeypatch) -> None:
    # Only the cuda family is detectable, regardless of the host's real accelerators.
    monkeypatch.setattr(device, "_is_torch_device_module_available", lambda name: name == "cuda")
    _patch_family_probe(monkeypatch, AcceleratorType.ROCM, lambda: True)
    _patch_family_probe(monkeypatch, AcceleratorType.KLX, lambda: True)
    device._detect_accelerator.cache_clear()
    try:
        assert device._detect_accelerator() is AcceleratorType.ROCM
    finally:
        device._detect_accelerator.cache_clear()


def test_device_detection_cuda_when_no_subfamily(monkeypatch) -> None:
    monkeypatch.setattr(device, "_is_torch_device_module_available", lambda name: name == "cuda")
    _patch_family_probe(monkeypatch, AcceleratorType.ROCM, lambda: False)
    _patch_family_probe(monkeypatch, AcceleratorType.KLX, lambda: False)
    device._detect_accelerator.cache_clear()
    try:
        assert device._detect_accelerator() is AcceleratorType.CUDA
    finally:
        device._detect_accelerator.cache_clear()


def test_device_detection_plugin_namespace_before_cuda_family(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "npu", types.SimpleNamespace(is_available=lambda: True), raising=False)
    device._detect_accelerator.cache_clear()
    try:
        assert device._detect_accelerator() is AcceleratorType.NPU
    finally:
        device._detect_accelerator.cache_clear()


def test_device_detection_plugin_unavailable_falls_through(monkeypatch) -> None:
    monkeypatch.setattr(torch, "npu", types.SimpleNamespace(is_available=lambda: False), raising=False)
    device._detect_accelerator.cache_clear()
    try:
        assert device._detect_accelerator() is not AcceleratorType.NPU
    finally:
        device._detect_accelerator.cache_clear()


def test_device_is_rocm_detects_hip_build(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "6.2")
    assert device._is_rocm() is True


def test_device_is_rocm_false_on_non_hip_build() -> None:
    if getattr(torch.version, "hip", None) is not None:
        pytest.skip("this host runs a HIP build")
    assert device._is_rocm() is False


def test_device_is_klx_probes_xpuctrl_node(monkeypatch) -> None:
    fake_os = types.SimpleNamespace(path=types.SimpleNamespace(exists=lambda p: p == "/dev/xpuctrl"))
    monkeypatch.setattr(device, "os", fake_os)
    assert device._is_klx() is True
    monkeypatch.setattr(device, "os", types.SimpleNamespace(path=types.SimpleNamespace(exists=lambda p: False)))
    assert device._is_klx() is False


# ---------------------------------------------------------------------------
# Config-query APIs (stable on a Ray-less host: CUDA-default fallback)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _CPU_ONLY, reason="expects a CPU-only host without a Ray runtime")
class TestDeviceCpuHost:
    def test_get_device_name(self) -> None:
        assert device.get_device_name() == "cpu"

    def test_get_torch_device_module(self) -> None:
        assert device.get_torch_device_module() is torch.cpu

    def test_get_accelerator_type(self) -> None:
        assert device.get_accelerator_type() is AcceleratorType.CPU

    def test_config_query_apis_use_cuda_default(self, monkeypatch) -> None:
        # Pin the Ray state: other modules' collection-time ray.init() would flip this to "gloo".
        ray = pytest.importorskip("ray")
        monkeypatch.setattr(ray, "is_initialized", lambda: False)
        assert device.get_dist_backend() == "nccl"
        assert device.get_visible_devices_env_var() == "CUDA_VISIBLE_DEVICES"
        assert device.get_ray_accelerator_name() == "GPU"

    def test_config_query_apis_use_gloo_on_initialized_cpu_cluster(self, monkeypatch) -> None:
        ray = pytest.importorskip("ray")
        monkeypatch.setattr(ray, "is_initialized", lambda: True)
        monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": 8})
        assert device.get_dist_backend() == "gloo"
        assert device.get_visible_devices_env_var() == ""
        assert device.get_ray_accelerator_name() == "CPU"

    def test_make_device_string(self) -> None:
        assert device.make_device_string() == "cpu"
        assert device.make_device_string(3) == "cpu"

    def test_make_current_torch_device(self) -> None:
        assert device.make_current_torch_device() == torch.device("cpu")

    def test_capability_flags_default(self) -> None:
        assert device.use_non_blocking_copy() is True
        assert device.use_pinned_host_memory() is True

    def test_is_available_false(self) -> None:
        assert device.is_available() is False

    def test_mod_current_device(self) -> None:
        assert device_module.current_device() == "cpu"

    def test_mod_attribute_proxy_forwards_to_torch_module(self) -> None:
        assert device_module.Event is torch.cpu.Event

    def test_mod_unknown_attribute_raises(self) -> None:
        with pytest.raises(AttributeError):
            device_module.definitely_not_a_torch_api
