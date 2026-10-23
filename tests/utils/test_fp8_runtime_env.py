# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""FP8 import-time defaults without loading Torch, Ray, or GPU dependencies."""

import sys
from types import SimpleNamespace

import pytest

import relax.utils
from relax.utils.env import configure_fp8_block_scaling


KEY = "NVTE_FP8_BLOCK_SCALING_FP32_SCALES"


def _runtime(monkeypatch, *, initialized=True, assigned=True, kind="cuda", major=9, gpu_id="0"):
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delitem(sys.modules, "transformer_engine.common.recipe", raising=False)
    resource = "NPU" if kind == "npu" else "GPU"
    queried_devices = []

    def get_device_capability(index):
        queried_devices.append(index)
        return major, 0

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            is_initialized=lambda: initialized,
            get_runtime_context=lambda: SimpleNamespace(
                get_accelerator_ids=lambda: {resource: [gpu_id]} if assigned else {}
            ),
        ),
    )
    device = SimpleNamespace(
        AcceleratorType=SimpleNamespace(CUDA="cuda"),
        get_accelerator_type=lambda: kind,
        device_module=SimpleNamespace(get_device_capability=get_device_capability),
    )
    monkeypatch.setattr(relax.utils, "device", device, raising=False)
    return queried_devices


@pytest.mark.parametrize(
    "initialized,assigned,kind,major,expected",
    [
        (False, True, "cuda", 9, None),
        (True, False, "cuda", 9, None),
        (True, True, "cuda", 9, "1"),
        (True, True, "cuda", 10, "0"),
        (True, True, "rocm", 10, "1"),
        (True, True, "npu", 10, "1"),
    ],
)
def test_fp8_default_uses_assigned_accelerator(monkeypatch, initialized, assigned, kind, major, expected):
    _runtime(monkeypatch, initialized=initialized, assigned=assigned, kind=kind, major=major)
    configure_fp8_block_scaling()
    assert relax.utils.env.Envs.NVTE_FP8_BLOCK_SCALING_FP32_SCALES == (expected or "1")
    assert relax.utils.env.os.environ.get(KEY) == expected


@pytest.mark.parametrize("explicit", ["0", "1"])
@pytest.mark.parametrize("initialized", [True, False])
def test_fp8_explicit_setting_is_preserved(monkeypatch, explicit, initialized):
    queried_devices = _runtime(monkeypatch, major=10, initialized=initialized)
    if not initialized:
        monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv(KEY, explicit)
    configure_fp8_block_scaling()
    assert relax.utils.env.os.environ[KEY] == explicit
    assert queried_devices == []


@pytest.mark.parametrize(
    "gpu_id,visible_devices,expected_index",
    [("1", None, 1), ("3", "5,3", 1), ("3", "3", 0), ("GPU-b", "GPU-a,GPU-b", 1)],
)
def test_fp8_queries_assigned_cuda_device(monkeypatch, gpu_id, visible_devices, expected_index):
    queried_devices = _runtime(monkeypatch, major=10, gpu_id=gpu_id)
    if visible_devices is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    configure_fp8_block_scaling()
    assert queried_devices == [expected_index]
    assert relax.utils.env.os.environ[KEY] == "0"


@pytest.mark.parametrize("major,expected", [(9, "1"), (10, "0")])
def test_fp8_torchrun_queries_local_rank(monkeypatch, major, expected):
    queried_devices = _runtime(monkeypatch, initialized=False, major=major)
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,3")
    configure_fp8_block_scaling()
    assert queried_devices == [1]
    assert relax.utils.env.os.environ[KEY] == expected


def test_fp8_ray_driver_ignores_inherited_local_rank(monkeypatch):
    queried_devices = _runtime(monkeypatch, assigned=False)
    monkeypatch.setenv("LOCAL_RANK", "1")
    configure_fp8_block_scaling()
    assert queried_devices == []
    assert KEY not in relax.utils.env.os.environ


def test_fp8_rejects_already_imported_inconsistent_recipe(monkeypatch):
    _runtime(monkeypatch)
    params = SimpleNamespace(power_2_scale=True)
    recipe = SimpleNamespace(fp8_quant_fwd_inp=params, fp8_quant_fwd_weight=params, fp8_quant_bwd_grad=params)
    monkeypatch.setitem(sys.modules, "transformer_engine.common.recipe", SimpleNamespace(Float8BlockScaling=recipe))
    with pytest.raises(RuntimeError, match="before importing Transformer Engine"):
        configure_fp8_block_scaling()
