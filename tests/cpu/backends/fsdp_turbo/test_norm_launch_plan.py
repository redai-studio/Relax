# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def test_device_query_failure_is_not_cached(monkeypatch):
    # Only isolate Triton compilation/driver access; execute the production planner.
    triton = ModuleType("triton")
    triton.jit = lambda **kwargs: lambda fn: fn
    language = ModuleType("triton.language")
    language.constexpr = object
    triton.language = language
    calls = []

    def query(device):
        calls.append(device)
        if len(calls) == 1:
            raise ValueError("invalid NPU_DEVICE_LIMIT")
        return {"num_vectorcore": 24}

    triton.runtime = SimpleNamespace(
        driver=SimpleNamespace(active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=query)))
    )
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    path = Path(__file__).resolve().parents[4] / "relax/backends/fsdp_turbo/performance_ops/grouped_rmsnorm_kernels.py"
    spec = importlib.util.spec_from_file_location("norm_launch_plan_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = SimpleNamespace(type="npu", index=0)
    with pytest.raises(ValueError, match="NPU_DEVICE_LIMIT"):
        module._num_cores(device)
    assert module._num_cores(device) == 24
    assert module._num_cores(device) == 24
    assert len(calls) == 2
