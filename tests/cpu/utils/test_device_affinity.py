import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.utils import device


@pytest.mark.parametrize("gpu_mask,expected", [(1 << 8, {8}), (3, {8, 9})])
def test_numa_affinity_uses_visible_uuid_and_respects_launcher_mask(monkeypatch, gpu_mask, expected):
    nvml = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        nvmlDeviceGetHandleByUUID=Mock(return_value="physical-gpu-4"),
        nvmlDeviceGetCpuAffinity=Mock(return_value=[gpu_mask]),
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(device, "_detect_accelerator", lambda: device.AcceleratorType.CUDA)
    monkeypatch.setattr(device.torch.cuda, "get_device_properties", lambda rank: SimpleNamespace(uuid="GPU-four"))
    monkeypatch.setattr(device.os, "sched_getaffinity", lambda pid: {8, 9})
    setter = Mock()
    monkeypatch.setattr(device.os, "sched_setaffinity", setter)
    device.set_numa_affinity(0)
    nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-four")
    nvml.nvmlDeviceGetCpuAffinity.assert_called_once_with("physical-gpu-4", 1)
    setter.assert_called_once_with(0, expected)
    nvml.nvmlShutdown.assert_called_once()


def test_numa_failure_releases_nvml_without_changing_cpu_mask(monkeypatch):
    nvml = SimpleNamespace(
        nvmlInit=Mock(), nvmlShutdown=Mock(), nvmlDeviceGetHandleByUUID=Mock(side_effect=OSError("gone"))
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(device, "_detect_accelerator", lambda: device.AcceleratorType.CUDA)
    monkeypatch.setattr(device.torch.cuda, "get_device_properties", lambda rank: SimpleNamespace(uuid="GPU-four"))
    monkeypatch.setattr(device.os, "sched_getaffinity", lambda pid: {8, 9})
    setter = Mock()
    monkeypatch.setattr(device.os, "sched_setaffinity", setter)
    device.set_numa_affinity(0)
    setter.assert_not_called()
    nvml.nvmlShutdown.assert_called_once()
