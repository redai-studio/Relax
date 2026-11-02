# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU-only state-machine tests, loaded without Megatron/SGLang imports."""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_PATH = Path(__file__).resolve().parents[3] / "relax/backends/megatron/weight_update/ipc_allocator.py"
_SPEC = importlib.util.spec_from_file_location("tested_ipc_allocator", _PATH)
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


class Allocator:
    def __init__(self, settings, enabled, modern=True, backend="native", hip=None):
        self.settings, self.enabled = settings, enabled
        self.calls = []
        self.snapshots = 0
        self.fail_once = False
        self.metadata_missing = False
        core = SimpleNamespace()
        if modern:
            core._accelerator_getAllocatorSettings = lambda: self.settings
            core._accelerator_setAllocatorSettings = self.set
        memory = SimpleNamespace(
            get_allocator_backend=lambda: backend, _snapshot=self.snapshot, _set_allocator_settings=self.set
        )
        self.torch = SimpleNamespace(_C=core, cuda=SimpleNamespace(memory=memory), version=SimpleNamespace(hip=hip))

    def set(self, settings):
        self.calls.append(settings)
        self.settings = settings
        matches = re.findall(r"expandable_segments\s*:\s*(True|False)", settings)
        if matches:
            self.enabled = matches[-1] == "True"
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("setter failure after partial mutation")

    def snapshot(self):
        self.snapshots += 1
        if self.metadata_missing:
            return {}
        return {"allocator_settings": {"PYTORCH_CUDA_ALLOC_CONF": self.settings, "expandable_segments": self.enabled}}


def make(monkeypatch, settings="expandable_segments:True", enabled=True, device="cuda", **kwargs):
    allocator = Allocator(settings, enabled, **kwargs)
    monkeypatch.setattr(module, "torch", allocator.torch)
    return allocator, module.IpcBucketAllocator(SimpleNamespace(type=device))


def test_preserves_other_options_and_restores_before_serialization(monkeypatch):
    raw = "max_split_size_mb:256,roundup_power2_divisions:[256:1,512:2,>:4],expandable_segments:True"
    allocator, scope = make(monkeypatch, raw)
    for _ in range(3):
        with scope.allocation():
            assert not allocator.enabled
            assert allocator.settings == raw.replace("segments:True", "segments:False")
        # Metadata serialization and collectives occur after this scope.
        assert allocator.enabled and allocator.settings == raw
    assert allocator.snapshots == 0


def test_nested_active_scope_does_not_restore_early(monkeypatch):
    allocator, scope = make(monkeypatch)
    with scope.allocation():
        with scope.allocation():
            assert not allocator.enabled
        assert not allocator.enabled
    assert allocator.enabled
    assert len(allocator.calls) == 2


def test_explicit_disabled_and_nested_noop(monkeypatch):
    allocator, scope = make(monkeypatch, "max_split_size_mb:128,expandable_segments:False", False)
    with scope.allocation():
        with scope.allocation():
            assert not allocator.enabled
    assert allocator.calls == [] and allocator.snapshots == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_partial_config_reads_effective_state_only_once(monkeypatch, enabled):
    raw = "max_split_size_mb:256"
    allocator, scope = make(monkeypatch, raw, enabled)
    assert allocator.snapshots == 1
    for _ in range(4):
        with scope.allocation():
            assert not allocator.enabled
        assert allocator.enabled == enabled and allocator.settings == raw
    assert allocator.snapshots == 1
    if enabled:
        assert allocator.calls[-2:] == [raw + ",expandable_segments:True", raw]
    else:
        assert allocator.calls == []


def test_bucket_exception_restores_original_state(monkeypatch):
    allocator, scope = make(monkeypatch)
    with pytest.raises(ValueError, match="bucket failed"):
        with scope.allocation():
            raise ValueError("bucket failed")
    assert allocator.enabled and allocator.settings == "expandable_segments:True"


def test_setter_partial_failure_restores_original_state(monkeypatch):
    allocator, scope = make(monkeypatch)
    allocator.fail_once = True
    with pytest.raises(RuntimeError, match="setter failure"):
        with scope.allocation():
            pytest.fail("must not run allocation after setter failure")
    assert allocator.enabled and allocator.settings == "expandable_segments:True"


def test_external_config_change_fails_without_overwriting(monkeypatch):
    allocator, scope = make(monkeypatch)
    allocator.settings = "max_split_size_mb:512,expandable_segments:True"
    with pytest.raises(RuntimeError, match="configuration changed"):
        with scope.allocation():
            pass
    assert allocator.calls == []


def test_legacy_getter_uses_one_snapshot(monkeypatch):
    allocator, scope = make(monkeypatch, modern=False)
    for _ in range(3):
        with scope.allocation():
            assert not allocator.enabled
    assert allocator.enabled and allocator.snapshots == 1


@pytest.mark.parametrize(
    "kwargs", [dict(device="cpu"), dict(device="npu"), dict(hip="6.4"), dict(backend="cudaMallocAsync")]
)
def test_other_backends_untouched(monkeypatch, kwargs):
    allocator, scope = make(monkeypatch, **kwargs)
    with scope.allocation():
        pass
    assert allocator.calls == [] and allocator.snapshots == 0


def test_missing_effective_state_fails_explicitly(monkeypatch):
    allocator = Allocator("", True)
    allocator.metadata_missing = True
    monkeypatch.setattr(module, "torch", allocator.torch)
    with pytest.raises(RuntimeError, match="Cannot determine"):
        module.IpcBucketAllocator(SimpleNamespace(type="cuda"))


@pytest.mark.skipif(
    module.torch.version.cuda is None or module.torch.version.hip is not None,
    reason="native allocator parser requires a CUDA-enabled PyTorch build (no GPU needed)",
)
@pytest.mark.parametrize("settings", ["", "   ", "max_split_size_mb:128,", "max_split_size_mb:128,   "])
def test_partial_config_native_parser_restores_exact_state(settings):
    # Isolate process-global configuration; use the real parser without allocating GPU memory.
    script = """
import importlib.util
import sys
import torch

spec = importlib.util.spec_from_file_location("tested_ipc_allocator", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.cuda.memory._set_allocator_settings("expandable_segments:True")
torch.cuda.memory._set_allocator_settings(sys.argv[2])
before = torch.cuda.memory._snapshot()["allocator_settings"]
scope = module.IpcBucketAllocator(torch.device("cuda"))
for fail in (False, True):
    try:
        with scope.allocation():
            during = torch.cuda.memory._snapshot()["allocator_settings"]
            assert during["expandable_segments"] is False
            for key, value in before.items():
                if key not in ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments"):
                    assert during[key] == value
            if fail:
                raise ValueError("bucket failed")
    except ValueError:
        assert fail
    else:
        assert not fail
    assert torch.cuda.memory._snapshot()["allocator_settings"] == before
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(_PATH), settings],
        env={**os.environ, "PYTORCH_ALLOC_CONF": "backend:native", "PYTORCH_CUDA_ALLOC_CONF": "backend:native"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
