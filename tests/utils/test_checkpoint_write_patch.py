# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from types import ModuleType

import pytest
import torch

from relax.utils import checkpoint_write_patch as patch


@pytest.fixture
def modern_writer(monkeypatch):
    class Writer:
        calls = []

        @staticmethod
        def preload_tensors(buckets, non_blocking=True):
            Writer.calls.append(non_blocking)
            return [
                (path, key, (data, [(item, tensor.to("cpu", non_blocking=non_blocking)) for item, tensor in tensors]))
                for path, key, (data, tensors) in buckets
            ]

    module = ModuleType("megatron.core.dist_checkpointing.strategies.filesystem_async")
    module.FileSystemWriterAsync = Writer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(patch, "_patched", False)
    monkeypatch.setattr(patch, "_blocking_staging_patched", False)
    return Writer


def test_checkpoint_staging_preserves_bucket_contents_and_is_idempotent(modern_writer):
    patch.patch_checkpoint_write(blocking_staging=True)
    installed = modern_writer.preload_tensors
    patch.patch_checkpoint_write(blocking_staging=True)
    assert modern_writer.preload_tensors is installed
    data = [("metadata", b"checkpoint")]
    value = torch.arange(12).reshape(3, 4).t()
    buckets = [("shard.distcp", "rank0", (data, [("weight", value)]))]
    result = modern_writer.preload_tensors(buckets, True)
    assert modern_writer.calls == [False]
    assert result[0][:2] == buckets[0][:2]
    assert result[0][2][0] is data
    assert result[0][2][1][0][0] == "weight"
    torch.testing.assert_close(result[0][2][1][0][1], value)


def test_checkpoint_staging_default_keeps_async_path(modern_writer):
    original = modern_writer.preload_tensors
    patch.patch_checkpoint_write()
    assert modern_writer.preload_tensors is original
    assert modern_writer.preload_tensors([]) == []
    assert modern_writer.calls == [True]


def test_checkpoint_staging_propagates_copy_failure(modern_writer):
    class FailedTensor:
        def to(self, *args, **kwargs):
            raise RuntimeError("CUDA copy failed")

    patch.patch_checkpoint_write(blocking_staging=True)
    with pytest.raises(RuntimeError, match="CUDA copy failed"):
        modern_writer.preload_tensors([("shard", "rank0", ([], [("weight", FailedTensor())]))])
