# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DeepEP initialization allocates a CUDA tensor")
def test_deepep_buffer_initializes_once_on_first_use(monkeypatch):
    from megatron.core.transformer.moe import fused_a2a

    group = object()
    allocations = []
    primed = []

    def all_reduce(tensor, *, group):
        assert tensor.is_cuda and tensor.numel() == 1
        primed.append(group)

    class Buffer:
        @staticmethod
        def get_buffer_size_hint(group, **kwargs):
            assert primed == [group]
            return 64

        def __init__(self, group, **kwargs):
            self.group = group
            self.__dict__.update(kwargs)
            allocations.append(self)

    monkeypatch.setitem(sys.modules, "_relax_deep_ep_v2", SimpleNamespace(ElasticBuffer=Buffer))
    monkeypatch.setattr(fused_a2a, "ElasticBuffer", None, raising=False)
    monkeypatch.setattr(fused_a2a, "_elastic_buffer", None)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    buffer = fused_a2a.get_elastic_buffer(group, 256, 2048, 8)
    assert buffer is fused_a2a.get_elastic_buffer(group, 256, 2048, 8)
    assert allocations == [buffer]
    assert primed == [group]
