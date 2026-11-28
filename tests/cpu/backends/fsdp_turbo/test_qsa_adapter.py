# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU contracts for QSA metadata and routing; kernels remain external
boundaries."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from relax.backends.fsdp_turbo import qwen4_attention as adapter
from relax.backends.fsdp_turbo.performance_config import PerformanceConfig


class Qwen4ExpModel(nn.Module):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        return kwargs


def config(implementation="npu"):
    return SimpleNamespace(
        relax_performance=PerformanceConfig(attention_implementation=implementation),
        distributed=SimpleNamespace(ulysses_parallel_size=2),
    )


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("positional", [False, True])
def test_qsa_sequence_extent_survives_model_call(embedded, positional, monkeypatch):
    calls = []
    original = adapter._capture_cp_sequence_length

    def capture(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(adapter, "_capture_cp_sequence_length", capture)
    model = Qwen4ExpModel()
    adapter.install(model, config())
    adapter.install(model, config())
    tensor = torch.zeros((1, 7, 3) if embedded else (1, 7))
    # Valid tokens differ from the unpadded sequence extent.
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0]])
    if positional:
        args = (None, mask, None, None, tensor) if embedded else (tensor, mask)
        result = model(*args, cp_sequence_length=999, sentinel="retained")
    else:
        inputs = {"inputs_embeds" if embedded else "input_ids": tensor}
        result = model(**inputs, attention_mask=mask, cp_sequence_length=999, sentinel="retained")
    assert result == {"cp_sequence_length": 7, "sentinel": "retained"}
    assert calls == [1]


def test_qsa_native_leaves_model_kwargs_unchanged():
    model = Qwen4ExpModel()
    adapter.install(model, config("native"))
    assert model(torch.zeros((1, 7)), sentinel="retained") == {"sentinel": "retained"}


def backend_boundaries(monkeypatch, cp_size=2, ulysses_size=2):
    group = object()
    state = SimpleNamespace(is_group_enable=lambda name: True, get_cp_group_size=lambda: cp_size)
    modules = {
        "fsdp_turbo.distributed.parallel_state": {"get_parallel_state": lambda: state},
        "fsdp_turbo.distributed.context_parallel.ulysses.utils": {
            "get_ulysses_context": lambda: (group, 0, ulysses_size)
        },
        "fsdp_turbo.distributed.context_parallel.kv_allgather.utils": {
            "get_kv_allgather_context": lambda: (None, 0, 1)
        },
    }
    for name, members in modules.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    return group


@pytest.mark.parametrize("grouped", [False, True])
def test_qsa_dispatch_preserves_group_and_length(monkeypatch, grouped):
    group = backend_boundaries(monkeypatch)
    calls = []

    def kernel(name):
        def capture(self, hidden, positions, mask, **kwargs):
            calls.append((name, hidden, positions, mask, kwargs))
            return "kernel-result"

        return capture

    boundary = ModuleType("relax.backends.fsdp_turbo.performance_ops.qsa_context_parallel")
    boundary.qsa_forward = kernel("standard")
    boundary.qsa_grouped_forward = kernel("grouped")
    monkeypatch.setitem(sys.modules, boundary.__name__, boundary)
    layer = SimpleNamespace(_relax_grouped_qsa=grouped)
    result = adapter.qsa_context_parallel_forward(layer, "hidden", "positions", "mask", cp_sequence_length=7)
    assert result == "kernel-result"
    assert calls == [
        (
            "grouped" if grouped else "standard",
            "hidden",
            "positions",
            "mask",
            {"ulysses_group": group, "cp_sequence_length": 7},
        )
    ]


@pytest.mark.parametrize("topology,cache", [(True, None), (False, object())])
def test_qsa_rejects_unsupported_runtime_before_kernel(monkeypatch, topology, cache):
    backend_boundaries(monkeypatch, cp_size=4 if topology else 2)
    with pytest.raises(ValueError, match="Ulysses CP only" if topology else "without a KV cache"):
        adapter.qsa_context_parallel_forward(SimpleNamespace(), None, None, None, past_key_values=cache)


def test_qsa_cp1_uses_original_attention(monkeypatch):
    backend_boundaries(monkeypatch, cp_size=1, ulysses_size=1)
    module = ModuleType("fsdp_turbo.models.qwen.modeling_qwen4_exp")

    class OriginalAttention:
        def forward(self, hidden, positions, mask, cache, **kwargs):
            return hidden, positions, mask, cache, kwargs

    module.Qwen4ExpTextAttention = OriginalAttention
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert adapter.qsa_context_parallel_forward(SimpleNamespace(), "hidden", "positions", "mask", sentinel=7) == (
        "hidden",
        "positions",
        "mask",
        None,
        {"sentinel": 7},
    )


def test_chunk_mask_preserves_padding_and_causal_positions():
    from relax.backends.fsdp_turbo.performance_ops.qsa_context_parallel import _ids_chunk_mask

    ids = torch.tensor([[[0, -1], [0, 1], [0, 3], [2, 3]]])
    validity = torch.tensor([[True, True, True, False]])
    mask = _ids_chunk_mask(ids, validity, 1, 4)
    assert mask[0, 0].tolist() == [
        [True, True, False, False],
        [True, False, False, False],
        [False, False, True, False],
    ]


def test_configured_patch_installs_on_real_turbo_attention(monkeypatch):
    patches = pytest.importorskip("fsdp_turbo.utils.patch")
    model_module = pytest.importorskip("fsdp_turbo.models.qwen.modeling_qwen4_exp")
    # No weights or accelerator allocation: test the real target class and installer.
    layer = model_module.Qwen4ExpTextAttention.__new__(model_module.Qwen4ExpTextAttention)
    nn.Module.__init__(layer)
    model = nn.Sequential(layer)
    cfg = config()
    cfg.module_patches = []
    adapter.configure(cfg, SimpleNamespace(mtp_num_layers=0))
    adapter.install(model, cfg)
    patches.apply_module_patches(model, cfg.module_patches)
    group = backend_boundaries(monkeypatch)
    boundary = ModuleType("relax.backends.fsdp_turbo.performance_ops.qsa_context_parallel")
    boundary.qsa_forward = lambda self, hidden, positions, mask, **kwargs: kwargs
    boundary.qsa_grouped_forward = boundary.qsa_forward
    monkeypatch.setitem(sys.modules, boundary.__name__, boundary)
    assert layer(None, None, None, cp_sequence_length=7) == {"ulysses_group": group, "cp_sequence_length": 7}


def test_triton_indexer_install_preserves_forward_and_updates_options():
    from dataclasses import replace

    class Qwen4ExpTextQSAIndexer(nn.Module):
        def forward(self, value):
            return value

    indexer = Qwen4ExpTextQSAIndexer()
    cfg = config()
    cfg.relax_performance = replace(
        cfg.relax_performance, qsa_triton_validate_prefix_mask=False, qsa_triton_query_chunk_size=64
    )
    adapter.install(indexer, cfg)
    assert indexer.qsa_triton_validate_prefix_mask is False
    assert indexer.qsa_triton_query_chunk_size == 64
    assert indexer.forward.__func__ is Qwen4ExpTextQSAIndexer.forward
    cfg.relax_performance = replace(
        cfg.relax_performance, qsa_triton_validate_prefix_mask=True, qsa_triton_query_chunk_size=0
    )
    adapter.install(indexer, cfg)
    assert indexer.qsa_triton_validate_prefix_mask is True
    assert indexer.qsa_triton_query_chunk_size is None


def test_grouped_validation_checks_once_and_invalidates_on_mutation():
    from torch.utils._python_dispatch import TorchDispatchMode

    from relax.backends.fsdp_turbo.performance_ops.qsa_grouped_local import validate_grouped_mask

    class CountHostReads(TorchDispatchMode):
        reads = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                self.reads += 1
            return func(*args, **(kwargs or {}))

    mask = torch.ones(1, 1, 257, 257, dtype=torch.bool).tril()
    with CountHostReads() as counter:
        validate_grouped_mask(mask, 257)
        validate_grouped_mask(mask, 257)
        assert counter.reads == 1
        mask[..., -1] = False
        with pytest.raises(ValueError, match="unpadded causal"):
            validate_grouped_mask(mask, 257)
        assert counter.reads == 2
