# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU checks for mixed dispatch and rejection of removed community
backends."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
ENV = "RELAX_QWEN38_QSA_COMMUNITY_BACKEND"


@pytest.fixture
def adapter(monkeypatch):
    variant = ModuleType("megatron.core.transformer.experimental_attention_variant")
    native = Mock(return_value="native")
    variant.qsa = SimpleNamespace(
        QSACoreAttention=SimpleNamespace(_relax_native_id_sparse_forward=native), QSASelection=object
    )
    monkeypatch.setitem(sys.modules, variant.__name__, variant)
    spec = importlib.util.spec_from_file_location("mixed_adapter_test", ROOT / "relax/models/qwen3_8/qsa_community.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, native


@pytest.mark.parametrize("backend", ["pr7553", "tilelang7060", "unknown"])
def test_removed_community_backends_fail_with_migration_hint(adapter, monkeypatch, backend):
    module, _ = adapter
    monkeypatch.setenv(ENV, backend)
    with pytest.raises(ValueError, match="only 'mixed' is retained") as error:
        module.validate_backend()
    assert "Clear RELAX_QWEN38_QSA_COMMUNITY_BACKEND" in str(error.value)


@pytest.mark.parametrize("sparse_backend", [None, "id_sparse", "flex", "dense_masked"])
def test_mixed_validation_keeps_native_selected_id_contract(adapter, monkeypatch, sparse_backend):
    module, _ = adapter
    monkeypatch.setenv(ENV, "mixed")
    monkeypatch.delenv("MCORE_QSA_SPARSE_BACKEND", raising=False)
    if sparse_backend is not None:
        monkeypatch.setenv("MCORE_QSA_SPARSE_BACKEND", sparse_backend)
    if sparse_backend in (None, "id_sparse"):
        module.validate_backend()
    else:
        with pytest.raises(ValueError, match="requires MCORE_QSA_SPARSE_BACKEND=id_sparse"):
            module.validate_backend()


@pytest.mark.parametrize("selected_ids", [None, object()])
def test_native_fallback_keeps_original_arguments(adapter, selected_ids):
    module, native = adapter
    owner, selection = object(), SimpleNamespace(selected_ids=selected_ids)
    query, key, value = (torch.zeros(5, 1, 256) for _ in range(3))
    assert module._community_id_sparse_forward(owner, query, key, value, selection, True) == "native"
    native.assert_called_once_with(owner, query, key, value, selection, True)


@pytest.mark.parametrize("is_thd", [False, True])
def test_mixed_dispatch_preserves_tensor_layout(adapter, monkeypatch, is_thd):
    """Only the CUDA gate is simulated; layout operations run on real CPU
    tensors."""
    module, native = adapter

    class CudaLikeTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    batch, length = (1 if is_thd else 2), 5
    values = [
        torch.arange(length * batch * heads * 256, dtype=torch.float32)
        .reshape(length, batch, heads, 256)
        .to(torch.bfloat16)
        .as_subclass(CudaLikeTensor)
        for heads in (12, 1, 1)
    ]
    selection = SimpleNamespace(
        selected_ids=object(), compress_ratio=4, positions=torch.zeros(batch, length, dtype=torch.int32)
    )
    attention = Mock(side_effect=lambda q, k, v, *args: q + k + v)
    monkeypatch.setitem(
        sys.modules, "relax.models.qwen3_8.ops.qsa_mixed_ops", SimpleNamespace(attention_for_selection=attention)
    )
    inputs = [value.squeeze(1) if is_thd else value for value in values]
    output = module._community_id_sparse_forward(SimpleNamespace(softmax_scale=0.125), *inputs, selection, is_thd)
    expected = sum(values).reshape(length, batch, 12 * 256)
    torch.testing.assert_close(output, expected.squeeze(1) if is_thd else expected)
    q, k, v, selected, scale, packed = attention.call_args.args
    assert q.shape == (batch, length, 12, 256)
    assert k.shape == v.shape == (batch, length, 1, 256)
    assert selected is selection and scale == 0.125 and packed is is_thd
    native.assert_not_called()
