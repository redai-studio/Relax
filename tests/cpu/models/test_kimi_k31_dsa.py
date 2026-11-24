# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Numerical K3.1 sparse MLA checks using real Core selection and attention.

Only Bridge's GPU projection construction is replaced with CPU linears. The
indexer, Core DSA masking/top-k, absorption, and autograd paths remain real.
"""

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


class _Linear(torch.nn.Linear):
    def forward(self, value):
        return super().forward(value), None


class _CPUAttention(torch.nn.Module):
    def _init_mla(self, config):
        self.local_num_heads = 3
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = 2
        self.qk_extra_head_dim = config.qk_pos_emb_head_dim
        self.v_head_dim = 3
        self.q_head_dim = self.qk_nope_head_dim + self.qk_extra_head_dim
        self.q_a_proj = _Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = torch.nn.RMSNorm(self.q_lora_rank, eps=1e-5)
        self.q_b_proj = _Linear(self.q_lora_rank, self.local_num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = _Linear(config.hidden_size, self.kv_lora_rank + self.qk_extra_head_dim, bias=False)
        self.kv_a_layernorm = torch.nn.RMSNorm(self.kv_lora_rank, eps=1e-5)
        self.kv_b_proj = _Linear(
            self.kv_lora_rank, self.local_num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.g_proj = _Linear(config.hidden_size, self.local_num_heads * self.v_head_dim, bias=False)
        self.o_proj = _Linear(self.local_num_heads * self.v_head_dim, config.hidden_size, bias=False)


@pytest.fixture
def dsa_module(monkeypatch, tmp_path):
    native = pytest.importorskip("megatron.core.transformer.experimental_attention_variant.dsa")
    if not hasattr(native.DSAttention, "consumes_absorbed_v_up_projection"):
        pytest.skip("Installed Megatron Core lacks the pinned absorbed DSA implementation.")
    upstream = ModuleType("megatron.bridge.models.kimi.kimi_k3_layers")
    upstream.KimiK3Attention = _CPUAttention
    upstream._linear = lambda module, value: module(value)[0]
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    path = Path(__file__).resolve().parents[3] / "relax/models/kimi_k3/dsa.py"
    spec = importlib.util.spec_from_file_location("_relax_k31_dsa_numerics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owns_group = not torch.distributed.is_initialized()
    if owns_group:
        torch.distributed.init_process_group(
            "gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1, timeout=timedelta(seconds=30)
        )
    try:
        yield module
    finally:
        if owns_group:
            torch.distributed.destroy_process_group()


def _config(topk=2, dtype=torch.float32):
    return SimpleNamespace(
        hidden_size=6,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_pos_emb_head_dim=1,
        use_cpu_initialization=True,
        params_dtype=dtype,
        sequence_parallel=True,
        dsa_kernel_backend="none",
        attention_backend="unfused",
        dsa_indexer_use_sparse_loss=False,
        calculate_per_token_loss=False,
        kimi_dsa_config={"index_num_heads": 3, "index_head_dim": 4, "index_topk": topk, "dsa_layers": [2]},
    )


def _attention(module, topk=2):
    config = _config(topk)
    attention = module.KimiK31DSAAttention.__new__(module.KimiK31DSAAttention)
    torch.nn.Module.__init__(attention)
    attention.config = config
    attention.layer_idx = 1
    attention.tp_group = torch.distributed.group.WORLD
    attention.pg_collection = SimpleNamespace(tp=attention.tp_group, cp=None)
    attention._init_mla(config)
    return attention


def _reference(attention, hidden, boundaries, topk):
    """Independent, expanded MLA with explicit per-sample causal top-k."""
    q_lora = attention.q_a_layernorm(attention.q_a_proj(hidden)[0])
    query = attention.q_b_proj(q_lora)[0].view(-1, 3, 3)
    kv_latent, key_extra = attention.kv_a_proj_with_mqa(hidden)[0].split((4, 1), dim=-1)
    key_nope, value = attention.kv_b_proj(attention.kv_a_layernorm(kv_latent))[0].view(-1, 3, 5).split((2, 3), -1)
    key = torch.cat((key_nope, key_extra.view(-1, 1, 1).expand(-1, 3, -1)), -1)
    indexer = attention.core_attention.indexer
    index_q = F.linear(q_lora, indexer.wq_b.weight).view(-1, indexer.num_heads, indexer.head_dim)
    index_k = F.layer_norm(
        F.linear(hidden, indexer.wk.weight).float(),
        (indexer.head_dim,),
        indexer.k_norm.weight.float(),
        indexer.k_norm.bias.float(),
        1e-5,
    ).squeeze(1)
    weights = (F.linear(hidden, indexer.weights_proj.weight) * indexer.num_heads**-0.5).float()
    weights = weights.squeeze(1) * indexer.head_dim**-0.5
    outputs = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        for row in range(start, stop):
            score = (F.relu(index_q[row] @ index_k[start : row + 1].T) * weights[row, :, None]).sum(0)
            selected = score.topk(min(topk, row - start + 1)).indices + start
            logits = torch.einsum("hd,khd->hk", query[row], key[selected]) * 3**-0.5
            outputs.append(torch.einsum("hk,khd->hd", logits.softmax(-1), value[selected]))
    output = torch.stack(outputs).reshape(hidden.size(0), 1, -1)
    return attention.o_proj(output * attention.g_proj(hidden)[0].sigmoid())[0]


@pytest.mark.parametrize("topk", [2, 4, 16])
def test_kimi_k31_absorbed_matches_expanded_sparse_attention_and_gradients(dsa_module, topk):
    from megatron.core.packed_seq_params import PackedSeqParams

    torch.manual_seed(421)
    attention = _attention(dsa_module, topk)
    reference = _attention(dsa_module, topk)
    reference.load_state_dict(attention.state_dict())
    hidden = torch.randn(7, 1, 6, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    boundaries = [0, 4, 7]
    cu = torch.tensor(boundaries, dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=4, max_seqlen_kv=4)
    packed.cu_seqlens_q_cpu = boundaries
    actual = attention._forward_mla(hidden, packed)
    expected = _reference(reference, reference_hidden, boundaries, topk)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    expected.backward(cotangent)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=3e-6, rtol=3e-5)
    reference_params = dict(reference.named_parameters())
    for name, parameter in attention.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, reference_params[name].grad, atol=3e-6, rtol=3e-5)
        else:
            assert parameter.grad is None
    assert attention.config.sequence_parallel is True
    assert attention.core_attention.config.sequence_parallel is False
    assert attention.core_attention.config.dsa_kernel_backend == "tilelang"


def test_kimi_k31_indexer_bf16_rounding_and_fp32_layernorm(dsa_module):
    torch.manual_seed(17)
    attention = _attention(dsa_module)
    indexer = attention.core_attention.indexer.to(torch.bfloat16)
    hidden = torch.randn(11, 1, 6, dtype=torch.bfloat16)
    qr = torch.randn(11, 1, 4, dtype=torch.bfloat16)
    query, key, weights = indexer.forward_before_topk(hidden, qr)
    raw = F.linear(hidden, indexer.weights_proj.weight)
    expected = (raw * 3**-0.5).float() * 4**-0.5
    torch.testing.assert_close(weights, expected, atol=0, rtol=0)
    assert not torch.equal(weights, raw.float() * (3**-0.5 * 4**-0.5))
    expected_key = F.layer_norm(
        F.linear(hidden, indexer.wk.weight).float(),
        (4,),
        indexer.k_norm.weight.float(),
        indexer.k_norm.bias.float(),
        1e-5,
    ).bfloat16()
    torch.testing.assert_close(key, expected_key, atol=0, rtol=0)
    torch.testing.assert_close(query, F.linear(qr, indexer.wq_b.weight).unflatten(-1, (3, 4)), atol=0, rtol=0)
    assert all(not parameter.requires_grad for parameter in indexer.parameters())


def test_kimi_k31_absorbed_weight_accumulates_into_core_main_grad(dsa_module):
    from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel

    torch.manual_seed(73)
    attention = _attention(dsa_module)
    reference = _attention(dsa_module)
    reference.load_state_dict(attention.state_dict())
    weight = attention.kv_b_proj.weight
    weight.main_grad = torch.zeros_like(weight)
    weight.grad_added_to_main_grad = False
    ddp = SimpleNamespace(param_to_bucket_group={weight: None}, ddp_config=SimpleNamespace(overlap_grad_reduce=False))
    hook = weight.register_post_accumulate_grad_hook(DistributedDataParallel._make_backward_post_hook(ddp, weight))
    try:
        for _ in range(2):
            hidden = torch.randn(5, 1, 6)
            attention._forward_mla(hidden, None).square().sum().backward()
            _reference(reference, hidden, [0, 5], 2).square().sum().backward()
        assert weight.grad is None
        torch.testing.assert_close(weight.main_grad, reference.kv_b_proj.weight.grad, atol=3e-6, rtol=3e-5)
    finally:
        hook.remove()


class _Adapter(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.to_wrap = base
        self.adapter = torch.nn.Linear(base.in_features, base.out_features, bias=False)

    @property
    def weight(self):
        raise AssertionError("Adapter weights must be consumed through their forward method.")

    def forward(self, hidden):
        return self.to_wrap(hidden)[0] + self.adapter(hidden), None


def test_kimi_k31_kv_adapter_preserves_sparse_attention_and_adapter_gradients(dsa_module):
    torch.manual_seed(413)
    attention = _attention(dsa_module)
    attention.kv_b_proj = _Adapter(attention.kv_b_proj)
    reference = _attention(dsa_module)
    reference.kv_b_proj = _Adapter(reference.kv_b_proj)
    reference.load_state_dict(attention.state_dict())
    hidden = torch.randn(5, 1, 6, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    actual = attention._forward_mla(hidden, None)
    expected = _reference(reference, reference_hidden, [0, 5], 2)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(
        attention.kv_b_proj.adapter.weight.grad, reference.kv_b_proj.adapter.weight.grad, atol=3e-6, rtol=3e-5
    )


@pytest.fixture
def recompute_module(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "relax/models/kimi_k3/dsa_recompute.py"
    spec = importlib.util.spec_from_file_location("_k31_dsa_recompute", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    # CPU tests use torch's real reentrant checkpoint; the GPU test additionally
    # exercises Core's checkpoint/RNG implementation and this native flag.
    monkeypatch.setattr(module, "_is_checkpointing", lambda: True)
    return module


def test_kimi_k31_selection_reuse_isolates_inflight_microbatches_and_repeated_backward(dsa_module, recompute_module):
    import weakref

    from megatron.core.packed_seq_params import PackedSeqParams
    from torch.utils.checkpoint import checkpoint

    class Block:
        config = SimpleNamespace(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)

        def __init__(self, attention):
            self.attention = attention
            self.scopes = []

        def _checkpointed_forward(self, hidden, packed_seq_params):
            scope = getattr(packed_seq_params, recompute_module._SCOPE_ATTR, None)
            if scope is not None:
                self.scopes.append(weakref.ref(scope))
            return checkpoint(lambda x: self.attention._forward_mla(x, packed_seq_params), hidden, use_reentrant=True)

    torch.manual_seed(71)
    attention, reference = _attention(dsa_module, 2), _attention(dsa_module, 2)
    reference.load_state_dict(attention.state_dict())
    block, baseline = Block(attention), Block(reference)
    recompute_module.configure_checkpoint_scope(block)
    recompute_module.configure_indexer_recompute_reuse(attention.core_attention)
    calls = []
    native = attention.core_attention.indexer.forward_before_topk

    def counted(*args, **kwargs):
        calls.append(1)
        return native(*args, **kwargs)

    attention.core_attention.indexer.forward_before_topk = counted
    cu = torch.tensor([0, 4, 7], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=4, max_seqlen_kv=4)
    inputs = [torch.randn(7, 1, 6, requires_grad=True) for _ in range(2)]
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    actual = [block._checkpointed_forward(x, packed) for x in inputs]
    expected = [baseline._checkpointed_forward(x, packed) for x in refs]
    assert len(calls) == 2
    assert not hasattr(packed, recompute_module._SCOPE_ATTR)
    assert block.scopes[0]() is not block.scopes[1]()
    assert all(len(scope().selections) == 1 for scope in block.scopes)
    cotangents = [torch.randn_like(x) for x in actual]
    for index in (1, 0):
        torch.testing.assert_close(actual[index], expected[index])
        actual[index].backward(cotangents[index], retain_graph=True)
        expected[index].backward(cotangents[index], retain_graph=True)
    assert len(calls) == 2, "Checkpoint replay must reuse the original top-k"
    assert all(not scope().selections for scope in block.scopes)
    # Repeated backward deliberately recomputes after the one-use entry was consumed.
    actual[0].backward(cotangents[0])
    expected[0].backward(cotangents[0])
    assert len(calls) == 3
    for x, ref in zip(inputs, refs, strict=True):
        torch.testing.assert_close(x.grad, ref.grad, atol=3e-6, rtol=3e-5)
    for (name, parameter), (_, ref) in zip(attention.named_parameters(), reference.named_parameters(), strict=True):
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, ref.grad, atol=3e-6, rtol=3e-5, msg=name)
    assert not attention.core_attention.index_share


def test_kimi_k31_selection_reuse_releases_abandoned_graph(dsa_module, recompute_module):
    import gc
    import weakref

    from megatron.core.packed_seq_params import PackedSeqParams
    from torch.utils.checkpoint import checkpoint

    attention = _attention(dsa_module, 2)
    recompute_module.configure_indexer_recompute_reuse(attention.core_attention)
    owners = []

    class Block:
        config = SimpleNamespace(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)

        def _checkpointed_forward(self, hidden, packed_seq_params):
            owners.append(weakref.ref(getattr(packed_seq_params, recompute_module._SCOPE_ATTR)))
            return checkpoint(lambda x: attention._forward_mla(x, packed_seq_params), hidden, use_reentrant=True)

    block = Block()
    recompute_module.configure_checkpoint_scope(block)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=4, max_seqlen_kv=4)
    out = block._checkpointed_forward(torch.randn(4, 1, 6, requires_grad=True), packed)
    assert owners[0]().selections
    del out
    gc.collect()
    assert owners[0]() is None
    out = block._checkpointed_forward(torch.randn(4, 1, 6, requires_grad=True), packed)
    out.sum().backward()
    assert not hasattr(packed, recompute_module._SCOPE_ATTR)


def test_kimi_k31_selection_scope_clears_partial_forward_on_error(dsa_module, recompute_module):
    from megatron.core.packed_seq_params import PackedSeqParams
    from torch.utils.checkpoint import checkpoint

    attention = _attention(dsa_module, 2)
    recompute_module.configure_indexer_recompute_reuse(attention.core_attention)
    scopes = []

    class Block:
        config = SimpleNamespace(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
        fail = True

        def _checkpointed_forward(self, hidden, packed_seq_params):
            scope = getattr(packed_seq_params, recompute_module._SCOPE_ATTR)
            scopes.append(scope)
            output = checkpoint(lambda x: attention._forward_mla(x, packed_seq_params), hidden, use_reentrant=True)
            assert scope.selections
            if self.fail:
                raise RuntimeError("later layer failed")
            return output

    block = Block()
    recompute_module.configure_checkpoint_scope(block)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=4, max_seqlen_kv=4)
    with pytest.raises(RuntimeError, match="later layer failed"):
        block._checkpointed_forward(torch.randn(4, 1, 6, requires_grad=True), packed)
    assert not scopes[0].selections
    block.fail = False
    output = block._checkpointed_forward(torch.randn(4, 1, 6, requires_grad=True), packed)
    output.sum().backward()
    assert not scopes[1].selections
