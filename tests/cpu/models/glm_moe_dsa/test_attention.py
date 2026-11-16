# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execute GLM construction/forward with CPU substitutes for TE and GPU
kernels.

The actual MCore Attention base, GLM projections/control flow, indexer wrapper,
and autograd are exercised. This does not validate CUDA kernels or collectives.
"""

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F


class _Linear(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * 0.1)

    def forward(self, value):
        return F.linear(value, self.weight), None


class _NormLinear(_Linear):
    def __init__(self, input_size, output_size, config, **kwargs):
        super().__init__(input_size, output_size)
        self.layer_norm_weight = nn.Parameter(torch.ones(input_size))
        self.eps = config.layernorm_epsilon

    def forward(self, value):
        value = F.rms_norm(value, (value.shape[-1],), self.layer_norm_weight, self.eps)
        return super().forward(value)


class _Norm(nn.LayerNorm):
    def __init__(self, hidden_size, eps, **kwargs):
        super().__init__(hidden_size, eps=eps)


class _Identity(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, value):
        return value


class _Rotary(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.dim = dim

    def get_rotary_seq_len(self, inference_context, transformer, hidden_states, config, packed_seq_params):
        return hidden_states.shape[0]

    def forward(self, seq_len, **kwargs):
        return torch.zeros(1, seq_len, 1, self.dim)


def _sparse_attention(q, kv, indices, scale):
    selected = indices[:, 0].long()
    keys = kv[:, 0][selected.clamp_min(0)]
    scores = torch.einsum("thd,tkd->thk", q, keys) * scale
    scores = scores.masked_fill(selected[:, None] < 0, -torch.inf)
    out = torch.einsum("thk,tkd->thd", scores.softmax(-1), keys[..., :8])
    return out, scores.logsumexp(-1)


@pytest.fixture
def attention_runtime(glm_modules, monkeypatch):
    pytest.importorskip("megatron.core", reason="Requires the Dockerfile-pinned MCore source")
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    module = glm_modules("dsa_attention")
    monkeypatch.setattr(module, "TELinear", _Linear)
    monkeypatch.setattr(module, "RotaryEmbedding", _Rotary)
    monkeypatch.setattr(module, "gather_from_sequence_parallel_region", lambda value, **kwargs: value)
    monkeypatch.setattr(module, "scatter_to_sequence_parallel_region", lambda value, **kwargs: value)
    monkeypatch.setattr(module.parallel_state, "get_context_parallel_group", lambda **kwargs: None)
    monkeypatch.setattr(module.parallel_state, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(module.parallel_state, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        module,
        "WeightLinearFunction",
        SimpleNamespace(apply=lambda x, w, bias, dtype: F.linear(x.float(), w.float())),
    )
    monkeypatch.setattr(module, "SparseMLA", SimpleNamespace(apply=_sparse_attention))

    # Zero-angle rotary kernel and CPU indexer scoring replace GPU-only calls.
    apex = ModuleType("apex.transformer.functional")
    apex.fused_apply_rotary_pos_emb_thd = lambda value, cu, rotary: value
    monkeypatch.setitem(sys.modules, apex.__name__, apex)
    indexer_kernel = ModuleType("relax.models.glm_moe_dsa.ops.tilelang_indexer_fwd")
    calls = []

    def score(q, k, weights, starts, ends, **kwargs):
        calls.append(q.shape[0])
        logits = (torch.einsum("thd,kd->thk", q.float(), k.float()).relu() * weights[..., None]).sum(1)
        positions = torch.arange(k.shape[0])[None, :]
        return logits.masked_fill((positions < starts[:, None]) | (positions >= ends[:, None]), -torch.inf)

    indexer_kernel.indexer_fwd_interface = score
    monkeypatch.setitem(sys.modules, indexer_kernel.__name__, indexer_kernel)
    config = MLATransformerConfig(
        num_layers=6,
        hidden_size=8,
        num_attention_heads=2,
        num_query_groups=2,
        ffn_hidden_size=16,
        kv_channels=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_head_dim=4,
        qk_pos_emb_head_dim=2,
        v_head_dim=4,
        rope_type="rope",
        normalization="RMSNorm",
        add_bias_linear=False,
        use_cpu_initialization=True,
        perform_initialization=False,
    )
    config.glm_dsa_index_sources = (1, 2, 3, 3, 3, 3)
    config.index_num_attention_heads = 2
    config.index_head_dim = 4
    config.dsa_indexer_topk = 8
    submodules = module.DSASelfAttentionSubmodules(
        linear_q_down_proj=_Linear,
        linear_q_up_proj=_NormLinear,
        linear_kv_down_proj=_Linear,
        linear_kv_up_proj=_NormLinear,
        linear_proj=_Linear,
        core_attention=_Identity,
        q_layernorm=_Identity,
        kv_layernorm=_Identity,
        wq_b=_Linear,
        wk=_Linear,
        k_norm=_Norm,
        weights_proj=_Linear,
    )

    def build(layer, **kwargs):
        return module.DSAMLASelfAttention(
            config,
            submodules,
            layer_number=layer,
            pg_collection=SimpleNamespace(tp=None, cp=None),
            name=f"decoder.layers.{layer - 1}.self_attention",
            pp_layer_offset=0,
            **kwargs,
        )

    return module, build, calls


def _packed():
    from megatron.core.packed_seq_params import PackedSeqParams

    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    return PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)


def test_glm_shared_layers_omit_indexer_parameters_and_skip_computation(attention_runtime):
    _, build, calls = attention_runtime
    full, shared = build(3), build(4)
    assert {n for n in full.state_dict() if n.startswith(("wq_b", "wk", "k_norm", "weights_proj"))} == {
        "wq_b.weight",
        "wk.weight",
        "k_norm.weight",
        "k_norm.bias",
        "weights_proj.weight",
    }
    assert not any(n.startswith(("wq_b", "wk", "k_norm", "weights_proj")) for n in shared.state_dict())
    value = torch.randn(5, 1, 8, requires_grad=True)
    params = _packed()
    out, _ = full(value, None, packed_seq_params=params)
    out, _ = shared(out, None, packed_seq_params=params)
    out.square().sum().backward()
    assert len(calls) == 1
    assert torch.isfinite(value.grad).all()
    assert shared.linear_q_up_proj.weight.grad.abs().sum() > 0
    # An MTP layer with local number 4 still owns an independent indexer.
    assert "wq_b.weight" in build(4, is_mtp_layer=True).state_dict()


def test_glm_spec_validates_real_core_pipeline_split_and_transforms_mtp(attention_runtime, monkeypatch):
    from megatron.core.transformer.spec_utils import ModuleSpec

    module, build, _ = attention_runtime
    config = build(3).config
    config.num_layers = 78
    config.glm_dsa_index_sources = tuple(i if i <= 3 else 3 + ((i - 3) // 4) * 4 for i in range(1, 79))
    config.pipeline_model_parallel_size = 4
    config.num_layers_in_first_pipeline_stage = 22
    config.num_layers_in_last_pipeline_stage = 16
    # The external GPT spec builder does not allocate layers; Core's real PP
    # offset/layer-count functions still determine what the GLM callback builds.
    original = ModuleSpec(module=nn.Module, submodules=SimpleNamespace(self_attention=_Identity))
    monkeypatch.setattr(
        module, "get_gpt_decoder_block_spec", lambda config, **kwargs: SimpleNamespace(layer_specs=[original] * 78)
    )
    for rank, count in enumerate((22, 20, 20, 16)):
        monkeypatch.setattr(module.parallel_state, "get_pipeline_model_parallel_rank", lambda rank=rank: rank)
        result = module.get_glm5_dsa_spec(config)
        assert all(
            s.submodules.self_attention.module is module.DSAMLASelfAttention for s in result.layer_specs[:count]
        )
    monkeypatch.setattr(module.parallel_state, "get_pipeline_model_parallel_rank", lambda: 1)
    config.num_layers_in_first_pipeline_stage = None
    config.num_layers_in_last_pipeline_stage = 18
    with pytest.raises(ValueError, match="shared layer 21"):
        module.get_glm5_dsa_spec(config)
    # Bridge calls this hook for standalone MTP stages with no decoder spec.
    mtp_spec = module.transform_glm5_layer_spec(config, original)
    assert mtp_spec.submodules.self_attention.module is module.DSAMLASelfAttention
    assert original.submodules.self_attention is _Identity


def test_glm_full_recompute_preserves_two_microbatch_outputs_and_gradients(attention_runtime, monkeypatch):
    from megatron.core.tensor_parallel import random as core_random
    from megatron.core.transformer.transformer_block import TransformerBlock

    # Only CUDA RNG access is substituted; use the real Core checkpoint loop
    # and CheckpointFunction, including their packed-metadata closure.
    monkeypatch.setattr(core_random, "_get_all_rng_states", lambda: (torch.get_rng_state(),))
    monkeypatch.setattr(core_random, "_set_all_rng_states", torch.set_rng_state)

    class Layer(nn.Module):
        def __init__(self, attention):
            super().__init__()
            self.attention = attention

        def forward(self, hidden_states, packed_seq_params, **kwargs):
            return self.attention(hidden_states, None, packed_seq_params=packed_seq_params)[0], None

    _, build, calls = attention_runtime
    torch.manual_seed(7)
    layers = nn.ModuleList([build(i) for i in (3, 4, 5, 6)])
    reference = copy.deepcopy(layers)
    block = TransformerBlock.__new__(TransformerBlock)
    nn.Module.__init__(block)
    block.config = SimpleNamespace(
        recompute_method="uniform", recompute_num_layers=1, fp8=None, fp4=None, distribute_saved_activations=False
    )
    block.layers = nn.ModuleList([Layer(layer) for layer in layers])
    block.num_layers_per_pipeline_rank = len(layers)
    batches = []
    for _ in range(2):
        x = torch.randn(5, 1, 8, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        params, ref_params = _packed(), _packed()
        actual = block._checkpointed_forward(x, None, None, None, None, None, params, False)
        assert type(actual.grad_fn).__name__ == "CheckpointFunctionBackward"
        expected = ref_x
        for ref_layer in reference:
            expected = ref_layer(expected, None, packed_seq_params=ref_params)[0]
        torch.testing.assert_close(actual, expected)
        batches.append((x, ref_x, actual, expected))
    for x, ref_x, actual, expected in reversed(batches):
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad)
    for actual, expected in zip(layers.parameters(), reference.parameters(), strict=True):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad)
    assert len(calls) == 6  # two original forwards, two references, two recomputes


def test_glm_physical_padding_has_finite_gradients_and_preserves_real_tokens(attention_runtime, monkeypatch):
    _, build, _ = attention_runtime

    def rotary_written_rows(value, cu, rotary):
        # Apex writes only rows covered by the supplied THD offsets. Poison
        # unwritten storage deterministically instead of relying on allocator reuse.
        return torch.where(
            torch.arange(value.shape[0])[:, None, None] < cu[-1], value, torch.full_like(value, torch.nan)
        )

    monkeypatch.setattr(
        sys.modules["apex.transformer.functional"], "fused_apply_rotary_pos_emb_thd", rotary_written_rows
    )
    model = build(3)
    value = torch.randn(8, 1, 8, requires_grad=True)
    packed = _packed()
    packed.cu_seqlens_q_padded = torch.tensor([0, 2, 8], dtype=torch.int32)
    packed.cu_seqlens_kv_padded = packed.cu_seqlens_q_padded
    actual, _ = model(value, None, packed_seq_params=packed)
    expected, _ = model(value[:5], None, packed_seq_params=_packed())
    torch.testing.assert_close(actual[:5], expected)
    assert torch.isfinite(actual).all()
    actual[:5].square().sum().backward()
    assert torch.isfinite(value.grad).all()
    assert torch.count_nonzero(value.grad[5:]) == 0


def test_glm_lora_wrapped_up_projections_train_through_absorbed_attention(attention_runtime):
    class WrappedProjection(nn.Module):
        """CPU stand-in for Bridge's fused-norm LoRA wrapper boundary."""

        def __init__(self, base):
            super().__init__()
            self.to_wrap = base.requires_grad_(False)
            self._adapter_enabled = True
            self.adapter = nn.Module()
            self.adapter.dim = 2
            self.adapter.alpha = 4
            self.adapter.tp_group = None
            self.adapter.dropout = nn.Identity()
            self.adapter.linear_in = nn.Linear(base.weight.shape[1], 2, bias=False)
            self.adapter.linear_out = nn.Linear(2, base.weight.shape[0], bias=False)

        def forward(self, value):
            norm = F.rms_norm(value, (value.shape[-1],), self.to_wrap.layer_norm_weight, self.to_wrap.eps)
            delta = self.adapter.linear_out(self.adapter.linear_in(norm)) * 2
            return self.to_wrap(value)[0] + delta, None

    _, build, _ = attention_runtime
    model = build(3).requires_grad_(False)
    model.linear_q_up_proj = WrappedProjection(model.linear_q_up_proj)
    model.linear_kv_up_proj = WrappedProjection(model.linear_kv_up_proj)
    value = torch.randn(5, 1, 8, requires_grad=True)
    output, _ = model(value, None, packed_seq_params=_packed())
    output.square().sum().backward()
    assert torch.isfinite(value.grad).all()
    for name, param in model.named_parameters():
        if ".adapter." in name:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            assert param.grad.abs().sum() > 0, name
        else:
            assert param.grad is None, name
