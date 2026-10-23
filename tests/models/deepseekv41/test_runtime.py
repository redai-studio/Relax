# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU state-transport tests; run in the remote Megatron/Bridge environment."""

from __future__ import annotations

from math import prod
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge.models.conversion.model_bridge")

from relax.models.deepseekv41.runtime import (  # noqa: E402
    V41CarryPlan,
    V41IdsOnlyPlan,
    V41Runtime,
    v41_aggregate,
    v41_kv_source_layer,
    v41_pack_carry,
    v41_pack_pre,
    v41_unpack_carry,
    v41_unpack_pre,
)


@pytest.fixture
def runtime_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_layers=8,
        hidden_size=8,
        num_residual_streams=4,
        csa_compress_ratios=[0, 2, 2, 2, 1, 1, 1, 1],
        v41_kv_source_layer_ids=(1, 4),
        v41_index_source_layer_ids=(1, 4, 6),
        v41_engram_layer_ids=(0, 5),
        v41_candidate_source_layer_id=4,
        v41_candidate_block_size=2,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        kv_lora_rank=16,
        dsa_indexer_head_dim=6,
        dsa_indexer_topk=3,
    )


def _populate(plan: V41CarryPlan, s: int, b: int, *, gradients: bool = False) -> V41Runtime:
    runtime = V41Runtime()
    for section in plan.sections(s, b):
        if section.encoding == "pre":
            value = torch.randn(section.shape, dtype=torch.float32, requires_grad=gradients)
        elif section.encoding == "int":
            value = torch.arange(prod(section.shape), dtype=torch.int64).view(section.shape)
            value = value.remainder(7) - 1
        elif section.encoding == "candidate":
            shape = (
                *section.shape[:-1],
                (section.shape[-1] + plan.candidate_block_size - 1) // plan.candidate_block_size,
            )
            value = torch.randint(0, 2, shape, dtype=torch.bool)
            value = value.repeat_interleave(plan.candidate_block_size, dim=-1)[..., : section.shape[-1]]
        else:
            value = torch.randn(section.shape, dtype=torch.bfloat16, requires_grad=gradients)
        if section.key is None:
            setattr(runtime, section.name, value)
        else:
            getattr(runtime, section.name)[section.key] = value
    return runtime


def _assert_state_equal(expected: V41Runtime, actual: V41Runtime) -> None:
    if expected.pre is None:
        assert actual.pre is None
    else:
        torch.testing.assert_close(actual.pre, expected.pre, atol=2e-5, rtol=2e-5)
    for name in ("input_ids", "topk", "candidates"):
        value = getattr(expected, name)
        if value is None:
            assert getattr(actual, name) is None
        else:
            assert torch.equal(getattr(actual, name), value)
    for name in ("latent", "index_k"):
        wanted = getattr(expected, name)
        got = getattr(actual, name)
        assert got.keys() == wanted.keys()
        for key in wanted:
            assert torch.equal(got[key], wanted[key])


def test_runtime_reset_is_microbatch_and_instance_local() -> None:
    first, second = V41Runtime(), V41Runtime()
    first.latent[1] = torch.ones(1)
    first.index_k[1] = torch.ones(1)
    first.pre = first.input_ids = first.topk = first.candidates = torch.ones(1)
    second.latent[2] = torch.zeros(1)
    first.reset()
    assert not first.latent and not first.index_k
    assert first.pre is first.input_ids is first.topk is first.candidates is None
    assert second.latent.keys() == {2}


@pytest.mark.parametrize("layer,source", [(0, None), (1, 1), (3, 1), (4, 4), (7, 4)])
def test_runtime_kv_source_matches_compression_family(runtime_config, layer, source) -> None:
    assert v41_kv_source_layer(runtime_config, layer) == source


def test_runtime_kv_source_missing_raises(runtime_config) -> None:
    runtime_config.v41_kv_source_layer_ids = (4,)
    with pytest.raises(ValueError, match="no KV source"):
        v41_kv_source_layer(runtime_config, 1)


def test_runtime_carry_plan_boundary_retains_only_needed_state(runtime_config) -> None:
    before = V41CarryPlan(runtime_config, 3)
    assert before.latent_keys == before.index_k_keys == ()
    assert not before.topk and not before.candidates
    active = V41CarryPlan(runtime_config, 4)
    assert active.latent_keys == active.index_k_keys == (4,)
    assert active.topk and active.candidates and active.input_ids
    before_index = V41CarryPlan(runtime_config, 5)
    assert not before_index.topk and not before_index.input_ids
    after_index = V41CarryPlan(runtime_config, 6)
    assert after_index.latent_keys == (4,)
    assert after_index.index_k_keys == ()
    assert after_index.topk and after_index.candidates


@pytest.mark.parametrize("layer", [-2, 8])
def test_runtime_invalid_carry_boundary_raises(runtime_config, layer) -> None:
    with pytest.raises(ValueError, match="invalid carry boundary"):
        V41CarryPlan(runtime_config, layer)


@pytest.mark.parametrize(
    "sp,tp,cp,s,b", [(False, 1, 1, 3, 1), (False, 2, 2, 5, 2), (True, 2, 1, 3, 1), (True, 2, 2, 5, 2)]
)
def test_runtime_carry_roundtrip_parallel_shapes(runtime_config, sp, tp, cp, s, b) -> None:
    runtime_config.sequence_parallel = sp
    runtime_config.tensor_model_parallel_size = tp
    runtime_config.context_parallel_size = cp
    plan = V41CarryPlan(runtime_config, 4)
    hidden = torch.randn(s, b, plan.base_cols, dtype=torch.bfloat16)
    original = _populate(plan, s, b)
    restored = V41Runtime()
    packed = v41_pack_carry(hidden, original, plan)
    result = v41_unpack_carry(packed, restored, plan)
    assert torch.equal(hidden, result)
    _assert_state_equal(original, restored)
    local_length = s * tp if sp else s
    assert restored.input_ids.shape == (b, local_length)
    assert restored.pre.shape == (s, b, 4)
    assert restored.latent[4].shape == (b, local_length * cp, 16)
    assert restored.index_k[4].shape == (b, local_length * cp, 6)
    assert restored.topk.shape == (b, local_length, 3)


def test_runtime_initial_plan_carries_ids_without_pre(runtime_config) -> None:
    plan = V41IdsOnlyPlan(runtime_config)
    assert not plan.pre and plan.input_ids
    assert plan.latent_keys == plan.index_k_keys == ()
    assert not plan.topk and not plan.candidates
    original = V41Runtime(input_ids=torch.tensor([[0, 1, 128799]], dtype=torch.int64))
    hidden = torch.randn(3, 1, plan.base_cols, dtype=torch.bfloat16)
    restored = V41Runtime()
    result = v41_unpack_carry(v41_pack_carry(hidden, original, plan), restored, plan)
    assert torch.equal(hidden, result)
    _assert_state_equal(original, restored)


def test_runtime_initial_plan_without_engram_is_identity(runtime_config) -> None:
    runtime_config.v41_engram_layer_ids = ()
    plan = V41IdsOnlyPlan(runtime_config)
    hidden = torch.randn(3, 2, plan.base_cols, dtype=torch.bfloat16)
    restored = V41Runtime(pre=torch.ones(1), latent={4: torch.ones(1)})
    assert v41_pack_carry(hidden, V41Runtime(), plan) is hidden
    assert torch.equal(v41_unpack_carry(hidden, restored, plan), hidden)
    assert restored.pre is None and not restored.latent


def test_runtime_encoded_int32_extremes_are_bit_exact(runtime_config) -> None:
    plan = V41IdsOnlyPlan(runtime_config)
    ids = torch.tensor([[0, -1, -(2**31), 2**31 - 1, 128799]], dtype=torch.int64)
    hidden = torch.zeros(5, 1, plan.base_cols, dtype=torch.bfloat16)
    restored = V41Runtime()
    v41_unpack_carry(v41_pack_carry(hidden, V41Runtime(input_ids=ids), plan), restored, plan)
    assert torch.equal(restored.input_ids, ids)


def test_runtime_carry_preserves_hidden_pre_and_latent_gradients(runtime_config) -> None:
    plan = V41CarryPlan(runtime_config, 4)
    original = _populate(plan, 3, 2, gradients=True)
    hidden = torch.randn(3, 2, plan.base_cols, dtype=torch.bfloat16, requires_grad=True)
    restored = V41Runtime()
    result = v41_unpack_carry(v41_pack_carry(hidden, original, plan), restored, plan)
    loss = result.float().sum() + 3 * restored.pre.sum() + 2 * restored.latent[4].float().sum()
    loss.backward()
    assert torch.equal(hidden.grad, torch.ones_like(hidden))
    assert torch.equal(original.pre.grad, torch.full_like(original.pre, 3))
    assert torch.equal(original.latent[4].grad, torch.full_like(original.latent[4], 2))
    assert not restored.topk.requires_grad
    assert not restored.input_ids.requires_grad
    assert not restored.candidates.requires_grad


def test_runtime_unpack_replaces_prior_microbatch_state(runtime_config) -> None:
    plan = V41CarryPlan(runtime_config, 4)
    hidden = torch.randn(3, 1, plan.base_cols, dtype=torch.bfloat16)
    first, second = _populate(plan, 3, 1), _populate(plan, 3, 1)
    restored = V41Runtime()
    v41_unpack_carry(v41_pack_carry(hidden, first, plan), restored, plan)
    old_latent = restored.latent[4]
    restored.latent[99] = torch.ones(1)
    v41_unpack_carry(v41_pack_carry(hidden, second, plan), restored, plan)
    _assert_state_equal(second, restored)
    assert restored.latent[4] is not old_latent
    assert 99 not in restored.latent


def test_runtime_missing_section_raises_before_pack(runtime_config) -> None:
    plan = V41CarryPlan(runtime_config, 4)
    hidden = torch.zeros(3, 1, plan.base_cols, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="carry pre.*expected"):
        v41_pack_carry(hidden, V41Runtime(), plan)


def test_runtime_wrong_shape_and_width_raise(runtime_config) -> None:
    plan = V41CarryPlan(runtime_config, 4)
    hidden = torch.zeros(3, 1, plan.base_cols, dtype=torch.bfloat16)
    runtime = _populate(plan, 3, 1)
    runtime.topk = torch.zeros(1, 3, 4, dtype=torch.int64)
    with pytest.raises(ValueError, match="carry topk.*expected"):
        v41_pack_carry(hidden, runtime, plan)
    with pytest.raises(ValueError, match="expected packed width"):
        v41_unpack_carry(hidden, V41Runtime(), plan)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_runtime_carry_rejects_dtype_conversion(runtime_config, dtype) -> None:
    plan = V41IdsOnlyPlan(runtime_config)
    hidden = torch.zeros(3, 1, plan.base_cols, dtype=dtype)
    with pytest.raises(ValueError, match="carry requires BF16"):
        v41_pack_carry(hidden, V41Runtime(), plan)
    with pytest.raises(ValueError, match="carry requires BF16"):
        v41_unpack_carry(hidden, V41Runtime(), plan)


def test_runtime_cp_compression_requires_aligned_local_chunk(runtime_config) -> None:
    runtime_config.context_parallel_size = 2
    plan = V41CarryPlan(runtime_config, 1)
    with pytest.raises(ValueError, match="CP chunk length"):
        plan.sections(3, 1)


def test_runtime_pre_high_low_retains_precision_and_gradient() -> None:
    pre = torch.tensor([[[0.50123, 0.99876, 1.00321, 0.007891]]], requires_grad=True)
    restored = v41_unpack_pre(v41_pack_pre(pre, torch.bfloat16), 4)
    torch.testing.assert_close(restored, pre, rtol=2e-5, atol=1e-6)
    restored.sum().backward()
    assert torch.equal(pre.grad, torch.ones_like(pre))


def test_runtime_aggregate_uses_previous_pre_or_first_stream() -> None:
    hidden = torch.arange(16, dtype=torch.bfloat16).view(2, 1, 8).requires_grad_()
    assert torch.equal(v41_aggregate(hidden, None, 4), hidden.view(2, 1, 4, 2)[:, :, 0])
    pre = torch.tensor([[[1.0, 0.5, 0.0, 0.25]], [[0.0, 1.0, 0.5, 0.0]]], requires_grad=True)
    output = v41_aggregate(hidden, pre, 4)
    expected = (hidden.view(2, 1, 4, 2).float() * pre.unsqueeze(-1)).sum(2).to(torch.bfloat16)
    assert torch.equal(output, expected)
    output.float().sum().backward()
    assert pre.grad is not None and hidden.grad is not None
