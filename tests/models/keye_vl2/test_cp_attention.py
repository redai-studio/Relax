# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _cp_worker(rank: int, init_method: str, sparse: bool) -> None:
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler

    from relax.backends.megatron.cp_utils import gdn_cp_slice
    from relax.models.keye_vl2.attention import KeyeCoreAttention

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=init_method)
    group = dist.new_group([0, 1])
    torch.manual_seed(42)
    DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))

    def config(cp):
        return SimpleNamespace(
            tensor_model_parallel_size=1,
            context_parallel_size=cp,
            hidden_size=32,
            layernorm_epsilon=1e-6,
            rotary_base=1e7,
            keye_indexer_loss_coeff=0.01,
        )

    base = KeyeCoreAttention(config(1), 1).cuda().bfloat16()
    shard = KeyeCoreAttention(config(2), 1, pg_collection=SimpleNamespace(cp=group)).cuda().bfloat16()
    shard.load_state_dict(base.state_dict())
    last = 2240 if sparse else 144
    cu = [0, 64, last]
    pos = torch.cat([torch.arange(64), torch.arange(last - 64)]).cuda()
    pos = torch.stack([pos // 2, pos // 3, pos // 5], -1)
    valid = torch.ones(last, device="cuda", dtype=torch.bool)
    valid[61:64] = False
    valid[-4:] = False
    packed = SimpleNamespace(cu_seqlens_q_cpu=cu, keye_positions=pos, keye_valid_mask=valid)
    full = [
        torch.randn(last, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for h, d in [(4, 32), (2, 32), (2, 32), (1, 32)]
    ]
    local = [gdn_cp_slice(x.detach(), cu, 2, rank).clone().requires_grad_() for x in full]
    out = base(*full[:3], hidden_states=full[3], packed_seq_params=packed)
    local_out = shard(*local[:3], hidden_states=local[3], packed_seq_params=packed)
    torch.testing.assert_close(local_out, gdn_cp_slice(out, cu, 2, rank), atol=0.02, rtol=0.02)
    # Identical full loss split across CP ranks, including cross-rank key gradients.
    weight = torch.randn_like(out)
    (out * weight).float().sum().backward()
    (local_out * gdn_cp_slice(weight, cu, 2, rank)).float().sum().backward()
    for i in range(3):
        torch.testing.assert_close(local[i].grad, gdn_cp_slice(full[i].grad, cu, 2, rank), atol=0.06, rtol=0.08)
    for (name, p), (_, bp) in zip(shard.named_parameters(), base.named_parameters()):
        dist.all_reduce(p.grad, group=group)
        torch.testing.assert_close(p.grad, bp.grad, atol=0.02, rtol=0.1)

    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Keye CP gradient regression needs two CUDA devices")
@pytest.mark.parametrize("sparse", [False, True])
def test_keye_cp2_matches_cp1_outputs_and_cross_rank_gradients(tmp_path, sparse):
    mp.spawn(_cp_worker, args=(f"file://{tmp_path / 'cp-init'}", sparse), nprocs=2, join=True)


def _text_cp_worker(rank: int, init_method: str, lengths: list[int], pad_multiplier: int) -> None:
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler

    from relax.backends.megatron import data
    from relax.backends.megatron.cp_utils import gdn_cp_slice
    from relax.models.keye_vl2.attention import KeyeCoreAttention

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=init_method)
    group = dist.new_group([0, 1])
    torch.manual_seed(83)
    DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
    raw = {
        "tokens": [torch.arange(length, device="cuda") for length in lengths],
        "total_lengths": lengths,
        "response_lengths": lengths,
        "loss_masks": [torch.ones(length, device="cuda", dtype=torch.int) for length in lengths],
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(data, "get_args", lambda: SimpleNamespace())
        patch.setattr(data.mpu, "get_context_parallel_world_size", lambda: 2)
        patch.setattr(data.mpu, "get_context_parallel_rank", lambda: rank)
        patch.setattr(data.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
        batch = data.get_batch(iter([(raw, None)]), list(raw), pad_multiplier=pad_multiplier)
    packed = batch["packed_seq_params"]
    assert not hasattr(packed, "keye_valid_mask")
    cu = packed.cu_seqlens_q_cpu
    real_indices = torch.cat([torch.arange(start, start + length) for start, length in zip(cu, lengths)]).cuda()
    config = SimpleNamespace(
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        hidden_size=32,
        layernorm_epsilon=1e-6,
        rotary_base=1e7,
        keye_indexer_loss_coeff=0.01,
    )
    base = KeyeCoreAttention(config, 1).cuda().bfloat16()
    config = SimpleNamespace(**{**vars(config), "context_parallel_size": 2})
    shard = KeyeCoreAttention(config, 1, pg_collection=SimpleNamespace(cp=group)).cuda().bfloat16()
    shard.load_state_dict(base.state_dict())
    full = [torch.randn(cu[-1], h, 32, device="cuda", dtype=torch.bfloat16) for h in (4, 2, 2, 1)]
    reference = [x.index_select(0, real_indices).requires_grad_() for x in full]
    local = [gdn_cp_slice(x, cu, 2, rank).clone().requires_grad_() for x in full]
    reference_cu = [0, lengths[0], sum(lengths)]
    expected = base(
        *reference[:3], hidden_states=reference[3], packed_seq_params=SimpleNamespace(cu_seqlens_q_cpu=reference_cu)
    )
    actual = shard(*local[:3], hidden_states=local[3], packed_seq_params=packed)
    padded_expected = expected.new_zeros(cu[-1], expected.shape[-1]).index_copy(0, real_indices, expected)
    valid = torch.zeros(cu[-1], dtype=torch.bool, device="cuda")
    valid[real_indices] = True
    local_valid = gdn_cp_slice(valid, cu, 2, rank)
    torch.testing.assert_close(
        actual[local_valid], gdn_cp_slice(padded_expected, cu, 2, rank)[local_valid], atol=0.02, rtol=0.02
    )
    weight = torch.randn_like(padded_expected).masked_fill(~valid[:, None], 0)
    (expected * weight[real_indices]).float().sum().backward()
    (actual * gdn_cp_slice(weight, cu, 2, rank)).float().sum().backward()
    for index in range(3):
        expected_grad = torch.zeros_like(full[index]).index_copy(0, real_indices, reference[index].grad)
        torch.testing.assert_close(local[index].grad, gdn_cp_slice(expected_grad, cu, 2, rank), atol=0.06, rtol=0.08)
    for parameter, reference_parameter in zip(shard.parameters(), base.parameters()):
        dist.all_reduce(parameter.grad, group=group)
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=0.02, rtol=0.1)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Keye text CP regression needs two CUDA devices")
@pytest.mark.parametrize("lengths, pad_multiplier", [([5, 5], 1), ([5, 9], 16)])
def test_keye_text_cp_padding_matches_unpadded_outputs_and_gradients(tmp_path, lengths, pad_multiplier):
    mp.spawn(
        _text_cp_worker, args=(f"file://{tmp_path / 'text-cp-init'}", lengths, pad_multiplier), nprocs=2, join=True
    )
