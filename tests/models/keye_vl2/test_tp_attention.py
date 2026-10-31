# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint


def _tp_worker(rank: int, init_method: str) -> None:
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
    from megatron.core.transformer.transformer_config import TransformerConfig

    from relax.models.common.chunked_grad_coalesce import _allreduce_non_tensor_model_parallel_grads
    from relax.models.keye_vl2.attention import KeyeCoreAttention
    from relax.models.keye_vl2.linear import KeyeRowParallelLinear

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=4, init_method=init_method)
    group = dist.new_group([0, 1, 2, 3])
    torch.manual_seed(42)
    DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
    projection_config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        tensor_model_parallel_size=4,
        sequence_parallel=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        gradient_accumulation_fusion=False,
    )
    projection = KeyeRowParallelLinear(
        input_size=32,
        output_size=16,
        config=projection_config,
        init_method=torch.nn.init.normal_,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=True,
        is_expert=False,
        tp_group=group,
        tp_comm_buffer_name="proj",
    ).cuda()
    x = torch.randn(64, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    local_x = x.detach().chunk(4, -1)[rank].clone().requires_grad_()
    with torch.no_grad():
        projection.weight.copy_(w.chunk(4, -1)[rank])
    expected = torch.nn.functional.linear(x, w)
    for fused in (False, True):
        projection.config.gradient_accumulation_fusion = fused
        projection.weight.grad = None
        local_x.grad = None
        if fused:
            projection.weight.main_grad = torch.zeros_like(projection.weight, dtype=torch.float32)
        actual, bias = projection(local_x)
        assert bias is None
        torch.testing.assert_close(actual, expected.detach().chunk(4, 0)[rank], atol=0, rtol=0)
        actual.float().square().sum().backward()
        dy = (2 * expected.detach()).reshape(-1, 16)
        dx = dy.matmul(w.detach()).reshape_as(x).chunk(4, -1)[rank]
        torch.testing.assert_close(local_x.grad, dx, atol=0, rtol=0)
        dw = torch.mm(dy.t(), x.detach().reshape(-1, 32), out_dtype=torch.float32).chunk(4, -1)[rank]
        if fused:
            assert projection.weight.grad_added_to_main_grad
            torch.testing.assert_close(projection.weight.main_grad, dw, atol=0, rtol=0)
            assert torch.count_nonzero(projection.weight.grad) == 0
        else:
            torch.testing.assert_close(projection.weight.grad, dw.bfloat16(), atol=0, rtol=0)

    def config(tp: int) -> SimpleNamespace:
        return SimpleNamespace(
            tensor_model_parallel_size=tp,
            context_parallel_size=1,
            sequence_parallel=tp > 1,
            qk_layernorm=False,
            hidden_size=32,
            layernorm_epsilon=1e-6,
            rotary_base=1e7,
            keye_indexer_loss_coeff=0.01,
        )

    base = KeyeCoreAttention(config(1), 1).cuda().bfloat16()
    shard = KeyeCoreAttention(config(4), 1, pg_collection=SimpleNamespace(tp=group)).cuda().bfloat16()
    shard.load_state_dict(base.state_dict())
    shard.ddp_config = SimpleNamespace()
    for boundaries, dummy, recompute in [
        ([0, 32, 64], False, False),
        ([0, 2080, 2112], False, True),
        ([0, 64], True, False),
    ]:
        base.zero_grad(set_to_none=True)
        shard.zero_grad(set_to_none=True)
        length = boundaries[-1]
        valid = torch.arange(length, device="cuda") < length - 3
        packed = SimpleNamespace(cu_seqlens_q_cpu=boundaries, keye_valid_mask=valid, is_dummy=dummy)
        full = [
            torch.randn(length, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for heads, dim in [(32, 128), (4, 128), (4, 128), (1, 32)]
        ]
        local = [
            tensor.detach()[:, rank * (tensor.shape[1] // 4) : (rank + 1) * (tensor.shape[1] // 4)]
            .clone()
            .requires_grad_()
            for tensor in full[:3]
        ]
        hidden = full[3].detach().chunk(4, dim=0)[rank].clone().requires_grad_()
        expected = base(*full[:3], hidden_states=full[3], packed_seq_params=packed)

        def forward(q, k, v, hidden):
            return shard(q, k, v, hidden_states=hidden, packed_seq_params=packed)

        actual = checkpoint(forward, *local, hidden, use_reentrant=True) if recompute else forward(*local, hidden)
        expected_local = expected.reshape(length, 32, 128)[:, rank * 8 : (rank + 1) * 8].reshape(length, -1)
        torch.testing.assert_close(actual, expected_local, atol=0, rtol=0)
        weight = torch.randn_like(expected)
        local_weight = weight.reshape(length, 32, 128)[:, rank * 8 : (rank + 1) * 8].reshape(length, -1)
        (expected * weight).float().sum().backward()
        (actual * local_weight).float().sum().backward()
        assert hidden.grad is None
        for tensor, full_tensor in zip(local, full[:3]):
            heads = full_tensor.shape[1] // 4
            target = full_tensor.grad[:, rank * heads : (rank + 1) * heads]
            assert tensor.grad.isfinite().all()
            torch.testing.assert_close(tensor.grad, target, atol=0.06, rtol=0.08)
        for parameter in shard.parameters():
            assert parameter.average_gradients_across_tp_domain
            parameter.main_grad = parameter.grad
        _allreduce_non_tensor_model_parallel_grads([shard], shard.config, tp_group=group)
        for parameter, reference in zip(shard.parameters(), base.parameters()):
            torch.testing.assert_close(parameter.grad, reference.grad, atol=0.02, rtol=0.1)
            if dummy:
                assert torch.count_nonzero(parameter.grad) == 0
            with torch.no_grad():
                parameter.add_(parameter.grad, alpha=-1e-3)
                reference.add_(reference.grad, alpha=-1e-3)
                replica = parameter.clone()
                dist.broadcast(replica, src=0, group=group)
                torch.testing.assert_close(parameter, replica, atol=0, rtol=0)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Keye TP4 gradient regression needs four CUDA devices")
def test_keye_tp4_sp_matches_tp1_outputs_gradients_and_indexer_update(tmp_path):
    mp.spawn(_tp_worker, args=(f"file://{tmp_path / 'tp-init'}",), nprocs=4, join=True)


def test_keye_custom_output_projection_uses_explicit_row_mapping():
    from megatron.bridge.models.conversion.param_mapping import RowParallelMapping

    from relax.models.keye_vl2.bridge import KeyeBridge

    mapping = (
        KeyeBridge().mapping_registry().megatron_to_hf_lookup("decoder.layers.0.self_attention.linear_proj.weight")
    )
    assert isinstance(mapping, RowParallelMapping)
    assert mapping.hf_param == "model.layers.0.self_attn.o_proj.weight"
