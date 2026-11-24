# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class _Group:
    def __init__(self, group, rank: int) -> None:
        self.group = group
        self.rank_in_group = rank
        self.world_size = 2

    def all_gather(self, value: torch.Tensor, dim: int) -> torch.Tensor:
        outputs = [torch.empty_like(value) for _ in range(self.world_size)]
        dist.all_gather(outputs, value.contiguous(), group=self.group)
        return torch.cat(outputs, dim=dim)


def _tp_worker(rank: int, init_method: str) -> None:
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config

    from relax.models.keye_vl2.sglang import model
    from relax.models.keye_vl2.sglang.ops import expert_forward

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=init_method)
    group = dist.new_group([0, 1])
    adapter = _Group(group, rank)
    model.get_parallel = lambda: SimpleNamespace(attn_tp_group=adapter)
    fused_moe_triton_config.get_exec = lambda: SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=False)
    )
    torch.manual_seed(73)
    experts, hidden, intermediate = 16, 2048, 768
    gate_up = torch.randn(experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16) * 0.01
    down = torch.randn(experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16) * 0.01
    weight = torch.randn(hidden, 4096, device="cuda", dtype=torch.bfloat16) * 0.01
    for rows in [3, 17, 128]:
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        ids = torch.rand(rows, experts, device="cuda").topk(8, dim=-1).indices.int()
        probabilities = torch.randn(rows, 8, device="cuda").softmax(-1)
        width = intermediate // 2
        start = rank * width
        local_gate_up = torch.cat(
            [gate_up[:, start : start + width], gate_up[:, intermediate + start : intermediate + start + width]], dim=1
        )
        # A second weight version must be read through the same parameter storage.
        for scale in [1.0, 0.5]:
            down.mul_(scale)
            expected = expert_forward(x, gate_up, down, probabilities, ids)
            actual = expert_forward(x, local_gate_up, down, probabilities, ids, adapter)
            dist.all_reduce(actual, group=group)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        attention = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
        local_attention = attention.chunk(2, dim=-1)[rank]
        expected = model.padded_linear(attention, weight)
        actual = model.KeyeRowLinearMethod().apply(SimpleNamespace(weight=weight), local_attention)
        dist.all_reduce(actual, group=group)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Keye TP alignment needs two CUDA devices")
def test_tp2_matches_unsharded_experts_and_output_projection(tmp_path):
    mp.spawn(_tp_worker, args=(f"file://{tmp_path / 'tp-init'}",), nprocs=2, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SGLang model imports require CUDA")
def test_replicated_expert_loader_updates_full_down_weights():
    from relax.models.keye_vl2.sglang.model import _replicated_expert_weight_loader

    parameter = torch.nn.Parameter(torch.zeros(3, 7, 8), requires_grad=False)
    address = parameter.data_ptr()
    weight = torch.randn(7, 8)
    _replicated_expert_weight_loader(parameter, weight, shard_id="w2", expert_id=2)
    torch.testing.assert_close(parameter[2], weight, atol=0, rtol=0)
    assert torch.count_nonzero(parameter[:2]) == 0
    replacement = torch.randn_like(parameter)
    _replicated_expert_weight_loader(parameter, replacement)
    torch.testing.assert_close(parameter, replacement, atol=0, rtol=0)
    assert parameter.data_ptr() == address
