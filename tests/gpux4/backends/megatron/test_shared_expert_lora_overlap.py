# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.utils.megatron_peft_utils import build_lora_peft, fix_shared_expert_lora_overlap


def _shared_expert(group: dist.ProcessGroup, overlap: bool, sequence_parallel: bool) -> torch.nn.Module:
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        tensor_model_parallel_size=group.size(),
        sequence_parallel=sequence_parallel,
        moe_shared_expert_intermediate_size=32,
        moe_shared_expert_overlap=overlap,
        moe_token_dispatcher_type="alltoall",
        add_bias_linear=False,
        bias_activation_fusion=False,
        use_cpu_initialization=True,
    )
    groups = ProcessGroupCollection(tp=group)
    model = torch.nn.Module()
    shared = model.shared_experts = SharedExpertMLP(
        config, MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear), False, groups
    )
    shared.config.pg_collection = groups
    shared.requires_grad_(False)
    peft = build_lora_peft(Namespace(lora_rank=4, lora_alpha=8, lora_target_modules=["linear_fc2"], lora_dropout=0))
    shared.linear_fc2 = peft.transform(shared.linear_fc2, name="linear_fc2", prefix="mlp.shared_experts")
    fix_shared_expert_lora_overlap(model)
    return shared.cuda()


def _compare_overlap(group: dist.ProcessGroup, sequence_parallel: bool) -> None:
    torch.manual_seed(651 + group.rank())
    standard = _shared_expert(group, False, sequence_parallel)
    overlapped = _shared_expert(group, True, sequence_parallel)
    with torch.no_grad():
        # Bridge initializes B to zero, which would hide both the forward error and the A gradient.
        standard.linear_fc2.adapter.linear_out.weight.normal_(std=0.2)
        for name, param in standard.named_parameters():
            overlapped.get_parameter(name).copy_(param)
    torch.manual_seed(813 + (group.rank() if sequence_parallel else 0))
    inputs = torch.randn(8, 1, 16, device="cuda", requires_grad=True)
    other_inputs = inputs.detach().clone().requires_grad_()
    expected = standard(inputs)
    overlapped.pre_forward_comm(other_inputs)
    overlapped.linear_fc1_forward_and_act()
    overlapped.linear_fc2_forward()
    overlapped.post_forward_comm()
    actual = overlapped.get_output()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(other_inputs.grad, inputs.grad, atol=1e-6, rtol=1e-5)
    for name, param in standard.linear_fc2.adapter.named_parameters():
        torch.testing.assert_close(
            overlapped.linear_fc2.adapter.get_parameter(name).grad, param.grad, atol=1e-6, rtol=1e-5
        )


def _worker(rank: int, rendezvous: str) -> None:
    from megatron.core import parallel_state

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
        group = parallel_state.get_tensor_model_parallel_group()
        for sequence_parallel in (False, True):
            _compare_overlap(group, sequence_parallel)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices for tensor parallelism")
def test_shared_expert_lora_overlap_preserves_forward_and_gradients(tmp_path: Path) -> None:
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
