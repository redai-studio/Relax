# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check fused MFSDP gradients against explicit sums before the optimizer
runs."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


def _worker(rank: int, rendezvous: str) -> None:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.transformer_config import TransformerConfig

    class Unit(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            kwargs = dict(config=config, init_method=lambda w: w, bias=False, skip_bias_add=True)
            self.experts = TEColumnParallelGroupedLinear(
                num_gemms=2, input_size=64, output_size=64, is_expert=True, **kwargs
            )
            self.head = ColumnParallelLinear(64, 64, **kwargs)

        def forward(self, x, splits):
            experts, _ = self.experts(x, splits)
            # A single output weight is reused for primary chunks and an additional projection.
            head = torch.cat([self.head(part[:, None])[0][:, 0] for part in x.tensor_split(3)])
            return experts, head, self.head(x[:4, None])[0][:, 0]

    class Model(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.unit = Unit(config)

        def forward(self, x, splits):
            return self.unit(x, splits)

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(minutes=3))
    parallel_state.initialize_model_parallel(expert_model_parallel_size=4, expert_tensor_parallel_size=1)
    try:
        tensor_parallel.model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            expert_model_parallel_size=4,
            expert_tensor_parallel_size=1,
            num_moe_experts=8,
            params_dtype=torch.bfloat16,
            bf16=True,
            gradient_accumulation_fusion=True,
            perform_initialization=False,
            calculate_per_token_loss=True,
        )
        ddp = DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            data_parallel_sharding_strategy="optim_grads_params",
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            megatron_fsdp_main_params_dtype=torch.float32,
            megatron_fsdp_main_grads_dtype=torch.float32,
        )
        model = Model(config).cuda()
        master = {}
        with torch.no_grad():
            for index, (name, param) in enumerate(model.named_parameters()):
                generator = torch.Generator(device="cuda").manual_seed(93 + index)
                param.copy_(torch.randn(param.shape, generator=generator, device="cuda") * 0.03)
                master[name] = param.detach().float().clone()

        wrapped = FullyShardedDataParallel(config, ddp, model, fsdp_unit_modules=[Unit])
        owner = wrapped.module
        local_weights = {}
        local_moments = {}
        reference_moments = {}
        for step in range(2):
            owner.zero_grad_buffer()
            owner.start_param_sync()
            grads = {name: torch.zeros_like(w) for name, w in master.items()}
            pending = []
            for mb, splits in enumerate(([7, 13], [16, 0], [0, 12])):
                gen = torch.Generator(device="cuda").manual_seed(1234 + step * 100 + mb * 8 + rank)
                x = torch.randn((sum(splits), 64), generator=gen, device="cuda", dtype=torch.bfloat16).requires_grad_()
                x_ref = x.detach().clone().requires_grad_()
                projected = wrapped(x, splits)
                expert_weights = [master[f"unit.experts.weight{i}"].bfloat16() for i in range(2)]
                head_weight = master["unit.head.weight"].bfloat16()
                expected = (
                    torch.cat([F.linear(p, w) for p, w in zip(x_ref.split(splits), expert_weights, strict=True)]),
                    F.linear(x_ref, head_weight),
                    F.linear(x_ref[:4], head_weight),
                )
                dy = [torch.randn(t.shape, device="cuda", dtype=torch.bfloat16, generator=gen) for t in projected]
                for actual, wanted in zip(projected, expected, strict=True):
                    torch.testing.assert_close(actual, wanted, rtol=0.02, atol=0.008)
                for i, (d, p) in enumerate(zip(dy[0].split(splits), x.detach().split(splits), strict=True)):
                    grads[f"unit.experts.weight{i}"].addmm_(d.float().T, p.float())
                grads["unit.head.weight"].addmm_(dy[1].float().T, x.detach().float())
                grads["unit.head.weight"].addmm_(dy[2].float().T, x[:4].detach().float())
                dx = torch.autograd.grad(expected, x_ref, grad_outputs=dy)[0]
                pending.append((projected, dy, x, dx))
            for index, (projected, dy, x, dx) in enumerate(reversed(pending)):
                owner.is_last_microbatch = index == len(pending) - 1
                torch.autograd.backward(projected, dy)
                torch.testing.assert_close(x.grad.float(), dx.float(), rtol=0.03, atol=0.02)
            owner.finish_grad_sync()
            for name, param in sorted(owner.param_and_grad_buffer.optimizer_named_parameters, reverse=True):
                raw = param.orig_param
                grad = grads[name]
                dist.all_reduce(grad, group=raw._gbuf.data_parallel_group)
                start, end = raw._gbuf._get_item_slice_in_shard(raw._item_id)
                # Maintain an independent full-size Adam reference for the next forward.
                m, v = reference_moments.setdefault(name, (torch.zeros_like(grad), torch.zeros_like(grad)))
                m.lerp_(grad, 0.1)
                v.mul_(0.999).addcmul_(grad, grad, value=0.001)
                denom = (v / (1 - 0.999 ** (step + 1))).sqrt().add_(1e-8)
                master[name].addcdiv_(m, denom, value=-0.0001 / (1 - 0.9 ** (step + 1)))
                if start == end:
                    continue
                actual_grad = param.grad.to_local().detach().flatten()
                wanted = grad.flatten()[start:end]
                torch.testing.assert_close(actual_grad, wanted, rtol=1e-4, atol=1e-4, msg=lambda msg: f"{name}: {msg}")
                if step == 0:
                    weight = param.to_local().detach().flatten()
                    local_weights[name] = weight
                    local_moments[name] = torch.optim.Adam([weight], lr=0.0001)
                weight = local_weights[name]
                optimizer = local_moments[name]
                weight.grad = actual_grad
                optimizer.step()
                for key, value in (("exp_avg", m), ("exp_avg_sq", v)):
                    torch.testing.assert_close(
                        optimizer.state[weight][key], value.flatten()[start:end], rtol=1e-4, atol=1e-4
                    )
                torch.testing.assert_close(weight, master[name].flatten()[start:end], rtol=1e-5, atol=1e-6)
            owner.install_optimized_model_weights()
        owner.synchronize_param_gather()
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four GPUs for dense DP4 and expert DP1")
def test_fsdp_fused_experts_and_shared_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rdzv'}",), nprocs=4, join=True)


def _cold_mtp_worker(rank: int, rendezvous: str) -> None:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import BucketStatus
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.transformer_config import TransformerConfig

    from relax.models.common.chunked_mtp_loss import process_mtp_loss

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(minutes=2))
    parallel_state.initialize_model_parallel()
    try:
        tensor_parallel.model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            params_dtype=torch.bfloat16,
            bf16=True,
            gradient_accumulation_fusion=False,
            perform_initialization=False,
            mtp_num_layers=1,
            calculate_per_token_loss=True,
        )
        model = torch.nn.Module()
        model.config = config
        model.output_layer = ColumnParallelLinear(
            64, 64, config=config, init_method=lambda w: w, bias=False, skip_bias_add=True
        ).cuda()
        generator = torch.Generator(device="cuda").manual_seed(214)
        weight = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16, generator=generator)
        with torch.no_grad():
            model.output_layer.weight.copy_(weight)
        ddp = DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            data_parallel_sharding_strategy="optim_grads_params",
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            bucket_size=1,
            suggested_communication_unit_size=1,
            fsdp_all_gather_in_start_param_sync=False,
        )
        owner = FullyShardedDataParallel(config, ddp, model).module
        owner.start_param_sync()
        head = model.output_layer
        bucket_id = owner.param_and_grad_buffer.param_to_param_group[head.weight]
        pipeline = owner.all_gather_pipeline
        assert pipeline.bucket_status[pipeline.get_bucket_key(bucket_id, False)] == BucketStatus.EMPTY

        # The untied output head has not run. A detached weight must bind to
        # the gathered storage before the first native MTP projection.
        seq_len = 1027  # Cross the default 1024-token MTP chunk boundary.
        hidden = torch.randn(2 * seq_len, 1, 64, device="cuda", dtype=torch.bfloat16, generator=generator)
        labels = torch.arange(seq_len, device="cuda").remainder(64).unsqueeze(0)
        projections = []

        def token_loss(targets, logits):
            projections.append(logits.detach())
            return F.cross_entropy(logits.squeeze(1).float(), targets.flatten(), reduction="none").unsqueeze(0)

        process_mtp_loss(
            hidden,
            labels,
            torch.ones_like(labels),
            head,
            None,
            False,
            False,
            token_loss,
            config,
            cp_group=parallel_state.get_context_parallel_group(),
            scale_logits_fn=lambda x: x,
        )
        torch.testing.assert_close(torch.cat(projections), F.linear(hidden[seq_len:], weight))
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs for a sharded output head")
def test_native_chunked_mtp_gathers_cold_untied_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    monkeypatch.setenv("RELAX_MTP_CE_IMPL", "native")
    mp.spawn(_cold_mtp_worker, args=(f"file://{tmp_path / 'cold-mtp'}",), nprocs=2, join=True)
