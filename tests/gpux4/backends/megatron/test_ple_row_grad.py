# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise PLE row gradients through TP lookup, leaf accumulation and DP-
sharded Adam."""

from argparse import Namespace
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, rendezvous: str) -> None:
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.per_layer_embedding import NGramEmbedding
    from megatron.core.transformer.transformer_config import TransformerConfig
    from torch.utils.checkpoint import checkpoint

    from relax.models.qwen3_8.ops.ple_row_grad import configure

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=180))
    ps.initialize_model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(731)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=320,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            ple_ngram_size=2,
            ple_heads_per_ngram=2,
            ple_ngram_vocab_size_base=4093,
            ple_unigram_vocab_size=2048,
        )
        cases = []
        for compact in (False, True):
            table = NGramEmbedding(config, 320, 0)
            weight = table.ngram_embedding.weight
            with torch.no_grad():
                if cases:
                    weight.copy_(cases[0][0].module.ngram_embedding.weight)
            if compact:
                configure(table, Namespace(use_distributed_optimizer=True))
            set_defaults_if_not_set_tensor_model_parallel_attributes(weight)
            ddp = DistributedDataParallel(
                config,
                DistributedDataParallelConfig(
                    use_distributed_optimizer=True, grad_reduce_in_fp32=True, overlap_grad_reduce=True
                ),
                table,
            )
            optimizer = get_megatron_optimizer(
                OptimizerConfig(
                    optimizer="adam",
                    lr=0.01,
                    min_lr=0.001,
                    weight_decay=0.1,
                    bf16=True,
                    params_dtype=torch.bfloat16,
                    use_distributed_optimizer=True,
                    clip_grad=0.0,
                ),
                [ddp],
            )
            layouts = []
            weight.register_hook(lambda grad, seen=layouts: seen.append(grad.layout))
            cases.append((ddp, optimizer, layouts))

        dp_rank = ps.get_data_parallel_rank()
        split = cases[0][0].module.ngram_embedding.num_embeddings_per_partition
        # Each TP owner sees valid local row zero alongside masked IDs and repeated rows.
        base = torch.tensor([0, split, 7, split + 5, 11 + dp_rank, split + 13 + dp_rank], device="cuda")
        original_keys = cases[0][0].module.state_dict().keys()
        assert cases[1][0].module.state_dict().keys() == original_keys
        for step in range(2):
            states = []
            for compact_enabled, (ddp, optimizer, layouts) in enumerate(cases):
                layouts.clear()
                ddp.zero_grad_buffer()
                optimizer.zero_grad()
                outputs = []
                for microbatch in range(2):
                    indices = base.repeat(1024).reshape(1, -1, 2) + step * 41 + microbatch
                    with ddp.no_sync() if microbatch == 0 else nullcontext():
                        out = checkpoint(ddp, indices, use_reentrant=False)
                        outputs.append(out.detach().clone())
                        out.float().square().mean().backward()
                ddp.finish_grad_sync()
                assert layouts == [torch.sparse_coo if compact_enabled else torch.strided] * 2
                child = optimizer.chained_optimizers[0]
                weight = ddp.module.ngram_embedding.weight
                shard = child._get_model_param_range_map(weight)["param"]
                reduced = weight.main_grad.view(-1)[shard.start : shard.end].detach().clone()
                assert optimizer.step()[0]
                (master,) = [p for group in child.shard_fp32_from_float16_groups for p in group]
                adam = child.optimizer.state[master]
                states.append(
                    [
                        *outputs,
                        reduced,
                        weight.detach().clone(),
                        master.detach().clone(),
                        adam["exp_avg"].clone(),
                        adam["exp_avg_sq"].clone(),
                    ]
                )
            for dense, compact in zip(*states, strict=True):
                torch.testing.assert_close(compact, dense, atol=0, rtol=0)
    finally:
        ps.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Real TP2 / DP2 PLE gradient regression requires four GPUs")
def test_ple_row_grad_matches_dense_adam(tmp_path: Path) -> None:
    mp.spawn(_worker, args=((tmp_path / "ple-row-grad").as_uri(),), nprocs=4, join=True)


def _local_copy(value: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    local = value.to_local() if isinstance(value, DTensor) else value
    return local.detach().clone()


def _fsdp_worker(rank: int, rendezvous: str) -> None:
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.per_layer_embedding import NGramEmbedding
    from megatron.core.transformer.transformer_config import TransformerConfig

    from relax.models.qwen3_8.ops.ple_row_grad import configure

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=180))
    ps.initialize_model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(731)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=320,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=False,
            ple_ngram_size=2,
            ple_heads_per_ngram=2,
            ple_ngram_vocab_size_base=4093,
            ple_unigram_vocab_size=2048,
        )
        for strategy in ("optim_grads_params", "optim_grads"):
            runs = []
            initial = None
            for compact in (False, True):
                table = NGramEmbedding(config, 320, 0)
                weight = table.ngram_embedding.weight
                split = table.ngram_embedding.num_embeddings_per_partition
                if initial is None:
                    initial = weight.detach().clone()
                else:
                    with torch.no_grad():
                        weight.copy_(initial)
                if compact:
                    configure(
                        table,
                        Namespace(
                            use_distributed_optimizer=True,
                            use_megatron_fsdp=True,
                            data_parallel_sharding_strategy=strategy,
                            megatron_fsdp_main_grads_dtype=torch.float32,
                        ),
                    )
                set_defaults_if_not_set_tensor_model_parallel_attributes(weight)
                fsdp = FullyShardedDataParallel(
                    config,
                    DistributedDataParallelConfig(
                        use_megatron_fsdp=True,
                        use_distributed_optimizer=True,
                        grad_reduce_in_fp32=True,
                        overlap_grad_reduce=True,
                        overlap_param_gather=True,
                        data_parallel_sharding_strategy=strategy,
                        megatron_fsdp_main_params_dtype=torch.float32,
                        megatron_fsdp_main_grads_dtype=torch.float32,
                        megatron_fsdp_grad_comm_dtype=torch.float32,
                    ),
                    torch.nn.Sequential(table),
                    fsdp_unit_modules=[NGramEmbedding] if strategy == "optim_grads_params" else [],
                )
                optimizer = get_megatron_optimizer(
                    OptimizerConfig(
                        optimizer="adam",
                        lr=0.01,
                        min_lr=0.001,
                        weight_decay=0.1,
                        bf16=True,
                        params_dtype=torch.bfloat16,
                        use_distributed_optimizer=True,
                        store_param_remainders=False,
                        clip_grad=0.0,
                    ),
                    [fsdp],
                )
                ((_, master),) = fsdp.param_and_grad_buffer.optimizer_named_parameters
                layouts = []
                master.orig_param.register_hook(lambda grad, seen=layouts: seen.append(grad.layout))
                runs.append((fsdp, optimizer, master, layouts))

            dp = ps.get_data_parallel_rank()
            indices = torch.tensor([0, split, 7, split + 5, 11 + dp, split + 13 + dp], device="cuda")
            for step in range(2):
                states = []
                for compact, (fsdp, optimizer, master, layouts) in enumerate(runs):
                    optimizer.zero_grad()
                    layouts.clear()
                    result = []
                    for microbatch in range(2):
                        x = indices.repeat(1024).reshape(1, -1, 2) + step * 41 + microbatch
                        with fsdp.no_sync() if microbatch == 0 else nullcontext():
                            output = fsdp(x)
                            result.append(_local_copy(output))
                            output.float().square().mean().backward()
                    fsdp.finish_grad_sync()
                    assert layouts == [torch.sparse_coo if compact else torch.strided] * 2
                    assert master.grad.dtype == torch.float32
                    result.append(_local_copy(master.grad))
                    assert optimizer.step()[0]
                    adam = optimizer.optimizer.state[master]
                    result.extend(_local_copy(value) for value in (master, adam["exp_avg"], adam["exp_avg_sq"]))
                    states.append(result)
                for dense, compact in zip(*states, strict=True):
                    torch.testing.assert_close(compact, dense, atol=0, rtol=0)
    finally:
        ps.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Real TP2 / DP2 FSDP PLE regression requires four GPUs")
def test_ple_row_grad_matches_fsdp_adam(tmp_path: Path) -> None:
    mp.spawn(_fsdp_worker, args=((tmp_path / "ple-fsdp-row-grad").as_uri(),), nprocs=4, join=True)
