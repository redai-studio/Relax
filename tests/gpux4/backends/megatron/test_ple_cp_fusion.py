# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real TP2/CP2 PLE fusion, native ownership and checkpoint recompute."""

import os
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, rendezvous: str) -> None:
    from megatron.core import parallel_state as ps
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.ssm.mamba_context_parallel import split_tensor_cp
    from megatron.core.tensor_parallel.random import checkpoint, model_parallel_cuda_manual_seed
    from megatron.core.transformer.per_layer_embedding import PerLayerEmbedding
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=240))
    ps.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
    model_parallel_cuda_manual_seed(92)
    os.environ.update(
        RELAX_QWEN38_PLE_STREAMED="0",
        RELAX_QWEN38_PLE_FUSED_CONV="0",
        RELAX_QWEN38_PLE_STREAM_CHUNK_SIZE="0",
    )
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            context_parallel_size=2,
            sequence_parallel=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            enable_hyper_connections=True,
            num_residual_streams=2,
            ple_layer_ids=[1],
            ple_embed_dim=32,
            ple_heads_per_ngram=2,
            ple_ngram_vocab_size_base=31,
            ple_ngram_vocab_divisible_by=8,
            ple_eos_token_id=1,
            ple_unigram_vocab_size=64,
            ple_conv_kernel_size=3,
        )
        model = PerLayerEmbedding(config, 1).cuda()
        torch.manual_seed(93)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(0, 0.1)
                dist.broadcast(parameter, src=0, group=dist.group.WORLD)
        cu = torch.tensor([0, 16, 40, 80], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
        hidden = torch.randn(80, 1, 128, device="cuda", dtype=torch.bfloat16)
        tokens = torch.randint(2, 64, (1, 80), device="cuda")
        tokens[:, 9] = 1
        upstream = torch.randn_like(hidden)
        # One CP rank has zero supervision, but must participate in backward.
        indices = split_tensor_cp(torch.arange(80, device="cuda"), packed, dim=0)
        zero_indices = indices.clone()
        dist.broadcast(
            zero_indices,
            src=dist.get_process_group_ranks(ps.get_context_parallel_group())[0],
            group=ps.get_context_parallel_group(),
        )
        upstream[zero_indices] = 0
        tp_rank = dist.get_rank(ps.get_tensor_model_parallel_group())
        local_indices = indices.chunk(2)[tp_rank]

        def run(value: torch.Tensor, derivative: torch.Tensor, recompute: bool = False):
            model.zero_grad(set_to_none=True)
            value = value.detach().clone().requires_grad_()
            output = checkpoint(model, False, value) if recompute else model(value)
            output.backward(derivative)
            return (
                output.detach(),
                value.grad,
                {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None},
            )

        for frozen in (False, True):
            model.requires_grad_(not frozen)
            os.environ["RELAX_QWEN38_PLE_STREAMED"] = "0"
            os.environ["RELAX_QWEN38_PLE_CP_FUSED_CONV"] = "0"
            # Full-sequence native oracle uses the same real TP vocabulary table,
            # without SP output sharding or CP reconstruction.
            with patch.object(ps, "get_context_parallel_world_size", return_value=1):
                config.sequence_parallel = False
                model.ple_embedding.ngram_embedding.reduce_scatter_embeddings = False
                model._cp_packed_seq_params = None
                model.prepare(tokens, cu)
                reference = run(hidden, upstream)
            config.sequence_parallel = True
            model.ple_embedding.ngram_embedding.reduce_scatter_embeddings = True
            model.prepare(tokens[:, indices], cu)
            native = run(hidden[local_indices], upstream[local_indices])
            for fused, recompute in ((False, False), (True, False), (True, True)):
                os.environ["RELAX_QWEN38_PLE_CP_FUSED_CONV"] = str(int(fused))
                actual = run(hidden[local_indices], upstream[local_indices], recompute) if fused else native
                torch.testing.assert_close(actual[0], native[0], atol=0, rtol=0)
                torch.testing.assert_close(actual[0], reference[0][local_indices], atol=0, rtol=0)
                torch.testing.assert_close(actual[1].float(), native[1].float(), atol=0.004, rtol=0.03)
                torch.testing.assert_close(
                    actual[1].float(), reference[1][local_indices].float(), atol=0.004, rtol=0.03
                )
                for name, gradient in actual[2].items():
                    torch.testing.assert_close(gradient.float(), native[2][name].float(), atol=0.01, rtol=0.03)
                    group = ps.get_context_parallel_group() if name.startswith("ple_embedding.") else dist.group.WORLD
                    summed = gradient.float()
                    dist.all_reduce(summed, group=group)
                    torch.testing.assert_close(summed, reference[2][name].float(), atol=0.03, rtol=0.04, msg=name)
    finally:
        ps.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4 or torch.cuda.get_device_capability()[0] != 10,
    reason="PLE CP fusion requires four Blackwell GPUs and the patched MCore image",
)
def test_ple_cp_fusion_native_gradients_and_recompute(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "ple-cp").as_uri(),), nprocs=4, join=True)
