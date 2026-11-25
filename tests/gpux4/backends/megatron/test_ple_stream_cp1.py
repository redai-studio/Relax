# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, rendezvous: str) -> None:
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import checkpoint, model_parallel_cuda_manual_seed
    from megatron.core.transformer.per_layer_embedding import PerLayerEmbedding
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    ps.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=1)
    model_parallel_cuda_manual_seed(92)
    os.environ.update(
        QWEN38_PLE_CPU_OFFLOAD="0",
        QWEN38_PLE_INPUT_HALO_CP="0",
        RELAX_QWEN38_PLE_STREAMED="0",
        RELAX_QWEN38_PLE_FUSED_CONV="0",
    )
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            context_parallel_size=1,
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
        model = PerLayerEmbedding(config, 1).cuda().requires_grad_(False)
        torch.manual_seed(93)
        for parameter in model.parameters():
            parameter.normal_(0, 0.1)

        # Both ranks cross an 8192-row chunk boundary. The second document
        # starts inside rank one's first convolution halo.
        local_rows = 8196
        sequence = local_rows * 2
        cu = torch.tensor([0, local_rows + 3, sequence], device="cuda", dtype=torch.int32)
        tokens = torch.randint(2, 64, (1, sequence), device="cuda")
        hidden = torch.randn(sequence, 1, 128, device="cuda", dtype=torch.bfloat16)
        upstream = torch.randn_like(hidden)
        upstream[:local_rows] = 0

        def run(value: torch.Tensor, derivative: torch.Tensor, recompute: bool = False):
            value = value.detach().clone().requires_grad_()
            output = checkpoint(model, False, value) if recompute else model(value)
            output.backward(derivative)
            return output.detach(), value.grad

        model.prepare(tokens, cu)
        reference, gradient = run(hidden, upstream)
        # Rank zero receives gradient only through rank one's convolution.
        assert gradient[local_rows - model.halo : local_rows].count_nonzero() > 0

        os.environ["RELAX_QWEN38_PLE_STREAMED"] = "1"
        output, hidden_grad = run(hidden, upstream)
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
        torch.testing.assert_close(hidden_grad, gradient, atol=0.004, rtol=0.03)

        config.sequence_parallel = True
        model.ple_embedding.ngram_embedding.reduce_scatter_embeddings = True
        model.prepare(tokens, cu)
        owned = slice(rank * local_rows, (rank + 1) * local_rows)
        output, hidden_grad = run(hidden[owned], upstream[owned], recompute=True)
        torch.testing.assert_close(output, reference[owned], atol=0, rtol=0)
        if rank == 0:
            assert hidden_grad[-model.halo :].count_nonzero() > 0
        # Splitting the convolution halo changes BF16 gradient accumulation order.
        torch.testing.assert_close(hidden_grad, gradient[owned], atol=0.004, rtol=0.03)
    finally:
        ps.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs for the PLE TP halo collective")
def test_streamed_ple_cp1_matches_native_forward_and_gradient(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "ple-cp1").as_uri(),), nprocs=2, join=True)
