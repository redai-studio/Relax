# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Static TP4/CP2 PLE recompute, DDP gradient scaling and convolution
rounding."""

import os
from datetime import timedelta
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


class _PLELayer(torch.nn.Module):
    def __init__(self, ple):
        super().__init__()
        self.per_layer_embedding = ple

    def forward(self, hidden_states, context, **kwargs):
        return self.per_layer_embedding(hidden_states), context


class _CheckpointBlock(torch.nn.Module):
    """One real PLE layer inside the same Bridge recomputation closure used for
    training."""

    def __init__(self, ple):
        super().__init__()
        self.layer = _PLELayer(ple)
        self.config = ple.config
        self.pre_process = False
        self.num_layers_per_pipeline_rank = 1

    def _get_layer(self, index):
        assert index == 0
        return self.layer

    def forward(self, hidden, packed):
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import Qwen3VLTransformerBlock

        return Qwen3VLTransformerBlock._relax_native_checkpointed_forward(
            self, hidden, None, None, None, None, None, packed, None, False
        )


def _worker(rank: int, rendezvous: str) -> None:
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.ssm.mamba_context_parallel import split_tensor_cp
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.per_layer_embedding import PerLayerEmbedding
    from megatron.core.transformer.transformer_config import TransformerConfig

    from relax.backends.megatron.qsa_layout_metadata import bind_physical_layout
    from relax.models.qwen3_8.ops import ple_sharded

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=8, timeout=timedelta(minutes=8))
    try:
        ps.initialize_model_parallel(tensor_model_parallel_size=4, context_parallel_size=2)
        model_parallel_cuda_manual_seed(92)
        os.environ.update(
            QWEN38_PLE_CPU_OFFLOAD="1",
            QWEN38_PLE_INPUT_HALO_CP="0",
            RELAX_QWEN38_PLE_STREAMED="0",
            RELAX_QWEN38_PLE_FUSED_CONV="0",
            RELAX_QWEN38_PLE_CP_FUSED_CONV="1",
        )
        config = TransformerConfig(
            num_layers=1,
            hidden_size=2560,
            num_attention_heads=16,
            tensor_model_parallel_size=4,
            context_parallel_size=2,
            sequence_parallel=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            calculate_per_token_loss=True,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
            enable_hyper_connections=True,
            num_residual_streams=4,
            ple_layer_ids=[1],
            ple_embed_dim=64,
            ple_heads_per_ngram=2,
            ple_ngram_size=3,
            ple_ngram_vocab_size_base=31,
            ple_ngram_vocab_divisible_by=8,
            ple_eos_token_id=1,
            ple_unigram_vocab_size=64,
            ple_conv_kernel_size=4,
        )
        ple = PerLayerEmbedding(config, 1).cuda()
        table = ple.ple_embedding.ngram_embedding
        tp_rank = ps.get_tensor_model_parallel_rank()
        generator = torch.Generator().manual_seed(203 + tp_rank)
        table.load_state_dict({"weight": torch.randn(table.weight.shape, generator=generator) * 0.2})
        torch.manual_seed(93)
        with torch.no_grad():
            for param in ple.parameters():
                param.normal_(0, 0.1)
                dist.broadcast(param, src=0, group=dist.group.WORLD)
        params = dict(ple.named_parameters())
        assert set(params) == {
            "key_proj.weight",
            "value_proj.weight",
            "norm_key.weight",
            "norm_query.weight",
            "norm_conv.weight",
            "conv1d.weight",
        }
        block = _CheckpointBlock(ple)
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=False,
                use_distributed_optimizer=False,
                average_in_collective=False,
            ),
            block,
        )
        gpt = SimpleNamespace(config=config, decoder=SimpleNamespace(layers=[block.layer]))
        batches = []
        expected_tokens = 0
        for i, lengths in enumerate(((32, 80, 16, 48), (16, 32, 64))):
            bounds = tuple(accumulate(lengths, initial=0))
            total = bounds[-1]
            cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
            packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
            physical = packed
            if i == 1:
                # The producer may pad each document. PLE routes physical token rows.
                logical_cu = torch.tensor(
                    tuple(accumulate((length - 1 for length in lengths), initial=0)),
                    device="cuda",
                    dtype=torch.int32,
                )
                packed = PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=logical_cu,
                    cu_seqlens_kv=logical_cu,
                    cu_seqlens_q_padded=cu,
                    cu_seqlens_kv_padded=cu,
                )
            bind_physical_layout(packed, bounds)
            ids = torch.randint(2, 64, (1, total), device="cuda")
            ids[:, lengths[0] + 1] = 1
            x = torch.randn(total, 1, ple.n * config.hidden_size, device="cuda", dtype=torch.bfloat16)
            dy = torch.randn_like(x)
            for tensor in (ids, x, dy):
                dist.broadcast(tensor, src=0, group=dist.group.WORLD)
            # Physical ownership from the native TP4/CP2 packer. The first batch leaves
            # one whole CP rank unsupervised while its halos can affect a peer's loss.
            cp_rows = split_tensor_cp(torch.arange(total, device="cuda"), physical, dim=0)
            rows = cp_rows.chunk(4)[tp_rank]
            mask = torch.arange(total, device="cuda") % 3 != 0
            if i == 0:
                peer = cp_rows.clone()
                group = ps.get_context_parallel_group()
                dist.broadcast(peer, dist.get_process_group_ranks(group)[0], group=group)
                mask[peer] = False
            else:
                mask[cu[1:].long() - 1] = False
            expected_tokens += int(mask.sum())
            dy = dy * mask[:, None, None]
            batches.append((ids[:, cp_rows], x[rows], dy[rows], packed, mask[cp_rows].sum()))

        def run(sharded):
            os.environ["RELAX_QWEN38_PLE_SHARDED"] = str(int(sharded))
            ddp.zero_grad_buffer()
            outs, inputs = [], []
            for ids, hidden, _, packed, _ in batches:
                GPTModel._prepare_per_layer_embeddings(gpt, ids, packed, None)
                if sharded:
                    assert getattr(ple._cp_packed_seq_params, "sharded_ple", None) is not None
                local = hidden.detach().requires_grad_()
                inputs.append(local)
                outs.append(ddp(local, packed))
            for output, batch in zip(reversed(outs), reversed(batches), strict=True):
                output.backward(batch[2])

            # Separate oracle for the reduction factor. All eight owners contribute
            # to these replicated parameters; count supervised tokens once per CP
            # shard (replicated on TP). We never manually alter the DDP buffers.
            reduced = {n: p.main_grad.clone() for n, p in params.items()}
            for grad in reduced.values():
                dist.all_reduce(grad, group=dist.group.WORLD)
                grad.div_(expected_tokens)
            local_tokens = sum(batch[-1] for batch in batches).to(torch.int)
            finalize_model_grads([ddp], num_tokens=local_tokens)
            assert int(local_tokens) == expected_tokens
            grads = {n: p.main_grad.clone() for n, p in params.items()}
            for name in params:
                torch.testing.assert_close(grads[name], reduced[name], rtol=2e-5, atol=1e-6, msg=name)
            return [t.detach() for t in outs], [t.grad for t in inputs], grads

        reference = run(False)
        # Replaying the two microbatches in reverse must restore their original routes and hashes.

        seen = []
        sharded_forward = ple_sharded.sharded_forward

        def record_forward(layer, hidden, state):
            seen.append(state)
            return sharded_forward(layer, hidden, state)

        with (
            patch.object(ple_sharded, "_build_route", wraps=ple_sharded._build_route) as build,
            patch.object(ple_sharded, "sharded_forward", side_effect=record_forward),
        ):
            actual = run(True)
        assert build.call_count == 2
        assert len(seen) == 4
        assert seen[0].route is not seen[1].route
        assert seen[2] is seen[1] and seen[3] is seen[0]
        for ref, got in zip(reference[0] + reference[1], actual[0] + actual[1], strict=True):
            torch.testing.assert_close(got.float(), ref.float(), atol=0.02, rtol=0.035)
        for name, ref in reference[2].items():
            got = actual[2][name]
            norm = ref.norm()
            assert norm > 0, name
            rel_l2 = (got - ref).norm() / norm
            scale = (got.flatten() @ ref.flatten()) / norm.square()
            assert rel_l2 < 0.04, (name, float(rel_l2), float(scale))
            assert abs(scale - 1) < 0.04, (name, float(scale))
    finally:
        ps.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs with the patched Megatron PLE image")
def test_ple_sharded_matches_native_ddp_gradients_under_recompute(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "ple-ddp").as_uri(),), nprocs=8, join=True)


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="requires CUDA BF16")
def test_ple_sharded_unit_tap_matches_compiled_silu_gradient():
    from relax.models.qwen3_8.ops.ple_fused_conv import _compiled_conv
    from relax.models.qwen3_8.ops.ple_sharded import _build_route, sharded_convolution

    # These two BF16 inputs straddle different rounding thresholds in eager and
    # compiled SiLU backward on Hopper. An identity kernel isolates the activation.
    values = torch.tensor([0.007537841796875, -7.9375], device="cuda", dtype=torch.bfloat16)
    upstream = torch.tensor([1.5546875, 1.0078125], device="cuda", dtype=torch.bfloat16)
    rows = 16
    normed = values.repeat(rows, 1, 1).requires_grad_()
    gated = torch.zeros_like(normed)
    weight = torch.zeros(values.numel(), 1, 4, device="cuda", dtype=normed.dtype)
    weight[..., -1] = 1
    weight.requires_grad_()
    dy = upstream.expand_as(normed).contiguous()
    positions = torch.arange(rows, device="cuda")[:, None]
    dilation = 3
    halo = (weight.size(-1) - 1) * dilation
    aggregate = _compiled_conv()(F.pad(normed, (0, 0, 0, 0, halo, 0)), normed, gated, weight, positions, dilation)
    _, taps, _, capacity = _build_route((0, rows), ((0, 0),), 0, weight.size(-1), dilation)
    assert capacity == 0
    route = SimpleNamespace(capacity=capacity, taps=torch.as_tensor(taps, device="cuda"))
    sharded = sharded_convolution(SimpleNamespace(route=route), normed, gated, weight)
    torch.testing.assert_close(sharded, aggregate, rtol=0, atol=0)
    reference = torch.autograd.grad(aggregate, normed, dy)[0]
    actual = torch.autograd.grad(sharded, normed, dy)[0]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
