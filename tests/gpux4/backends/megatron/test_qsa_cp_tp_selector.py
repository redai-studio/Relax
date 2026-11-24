# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""TP2/CP2 selected-ID partition with real indexer, RoPE and attention."""

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, rendezvous: str) -> None:
    from megatron.core import parallel_state as ps
    from megatron.core.extensions.transformer_engine import TELinear, TENorm
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.experimental_attention_variant.qsa import QSAIndexer, QSAIndexerSubmodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    from relax.backends.megatron.qsa_layout_metadata import bind_physical_layout
    from relax.models.qwen3_8.ops.qsa_cute import cp
    from relax.models.qwen3_8.ops.qsa_cute.triton_qsa_select import qsa_cp_layout_static

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=240))
    ps.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
    model_parallel_cuda_manual_seed(103)
    os.environ["RELAX_QWEN38_QSA_SELECTOR_BACKEND"] = "torch"
    try:
        groups = ProcessGroupCollection(tp=ps.get_tensor_model_parallel_group(), cp=ps.get_context_parallel_group())
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=256,
            tensor_model_parallel_size=2,
            context_parallel_size=2,
            sequence_parallel=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            normalization="RMSNorm",
            layernorm_zero_centered_gamma=True,
            attention_dropout=0,
            hidden_dropout=0,
            apply_rope_fusion=False,
            qsa_indexer_n_heads=4,
            qsa_indexer_kv_heads=1,
            qsa_indexer_head_dim=128,
            qsa_indexer_budget=2048,
            qsa_indexer_compress_ratio=4,
            qsa_indexer_loss_coeff=0,
        )
        config.mrope_section = [11, 11, 10]
        indexer = (
            QSAIndexer(
                config, QSAIndexerSubmodules(linear_qk=TELinear, q_layernorm=TENorm, k_layernorm=TENorm), 1, groups
            )
            .cuda()
            .bfloat16()
            .requires_grad_(False)
        )
        for parameter in indexer.parameters():
            dist.broadcast(parameter, src=0, group=dist.group.WORLD)
        bounds = (0, 0, 32, 4168, 6240)
        cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cu, cu_seqlens_kv_padded=cu
        )
        packed.cp_partition_mode, packed.qsa_select_offsets_cpu = "zigzag", bounds
        bind_physical_layout(packed, bounds)
        rows, _ = qsa_cp_layout_static(cu, 3120, 2, dist.get_rank(groups.cp), "per_document")
        tp_rank = dist.get_rank(groups.tp)
        generator = torch.Generator(device="cuda").manual_seed(107)
        full_hidden = torch.randn(6240, 1, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
        hidden = full_hidden[rows].chunk(2)[tp_rank].contiguous()
        inverse = 10000000 ** (-torch.arange(0, 64, 2, device="cuda").float() / 64)
        freqs = torch.cat([torch.arange(6240, device="cuda")[:, None] * inverse] * 2, -1)[:, None, None]
        real_select = cp.select_ids
        work = []

        def measured_select(query, *args, **kwargs):
            work.append(query.shape[0])
            return real_select(query, *args, **kwargs)

        def select(enabled: bool, frequencies: torch.Tensor):
            os.environ["RELAX_QWEN38_QSA_CP_TP_SELECTOR"] = str(int(enabled))
            work.clear()
            with patch.object(cp, "select_ids", measured_select):
                result = cp.forward(indexer, hidden, frequencies, packed)
            return result, sum(work)

        off, old_work = select(False, freqs)
        on, new_work = select(True, freqs)
        _, counts = cp.selector_query_plan(bounds, 2, 2, 2048)
        assert old_work == sum(counts) and new_work == counts[tp_rank] < old_work
        torch.testing.assert_close(on.selected_ids, off.selected_ids, atol=0, rtol=0)
        local_freq, _ = select(True, freqs[rows].contiguous())
        torch.testing.assert_close(local_freq.selected_ids, off.selected_ids, atol=0, rtol=0)
        for name in ("doc_ids", "positions", "_relax_qsa_query_positions", "_relax_qsa_key_reorder"):
            torch.testing.assert_close(getattr(on, name), getattr(off, name), atol=0, rtol=0)

        # Exercise actual selected IDs in the attention path, including remote dKV.
        full = [
            torch.randn(6240, heads, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
            for heads in (12, 1, 1)
        ]
        derivative = torch.randn(3120, 12 * 256, device="cuda", dtype=torch.bfloat16, generator=generator)
        core = SimpleNamespace(pg_collection=groups, softmax_scale=256**-0.5)
        outputs = []
        for selection in (off, on):
            local = [x[rows].detach().requires_grad_() for x in full]
            output = cp.core_forward(core, *local, selection, packed)
            output.backward(derivative)
            outputs.append((output.detach(), [x.grad for x in local]))
        torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=0, rtol=0)
        for before, after in zip(outputs[0][1], outputs[1][1], strict=True):
            torch.testing.assert_close(before, after, atol=0.002, rtol=0.002)

        # Zero projection gives exact score ties; keep the native top-k order.
        with torch.no_grad():
            indexer.index_qk_proj.weight.zero_()
        tied_off, _ = select(False, freqs)
        tied_on, _ = select(True, freqs)
        torch.testing.assert_close(tied_on.selected_ids, tied_off.selected_ids, atol=0, rtol=0)
    finally:
        ps.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4 or torch.cuda.get_device_capability()[0] != 10,
    reason="QSA CP TP selector requires four Blackwell GPUs and patched MCore",
)
def test_qsa_cp_tp_selector_ids_and_attention_gradients(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "qsa-cp-tp").as_uri(),), nprocs=4, join=True)
