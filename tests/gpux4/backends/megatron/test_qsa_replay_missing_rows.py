# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run with torchrun --nproc-per-node=4 -m pytest on this file."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.gpux4.backends.megatron.test_qsa_tp_selector import indexer, inputs


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="Requires torchrun with four CUDA ranks",
)


@pytest.fixture(scope="module")
def groups():
    from megatron.core import parallel_state

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=4, context_parallel_size=1)
    yield SimpleNamespace(
        tp=parallel_state.get_tensor_model_parallel_group(), cp=parallel_state.get_context_parallel_group()
    )
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.parametrize("case", ["complete", "one", "documents_padding"])
def test_qsa_missing_rows_reduce_matches_full_reference_gather(groups, monkeypatch, case):
    from relax.models.qwen3_8.ops.qsa_mixed_ops import selector

    torch.manual_seed(918)
    monkeypatch.setenv("RELAX_QWEN38_QSA_SELECTOR_COMPILE", "0")
    lengths = (2052,) if case != "documents_padding" else (2052, 2052, 2052, 2052)
    value = indexer(groups, 512)
    hidden, freqs, packed = inputs(lengths)
    local = hidden.chunk(4)[dist.get_rank(group=groups.tp)].unsqueeze(1).contiguous()
    native = selector.forward(value, local, freqs, packed, "ids")
    total = sum(lengths)
    missing = () if case == "complete" else (total - 1,)
    if case == "documents_padding":
        # Doc-final rows span every query owner; extra holes model padding.
        missing = (0, 2051, 2052, 4103, 4104, 6155, 6156, 8207)
    captured = native.selected_ids.sort().values
    mask = torch.ones(total, dtype=torch.bool, device="cuda")
    mask[list(missing)] = False
    # No CPU missing metadata executes the previous full-gather reference path.
    original = selector.forward(value, local, freqs, packed, "ids", replay_data=(captured, mask))
    captured._relax_indexer_missing_rows = missing
    reductions, gathers = [], []
    reduce, gather = dist.all_reduce, dist.all_gather_into_tensor

    def track_reduce(tensor, *args, **kwargs):
        reductions.append((tensor.numel() * tensor.element_size(), kwargs.get("group"), kwargs.get("op")))
        return reduce(tensor, *args, **kwargs)

    def track_gather(output, source, *args, **kwargs):
        gathers.append(tuple(output.shape))
        return gather(output, source, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(dist, "all_reduce", track_reduce)
        scoped.setattr(dist, "all_gather_into_tensor", track_gather)
        scoped.setattr(torch.Tensor, "item", lambda *args: (_ for _ in ()).throw(AssertionError("GPU scalar read")))
        scoped.setattr(torch.Tensor, "tolist", lambda *args: (_ for _ in ()).throw(AssertionError("GPU list read")))
        actual = selector.forward(value, local, freqs, packed, "ids", replay_data=(captured, mask))
    torch.testing.assert_close(actual.selected_ids, original.selected_ids, atol=0, rtol=0)
    torch.testing.assert_close(actual.selected_ids.sort().values, native.selected_ids.sort().values, atol=0, rtol=0)
    assert actual._relax_qsa_mixed_bounds == original._relax_qsa_mixed_bounds
    assert reductions == ([(len(missing) * 512 * 4, groups.tp, dist.ReduceOp.SUM)] if missing else [])
    # The hidden SP gather remains necessary, but the full [total,512] IDs do not.
    assert (total, 512) not in gathers
    assert torch.cuda.max_memory_allocated() < 1 << 30


def test_qsa_missing_rows_full_module_output_and_gradients(groups, monkeypatch):
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import get_qsa_module_spec_for_backend
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig

    from relax.models.qwen3_8.ops.qsa_mixed_ops import selector

    for name, value in {
        "RELAX_QWEN38_QSA_COMMUNITY_BACKEND": "mixed",
        "MCORE_QSA_SPARSE_BACKEND": "id_sparse",
        "RELAX_QWEN38_QSA_FUSED_ATTENTION": "0",
        "RELAX_QWEN38_QSA_FUSED_SELECTOR": "0",
        "RELAX_QWEN38_QSA_TP_SELECTOR": "1",
        "RELAX_QWEN38_QSA_SELECTOR_COMPILE": "0",
        "RELAX_QWEN38_QSA_ROUTED_QKV": "1",
    }.items():
        monkeypatch.setenv(name, value)
    torch.manual_seed(93)
    model_parallel_cuda_manual_seed(93)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=256,
        ffn_hidden_size=256,
        tensor_model_parallel_size=4,
        context_parallel_size=1,
        sequence_parallel=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        layernorm_epsilon=1e-6,
        qk_layernorm=True,
        attention_output_gate=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        transformer_impl="transformer_engine",
        gradient_accumulation_fusion=False,
        qsa_indexer_n_heads=4,
        qsa_indexer_kv_heads=1,
        qsa_indexer_head_dim=128,
        qsa_indexer_budget=2048,
        qsa_indexer_compress_ratio=4,
        qsa_indexer_loss_coeff=0,
        apply_rope_fusion=False,
    )
    config.rotary_percent = 0.25
    config.relax_use_indexer_replay = False
    model = build_module(
        get_qsa_module_spec_for_backend(config),
        config=config,
        layer_number=1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    ).cuda()
    model.indexer.requires_grad_(False)
    hidden, freqs, packed = inputs((2052,))
    from relax.backends.megatron.qsa_layout_metadata import bind_physical_layout

    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=packed.cu_seqlens_q,
        cu_seqlens_kv=packed.cu_seqlens_q,
        max_seqlen_q=2052,
        max_seqlen_kv=2052,
    )
    packed.qsa_select_offsets_cpu = (0, 2052)
    bind_physical_layout(packed, (0, 2052))
    hidden = hidden.repeat(1, 4)  # Match this smaller real module's hidden width.
    local = hidden.chunk(4)[dist.get_rank(group=groups.tp)].unsqueeze(1).contiguous()
    native = selector.forward(model.indexer, local, freqs, packed, "ids")
    captured = native.selected_ids.sort().values
    mask = torch.ones(2052, dtype=torch.bool, device="cuda")
    mask[-1] = False
    real_forward = selector.forward

    def run(missing):
        data = captured.clone()
        if missing is not None:
            data._relax_indexer_missing_rows = missing

        def replay(owner, *args, **kwargs):
            return real_forward(owner, *args, **kwargs, replay_data=(data, mask))

        model.zero_grad(set_to_none=True)
        value = local.detach().clone().requires_grad_()
        with monkeypatch.context() as scoped:
            scoped.setattr(selector, "forward", replay)
            output, bias = model(value, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=packed)
            output = output if bias is None else output + bias
            output.float().square().mean().backward()
        return (
            output.detach().clone(),
            value.grad.clone(),
            {
                name: parameter.grad.clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            },
        )

    before, after = run(None), run((2051,))
    torch.testing.assert_close(after[0], before[0], atol=0, rtol=0)
    torch.testing.assert_close(after[1], before[1], atol=0, rtol=0)
    assert after[2].keys() == before[2].keys()
    for name in before[2]:
        torch.testing.assert_close(after[2][name], before[2][name], atol=0, rtol=0)
    assert torch.cuda.max_memory_allocated() < 1 << 30
