# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare selector routes, query sharding and replay with pinned image
sources."""

from __future__ import annotations

import ast
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.distributed as dist


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="TP selector validation requires torchrun with two CUDA ranks",
)


@pytest.fixture(scope="module")
def groups():
    from megatron.core import parallel_state

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=1)
    yield SimpleNamespace(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def image():
    from megatron.core import parallel_state
    from megatron.core.models.common.embeddings import rope_utils

    root = Path(os.environ["QSA_TP_SELECTOR_REFERENCE_RUNTIME"])
    sys.path.insert(0, str(root))
    # Execute the image's actual norm functions, without importing its model/plugin.
    norm = types.ModuleType("qsa_norm_reference")
    norm.__dict__.update(torch=torch, Tensor=torch.Tensor)
    tree = ast.parse((root / "qsa_indexer.py").read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_indexer_acc_dtype", "gemma_rmsnorm_last_dim")
    ]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(root / "qsa_indexer.py"), "exec"), norm.__dict__)
    source = types.ModuleType("qsa_packed_image_reference")
    # Inject the original norm function, keeping the reference's arithmetic intact.
    tree = ast.parse((root / "qsa_packed.py").read_text())
    select = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "select")
    imports = [
        node
        for node in select.body
        if isinstance(node, ast.ImportFrom)
        and len(node.names) == 1
        and node.names[0].name == "gemma_rmsnorm_last_dim"
        and node.names[0].asname is None
    ]
    assert len(imports) == 1, "Reference selector must import the expected norm function"
    select.body.remove(imports[0])
    source.gemma_rmsnorm_last_dim = norm.gemma_rmsnorm_last_dim
    exec(compile(tree, str(root / "qsa_packed.py"), "exec"), source.__dict__)
    # Compare query RoPE against the image's unfused CP1 helper, rather than
    # silently using the target helper for both sides of the comparison.
    helpers = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "Optional": Optional,
        "parallel_state": parallel_state,
        "warnings": __import__("warnings"),
    }
    tree = ast.parse((root / "rope_utils.py").read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_rotate_half", "_apply_rotary_pos_emb_bshd", "_apply_rotary_pos_emb_thd")
    ]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(root / "rope_utils.py"), "exec"), helpers)

    def apply(q, freqs, config, cu_seqlens=None, max_seqlen=None, **kwargs):
        assert not config.apply_rope_fusion
        return helpers["_apply_rotary_pos_emb_thd"](
            q,
            cu_seqlens,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            cp_group=parallel_state.get_context_parallel_group(),
            max_seqlen=max_seqlen,
        )

    yield SimpleNamespace(
        module=source, norm=norm.gemma_rmsnorm_last_dim, query_rope=apply, target_rope=rope_utils, root=root
    )
    sys.path.remove(str(root))


class Projection(torch.nn.Module):
    def __init__(self, width: int, *, ties: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(640, width, device="cuda", dtype=torch.bfloat16) * 0.03, requires_grad=False
        )
        if ties:
            self.weight.data.zero_()
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        return torch.nn.functional.linear(hidden, self.weight), None


def indexer(groups, budget: int, *, ties: bool = False):
    from megatron.core.transformer.experimental_attention_variant import qsa

    value = qsa.QSAIndexer.__new__(qsa.QSAIndexer)
    torch.nn.Module.__init__(value)
    value.config = SimpleNamespace(
        context_parallel_size=1,
        qsa_indexer_loss_coeff=0,
        layernorm_zero_centered_gamma=True,
        layernorm_epsilon=1e-6,
        sequence_parallel=True,
        rotary_interleaved=False,
        mrope_section=None,
        apply_rope_fusion=False,
        multi_latent_attention=False,
    )
    value.n_heads, value.kv_heads, value.head_dim = 4, 1, 128
    value.token_budget, value.compress_ratio, value.block_topk = budget * 4, 4, budget
    value.pg_collection = groups
    value.index_qk_proj = Projection(64, ties=ties)
    for name in ("q_layernorm", "k_layernorm"):
        norm = torch.nn.Module()
        norm.weight = torch.nn.Parameter(
            torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.02, requires_grad=False
        )
        setattr(value, name, norm)
    return value


def inputs(lengths):
    from relax.backends.megatron.qsa_layout_metadata import bind_physical_layout

    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    packed = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_q_padded=None,
        max_seqlen_q=max(lengths),
        qsa_select_offsets_cpu=tuple(bounds),
    )
    bind_physical_layout(packed, tuple(bounds))
    hidden = torch.randn(bounds[-1], 64, device="cuda", dtype=torch.bfloat16)
    inverse = 10000 ** (-torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64)
    base = torch.arange(max(lengths), device="cuda", dtype=torch.float32)[:, None] * inverse[None]
    freqs = torch.cat((base, base), dim=-1)[:, None, None]
    return hidden, freqs, packed


def expanded(selection):
    from relax.models.qwen3_8.ops.qsa_mixed_ops.routes import token_routes

    result = torch.nn.functional.pad(token_routes(selection, 1, selection.positions.shape[1])[0], (0, 1), value=-1)
    result._qsa_bounds = selection._relax_qsa_mixed_bounds
    return result


def reference(image, value, hidden, freqs, packed, monkeypatch):
    proxy = SimpleNamespace(
        index_qk_proj=value.index_qk_proj,
        n_heads=4,
        head_dim=128,
        compress_ratio=4,
        token_topk=value.token_budget,
        q_layernorm=value.q_layernorm.weight,
        k_layernorm=value.k_layernorm.weight,
        norm_eps=value.config.layernorm_epsilon,
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(image.target_rope, "apply_rotary_pos_emb", image.query_rope)
        return image.module.select(
            proxy, hidden, freqs, packed.cu_seqlens_q, value.config, query_group=value.pg_collection.tp
        )


def report(row):
    if dist.get_rank(group=dist.group.WORLD) != 0:
        return
    print("QSA_TP_SELECTOR " + json.dumps(row), flush=True)
    destination = os.environ.get("QSA_TP_SELECTOR_REPORT")
    if destination:
        with Path(destination).open("a") as stream:
            stream.write(json.dumps(row) + "\n")


@pytest.mark.parametrize(
    "lengths,budget,ties",
    [
        ((0, 1, 3, 4, 5, 7), 8, False),
        ((35, 1), 8, False),
        ((3, 36, 5), 8, False),
        ((2051, 1), 512, False),
        ((2052,), 512, False),
        ((1, 2053, 2), 512, False),
        ((16384,), 512, False),
        ((3, 16385, 0, 4), 512, False),
        ((37, 3), 8, True),
        ((2047, 2052, 1), 512, False),
    ],
)
def test_selector_matches_image_routes_and_replay(groups, image, monkeypatch, lengths, budget, ties):
    from relax.models.qwen3_8 import qsa_cp as qsa_cp_patch
    from relax.models.qwen3_8.ops.qsa_mixed_ops import selector

    torch.manual_seed(20261014)
    value = indexer(groups, budget, ties=ties)
    hidden, freqs, packed = inputs(lengths)
    want = reference(image, value, hidden, freqs, packed, monkeypatch)
    value.index_qk_proj.calls = 0
    local = hidden.chunk(2)[dist.get_rank(group=groups.tp)].unsqueeze(1).contiguous()
    score_rows, gathers = [], []
    original_scores, original_gather = selector.index_scores, dist.all_gather_into_tensor

    def scores(q, keys):
        score_rows.append(q.shape[0])
        return original_scores(q, keys)

    def gather(*args, **kwargs):
        gathers.append(kwargs.get("group"))
        return original_gather(*args, **kwargs)

    monkeypatch.setattr(selector, "index_scores", scores)
    monkeypatch.setattr(dist, "all_gather_into_tensor", gather)
    run = qsa_cp_patch._with_replay(lambda: value(local, freqs, packed, output_format="ids"))
    # Production must use CPU metadata, including the RoPE helper's known max length.
    with monkeypatch.context() as scoped:

        def scalar_read(*args, **kwargs):
            raise AssertionError("selector performed an accelerator scalar read")

        scoped.setattr(torch.Tensor, "item", scalar_read)
        scoped.setattr(torch.Tensor, "tolist", scalar_read)
        actual = run()
        counts = value.index_qk_proj.calls, len(score_rows), len(gathers)
        replayed = run()
    assert actual.selected_ids is replayed.selected_ids
    assert counts == (value.index_qk_proj.calls, len(score_rows), len(gathers))
    # The static replay stage delegates to the TP selector through qsa_dispatch.
    assert qsa_cp_patch.qsa_dispatch.tp_forward.__module__ == "relax.models.qwen3_8.qsa_dispatch"
    got = expanded(actual)
    if want.shape[0]:
        assert torch.equal(got, want), (lengths, (got != want).sum().item())
    else:
        # Image all-short sentinel means dense causal attention; verify the native
        # materialization independently, including empty and sub-block documents.
        for start, end in zip(packed.qsa_select_offsets_cpu, packed.qsa_select_offsets_cpu[1:]):
            positions = torch.arange(end - start, device="cuda")
            ids = torch.arange(budget, device="cuda")
            expected = torch.where(ids[None] < (positions[:, None] + 1) // 4, ids[None], -1).int()
            assert torch.equal(actual.selected_ids[start:end], expected)
    rank = dist.get_rank(group=groups.tp)
    first, last = rank * (hidden.shape[0] // 2), (rank + 1) * (hidden.shape[0] // 2)
    expected_rows = sum(
        max(0, min(b, last) - max(a, first))
        for a, b in zip(packed.qsa_select_offsets_cpu, packed.qsa_select_offsets_cpu[1:])
        if b - a > budget * 4 + 3
    )
    assert sum(score_rows) == expected_rows
    assert counts[0] == int(max(lengths) > budget * 4 + 3)
    assert all(group is groups.tp for group in gathers)
    report(
        {
            "lengths": lengths,
            "block_budget": budget,
            "ties": ties,
            "routes_match_image": True,
            "score_rows_rank0": sum(score_rows),
            "projection_calls": counts[0],
            "replay_reused": True,
        }
    )


def test_selector_norm_scores_loss_and_gradients(groups, image, monkeypatch):
    from relax.models.qwen3_8.ops.qsa_mixed_ops import attention_for_selection, selector

    torch.manual_seed(17)
    x = torch.randn(17, 4, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(selector.gemma_rmsnorm(x, weight, 1e-6), image.norm(x, weight, 1e-6))
    keys = torch.randn(513, 128, device="cuda", dtype=torch.bfloat16)
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    assert torch.equal(selector.index_scores(x, keys), image.module.index_scores(x, keys))
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == previous
    value = indexer(groups, 8)
    hidden, freqs, packed = inputs((3, 36, 5))
    routes = reference(image, value, hidden, freqs, packed, monkeypatch)
    local = hidden.chunk(2)[dist.get_rank(group=groups.tp)].unsqueeze(1).contiguous()
    selection = value(local, freqs, packed, output_format="ids")
    assert torch.equal(expanded(selection), routes)
    from qsa_mixed_tp2 import attention as source_attention

    tensors = [torch.randn(44, heads, 256, device="cuda", dtype=torch.bfloat16) for heads in (12, 1, 1)]
    upstream = torch.randn_like(tensors[0])
    left = [x.detach().requires_grad_() for x in tensors]
    right = [x.detach().clone().requires_grad_() for x in tensors]
    got = attention_for_selection(*(x.unsqueeze(0) for x in left), selection, 256**-0.5, True)[0]
    want = source_attention(*right, routes, 256**-0.5)
    actual_grads = torch.autograd.grad(got, left, upstream)
    source_grads = torch.autograd.grad(want, right, upstream)
    errors = {}
    for name, actual, expected in zip(("output", "dq", "dk", "dv"), (got, *actual_grads), (want, *source_grads)):
        relative = ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item()
        errors[name] = relative
        assert relative < 0.002, (name, relative)
    loss_error = ((got.float() * upstream.float()).sum() - (want.float() * upstream.float()).sum()).abs().item()
    report({"norm_scores_match_image": True, "relative_l2": errors, "weighted_loss_absolute_difference": loss_error})


def test_selector_rejects_stale_layout_and_unsupported_mode(groups):
    from relax.models.qwen3_8.ops.qsa_mixed_ops.selector import document_bounds

    hidden, freqs, packed = inputs((4, 4))
    packed.cu_seqlens_q = packed.cu_seqlens_q.clone()
    with pytest.raises(ValueError, match="stale"):
        document_bounds(packed, hidden.shape[0])
    value = indexer(groups, 8)
    local = hidden.chunk(2)[dist.get_rank(group=groups.tp)].unsqueeze(1).contiguous()
    with pytest.raises(NotImplementedError, match="Stage-1"):
        value(local, freqs, packed, output_format="bits")
    value.config.qsa_indexer_loss_coeff = 1
    with pytest.raises(NotImplementedError, match="Stage-1"):
        value(local, freqs, packed, output_format="ids")
