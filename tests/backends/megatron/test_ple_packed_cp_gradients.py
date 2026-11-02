# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU contracts for the shipped PLE forward and its packed CP/SP gradient
ownership.

Replicas share gathered inputs and parameters in one autograd graph, so
backward sums their contributions exactly like the input gathers and parameter
reductions. This tests gradient ownership, not a distributed transport
implementation.
"""

import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


PATCH = Path(__file__).resolve().parents[3] / "docker/patch/megatron/20261016-qwen38-models.patch"


def _added_source(relative: str) -> str:
    text = PATCH.read_text()
    section = text.split(f"diff --git a/{relative} b/{relative}\n", 1)[1].split("\ndiff --git ", 1)[0]
    return "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))


def _load_ple(parallel_state):
    namespace = {"torch": torch, "Tensor": torch.Tensor, "F": F, "math": math, "parallel_state": parallel_state}
    hyper = ast.parse(_added_source("megatron/core/transformer/hyper_connection.py"))
    norm = next(
        node
        for node in hyper.body
        if isinstance(node, ast.FunctionDef) and node.name == "gated_residual_group_rmsnorm"
    )
    exec(compile(ast.Module(body=[norm], type_ignores=[]), str(PATCH), "exec"), namespace)
    source = ast.parse(_added_source("megatron/core/transformer/per_layer_embedding.py"))
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "PerLayerEmbedding")
    names = {"forward", "_local_positions", "_causal_dilated_conv"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(PATCH), "exec"), namespace)
    return type("PatchedPLE", (torch.nn.Module,), {name: namespace[name] for name in names})


class _GatheringScatter(torch.autograd.Function):
    """Legacy scatter backward for the uniform upstream derivative used
    below."""

    @staticmethod
    def forward(ctx, value, rank, size):
        ctx.size = size
        return value.chunk(size, dim=0)[rank]

    @staticmethod
    def backward(ctx, gradient):
        # Each peer's output.sum() contributes ones, so the real all-gather
        # produces this exact repeated derivative. No network is needed.
        assert torch.equal(gradient, torch.ones_like(gradient))
        return gradient.repeat(ctx.size, 1, 1), None, None


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("frozen", [False, True])
def test_packed_cp_sp_preserves_input_and_parameter_gradients(monkeypatch, tp_size, frozen):
    state = SimpleNamespace(cp_size=1, cp_rank=0, tp_rank=0)
    parallel_state = SimpleNamespace(get_context_parallel_world_size=lambda: state.cp_size)
    ple_class = _load_ple(parallel_state)

    # Two packed documents, each split into 2*CP pieces in zigzag order.
    documents = (8, 24)
    cp_indices = []
    for rank in range(2):
        indices, start = [], 0
        for length in documents:
            chunks = torch.arange(start, start + length).chunk(4)
            indices.extend((chunks[rank], chunks[3 - rank]))
            start += length
        cp_indices.append(torch.cat(indices))

    def split(value, *_args, **_kwargs):
        return value if state.cp_size == 1 else value[cp_indices[state.cp_rank]]

    monkeypatch.setitem(
        sys.modules,
        "megatron.core.ssm.mamba_context_parallel",
        SimpleNamespace(
            reconstruct_tensor_cp=lambda value, *_args, **_kwargs: value,
            split_tensor_cp=split,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.tensor_parallel.mappings",
        SimpleNamespace(
            gather_from_sequence_parallel_region=lambda value, **_kwargs: value,
            scatter_to_sequence_parallel_region=lambda value, **_kwargs: _GatheringScatter.apply(
                value, state.tp_rank, tp_size
            ),
        ),
    )
    torch.manual_seed(67)
    model = ple_class()
    model.n, model.hidden_size, model.norm_eps = 2, 4, 1e-6
    model.config = SimpleNamespace(sequence_parallel=False)
    model.tp_group = SimpleNamespace(size=lambda: tp_size, rank=lambda: state.tp_rank)
    model._ngram_ids = torch.arange(32)
    model._cp_packed_seq_params = None
    model._position_in_sequence = torch.cat([torch.arange(length) for length in documents]).unsqueeze(0)
    model.ple_embedding = torch.nn.Embedding(32, 6)
    # Match the production [sequence, batch, channels] lookup output.
    model.ple_embedding.register_forward_hook(lambda _module, _inputs, output: output.unsqueeze(1))
    model.key_proj = torch.nn.Linear(6, 8, bias=False)
    model.value_proj = torch.nn.Linear(6, 4, bias=False)
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm = torch.nn.Module()
        norm.weight = torch.nn.Parameter(torch.randn(8) * 0.1)
        setattr(model, name, norm)
    model.conv_kernel_size, model.conv_dilation, model.halo = 3, 2, 4
    model.conv1d = torch.nn.Conv1d(8, 8, 3, groups=8, bias=False)
    model.requires_grad_(not frozen)
    hidden = torch.randn(32, 1, 8, requires_grad=True)
    expected = model(hidden)
    expected.sum().backward()
    expected_input_grad = hidden.grad.clone()
    expected_parameter_grads = {
        name: value.grad.clone() for name, value in model.named_parameters() if value.grad is not None
    }
    model.zero_grad(set_to_none=True)
    hidden.grad = None

    state.cp_size = 2
    model.config.sequence_parallel = True
    model._cp_packed_seq_params = object()
    losses = []
    for cp_rank in range(2):
        state.cp_rank = cp_rank
        for tp_rank in range(tp_size):
            state.tp_rank = tp_rank
            local = model(hidden)
            indices = cp_indices[cp_rank].chunk(tp_size)[tp_rank]
            torch.testing.assert_close(local, expected.detach()[indices])
            losses.append(local.sum())
    sum(losses).backward()
    torch.testing.assert_close(hidden.grad, expected_input_grad, rtol=2e-5, atol=2e-6)
    for name, parameter in model.named_parameters():
        if not frozen:
            torch.testing.assert_close(parameter.grad, expected_parameter_grads[name], rtol=2e-5, atol=2e-6, msg=name)
    assert frozen or len(expected_parameter_grads) == 7  # table + six replicated parameter tensors
