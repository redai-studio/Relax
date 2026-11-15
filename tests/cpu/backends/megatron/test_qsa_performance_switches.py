# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU control-flow tests; kernel arithmetic requires the separate GPU
gates."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("bf16_products,override", [(True, False), (True, True), (False, True)])
def test_backward_launch_override_preserves_forward(monkeypatch, bf16_products, override):
    from relax.models.qwen3_8.ops.qsa_mixed_ops import grouped

    calls = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def invoke(*args, **kwargs):
                calls.append((self.name, grid, kwargs))
                for output in args[4:6] if self.name == "forward" else args[7:10]:
                    output.zero_()

            return invoke

    monkeypatch.setattr(grouped, "_forward", Kernel("forward"))
    monkeypatch.setattr(grouped, "_backward", Kernel("backward"))
    for key, value in (("BLOCK_K", "32"), ("WARPS", "4"), ("STAGES", "3")):
        key = "RELAX_QWEN38_QSA_GROUPED_BACKWARD_" + key
        if override:
            monkeypatch.setenv(key, value)
        else:
            monkeypatch.delenv(key, raising=False)
    q = torch.zeros(7, 12, 256, dtype=torch.bfloat16)
    kv = torch.zeros(7, 1, 256, dtype=torch.bfloat16)
    indices = torch.full((7, 2052), -1, dtype=torch.int32)
    ctx = SimpleNamespace(needs_input_grad=(True, True, True, False, False, False, False, False))
    ctx.save_for_backward = lambda *values: setattr(ctx, "saved_tensors", values)
    out = grouped._Grouped.forward(ctx, q, kv, kv, indices, 0.0625, bf16_products, 64, None)
    grads = grouped._Grouped.backward(ctx, torch.ones_like(out))
    forward, backward = calls
    assert forward[0] == "forward" and backward[0] == "backward"
    assert (forward[2]["BK"], forward[2]["num_warps"], forward[2]["num_stages"]) == (64, 4, 1 if bf16_products else 3)
    assert (backward[2]["BK"], backward[2]["num_warps"], backward[2]["num_stages"]) == (
        32 if override and bf16_products else 64,
        4,
        3 if override or not bf16_products else 1,
    )
    assert [value.shape for value in grads[:3]] == [q.shape, kv.shape, kv.shape]


class _PretendCudaTensor(torch.Tensor):
    """Allow CPU-only exercise of forward's control flow, never kernel
    numerics."""

    @property
    def is_cuda(self):
        return True


@pytest.fixture
def selector_inputs(monkeypatch):
    from megatron.core.models.common.embeddings import rope_utils

    from relax.models.qwen3_8.ops.qsa_mixed_ops import selector

    monkeypatch.setattr(selector.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(selector.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(rope_utils, "_apply_rotary_pos_emb_bshd", lambda tensor, *_args, **_kwargs: tensor)
    config = SimpleNamespace(
        context_parallel_size=1,
        qsa_indexer_loss_coeff=0,
        layernorm_zero_centered_gamma=True,
        sequence_parallel=False,
        rotary_interleaved=False,
        mrope_section=None,
        layernorm_epsilon=1e-6,
    )
    model = SimpleNamespace(
        config=config,
        parameters=lambda: (),
        n_heads=4,
        kv_heads=1,
        head_dim=128,
        compress_ratio=4,
        token_budget=4,
        block_topk=1,
        pg_collection=SimpleNamespace(tp=object(), cp=object()),
        index_qk_proj=lambda hidden: (torch.ones(hidden.shape[0], 640, dtype=torch.bfloat16), None),
        q_layernorm=SimpleNamespace(weight=torch.zeros(128)),
        k_layernorm=SimpleNamespace(weight=torch.zeros(128)),
    )
    hidden = torch.zeros(8, 1, 4, dtype=torch.bfloat16).as_subclass(_PretendCudaTensor)
    freqs = torch.zeros(8, 1, 1, 128)
    return selector, model, hidden, freqs


@pytest.mark.parametrize("enabled", [None, "0", "1"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("initial_flag", [False, True])
def test_selector_compile_branch_restores_global_reduction(monkeypatch, selector_inputs, enabled, fail, initial_flag):
    selector, model, hidden, freqs = selector_inputs
    if enabled is None:
        monkeypatch.delenv("RELAX_QWEN38_QSA_SELECTOR_COMPILE", raising=False)
    else:
        monkeypatch.setenv("RELAX_QWEN38_QSA_SELECTOR_COMPILE", enabled)
    seen = []

    def compiled(query, keys, positions, topk):
        seen.append("compiled")
        assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        if fail:
            raise RuntimeError("injected compile failure")
        return torch.full((query.shape[0], topk), -1, dtype=torch.int32)

    def eager(query, keys):
        seen.append("eager")
        if fail:
            raise RuntimeError("injected eager failure")
        return torch.zeros(query.shape[0], keys.shape[0], 4)

    monkeypatch.setattr(selector, "compiled_score_topk", compiled)
    monkeypatch.setattr(selector, "index_scores", eager)
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = initial_flag
        if fail:
            with pytest.raises(RuntimeError, match="injected"):
                selector.forward(model, hidden, freqs, output_format="ids")
        else:
            selector.forward(model, hidden, freqs, output_format="ids")
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == initial_flag
        assert seen == ["compiled" if enabled == "1" else "eager"]
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous


@pytest.mark.parametrize("attribute,value", [("head_dim", 64), ("compress_ratio", 2), ("n_heads", 8)])
def test_selector_rejects_geometry_before_compiled_kernel(monkeypatch, selector_inputs, attribute, value):
    selector, model, hidden, freqs = selector_inputs
    monkeypatch.setenv("RELAX_QWEN38_QSA_SELECTOR_COMPILE", "1")
    setattr(model, attribute, value)

    def forbidden(*_args):
        pytest.fail("Unsupported geometry reached compiled scoring")

    monkeypatch.setattr(selector, "compiled_score_topk", forbidden)
    with pytest.raises(NotImplementedError, match="4Q/1KV/D128"):
        selector.forward(model, hidden, freqs, output_format="ids")
