# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from tests.backends.megatron.test_qsa_block_mask_patch import _selection


def _reference(query, key, value, dense_mask, scale, query_ids=None):
    """Dense masked SDPA in FP32 over the same admitted token pairs."""
    heads, kv_heads = query.shape[1], key.shape[1]
    q = query.float().transpose(0, 1).unsqueeze(0)
    k = key.float().transpose(0, 1).unsqueeze(0).repeat_interleave(heads // kv_heads, dim=1)
    v = value.float().transpose(0, 1).unsqueeze(0).repeat_interleave(heads // kv_heads, dim=1)
    mask = dense_mask if query_ids is None else dense_mask.index_select(0, query_ids.long())
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask[None, None], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.nan_to_num(probabilities, nan=0.0)
    return torch.matmul(probabilities, v).squeeze(0).transpose(0, 1)


def _run(
    lengths,
    pattern,
    heads,
    kv_heads,
    dim,
    query_ids=None,
    dtype=torch.bfloat16,
    backend="triton",
    selection=None,
    seed=7,
):
    from megatron.core.transformer.experimental_attention_variant import qsa

    if backend == "triton_v3":
        from relax.models.qwen3_8.ops import qsa_triton_v3 as fused
    elif backend == "triton_v2":
        from relax.models.qwen3_8.ops import qsa_triton_v2 as fused
    else:
        from relax.models.qwen3_8.ops import qsa_fused_attention as fused

    sequence_length = sum(lengths)
    selection = _selection(lengths, pattern) if selection is None else selection
    dense_mask = qsa.build_qsa_dense_mask(selection, sequence_length)[0]
    scale = dim**-0.5
    torch.manual_seed(seed)
    q_len = sequence_length if query_ids is None else query_ids.numel()
    inputs = [
        torch.randn(q_len, heads, dim, device="cuda", dtype=dtype),
        torch.randn(sequence_length, kv_heads, dim, device="cuda", dtype=dtype),
        torch.randn(sequence_length, kv_heads, dim, device="cuda", dtype=dtype),
    ]
    upstream = torch.randn(q_len, heads, dim, device="cuda", dtype=dtype)

    actual_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    plan = fused.build_plan(selection, sequence_length, query_ids=query_ids)
    actual = fused.qsa_fused_attention(*actual_inputs, selection, plan, query_ids, scale)
    actual.backward(upstream)

    # Keep reference leaves in FP32 too: BF16 leaves would round the oracle gradients.
    expected_inputs = [x.detach().float().requires_grad_() for x in inputs]
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        expected = _reference(*expected_inputs, dense_mask, scale, query_ids)
        expected.backward(upstream.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return (
        [actual, *[x.grad for x in actual_inputs]],
        [expected, *[x.grad for x in expected_inputs]],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The fused QSA kernel requires CUDA")
@pytest.mark.parametrize(
    "lengths,pattern",
    [
        ((256,), "sparse"),
        ((512,), "sparse"),
        ((7, 129, 4, 116), "sparse"),
        ((12, 252, 248), "sparse"),
        ((512,), "all"),
        ((4, 124, 128, 256), "sparse"),
        ((129,), "sparse"),
        ((7, 129, 4, 117), "sparse"),
        ((513,), "all"),
        ((517, 263), "sparse"),
    ],
)
@pytest.mark.parametrize("heads,kv_heads,dim", [(3, 1, 256), (4, 2, 128), (2, 2, 64)])
@pytest.mark.parametrize("backend", ["triton", "triton_v2"])
def test_fused_attention_matches_dense_masked_reference(lengths, pattern, heads, kv_heads, dim, backend):
    actual, expected = _run(lengths, pattern, heads, kv_heads, dim, backend=backend)
    names = ["output", "dq", "dk", "dv"]
    for name, got, want in zip(names, actual, expected):
        torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2e-2, msg=lambda m: f"{name}: {m}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The fused QSA kernel requires CUDA")
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("backend", ["triton", "triton_v2"])
def test_fused_attention_supports_packed_cp_local_queries(rank, backend):
    from megatron.core.ssm.mamba_context_parallel import _thd_partitioned_indices

    lengths = (12, 252, 248)
    sequence_length = sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    query_ids = _thd_partitioned_indices(cu, sequence_length, 2, rank)
    actual, expected = _run(lengths, "sparse", 3, 1, 256, query_ids=query_ids, backend=backend)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The fused QSA kernel requires CUDA")
@pytest.mark.parametrize("backend", ["triton", "triton_v2"])
def test_fused_attention_backward_is_deterministic(backend):
    actual_first, _ = _run((4, 124, 128, 256), "sparse", 3, 1, 256, backend=backend)
    actual_second, _ = _run((4, 124, 128, 256), "sparse", 3, 1, 256, backend=backend)
    for first, second in zip(actual_first, actual_second):
        assert torch.equal(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The fused QSA kernel requires CUDA")
@pytest.mark.parametrize("rank", [0, 1])
def test_fused_attention_head_dim_256_long_packed_cp_matches_reference(rank):
    from megatron.core.ssm.mamba_context_parallel import _thd_partitioned_indices
    from megatron.core.transformer.experimental_attention_variant import qsa

    # Qwen3.8 TP=4: six local query heads, one replicated KV head. Two
    # documents with unaligned 128-token boundaries exercise the classifier
    # when each block is processed as multiple smaller attention tiles.
    lengths = (4092, 4100)
    sequence_length = sum(lengths)
    positions = torch.cat([torch.arange(n, device="cuda") for n in lengths]).to(torch.int32)
    doc_ids = torch.repeat_interleave(torch.arange(2, device="cuda"), torch.tensor(lengths, device="cuda"))
    bytes_per_row = (((max(lengths) + 3) // 4 + 1) + 7) // 8
    rows = torch.arange(sequence_length, device="cuda")[:, None]
    byte_offsets = torch.arange(bytes_per_row, device="cuda")[None, :] * 8
    bits = torch.zeros((sequence_length, bytes_per_row), device="cuda", dtype=torch.uint8)
    for bit in range(8):
        block = byte_offsets + bit
        selected = (block < (positions[:, None] + 1) // 4) & ((block * 7 + rows * 3) % 5 < 2)
        bits |= selected.to(torch.uint8) << bit
    selection = qsa.QSASelection(
        doc_ids=doc_ids.to(torch.int32).view(1, -1),
        positions=positions.view(1, -1),
        selected_bits=bits.flatten(),
        bits_per_row=bytes_per_row,
        bits_per_row_t=torch.tensor(bytes_per_row, device="cuda"),
        compress_ratio=4,
        all_selected=False,
    )
    cu = torch.tensor([0, lengths[0], sequence_length], device="cuda", dtype=torch.int32)
    query_ids = _thd_partitioned_indices(cu, sequence_length, 2, rank)
    actual, expected = _run(lengths, "sparse", 6, 1, 256, query_ids=query_ids, selection=selection)
    for name, got, want in zip(("output", "dq", "dk", "dv"), actual, expected):
        torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2e-2, msg=lambda m: f"{name}: {m}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The fused QSA kernel requires CUDA")
@pytest.mark.parametrize("backend", ["triton", "triton_v2"])
def test_fused_attention_constant_keys_do_not_create_query_gradient(backend):
    """A query-independent softmax must not acquire a large rounding
    gradient."""
    if backend == "triton":
        from relax.models.qwen3_8.ops import qsa_fused_attention as fused
    else:
        from relax.models.qwen3_8.ops import qsa_triton_v2 as fused

    torch.manual_seed(23)
    length, heads, dim = 129, 3, 256
    selection = _selection((length,), "all")
    query = (torch.randn(length, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
    key = torch.full((length, 1, dim), 0.125, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = (1.3 + torch.randn_like(key) * 0.1).requires_grad_()
    upstream = torch.randn_like(query) * 0.1
    plan = fused.build_plan(selection, length)
    output = fused.qsa_fused_attention(query, key, value, selection, plan, None, dim**-0.5)
    dq = torch.autograd.grad(output, query, upstream)[0]
    # With every key identical, changing Q adds the same logit to all admitted keys.
    torch.testing.assert_close(dq.float(), torch.zeros_like(dq.float()), atol=2e-5, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The fused QSA kernel requires CUDA")
@pytest.mark.parametrize("backend", ["triton", "triton_v2"])
def test_fused_attention_recompute_preserves_forward_and_saves_residual_only_for_backward(backend):
    """Checkpoint forward and recomputed forward must return the same
    values."""
    if backend == "triton":
        from relax.models.qwen3_8.ops import qsa_fused_attention as fused
    else:
        from relax.models.qwen3_8.ops import qsa_triton_v2 as fused

    torch.manual_seed(23)
    selection = _selection((7, 129, 4, 117), "sparse")
    q = torch.randn(257, 3, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(257, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    plan = fused.build_plan(selection, 257)
    with torch.no_grad():
        initial = fused.qsa_fused_attention(q, k, v, selection, plan, None, 256**-0.5)
    recomputed = fused.qsa_fused_attention(q, k, v, selection, plan, None, 256**-0.5)
    assert torch.equal(initial, recomputed)
    residual = recomputed.grad_fn.saved_tensors[-2]
    assert residual.shape == q.shape and residual.dtype == q.dtype
    assert residual.requires_grad is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The shared dS kernel requires CUDA")
@pytest.mark.parametrize("capacity_bytes", [0, 64 * 2**20])
def test_v3_shared_ds_fit_and_overflow_match_original_backward(monkeypatch, capacity_bytes):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Shared dS is enabled only on SM103")
    from relax.models.qwen3_8.ops import qsa_triton_v3 as fused
    from relax.models.qwen3_8.ops.qsa_triton_v3 import shared_backward

    with monkeypatch.context() as control:
        control.setattr(fused, "_can_share_ds", lambda *args: False)
        reference, _ = _run((4, 124, 128, 257), "sparse", 3, 1, 256, backend="triton_v3")
    allocate = shared_backward.allocate_workspace
    monkeypatch.setattr(
        shared_backward,
        "allocate_workspace",
        lambda q, k, plan, budget: allocate(q, k, plan, capacity_bytes),
    )
    actual, _ = _run((4, 124, 128, 257), "sparse", 3, 1, 256, backend="triton_v3")
    for left, right in zip(actual, reference, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
