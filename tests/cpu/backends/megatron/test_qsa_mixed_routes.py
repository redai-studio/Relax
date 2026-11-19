# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU route, packed-causality and gradient contracts retained by mixed QSA."""

from types import SimpleNamespace

import pytest
import torch

from relax.models.qwen3_8.ops.qsa_mixed_ops.routes import token_routes


def selection_for(lengths, batch, dtype=torch.int32):
    rows = [lengths if row % 2 == 0 else tuple(reversed(lengths)) for row in range(batch)]
    positions = torch.tensor([[pos for length in row for pos in range(length)] for row in rows], dtype=torch.int32)
    docs = torch.tensor(
        [[doc for doc, length in enumerate(row) for _ in range(length)] for row in rows], dtype=torch.int32
    )
    visible = (positions + 1) // 4
    ids = torch.stack((torch.where(visible > 0, 0, -1), torch.where(visible > 1, visible - 1, -1)), dim=-1).to(dtype)
    return SimpleNamespace(positions=positions, doc_ids=docs, selected_ids=ids.flatten(0, 1), compress_ratio=4)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("lengths", [(1, 2, 3), (7, 9, 4, 1), (2052, 7), (0, 4, 0, 1)])
def test_mixed_routes_preserve_document_local_blocks_and_causal_tails(lengths, batch, dtype):
    selection = selection_for(lengths, batch, dtype)
    length = sum(lengths)
    routes = token_routes(selection, batch, length)
    assert routes.shape == (batch, length, 11)
    assert routes.is_contiguous() and routes.dtype == dtype
    ids = selection.selected_ids.view(batch, length, -1)
    for row in range(batch):
        for query in range(length):
            position = int(selection.positions[row, query])
            start = query - position
            expected = []
            for block in ids[row, query].tolist():
                if block >= 0:
                    expected.extend(range(start + block * 4, start + block * 4 + 4))
            expected.extend(range(start + ((position + 1) // 4) * 4, query + 1))
            actual = routes[row, query]
            actual = actual[actual >= 0].tolist()
            assert actual == expected
            assert all(start <= key <= query for key in actual)


def test_mixed_route_attention_and_qkv_gradients_match_independent_mask():
    selection = selection_for((7, 4, 2, 9), 1)
    positions = selection.positions[0]
    docs = selection.doc_ids[0]
    length = positions.numel()
    ids = selection.selected_ids
    selected = ((ids[:, :, None] >= 0) & (ids[:, :, None] == positions[None, None, :] // 4)).any(dim=1)
    expected_mask = (
        (docs[:, None] == docs[None, :])
        & (positions[None, :] <= positions[:, None])
        & (selected | (positions[None, :] >= ((positions[:, None] + 1) // 4) * 4))
    )
    actual_mask = torch.zeros_like(expected_mask)
    for query, route in enumerate(token_routes(selection, 1, length)[0]):
        actual_mask[query, route[route >= 0].long()] = True
    torch.testing.assert_close(actual_mask, expected_mask)

    generator = torch.Generator().manual_seed(42)
    q, k, v = [torch.randn(length, 3, dtype=torch.float64, generator=generator, requires_grad=True) for _ in range(3)]
    scores = q @ k.T
    actual = scores.masked_fill(~actual_mask, -torch.inf).softmax(-1) @ v
    expected = scores.masked_fill(~expected_mask, -torch.inf).softmax(-1) @ v
    upstream = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
    actual_grads = torch.autograd.grad(actual, (q, k, v), upstream, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (q, k, v), upstream)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want)
