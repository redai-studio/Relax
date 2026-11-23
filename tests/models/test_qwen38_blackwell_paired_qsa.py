# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise QSA's compressed-block addressing, causal tail and KV gradient
accumulation."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The QSA kernels execute on CUDA")
@pytest.mark.parametrize("paired", [False, True], ids=["direct", "paired"])
def test_suffix_attention_matches_each_rows_selected_tokens(paired):
    from relax.models.qwen3_8.ops.blackwell_sft.qsa_direct import DirectGrouped
    from relax.models.qwen3_8.ops.blackwell_sft.qsa_paired import Paired

    torch.manual_seed(1729)
    # The first pair shares two blocks; the second pair crosses documents; the last query has no partner.
    documents = (0,) * 3 + (16,) * 2
    rows = [12, 13, 15, 24, 26]
    blocks = [[0, 1], [1, 0], [1, 3], [0], [1, 0]]
    device = torch.device("cuda")
    positions = torch.arange(16, device=device, dtype=torch.int32).repeat(2)
    ids = torch.full((32, 512), -1, dtype=torch.int32, device=device)
    for row, selected in zip(rows, blocks, strict=True):
        ids[row, : len(selected)] = torch.tensor(selected, device=device)
    q, k, v = [
        torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        for shape in ((5, 12, 256), (32, 1, 256), (32, 1, 256))
    ]
    scale = q.shape[-1] ** -0.5
    kernel = Paired if paired else DirectGrouped
    actual = kernel.apply(q, k, v, ids, positions, torch.tensor(rows, device=device, dtype=torch.int32), scale)
    query, keys, values = [t.detach().float().requires_grad_() for t in (q, k, v)]
    outputs = []
    for i, (row, start, selected) in enumerate(zip(rows, documents, blocks, strict=True)):
        indices = [start + 4 * block + j for block in selected for j in range(4)]
        tail = (row - start + 1) % 4
        indices.extend(range(row + 1 - tail, row + 1))
        scores = query[i] @ keys[indices, 0].T * scale
        outputs.append(scores.softmax(-1) @ values[indices, 0])
    reference = torch.stack(outputs)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    reference.backward(grad.float())
    torch.testing.assert_close(actual.float(), reference, rtol=0.02, atol=0.02)
    for tensor, expected in zip((q, k, v), (query, keys, values), strict=True):
        torch.testing.assert_close(tensor.grad.float(), expected.grad, rtol=0.03, atol=0.04)
