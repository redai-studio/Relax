# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye FlashInfer DSA selection requires CUDA")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [1, 73, 2048, 2049, 4096, 32768])
def test_dsa_topk_prefers_earlier_tokens_at_tied_boundary(dtype: torch.dtype, width: int) -> None:
    from relax.models.keye_vl2.topk import dsa_topk_indices

    # Include all-tied and mixed-score rows, with many ties across the 2048th boundary.
    positions = torch.arange(width, device="cuda")
    scores = torch.stack([positions % 7, torch.zeros_like(positions), positions.flip(0) % 11]).to(dtype)
    expected = scores.argsort(dim=-1, descending=True, stable=True)[:, : min(2048, width)].sort(-1).values
    actual = dsa_topk_indices(scores)
    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual.sort(-1).values, expected, atol=0, rtol=0)
    torch.testing.assert_close(dsa_topk_indices(scores), actual, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dsa_topk_prefix_and_padded_decode_select_same_keys(dtype: torch.dtype) -> None:
    from relax.models.keye_vl2.sparse_attention import compact_indices
    from relax.models.keye_vl2.topk import dsa_topk_indices

    width = 4096
    for length in (73, 2048, 2049, 3073):
        prefix = (torch.arange(length, device="cuda") % 3).to(dtype)[None]
        expected = prefix.argsort(dim=-1, descending=True, stable=True)[:, : min(2048, length)].sort(-1).values
        # Training uses the dtype's minimum; paged decode uses -inf for invalid keys.
        for padding in (torch.finfo(dtype).min, float("-inf")):
            padded = torch.full((1, width), padding, dtype=dtype, device="cuda")
            padded[:, :length] = prefix
            selected = dsa_topk_indices(padded)
            ordered, counts = compact_indices(selected, selected < length)
            torch.testing.assert_close(counts, torch.tensor([min(2048, length)], device="cuda", dtype=torch.int32))
            torch.testing.assert_close(ordered[:, : min(2048, length)].long(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dsa_topk_cuda_graph_replays_changed_scores_and_lengths(dtype: torch.dtype) -> None:
    from relax.models.keye_vl2.sparse_attention import compact_indices
    from relax.models.keye_vl2.topk import dsa_topk_indices

    width = 4096
    scores = torch.zeros((4, width), device="cuda", dtype=dtype)
    lengths = torch.full((4,), width, device="cuda", dtype=torch.int32)

    def select() -> tuple[torch.Tensor, torch.Tensor]:
        selected = dsa_topk_indices(scores)
        return compact_indices(selected, selected < lengths[:, None])

    for _ in range(3):
        select()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ordered, counts = select()
    positions = torch.arange(width, device="cuda")
    for iteration, new_lengths in enumerate(((73, 2048, 2051, 3073), (4096, 2100, 125, 2049))):
        lengths.copy_(torch.tensor(new_lengths, device="cuda", dtype=torch.int32))
        changed = ((positions[None] + torch.arange(4, device="cuda")[:, None]) % (3 + iteration)).to(dtype)
        changed.masked_fill_(positions[None] >= lengths[:, None], float("-inf"))
        scores.copy_(changed)
        graph.replay()
        for row, length in enumerate(new_lengths):
            count = min(2048, length)
            expected = changed[row, :length].argsort(descending=True, stable=True)[:count].sort().values
            torch.testing.assert_close(ordered[row, :count].long(), expected, atol=0, rtol=0)
        torch.testing.assert_close(counts, lengths.clamp_max(2048), atol=0, rtol=0)
