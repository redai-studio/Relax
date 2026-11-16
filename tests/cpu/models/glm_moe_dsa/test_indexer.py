# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch


def test_glm_short_packed_sequences_preserve_causal_topk_and_invalid_padding(glm_modules):
    module = glm_modules("ops.indexer")
    starts, ends = module.generate_varlen_mask_params(torch.tensor([0, 2, 5]), seq_len=5)
    torch.testing.assert_close(starts, torch.tensor([0, 0, 2, 2, 2]))
    torch.testing.assert_close(ends, torch.tensor([1, 2, 3, 4, 5]))
    keys = torch.arange(5)
    mask = (keys[None, :] >= starts[:, None]) & (keys[None, :] < ends[:, None])
    logits = keys.float().expand(5, -1).masked_fill(~mask, -torch.inf)
    scores, indices = module.pytorch_topk_with_invalid_padding(logits, 8)
    assert indices.shape == scores.shape == (5, 8)
    assert indices.dtype == torch.int32
    for row in range(5):
        valid = indices[row][indices[row] >= 0]
        assert set(valid.tolist()) == set(range(int(starts[row]), int(ends[row])))
    assert torch.all(indices[:, 5:] == -1)
    assert torch.all(scores[indices == -1] == -torch.inf)
    torch.testing.assert_close(module.pytorch_extract_topk_scores(logits, indices), scores)


def test_glm_topk_respects_configured_width(glm_modules):
    module = glm_modules("ops.indexer")
    scores, indices = module.pytorch_topk_with_invalid_padding(torch.tensor([[1.0, 4.0, 3.0]]), 2)
    torch.testing.assert_close(scores, torch.tensor([[4.0, 3.0]]))
    torch.testing.assert_close(indices, torch.tensor([[1, 2]], dtype=torch.int32))
