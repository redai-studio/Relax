# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch
import torch.nn.functional as F

from examples.seq_contrast_sft.tools.retrieve import search


def test_exact_top_k_crosses_candidate_blocks():
    queries = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), dim=-1)
    candidates = F.normalize(torch.tensor([[0.8, 0.2], [0.3, -1.0], [0.2, 0.6], [-0.7, 0.3], [-0.2, -0.3]]), dim=-1)
    values, expected = (queries @ candidates.T).topk(3, dim=1)
    actual, scores = search(queries, candidates, 3, torch.device("cpu"), query_batch=2, candidate_batch=2)
    assert actual == expected.tolist()
    torch.testing.assert_close(torch.tensor(scores), values)
