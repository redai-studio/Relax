# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import ast
from enum import Enum, auto
from types import ModuleType

import pytest

from tests.tools.sglang_patch_source import extract_functions, patched_source


torch = pytest.importorskip("torch")
target = "python/sglang/srt/layers/logprob_processor.py"
source = patched_source(target, include_per_position=True)
functions = extract_functions(
    source,
    target,
    {"get_token_ids_logprobs_raw", "get_token_ids_logprobs_chunk", "get_token_ids_logprobs_batch_optimized"},
)
stage = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "LogprobStage")
logprob = ModuleType("patched_logprob_processor")
logprob.__dict__.update(
    torch=torch,
    Enum=Enum,
    auto=auto,
    is_pin_memory_available=lambda device: False,
    async_d2h=lambda value: value.cpu(),
)
exec(compile(ast.Module(body=[stage, *functions], type_ignores=[]), target, "exec"), logprob.__dict__)


@pytest.mark.parametrize("top_k", [1, 3])
def test_per_position_raw_gather_keeps_mixed_request_row_offsets(top_k):
    scores = torch.arange(70, dtype=torch.float32).reshape(7, 10)
    queries = [[(row + col) % 10 for col in range(top_k)] for row in range(3)]
    values, ids = logprob.get_token_ids_logprobs_raw(
        scores, [None, queries, [2, 8]], logprob.LogprobStage.PREFILL, [2, 3, 2]
    )
    assert values[0] == []
    assert ids[1] == queries
    torch.testing.assert_close(torch.tensor(values[1]), scores[2:5].gather(1, torch.tensor(queries)))
    torch.testing.assert_close(torch.tensor(values[2]), scores[5:, [2, 8]])


def test_per_position_output_uses_final_dummy_query_row():
    scores = torch.arange(30, dtype=torch.float32).reshape(3, 10)
    values, ids = logprob.get_token_ids_logprobs_raw(scores, [[[3], [0]], [2, 8], None], logprob.LogprobStage.DECODE)
    assert ids == [[0], [2, 8], []]
    assert values == [[0.0], [12.0, 18.0], []]
    packed_values, packed_ids = logprob.get_token_ids_logprobs_batch_optimized(scores, [[[3], [0]], [2, 8], None])
    assert packed_ids == ids
    assert [row.tolist() for row in packed_values] == values


@pytest.mark.parametrize("top_k", [1, 2])
def test_per_position_logits_chunks_resume_with_mixed_queries(top_k):
    scores = torch.arange(110, dtype=torch.float32).reshape(11, 10)
    query_a = [[(row + col) % 10 for col in range(top_k)] for row in range(4)]
    query_d = [[row + 1] for row in range(4)]
    values, ids = [], []
    split = logprob.get_token_ids_logprobs_chunk(scores[:2], [query_a], [4], values, ids)
    assert split == 2
    split = logprob.get_token_ids_logprobs_chunk(
        scores[2:9], [query_a, None, [2, 8], query_d], [4, 1, 2, 4], values, ids, split
    )
    assert split == 2
    split = logprob.get_token_ids_logprobs_chunk(scores[9:], [query_d], [4], values, ids, split)
    assert split == 0
    assert ids == [query_a, [], [[2, 8], [2, 8]], query_d]
    torch.testing.assert_close(torch.tensor(values[0]), scores[:4].gather(1, torch.tensor(query_a)))
    assert values[1] == []
    torch.testing.assert_close(torch.tensor(values[2]), scores[5:7, [2, 8]])
    torch.testing.assert_close(torch.tensor(values[3]), scores[7:].gather(1, torch.tensor(query_d)))


def test_per_position_zero_row_request_does_not_consume_scores():
    scores = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    values, ids = [], []
    split = logprob.get_token_ids_logprobs_chunk(scores, [[], [[0], [1]]], [0, 2], values, ids)
    assert split == 0
    assert values == [[], [[0.0], [11.0]]]
    assert ids == [[], [[0], [1]]]


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("copy_to_pinned_cpu", [False, True])
def test_per_position_chunks_preserve_normalization_and_async_copy(monkeypatch, normalize, copy_to_pinned_cpu):
    scores = torch.arange(50, dtype=torch.float32).reshape(5, 10) / 10
    queries = [[1, 8], [2, 6], [0, 9]]
    row_max = scores.max(dim=1).values
    normalizer = torch.logsumexp(scores - row_max[:, None], dim=1)
    copies = []

    def copy_to_cpu(values):
        copies.append(values.clone())
        return copies[-1]

    monkeypatch.setattr(logprob, "async_d2h", copy_to_cpu)
    values, ids = [], []
    split = logprob.get_token_ids_logprobs_chunk(
        scores[:2],
        [queries],
        [3],
        values,
        ids,
        log_normalizer=(row_max[:2], normalizer[:2]) if normalize else None,
        copy_to_pinned_cpu=copy_to_pinned_cpu,
    )
    assert split == 2
    split = logprob.get_token_ids_logprobs_chunk(
        scores[2:],
        [queries, [2, 8]],
        [3, 2],
        values,
        ids,
        split,
        log_normalizer=(row_max[2:], normalizer[2:]) if normalize else None,
        copy_to_pinned_cpu=copy_to_pinned_cpu,
    )
    expected = scores.log_softmax(dim=1) if normalize else scores
    actual = [torch.cat(rows) if copy_to_pinned_cpu else torch.tensor(rows) for rows in values]
    assert split == 0 and ids == [queries, [[2, 8], [2, 8]]]
    torch.testing.assert_close(actual[0], expected[:3].gather(1, torch.tensor(queries)))
    torch.testing.assert_close(actual[1], expected[3:, [2, 8]])
    assert len(copies) == (3 if copy_to_pinned_cpu else 0)
