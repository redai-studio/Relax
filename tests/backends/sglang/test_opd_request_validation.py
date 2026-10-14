# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks against the SGLang patch installed in the training image."""

from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from relax.utils.opd.opd_main_worker import LogprobResponse


torch = pytest.importorskip("torch")
tokenizer_manager = pytest.importorskip("sglang.srt.managers.tokenizer_manager")
schedule_batch = pytest.importorskip("sglang.srt.managers.schedule_batch")


@pytest.fixture
def tokenized_request():
    request = tokenizer_manager.GenerateReqInput(
        rid="opd-test",
        input_ids=[1, 99, 2, 10, 11],
        sampling_params={"max_new_tokens": 0, "custom_params": {"relax_opd_token_ids_logprob_k": 2}},
        return_logprob=True,
        logprob_start_len=4,
        token_ids_logprob=[0, 5, 7, 9, 0, 0],
    )
    manager = SimpleNamespace(
        preferred_sampling_params=None,
        sampling_params_class=schedule_batch.SamplingParams,
        tokenizer=None,
        model_config=SimpleNamespace(vocab_size=128),
        server_args=SimpleNamespace(disaggregation_transfer_backend="none"),
        rid_to_state={request.rid: SimpleNamespace(time_stats=Mock())},
    )

    def create(input_ids=None):
        return tokenizer_manager.TokenizerManager._create_tokenized_object(manager, request, None, input_ids)

    return request, create


@pytest.mark.parametrize("top_k", [1, 2])
def test_opd_query_validation_uses_expanded_prompt_length(tokenized_request, top_k):
    request, create = tokenized_request
    request.sampling_params["custom_params"]["relax_opd_token_ids_logprob_k"] = top_k
    request.token_ids_logprob = [0, 127] * top_k + [0] * top_k
    tokenized = create([1, 99, 99, 99, 2, 10, 11])
    assert list(tokenized.input_ids) == [1, 99, 99, 99, 2, 10, 11]
    assert tokenized.token_ids_logprob == request.token_ids_logprob
    assert tokenized.sampling_params.custom_params["relax_opd_token_ids_logprob_k"] == top_k


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "2"])
def test_opd_query_validation_rejects_invalid_width(tokenized_request, top_k):
    request, create = tokenized_request
    request.sampling_params["custom_params"]["relax_opd_token_ids_logprob_k"] = top_k
    with pytest.raises(ValueError, match="positive integer"):
        create([1, 99, 99, 99, 2, 10, 11])


@pytest.mark.parametrize(
    "field,value",
    [("return_logprob", False), ("logprob_start_len", None), ("logprob_start_len", -1), ("logprob_start_len", 7)],
)
def test_opd_query_validation_requires_input_logprobs(tokenized_request, field, value):
    request, create = tokenized_request
    setattr(request, field, value)
    with pytest.raises(ValueError, match="prefill-only input logprobs"):
        create([1, 99, 99, 99, 2, 10, 11])


@pytest.mark.parametrize("missing_ids", [False, True])
def test_opd_query_validation_requires_prefill_token_ids(tokenized_request, missing_ids):
    request, create = tokenized_request
    if not missing_ids:
        request.sampling_params["max_new_tokens"] = 1
    with pytest.raises(ValueError, match="prefill-only input logprobs"):
        create(None if missing_ids else [1, 99, 99, 99, 2, 10, 11])


@pytest.mark.parametrize(
    "query_ids",
    [None, (0, 5, 7, 9, 0, 0), [0, 0], [0] * 5, [0] * 7, [-1] * 6, [128] * 6, [False] * 6, [0.5] * 6],
)
def test_opd_query_validation_rejects_wrong_size_or_vocabulary(tokenized_request, query_ids):
    request, create = tokenized_request
    request.token_ids_logprob = query_ids
    with pytest.raises(ValueError, match="exactly 6 valid token IDs"):
        create([1, 99, 99, 99, 2, 10, 11])


def test_regular_query_does_not_require_per_position_shape(tokenized_request):
    request, create = tokenized_request
    request.sampling_params["custom_params"] = None
    request.sampling_params["max_new_tokens"] = 2
    request.token_ids_logprob = [0, 127]
    assert create([1, 99, 99, 99, 2, 10, 11]).token_ids_logprob == [0, 127]


@pytest.mark.parametrize("top_k", [1, 3])
def test_opd_scheduler_slices_queries_across_prefill_chunks(monkeypatch, top_k):
    monkeypatch.setattr(
        schedule_batch, "get_server_args", lambda: SimpleNamespace(enable_mamba_extra_buffer=lambda: False)
    )
    monkeypatch.setattr(
        schedule_batch,
        "alloc_for_extend",
        lambda batch: (
            torch.arange(batch.extend_num_tokens),
            torch.arange(len(batch.reqs)),
            list(range(len(batch.reqs))),
        ),
    )
    monkeypatch.setattr(schedule_batch.SamplingBatchInfo, "from_schedule_batch", lambda *args: None)

    def request(query, width):
        params = schedule_batch.SamplingParams(
            max_new_tokens=0,
            custom_params={"relax_opd_token_ids_logprob_k": width} if width is not None else None,
        )
        req = schedule_batch.Req("opd-test", "", array("q", range(10)), params, token_ids_logprob=query)
        req.logprob_start_len = 4
        req.full_untruncated_fill_ids = req.origin_input_ids[:]
        return req

    queries = [[row * top_k + col for col in range(top_k)] for row in range(6)]
    flat_queries = [token for row in queries for token in row]
    per_position = request(flat_queries[:], top_k)
    ordinary = request([2, 8], None)
    no_query = request(None, None)
    batch = schedule_batch.ScheduleBatch(
        reqs=[per_position, ordinary, no_query],
        model_config=SimpleNamespace(vocab_size=128, is_encoder_decoder=False),
        device="cpu",
        return_logprob=True,
    )
    collected = []
    for start, end, expected_rows in [(0, 3, []), (3, 7, queries[:3]), (7, 10, queries[3:])]:
        for req in batch.reqs:
            req.prefix_indices = torch.arange(start)
            req.extend_range = schedule_batch.Range(start, end)
        batch.prepare_for_extend()
        assert batch.token_ids_logprobs == [expected_rows, [2, 8], None]
        assert per_position.logprob.token_ids_logprob == flat_queries
        collected.extend(batch.token_ids_logprobs[0])
    assert collected == queries


def test_compact_logprobs_preserve_scheduler_flat_topk():
    state = SimpleNamespace()
    for side in ("input", "output"):
        for group in ("token_logprobs", "top_logprobs"):
            for field in ("val", "idx"):
                setattr(state, f"{side}_{group}_{field}", [])
    expected_ids = np.array([[0, 5], [7, 9]], dtype=np.int32)
    expected_values = np.array([[-1, -2], [-3, -4]], dtype=np.float32)
    state.input_top_logprobs_scheduler_flat = (expected_values, expected_ids, 1)
    meta = {}
    tokenizer_manager.TokenizerManager._add_relax_logprob_base64(meta, state, 2, None)
    ids, values = LogprobResponse({"meta_info": meta}).strict_topk("input_top_logprobs", 2, 2)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_array_equal(values, expected_values)
