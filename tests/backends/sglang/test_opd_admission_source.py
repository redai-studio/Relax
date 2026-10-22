# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU validation of patched admission and chunk slicing without CUDA
imports."""

import ast
from types import SimpleNamespace

import pytest

from tests.backends.sglang.source_utils import sglang_source_path


def _source_method(relative_path, name):
    path = sglang_source_path(relative_path)
    return next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _compile_body(signature, body, namespace):
    module = ast.parse(f"def {signature}:\n    pass\n")
    module.body[0].body = body
    exec(compile(ast.fix_missing_locations(module), "<patched-sglang-source>", "exec"), namespace)
    return namespace[module.body[0].name]


class GenerateReqInput(SimpleNamespace):
    pass


@pytest.fixture
def admission():
    method = _source_method("srt/managers/tokenizer_manager.py", "_create_tokenized_object")
    index = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "opd_k" for target in node.targets)
    )
    validate = _compile_body(
        "validate(self, obj, sampling_params, input_ids)",
        method.body[index : index + 2],
        {"GenerateReqInput": GenerateReqInput},
    )
    request = GenerateReqInput(return_logprob=True, logprob_start_len=4, token_ids_logprob=[0, 5, 7, 9, 0, 0])
    params = SimpleNamespace(custom_params={"relax_opd_token_ids_logprob_k": 2}, max_new_tokens=0)
    manager = SimpleNamespace(model_config=SimpleNamespace(vocab_size=128))
    return request, params, lambda ids: validate(manager, request, params, ids)


@pytest.mark.parametrize("width", [0, -1, True, 1.5, "2"])
def test_opd_admission_rejects_invalid_width(admission, width):
    _, params, validate = admission
    params.custom_params["relax_opd_token_ids_logprob_k"] = width
    with pytest.raises(ValueError, match="positive integer"):
        validate(list(range(7)))


@pytest.mark.parametrize(
    "field,value",
    [("return_logprob", False), ("logprob_start_len", None), ("logprob_start_len", -1), ("logprob_start_len", 7)],
)
def test_opd_admission_requires_input_logprobs(admission, field, value):
    request, _, validate = admission
    setattr(request, field, value)
    with pytest.raises(ValueError, match="prefill-only"):
        validate(list(range(7)))


@pytest.mark.parametrize("missing_ids", [False, True])
def test_opd_admission_requires_prefill_tokens(admission, missing_ids):
    _, params, validate = admission
    if not missing_ids:
        params.max_new_tokens = 1
    with pytest.raises(ValueError, match="prefill-only"):
        validate(None if missing_ids else list(range(7)))


@pytest.mark.parametrize("query", [None, (0,) * 6, [0] * 5, [0] * 7, [-1] * 6, [128] * 6, [False] * 6, [0.5] * 6])
def test_opd_admission_rejects_invalid_queries(admission, query):
    request, _, validate = admission
    request.token_ids_logprob = query
    with pytest.raises(ValueError, match="exactly 6 valid token IDs"):
        validate(list(range(7)))


@pytest.mark.parametrize("width", [1, 3])
def test_opd_admission_uses_expanded_prompt_length(admission, width):
    request, params, validate = admission
    params.custom_params["relax_opd_token_ids_logprob_k"] = width
    request.token_ids_logprob = [0, 127, 0] * width
    validate(list(range(7)))
    params.custom_params = None
    request.token_ids_logprob = [1, 2]
    params.max_new_tokens = 3
    validate(list(range(7)))


@pytest.mark.parametrize("width", [1, 3])
def test_opd_scheduler_source_preserves_queries_across_chunks(width):
    method = _source_method("srt/managers/schedule_batch.py", "prepare_for_extend")
    block = next(
        node
        for node in method.body
        if isinstance(node, ast.If) and "self.token_ids_logprobs = []" in ast.unparse(node)
    )
    prepare = _compile_body("prepare(self, reqs, extend_logprob_start_lens)", [block], {})
    queries = [[row * width + col for col in range(width)] for row in range(6)]
    flat = [token for row in queries for token in row]

    def req(ids, custom):
        return SimpleNamespace(
            logprob=SimpleNamespace(top_logprobs_num=0, token_ids_logprob=ids),
            sampling_params=SimpleNamespace(custom_params=custom),
            logprob_start_len=4,
        )

    requests = [req(flat[:], {"relax_opd_token_ids_logprob_k": width}), req([2, 8], None), req(None, None)]
    batch = SimpleNamespace(return_logprob=True)
    collected = []
    for start, end, expected in [(0, 3, []), (3, 7, queries[:3]), (7, 10, queries[3:])]:
        for request in requests:
            request.prefix_indices = list(range(start))
            request.extend_range = SimpleNamespace(length=end - start)
        prepare(batch, requests, [min(max(4 - start, 0), end - start)] * 3)
        assert batch.token_ids_logprobs == [expected, [2, 8], None]
        assert requests[0].logprob.token_ids_logprob == flat
        collected.extend(batch.token_ids_logprobs[0])
    assert collected == queries
