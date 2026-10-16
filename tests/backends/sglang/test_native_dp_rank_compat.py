# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Native generate requests must honor the DP rank sent by older routers."""

import importlib
import importlib.util
import json
import warnings

import pytest
from pydantic import TypeAdapter


if importlib.util.find_spec("sglang") is None:
    pytest.skip("Requires the SGLang package installed in the training image", allow_module_level=True)

GenerateReqInput = importlib.import_module("sglang.srt.managers.io_struct").GenerateReqInput


@pytest.mark.parametrize("transport", ["python", "json"])
@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, None),
        ({"routed_dp_rank": 1}, 1),
        ({"data_parallel_rank": 0}, 0),
        ({"data_parallel_rank": 1}, 1),
        ({"routed_dp_rank": 0, "data_parallel_rank": 1}, 0),
        ({"routed_dp_rank": 1, "data_parallel_rank": 0}, 1),
        ({"routed_dp_rank": None, "data_parallel_rank": 1}, 1),
        ({"routed_dp_rank": 1, "data_parallel_rank": None}, 1),
    ],
)
def test_native_dp_rank_compatibility(fields: dict[str, int | None], expected: int | None, transport: str) -> None:
    payload = {"input_ids": [1, 2, 3], **fields}
    adapter = TypeAdapter(GenerateReqInput)
    req = adapter.validate_python(payload) if transport == "python" else adapter.validate_json(json.dumps(payload))
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", DeprecationWarning)
        req.normalize_batch_and_arguments()
    assert req.routed_dp_rank == expected
    assert req.data_parallel_rank is None
    deprecated = [warning for warning in emitted if warning.category is DeprecationWarning]
    assert len(deprecated) == int(fields.get("data_parallel_rank") is not None)


def test_native_dp_rank_survives_batch_splitting() -> None:
    req = GenerateReqInput(input_ids=[[1, 2], [3, 4]], data_parallel_rank=1)
    with pytest.warns(DeprecationWarning, match="data_parallel_rank"):
        req.normalize_batch_and_arguments()
    assert [req[index].routed_dp_rank for index in range(2)] == [1, 1]
    assert all(req[index].data_parallel_rank is None for index in range(2))
