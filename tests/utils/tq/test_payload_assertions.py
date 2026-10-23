# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Raw-byte identity contracts shared by dataplane and acceptance tests."""

from __future__ import annotations

import struct
import warnings
from typing import Any

import numpy as np
import pytest
import torch

from relax.utils.tq.correctness import diff_digests, leaf_digests, payload_nbytes, payload_rows


class NonTensorData:
    def __init__(self, data: Any) -> None:
        self.data = data


class NonTensorStack:
    def __init__(self, data: Any) -> None:
        self._data = data

    def tolist(self) -> Any:
        return self._data


def test_digest_preserves_raw_bytes_dtype_shape_and_paths() -> None:
    positive = leaf_digests(torch.tensor([0.0], dtype=torch.float32))
    negative = leaf_digests(torch.tensor([-0.0], dtype=torch.float32))
    assert positive["payload"][:2] == negative["payload"][:2]
    assert positive["payload"][2] != negative["payload"][2]

    first_nan = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
    second_nan = struct.unpack("!d", bytes.fromhex("7ff8000000000002"))[0]
    assert repr(first_nan) == repr(second_nan) == "nan"
    assert leaf_digests(first_nan) != leaf_digests(second_nan)

    paths = leaf_digests({"a.b": 1, "a": {"b": 2}})
    assert set(paths) == {"payload['a.b']", "payload.a.b"}


def test_digest_supports_nested_numpy_and_non_tensor_containers() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        nested = torch.nested.nested_tensor(
            [torch.tensor([1, 2], dtype=torch.int16), torch.tensor([3], dtype=torch.int16)]
        )
    digests = leaf_digests(
        {
            "array": np.array([[1, 2]], dtype=np.uint16),
            "nested": nested,
            "wrapped": NonTensorStack([NonTensorData({"text": "hello"}), NonTensorData(None)]),
        }
    )
    assert digests["payload.array"][:2] == ("np.uint16", "1x2")
    assert digests["payload.nested[0]"][:2] == ("torch.int16", "2")
    assert digests["payload.nested[1]"][:2] == ("torch.int16", "1")
    assert leaf_digests(NonTensorStack([NonTensorData(None)])) == leaf_digests([None])


def test_rows_nbytes_and_diff_cover_non_tensor_payloads() -> None:
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int16)
    wrapped = NonTensorStack([NonTensorData({"text": "é"}), NonTensorData(None)])
    assert all(torch.equal(got, want) for got, want in zip(payload_rows(tensor), tensor.unbind(), strict=True))
    assert payload_rows(wrapped) == [{"text": "é"}, None]
    assert payload_nbytes({"tensor": tensor, "wrapped": wrapped}) == 8 + 2 + len(repr(None))
    problems = diff_digests(leaf_digests({"expected": 1}), leaf_digests({"actual": 1}))
    assert len(problems) == 2 and "unexpected extra" in problems[0] and "missing" in problems[1]


@pytest.mark.parametrize("payload", [{1: "value"}, {"bad": object()}], ids=["non-string-key", "unknown-leaf"])
def test_digest_rejects_unsupported_payload(payload: dict[Any, Any]) -> None:
    with pytest.raises(TypeError, match="Unsupported payload"):
        leaf_digests(payload)
