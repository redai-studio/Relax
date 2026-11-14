# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Compaction preserves copy hooks, graph ownership and failure ordering."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pytest

from relax.agentic.pipeline import (
    TrainingFieldArtifact,
    _can_compact_without_deepcopy_hooks,
    _compact_sample_payload,
)
from relax.utils.types import Sample


class _CopiedList(list[int]):
    def __deepcopy__(self, memo: dict[int, Any]) -> list[int]:
        result = [value + 100 for value in self]
        memo[id(self)] = result
        return result


class _CopiedInt(int):
    def __deepcopy__(self, memo: dict[int, Any]) -> int:
        return int(self) + 100


class _MutatingMetadata:
    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, str]:
        self.tokens[0] = 101
        return {"copied": "metadata"}


class _MetadataCopyError(RuntimeError):
    pass


class _FailingMetadata:
    def __deepcopy__(self, memo: dict[int, Any]) -> None:
        raise _MetadataCopyError("metadata copy precedes array conversion")


def test_from_sample_preserves_list_subclass_deepcopy() -> None:
    source = Sample(tokens=_CopiedList([1, 2]), rollout_tokens=[3, 4])
    artifact = TrainingFieldArtifact.from_sample(source)
    np.testing.assert_array_equal(artifact.sample_payload["tokens"], [101, 102])
    assert artifact.sample_payload["tokens"].dtype == np.int32
    assert list(source.tokens) == [1, 2]


def test_from_sample_preserves_numeric_element_deepcopy() -> None:
    source = Sample(tokens=[_CopiedInt(1), _CopiedInt(2)], rollout_tokens=[3, 4])
    artifact = TrainingFieldArtifact.from_sample(source)
    np.testing.assert_array_equal(artifact.sample_payload["tokens"], [101, 102])
    assert [int(value) for value in source.tokens] == [1, 2]


def test_deepcopy_result_determines_shared_rollout_marker() -> None:
    source = Sample(tokens=_CopiedList([1, 2]), rollout_tokens=[101, 102])
    artifact = TrainingFieldArtifact.from_sample(source)
    assert artifact.sample_payload["_rollout_tokens_shared"] is True
    assert artifact.sample_payload["rollout_tokens"] is None
    restored = artifact.to_sample()
    assert restored.tokens == restored.rollout_tokens == [101, 102]
    assert restored.tokens is not restored.rollout_tokens


def test_metadata_copy_order_precedes_numeric_conversion() -> None:
    tokens = [1, 2]
    payload = {"metadata": _MutatingMetadata(tokens), "tokens": tokens}
    compacted = _compact_sample_payload(payload)
    np.testing.assert_array_equal(compacted["tokens"], [101, 2])
    assert compacted["metadata"] == {"copied": "metadata"}


def test_compaction_preserves_payload_root_cycle() -> None:
    payload: dict[str, Any] = {"tokens": [1, 2]}
    payload["metadata"] = {"root": payload}
    compacted = _compact_sample_payload(payload)
    assert compacted["metadata"]["root"] is compacted
    np.testing.assert_array_equal(compacted["tokens"], [1, 2])
    compacted["tokens"][0] = 99
    assert payload["tokens"] == [1, 2]


def test_deepcopy_error_precedes_invalid_numeric_conversion() -> None:
    payload = {"metadata": _FailingMetadata(), "tokens": [object()]}
    with pytest.raises(_MetadataCopyError, match="metadata copy precedes array conversion"):
        _compact_sample_payload(payload)


def test_repeated_container_aliases_preserve_internal_ownership() -> None:
    shared = [1, 2]
    payload = {"tokens": shared, "alias": shared, "metadata": {"same": shared}}
    compacted = _compact_sample_payload(payload)
    assert compacted["alias"] is compacted["metadata"]["same"]
    assert compacted["alias"] is not shared
    compacted["alias"][0] = 99
    assert shared == [1, 2]
    np.testing.assert_array_equal(compacted["tokens"], [1, 2])


def test_eligibility_does_not_invoke_unknown_object_or_metaclass_hooks() -> None:
    calls = {"object_eq": 0, "object_hash": 0, "object_iter": 0, "type_eq": 0, "type_hash": 0}

    class NoHooksMeta(type):
        def __eq__(cls, other: Any) -> bool:
            calls["type_eq"] += 1
            raise AssertionError("eligibility called custom metaclass equality")

        def __hash__(cls) -> int:
            calls["type_hash"] += 1
            raise AssertionError("eligibility called custom metaclass hashing")

    class Opaque(metaclass=NoHooksMeta):
        def __eq__(self, other: Any) -> bool:
            calls["object_eq"] += 1
            raise AssertionError("eligibility called custom object equality")

        def __hash__(self) -> int:
            calls["object_hash"] += 1
            raise AssertionError("eligibility called custom object hashing")

        def __iter__(self) -> Iterator[Any]:
            calls["object_iter"] += 1
            raise AssertionError("eligibility called custom object iteration")

    assert _can_compact_without_deepcopy_hooks({"tokens": [1, 2], "metadata": {"opaque": Opaque()}}) is False
    assert calls == {"object_eq": 0, "object_hash": 0, "object_iter": 0, "type_eq": 0, "type_hash": 0}
