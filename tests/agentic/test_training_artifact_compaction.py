# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Values and ownership contracts for faster training artifact compaction."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest
import torch

from relax.agentic.pipeline import (
    _TRAINING_ARTIFACT_ARRAY_FIELDS,
    TrainingFieldArtifact,
    _compact_sample_payload,
    _expand_sample_payload,
)
from relax.utils.types import Sample


def _original_compact(payload: dict[str, Any]) -> dict[str, Any]:
    """Frozen pre-optimization implementation from 9d4461d; no Git
    dependency."""
    compacted = copy.deepcopy(payload)
    tokens = compacted.get("tokens")
    rollout_tokens = compacted.get("rollout_tokens")
    if isinstance(tokens, list) and isinstance(rollout_tokens, list) and rollout_tokens == tokens:
        compacted["rollout_tokens"] = None
        compacted["_rollout_tokens_shared"] = True
    for field_name, dtype in _TRAINING_ARTIFACT_ARRAY_FIELDS:
        value = compacted.get(field_name)
        if not isinstance(value, list) or not value:
            continue
        compacted[field_name] = np.asarray(value, dtype=dtype)
    return compacted


@dataclass
class _Box:
    values: list[int]


class _CopiedList(list):
    def __deepcopy__(self, memo: dict[int, Any]) -> list[int]:
        copied = [value + 100 for value in self]
        memo[id(self)] = copied
        return copied


class _CopiedNumber:
    def __init__(self, value: int) -> None:
        self.value = value

    def __deepcopy__(self, memo: dict[int, Any]) -> int:
        del memo
        return self.value + 100


def _assert_same_graph(expected: Any, actual: Any, seen: dict[int, int] | None = None) -> None:
    """Compare values, dtypes and mutable aliases while supporting cycles."""
    assert type(actual) is type(expected)
    if seen is None:
        seen = {}
    if isinstance(expected, (dict, list, np.ndarray, torch.Tensor, _Box)):
        previous = seen.get(id(expected))
        if previous is not None:
            assert id(actual) == previous
            return
        assert id(actual) not in seen.values()
        seen[id(expected)] = id(actual)
    if isinstance(expected, np.ndarray):
        assert expected.dtype == actual.dtype
        np.testing.assert_array_equal(expected, actual)
    elif isinstance(expected, torch.Tensor):
        assert expected.dtype == actual.dtype
        assert torch.equal(expected, actual)
        assert expected.requires_grad == actual.requires_grad
        _assert_same_graph(expected.grad, actual.grad, seen)
        _assert_same_graph(expected.__dict__, actual.__dict__, seen)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_same_graph(expected[key], actual[key], seen)
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _assert_same_graph(left, right, seen)
    elif isinstance(expected, _Box):
        _assert_same_graph(expected.__dict__, actual.__dict__, seen)
    else:
        assert expected == actual


@pytest.mark.parametrize("rollout_kind", ["same-object", "equal-values", "different", "absent"])
def test_compaction_matches_original_numeric_training_fields(rollout_kind: str) -> None:
    tokens = [1, 2, 3]
    payload: dict[str, Any] = {
        "tokens": tokens,
        "loss_mask": [True, False, True],
        "rollout_log_probs": [-0.125, -np.inf, np.nan],
        "teacher_log_probs": [-0.5, 0.0, -1.25],
        "teacher_topk_token_ids": [[1, 5], [2, 6], [3, 7]],
        "metadata": {"nested": ["context", {"turn": 2}]},
        "custom_advantage": [[0, 2]],
    }
    if rollout_kind != "absent":
        payload["rollout_tokens"] = tokens if rollout_kind == "same-object" else [1, 2, 3]
        if rollout_kind == "different":
            payload["rollout_tokens"][1] = 9
    expected = _original_compact(payload)
    actual = _compact_sample_payload(payload)
    _assert_same_graph(expected, actual)
    for field_name, dtype in _TRAINING_ARTIFACT_ARRAY_FIELDS:
        value = actual.get(field_name)
        if value is not None:
            assert isinstance(value, np.ndarray)
            assert value.dtype == np.dtype(dtype)
    assert tokens == [1, 2, 3]
    assert isinstance(payload["loss_mask"], list)


@pytest.mark.parametrize("empty_kind", ["empty-list", "none", "missing"])
def test_compaction_preserves_empty_none_and_missing_fields(empty_kind: str) -> None:
    payload = {"metadata": {"kind": empty_kind}}
    if empty_kind != "missing":
        payload.update(
            {
                field_name: [] if empty_kind == "empty-list" else None
                for field_name, _ in _TRAINING_ARTIFACT_ARRAY_FIELDS
            }
        )
    _assert_same_graph(_original_compact(payload), _compact_sample_payload(payload))


def test_compaction_preserves_unknown_field_aliases_cycles_and_ownership() -> None:
    shared = {"nested": [1, 2]}
    array = np.arange(4, dtype=np.int64)
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    box = _Box([4, 5])
    metadata = {"shared": shared, "array": array, "tensor": tensor, "box": box}
    metadata["cycle"] = metadata
    payload = {
        "tokens": [1, 2],
        "rollout_tokens": [1, 2],
        "metadata": metadata,
        "train_metadata": metadata,
        "unknown_shared": shared,
        "unknown_array": array,
        "unknown_tensor": tensor,
        "unknown_box": box,
    }
    payload["self"] = payload
    actual = _compact_sample_payload(payload)
    _assert_same_graph(_original_compact(payload), actual)
    assert actual["self"] is actual
    assert actual["metadata"] is actual["train_metadata"]
    assert actual["metadata"]["cycle"] is actual["metadata"]
    assert actual["metadata"]["shared"] is actual["unknown_shared"]
    assert actual["metadata"]["box"] is actual["unknown_box"]
    assert not np.shares_memory(actual["unknown_array"], array)
    assert actual["unknown_tensor"].data_ptr() != tensor.data_ptr()
    assert actual["metadata"] is not metadata
    shared["nested"].append(3)
    assert actual["unknown_shared"]["nested"] == [1, 2]
    actual["unknown_box"].values.append(6)
    assert box.values == [4, 5]


def test_compaction_copies_metadata_aliases_of_converted_training_lists() -> None:
    tokens = [1, 2, 3]
    payload = {
        "tokens": tokens,
        "rollout_tokens": tokens,
        "teacher_tokens": tokens,
        "metadata": {"tokens": tokens},
    }
    actual = _compact_sample_payload(payload)
    _assert_same_graph(_original_compact(payload), actual)
    assert actual["metadata"]["tokens"] is actual["teacher_tokens"]
    assert isinstance(actual["metadata"]["tokens"], list)
    tokens[0] = 9
    assert actual["tokens"].tolist() == [1, 2, 3]
    assert actual["metadata"]["tokens"] == [1, 2, 3]
    actual["metadata"]["tokens"][1] = 8
    assert actual["tokens"].tolist() == [1, 2, 3]
    actual["tokens"][2] = 7
    assert actual["metadata"]["tokens"] == [1, 8, 3]
    assert tokens == [9, 2, 3]


def test_compaction_keeps_existing_array_dtype_and_aliases() -> None:
    array = np.array([1, 2, 3], dtype=np.int64)
    payload = {"tokens": array, "rollout_tokens": array, "metadata": {"tokens": array}}
    actual = _compact_sample_payload(payload)
    _assert_same_graph(_original_compact(payload), actual)
    assert actual["tokens"].dtype == np.int64
    assert actual["tokens"] is actual["rollout_tokens"] is actual["metadata"]["tokens"]
    assert not np.shares_memory(actual["tokens"], array)
    array[0] = 9
    assert actual["tokens"].tolist() == [1, 2, 3]


@pytest.mark.parametrize("value", [_CopiedList([1, 2]), [_CopiedNumber(7)], [np.int64(1), np.int64(2)]])
def test_compaction_preserves_nonbuiltin_list_deepcopy_semantics(value: Any) -> None:
    payload = {"tokens": value, "metadata": {"input": value}}
    _assert_same_graph(_original_compact(payload), _compact_sample_payload(payload))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("tokens", ["not-an-integer"]), ("teacher_topk_token_ids", [[1, 2], [3]]), ("tokens", [2**100])],
)
def test_compaction_preserves_invalid_conversion_errors(field_name: str, value: list[Any]) -> None:
    payload = {field_name: value, "metadata": {"unchanged": [1, 2]}}
    before = copy.deepcopy(payload)
    with pytest.raises((ValueError, TypeError, OverflowError)) as old_error:
        _original_compact(payload)
    with pytest.raises(type(old_error.value)):
        _compact_sample_payload(payload)
    _assert_same_graph(before, payload)


def test_training_artifact_roundtrip_keeps_values_and_independent_buffers() -> None:
    sample = Sample(
        tokens=[1, 2, 3, 4],
        rollout_tokens=[1, 2, 3, 4],
        loss_mask=[1, 0],
        rollout_log_probs=[-0.25, 0.0],
        teacher_topk_token_ids=[[3, 5], [4, 6]],
        metadata={"nested": {"values": [7]}},
        train_metadata={"weight": torch.tensor([2.0])},
    )
    artifact = TrainingFieldArtifact.from_sample(sample)
    restored = artifact.to_sample()
    expected = Sample.from_dict(_expand_sample_payload(_original_compact(sample.to_dict())))
    _assert_same_graph(expected.to_dict(), restored.to_dict())
    sample.tokens[0] = 99
    sample.metadata["nested"]["values"].append(8)
    assert artifact.sample_payload["tokens"].tolist() == [1, 2, 3, 4]
    restored.tokens[1] = 88
    restored.metadata["nested"]["values"].append(9)
    assert artifact.sample_payload["tokens"].tolist() == [1, 2, 3, 4]
    assert artifact.sample_payload["metadata"]["nested"]["values"] == [7]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        pytest.param("tokens", [-(2**31), -(2**31) + 1, 0, 2**31 - 1], id="int32-boundaries"),
        pytest.param("tokens", [-(2**31) - 1, 2**31, 2**32 + 1], id="int32-overflow-wrap"),
        pytest.param("tokens", [-2.99, -0.99, 0.99, 2.99], id="float-to-int32-truncation"),
        pytest.param("loss_mask", [-257, -1, 0, 1, 255, 256, 257], id="uint8-boundaries-and-wrap"),
        pytest.param("loss_mask", [-1.9, 0.9, 255.9, 256.9], id="float-to-uint8-truncation"),
        pytest.param(
            "rollout_log_probs",
            [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, True, False, 1, 2.5],
            id="float64-special-values-and-bools",
        ),
        pytest.param("teacher_log_probs", [-0.0, True, False, 2.5], id="teacher-float64-scalar-types"),
        pytest.param("rollout_tokens", [True, False, 7], id="rollout-int32-bools"),
        pytest.param("teacher_topk_token_ids", [True, False, 7], id="flat-teacher-int32-bools"),
    ],
)
def test_compaction_primitive_conversion_matches_original_bytes_and_warnings(
    field_name: str, value: list[int | float | bool]
) -> None:
    payload = {field_name: value}
    with warnings.catch_warnings(record=True) as old_warnings:
        warnings.simplefilter("always")
        expected = _original_compact(payload)
    with warnings.catch_warnings(record=True) as new_warnings:
        warnings.simplefilter("always")
        actual = _compact_sample_payload(payload)
    _assert_same_graph(expected, actual)
    assert actual[field_name].tobytes() == expected[field_name].tobytes()
    assert [(item.category, str(item.message)) for item in new_warnings] == [
        (item.category, str(item.message)) for item in old_warnings
    ]
    assert payload[field_name] is value


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        pytest.param("tokens", [float("nan")], id="int32-nan"),
        pytest.param("tokens", [float("inf")], id="int32-positive-infinity"),
        pytest.param("tokens", [-float("inf")], id="int32-negative-infinity"),
        pytest.param("tokens", [2**100], id="int32-huge-int"),
        pytest.param("loss_mask", [float("nan")], id="uint8-nan"),
        pytest.param("loss_mask", [float("inf")], id="uint8-infinity"),
        pytest.param("loss_mask", [2**100], id="uint8-huge-int"),
        pytest.param("teacher_log_probs", [2**10000], id="float64-huge-int"),
    ],
)
def test_compaction_primitive_conversion_errors_match_original(field_name: str, value: list[int | float]) -> None:
    payload = {field_name: value}
    with warnings.catch_warnings(record=True) as old_warnings:
        warnings.simplefilter("always")
        with pytest.raises((ValueError, TypeError, OverflowError)) as old_error:
            _original_compact(payload)
    with warnings.catch_warnings(record=True) as new_warnings:
        warnings.simplefilter("always")
        with pytest.raises(type(old_error.value)) as new_error:
            _compact_sample_payload(payload)
    # Compare the two implementations dynamically, without freezing a NumPy
    # release's error/warning wording in the test expectations.
    assert str(new_error.value) == str(old_error.value)
    assert [(item.category, str(item.message)) for item in new_warnings] == [
        (item.category, str(item.message)) for item in old_warnings
    ]
    assert payload[field_name] is value


class _MutateSourceOnCopy:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        self.values[0] = 99
        return {"copied_values": copy.deepcopy(self.values, memo)}


class _ObserveRootOnCopy:
    def __init__(self, root: dict[str, Any]) -> None:
        self.root = root

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        root = memo[id(self.root)]
        return {"keys_seen": tuple(root), "tokens_seen": root.get("tokens"), "root": root}


def _compare_compaction(factory: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    # A deepcopy hook may mutate its source, so give each algorithm fresh input.
    expected_source, actual_source = factory(), factory()
    expected = _original_compact(expected_source)
    actual = _compact_sample_payload(actual_source)
    _assert_same_graph(expected, actual)
    assert expected_source["tokens"] == actual_source["tokens"]
    return actual_source, actual


@pytest.mark.parametrize("hook", ["mutate-source", "observe-root"])
@pytest.mark.parametrize("hook_first", [False, True])
def test_compaction_custom_deepcopy_keeps_original_order_and_root_memo(hook: str, hook_first: bool) -> None:
    def factory() -> dict[str, Any]:
        values = [1, 2]
        payload: dict[str, Any] = {}
        metadata = _MutateSourceOnCopy(values) if hook == "mutate-source" else _ObserveRootOnCopy(payload)
        if hook_first:
            payload["metadata"] = metadata
        payload.update(tokens=values, loss_mask=[1, 1], extra={"numbers": [3, 4]})
        if not hook_first:
            payload["metadata"] = metadata
        return payload

    source, result = _compare_compaction(factory)
    if hook == "mutate-source":
        assert source["tokens"][0] == 99
        assert result["tokens"][0] == (99 if hook_first else 1)
    else:
        assert result["metadata"]["root"] is result
        assert result["metadata"]["keys_seen"] == (() if hook_first else ("tokens", "loss_mask", "extra"))


@pytest.mark.parametrize("tensor_kind", ["plain", "attrs", "grad-attrs", "root-attrs", "attrs-no-grad"])
def test_compaction_tensor_grad_attrs_hooks_and_storage_remain_independent(tensor_kind: str) -> None:
    def factory() -> dict[str, Any]:
        values = [1, 2]
        payload: dict[str, Any] = {"tokens": values}
        tensor = torch.tensor([0.25, 0.5], dtype=torch.float64, requires_grad=tensor_kind != "attrs-no-grad")
        if tensor_kind != "attrs-no-grad":
            tensor.grad = torch.tensor([1.0, 2.0], dtype=torch.float64)
        if tensor_kind in {"attrs", "attrs-no-grad"}:
            tensor.copy_hook = _MutateSourceOnCopy(values)
        elif tensor_kind == "grad-attrs":
            tensor.grad.copy_hook = _MutateSourceOnCopy(values)
        elif tensor_kind == "root-attrs":
            tensor.copy_hook = _ObserveRootOnCopy(payload)
        payload.update(multimodal_tensor=tensor, metadata={"tensor": tensor})
        return payload

    source, result = _compare_compaction(factory)
    tensor = result["multimodal_tensor"]
    assert tensor is result["metadata"]["tensor"]
    assert tensor.data_ptr() != source["multimodal_tensor"].data_ptr()
    if tensor.grad is not None:
        assert tensor.grad.data_ptr() != source["multimodal_tensor"].grad.data_ptr()
    with torch.no_grad():
        tensor[0] = 9
    assert source["multimodal_tensor"][0].item() == 0.25
    if tensor_kind in {"attrs", "attrs-no-grad", "grad-attrs"}:
        assert source["tokens"][0] == 99
        assert result["tokens"][0] == 1


@pytest.mark.parametrize("hook_kind", ["mutate-source", "observe-root"])
def test_compaction_hidden_tensor_storage_hooks_keep_original_copy_order(hook_kind: str) -> None:
    def factory() -> dict[str, Any]:
        values = [1, 2]
        observations: list[tuple[str, ...]] = []
        payload: dict[str, Any] = {"tokens": values, "storage_observations": observations}
        tensor = torch.tensor([0.25, 0.5])
        storage = tensor.untyped_storage()
        original_deepcopy = storage.__deepcopy__

        def copy_storage(memo: dict[int, Any]) -> Any:
            if hook_kind == "mutate-source":
                values[0] = 99
            else:
                observations.append(tuple(memo[id(payload)]))
            return original_deepcopy(memo)

        storage.__deepcopy__ = copy_storage
        assert tensor.__dict__ == {}, "the hook must only be on its hidden storage"
        payload["metadata"] = {"tensor": tensor}
        return payload

    source, result = _compare_compaction(factory)
    if hook_kind == "mutate-source":
        assert source["tokens"][0] == 99
        assert result["tokens"][0] == 1
    else:
        assert source["storage_observations"] == [("tokens", "storage_observations")]
        assert result["storage_observations"] == []


def test_compaction_nonleaf_tensor_keeps_original_deepcopy_error() -> None:
    def factory() -> dict[str, Any]:
        return {"tokens": [1, 2], "metadata": {"nonleaf": torch.tensor([1.0], requires_grad=True) * 2}}

    with pytest.raises(RuntimeError) as expected:
        _original_compact(factory())
    with pytest.raises(type(expected.value)) as actual:
        _compact_sample_payload(factory())
    assert str(actual.value) == str(expected.value)


class _CopiedPayload(dict):
    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        copied: dict[str, Any] = {"root_copy_hook": True}
        memo[id(self)] = copied
        for key, value in self.items():
            copied[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return copied


class _MutatingList(list):
    def __init__(self, source: list[int]) -> None:
        super().__init__([5, 6])
        self.source = source

    def __deepcopy__(self, memo: dict[int, Any]) -> list[int]:
        self.source[0] = 99
        copied = [value + 100 for value in self]
        memo[id(self)] = copied
        return copied


class _MutatingTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, source: list[int]) -> "_MutatingTensor":
        result = torch.Tensor._make_subclass(cls, torch.tensor([0.25]), False)
        result.source = source
        return result

    def __deepcopy__(self, memo: dict[int, Any]) -> torch.Tensor:
        self.source[0] = 99
        copied = torch.tensor([0.5])
        memo[id(self)] = copied
        return copied


def test_compaction_root_dict_subclass_keeps_deepcopy_hook_and_root_cycle() -> None:
    def factory() -> dict[str, Any]:
        payload = _CopiedPayload(tokens=[1, 2])
        payload["self"] = payload
        return payload

    _source, result = _compare_compaction(factory)
    assert result["root_copy_hook"] is True
    assert result["self"] is result


@pytest.mark.parametrize("kind", ["list", "tensor"])
def test_compaction_nested_builtin_and_tensor_subclasses_keep_copy_hooks(kind: str) -> None:
    def factory() -> dict[str, Any]:
        values = [1, 2]
        hook = _MutatingList(values) if kind == "list" else _MutatingTensor(values)
        return {"tokens": values, "metadata": {"hook": hook}}

    source, result = _compare_compaction(factory)
    assert source["tokens"][0] == 99
    assert result["tokens"][0] == 1


def test_compaction_plain_builtin_graph_preserves_root_and_tuple_cycles() -> None:
    metadata: dict[str, Any] = {"set": {1, 2}, "frozen": frozenset({3, 4})}
    cycle: list[Any] = []
    pair = (cycle,)
    cycle.append(pair)
    metadata["tuple_cycle"] = pair
    payload: dict[str, Any] = {"tokens": [1, 2], "metadata": metadata, "train_metadata": metadata}
    payload["self"] = payload
    result = _compact_sample_payload(payload)
    _assert_same_graph(_original_compact(payload), result)
    assert result["self"] is result
    assert result["metadata"] is result["train_metadata"]
    assert result["metadata"]["tuple_cycle"][0][0] is result["metadata"]["tuple_cycle"]
    assert result["metadata"] is not metadata
