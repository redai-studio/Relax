# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Transport compaction must own every mutable training field."""

import copy

import numpy as np
import pytest

from relax.agentic.pipeline import TrainingFieldArtifact, _compact_sample_payload
from relax.utils.types import Sample


@pytest.mark.parametrize(
    ("field", "values", "dtype"),
    [
        ("tokens", [1, 2, 3], np.int32),
        ("rollout_tokens", [3, 4], np.int32),
        ("loss_mask", [0, 1], np.uint8),
        ("rollout_log_probs", [-0.125, -0.5], np.float64),
        ("teacher_log_probs", [-0.75, -1.0], np.float64),
        ("teacher_topk_token_ids", [[1, 2], [3, 4]], np.int32),
    ],
)
def test_compacted_numeric_arrays_are_independently_owned(field, values, dtype):
    expected = np.asarray(values, dtype=dtype)
    payload = {field: values, "metadata": {"nested": [1]}}
    compacted = _compact_sample_payload(payload)
    np.testing.assert_array_equal(compacted[field], expected)
    assert compacted[field].dtype == expected.dtype
    assert compacted[field].flags.owndata
    compacted[field].flat[0] = 99
    compacted["metadata"]["nested"].append(2)
    np.testing.assert_array_equal(np.asarray(values, dtype=dtype), expected)
    assert payload["metadata"] == {"nested": [1]}


def test_compaction_preserves_dynamic_fields_without_aliasing():
    sample = Sample(tokens=[1, 2, 3], rollout_tokens=[1, 2, 3], response_length=2)
    sample.custom_payload = {"values": [4, 5], "array": np.array([6, 7])}
    artifact = TrainingFieldArtifact.from_sample(sample)
    assert artifact.sample_payload["_rollout_tokens_shared"] is True
    restored = artifact.to_sample()
    assert restored.tokens == sample.tokens
    assert restored.rollout_tokens == sample.rollout_tokens
    assert restored.rollout_tokens is not restored.tokens
    artifact.sample_payload["custom_payload"]["values"].append(9)
    artifact.sample_payload["custom_payload"]["array"][0] = 0
    assert sample.custom_payload["values"] == [4, 5]
    np.testing.assert_array_equal(sample.custom_payload["array"], [6, 7])


def test_compaction_preserves_empty_none_and_existing_array_fields():
    payload = {"tokens": [], "loss_mask": None, "rollout_log_probs": np.array([-0.25]), "extra": []}
    compacted = _compact_sample_payload(payload)
    assert compacted["tokens"] == []
    assert compacted["loss_mask"] is None
    assert compacted["extra"] == []
    assert compacted["extra"] is not payload["extra"]
    compacted["rollout_log_probs"][0] = 0
    np.testing.assert_array_equal(payload["rollout_log_probs"], [-0.25])


def _assert_same_and_independent(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)
    elif isinstance(expected, dict):
        assert actual is not expected
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same_and_independent(actual[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list) and actual is not expected
        assert len(actual) == len(expected)
        for value, original in zip(actual, expected):
            _assert_same_and_independent(value, original)
    else:
        assert actual == expected


@pytest.mark.parametrize("shared_rollout", [False, True])
def test_complete_nonempty_sample_round_trip_preserves_values_and_isolation(shared_rollout):
    ids = np.array([[5, 6], [7, 8]], dtype=np.int32)
    probabilities = np.array([[-0.125, -0.25], [-0.5, -0.75]], dtype=np.float32)
    sample = Sample(
        group_index=3,
        index=4,
        prompt=[{"role": "user", "content": "Describe this image"}],
        tokens=[1, 2, 3, 4],
        rollout_tokens=[1, 2, 3, 4] if shared_rollout else [1, 5, 3, 4],
        multimodal_inputs={"images": ["image-reference"]},
        multimodal_train_inputs={"pixels": np.array([[0.25, 0.5]], dtype=np.float32)},
        response="answer",
        response_length=2,
        label="label",
        reward={"score": 0.5},
        custom_advantage=[[0.125, 0.25]],
        loss_mask=[0, 1],
        weight_versions=["v1", "v2"],
        rollout_log_probs=[-0.125, -0.25],
        rollout_topk_token_ids=ids,
        rollout_topk_log_probs=probabilities,
        rollout_routed_experts=np.arange(4, dtype=np.int16).reshape(2, 1, 2),
        remove_sample=True,
        abort_count=2,
        teacher_log_probs=[-0.5, -0.75],
        student_topk_token_ids=ids,
        student_topk_log_probs=probabilities,
        teacher_at_student_topk_log_probs=probabilities,
        teacher_topk_token_ids=[[5, 6], [7, 8]],
        teacher_topk_log_probs=probabilities,
        student_at_teacher_topk_log_probs=probabilities,
        opd_topk_token_ids=[ids],
        opd_topk_student_log_probs=[probabilities],
        opd_topk_teacher_log_probs=[probabilities],
        opd_topk_ksz=np.array([2, 2], dtype=np.int32),
        teacher_prompt=[{"role": "user", "content": "teacher prompt"}],
        teacher_multimodal_inputs={"images": ["teacher-image"]},
        teacher_tokens=[9, 10, 3, 4],
        teacher_rollout_tokens=[9, 10, 3, 4],
        teacher_prompt_length=2,
        teacher_image_data={"images": ["encoded-image"]},
        status=Sample.Status.COMPLETED,
        metadata={"nested": {"tags": ["original"]}},
        generate_function_path="example.generate",
        train_metadata={"loss": {"weight": [0.5]}},
        session_id="session",
        non_generation_time=1.25,
        spec_info=Sample.SpecInfo(1, 2, 3, 4),
        prefix_cache_info=Sample.PrefixCacheInfo(5, 6),
    )
    sample.dynamic_training = {"values": [ids, {"nested": [7]}]}
    expected = copy.deepcopy(sample.to_dict())
    artifact = TrainingFieldArtifact.from_sample(sample)
    assert artifact.sample_payload["tokens"].dtype == np.int32
    assert artifact.sample_payload["loss_mask"].dtype == np.uint8
    assert artifact.sample_payload["rollout_log_probs"].dtype == np.float64
    assert artifact.sample_payload["teacher_log_probs"].dtype == np.float64
    assert artifact.sample_payload["teacher_topk_token_ids"].dtype == np.int32
    restored = artifact.to_sample()
    _assert_same_and_independent(restored.to_dict(), sample.to_dict())
    assert restored.tokens is not restored.rollout_tokens
    sample.tokens[0] = 99
    sample.multimodal_train_inputs["pixels"][0, 0] = 9
    sample.metadata["nested"]["tags"].append("changed")
    sample.dynamic_training["values"][0][0, 0] = 99
    _assert_same_and_independent(artifact.to_sample().to_dict(), expected)
    restored.train_metadata["loss"]["weight"].append(9)
    _assert_same_and_independent(artifact.to_sample().to_dict(), expected)
