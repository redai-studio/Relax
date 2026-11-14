# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch

from relax.utils.multimodal.stats import get_multimodal_token_counts, get_sample_multimodal_stats
from relax.utils.training.train_dump_utils import _sample_to_summary_record
from relax.utils.types import Sample


def test_multimodal_stats_counts_images_and_tokens():
    sample = Sample(
        tokens=list(range(100)),
        response_length=20,
        multimodal_inputs={"images": ["a.png", "b.png"]},
        multimodal_train_inputs={
            "image_grid_thw": torch.tensor([[1, 14, 14], [2, 10, 10]]),
            "video_grid_thw": torch.tensor([[3, 8, 8]]),
        },
        metadata={"rollout_turns": 3},
    )

    token_counts = get_multimodal_token_counts(sample.multimodal_train_inputs)
    assert token_counts == {
        "image": 99,
        "video": 48,
        "audio": 0,
        "total": 147,
    }
    assert get_sample_multimodal_stats(sample) == {
        "image_count": 2,
        "image_token_count": 99,
        "video_token_count": 48,
        "audio_token_count": 0,
        "multimodal_token_count": 147,
    }


def test_rollout_summary_record_includes_token_and_agent_stats():
    sample = Sample(
        prompt="hello",
        response="world",
        tokens=list(range(12)),
        response_length=5,
        reward=1.0,
        multimodal_inputs={"images": ["image.png"]},
        multimodal_train_inputs={"image_grid_thw": [[1, 8, 8]]},
        metadata={"rollout_turns": 2},
    )

    record = _sample_to_summary_record(sample, rollout_id=7, idx=0)

    assert record["prompt_token_count"] == 7
    assert record["response_token_count"] == 5
    assert record["total_token_count"] == 12
    assert record["prompt_length"] == 7
    assert record["image_count"] == 1
    assert record["image_token_count"] == 16
    assert record["multimodal_token_count"] == 16
    assert record["agent_turns"] == 2
