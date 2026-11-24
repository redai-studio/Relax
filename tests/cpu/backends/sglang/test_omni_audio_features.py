# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration


@pytest.fixture
def audio_tower():
    tower = Mock(dtype=torch.bfloat16)
    tower.parameters.side_effect = lambda: iter([torch.empty(0)])
    tower.return_value = SimpleNamespace(last_hidden_state=torch.empty(0))
    return tower


def test_audio_features_pack_variable_lengths_and_preserve_sample_boundaries(audio_tower):
    first = torch.arange(2 * 128 * 5, dtype=torch.float32).reshape(2, 128, 5)
    second = torch.arange(128 * 3, dtype=torch.float32).reshape(1, 128, 3)
    items = [
        SimpleNamespace(feature=first, feature_attention_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])),
        SimpleNamespace(feature=second, feature_attention_mask=torch.tensor([[1, 1, 0]])),
    ]

    result = Qwen3OmniMoeThinkerForConditionalGeneration.get_audio_feature(
        SimpleNamespace(audio_tower=audio_tower), items
    )

    audio_tower.assert_called_once()
    packed = audio_tower.call_args.args[0]
    lengths = audio_tower.call_args.kwargs["feature_lens"]
    expected = torch.cat([first[0], first[1, :, :3], second[0, :, :2]], dim=1).to(torch.bfloat16)
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    torch.testing.assert_close(lengths, torch.tensor([5, 3, 2]))
    assert result is audio_tower.return_value.last_hidden_state


def test_audio_features_preserve_equal_length_behavior(audio_tower):
    features = [torch.arange(128 * 4, dtype=torch.float32).reshape(1, 128, 4) + offset for offset in (0, 1000)]
    masks = [torch.tensor([[1, 1, 0, 0]]), torch.tensor([[1, 0, 1, 0]])]
    items = [SimpleNamespace(feature=feature, feature_attention_mask=mask) for feature, mask in zip(features, masks)]
    old_features = torch.cat(features).to(torch.bfloat16)
    old_mask = torch.cat(masks)
    expected = old_features.permute(0, 2, 1)[old_mask.bool()].permute(1, 0)

    Qwen3OmniMoeThinkerForConditionalGeneration.get_audio_feature(SimpleNamespace(audio_tower=audio_tower), items)

    audio_tower.assert_called_once()
    torch.testing.assert_close(audio_tower.call_args.args[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(audio_tower.call_args.kwargs["feature_lens"], old_mask.sum(dim=1))
