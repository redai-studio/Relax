# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import pybase64
import pytest
import torch
from safetensors.torch import load

from relax.utils.data import kimi_k3
from relax.utils.types import Sample


def test_video_rollout_sends_the_training_prompt_and_visual_tensors(monkeypatch):
    pytest.importorskip("sglang_router")
    rollout = pytest.importorskip("relax.engine.rollout.sglang_rollout", exc_type=ModuleNotFoundError)
    processor = SimpleNamespace(IMAGE_TOKEN="]<]image[>[")
    state = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [90]),
        processor=processor,
        opd_manager=None,
        use_preprocessed_video=True,
        encode_executor=None,
    )
    monkeypatch.setattr(rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(kimi_k3, "is_kimi_k3_tokenizer", lambda tokenizer: False)

    prompt_ids = [1, 2, 2, 5, 5, 3]
    processor_output = {
        "pixel_values": torch.ones(2, 3),
        "image_grid_thw": torch.tensor([[1, 1, 2]]),
        "pixel_values_videos": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "video_grid_thw": torch.tensor([[2, 1, 1]]),
    }
    expected_ids = prompt_ids.copy()

    async def run_processor(state, args, prompt, media):
        return prompt_ids.copy(), processor_output.copy(), 0.0

    async def send(args, state, url, payload, headers, evaluation):
        assert payload["input_ids"] == expected_ids
        assert "image_data" not in payload
        assert payload["return_prompt_token_ids"] is True
        video = payload["video_data"][0]
        assert video["format"] == "processor_output"
        received = load(pybase64.b64decode(video["tensor_data"]))
        assert set(received) == set(processor_output)
        for key, value in processor_output.items():
            torch.testing.assert_close(received[key], value, atol=0, rtol=0)
        return {
            "text": "a",
            "prompt_token_ids": expected_ids.copy(),
            "meta_info": {"output_token_logprobs": [(-0.1, 97)], "finish_reason": {"type": "abort"}},
        }

    monkeypatch.setattr(rollout, "_run_image_processor", run_processor)
    monkeypatch.setattr(rollout, "_post_generation", send)
    args = SimpleNamespace(
        ci_test=False,
        fully_async=False,
        sglang_router_ip="localhost",
        sglang_router_port=1,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        lora_rank=0,
        lora_adapter_mode=False,
        slime_router_sticky=False,
        sglang_router_policy="cache_aware",
        use_slime_router=False,
        sglang_speculative_algorithm=None,
    )
    sample = Sample(
        prompt="question",
        multimodal_inputs={
            "images": [object()],
            "videos": [object()],
            "video_metadata": [{"fps": 2, "frames_indices": [0, 1], "total_num_frames": 2}],
        },
    )
    asyncio.run(rollout.generate(args, sample, {"max_new_tokens": 8}))
    assert sample.tokens == sample.rollout_tokens == prompt_ids + [97]
    expected_ids.append(97)
    asyncio.run(rollout.generate(args, sample, {"max_new_tokens": 8}))
    assert sample.tokens == sample.rollout_tokens == prompt_ids + [97, 97]
