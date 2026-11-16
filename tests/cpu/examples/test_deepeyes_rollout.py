# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest


class _TextTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("The text tokenizer requires a string")
        return [1] if text == "preamble" else [1, 2, 3]

    def apply_chat_template(self, messages: list[dict], **kwargs) -> str:
        return "preamble" if len(messages) == 2 else "preamble observation"


@pytest.mark.parametrize("structured", [False, True])
def test_deepeyes_initial_inputs_preserve_processor_messages(monkeypatch, structured: bool) -> None:
    rollout = pytest.importorskip("examples.deepeyes.rollout")
    from relax.utils.types import Sample

    prompt = [{"role": "user", "content": "hello"}] if structured else "hello"
    image = object()
    processor_calls = []

    def processor(*, text, images):
        processor_calls.append((text, images))
        return {"input_ids": [[10, 11, 12]], "attention_mask": [[1, 1, 1]], "pixel_values": "pixels"}

    monkeypatch.setattr(rollout, "encode_image_for_rollout_engine", lambda img: "image" if img is image else None)
    train_ids, rollout_ids, images, train_inputs = rollout._prepare_initial_inputs(
        Sample(prompt=prompt, multimodal_inputs={"images": [image]}), processor, _TextTokenizer()
    )

    assert processor_calls == [(prompt, [image])]
    assert train_ids == [10, 11, 12]
    assert rollout_ids == ([10, 11, 12] if structured else [1, 2, 3])
    assert images == ["image"]
    assert train_inputs == {"pixel_values": "pixels"}


@pytest.mark.parametrize("apply_chat_template", [False, True])
def test_deepeyes_observation_inputs_preserve_processor_messages(monkeypatch, apply_chat_template: bool) -> None:
    rollout = pytest.importorskip("examples.deepeyes.rollout")
    from relax.utils.data import processing_utils

    message = {"role": "user", "content": "observation"}
    image = object()
    processor_calls = []

    def processor(*, text, images):
        processor_calls.append((text, images))
        return {"input_ids": [[10, 11, 12]], "attention_mask": [[1, 1, 1]], "pixel_values": "pixels"}

    def process_vision_info(messages, received_processor, *, use_audio_in_video):
        assert messages == [message]
        assert received_processor is processor
        assert use_audio_in_video is False
        return {"images": [image]}

    monkeypatch.setattr(processing_utils, "process_vision_info", process_vision_info)
    monkeypatch.setattr(rollout, "encode_image_for_rollout_engine", lambda img: "image" if img is image else None)
    train_ids, rollout_ids, images, multimodal_inputs, train_inputs = rollout._encode_observation_for_generation(
        _TextTokenizer(), processor, message, None, apply_chat_template, None
    )

    expected_text = "preamble observation" if apply_chat_template else [message]
    assert processor_calls == [(expected_text, [image])]
    assert train_ids == ([11, 12] if apply_chat_template else [10, 11, 12])
    assert rollout_ids == ([2, 3] if apply_chat_template else [10, 11, 12])
    assert images == ["image"]
    assert multimodal_inputs == {"images": [image]}
    assert train_inputs == {"pixel_values": "pixels"}
