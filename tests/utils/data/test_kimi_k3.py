# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""K3 segment/token boundary regressions; no model or tokenizer weights
needed."""

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from PIL import Image

from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample
from relax.utils.data import image_rebuild
from relax.utils.data.kimi_k3 import encode_kimi_k3_sft, is_kimi_k3_tokenizer, make_kimi_k3_sft_request


@dataclass
class _Segment:
    text: str
    allow_special: bool = False


def _tag(name: str, *, close: bool = False) -> list[_Segment]:
    return [_Segment("<|close|>" if close else "<|open|>", True), _Segment(name), _Segment("<|sep|>", True)]


def build_chat_segments(messages, *, thinking=True, add_generation_prompt=True, image_prompts=None, **kwargs):
    """Small protocol fixture; optional tests below exercise NVIDIA/HF's
    encoder."""
    del kwargs
    assert add_generation_prompt is False
    segments = []
    prompts = iter(image_prompts or ())
    for message in messages:
        role = message["role"]
        segments.extend(_tag(f'message role="{role}"'))
        if role == "assistant":
            if thinking:
                segments.extend(_tag("think"))
                if message.get("reasoning_content"):
                    segments.append(_Segment(message["reasoning_content"]))
                segments.extend(_tag("think", close=True))
            segments.extend(_tag("response"))
        content = message["content"]
        if isinstance(content, list):
            for part in content:
                if part["type"] in ("image", "image_url"):
                    segments.append(_Segment(next(prompts, "<|kimi_image_placeholder|>"), True))
                else:
                    segments.append(_Segment(part["text"]))
        else:
            segments.append(_Segment(content))
        if role == "assistant":
            segments.extend(_tag("response", close=True))
        segments.extend(_tag("message", close=True))
        segments.append(_Segment("<|end_of_msg|>", True))
    return segments


class _SegmentTokenizer:
    chat_template = None
    special_tokens = {
        token: 1000 + index
        for index, token in enumerate(("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>", "<|media_pad|>"))
    }

    def _encode_text_piece(self, text, *, allow_special_tokens):
        ids = []
        while text:
            special = next((token for token in self.special_tokens if text.startswith(token)), None)
            if special is not None and allow_special_tokens:
                ids.append(self.special_tokens[special])
                text = text[len(special) :]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return ids

    def convert_tokens_to_ids(self, token):
        return self.special_tokens[token]

    def apply_chat_template(self, messages, *, tokenize=False, **kwargs):
        kwargs.setdefault("thinking_effort", "max")
        segments = build_chat_segments(messages, **kwargs)
        if not tokenize:
            return "".join(segment.text for segment in segments)
        return [
            token
            for segment in segments
            for token in self._encode_text_piece(segment.text, allow_special_tokens=segment.allow_special)
        ]

    def decode(self, ids):
        inverse = {token_id: token for token, token_id in self.special_tokens.items()}
        return "".join(inverse[token_id] if token_id in inverse else chr(token_id) for token_id in ids)


def test_kimi_k3_raw_text_override_preserves_first_target_supervision(monkeypatch):
    from relax.engine.sft.dataset import chat_template

    class RawTokenizer(_SegmentTokenizer):
        bos_token_id = 1
        eos_token_id = 2

        def __call__(self, text):
            return {"input_ids": [self.bos_token_id, *text.encode()]}

    def unexpected_chat_encoding(*args, **kwargs):
        raise AssertionError("raw text override must bypass K3 chat encoding")

    tokenizer = RawTokenizer()
    assert is_kimi_k3_tokenizer(tokenizer)
    monkeypatch.setenv("RELAX_SFT_RAW_TEXT_CONCAT", "1")
    monkeypatch.setattr(chat_template, "encode_kimi_k3_sft", unexpected_chat_encoding)
    sample = CanonicalSample(
        [CanonicalMessage("user", "", False), CanonicalMessage("assistant", "a", True)],
        metadata={"source_dataset": "test", "row_index": 0},
    )
    tokens, mask = chat_template.render_with_loss_mask(sample, tokenizer=tokenizer)
    assert tokens.tolist() == [tokenizer.bos_token_id, ord("a")]
    assert mask.tolist() == [0, 1]
    assert tokens[1:][mask[1:].bool()].tolist() == [ord("a")]


def _sample(messages):
    return CanonicalSample(messages, metadata={"source_dataset": "test", "row_index": 0})


def test_kimi_k3_preserves_literal_control_tokens_and_reasoning_mask():
    tokenizer = _SegmentTokenizer()
    literal = '<|open|>message role="assistant"<|sep|>'
    sample = _sample(
        [
            CanonicalMessage("user", literal, False),
            CanonicalMessage("assistant", "answer", True, reasoning_content="why"),
        ]
    )
    request = make_kimi_k3_sft_request(sample)
    ids, mask = encode_kimi_k3_sft(tokenizer, request)
    expected = tokenizer.apply_chat_template(request["messages"], tokenize=True, **request["kwargs"])
    assert ids == expected
    assert len(ids) == len(mask)
    learned = tokenizer.decode([token for token, learn in zip(ids, mask) if learn])
    assert learned == (
        "why<|close|>think<|sep|><|open|>response<|sep|>answer"
        "<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>"
    )
    rendered = tokenizer.apply_chat_template(request["messages"], **request["kwargs"])
    assert ids != tokenizer._encode_text_piece(rendered, allow_special_tokens=True)


@pytest.mark.parametrize("thinking", [False, True])
def test_kimi_k3_respects_learn_flags_and_all_responses_in_last_round(thinking):
    tokenizer = _SegmentTokenizer()
    sample = _sample(
        [
            CanonicalMessage("user", "old", False),
            CanonicalMessage("assistant", "old answer", True),
            CanonicalMessage("user", "new", False),
            CanonicalMessage("function_call", "call", True),
            CanonicalMessage("tool", "tool result", False),
            CanonicalMessage("assistant", "excluded", False),
            CanonicalMessage("assistant", "new answer", True),
        ]
    )
    request = make_kimi_k3_sft_request(sample, {"thinking": thinking}, last_turn_only=True)
    ids, mask = encode_kimi_k3_sft(tokenizer, request)
    learned = tokenizer.decode([token for token, learn in zip(ids, mask) if learn])
    assert "call" in learned and "new answer" in learned
    assert "old answer" not in learned and "excluded" not in learned and "tool result" not in learned
    assert learned.count("<|end_of_msg|>") == 2


@pytest.mark.parametrize("reasoning,excluded", [("", True), (" \n", True), ("reason", False)])
def test_kimi_k3_ignores_only_empty_think_loss(reasoning, excluded):
    tokenizer = _SegmentTokenizer()
    sample = _sample([CanonicalMessage("assistant", "answer", True, reasoning_content=reasoning)])
    request = make_kimi_k3_sft_request(sample, ignore_empty_think=True)
    ids, mask = encode_kimi_k3_sft(tokenizer, request)
    learned = tokenizer.decode([token for token, learn in zip(ids, mask) if learn])
    assert ("think" not in learned) is excluded
    assert "answer" in learned


def test_kimi_k3_request_keeps_sample_overrides_and_omits_media_bytes():
    sample = _sample([CanonicalMessage("user", [{"type": "image_url", "image_url": "image-data"}], False)])
    sample.metadata["apply_chat_template_kwargs"] = {"thinking_effort": "low"}
    request = make_kimi_k3_sft_request(sample, {"thinking_effort": "high"})
    assert request["kwargs"]["thinking_effort"] == "low"
    assert request["kwargs"]["add_generation_prompt"] is False
    assert request["messages"][0]["content"] == [{"type": "image"}]
    assert sample.messages[0].content[0]["image_url"] == "image-data"


@pytest.mark.parametrize("option", [{"add_generation_prompt": True}, {"chat_template": "jinja"}, {"truncation": True}])
def test_kimi_k3_rejects_options_that_change_sft_alignment(option):
    with pytest.raises(ValueError):
        make_kimi_k3_sft_request(_sample([]), option)


@pytest.mark.parametrize("kind", ["videos", "audios"])
def test_kimi_k3_rejects_unsupported_media_before_loading(kind):
    sample = _sample([])
    setattr(sample, kind, ["unopened-path"])
    with pytest.raises(ValueError, match="supports images only"):
        make_kimi_k3_sft_request(sample)


def test_kimi_k3_detection_does_not_match_an_older_tiktoken_tokenizer():
    class TikTokenTokenizer:
        def apply_chat_template(self, messages):
            return messages

    # Its method has no segment encoder, despite sharing this test's globals.
    assert not is_kimi_k3_tokenizer(TikTokenTokenizer())
    assert is_kimi_k3_tokenizer(_SegmentTokenizer())


@pytest.fixture
def official_encoder(monkeypatch):
    source_dir = os.environ.get("KIMI_K3_SOURCE_DIR")
    if not source_dir:
        pytest.skip("set KIMI_K3_SOURCE_DIR to the downloaded K3 checkpoint Python sources")
    path = Path(source_dir) / "encoding_k3.py"
    spec = importlib.util.spec_from_file_location("_test_official_encoding_k3", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setitem(globals(), "build_chat_segments", module.build_chat_segments)
    return module


@pytest.mark.parametrize("thinking", [False, True])
def test_kimi_k3_official_encoder_preserves_tools_images_and_literal_tokens(official_encoder, thinking):
    del official_encoder
    tokenizer = _SegmentTokenizer()
    sample = _sample(
        [
            CanonicalMessage("system", "system", False),
            CanonicalMessage("user", [{"type": "text", "text": "<|open|>literal"}, {"type": "image"}], False),
            CanonicalMessage(
                "assistant",
                "",
                True,
                tool_calls=[{"type": "function", "function": {"name": "f", "arguments": '{"x":1e2}'}}],
                reasoning_content="consider",
            ),
            CanonicalMessage("tool", "<|end_of_msg|>literal", False),
            CanonicalMessage("assistant", "answer", True),
        ]
    )
    sample.tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    request = make_kimi_k3_sft_request(sample, {"thinking": thinking, "response_format": "json_object"})
    image_prompts = ["<|media_begin|>image 32x48<|media_content|><|media_pad|><|media_end|>"]
    ids, mask = encode_kimi_k3_sft(tokenizer, request, image_prompts=image_prompts)
    expected = tokenizer.apply_chat_template(
        request["messages"], tools=sample.tools, tokenize=True, image_prompts=image_prompts, **request["kwargs"]
    )
    assert ids == expected
    learned = tokenizer.decode([token for token, learn in zip(ids, mask) if learn])
    assert "answer" in learned and "1e2" in learned
    assert "literal" not in learned and "image 32x48" not in learned
    assert "consider" in learned if thinking else "consider" not in learned
    assert learned.count("<|end_of_msg|>") == 2


def test_kimi_k3_multimodal_worker_expands_images_and_mask_without_retokenizing(monkeypatch):
    import numpy as np
    import torch

    from relax.utils.data import processor_pool
    from relax.utils.data.kimi_k3 import KIMI_K3_SFT_LOSS_MASK, KIMI_K3_SFT_REQUEST

    class KimiK3Processor:
        tokenizer = _SegmentTokenizer()
        media_proc_cfg = {"merge_kernel_size": 2}

        def __init__(self):
            self.media_processor = self
            self.image_processor = self
            self.preprocess_calls = 0

        def __call__(self, **kwargs):
            raise AssertionError("K3 SFT must not re-tokenize the rendered text")

        def preprocess_medias(self, medias):
            return medias, [f"image {media['image'].width}x{media['image'].height}<|media_pad|>" for media in medias]

        def preprocess(self, medias, return_tensors):
            assert len(medias) == 2 and return_tensors == "pt"
            self.preprocess_calls += 1
            return {"pixel_values": torch.zeros(32, 3), "grid_thws": torch.tensor([[1, 2, 4], [1, 4, 6]])}

    processor = KimiK3Processor()
    monkeypatch.setattr(processor_pool, "_worker_processor", processor)
    monkeypatch.setattr(processor_pool, "_worker_multimodal_config", None)
    sample = _sample(
        [
            CanonicalMessage("user", [{"type": "image"}, {"type": "image"}], False),
            CanonicalMessage("assistant", "answer", True),
        ]
    )
    ids, inputs = processor_pool.process_sample_in_worker(
        "unused rendered text",
        {"images": [np.zeros((12, 8, 3), dtype=np.uint8), np.zeros((16, 10, 3), dtype=np.uint8)]},
        {KIMI_K3_SFT_REQUEST: make_kimi_k3_sft_request(sample)},
    )
    mask = inputs[KIMI_K3_SFT_LOSS_MASK]
    assert processor.preprocess_calls == 1
    assert inputs["image_grid_thw"].tolist() == [[1, 2, 4], [1, 4, 6]]
    assert "grid_thws" not in inputs
    assert len(ids) == len(mask)
    pad_id = processor.tokenizer.convert_tokens_to_ids("<|media_pad|>")
    assert ids.count(pad_id) == 8
    assert all(mask[index] == 0 for index, token in enumerate(ids) if token == pad_id)
    assert "image 8x12" in processor.tokenizer.decode(ids)
    assert "image 10x16" in processor.tokenizer.decode(ids)
    assert "answer" in processor.tokenizer.decode([token for token, learn in zip(ids, mask) if learn])
    assert inputs["pixel_values"].dtype == torch.bfloat16
    assert all(tensor.is_shared() for tensor in inputs.values())


class _RebuildProcessor:
    """Pixel-only K3 processor fake for the rank-side rebuild worker."""

    media_proc_cfg = {"merge_kernel_size": 2}

    def __init__(self):
        self.media_processor = self
        self.pixel_values = torch.zeros(32, 3)
        self.grid_thws = torch.tensor([[1, 2, 4], [1, 4, 6]])

    def preprocess_medias(self, medias):
        return medias, [f"image {media['image'].width}x{media['image'].height}<|media_pad|>" for media in medias]

    def preprocess(self, medias, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": self.pixel_values, "grid_thws": self.grid_thws}


def _rebuild_descriptor(**overrides):
    descriptor = {
        "version": 1,
        "kind": "kimi_k3_sft_image_v1",
        "image_refs": ["/data/a.jpg", "/data/b.jpg"],
        "pixel_shape": [32, 3],
        "image_grid_thw": [[1, 2, 4], [1, 4, 6]],
    }
    descriptor.update(overrides)
    return descriptor


def test_kimi_k3_multimodal_official_tokenizer_ids_match_chat_template(official_encoder, monkeypatch, tmp_path):
    import base64

    from tokenizers import AddedToken

    # A byte vocabulary exercises the real checkpoint tokenizer without
    # downloading model weights or depending on the production BPE vocabulary.
    vocab = tmp_path / "tiktoken.model"
    vocab.write_text("".join(f"{base64.b64encode(bytes([i])).decode()} {i}\n" for i in range(256)))
    monkeypatch.setitem(sys.modules, "encoding_k3", official_encoder)
    source_path = Path(os.environ["KIMI_K3_SOURCE_DIR"]) / "tokenization_kimi.py"
    spec = importlib.util.spec_from_file_location("_test_official_tokenization_kimi", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    specials = [
        "[BOS]",
        "[EOS]",
        "[UNK]",
        "[PAD]",
        "<|open|>",
        "<|close|>",
        "<|sep|>",
        "<|end_of_msg|>",
        "<|media_pad|>",
    ]
    tokenizer = module.TikTokenTokenizer(
        str(vocab),
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=specials[4:],
        added_tokens_decoder={256 + i: AddedToken(token, special=True) for i, token in enumerate(specials)},
    )
    sample = _sample(
        [
            CanonicalMessage("user", [{"type": "text", "text": "图片 <|open|> literal"}, {"type": "image"}], False),
            CanonicalMessage("assistant", "答案", True, reasoning_content="推理"),
        ]
    )
    request = make_kimi_k3_sft_request(sample)
    image_prompts = ["image 28x56<|media_pad|>"]
    ids, mask = encode_kimi_k3_sft(tokenizer, request, image_prompts=image_prompts)
    assert ids == tokenizer.apply_chat_template(
        request["messages"], tokenize=True, image_prompts=image_prompts, **request["kwargs"]
    )
    learned = tokenizer.decode([token for token, learn in zip(ids, mask) if learn])
    assert "推理" in learned and "答案" in learned
    assert "图片" not in learned and "literal" not in learned


@pytest.mark.parametrize("use_async", [False, True])
def test_kimi_k3_multimodal_streaming_consumes_expanded_mask(monkeypatch, tmp_path, use_async):
    import asyncio

    import torch

    from relax.engine.sft.dataset import streaming
    from relax.utils.data.kimi_k3 import KIMI_K3_SFT_LOSS_MASK, KIMI_K3_SFT_REQUEST

    path = tmp_path / "sft.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "<image>Describe."},
            {"role": "assistant", "content": "answer"},
        ],
        "images": ["unopened.png"],
    }
    path.write_text(json.dumps(row) + "\n")
    dataset = streaming.SFTStreamingDataset(
        str(path),
        tokenizer=_SegmentTokenizer(),
        prompt_key="messages",
        multimodal_keys={"image": "images"},
        prefetch_max_cached=0,
    )

    def preprocess(sample, *, processor_pool, rendered_text, processor_kwargs):
        del processor_pool
        assert sample.images == ["unopened.png"]
        assert rendered_text == ""
        assert processor_kwargs[KIMI_K3_SFT_REQUEST]["kwargs"]["add_generation_prompt"] is False
        # Intentionally incompatible with the unexpanded token sequence: K3
        # image dimension strings require the worker's newly computed mask.
        return [17, 18, 19], {
            "pixel_values": torch.zeros(1, 3),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            KIMI_K3_SFT_LOSS_MASK: torch.tensor([0, 0, 1]),
        }

    async def preprocess_async(*args, **kwargs):
        return preprocess(*args, **kwargs)

    monkeypatch.setattr(streaming, "preprocess_multimodal", preprocess)
    monkeypatch.setattr(streaming, "preprocess_multimodal_async", preprocess_async)
    try:
        sample = asyncio.run(dataset._finalize_async(dataset._render_one(0))) if use_async else dataset._process_one(0)
        assert sample.tokens.tolist() == [17, 18, 19]
        assert sample.loss_mask.tolist() == [0, 0, 1]
        assert KIMI_K3_SFT_LOSS_MASK not in sample.multimodal_train_inputs
    finally:
        dataset.stop()


@pytest.mark.parametrize(
    ("strategy", "capacity", "expected"),
    [
        ("truncate_right", 3, None),
        ("truncate_left", 3, None),
        ("truncate_right", 5, [17, 18, 1004, 1004, 19]),
        ("truncate_left", 5, [18, 1004, 1004, 19, 20]),
    ],
)
def test_kimi_k3_multimodal_truncation_keeps_all_image_slots(tmp_path, strategy, capacity, expected):
    import torch

    from relax.engine.sft.dataset.streaming import SFTStreamingDataset, _RenderedSample
    from relax.utils.data.kimi_k3 import KIMI_K3_SFT_LOSS_MASK

    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps({"input": [{"role": "assistant", "content": "answer"}]}) + "\n")
    dataset = SFTStreamingDataset(
        str(path),
        tokenizer=_SegmentTokenizer(),
        prefetch_max_cached=0,
        oversize_strategy=strategy,
        capacity=capacity,
    )
    rendered = _RenderedSample(
        idx=0,
        sample=_sample([CanonicalMessage("assistant", "answer", True)]),
        short_ids=torch.tensor([1]),
        short_mask=torch.tensor([1]),
        rendered_text=None,
        total_length=1,
        classification_label=None,
    )
    inputs = {
        "pixel_values": torch.zeros(8, 3),
        "image_grid_thw": torch.tensor([[1, 2, 4]]),
        KIMI_K3_SFT_LOSS_MASK: torch.tensor([0, 0, 0, 0, 1, 1]),
    }
    try:
        if expected is None:
            with pytest.raises(ValueError, match="truncation changed the image token count"):
                dataset._build_processed(rendered, [17, 18, 1004, 1004, 19, 20], inputs)
        else:
            processed = dataset._build_processed(rendered, [17, 18, 1004, 1004, 19, 20], inputs)
            assert processed.tokens.tolist() == expected
            assert len(processed.loss_mask) == capacity
    finally:
        dataset.stop()


def test_kimi_k3_streaming_preserves_reasoning_content():
    from relax.engine.sft.dataset.streaming import _canonicalize_messages

    messages = _canonicalize_messages(
        [{"role": "assistant", "content": "answer", "reasoning_content": "reason"}], require_response=True
    )
    assert messages[0].reasoning_content == "reason"
    ids, mask = encode_kimi_k3_sft(_SegmentTokenizer(), make_kimi_k3_sft_request(_sample(messages)))
    learned = _SegmentTokenizer().decode([token for token, learn in zip(ids, mask) if learn])
    assert "reason" in learned and "answer" in learned


def test_kimi_k3_rank_side_rebuild_matches_producer_contract(monkeypatch):
    from concurrent.futures import Future

    from relax.utils.data import processor_pool
    from relax.utils.data.kimi_k3 import build_kimi_k3_image_features

    processor = _RebuildProcessor()
    monkeypatch.setattr(processor_pool, "_worker_processor", processor)
    monkeypatch.setattr(image_rebuild, "_worker_threads_limited", True)
    # The worker must normalize images exactly like the producer IPC path
    # (numpy round-trip) so the processor sees identical inputs.
    seen_arrays = []

    def fake_load_image(ref):
        seen_arrays.append(ref)
        return Image.new("RGB", (8, 12)) if ref == "/data/a.jpg" else Image.new("RGB", (10, 16))

    monkeypatch.setattr(image_rebuild, "load_image", fake_load_image)

    features, timings = image_rebuild.build_image_features_in_worker(_rebuild_descriptor())

    assert seen_arrays == ["/data/a.jpg", "/data/b.jpg"]
    assert features["pixel_values"].dtype == torch.bfloat16
    assert features["pixel_values"].is_shared()
    assert features["image_grid_thw"].tolist() == [[1, 2, 4], [1, 4, 6]]
    assert set(timings) == {"read_s", "process_s"}
    # The pixel-only rebuild shares the producer's pipeline.
    shared = build_kimi_k3_image_features(processor, [Image.new("RGB", (8, 12))])
    assert shared["pixel_values"].dtype == torch.bfloat16

    class _StubPool:
        def __init__(self):
            self.executor = self

        def submit(self, fn, *args):
            future: Future = Future()
            future.set_result(fn(*args))
            return future

    batch, _ = image_rebuild.build_batch_image_features(_StubPool(), [None, _rebuild_descriptor()])
    assert batch[0] is None
    assert batch[1]["image_grid_thw"].tolist() == [[1, 2, 4], [1, 4, 6]]


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "unknown_model_v1"},
        {"pixel_shape": [8, 3]},
        {"image_grid_thw": [[1, 2, 4]]},
    ],
)
def test_kimi_k3_rank_side_rebuild_fails_on_descriptor_mismatch(monkeypatch, overrides):
    from relax.utils.data import processor_pool

    monkeypatch.setattr(processor_pool, "_worker_processor", _RebuildProcessor())
    monkeypatch.setattr(image_rebuild, "_worker_threads_limited", True)
    monkeypatch.setattr(image_rebuild, "load_image", lambda ref: Image.new("RGB", (8, 12)))

    with pytest.raises(RuntimeError):
        image_rebuild.build_image_features_in_worker(_rebuild_descriptor(**overrides))


def test_kimi_k3_rank_side_rebuild_resizes_like_producer(monkeypatch):
    class _SizedProcessor(_RebuildProcessor):
        def __init__(self):
            super().__init__()
            self.image_processor = self
            self.patch_size = 14
            self.received_sizes = []

        def preprocess(self, medias, return_tensors):
            self.received_sizes = [media["image"].size for media in medias]
            return {"pixel_values": self.pixel_values, "grid_thws": self.grid_thws}

    from relax.utils.data import processor_pool
    from relax.utils.multimodal.config import MultimodalConfig
    from relax.utils.multimodal.image_utils import fetch_image

    processor = _SizedProcessor()
    monkeypatch.setattr(processor_pool, "_worker_processor", processor)
    monkeypatch.setattr(image_rebuild, "_worker_threads_limited", True)
    monkeypatch.setattr(processor_pool, "_worker_multimodal_config", MultimodalConfig(image_max_token_num=4))
    monkeypatch.setattr(image_rebuild, "load_image", lambda ref: Image.new("RGB", (200, 100)))

    features, _ = image_rebuild.build_image_features_in_worker(_rebuild_descriptor())

    expected = fetch_image(
        {"image": Image.new("RGB", (200, 100))},
        image_patch_size=14,
        config=MultimodalConfig(image_max_token_num=4),
    )
    assert processor.received_sizes == [expected.size, expected.size]
    assert features["image_grid_thw"].tolist() == [[1, 2, 4], [1, 4, 6]]
