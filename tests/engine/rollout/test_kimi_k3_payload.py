# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import pytest

from relax.utils.data import kimi_k3
from relax.utils.types import Sample


class Tokenizer:
    def __call__(self, text: str, **kwargs) -> dict:
        return {"input_ids": self.encode(text, **kwargs)}

    def encode(self, text: str, **kwargs) -> list[int]:
        return [900 if part == "<|media_pad|>" else ord(part) for part in self._pieces(text)]

    def _pieces(self, text: str):
        while text:
            if text.startswith("<|media_pad|>"):
                yield "<|media_pad|>"
                text = text[len("<|media_pad|>") :]
            else:
                yield text[0]
                text = text[1:]

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|media_pad|>"
        return 900


@pytest.mark.parametrize("count", [1, 2])
def test_k3_rollout_encodes_one_slot_per_image(count):
    prompt = "question" + "<|kimi_image_placeholder|>" * count
    ids = kimi_k3.encode_kimi_k3_rollout_prompt(Tokenizer(), prompt, count)
    assert ids == list(map(ord, "question")) + [900] * count


@pytest.mark.parametrize(
    "prompt,count",
    [("<|kimi_image_placeholder|>", 2), ("<|media_pad|>", 1), ("<|kimi_image_placeholder|><|media_pad|>", 1)],
)
def test_k3_rollout_rejects_ambiguous_or_missing_image_slots(prompt, count):
    with pytest.raises(ValueError):
        kimi_k3.encode_kimi_k3_rollout_prompt(Tokenizer(), prompt, count)


@pytest.mark.parametrize("is_k3,has_image", [(True, True), (True, False), (False, True)])
def test_generate_keeps_training_ids_separate_and_resumes_rollout_ids(monkeypatch, is_k3, has_image):
    pytest.importorskip("sglang_router")
    rollout = pytest.importorskip("relax.engine.rollout.sglang_rollout", exc_type=ModuleNotFoundError)

    tokenizer = Tokenizer()
    state = SimpleNamespace(tokenizer=tokenizer, processor=object(), opd_manager=None)
    monkeypatch.setattr(rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(kimi_k3, "is_kimi_k3_tokenizer", lambda tok: is_k3)
    original = "question" + ("<|kimi_image_placeholder|>" if has_image else "")
    trained_ids = [80, 900, 900, 900, 81]
    payloads = []

    async def processor(state, args, prompt, images):
        assert prompt == original
        return trained_ids.copy(), {"image_grid_thw": [[1, 2, 6]]}, 0.0

    async def encode(images):
        return {"image_data": ["encoded"]}, 0.0

    async def post(url, payload, headers=None, *, fail_fast_no_workers=False):
        payloads.append({**payload, "input_ids": payload["input_ids"].copy()})
        return {"text": "a", "meta_info": {"output_token_logprobs": [(-0.1, 97)], "finish_reason": {"type": "abort"}}}

    monkeypatch.setattr(rollout, "_run_image_processor", processor)
    monkeypatch.setattr(rollout, "_encode_multimodal_inputs", encode)
    monkeypatch.setattr(rollout, "post", post)
    args = SimpleNamespace(
        ci_test=False,
        fully_async=False,
        sglang_router_ip="localhost",
        sglang_router_port=1,
        use_rollout_routing_replay=False,
        sglang_router_policy="cache_aware",
        use_slime_router=False,
        sglang_speculative_algorithm=None,
    )
    sample = Sample(
        prompt=original, multimodal_inputs={"images": [object()] if has_image else [], "videos": [], "audio": []}
    )
    raw_ids = tokenizer.encode(original)
    expected = kimi_k3.encode_kimi_k3_rollout_prompt(tokenizer, original, 1) if is_k3 and has_image else raw_ids
    asyncio.run(rollout.generate(args, sample, {"max_new_tokens": 8}))
    assert payloads[0]["input_ids"] == expected
    assert sample.rollout_tokens == expected + [97]
    assert sample.tokens == (trained_ids if has_image else raw_ids) + [97]
    assert sample.prompt == original
    asyncio.run(rollout.generate(args, sample, {"max_new_tokens": 8}))
    assert payloads[1]["input_ids"] == expected + [97]
    assert sample.rollout_tokens == expected + [97, 97]
