# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.engine.sft.dataset.chat_template import render_with_loss_mask
from relax.engine.sft.dataset.raw_text import _render_raw_text_concat, raw_text_concat_enabled
from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample


class _Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    all_special_ids = [1, 2]

    def __call__(self, text):
        return {"input_ids": [1, *text.encode(), 2]}

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("raw text mode must not use the chat template")


def test_raw_text_concat_masks_target_and_removes_duplicate_boundaries(monkeypatch):
    monkeypatch.setenv("RELAX_SFT_RAW_TEXT_CONCAT", "1")
    sample = CanonicalSample(
        [CanonicalMessage("user", "q", False), CanonicalMessage("assistant", "a", True)],
        metadata={"source_dataset": "test", "row_index": 0},
    )
    tokens, mask = render_with_loss_mask(sample, tokenizer=_Tokenizer())
    assert tokens.tolist() == [1, ord("q"), ord("a"), 2]
    assert mask.tolist() == [0, 0, 1, 1]


def test_raw_text_mode_is_opt_in_and_reads_current_environment(monkeypatch):
    monkeypatch.delenv("RELAX_SFT_RAW_TEXT_CONCAT", raising=False)
    assert not raw_text_concat_enabled()
    monkeypatch.setenv("RELAX_SFT_RAW_TEXT_CONCAT", "1")
    assert raw_text_concat_enabled()
    monkeypatch.setenv("RELAX_SFT_RAW_TEXT_CONCAT", "0")
    assert not raw_text_concat_enabled()


@pytest.mark.parametrize(
    "messages,error",
    [
        ([CanonicalMessage("user", "q", False)], ValueError),
        ([CanonicalMessage("assistant", [{"type": "text", "text": "a"}], True)], TypeError),
    ],
)
def test_raw_text_rejects_unsupported_samples(messages, error):
    with pytest.raises(error):
        _render_raw_text_concat(
            CanonicalSample(messages, metadata={"source_dataset": "test", "row_index": 0}), tokenizer=_Tokenizer()
        )


def test_raw_text_keeps_tokens_when_tokenizer_has_no_special_ids():
    class BareTokenizer:
        def __call__(self, text):
            return {"input_ids": list(text.encode())}

    sample = CanonicalSample(
        [CanonicalMessage("user", "q", False), CanonicalMessage("assistant", "a", True)],
        metadata={"source_dataset": "test", "row_index": 0},
    )
    tokens, mask = _render_raw_text_concat(sample, tokenizer=BareTokenizer())
    assert tokens.tolist() == [ord("q"), ord("a")]
    assert mask.tolist() == [0, 1]


@pytest.mark.parametrize("target", ["a", "ab"])
@pytest.mark.parametrize("add_eos,shared_boundary", [(False, False), (True, False), (False, True), (True, True)])
def test_raw_text_empty_context_preserves_first_target_supervision(target, add_eos, shared_boundary):
    class Tokenizer(_Tokenizer):
        eos_token_id = 1 if shared_boundary else 2

        def __call__(self, text):
            return {"input_ids": [self.bos_token_id, *text.encode(), *([self.eos_token_id] if add_eos else [])]}

    tokenizer = Tokenizer()
    sample = CanonicalSample(
        [CanonicalMessage("user", "", False), CanonicalMessage("assistant", target, True)],
        metadata={"source_dataset": "test", "row_index": 0},
    )
    tokens, mask = _render_raw_text_concat(sample, tokenizer=tokenizer)
    expected_target = [*target.encode(), *([tokenizer.eos_token_id] if add_eos else [])]
    assert tokens.tolist() == [tokenizer.bos_token_id, *expected_target]
    assert mask.tolist() == [0, *([1] * len(expected_target))]
    # After the training loop's next-token shift, the first answer is still a label.
    assert tokens[1:][mask[1:].bool()].tolist() == expected_target


def test_raw_text_retains_target_bos_when_empty_context_has_no_tokens():
    class Tokenizer(_Tokenizer):
        def __call__(self, text):
            return {"input_ids": [self.bos_token_id, *text.encode()] if text else []}

    sample = CanonicalSample(
        [CanonicalMessage("assistant", "a", True)],
        metadata={"source_dataset": "test", "row_index": 0},
    )
    tokens, mask = _render_raw_text_concat(sample, tokenizer=Tokenizer())
    assert tokens.tolist() == [1, ord("a")]
    assert mask.tolist() == [0, 1]


def test_raw_text_keeps_special_tokens_that_are_not_bos_or_eos():
    class Tokenizer(_Tokenizer):
        all_special_ids = [1, 2, 3]

        def __call__(self, text):
            return {"input_ids": [1, ord("q"), 3] if text == "q" else [3, ord("a"), 2]}

    sample = CanonicalSample(
        [CanonicalMessage("user", "q", False), CanonicalMessage("assistant", "a", True)],
        metadata={"source_dataset": "test", "row_index": 0},
    )
    tokens, mask = _render_raw_text_concat(sample, tokenizer=Tokenizer())
    assert tokens.tolist() == [1, ord("q"), 3, 3, ord("a"), 2]
    assert mask.tolist() == [0, 0, 0, 1, 1, 1]
