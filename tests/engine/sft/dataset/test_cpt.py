# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU contracts for CPT encoding, batching and per-document supervision."""

import asyncio
import json

import pytest
import torch

from relax.engine.sft.dataset.streaming import SFTStreamingDataset, pack_samples_for_tq
from relax.utils.sft_utils import align_loss_mask_for_sft, compute_sft_response_chunk


class TextTokenizer:
    eos_token_id = 0
    eos_token = "\u0000"

    def encode(self, text, *, add_special_tokens=True):
        assert add_special_tokens is True
        return [ord(char) for char in text]

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("CPT must never invoke the chat template")


def make_dataset(tmp_path, rows, **kwargs):
    path = tmp_path / "text.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    options = {"tokenizer": TextTokenizer(), "training_mode": "cpt", "prefetch_max_cached": 0}
    options.update(kwargs)
    return SFTStreamingDataset(str(path), **options)


def test_text_and_swift_assistant_format_preserve_raw_content(tmp_path):
    text = "  原文\n<think>内容</think>\n\n "
    ds = make_dataset(tmp_path, [{"text": text}, {"messages": [{"role": "assistant", "content": text}]}])
    samples = ds.get_batch_in_order(0, 2)
    expected = [ord(char) for char in text] + [0]
    for sample in samples:
        assert sample.tokens.tolist() == expected
        assert sample.loss_mask.tolist() == [1] * len(expected)
        assert sample.multimodal_train_inputs is None


def test_auto_input_accepts_mixed_text_and_swift_assistant_rows(tmp_path):
    text = "<think></think>answer"
    ds = make_dataset(
        tmp_path,
        [{"text": text}, {"messages": [{"role": "assistant", "content": text}]}],
        prompt_key="auto",
        tokenizer=QwenTextTokenizer(),
        cpt_template="qwen3_5",
    )
    text_sample, messages_sample = ds.get_batch_in_order(0, 2)
    assert torch.equal(text_sample.tokens, messages_sample.tokens)
    assert torch.equal(text_sample.loss_mask, messages_sample.loss_mask)


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"text": "one", "messages": [{"role": "assistant", "content": "two"}]},
    ],
)
def test_auto_input_requires_exactly_one_supported_schema(tmp_path, row):
    ds = make_dataset(tmp_path, [row], prompt_key="auto")
    with pytest.raises(ValueError, match="requires exactly one of 'text' or 'messages'"):
        ds.get_batch_in_order(0, 1)


def test_yaml_mixture_applies_weights_to_text_and_messages(tmp_path):
    text_path = tmp_path / "text.jsonl"
    text_path.write_text("".join(json.dumps({"text": f"a{i}"}) + "\n" for i in range(3)))
    messages_path = tmp_path / "messages.jsonl"
    messages_path.write_text(json.dumps({"messages": [{"role": "assistant", "content": "b0"}]}) + "\n")
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(
        "datasets:\n"
        "  text:\n"
        "    path: text.jsonl\n"
        "    weight: 3\n"
        "  messages:\n"
        "    path: messages.jsonl\n"
        "    weight: 1\n"
    )

    dataset = SFTStreamingDataset(
        str(config_path),
        tokenizer=TextTokenizer(),
        training_mode="cpt",
        prompt_key="auto",
        prefetch_max_cached=0,
    )
    samples = dataset.get_batch_in_order(0, len(dataset))

    assert dataset.reader.names == ["text", "messages"]
    assert dataset.reader.quotas == [3, 1]
    assert [chr(sample.tokens[0]) for sample in samples] == ["a", "a", "a", "b"]
    batch = list(dataset.reader.iter_batch([3, 2, 0, 1]))
    assert [idx for idx, _ in batch] == [3, 2, 0, 1]
    assert [row.get("text", row.get("messages", [{}])[0].get("content"))[0] for _, row in batch] == [
        "b",
        "a",
        "a",
        "a",
    ]


def test_yaml_mixture_rejects_partial_weights(tmp_path):
    path = tmp_path / "text.jsonl"
    path.write_text(json.dumps({"text": "value"}) + "\n")
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text("datasets:\n  weighted:\n    path: text.jsonl\n    weight: 1\n  missing: text.jsonl\n")

    with pytest.raises(ValueError, match="set every weight or none"):
        SFTStreamingDataset(str(config_path), prefetch_max_cached=0)


def test_explicit_text_column_and_existing_eos(tmp_path):
    ds = make_dataset(tmp_path, [{"body": "abc\u0000", "text": "wrong"}], prompt_key="body")
    assert ds.get_batch_in_order(0, 1)[0].tokens.tolist() == [97, 98, 99, 0]


def test_tokenizer_bos_is_preserved(tmp_path):
    class BosTokenizer(TextTokenizer):
        def encode(self, text, *, add_special_tokens=True):
            return [9] + super().encode(text, add_special_tokens=add_special_tokens)

    ds = make_dataset(tmp_path, [{"text": "abc"}], tokenizer=BosTokenizer())
    assert ds.get_batch_in_order(0, 1)[0].tokens.tolist() == [9, 97, 98, 99, 0]


@pytest.mark.parametrize(
    "row",
    [
        {"text": ""},
        {"text": " \n "},
        {"text": 123},
        {"text": "\u0000"},
        {"messages": [{"role": "user", "content": "question"}]},
        {"messages": [{"role": "assistant", "content": "a", "loss": False}]},
        {"messages": [{"role": "assistant", "content": "a", "loss_scale": 0.5}]},
        {"messages": [{"role": "assistant", "content": [{"type": "image"}]}]},
        {"text": "caption", "images": ["image.png"]},
        {"text": "tools", "tools": [{"name": "tool"}]},
    ],
)
def test_invalid_cpt_rows_fail_before_training(tmp_path, row):
    ds = make_dataset(tmp_path, [row])
    with pytest.raises(ValueError, match="CPT"):
        ds.get_batch_in_order(0, 1)


def test_explicit_missing_column_does_not_fall_back(tmp_path):
    ds = make_dataset(tmp_path, [{"text": "valid"}], prompt_key="typo")
    with pytest.raises(ValueError, match="missing text column 'typo'"):
        ds.get_batch_in_order(0, 1)


def test_tokenizer_without_eos_is_rejected(tmp_path):
    tokenizer = TextTokenizer()
    tokenizer.eos_token_id = None
    ds = make_dataset(tmp_path, [{"text": "abc"}], tokenizer=tokenizer)
    with pytest.raises(ValueError, match="eos_token_id"):
        ds.get_batch_in_order(0, 1)


@pytest.mark.parametrize(
    "options",
    [
        {"task_type": "seq_cls"},
        {"label_key": "answer"},
        {"system_prompt": "assistant"},
        {"loss_last_turn_only": True},
        {"loss_ignore_empty_think": True},
        {"apply_chat_template_kwargs": {"enable_thinking": False}},
        {"capacity": 1},
        {"oversize_strategy": "custom", "oversize_custom_fn": lambda **kwargs: None},
    ],
)
def test_cpt_rejects_incompatible_dataset_options(tmp_path, options):
    with pytest.raises(ValueError, match="CPT"):
        make_dataset(tmp_path, [{"text": "abc"}], **options)


@pytest.mark.parametrize("strategy, expected", [("truncate_left", [100, 101, 0]), ("truncate_right", [97, 98, 99])])
def test_cpt_truncation_keeps_full_supervision(tmp_path, strategy, expected):
    ds = make_dataset(tmp_path, [{"text": "abcde"}], capacity=3, oversize_strategy=strategy)
    sample = ds.get_batch_in_order(0, 1)[0]
    assert sample.tokens.tolist() == expected
    assert align_loss_mask_for_sft(sample.loss_mask).tolist() == [1, 1, 0]


def test_packed_documents_never_train_the_next_document_boundary(tmp_path):
    ds = make_dataset(tmp_path, [{"text": "abc"}, {"text": "de"}])
    samples = ds.get_batch_in_order(0, 2)
    batch = pack_samples_for_tq(samples)
    assert batch["response_lengths"] == batch["total_lengths"] == [4, 3]
    masks = torch.cat([align_loss_mask_for_sft(mask) for mask in batch["loss_masks"]])
    assert masks.tolist() == [1, 1, 1, 0, 1, 1, 0]
    logits = torch.randn(7, 128)
    offset = 0
    targets = []
    for sample in samples:
        _, labels = compute_sft_response_chunk(logits, sample.tokens, offset, offset + sample.total_length)
        targets.extend(labels.tolist())
        offset += sample.total_length
    assert targets == [98, 99, 0, 0, 101, 0, 0]


@pytest.mark.parametrize("prefetch, use_async", [(0, False), (0, True), (8, False), (8, True)])
def test_batch_paths_and_resume_share_cpt_semantics(tmp_path, prefetch, use_async):
    rows = [{"text": f"document {i}"} for i in range(8)]
    ds = make_dataset(tmp_path, rows, prefetch_max_cached=prefetch, prefetch_chunk_size=2, seed=17)
    baseline = make_dataset(tmp_path, rows, seed=17)
    try:
        for dataset in (ds, baseline):
            dataset.restrict_training_indices((0, 2, 4, 6))
            dataset.shuffle(1, position=2)
        expected, _ = baseline.get_batch(4)
        actual, _ = asyncio.run(ds.get_batch_async(4)) if use_async else ds.get_batch(4)
        assert [s.source_idx for s in actual] == [s.source_idx for s in expected]
        for left, right in zip(actual, expected, strict=True):
            assert torch.equal(left.tokens, right.tokens)
            assert torch.equal(left.loss_mask, right.loss_mask)
            assert left.source_idx in (0, 2, 4, 6)
    finally:
        ds.stop()
        baseline.stop()


class QwenTextTokenizer(TextTokenizer):
    def encode(self, text, *, add_special_tokens=True):
        assert add_special_tokens is False
        return [ord(char) for char in text]


@pytest.mark.parametrize(
    "text, expected, supervised",
    [
        ("  正文\n\n", "正文\u0000", "文\u0000"),
        ("<think>推理</think>答案", "<think>\n推理\n</think>\n\n答案\u0000", "think>\n推理\n</think>\n\n答案\u0000"),
        (
            "prefix<think> a </think>  answer",
            "<think>\na\n</think>\n\n  answer\u0000",
            "think>\na\n</think>\n\n  answer\u0000",
        ),
        ("<think> \t</think> \u3000答案", "<think>\n\n</think>\n\n \u3000答案\u0000", "答案\u0000"),
        ("</think> 答案", "</think> 答案\u0000", "答案\u0000"),
        # Preserve the upstream re-split/re-match behavior, including masking
        # ANSWER when a second empty block starts the remainder.
        ("<think></think><think></think>ANSWER", "<think>\n\n</think>\n\n<think></think>ANSWER\u0000", "\u0000"),
        ("正文\u0000", "正文\u0000", "文\u0000"),
        ("正文<|endoftext|>", "正文<|endoftext|>", "文<|endoftext|>"),
    ],
)
def test_qwen_cpt_normalization_and_supervised_targets(tmp_path, text, expected, supervised):
    ds = make_dataset(tmp_path, [{"text": text}], tokenizer=QwenTextTokenizer(), cpt_template="qwen3_5")
    sample = ds.get_batch_in_order(0, 1)[0]
    ids = sample.tokens.tolist()
    assert "".join(map(chr, ids)) == expected
    assert (
        "".join(chr(token) for token, mask in zip(ids, sample.loss_mask.tolist(), strict=True) if mask) == supervised
    )
    assert align_loss_mask_for_sft(sample.loss_mask)[-1] == 0


def test_qwen_cpt_splits_mask_regions_before_tokenization(tmp_path):
    class BoundaryTokenizer(QwenTextTokenizer):
        def encode(self, text, *, add_special_tokens=True):
            assert add_special_tokens is False
            return {
                "<think>\n\n</think>\n\n ": [10, 11],
                "answer": [20],
            }[text]

    ds = make_dataset(
        tmp_path,
        [{"text": "<think></think> answer"}],
        tokenizer=BoundaryTokenizer(),
        cpt_template="qwen3_5",
    )
    sample = ds.get_batch_in_order(0, 1)[0]
    assert sample.tokens.tolist() == [10, 11, 20, 0]
    assert sample.loss_mask.tolist() == [0, 0, 1, 1]


def test_qwen_cpt_requires_explicit_cpt_mode(tmp_path):
    with pytest.raises(ValueError, match="requires training_mode=cpt"):
        make_dataset(tmp_path, [{"text": "abc"}], training_mode="sft", cpt_template="qwen3_5")


def test_qwen_cpt_explicit_unit_weight_overrides_empty_think_mask(tmp_path):
    ds = make_dataset(
        tmp_path,
        [{"messages": [{"role": "assistant", "content": "<think></think>answer", "loss_scale": 1}]}],
        tokenizer=QwenTextTokenizer(),
        cpt_template="qwen3_5",
    )
    sample = ds.get_batch_in_order(0, 1)[0]
    assert sample.loss_mask.tolist() == [0] + [1] * (sample.total_length - 1)


@pytest.mark.parametrize("strategy, expected", [("truncate_left", [65, 102, 0]), ("truncate_right", [97, 98, 65])])
def test_qwen_cpt_truncation_preserves_native_placeholders(tmp_path, strategy, expected):
    class PlaceholderTokenizer(QwenTextTokenizer):
        def convert_tokens_to_ids(self, name):
            return 65 if name == "<|image_pad|>" else 66

    ds = make_dataset(
        tmp_path,
        [{"text": "abcAdef\u0000"}],
        tokenizer=PlaceholderTokenizer(),
        cpt_template="qwen3_5",
        capacity=3,
        oversize_strategy=strategy,
    )
    sample = ds.get_batch_in_order(0, 1)[0]
    assert sample.tokens.tolist() == expected
    assert sample.loss_mask.tolist() == [0, 1, 1]
