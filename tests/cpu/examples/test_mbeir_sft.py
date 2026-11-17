# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""M-BEIR occurrence, multimedia routing and shared-reader resume contracts."""

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from examples.seq_contrast_sft.mbeir import MbeirPairDataset, _encode_mbeir_pair, retrieval_messages, select_visit
from examples.seq_contrast_sft.tools import prepare_mbeir
from relax.utils.data import processor_pool
from relax.utils.data.streaming_dataset import IndexManager
from relax.utils.training.embedding import expand_embedding_pairs


class _Processor:
    class tokenizer:
        def __new__(cls, values, **kwargs):
            return {"input_ids": [value[: kwargs["max_length"]] for value in values]}

        @staticmethod
        def batch_decode(values):
            return values

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": False}
        return json.dumps(messages)

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        assert padding is False and truncation is False and return_tensors == "pt"
        messages = json.loads(text[0])
        first_text = next(part["text"] for part in messages[0]["content"] if part["type"] == "text")
        marker = int(re.search(r"\d+", first_text).group())
        output = {
            "input_ids": torch.tensor([[marker, 99, 2]]),
            "attention_mask": torch.ones(1, 3),
            "_sft_hf_image_rebuild_supported": True,
        }
        if images:
            assert len(images) == 1 and images[0].mode == "RGB" and images[0].size == (84, 56)
            output.update(pixel_values=torch.full((4, 24), float(marker)), image_grid_thw=torch.tensor([[1, 2, 2]]))
        return output


def _row(marker, image=None):
    return {
        "native_mbeir": {
            "query": {
                "qid": "1:q",
                "query_modality": "text",
                "query_txt": str(marker),
                "query_img_path": str(image.name) if image else None,
                "pos_cand_list": ["p0"],
            },
            "positives": [{"did": "p0", "modality": "text", "txt": str(marker + 100)}],
            "instructions": [[["1", "text", "text"], ["find a match"]]],
            "image_root": str(image.parent) if image else ".",
        }
    }


def test_native_entity_prompts_preserve_whitespace_and_assistant_emb():
    for entity, suffix in [
        ({"txt": "Caption."}, "Caption.\nSummarize above sentence in one word: "),
        ({"image": "image.png"}, "\nSummarize above image in one word: "),
        ({"txt": "Caption.", "image": "image.png"}, "Caption.\nSummarize above image and sentence in one word: "),
    ]:
        messages = retrieval_messages(entity)
        assert messages[0]["content"][-1] == {"type": "text", "text": suffix}
        assert messages[1] == {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]}


def test_same_query_is_sampled_again_on_each_occurrence():
    query = {"qid": "1:q", "query_modality": "text", "query_txt": '"caption"', "pos_cand_list": ["p0", "p1"]}
    candidates = {name: {"modality": "text", "txt": name} for name in query["pos_cand_list"]}
    instructions = {("1", "text", "text"): ["find a match", "find another"]}
    choices = iter([0, 0, 1, 1])
    rng = SimpleNamespace(choice=lambda values: values[next(choices)])
    first = select_visit(query, candidates, instructions, rng)
    second = select_visit(query, candidates, instructions, rng)
    assert first["positive_id"] == "p0" and second["positive_id"] == "p1"
    assert first["instruction"] == "Find a match."
    assert second["instruction"] == "Find another."
    assert first["query_text"] == 'Find a match. "caption.'


def test_mm_pair_keeps_only_model_tensors_and_preserves_post_DP_image_order(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (93, 67)).save(image)
    monkeypatch.setattr(processor_pool, "get_worker_processor", lambda: _Processor())
    pairs = [_encode_mbeir_pair(_row(marker, image), 99, 16) for marker in (10, 20)]
    assert set(pairs[0]["query"]) == {"input_ids", "pixel_values", "image_grid_thw", "embedding_position"}
    assert set(pairs[0]["positive"]) == {"input_ids", "embedding_position"}
    expanded = expand_embedding_pairs({"multimodal_train_inputs": pairs[::-1]})
    assert [row[0].item() for row in expanded["tokens"]] == [20, 120, 10, 110]
    assert expanded["multimodal_train_inputs"][0]["pixel_values"][0, 0] == 20
    assert expanded["multimodal_train_inputs"][2]["pixel_values"][0, 0] == 10
    assert expanded["multimodal_train_inputs"][1] is None
    assert expanded["embedding_positions"] == [0] * 4
    assert "embedding_position" not in expanded["multimodal_train_inputs"][0]


def test_missing_image_and_overlength_fail_without_skipping(tmp_path, monkeypatch):
    monkeypatch.setattr(processor_pool, "get_worker_processor", lambda: _Processor())
    with pytest.raises(FileNotFoundError):
        _encode_mbeir_pair(_row(10, tmp_path / "absent.png"), 99, 16)
    with pytest.raises(ValueError, match="overlength"):
        _encode_mbeir_pair(_row(10), 99, 2)


def test_real_pair_reader_uses_default_shuffle_and_preserves_resume_cursor(tmp_path, monkeypatch):
    monkeypatch.setattr(processor_pool, "get_worker_processor", lambda: _Processor())
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(_row(index + 10)) + "\n" for index in range(8)))
    with ThreadPoolExecutor(max_workers=2) as executor:
        pool = SimpleNamespace(executor=executor)
        options = dict(emb_token_id=99, batch_size=4, max_length=16, processor_pool=pool)
        original = MbeirPairDataset(str(path), **options)
        first, _ = asyncio.run(original.get_batch_async(4))
        order = IndexManager(8)
        order.shuffle(0)
        assert order.indices != list(range(8))
        assert [sample.source_idx for sample in first] == order.indices[:4]
        expected, _ = asyncio.run(original.get_batch_async(4))
        resumed = MbeirPairDataset(str(path), **options)
        resumed.shuffle(0, position=4)
        actual, _ = asyncio.run(resumed.get_batch_async(4))
        assert (
            [sample.source_idx for sample in expected] == [sample.source_idx for sample in actual] == order.indices[4:]
        )
        for left, right in zip(expected, actual, strict=True):
            torch.testing.assert_close(left.tokens, right.tokens)
        original.index_manager.shuffle(1)
        assert original.index_manager.indices != order.indices


def test_native_instruction_loader_retains_literal_quotes(tmp_path):
    path = tmp_path / "instructions.tsv"
    path.write_text('q\tc\tu\tdataset\tprompt\ntext\timage\tx\t1\t"quoted prompt"\n')
    assert prepare_mbeir.load_instructions(path) == {("1", "text", "image"): ['"quoted prompt"']}


def test_preparation_retains_runtime_choices(tmp_path, monkeypatch):
    class Tokenizer:
        def get_vocab(self):
            return {"<emb>": 99}

        def __call__(self, texts, **kwargs):
            assert kwargs == {"truncation": True, "max_length": 480, "padding": False, "add_special_tokens": False}
            return {"input_ids": [[ord(char) for char in text[:480]] for text in texts]}

        def batch_decode(self, rows):
            return ["".join(chr(value) for value in row) for row in rows]

    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    queries = tmp_path / "queries.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    instructions = tmp_path / "instructions.tsv"
    output = tmp_path / "pairs.jsonl"
    queries.write_text(
        "".join(
            json.dumps(
                {"qid": f"1:{i}", "query_txt": "caption", "query_modality": "text", "pos_cand_list": ["p0", "p1"]}
            )
            + "\n"
            for i in range(5)
        )
    )
    candidates.write_text(
        "".join(json.dumps({"did": f"p{i}", "txt": f"candidate{i}", "modality": "text"}) + "\n" for i in range(2))
    )
    instructions.write_text("q\tc\tu\tdataset\tprompt\ntext\ttext\tx\t1\tfind a match\n")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_mbeir",
            "--queries",
            str(queries),
            "--candidates",
            str(candidates),
            "--instructions",
            str(instructions),
            "--image-root",
            str(tmp_path),
            "--tokenizer",
            str(tmp_path),
            "--output",
            str(output),
            "--batch-size",
            "4",
        ],
    )
    prepare_mbeir.main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    meta = json.loads(output.with_suffix(".meta.json").read_text())
    assert [row["metadata"]["source_index"] for row in rows] == list(range(5))
    assert len(rows) == 5 and meta["sampling"] == "native_runtime"
    assert [p["did"] for p in rows[0]["native_mbeir"]["positives"]] == ["p0", "p1"]
    assert rows[0]["native_mbeir"]["instructions"] == [[["1", "text", "text"], ["find a match"]]]
    assert meta["raw_queries"] == 5 and meta["pair_visits"] == 8 and meta["padding_visits"] == 3
    assert meta["updates"] == 2 and meta["global_batch_size"] == 4
