# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rerank data validity, supervision boundaries and complete evaluation
coverage."""

import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.lamra_rerank.data import format_text, listwise_messages, pointwise_messages
from examples.lamra_rerank.tools.prepare_data import make_triplet
from examples.lamra_rerank.tools.summarize import recall_metrics


def test_triplet_never_labels_a_known_positive_as_negative():
    query = {"qid": "2:1", "query_txt": "a red coat", "query_modality": "text", "pos_cand_list": ["2:10", "2:11"]}
    candidates = {did: {"modality": "image", "img_path": f"{did}.jpg"} for did in ["2:10", "2:11", "2:12", "2:13"]}
    ranking = {"gt_docs": ["2:10"], "top100_docs": ["2:10", "2:11", "2:12", "2:13"]}
    original = copy.deepcopy((query, candidates, ranking))
    for seed in range(20):
        rows = make_triplet(
            query,
            candidates,
            ranking,
            {("2", "text", "image"): ["Find a matching image"]},
            Path("/images"),
            lambda text: text,
            random.Random(seed),
        )
        positive, negative, multiple = rows
        assert positive["metadata"]["positive_id"] in query["pos_cand_list"]
        assert negative["metadata"]["candidate_ids"][0] in {"2:12", "2:13"}
        ids = multiple["metadata"]["candidate_ids"]
        assert len(ids) == len(set(ids))
        answer = multiple["metadata"]["answer"]
        assert ids[answer - 1] == positive["metadata"]["positive_id"]
        assert all(did in {"2:12", "2:13"} for index, did in enumerate(ids, 1) if index != answer)
        assert len(multiple["messages"][0]["content"]) > len(positive["messages"][0]["content"])
    assert (query, candidates, ranking) == original


def test_query_without_a_true_negative_is_skipped():
    query = {"qid": "2:1", "query_txt": "coat", "query_modality": "text", "pos_cand_list": ["2:10"]}
    rows = make_triplet(
        query,
        {"2:10": {"modality": "image", "img_path": "coat.jpg"}},
        {"gt_docs": ["2:10"], "top100_docs": ["2:10"]},
        {("2", "text", "image"): ["Find an image"]},
        Path("/images"),
        lambda text: text,
        random.Random(42),
    )
    assert rows is None


def test_messages_preserve_query_and_candidate_image_order():
    query = {"txt": "Query text.", "image": "/images/query.png"}
    candidate = {"txt": "Candidate text.", "image": "/images/candidate.png"}
    messages = pointwise_messages(query, candidate)
    assert len(messages) == 1  # Evaluation never includes the answer.
    assert [part["image"] for part in messages[0]["content"] if part["type"] == "image"] == [
        "/images/query.png",
        "/images/candidate.png",
    ]
    assert pointwise_messages(query, candidate, "No")[-1]["content"][0]["text"] == "No"
    assert listwise_messages(query, [candidate, query], 2)[-1]["content"][0]["text"] == "Ans: (2)"
    assert format_text('  "red coat"\r ') == "Red coat."


def _evaluation_pairs():
    pairs = [
        {
            "index": index,
            "qid": "2:1",
            "did": f"2:{index}",
            "ret_score": 1 - index / 100,
            "relevant": index == 20,
        }
        for index in range(50)
    ]
    scores = {
        row["index"]: {"index": row["index"], "qid": row["qid"], "did": row["did"], "p_yes": float(row["relevant"])}
        for row in pairs
    }
    return pairs, scores


def test_fixed_fusion_reports_retrieval_rank_and_candidate_ceiling():
    pairs, scores = _evaluation_pairs()
    metrics = recall_metrics(pairs, scores)
    assert metrics["queries"] == 1 and metrics["pairs"] == 50
    assert metrics["recall_percent"]["retrieval"]["10"] == 0
    assert metrics["recall_percent"]["equal_fusion"]["1"] == 100
    assert metrics["recall_percent"]["rank_only"]["1"] == 100
    assert metrics["recall_percent"]["retrieval"]["50"] == 100
    assert metrics["fusion_weight"] == 1


def test_evaluation_rejects_missing_pairs_and_mismatched_identity():
    pairs, scores = _evaluation_pairs()
    with pytest.raises(ValueError, match="pair indices"):
        recall_metrics(pairs, {index: score for index, score in scores.items() if index != 17})
    scores[17]["did"] = "wrong-candidate"
    with pytest.raises(ValueError, match="identity mismatch"):
        recall_metrics(pairs, scores)


def test_reference_mask_keeps_assistant_header_and_masks_images():
    torch = pytest.importorskip("torch")
    from examples.lamra_rerank.dataset import reference_loss_mask

    tokens = torch.tensor(
        [151644, *([100] * 10), 151644, 872, 198, 151655, 151655, 151644, 77091, 198, 9454, 151645, 198]
    )
    mask = reference_loss_mask(tokens, 151644, 151643)
    assert mask[:16].sum() == 0
    assert mask[16:].tolist() == [1] * 6
    with pytest.raises(ValueError, match="chat template"):
        reference_loss_mask(tokens[1:], 151644, 151643)


@pytest.mark.parametrize("invalid_key", [None, "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_evaluation_serializes_loading_sets_and_keeps_strict_checks(monkeypatch, tmp_path, invalid_key):
    torch = pytest.importorskip("torch")
    from examples.lamra_rerank.tools import evaluate

    info = {"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(), "error_msgs": []}
    if invalid_key is not None:
        info[invalid_key] = {"decoder.weight"}

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1 if text == "Yes" else 2]}

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, messages, **kwargs):
            return "query"

        def __call__(self, **kwargs):
            return Inputs(input_ids=torch.tensor([[1]]))

    model = SimpleNamespace(
        to=lambda device: model,
        eval=lambda: model,
        generate=lambda **kwargs: SimpleNamespace(scores=[torch.tensor([[0.0, 1.0, 0.0]])]),
    )
    transformers = SimpleNamespace(
        __version__="test",
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Processor()),
        Qwen2VLForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *args, **kwargs: (model, info)),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    for key, value in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}.items():
        monkeypatch.setenv(key, value)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"index": 0, "qid": "q", "did": "d", "messages": pointwise_messages({}, {})}) + "\n")
    output = tmp_path / "scores"
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--model", str(tmp_path / "model"), "--pairs", str(pairs), "--output", str(output)],
    )
    if invalid_key is not None:
        with pytest.raises(ValueError, match="Incomplete rank checkpoint load"):
            evaluate.main()
        assert not (output / "scores_rank0.jsonl").exists()
        return

    evaluate.main()
    loading = json.loads((output / "loading_rank0.json").read_text())
    assert loading["loading_info"] == {key: [] for key in info}
    scores = json.loads((output / "scores_rank0.jsonl").read_text())
    assert scores["qid"] == "q" and scores["did"] == "d"
    assert scores["p_yes"] == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item())
    assert json.loads((output / "completion_rank0.json").read_text())["pairs"] == 1
    assert isinstance(info["missing_keys"], set)
