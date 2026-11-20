# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from relax.engine.rewards.textpecker_reward import (
    TextPeckerEvalScorer,
    TextPeckerRewardScorer,
    extract_targets,
    ocr_sentence_reward,
    recognized_text,
    sentence_reward,
)


def test_sentence_reward_reference_gates_and_structural_markers():
    assert sentence_reward("你好", ["你好"], 0.0) == pytest.approx(1.0)
    assert sentence_reward("你好", ["你好"], 1.5) == pytest.approx(0.3)
    assert sentence_reward("你好", ["你好"], 0.4) == pytest.approx(0.72)
    assert sentence_reward("你好你好", ["你好"], 0.0) == pytest.approx(np.exp(-0.5))
    assert sentence_reward("你<#>", ["你好"], 0.0) == pytest.approx(0.35)
    assert sentence_reward("Buy One, Get!", ["Buy One Get"], 0.0) == pytest.approx(1.0)
    assert sentence_reward("buy one get", ["Buy One Get"], 0.0) < 1.0


@pytest.mark.parametrize("distance", [-0.1, float("nan"), float("inf")])
def test_invalid_lpips_is_not_a_neutral_reward(distance):
    with pytest.raises(ValueError):
        sentence_reward("target", ["target"], distance)


def test_targets_metadata_and_chinese_prompt_fallback():
    assert extract_targets({"metadata": {"target_texts": '["Hello", "你好"]'}}) == ["Hello", "你好"]
    assert extract_targets({"prompt": "把文字替换成“世界”"}) == ["世界"]
    assert extract_targets({"metadata": {"target_texts": ["first"]}, "prompt": '替换成"second"'}) == ["first"]


def test_recognition_parse_keeps_markers_and_normalizes_like_reference():
    assert recognized_text('```json\n{"recognized_text":"你<#> 好"}\n```') == "你<#> 好"
    assert recognized_text('{"recognized_text":""}') == ""
    with pytest.raises(ValueError):
        recognized_text("service failed")


def test_memory_and_file_candidates_use_gt_not_source(tmp_path):
    gt = tmp_path / "gt.png"
    generated = tmp_path / "gen.png"
    Image.new("RGB", (9, 7), "red").save(gt)
    Image.new("RGB", (4, 3), "blue").save(generated)
    scorer = TextPeckerRewardScorer(SimpleNamespace(reward_endpoint="http://test/v1"))
    request = {
        "metadata": {"gt_path": str(gt), "target_texts": ["你好"]},
        "outputs": [{"track": "image", "uri": str(generated)}],
    }
    image, target, texts = scorer._read_candidate(request)
    assert image.size == (4, 3) and target.getpixel((0, 0)) == (255, 0, 0) and texts == ["你好"]
    request["outputs"][0]["uri"] = "memory://candidate"
    request["memory_images"] = {"memory://candidate": np.zeros((5, 6, 3), dtype=np.uint8)}
    assert scorer._read_candidate(request)[0].size == (6, 5)


def test_batch_order_failures_and_real_zero_have_distinct_validity(monkeypatch, tmp_path):
    path = tmp_path / "gt.png"
    Image.new("RGB", (4, 4)).save(path)
    scorer = TextPeckerRewardScorer(SimpleNamespace(reward_endpoint="http://test/v1"))
    scorer._client = object()
    monkeypatch.setattr(scorer, "_complete", lambda _: '{"recognized_text":"你好"}')
    monkeypatch.setattr(scorer, "_lpips_distance", lambda *_: 0.0)

    def request(text):
        return {
            "metadata": {"gt_path": str(path), "target_texts": [text]},
            "outputs": [{"track": "image", "uri": str(path)}],
        }

    result = scorer.score_batch([request("你好"), {}, request("世界")])
    assert result == {"textpecker_lpips": [1.0, 0.0, 0.0], "__valid_mask__": [True, False, True]}
    monkeypatch.setattr(scorer, "_complete", lambda _: "unparseable")
    assert scorer.score_batch([request("你好")])["__valid_mask__"] == [False]
    assert scorer.score_batch([]) == {"textpecker_lpips": [], "__valid_mask__": []}


def test_wire_uses_reference_jpeg_prompt_and_decoding_settings():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"recognized_text":"x"}'))])

    scorer = TextPeckerRewardScorer(SimpleNamespace(reward_endpoint="http://test/v1"))
    scorer._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    scorer._complete(Image.new("RGB", (2, 2)))
    call = calls[0]
    assert call["model"] == "TextPecker" and call["temperature"] == 0.0 and call["max_tokens"] == 2048
    assert call["timeout"] == 120.0 and call["extra_body"] == {"repetition_penalty": 1.2}
    assert call["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("recovers", [True, False])
def test_malformed_success_response_retries_with_bounded_budget(monkeypatch, recovers):
    replies = iter(["{}", '{"wrong_key":"x"}', '{"recognized_text":"ok"}' if recovers else "{}"])
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))])

    monkeypatch.setattr("relax.engine.rewards.textpecker_reward.time.sleep", lambda _seconds: None)
    scorer = TextPeckerRewardScorer(SimpleNamespace(reward_endpoint="http://test/v1"))
    scorer._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(scorer, "_read_candidate", lambda _: (Image.new("RGB", (2, 2)), None, ["ok"]))
    monkeypatch.setattr(scorer, "_lpips_distance", lambda *_: 0.0)
    result = scorer.score_batch([{"metadata": {"sample_id": "retry-test"}}])
    assert len(calls) == 3
    assert result == {"textpecker_lpips": [1.0 if recovers else 0.0], "__valid_mask__": [recovers]}


def test_missing_offline_backbone_fails_before_download(monkeypatch, tmp_path):
    import torch

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    scorer = TextPeckerRewardScorer(SimpleNamespace(reward_endpoint="http://test/v1"))
    with pytest.raises(FileNotFoundError, match="Provision"):
        scorer.onload()


def test_eval_ocr_metric_omits_both_training_gates():
    assert ocr_sentence_reward("你好你好", ["你好"]) == 1.0
    assert sentence_reward("你好你好", ["你好"], 0.4) < 1.0
    assert ocr_sentence_reward("你<#>", ["你好"]) == pytest.approx(0.35)
    with pytest.raises(ValueError, match="target texts"):
        ocr_sentence_reward("你好", [])


def test_eval_without_gt_never_loads_or_calls_lpips(monkeypatch, tmp_path):
    path = tmp_path / "generated.png"
    Image.new("RGB", (4, 4)).save(path)
    scorer = TextPeckerEvalScorer(SimpleNamespace(reward_endpoint="http://test/v1"))
    scorer._client = object()
    monkeypatch.setattr(scorer, "_complete", lambda _: '{"recognized_text":"你好你好"}')

    def fail_lpips(*_args):
        raise AssertionError("Evaluation must not use LPIPS")

    monkeypatch.setattr(scorer, "_lpips_distance", fail_lpips)
    scorer.onload()
    assert scorer._model is None
    request = {
        "metadata": {"target_texts": ["你好"], "gt_path": "irrelevant-missing-gt.png"},
        "outputs": [{"track": "image", "uri": str(path)}],
    }
    assert scorer.score_batch([request, {}]) == {"textpecker_ocr": [1.0, 0.0], "__valid_mask__": [True, False]}
