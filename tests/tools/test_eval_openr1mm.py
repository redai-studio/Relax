# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import base64
import importlib.util
import io
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[2] / "scripts/tools/eval_openr1mm.py"
_SPEC = importlib.util.spec_from_file_location("openr1mm_eval", _PATH)
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)


def test_pass_at_k_uses_all_samples_and_correct_combinations() -> None:
    assert [evaluation.pass_at_k(16, 1, k) for k in (1, 4, 8, 16)] == [1 / 16, 1 / 4, 1 / 2, 1]
    assert evaluation.pass_at_k(16, 0, 16) == 0
    assert evaluation.pass_at_k(16, 16, 1) == 1
    with pytest.raises(ValueError):
        evaluation.pass_at_k(8, 1, 16)


def test_messages_remove_reference_and_replace_image_marker() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Before<image>After"},
            {"role": "assistant", "content": "SECRET REASONING<answer>42</answer>"},
        ],
        "images": ["data:image/png;base64,abc"],
    }
    messages, gold = evaluation.make_messages(row)
    assert gold == "42"
    assert "SECRET" not in str(messages)
    assert "42" not in str(messages)
    assert [x["type"] for x in messages[-1]["content"]] == ["text", "image_url", "text"]
    row["images"] = []
    with pytest.raises(ValueError, match="Image markers"):
        evaluation.make_messages(row)


@pytest.mark.parametrize(
    "pred,gold,correct",
    [
        ("<answer>\nB\n</answer>", "The correct answer is 151°, which corresponds to option B.", True),
        ("<answer>25%</answer>", "2013, 2014, and 2015 with a percentage above 25%", False),
        ("<answer>2</answer>", "The answer is Total surface area is 2(A_triangle) + 3(A_rectangle)", False),
        ("<answer>1/2</answer>", "0.5", True),
        (r"<answer>18\pi</answer>", "18", False),
        (r"<answer>18\pi</answer>", r"The volume is \( 18\pi \) cubic units.", True),
        ("<answer>A</answer><answer>B</answer>", "B", False),
        ("The answer is B", "B", False),
        ("<answer>2</answer>", "3", False),
    ],
)
def test_score_only_matches_complete_final_answers(pred: str, gold: str, correct: bool) -> None:
    assert evaluation.score_answer(pred, gold)["correct"] is correct


def test_summary_does_not_treat_missing_requests_as_complete() -> None:
    record = {"score": {"correct": True, "method": "exact"}, "finish_reason": "stop", "content": "A", "usage": {}}
    records = {(10, i): record for i in range(16)}
    records[(20, 0)] = record
    result = evaluation.summarize(records, [10, 20], 16, [1, 4, 8, 16])
    assert result["status"] == "partial"
    assert result["completed_questions"] == 1
    assert result["completed_requests"] == 17
    assert result["pass_at_k"] == {str(k): 1 for k in (1, 4, 8, 16)}


def test_scoring_error_is_not_a_wrong_model_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    import math_verify

    def fail(*args, **kwargs):
        raise RuntimeError("scorer infrastructure failed")

    monkeypatch.setattr(math_verify, "parse", fail)
    with pytest.raises(RuntimeError, match="infrastructure"):
        evaluation.score_answer("<answer>0.5</answer>", "1/2")


def test_image_preprocessing_caps_tokens_and_preserves_aspect_ratio() -> None:
    from PIL import Image

    image = Image.new("RGB", (2000, 1000))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    row, metadata = evaluation.prepare_images({"images": [url]}, 1024)
    width, height = metadata[0]["width"], metadata[0]["height"]
    assert width % 28 == height % 28 == 0
    assert width * height <= 1024 * 28 * 28
    assert abs(width / height - 2) < 0.1
    assert row["images"][0].startswith("data:image/png;base64,")
