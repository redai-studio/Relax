# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Schema validation for the deterministic UltraFeedback preparation script."""

import pytest

from scripts.data.prepare_ultrafeedback_preferences import _select, _validate_row


def _row() -> dict:
    return {
        "prompt_id": "pair-1",
        "chosen": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "good"},
        ],
        "rejected": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "bad"},
        ],
    }


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ("not-a-message", "message object"),
        ({"content": "question"}, "invalid role"),
        ({"role": "invalid", "content": "question"}, "invalid role"),
        ({"role": "user", "content": 123}, "content must be a string"),
    ],
)
@pytest.mark.parametrize("branch", ["chosen", "rejected"])
def test_validate_row_rejects_invalid_message_schema(replacement, match: str, branch: str):
    row = _row()
    row[branch][0] = replacement

    with pytest.raises(ValueError, match=match):
        _validate_row(row, split="train_prefs", index=0)


def test_select_classifies_non_object_rows_as_schema_rejections():
    selected, rejected = _select(["not-an-object", _row()], split="train_prefs", count=1)

    assert selected[0]["prompt_id"] == "pair-1"
    assert rejected == [
        {
            "prompt_id": None,
            "source_index": 0,
            "reason_code": "schema",
            "reason": "train_prefs[0] must be an object",
        }
    ]
