# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Schema validation for the deterministic UltraFeedback preparation script."""

import pytest

from scripts.data.prepare_ultrafeedback_preferences import _select, _validate_row


def _row() -> dict:
    return {
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

    assert selected[0]["metadata"]["source_index"] == 1
    assert rejected == [
        {
            "source_index": 0,
            "reason_code": "schema",
            "reason": "train_prefs[0] must be an object",
        }
    ]


@pytest.mark.parametrize("extra", [{}, {"prompt_id": "reused"}, {"prompt_id": 7}, {"prompt_id": None}])
def test_select_is_deterministic_without_external_pair_ids(extra: dict):
    rows = [{**_row(), **extra} for _ in range(4)]
    selected, rejected = _select(rows, split="train_prefs", count=4)

    assert rejected == []
    assert sorted(row["metadata"]["source_index"] for row in selected) == [0, 1, 2, 3]
    assert all("prompt_id" not in row for row in selected)
    assert _select(rows, split="train_prefs", count=4) == (selected, rejected)
