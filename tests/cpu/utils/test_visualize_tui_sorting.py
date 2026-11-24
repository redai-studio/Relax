# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.utils.visualize.tui import (
    _NO_SORT_VALUE,
    _numeric_fields,
    _parse_numeric_value,
    _sort_options,
    _sort_value_to_field,
)


def test_tui_numeric_value_parser_accepts_jsonl_number_strings():
    assert _parse_numeric_value("12") == 12.0
    assert _parse_numeric_value("-3.5") == -3.5
    assert _parse_numeric_value(7) == 7.0

    assert _parse_numeric_value("hello") is None
    assert _parse_numeric_value('{"score": 1}') is None
    assert _parse_numeric_value("[1, 2]") is None
    assert _parse_numeric_value(True) is None


def test_tui_sort_options_are_built_from_numeric_fields():
    samples = [
        {"reward": "1.0", "prompt": "question", "agent_turns": "3", "metadata": '{"x": 1}', "__IDX": 0},
        {"reward": "0.0", "prompt": "answer", "image_token_count": "128", "__IDX": 1},
    ]

    assert _numeric_fields(samples) == ["reward", "agent_turns", "image_token_count"]
    assert _sort_options(samples) == [
        ("no sort", _NO_SORT_VALUE),
        ("reward asc", "reward:asc"),
        ("reward desc", "reward:desc"),
        ("agent_turns asc", "agent_turns:asc"),
        ("agent_turns desc", "agent_turns:desc"),
        ("image_token_count asc", "image_token_count:asc"),
        ("image_token_count desc", "image_token_count:desc"),
    ]
    assert _sort_value_to_field("image_token_count:desc") == ("image_token_count", True)
    assert _sort_value_to_field(_NO_SORT_VALUE) is None
