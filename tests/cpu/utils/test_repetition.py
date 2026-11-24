# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
import random
import string
import zlib
from unittest.mock import patch

import pytest

from relax.utils.metrics.metric_utils import compression_ratio, detect_repetition, has_repetition


def _noise(length: int) -> str:
    return "".join(random.Random(42).choices(string.ascii_letters + string.digits, k=length))


@pytest.mark.parametrize("position", [0, 10_000, 20_000])
def test_repetition_detects_beginning_middle_and_end(position: int) -> None:
    noise = _noise(30_000)
    text = noise[:position] + "repeat! " * 1_250 + noise[position + 10_000 :]
    result = detect_repetition(text)

    assert result.has_repetition
    assert has_repetition(text) is True
    assert [(hit.start, hit.end) for hit in result.hit_windows] == [(position, position + 10_000)]
    assert result.max_compression_ratio == compression_ratio(text[position : position + 10_000])[0]
    assert result.window_count == 5
    if position < 20_000:
        assert compression_ratio(text[-10_000:])[0] < 10


def test_repetition_empty_input_has_no_infinite_ratio() -> None:
    result = detect_repetition("")
    assert not result.has_repetition
    assert result.hit_windows == ()
    assert result.window_count == result.max_compression_ratio == 0
    assert has_repetition("") is False


@pytest.mark.parametrize("text", ["a", "A short ordinary response.", _noise(30_000)])
def test_repetition_nonrepetitive_controls(text: str) -> None:
    result = detect_repetition(text)
    assert not result.has_repetition
    assert has_repetition(text) is False
    assert result.hit_windows == ()
    assert 0 < result.max_compression_ratio < 10


@pytest.mark.parametrize("length", [1_000, 10_000, 10_001, 15_000, 17_321, 20_000])
def test_repetition_short_exact_and_unaligned_windows(length: int) -> None:
    result = detect_repetition("a" * length)
    expected = {
        1_000: [(0, 1_000)],
        10_000: [(0, 10_000)],
        10_001: [(0, 10_000), (1, 10_001)],
        15_000: [(0, 10_000), (5_000, 15_000)],
        17_321: [(0, 10_000), (5_000, 15_000), (7_321, 17_321)],
        20_000: [(0, 10_000), (5_000, 15_000), (10_000, 20_000)],
    }[length]
    assert [(hit.start, hit.end) for hit in result.hit_windows] == expected
    assert result.window_count == len(expected)
    assert result.has_repetition
    assert has_repetition("a" * length) is True


def test_repetition_overlap_detects_across_window_boundary() -> None:
    noise = _noise(20_000)
    text = noise[:5_000] + "a" * 10_000 + noise[15_000:]
    assert compression_ratio(text[:10_000])[0] < 10
    assert compression_ratio(text[10_000:])[0] < 10
    assert [(hit.start, hit.end) for hit in detect_repetition(text).hit_windows] == [(5_000, 15_000)]
    assert has_repetition(text) is True


def test_repetition_unaligned_tail_is_scanned() -> None:
    text = _noise(7_321) + "a" * 10_000
    result = detect_repetition(text)
    assert [(hit.start, hit.end) for hit in result.hit_windows] == [(7_321, 17_321)]
    assert result.window_count == 3
    assert has_repetition(text) is True


def test_repetition_unicode_offsets_index_code_points() -> None:
    text = _noise(10_000) + "重复🙂e\u0301" * 2_000 + _noise(10_000)
    result = detect_repetition(text)
    assert [(hit.start, hit.end) for hit in result.hit_windows] == [(10_000, 20_000)]
    hit = result.hit_windows[0]
    assert text[hit.start : hit.end] == "重复🙂e\u0301" * 2_000
    assert len(text[hit.start : hit.end].encode("utf-8")) > hit.end - hit.start
    assert has_repetition(text) is True


def test_repetition_threshold_is_strict() -> None:
    text = "a" * 10_000
    ratio = compression_ratio(text)[0]
    equal = detect_repetition(text, threshold=ratio)
    assert not equal.has_repetition
    assert equal.max_compression_ratio == ratio
    assert detect_repetition(text, threshold=math.nextafter(ratio, 0.0)).has_repetition


def test_repetition_scans_all_windows_after_first_hit() -> None:
    text = "repeat! " * 1_250 + _noise(980_000) + "b" * 10_000
    result = detect_repetition(text)
    assert result.window_count == 199
    assert [(hit.start, hit.end) for hit in result.hit_windows] == [(0, 10_000), (990_000, 1_000_000)]
    assert result.max_compression_ratio == compression_ratio(text[-10_000:])[0]
    assert result.max_compression_ratio > result.hit_windows[0].compression_ratio


@pytest.mark.parametrize("position, expected_calls", [(0, 1), (10_000, 3), (20_000, 5), (None, 5)])
def test_repetition_boolean_stops_only_after_a_hit(position: int | None, expected_calls: int) -> None:
    text = _noise(30_000)
    if position is not None:
        text = text[:position] + "a" * 10_000 + text[position + 10_000 :]
    expected = detect_repetition(text).has_repetition

    with patch("relax.utils.repetition.zlib.compress", wraps=zlib.compress) as compress:
        assert has_repetition(text) is expected
        assert compress.call_count == expected_calls


@pytest.mark.parametrize("compressed_size, expected", [(999, True), (1_000, False), (1_001, False)])
def test_repetition_boolean_default_threshold_is_strict(compressed_size: int, expected: bool) -> None:
    with patch("relax.utils.repetition.zlib.compress", return_value=b"x" * compressed_size):
        assert has_repetition("a" * 10_000) is expected
        assert detect_repetition("a" * 10_000).has_repetition is expected


def test_repetition_early_exit_reports_only_scanned_windows() -> None:
    text = _noise(10_000) + "repeat! " * 1_250 + _noise(10_000) + "b" * 10_000
    result = detect_repetition(text, stop_after_first_hit=True)
    assert result.has_repetition
    assert [(hit.start, hit.end) for hit in result.hit_windows] == [(10_000, 20_000)]
    assert result.window_count == 3
    assert result.max_compression_ratio == compression_ratio(text[10_000:20_000])[0]
    assert result.max_compression_ratio < detect_repetition(text).max_compression_ratio


@pytest.mark.parametrize(
    "options",
    [
        {"window_size": 0},
        {"window_size": -1},
        {"stride": 0},
        {"stride": -1},
        {"stride": 10_001},
        {"threshold": 0},
        {"threshold": -1},
        {"threshold": float("inf")},
        {"threshold": float("nan")},
    ],
)
@pytest.mark.parametrize("stop_after_first_hit", [False, True])
def test_repetition_invalid_configuration_is_rejected_even_for_empty_input(
    options: dict, stop_after_first_hit: bool
) -> None:
    with pytest.raises(ValueError):
        detect_repetition("", stop_after_first_hit=stop_after_first_hit, **options)


def test_repetition_custom_window_and_stride() -> None:
    result = detect_repetition("a" * 250, window_size=100, stride=100, threshold=2)
    assert [(hit.start, hit.end) for hit in result.hit_windows] == [(0, 100), (100, 200), (150, 250)]
