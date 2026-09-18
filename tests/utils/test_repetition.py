# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Full-response repetition regressions using real compression and fixed
text."""

import hashlib
import json
import zlib
from pathlib import Path

import pytest

from relax.utils import repetition
from relax.utils.repetition import RepetitionConfig, analyze_repetition, has_repetition, iter_repetition_windows


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "repetition" / "middle_repetition.jsonl"


def _noise(length: int) -> str:
    return "".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range((length + 63) // 64))[:length]


@pytest.fixture
def middle_record() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (0, []),
        (17, [(0, 17)]),
        (10_000, [(0, 10_000)]),
        (10_001, [(0, 10_000), (1, 10_001)]),
        (15_000, [(0, 10_000), (5_000, 15_000)]),
        (22_000, [(0, 10_000), (5_000, 15_000), (10_000, 20_000), (12_000, 22_000)]),
    ],
)
def test_window_boundaries(length: int, expected: list[tuple[int, int]]) -> None:
    assert list(iter_repetition_windows(length)) == expected


@pytest.mark.parametrize("length", [100_000, 1_000_000, 1_000_003, 10_000_001])
def test_long_response_windows_never_skip_middle(length: int) -> None:
    windows = list(iter_repetition_windows(length))
    assert windows[0][0] == 0
    assert windows[-1][1] == length
    assert len(windows) == (length - 10_000 + 4_999) // 5_000 + 1
    assert len(windows) == len(set(windows))
    for previous, current in zip(windows, windows[1:]):
        assert 0 < current[0] - previous[0] <= 5_000
        assert current[0] <= previous[1]
        assert current[1] - current[0] == 10_000


def test_empty_response_is_json_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_compression(*args, **kwargs):
        pytest.fail("Empty input must not reach compression")

    monkeypatch.setattr(repetition, "compression_ratio", unexpected_compression)
    result = analyze_repetition("")
    assert result.to_dict() == {
        "has_repetition": False,
        "response_chars": 0,
        "scanned_windows": 0,
        "hits": [],
        "max_compression_ratio": None,
        "covered_chars": 0,
    }
    assert not has_repetition("")
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["max_compression_ratio"] is None


@pytest.mark.parametrize("length", [1_000, 9_999, 10_000])
def test_short_and_exact_window_responses_are_detected(length: int) -> None:
    result = analyze_repetition("a" * length)
    assert result.has_repetition
    assert result.scanned_windows == 1
    assert [(hit.start, hit.end) for hit in result.hits] == [(0, length)]


@pytest.mark.parametrize("position", ["beginning", "middle", "end"])
def test_repetition_at_every_position(position: str, middle_record: dict) -> None:
    noise = _noise(20_000)
    texts = {
        "beginning": "重复内容。" * 2_000 + noise,
        "middle": middle_record["response"],
        "end": noise + "重复内容。" * 2_000,
    }
    expected = {"beginning": (0, 10_000), "middle": (10_000, 20_000), "end": (20_000, 30_000)}
    result = analyze_repetition(texts[position])
    assert result.has_repetition
    assert expected[position] in [(hit.start, hit.end) for hit in result.hits]
    assert has_repetition(texts[position])


def test_fixed_middle_repetition_escapes_legacy_suffix_check(middle_record: dict) -> None:
    response = middle_record["response"]
    assert middle_record["sample_index"] == 6
    assert repetition.compression_ratio(response[-10_000:])[0] <= 10
    assert repetition.compression_ratio(response[:10_000])[0] <= 10
    assert analyze_repetition(response).has_repetition
    assert "<tool_observation>" in response


def test_unaligned_suffix_repetition_is_detected() -> None:
    result = analyze_repetition(_noise(12_345) + "尾" * 10_000)
    assert (12_345, 22_345) in [(hit.start, hit.end) for hit in result.hits]
    assert result.scanned_windows == 4


def test_overlap_finds_repeat_crossing_nonoverlapping_window_boundary() -> None:
    response = _noise(5_000) + "a" * 10_000 + _noise(5_000)
    assert repetition.compression_ratio(response[:10_000])[0] < 10
    assert repetition.compression_ratio(response[10_000:])[0] < 10
    assert [(hit.start, hit.end) for hit in analyze_repetition(response).hits] == [(5_000, 15_000)]


def test_unicode_offsets_are_code_points_and_compression_uses_utf8() -> None:
    text = "汉😀e\u0301" * 2_501
    result = analyze_repetition(text)
    assert result.response_chars == 10_004
    assert [(hit.start, hit.end) for hit in result.hits] == [(0, 10_000), (4, 10_004)]
    assert result.covered_chars == len(text)
    raw = text[:10_000].encode("utf-8")
    assert result.hits[0].compression_ratio == len(raw) / len(zlib.compress(raw, 9))


@pytest.mark.parametrize("length", [1, 37, 10_000, 100_000])
def test_nonrepeating_control(length: int) -> None:
    text = _noise(length)
    result = analyze_repetition(text)
    assert not result.has_repetition
    assert not has_repetition(text)
    assert result.hits == ()
    assert result.covered_chars == 0
    assert result.max_compression_ratio is not None


@pytest.mark.parametrize("ratio, expected", [(9.999, False), (10.0, False), (10.001, True)])
def test_threshold_is_strict(monkeypatch: pytest.MonkeyPatch, ratio: float, expected: bool) -> None:
    monkeypatch.setattr(repetition, "compression_ratio", lambda text: (ratio, 0.0))
    assert analyze_repetition("sample").has_repetition is expected
    assert has_repetition("sample") is expected


@pytest.mark.parametrize(
    ("ratios", "covered", "hit_starts"),
    [([11, 12, 2, 3, 19], 25, [0, 5, 20]), ([11, 12, 13, 14, 19], 30, [0, 5, 10, 15, 20])],
)
def test_complete_scan_maximum_and_union(
    monkeypatch: pytest.MonkeyPatch, ratios: list[float], covered: int, hit_starts: list[int]
) -> None:
    values = iter(ratios)
    monkeypatch.setattr(repetition, "compression_ratio", lambda text: (next(values), 0.0))
    result = analyze_repetition("x" * 30, RepetitionConfig(window_size=10, stride=5))
    assert result.scanned_windows == 5
    assert result.max_compression_ratio == 19
    assert result.covered_chars == covered
    assert [hit.start for hit in result.hits] == hit_starts
    assert [hit.compression_ratio for hit in result.hits] == [ratio for ratio in ratios if ratio > 10]
    assert json.loads(json.dumps(result.to_dict()))["hits"][0]["start"] == 0


def test_maximum_includes_nonhits(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter([1, 9, 3])
    monkeypatch.setattr(repetition, "compression_ratio", lambda text: (next(values), 0.0))
    result = analyze_repetition("x" * 20, RepetitionConfig(window_size=10, stride=5))
    assert not result.has_repetition
    assert result.max_compression_ratio == 9


@pytest.mark.parametrize(
    "ratios, expected, calls", [([11, 2, 3], True, 1), ([1, 2, 11], True, 3), ([1, 2, 3], False, 3)]
)
def test_boolean_early_exit_only_after_hit(
    monkeypatch: pytest.MonkeyPatch, ratios: list[float], expected: bool, calls: int
) -> None:
    observed = []

    def fake_compression(text: str) -> tuple[float, float]:
        observed.append(text)
        return ratios[len(observed) - 1], 0.0

    monkeypatch.setattr(repetition, "compression_ratio", fake_compression)
    assert has_repetition("x" * 20, RepetitionConfig(window_size=10, stride=5)) is expected
    assert len(observed) == calls


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 0},
        {"window_size": -1},
        {"window_size": True},
        {"window_size": 1.5},
        {"stride": 0},
        {"stride": -1},
        {"stride": True},
        {"stride": 1.5},
        {"window_size": 5, "stride": 6},
        {"threshold": 0},
        {"threshold": -1},
        {"threshold": True},
        {"threshold": "10"},
        {"threshold": float("nan")},
        {"threshold": float("inf")},
        {"threshold": float("-inf")},
    ],
)
def test_invalid_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RepetitionConfig(**kwargs)


@pytest.mark.parametrize("length", [-1, True, 1.5, "10"])
def test_invalid_window_length(length) -> None:
    with pytest.raises(ValueError):
        list(iter_repetition_windows(length))


@pytest.mark.parametrize("value", [None, b"bytes", 12, []])
def test_response_must_be_string(value) -> None:
    with pytest.raises(TypeError, match="response must be a string"):
        analyze_repetition(value)
    with pytest.raises(TypeError, match="response must be a string"):
        has_repetition(value)
