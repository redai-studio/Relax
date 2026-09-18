# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compression-based diagnostics on unmodified response strings.

Offsets are zero-based Python string indices (Unicode code points), with an
exclusive end. A hit identifies a suspicious window, not a repetition's exact
boundaries. Compression measures UTF-8 bytes, while window sizes measure code
points. Tool observations are deliberately treated like any other text.
"""

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Literal


DEFAULT_WINDOW_SIZE = 10_000
DEFAULT_STRIDE = 5_000
DEFAULT_THRESHOLD = 10.0


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    """Return original/compressed byte ratio and percentage saved.

    The historical empty-input result is retained for compatibility; detectors
    do not send empty windows to this function.
    """
    raw = data.encode(encoding) if isinstance(data, str) else data
    original = len(raw)
    if original == 0:
        return float("inf"), 0.0
    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0
    return original / comp_len, 100.0 * (1.0 - comp_len / original)


@dataclass(frozen=True)
class RepetitionConfig:
    window_size: int = DEFAULT_WINDOW_SIZE
    stride: int = DEFAULT_STRIDE
    threshold: float = DEFAULT_THRESHOLD

    def __post_init__(self) -> None:
        for name in ("window_size", "stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.stride > self.window_size:
            raise ValueError("stride must not exceed window_size: every character must be covered")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold)
            or self.threshold <= 0
        ):
            raise ValueError("threshold must be a positive finite number")


@dataclass(frozen=True)
class RepetitionWindow:
    start: int
    end: int
    compression_ratio: float


@dataclass(frozen=True)
class RepetitionResult:
    has_repetition: bool
    response_chars: int
    scanned_windows: int
    hits: tuple[RepetitionWindow, ...]
    max_compression_ratio: float | None
    covered_chars: int

    def to_dict(self) -> dict:
        """Return JSON-ready data without the response text."""
        result = asdict(self)
        result["hits"] = list(result["hits"])
        return result


def iter_repetition_windows(length: int, config: RepetitionConfig = RepetitionConfig()) -> Iterator[tuple[int, int]]:
    """Cover every character, including a full suffix window when unaligned.

    Nonempty short responses produce one window; empty responses produce none.
    There is no window cap or adaptive stride, regardless of response length.
    """
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if length == 0:
        return
    if length <= config.window_size:
        yield 0, length
        return
    last_start = length - config.window_size
    for start in range(0, last_start + 1, config.stride):
        yield start, start + config.window_size
    if last_start % config.stride:
        yield last_start, length


def _scan_windows(text: str, config: RepetitionConfig) -> Iterator[RepetitionWindow]:
    if not isinstance(text, str):
        raise TypeError("response must be a string")
    for start, end in iter_repetition_windows(len(text), config):
        ratio, _ = compression_ratio(text[start:end])
        yield RepetitionWindow(start, end, ratio)


def analyze_repetition(text: str, config: RepetitionConfig = RepetitionConfig()) -> RepetitionResult:
    """Scan all windows, retaining all hits and the maximum over all windows.

    Covered characters are the union of suspicious windows, not an estimate of
    the number of repeated characters. Empty text has no maximum (JSON null).
    """
    hits = []
    maximum = None
    count = covered = covered_end = 0
    for window in _scan_windows(text, config):
        count += 1
        maximum = window.compression_ratio if maximum is None else max(maximum, window.compression_ratio)
        if window.compression_ratio > config.threshold:
            hits.append(window)
            covered += window.end - max(window.start, covered_end)
            covered_end = window.end
    return RepetitionResult(bool(hits), len(text), count, tuple(hits), maximum, covered)


def has_repetition(text: str, config: RepetitionConfig = RepetitionConfig()) -> bool:
    """Return whether any window exceeds the threshold, stopping on a hit.

    Unlike detailed diagnostics, this boolean interface need not scan beyond a
    proven hit. A negative result always scans the entire response.
    """
    return any(window.compression_ratio > config.threshold for window in _scan_windows(text, config))
