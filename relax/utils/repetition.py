# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Full-response repetition detection without training dependencies."""

import math
import zlib
from dataclasses import dataclass
from itertools import chain


@dataclass(frozen=True)
class RepetitionWindow:
    """A suspicious window with zero-based, half-open character offsets."""

    start: int
    end: int
    compression_ratio: float


@dataclass(frozen=True)
class RepetitionResult:
    has_repetition: bool
    hit_windows: tuple[RepetitionWindow, ...]
    max_compression_ratio: float
    window_count: int


def detect_repetition(
    text: str,
    *,
    window_size: int = 10_000,
    stride: int = 5_000,
    threshold: float = 10.0,
    stop_after_first_hit: bool = False,
) -> RepetitionResult:
    """Scan windows using the UTF-8 byte compression ratio (zlib level 9).

    Offsets index Python string characters (Unicode code points), not bytes or
    tokens. Hits are suspicious windows, not exact repetition boundaries. A
    ratio strictly greater than ``threshold`` is a hit. Short responses use one
    window; an unaligned final window ends at ``len(text)``. Empty input has no
    windows and a maximum ratio of zero.

    With ``stop_after_first_hit=True``, return after the first hit. Result fields
    then describe only the scanned windows; keep the default for full diagnostics.
    """
    if window_size <= 0 or not 0 < stride <= window_size:
        raise ValueError("window_size and stride must be positive; stride must not exceed window_size")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if not text:
        return RepetitionResult(False, (), 0.0, 0)

    last_start = max(0, len(text) - window_size)
    starts = chain(range(0, last_start + 1, stride), (last_start,) if last_start % stride else ())
    hits: list[RepetitionWindow] = []
    max_ratio = 0.0
    window_count = 0
    for start in starts:
        end = min(start + window_size, len(text))
        raw = text[start:end].encode("utf-8")
        ratio = len(raw) / len(zlib.compress(raw, level=9))
        max_ratio = max(max_ratio, ratio)
        window_count += 1
        if ratio > threshold:
            hits.append(RepetitionWindow(start, end, ratio))
            if stop_after_first_hit:
                break
    return RepetitionResult(bool(hits), tuple(hits), max_ratio, window_count)
