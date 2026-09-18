# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Full-response repetition detection, free of training dependencies.

This module holds the detection core so that offline diagnosis works on a
plain CPU box: importing it pulls in only the standard library, never
``torch``, ``numpy`` or the Ray/metrics service stack.

:mod:`relax.utils.metrics.metric_utils` re-exports everything here, so the
online metric and the offline diagnostic share one implementation and one set
of thresholds.
"""

import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Iterator


REPETITION_WINDOW_SIZE_CHARS = 10_000
REPETITION_WINDOW_STRIDE_CHARS = 5_000
REPETITION_COMPRESSION_RATIO_THRESHOLD = 10.0


@dataclass(frozen=True)
class RepetitionWindow:
    """One scanned window and its compression ratio.

    ``start``/``end`` are character offsets into the scanned text, using Python
    slice semantics: the window is ``text[start:end]``, ``start`` is inclusive
    and ``end`` is exclusive.  Offsets count Unicode code points
    (``len(text)``), not bytes, so they stay meaningful for CJK responses.
    """

    start: int
    end: int
    compression_ratio: float

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class RepetitionReport:
    """Full-text repetition scan result for a single response."""

    text_length: int
    # Windows whose compression ratio exceeded the threshold.  Each interval
    # marks a *suspected* repetitive window, not an exact repetition boundary.
    hit_windows: list[RepetitionWindow] = field(default_factory=list)
    # How many windows the scan measured.  Only the count is kept: the ratios
    # of hit windows live in ``hit_windows`` and the largest is tracked below,
    # so retaining every ratio would hold K floats the caller never reads.
    num_windows_scanned: int = 0
    # Largest ratio seen across all scanned windows; None when nothing was
    # scanned (empty text).
    max_compression_ratio: float | None = None
    threshold: float = REPETITION_COMPRESSION_RATIO_THRESHOLD

    @property
    def has_repetition(self) -> bool:
        """Whether any scanned window exceeded the threshold."""
        return bool(self.hit_windows)

    @property
    def hit_intervals(self) -> list[tuple[int, int]]:
        """Hit windows as ``(start, end)`` character intervals."""
        return [(w.start, w.end) for w in self.hit_windows]

    @property
    def covered_chars(self) -> int:
        """Characters covered by hit windows, counting the union of the
        overlapping intervals so an overlap is never counted twice."""
        return _union_length(self.hit_intervals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_repetition": self.has_repetition,
            "text_length": self.text_length,
            "threshold": self.threshold,
            "max_compression_ratio": self.max_compression_ratio,
            "num_windows_scanned": self.num_windows_scanned,
            "hit_windows": [
                {"start": w.start, "end": w.end, "compression_ratio": w.compression_ratio} for w in self.hit_windows
            ],
            "covered_chars": self.covered_chars,
        }


def _union_length(intervals: list[tuple[int, int]]) -> int:
    """Total length of the union of half-open ``[start, end)`` intervals."""
    if not intervals:
        return 0
    total = 0
    cur_start = cur_end = None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total


def window_compression_ratio(text: str, level: int = 9) -> float:
    """UTF-8 byte length divided by its zlib-compressed length.

    Kept separate from :func:`relax.utils.metrics.metric_utils.compression_ratio`
    so this module stays standard-library only; for ``algorithm="zlib"`` and the
    same ``level`` the two agree exactly on the ratio.
    """
    raw = text.encode("utf-8")
    if not raw:
        return float("inf")
    compressed = zlib.compress(raw, level)
    if not compressed:
        return float("inf")
    return len(raw) / len(compressed)


def repetition_window_bounds(
    text_length: int,
    *,
    window_size: int = REPETITION_WINDOW_SIZE_CHARS,
    stride: int = REPETITION_WINDOW_STRIDE_CHARS,
) -> list[tuple[int, int]]:
    """Return all overlapping bounds, including short responses and the
    tail."""
    return list(_iter_window_bounds(text_length, window_size=window_size, stride=stride))


def _iter_window_bounds(text_length: int, *, window_size: int, stride: int) -> Iterator[tuple[int, int]]:
    """Generate bounds lazily so boolean detection needs no full-window
    list."""
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if stride > window_size:
        raise ValueError(f"stride ({stride}) must not exceed window_size ({window_size})")
    if text_length <= 0:
        return
    final_start = max(0, text_length - window_size)
    for start in range(0, final_start + 1, stride):
        yield start, min(start + window_size, text_length)
    if final_start % stride:
        yield final_start, text_length


def scan_repetition(
    text: str,
    *,
    window_size: int = REPETITION_WINDOW_SIZE_CHARS,
    stride: int = REPETITION_WINDOW_STRIDE_CHARS,
    threshold: float = REPETITION_COMPRESSION_RATIO_THRESHOLD,
    stop_at_first_hit: bool = False,
) -> RepetitionReport:
    """Scan the full text for repetition with overlapping windows.

    Reuses the compression-ratio criterion of the original suffix-only
    check: a window whose ratio is strictly greater than ``threshold`` is
    reported as a hit.  Set ``stop_at_first_hit`` to short-circuit once the
    boolean answer is settled, which is what the online metric wants; the
    offline diagnostic keeps it off so every window is reported. In early-exit
    mode the intervals and maximum describe only the scanned prefix. Nonempty
    short responses use one window; empty responses have no windows.
    """
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    bounds = _iter_window_bounds(len(text), window_size=window_size, stride=stride)

    scanned = 0
    max_ratio: float | None = None
    hits: list[RepetitionWindow] = []
    for start, end in bounds:
        ratio = window_compression_ratio(text[start:end])
        scanned += 1
        if max_ratio is None or ratio > max_ratio:
            max_ratio = ratio
        if ratio > threshold:
            hits.append(RepetitionWindow(start=start, end=end, compression_ratio=ratio))
            if stop_at_first_hit:
                break

    return RepetitionReport(
        text_length=len(text),
        hit_windows=hits,
        num_windows_scanned=scanned,
        max_compression_ratio=max_ratio,
        threshold=threshold,
    )


def has_repetition(text: str) -> bool:
    """Whether any window of the full text is repetitive.

    Boolean interface preserved for existing callers; the scan now covers the
    whole response instead of only its last 10,000 characters.
    """
    return scan_repetition(text, stop_at_first_hit=True).has_repetition
