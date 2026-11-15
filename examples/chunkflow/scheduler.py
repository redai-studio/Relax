# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Segment:
    seq_id: int
    prefix: int
    length: int


@dataclass
class Chunk:
    chunk_id: int
    segments: list[Segment]
    group: int | None = None
    group_idx: int = 0

    @property
    def num_tokens(self) -> int:
        return sum(s.length for s in self.segments)

    @property
    def is_dependent(self) -> bool:
        return self.group is not None


def _bin_pack_min_bins(lengths: list[int], cap: int) -> list[list[int]]:
    if not lengths:
        return []
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    lower = max(1, math.ceil(sum(lengths) / cap))
    for num_bins in range(lower, len(lengths) + 1):
        loads = [0] * num_bins
        bins: list[list[int]] = [[] for _ in range(num_bins)]
        ok = True
        for i in order:
            best, best_load = -1, -1
            for b in range(num_bins):
                if loads[b] + lengths[i] <= cap and loads[b] > best_load:
                    best, best_load = b, loads[b]
            if best < 0:
                ok = False
                break
            loads[best] += lengths[i]
            bins[best].append(i)
        if ok:
            return [b for b in bins if b]
    return [[i] for i in range(len(lengths))]


def build_chunks(seq_lengths: Sequence[int], chunk_size: int) -> list[Chunk]:
    long_ids = [i for i, n in enumerate(seq_lengths) if n > chunk_size]
    short_ids = [i for i, n in enumerate(seq_lengths) if n <= chunk_size]

    chunks: list[Chunk] = []
    cid = 0
    for seq_id in long_ids:
        length = seq_lengths[seq_id]
        num_chunks = math.ceil(length / chunk_size)
        for i in range(num_chunks):
            start = i * chunk_size
            seg_len = min(chunk_size, length - start)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    segments=[Segment(seq_id, start, seg_len)],
                    group=seq_id,
                    group_idx=i,
                )
            )
            cid += 1

    short_lengths = [seq_lengths[i] for i in short_ids]
    for members in _bin_pack_min_bins(short_lengths, chunk_size):
        segs = [Segment(short_ids[i], 0, short_lengths[i]) for i in members]
        chunks.append(Chunk(chunk_id=cid, segments=segs))
        cid += 1

    return chunks


def group_of(chunks: Sequence[Chunk]) -> dict[int, list[Chunk]]:
    groups: dict[int, list[Chunk]] = {}
    for c in chunks:
        if c.is_dependent:
            groups.setdefault(c.group, []).append(c)
    for members in groups.values():
        members.sort(key=lambda c: c.group_idx)
    return groups
