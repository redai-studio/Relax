# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChunkState:
    tensors: dict[str, torch.Tensor]


class StateStoreError(RuntimeError):
    pass


class StateStore:
    def __init__(self) -> None:
        self._chunks: dict[int, ChunkState] = {}
        self._next_expected_put: int = 0

    def put(self, chunk_id: int, state: ChunkState) -> None:
        if chunk_id != self._next_expected_put:
            raise StateStoreError(
                f"put() must be strictly ascending starting at 0: "
                f"expected chunk_id={self._next_expected_put}, got {chunk_id}"
            )
        self._chunks[chunk_id] = state
        self._next_expected_put += 1

    def lookup(self, upto_chunk_id: int) -> list[ChunkState]:
        missing = [i for i in range(upto_chunk_id + 1) if i not in self._chunks]
        if missing:
            raise StateStoreError(
                f"lookup(upto_chunk_id={upto_chunk_id}) requested missing/freed chunk id(s) {missing}"
            )
        return [self._chunks[i] for i in range(upto_chunk_id + 1)]

    def __len__(self) -> int:
        return len(self._chunks)
