# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Batch sparse HF entries by their actual Megatron process group."""

from __future__ import annotations

import torch

from .verl_sparse_gather import gather_slot_entries_to_rank0


class _GatherQueue:
    """Hold one source batch's projections until every rank votes to start its
    data collectives."""

    def __init__(self, max_round_bytes: int, is_r0: bool, consume, traffic):
        self.max_round_bytes = max_round_bytes
        self.is_r0 = is_r0
        self._consume = consume
        self._traffic = traffic
        self._queues: dict[tuple, list] = {}

    def put(self, pg, slots: list, dtype_str: str, counts: torch.Tensor, idx: torch.Tensor, val: torch.Tensor):
        if len(slots) != counts.numel() or idx.numel() != val.numel():
            raise ValueError("Invalid sparse projection lengths")
        if (
            pg is not None
            and idx.numel()
            and self.max_round_bytes < pg.size() * (idx.element_size() + val.element_size())
        ):
            raise ValueError("Sparse gather budget cannot hold one index and value per process-group rank")
        self._queues.setdefault((pg, dtype_str), []).append((slots, counts, idx, val))

    def flush_all(self) -> None:
        for (pg, dtype_str), batch in self._queues.items():
            if pg is None:
                if self.is_r0:
                    for slots, counts, idx, val in batch:
                        off = 0
                        for (name, shape), count in zip(slots, counts.tolist(), strict=True):
                            self._consume(name, dtype_str, shape, idx[off : off + count], val[off : off + count])
                            off += count
                batch.clear()
                continue
            slots = [slot for row, *_ in batch for slot in row]
            counts = torch.cat([row[1] for row in batch])
            idx = torch.cat([row[2] for row in batch])
            val = torch.cat([row[3] for row in batch])
            batch.clear()

            def deliver(slot, indices, values):
                name, shape = slots[slot]
                self._consume(name, dtype_str, shape, indices, values)

            gather_slot_entries_to_rank0(
                idx,
                val,
                counts,
                group=pg,
                max_round_bytes=self.max_round_bytes,
                consume=deliver,
                traffic=self._traffic[pg],
            )
