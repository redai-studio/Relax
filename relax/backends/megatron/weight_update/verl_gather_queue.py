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


def _prodshape(shape) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n


class _GatherQueue:
    """Per-gather-group batching of slot-keyed queue entries
    ``(slots, dtype_str, counts, idx, val)``. Entries carry FINAL-coordinate
    payloads (identity specs: one slot = the param itself; converter specs: the
    spec's hf_slots), so rank 0 never converts -- ``consume`` receives assembled
    per-slot pieces straight from the gather.

    One queue per ProcessGroup: separate queues stop pg alternation (dense fsdp
    group vs expert world group per layer) from shattering batches. The flush
    trigger is COUNT-ONLY: entry counts are identical on every rank while byte
    totals are not, so a count trigger is the only one that keeps the collective
    sequence identical across ranks (a per-rank byte trigger desyncs the gathers
    and deadlocks NCCL). Byte bounding happens INSIDE the batched gather via
    ``max_round_bytes``, decided from the all-gathered counts every rank sees."""

    __slots__ = ("batch_k", "max_round_bytes", "is_r0", "_consume", "_queues")

    def __init__(self, batch_k: int, max_round_bytes: int, is_r0: bool, consume):
        self.batch_k = max(int(batch_k), 1)
        self.max_round_bytes = int(max_round_bytes)
        self.is_r0 = is_r0
        self._consume = consume
        self._queues: dict[tuple[int, str], tuple] = {}

    def put(self, pg, slots: list, dtype_str: str, counts: torch.Tensor, idx: torch.Tensor, val: torch.Tensor):
        _pg, entries = self._queues.setdefault((id(pg), dtype_str), (pg, []))
        entries.append((slots, dtype_str, counts, idx, val))
        if len(entries) >= self.batch_k:
            self._flush(pg, entries)

    def flush_all(self) -> None:
        for pg, entries in self._queues.values():
            self._flush(pg, entries)

    def _flush(self, pg, entries: list) -> None:
        """One gather round for one group's queue."""
        if not entries:
            return
        batch = list(entries)
        entries.clear()
        if pg is None:
            # unsharded/replicated params: rank 0's local delta already is global
            if self.is_r0:
                for slots, dtype_str, counts, idx, val in batch:
                    off = 0
                    for (name, shape), c in zip(slots, counts.tolist(), strict=True):
                        self._consume(
                            name, dtype_str, tuple(shape), _prodshape(shape), idx[off : off + c], val[off : off + c]
                        )
                        off += c
            return
        dev = batch[0][3].device
        counts_concat = torch.cat([c for _, _, c, _, _ in batch]).to(dev)
        idx_concat = torch.cat([i for _, _, _, i, _ in batch])
        val_concat = torch.cat([v for _, _, _, _, v in batch])
        gathered = gather_slot_entries_to_rank0(
            idx_concat, val_concat, counts_concat, group=pg, max_round_bytes=self.max_round_bytes
        )
        if self.is_r0 and gathered is not None:
            slot_i = 0
            for slots, dtype_str, _counts, _i, _v in batch:
                for name, shape in slots:
                    aidx, aval = gathered[slot_i]
                    slot_i += 1
                    self._consume(name, dtype_str, tuple(shape), _prodshape(shape), aidx, aval)
