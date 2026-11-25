# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Project native row-streamed HF slots without assembling the embedding
table."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .shard_snapshot import CudaShardInputs
from .verl_sparse_gather import changed_words, shard_delta_indices


class UnsupportedStreamedLayout(ValueError):
    """The native HF stream is valid but its sparse row geometry cannot be
    established."""


@dataclass(frozen=True)
class RowWindow:
    name: str
    shape: tuple[int, int]
    owner: int
    local_start: int
    slot_start: int
    length: int

    def read(self, shards: Mapping, key: str, expected_shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
        # Borrow only the selected rows, including when the acknowledged version is pinned on the host.
        tensor = (shards.shards if isinstance(shards, CudaShardInputs) else shards)[key].detach()
        if tensor.shape != expected_shape or tensor.dtype != dtype:
            raise ValueError(f"Local streamed snapshot geometry changed for {key}")
        width = self.shape[1]
        row, col = divmod(self.local_start, width)
        if col or self.length < width:
            part = tensor[row, col : col + self.length]
        else:
            part = tensor[row : row + self.length // width]
        return part.contiguous().view(-1).to(device=device, non_blocking=True)


@dataclass(frozen=True)
class StreamedRowTable:
    slots: tuple[tuple[str, tuple[int, int]], ...]
    local_shape: tuple[int, int]
    tp_group: dist.ProcessGroup
    tp_size: int
    tp_rank: int

    @classmethod
    def for_mapping(cls, mapping, param, schema: dict, tp_group) -> StreamedRowTable | None:
        if not callable(getattr(mapping, "gather_shard", None)):
            return None
        try:
            from megatron.bridge.models.qwen.qwen4_exp_bridge import PLENGramEmbeddingMapping
        except ImportError as exc:
            raise UnsupportedStreamedLayout(f"Unrecognized streamed mapping {type(mapping).__qualname__}") from exc

        from .bridge_converter import _resolve_hf_name_against_checkpoint

        # Only this native mapping defines the row partition and SUM contract below.
        if type(mapping) is not PLENGramEmbeddingMapping:
            raise UnsupportedStreamedLayout(f"Unrecognized streamed mapping {type(mapping).__qualname__}")
        names = [
            _resolve_hf_name_against_checkpoint(mapping.hf_param[f"shard_{i}"], schema)
            for i in range(mapping.num_shards)
        ]
        if not names or len(set(names)) != len(names) or any(name not in schema for name in names):
            raise UnsupportedStreamedLayout("Streamed HF shard names disagree with the native FULL directory")
        slots = tuple((name, schema[name][0]) for name in names)
        if any(len(shape) != 2 or min(shape) <= 0 or shape[0] * shape[1] >= 2**31 for _, shape in slots):
            raise UnsupportedStreamedLayout(
                "Streamed HF slots require two dimensions and representable int32 coordinates"
            )
        tp_size, tp_rank = dist.get_world_size(group=tp_group), dist.get_rank(group=tp_group)
        native_group = mapping.tp_group
        if (
            mapping.tp_size != tp_size
            or mapping.tp_rank != tp_rank
            or (
                tp_size > 1
                and (
                    native_group is None
                    or dist.get_process_group_ranks(native_group) != dist.get_process_group_ranks(tp_group)
                )
            )
        ):
            raise UnsupportedStreamedLayout("Native streamed mapping and source TP ownership differ")
        total_rows = sum(shape[0] for _, shape in slots)
        shard_rows = (total_rows + len(slots) - 1) // len(slots)
        width = slots[0][1][1]
        if total_rows % tp_size or any(
            shape != (min(shard_rows, total_rows - i * shard_rows), width) for i, (_, shape) in enumerate(slots)
        ):
            raise UnsupportedStreamedLayout("Native FULL slots disagree with the streamed TP row partition")
        local_shape = (total_rows // tp_size, width)
        dtype = schema[names[0]][1]
        if not dtype.is_floating_point or any(schema[name][1] != dtype for name in names):
            raise UnsupportedStreamedLayout("Streamed HF slots require one native floating-point dtype")
        if param is not None and (tuple(param.shape) != local_shape or param.dtype != dtype):
            raise UnsupportedStreamedLayout("Local streamed rows disagree with native FULL shape or dtype")
        return cls(slots, local_shape, native_group if native_group is not None else tp_group, tp_size, tp_rank)

    def windows(self, limit_bytes: int, itemsize: int, group_size: int) -> Iterator[RowWindow]:
        # Reserve count exchange/control storage, transient int64 indices and typed pairs,
        # including the padded sparse gather on the publisher when only one rank has data.
        control_bytes = 16 + 12 * group_size
        bytes_per_word = max(48 + 8 * itemsize, 32 + (4 + itemsize) * (group_size + 2))
        max_words = (limit_bytes - control_bytes) // bytes_per_word
        if max_words < 1:
            raise ValueError("Streamed projection budget cannot hold one changed word")
        owner_size = self.local_shape[0] * self.local_shape[1]
        global_start = 0
        for name, shape in self.slots:
            size, width = shape[0] * shape[1], shape[1]
            start = 0
            while start < size:
                owner, local_start = divmod(global_start + start, owner_size)
                # Keep either full rows or a single partial row so strided sources can be sliced before copying.
                width_limit = max_words // width * width if width <= max_words else width - local_start % width
                length = min(size - start, owner_size - local_start, max_words, width_limit)
                yield RowWindow(name, shape, owner, local_start, start, length)
                start += length
            global_start += size

    def change_counts(
        self,
        windows: Sequence[RowWindow],
        current: Mapping,
        previous: Mapping,
        key: str | None,
        dtype: torch.dtype,
        device: torch.device,
        contributes: bool,
        control_device: torch.device,
    ) -> torch.Tensor:
        counts = torch.zeros(len(windows), dtype=torch.int32, device=device)
        if contributes:
            for i, window in enumerate(windows):
                if self.tp_rank == window.owner:
                    new = window.read(current, key, self.local_shape, dtype, device)
                    old = window.read(previous, key, self.local_shape, dtype, device)
                    counts[i] = torch.count_nonzero(changed_words(new, old))
        return counts.to(control_device)

    def local_changes(
        self,
        window: RowWindow,
        current: Mapping,
        previous: Mapping,
        key: str | None,
        dtype: torch.dtype,
        device: torch.device,
        contributes: bool,
        expected_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if contributes and self.tp_rank == window.owner:
            new = window.read(current, key, self.local_shape, dtype, device)
            old = window.read(previous, key, self.local_shape, dtype, device)
            indices, values = shard_delta_indices(new, old, 0)
            if indices.numel() != expected_count:
                raise ValueError("Streamed source changed while its publication held the weights")
            previous_values = old[indices]
        else:
            indices = torch.empty(0, dtype=torch.int64, device=device)
            values = previous_values = torch.empty(0, dtype=dtype, device=device)
        return indices, values, previous_values

    def native_entry(
        self,
        window: RowWindow,
        changes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        participates: bool,
        expected_count: int,
        reserve: Callable[[], AbstractContextManager],
    ) -> tuple[list, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        indices, new, old = changes
        payload_bytes = 0
        # The native HF shard is assembled by summing its owner's rows with zeros.
        # Use the same operator/backend on both versions, including signed zeros.
        n = expected_count if self.tp_size > 1 and participates else 0
        # Reach a common allocation verdict before the native TP data collective.
        with reserve():
            if n:
                pair = new.new_zeros((2, n))
                if self.tp_rank == window.owner:
                    pair[0].copy_(new)
                    pair[1].copy_(old)
        if n:
            dist.all_reduce(pair, op=dist.ReduceOp.SUM, group=self.tp_group)
            payload_bytes = pair.nbytes
            if self.tp_rank == window.owner:
                new, old = pair.unbind()
        if indices.numel():
            keep = (changed_words(new, old) & ~torch.isnan(new)).nonzero(as_tuple=True)[0]
            indices, new = indices[keep].add_(window.slot_start).int(), new[keep]
        else:
            indices = indices.int()
        return [(window.name, window.shape)], torch.tensor([indices.numel()]), indices, new, payload_bytes
