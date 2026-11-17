# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Host-owned packed layout; accelerator metadata is derived without device reads.

The segment and compression conventions follow Miles #4488 at
faecb81af3bf69f3684dcad69ef9e3f830036496. Unlike that implementation, physical
padding is not a real segment: Relax keeps its physical and valid ends separate.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any


def _host_ints(values: Any, name: str) -> tuple[int, ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"V4.1 THD requires {name} as host integers; accelerator metadata cannot be read here.")
    try:
        if any(not isinstance(value, Integral) or isinstance(value, bool) for value in values):
            raise TypeError
        return tuple(operator.index(value) for value in values)
    except TypeError as error:
        raise ValueError(f"V4.1 THD {name} must contain integers.") from error


@dataclass(frozen=True)
class ThdSpan:
    """The valid portion of one sample that belongs to this CP rank."""

    seq_index: int
    local_start: int
    local_end: int
    seq_start: int
    seq_length: int
    position_start: int


@dataclass(frozen=True)
class ThdLayout:
    """Immutable CPU metadata for one microbatch and one contiguous CP rank.

    Zero-valid-length segments are omitted from ``seq_starts``/``seq_lens``.
    ``sample_indices`` retains their original sample numbering. Physical gaps
    and the tail remain in the token stream but own no query or compressed key.
    The private cache contains only derived tensors and never changes the layout.
    It belongs to this microbatch, not to a model or a global cache.
    """

    seq_starts: tuple[int, ...]
    seq_lens: tuple[int, ...]
    sample_indices: tuple[int, ...]
    physical_cu_seqlens: tuple[int, ...]
    cp_rank: int
    cp_size: int
    local_length: int
    _tensor_cache: dict[tuple[str, bool], tuple[Any, Any]] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_packed_seq_params(
        cls,
        packed_seq_params: Any,
        *,
        cp_rank: int,
        cp_size: int,
        seqlen_local: int | None = None,
    ) -> ThdLayout:
        """Read Relax's host metadata, never scalar values from GPU tensors.

        The caller must select contiguous all-gather CP. Older callers without
        ``cp_partition_mode`` are accepted because the execution configuration
        owns that choice; an explicitly incompatible partition is rejected.
        """
        if getattr(packed_seq_params, "qkv_format", None) != "thd":
            raise ValueError("V4.1 THD layout requires qkv_format='thd'.")
        if cp_size < 1 or not 0 <= cp_rank < cp_size:
            raise ValueError("V4.1 THD requires a positive CP size and a rank within that group.")
        partition = getattr(packed_seq_params, "cp_partition_mode", None)
        if cp_size > 1 and partition not in (None, "contiguous"):
            raise ValueError("V4.1 THD supports only contiguous all-gather CP.")
        cu = _host_ints(getattr(packed_seq_params, "cu_seqlens_q_cpu", None), "cu_seqlens_q_cpu")
        valid = _host_ints(getattr(packed_seq_params, "valid_seqlens_q_cpu", None), "valid_seqlens_q_cpu")
        if not cu or cu[0] != 0 or len(valid) != len(cu) - 1:
            raise ValueError("V4.1 THD requires cumulative lengths starting at zero and one valid length per segment.")
        if any(end < start for start, end in zip(cu, cu[1:])):
            raise ValueError("V4.1 THD cumulative lengths must be nondecreasing.")
        if any(length < 0 or length > end - start for start, end, length in zip(cu, cu[1:], valid)):
            raise ValueError("V4.1 THD valid lengths must fit within their physical sample intervals.")

        physical = getattr(packed_seq_params, "cu_seqlens_q_padded_cpu", None)
        if physical is not None:
            physical = _host_ints(physical, "cu_seqlens_q_padded_cpu")
            if len(physical) != len(cu) or physical[:-1] != cu[:-1] or physical[-1] < cu[-1]:
                raise ValueError("V4.1 THD physical metadata may extend only the final sample interval.")
            total = physical[-1]
        elif seqlen_local is not None:
            total = seqlen_local * cp_size
            physical = (*cu[:-1], total)
        elif getattr(packed_seq_params, "cu_seqlens_q_padded", None) is None:
            # With no separate padded metadata, the host cumulative lengths
            # already include any padding-only tail emitted by get_batch.
            total, physical = cu[-1], cu
        else:
            raise ValueError("V4.1 THD needs host physical lengths to infer its CP-local token count.")
        if total < cu[-1] or total % cp_size:
            raise ValueError("V4.1 THD physical token count must cover all samples and divide evenly over CP.")
        if seqlen_local is not None and (seqlen_local < 0 or total != seqlen_local * cp_size):
            raise ValueError("V4.1 THD hidden-state length disagrees with its physical packed metadata.")
        if getattr(packed_seq_params, "num_valid_tokens", sum(valid)) != sum(valid):
            raise ValueError("V4.1 THD num_valid_tokens disagrees with the host sample lengths.")
        samples = tuple((index, start, length) for index, (start, length) in enumerate(zip(cu, valid)) if length)
        return cls(
            seq_starts=tuple(start for _, start, _ in samples),
            seq_lens=tuple(length for _, _, length in samples),
            sample_indices=tuple(index for index, _, _ in samples),
            physical_cu_seqlens=physical,
            cp_rank=cp_rank,
            cp_size=cp_size,
            local_length=total // cp_size,
        )

    @property
    def global_start(self) -> int:
        return self.cp_rank * self.local_length

    @property
    def total_tokens(self) -> int:
        return self.cp_size * self.local_length

    @property
    def num_valid_tokens(self) -> int:
        return sum(self.seq_lens)

    def spans(self) -> tuple[ThdSpan, ...]:
        result = []
        lo, hi = self.global_start, self.global_start + self.local_length
        for index, (start, length) in enumerate(zip(self.seq_starts, self.seq_lens)):
            first, last = max(start, lo), min(start + length, hi)
            if first < last:
                result.append(ThdSpan(index, first - lo, last - lo, start, length, first - start))
        return tuple(result)

    def compressed_lengths(self, ratio: int) -> tuple[int, ...]:
        if ratio < 1:
            raise ValueError("V4.1 THD compression ratio must be positive.")
        return tuple(length // ratio for length in self.seq_lens)

    def compressed_offsets(self, ratio: int) -> tuple[int, ...]:
        offsets = [0]
        for length in self.compressed_lengths(ratio):
            offsets.append(offsets[-1] + length)
        return tuple(offsets)

    def compressed_count(self, ratio: int) -> int:
        return sum(self.compressed_lengths(ratio))

    def candidate_width(self, ratio: int) -> int:
        lengths = self.compressed_lengths(ratio)
        return max((lengths[span.seq_index] for span in self.spans()), default=0)

    def _token_metadata(self, device: Any, *, global_stream: bool) -> tuple[Any, Any]:
        import torch

        device = torch.device(device)
        key = (str(device), global_stream)
        if key not in self._tensor_cache:
            length = self.total_tokens if global_stream else self.local_length
            start = 0 if global_stream else self.global_start
            rows = torch.arange(start, start + length, device=device, dtype=torch.int64)
            if self.seq_lens:
                starts = torch.tensor(self.seq_starts, device=device, dtype=torch.int64)
                lengths = torch.tensor(self.seq_lens, device=device, dtype=torch.int64)
                owners = torch.bucketize(rows, starts, right=True) - 1
                positions = rows - starts[owners.clamp_min(0)]
                valid = (owners >= 0) & (positions < lengths[owners.clamp_min(0)])
                positions = torch.where(valid, positions, 0)
            else:
                positions = torch.zeros_like(rows)
                valid = torch.zeros_like(rows, dtype=torch.bool)
            self._tensor_cache[key] = (positions, valid)
        return self._tensor_cache[key]

    def positions(self, device: Any, *, global_stream: bool = False) -> Any:
        """Sample-local positions, zero at padding rows; cached per device."""
        return self._token_metadata(device, global_stream=global_stream)[0]

    def valid_mask(self, device: Any, *, global_stream: bool = False) -> Any:
        return self._token_metadata(device, global_stream=global_stream)[1]
