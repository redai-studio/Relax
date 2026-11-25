# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exact, bounded-memory comparison across an explicitly requested native FULL
load."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import torch


def _registered(model: torch.nn.Module) -> dict:
    items = {}
    for kind, iterator in (("parameter", model.named_parameters), ("buffer", model.named_buffers)):
        for name, tensor in iterator(remove_duplicate=False):
            items[kind, name] = tensor.detach()
    return items


def _chunks(tensor: torch.Tensor, byte_limit: int) -> Iterator[bytes]:
    """Walk even a transposed or expanded view in logical order without
    materializing it whole."""
    if not tensor.numel():
        return
    if tensor.layout != torch.strided or tensor.is_quantized or tensor.device.type == "meta":
        raise ValueError("Raw installed-weight verification requires materialized strided tensors")
    if tensor.nbytes <= byte_limit:
        yield tensor.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    elif tensor.is_contiguous():
        flat = tensor.view(-1)
        for part in flat.split(max(1, byte_limit // flat.element_size())):
            yield part.view(torch.uint8).cpu().numpy().tobytes()
    else:
        row_bytes = tensor[0].nbytes
        if row_bytes > byte_limit:
            for index in range(tensor.shape[0]):
                yield from _chunks(tensor[index], byte_limit)
        else:
            for part in tensor.split(max(1, byte_limit // row_bytes), dim=0):
                yield from _chunks(part, byte_limit)


class InstalledWeightVerifier:
    """Keep the post-DELTA installation until native FULL has completed all its
    writes.

    A verification version uses one ephemeral file per scheduler. Only
    installed parameter and buffer bytes enter it. The ordinary update path has
    no snapshot.
    """

    def __init__(self, model: torch.nn.Module, *, chunk_bytes: int = 8 << 20) -> None:
        self.chunk_bytes = max(16, chunk_bytes)
        self.file = tempfile.TemporaryFile(dir=os.environ.get("RELAX_DELTA_VERIFY_TMPDIR"))
        self.schema = {}
        try:
            for key, value in _registered(model).items():
                self.schema[key] = (tuple(value.shape), value.dtype, value.device)
                for block in _chunks(value, self.chunk_bytes):
                    self.file.write(block)
            self.file.flush()
        except BaseException:
            self.close()
            raise

    def finish(self, model: torch.nn.Module) -> int:
        try:
            current = _registered(model)
            if current.keys() != self.schema.keys():
                raise ValueError("Native FULL changed the registered parameter or buffer names")
            self.file.seek(0)
            compared = 0
            for key, signature in self.schema.items():
                value = current[key]
                if (tuple(value.shape), value.dtype, value.device) != signature:
                    raise ValueError(f"Native FULL changed the installed weight schema for {key}")
                offset = 0
                for block in _chunks(value, self.chunk_bytes):
                    expected = self.file.read(len(block))
                    if expected != block:
                        first = next((i for i, (a, b) in enumerate(zip(expected, block)) if a != b), len(expected))
                        raise ValueError(f"Installed DELTA differs from native FULL at {key}, byte {offset + first}")
                    offset += len(block)
                compared += offset
            if self.file.read(1):
                raise ValueError("Incomplete installed-weight verification")
            return compared
        finally:
            self.close()

    def close(self) -> None:
        self.file.close()
