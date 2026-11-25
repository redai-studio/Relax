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
"""Apply verl delta_sharded flushes inside the SGLang scheduler."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from itertools import chain

import torch


CHUNK_BYTES = 512 << 20


def apply_delta(model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Decode one sparse delta flush and masked-apply it onto ``model`` in
    place."""
    tensors = dict(named_tensors)
    spec = json.loads(bytes(tensors["__delta_spec__"].cpu().numpy().tobytes()).decode())
    values = tensors["__values__"]
    positions = tensors.get("__positions__")
    if positions is None:  # values-only flush (the seed) carries no positions
        positions = torch.empty(0, dtype=torch.uint8, device=values.device)

    apply_flush(model, spec, positions, values)


def apply_flush(
    model: torch.nn.Module,
    spec: dict,
    positions: torch.Tensor,
    values: torch.Tensor,
    *,
    load_weights: Callable | None = None,
) -> None:
    """Apply a parsed verl manifest over the proxy's two device-local IPC
    views."""
    from relax.distributed.weight_sync.verl_codec import checksum as _checksum

    got = _checksum(positions, values)
    if got != int(spec["checksum"]):
        raise RuntimeError(
            f"delta checksum mismatch in sglang loader: got {got}, expected {spec['checksum']}; "
            "indicates corruption between sender encode and receiver apply"
        )

    if load_weights is None:
        load_weights = model.load_weights
    if spec["encoding"] == "dense":
        _apply_dense(load_weights, spec["params"], values)
        return

    encoding = spec["encoding"]
    with _masked_copy(model):
        chunk: list[tuple[str, torch.Tensor]] = []
        chunk_bytes = 0
        for p in spec["params"]:
            t = _decode_one(encoding, positions, values, p)
            nbytes = t.numel() * t.element_size()
            if chunk and chunk_bytes + nbytes > CHUNK_BYTES:
                load_weights(chunk)
                chunk, chunk_bytes = [], 0
            chunk.append((p["name"], t))
            chunk_bytes += nbytes
        if chunk:
            load_weights(chunk)


def _apply_dense(load_weights: Callable, params: list[dict], values: torch.Tensor) -> None:
    """Apply a dense (full-coverage) flush: plain chunked load, no masking
    needed."""
    chunk: list[tuple[str, torch.Tensor]] = []
    chunk_bytes = 0
    for p in params:
        dtype = getattr(torch, p["dtype"])
        t = values[p["val_start"] : p["val_end"]].to(dtype).view(p["shape"])
        nbytes = t.numel() * t.element_size()
        if chunk and chunk_bytes + nbytes > CHUNK_BYTES:
            load_weights(chunk)
            chunk, chunk_bytes = [], 0
        chunk.append((p["name"], t))
        chunk_bytes += nbytes
    if chunk:
        load_weights(chunk)


def _decode_one(encoding: str, positions: torch.Tensor, values: torch.Tensor, p: dict) -> torch.Tensor:
    """Densify one parameter using positions relative to its own allocation."""
    numel = math.prod(p["shape"])
    dtype = getattr(torch, p["dtype"])
    flat = torch.full((numel,), float("nan"), dtype=dtype, device=values.device)
    vals = values[p["val_start"] : p["val_end"]]
    if vals.numel() == 0:
        return flat.view(p["shape"])

    pos_b = positions[p["pos_start"] : p["pos_end"]]
    if encoding != "indices":
        raise ValueError(f"unsupported delta encoding: {encoding!r}")

    indices = pos_b.view(torch.int32)
    if flat.is_cuda and torch.version.cuda is not None:
        flat.scatter_(0, indices, vals.to(dtype))
    else:
        flat.index_copy_(0, indices.long(), vals.to(dtype))
    return flat.view(p["shape"])


@contextmanager
def _masked_copy(model: torch.nn.Module) -> Iterator[None]:
    """Preserve installed words at NaN positions during native sparse loads."""
    orig_copy = torch.Tensor.copy_
    orig_setitem = torch.Tensor.__setitem__
    # A loader may assemble its input in a temporary tensor before installing
    # it. Preserve placeholders in that tensor until a resident weight is written.
    installed = {t.untyped_storage() for t in chain(model.parameters(), model.buffers()) if t.layout == torch.strided}
    cuda_copy = None
    if torch.version.cuda is not None and any(storage.device.type == "cuda" for storage in installed):
        from torch.utils._triton import has_triton

        if has_triton():
            from relax.backends.sglang.verl_masked_copy_kernel import copy_present

            cuda_copy = copy_present

    def writes_weight(target: torch.Tensor, source: object) -> bool:
        return (
            isinstance(source, torch.Tensor)
            and source.is_floating_point()
            and target.layout == torch.strided
            and target.untyped_storage() in installed
        )

    def merge(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        # Keep the mask on-device. Native loaders can broadcast a source over a
        # destination slice, so use the original destination for every NaN.
        source = source.to(destination.device)
        return torch.where(torch.isnan(source), destination, source.to(destination.dtype))

    def masked_copy_(self: torch.Tensor, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if writes_weight(self, src):
            if cuda_copy is not None and not args and not kwargs and cuda_copy(self, src):
                return self
            src = merge(self, src)
        return orig_copy(self, src, *args, **kwargs)

    def masked_setitem(self: torch.Tensor, index, src) -> None:
        # Native indexed assignment invokes the C++ setter directly, bypassing
        # the Python copy_ hook even for an ordinary contiguous slice.
        if writes_weight(self, src):
            src = merge(self[index], src)
        orig_setitem(self, index, src)

    torch.Tensor.copy_ = masked_copy_
    torch.Tensor.__setitem__ = masked_setitem
    try:
        yield
    finally:
        torch.Tensor.__setitem__ = orig_setitem
        torch.Tensor.copy_ = orig_copy
