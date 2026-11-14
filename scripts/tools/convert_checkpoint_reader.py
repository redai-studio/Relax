# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Read ranges from large, uncompressed DCP tensor archives on CPU."""

from __future__ import annotations

import io
import itertools
import math
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


RANGE_READ_THRESHOLD = 1024**3
READ_BLOCK_BYTES = 64 * 1024**2


class _ArchiveView(io.RawIOBase):
    def __init__(self, stream: BinaryIO, offset: int, length: int) -> None:
        self.stream, self.offset, self.length = stream, offset, length
        self.seek(0)

    def tell(self) -> int:
        return self.stream.tell() - self.offset

    def seek(self, offset: int, whence: int = 0) -> int:
        position = (0, self.tell(), self.length)[whence] + offset
        if position < 0:
            raise ValueError("Negative archive position")
        self.stream.seek(self.offset + position)
        return position

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        remaining = max(0, self.length - self.tell())
        return self.stream.read(remaining if size < 0 else min(size, remaining))

    def readinto(self, buffer: Any) -> int:
        return self.stream.readinto(memoryview(buffer)[: max(0, self.length - self.tell())])


@dataclass(frozen=True)
class TensorRecord:
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Any


def tensor_record(path: Path, info: Any, shape: tuple[int, ...], dtype: Any) -> TensorRecord | None:
    import torch
    from torch._weights_only_unpickler import Unpickler

    if getattr(info, "transform_descriptors", None):
        return None
    with path.open("rb") as stream:
        if info.offset < 0 or info.offset + info.length > path.stat().st_size:
            raise ValueError(f"Truncated DCP archive: {path}")
        view = _ArchiveView(stream, info.offset, info.length)
        with zipfile.ZipFile(view) as archive:
            storages = [item for item in archive.infolist() if "/data/" in item.filename]
            if len(storages) != 1 or storages[0].compress_type != zipfile.ZIP_STORED:
                return None
            entry = storages[0]
            prefix = entry.filename.rsplit("/data/", 1)[0]
        view.seek(0)
        with torch.serialization._open_zipfile_reader(view) as archive:
            # Torch archives may intentionally omit CRCs, including on metadata.
            byteorder = archive.get_record("byteorder").decode() if archive.has_record("byteorder") else "little"
            if byteorder != sys.byteorder:
                return None
            descriptor = archive.get_record("data.pkl")

        class MetadataUnpickler(Unpickler):
            def persistent_load(self, saved_id: Any) -> Any:
                kind, storage_type, key, _, numel = saved_id
                if kind != "storage" or entry.filename != f"{prefix}/data/{key}":
                    raise ValueError("DCP tensor references an unexpected storage")
                storage_dtype = torch.uint8 if storage_type is torch.UntypedStorage else storage_type.dtype
                nbytes = numel * torch.empty((), dtype=storage_dtype, device="meta").element_size()
                if nbytes != entry.file_size:
                    raise ValueError("DCP storage size disagrees with its descriptor")
                # Use the serialized count: FakeTensor storage capacity varies
                # across Torch versions. Meta storage never loads the payload.
                return torch.storage.TypedStorage(
                    wrap_storage=torch.UntypedStorage(nbytes, device="meta"), dtype=storage_dtype, _internal=True
                )

        tensor = MetadataUnpickler(io.BytesIO(descriptor), encoding="utf-8").load()
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError("DCP tensor descriptor disagrees with checkpoint metadata")
        view.seek(entry.header_offset)
        header = view.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid tensor storage ZIP header")
        name_bytes, extra_bytes = struct.unpack_from("<HH", header, 26)
        payload = entry.header_offset + 30 + name_bytes + extra_bytes
        if payload + entry.file_size > info.length:
            raise ValueError("Tensor storage extends past its DCP archive")
        if not tensor.is_contiguous():
            return None
        end = (tensor.storage_offset() + tensor.numel()) * tensor.element_size()
        if end > entry.file_size:
            raise ValueError("Tensor extends past its serialized storage")
        return TensorRecord(
            info.offset + payload + tensor.storage_offset() * tensor.element_size(),
            shape,
            tuple(tensor.stride()),
            dtype,
        )


def prepare_range_reads(checkpoint: Path, metadata: Any, tensors: dict[str, Any]) -> dict[Any, TensorRecord]:
    records: dict[Any, TensorRecord] = {}
    shapes = {
        (key, tuple(chunk.offsets)): tuple(chunk.sizes) for key, value in tensors.items() for chunk in value.chunks
    }
    for index, info in metadata.storage_data.items():
        if index.fqn in tensors and info.length >= RANGE_READ_THRESHOLD:
            record = tensor_record(
                checkpoint / info.relative_path,
                info,
                shapes[index.fqn, tuple(index.offset)],
                tensors[index.fqn].properties.dtype,
            )
            if record is not None:
                records[index] = record
    return records


def read_tensor_range(stream: BinaryIO, record: TensorRecord, request: Any, target: Any) -> None:
    import torch

    offsets, lengths = tuple(request.storage_offsets), tuple(request.lengths)
    if (
        target.device.type != "cpu"
        or target.dtype != record.dtype
        or tuple(target.shape) != lengths
        or len(offsets) != len(record.shape)
        or any(
            start < 0 or size <= 0 or start + size > full for start, size, full in zip(offsets, lengths, record.shape)
        )
    ):
        raise ValueError("Invalid DCP range destination or bounds")
    axis = max(0, len(lengths) - 1)
    if lengths and target.stride(-1) != 1:
        raise ValueError("Range reads require a contiguous innermost destination dimension")
    while axis > 0 and record.stride[axis - 1] == target.stride(axis - 1) == math.prod(lengths[axis:]):
        axis -= 1
    element_bytes = target.element_size()
    start = record.offset + sum(offset * stride for offset, stride in zip(offsets, record.stride)) * element_bytes
    for prefix in itertools.product(*(range(size) for size in lengths[:axis])):
        row = target[prefix].view(-1)
        stream.seek(start + sum(index * stride for index, stride in zip(prefix, record.stride)) * element_bytes)
        for position in range(0, row.numel(), max(1, READ_BLOCK_BYTES // element_bytes)):
            block = row[position : position + max(1, READ_BLOCK_BYTES // element_bytes)]
            # NumPy shares the existing CPU destination; readinto allocates no
            # archive-sized buffer, including for BF16 viewed as raw bytes.
            buffer = memoryview(block.view(torch.uint8).numpy())
            completed = 0
            while completed < len(buffer):
                count = stream.readinto(buffer[completed:])
                if not count:
                    raise ValueError("Truncated DCP tensor payload")
                completed += count
