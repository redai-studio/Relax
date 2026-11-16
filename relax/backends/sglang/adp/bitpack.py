# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reversible U16 residuals.

Each 512-word tile uses zero, nibble, byte or full words.
"""

import hashlib
import json
import os
import struct

import numpy as np
import torch
import triton
import triton.language as tl


TILE = 512
_BATCH = 64 * 1024**2


@triton.jit
def _apply_tiles(src, descriptors, owners, REVERSE: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    desc = descriptors + tl.load(owners + tile).to(tl.int64) * 5
    dst = tl.load(desc).to(tl.pointer_type(tl.uint16))
    words = tl.load(desc + 1)
    frame = src + tl.load(desc + 2)
    local = tile - tl.load(desc + 3)
    tile_count = tl.load(desc + 4)
    entry = tl.load(frame.to(tl.pointer_type(tl.uint32)) + local)
    mode = entry & 3
    if mode != 0:
        base = frame + tile_count * 4 + (entry >> 2)
        lane = tl.arange(0, BLOCK)
        if mode == 3:
            z = tl.load(base + lane * 2).to(tl.int32) | (tl.load(base + lane * 2 + 1).to(tl.int32) << 8)
        else:
            if mode == 1:
                code = (tl.load(base + lane // 2).to(tl.int32) >> ((lane & 1) * 4)) & 15
                escape = code == 15
                extra = base + BLOCK // 2
            else:
                code = tl.load(base + lane).to(tl.int32)
                escape = code == 255
                extra = base + BLOCK
            rank = tl.cumsum(escape.to(tl.int32), axis=0) - 1
            lo = tl.load(extra + 2 * rank, mask=escape, other=0).to(tl.int32)
            hi = tl.load(extra + 2 * rank + 1, mask=escape, other=0).to(tl.int32)
            z = tl.where(escape, lo | (hi << 8), code)
        idx = local * BLOCK + lane
        delta = (z >> 1) ^ -(z & 1)
        current = tl.load(dst + idx, mask=idx < words, other=0).to(tl.int32)
        if REVERSE:
            delta = -delta
        tl.store(dst + idx, (current + delta).to(tl.uint16), mask=idx < words)


def _validate_frame(data, tiles):
    if len(data) < 4 * tiles or len(data) % 4:
        raise ValueError("Truncated U16 frame")
    table = np.frombuffer(data, dtype="<u4", count=tiles)
    starts = (table >> 2).astype(np.int64)
    modes = table & 3
    body = np.frombuffer(data, dtype=np.uint8, offset=4 * tiles)
    ends = np.append(starts[1:], len(body))
    if starts[0] != 0 or np.any(ends < starts) or np.any(starts > len(body)):
        raise ValueError("Invalid U16 tile offsets")
    expected = np.zeros(tiles, dtype=np.int64)
    for mode, width in ((1, TILE // 2), (2, TILE), (3, TILE * 2)):
        selected = np.flatnonzero(modes == mode)
        if not len(selected):
            continue
        if np.any(ends[selected] - starts[selected] < width):
            raise ValueError("Truncated U16 base")
        expected[selected] = width
        if mode != 3:
            codes = body[starts[selected, None] + np.arange(width)]
            escapes = ((codes & 15) == 15).sum(1) + ((codes >> 4) == 15).sum(1) if mode == 1 else (codes == 255).sum(1)
            expected[selected] += 2 * escapes
    padding = ends - starts - expected
    if np.any(padding[:-1] != 0) or padding[-1] != (-int(starts[-1] + expected[-1])) % 4:
        raise ValueError("Invalid U16 exception payload")


class PackedU16Plan:
    """One pinned bank and bounded H2D staging for both directions."""

    def __init__(self, path, tensors, header, digest):
        self.source = header["provenance"]["source"]
        intervals = {}
        groups, group, cursor, first = [], [], 0, 0
        for packet in header["packets"]:
            name, off, size = packet["name"], packet["offset"], packet["size"]
            n = packet["n"]
            tiles = triton.cdiv(n, 2 * TILE)
            # Each tile has a 4-byte offset. Byte mode with an escape for every word is the largest encoding.
            if (
                packet["mode"] != "U16_PACKED"
                or off < 0
                or n <= 0
                or (off | n) & 1
                or n > 16 * 1024**2
                or not 4 * tiles <= size <= (4 + 3 * TILE) * tiles
                or size % 4
                or packet["payload_offset"] != cursor
                or off + n > tensors[name].numel() * tensors[name].element_size()
            ):
                raise ValueError("Invalid U16 frame descriptor")
            intervals.setdefault(name, []).append((off, off + n))
            if group and cursor - first + size > _BATCH:
                groups.append((first, cursor, group))
                group = []
                first = cursor
            group.append((tensors[name].data_ptr() + off, n // 2, cursor - first, size))
            cursor += size
        if group:
            groups.append((first, cursor, group))
        for spans in intervals.values():
            spans.sort()
            if any(end > start for (_, end), (start, _) in zip(spans, spans[1:])):
                raise ValueError("Overlapping U16 frames")

        device = next(iter(tensors.values())).device
        sha = hashlib.sha256()
        with path.open("rb") as f:
            preamble = f.read(16)
            if preamble[:8] != b"ADP1\0\0\0\0" or len(preamble) != 16:
                raise ValueError("Invalid U16 package")
            length = struct.unpack("<Q", preamble[8:])[0]
            if not 0 < length <= _BATCH:
                raise ValueError("Invalid U16 header length")
            encoded = f.read(length)
            if json.loads(encoded) != header:
                raise ValueError("U16 header changed while opening package")
            if os.fstat(f.fileno()).st_size != 16 + length + cursor:
                raise ValueError("U16 payload length does not match frame sizes")
            self.bank = torch.empty(cursor, dtype=torch.uint8, device="cpu", pin_memory=True)
            raw = memoryview(self.bank.numpy())
            sha.update(preamble)
            sha.update(encoded)
            for start in range(0, cursor, _BATCH):
                part = raw[start : start + _BATCH]
                if f.readinto(part) != len(part):
                    raise ValueError("Truncated U16 package")
                sha.update(part)
            if f.read(1) or sha.hexdigest() != digest:
                raise ValueError("U16 package integrity mismatch")

        self.stream = torch.cuda.Stream(device=device)
        self.copy_stream = torch.cuda.Stream(device=device)
        capacity = max((end - start for start, end, _ in groups), default=1)
        self.buffers = [torch.empty(capacity, dtype=torch.uint8, device=device) for _ in range(min(2, len(groups)))]
        self.copied = [torch.cuda.Event() for _ in self.buffers]
        self.applied = [torch.cuda.Event() for _ in self.buffers]
        self.batches = []
        for start, end, frames in groups:
            rows, counts, tile = [], [], 0
            for dst, words, offset, size in frames:
                tiles = triton.cdiv(words, TILE)
                _validate_frame(raw[start + offset : start + offset + size], tiles)
                rows.append((dst, words, offset, tile, tiles))
                counts.append(tiles)
                tile += tiles
            rows = torch.tensor(rows, device=device, dtype=torch.int64)
            owners = torch.from_numpy(np.repeat(np.arange(len(counts), dtype=np.int32), counts)).to(device)
            self.batches.append((start, end, rows, owners))
        self.stats = {
            "additional_gpu_buffer_bytes": sum(t.numel() * t.element_size() for t in self.buffers),
            "additional_gpu_metadata_bytes": sum(
                t.numel() * t.element_size() for _, _, rows, owners in self.batches for t in (rows, owners)
            ),
        }
        self.stream.wait_stream(torch.cuda.current_stream(device))

    def apply(self, target):
        caller = torch.cuda.current_stream(self.stream.device)
        self.stream.wait_stream(caller)
        self.copy_stream.wait_stream(self.stream)
        with torch.cuda.device(self.stream.device), torch.cuda.stream(self.stream):
            for i, (begin, end, rows, owners) in enumerate(self.batches):
                slot = i % len(self.buffers)
                with torch.cuda.stream(self.copy_stream):
                    if i >= len(self.buffers):
                        self.copy_stream.wait_event(self.applied[slot])
                    self.buffers[slot][: end - begin].copy_(self.bank[begin:end], non_blocking=True)
                    self.copied[slot].record(self.copy_stream)
                self.stream.wait_event(self.copied[slot])
                _apply_tiles[(owners.numel(),)](self.buffers[slot], rows, owners, target == self.source, TILE)
                self.applied[slot].record(self.stream)
        self.stream.synchronize()
