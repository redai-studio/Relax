# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reversible U16 residuals.

Each 512-word tile uses zero, nibble, byte or full words.
"""

import json
import struct

import numpy as np
import torch
import triton
import triton.language as tl


MAGIC = b"ADTS\0\0\0\0"
TILE = 512
_BATCH_BYTES = 64 * 1024**2


def encode_frame(left, right):
    """Encode a U16 delta, returning None for identical slices."""
    tile_words = TILE

    # Compute the delta modulo 2^16 so subtracting it recovers the original bits.
    delta = right.view("<u2").astype(np.int32) - left.view("<u2").astype(np.int32)
    delta = ((delta + 32768) & 65535) - 32768
    if not np.any(delta):
        return None

    # ZigZag gives small codes to deltas near zero. The kernel masks writes to padded lanes.
    codes = ((delta << 1) ^ (delta >> 15)).astype("<u2")
    codes = np.pad(codes, (0, -len(codes) % tile_words)).reshape(-1, tile_words)

    # Include the two-byte escape values when choosing the smallest encoding for each tile.
    nibble_escapes = np.count_nonzero(codes >= 15, axis=1)
    byte_escapes = np.count_nonzero(codes >= 255, axis=1)
    candidate_bytes = np.stack(
        (tile_words // 2 + 2 * nibble_escapes, tile_words + 2 * byte_escapes, np.full(len(codes), tile_words * 2)),
        axis=1,
    )
    modes = np.argmin(candidate_bytes, axis=1) + 1
    tile_bytes = np.min(candidate_bytes, axis=1)
    zero_tiles = ~np.any(codes, axis=1)
    modes[zero_tiles] = tile_bytes[zero_tiles] = 0

    # Each entry packs a body-relative byte offset above two mode bits. Pad the frame to a four-byte boundary.
    tile_offsets = np.cumsum(tile_bytes, dtype=np.int64) - tile_bytes
    tile_table = ((tile_offsets << 2) | modes).astype("<u4")
    body = np.zeros((int(tile_bytes.sum()) + 3) & ~3, dtype=np.uint8)

    for mode, escape_marker in ((1, 15), (2, 255), (3, 65535)):
        tile_ids = np.flatnonzero(modes == mode)
        if not tile_ids.size:
            continue
        values = codes[tile_ids]
        if mode == 3:
            base_codes = values.view(np.uint8).reshape(len(tile_ids), tile_words * 2)
        else:
            literals = np.minimum(values, escape_marker).astype(np.uint8)
            base_codes = literals[:, ::2] | (literals[:, 1::2] << 4) if mode == 1 else literals
        body[tile_offsets[tile_ids, None] + np.arange(base_codes.shape[1])] = base_codes

        # Each tile stores its full escape values after its short codes, in lane order.
        if mode != 3:
            escape_rows, lanes = np.nonzero(values >= escape_marker)
            counts = (nibble_escapes if mode == 1 else byte_escapes)[tile_ids]
            first_escape = np.cumsum(counts) - counts
            escape_index = np.arange(len(escape_rows)) - first_escape[escape_rows]
            positions = tile_offsets[tile_ids[escape_rows]] + base_codes.shape[1] + 2 * escape_index
            escaped = values[escape_rows, lanes]
            body[positions] = escaped & 255
            body[positions + 1] = escaped >> 8

    return tile_table.tobytes() + body.tobytes()


@triton.jit
def _apply_tiles(batch_data, descriptors, tile_to_frame, REVERSE: tl.constexpr, BLOCK: tl.constexpr):
    # One program decodes one tile. All tiles in a frame share a descriptor.
    tile_id = tl.program_id(0)
    frame_row = descriptors + tl.load(tile_to_frame + tile_id).to(tl.int64) * 5
    dst = tl.load(frame_row).to(tl.pointer_type(tl.uint16))
    frame_words = tl.load(frame_row + 1)
    frame = batch_data + tl.load(frame_row + 2)
    tile_in_frame = tile_id - tl.load(frame_row + 3)
    tile_count = tl.load(frame_row + 4)

    entry = tl.load(frame.to(tl.pointer_type(tl.uint32)) + tile_in_frame)
    mode = entry & 3
    if mode != 0:
        tile_data = frame + tile_count * 4 + (entry >> 2)
        lane = tl.arange(0, BLOCK)
        if mode == 3:
            zigzag = tl.load(tile_data + lane * 2).to(tl.int32) | (tl.load(tile_data + lane * 2 + 1).to(tl.int32) << 8)
        else:
            if mode == 1:
                code = (tl.load(tile_data + lane // 2).to(tl.int32) >> ((lane & 1) * 4)) & 15
                escape = code == 15
                escape_data = tile_data + BLOCK // 2
            else:
                code = tl.load(tile_data + lane).to(tl.int32)
                escape = code == 255
                escape_data = tile_data + BLOCK

            # Escaped words follow the short codes in lane order. The prefix sum indexes that list.
            escape_index = tl.cumsum(escape.to(tl.int32), axis=0) - 1
            lo = tl.load(escape_data + 2 * escape_index, mask=escape, other=0).to(tl.int32)
            hi = tl.load(escape_data + 2 * escape_index + 1, mask=escape, other=0).to(tl.int32)
            zigzag = tl.where(escape, lo | (hi << 8), code)

        word_index = tile_in_frame * BLOCK + lane
        delta = (zigzag >> 1) ^ -(zigzag & 1)
        current = tl.load(dst + word_index, mask=word_index < frame_words, other=0).to(tl.int32)
        if REVERSE:
            delta = -delta
        tl.store(dst + word_index, (current + delta).to(tl.uint16), mask=word_index < frame_words)


def read_package_header(stream):
    """Read the ADTS header and leave the stream at the first payload byte."""
    magic, header_bytes = struct.unpack("<8sQ", stream.read(16))
    if magic != MAGIC:
        raise ValueError("Expected an ADTS package")
    return json.loads(stream.read(header_bytes))


class PackedU16Plan:
    """One pinned bank and reusable H2D buffers for both directions."""

    def __init__(self, stream, tensors, header):
        self.source = header["provenance"]["source"]
        self.target_names = frozenset(packet["name"] for packet in header["packets"])
        for name in self.target_names:
            if not tensors[name].is_contiguous():
                raise ValueError(f"ADTS requires contiguous weight storage: {name}")

        batch_layouts, pending_frames = [], []
        payload_bytes = batch_start = 0
        for packet in header["packets"]:
            tensor = tensors[packet["name"]]
            # offset and n describe the destination byte range. size counts compressed bytes.
            weight_offset, weight_bytes, packed_bytes = packet["offset"], packet["n"], packet["size"]
            if not 0 <= weight_offset < weight_offset + weight_bytes <= tensor.nbytes:
                raise ValueError(f"Frame exceeds target tensor boundary: {packet['name']}")

            if pending_frames and payload_bytes - batch_start + packed_bytes > _BATCH_BYTES:
                batch_layouts.append((batch_start, payload_bytes, pending_frames))
                pending_frames = []
                batch_start = payload_bytes

            # The kernel counts destination words and addresses compressed bytes relative to the batch.
            pending_frames.append((tensor.data_ptr() + weight_offset, weight_bytes // 2, payload_bytes - batch_start))
            payload_bytes += packed_bytes
        if pending_frames:
            batch_layouts.append((batch_start, payload_bytes, pending_frames))

        device = next(iter(tensors.values())).device
        self.bank = torch.empty(payload_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
        if stream.readinto(self.bank.numpy()) != payload_bytes:
            raise ValueError("Truncated U16 package")

        self.stream = torch.cuda.Stream(device=device)
        self.copy_stream = torch.cuda.Stream(device=device)
        capacity = max((end - start for start, end, _ in batch_layouts), default=1)
        self.buffers = [
            torch.empty(capacity, dtype=torch.uint8, device=device) for _ in range(min(2, len(batch_layouts)))
        ]
        self.copied = [torch.cuda.Event() for _ in self.buffers]
        self.applied = [torch.cuda.Event() for _ in self.buffers]

        self.batches = []
        for start, end, frames in batch_layouts:
            frame_rows, tile_counts, first_tile = [], [], 0
            for weight_ptr, frame_words, offset_in_batch in frames:
                tile_count = triton.cdiv(frame_words, TILE)
                frame_rows.append((weight_ptr, frame_words, offset_in_batch, first_tile, tile_count))
                tile_counts.append(tile_count)
                first_tile += tile_count

            descriptors = torch.tensor(frame_rows, device=device, dtype=torch.int64)
            # Frames with four tiles and one tile give tile_to_frame = [0, 0, 0, 0, 1].
            tile_to_frame = torch.from_numpy(np.repeat(np.arange(len(tile_counts), dtype=np.int32), tile_counts)).to(
                device
            )
            self.batches.append((start, end, descriptors, tile_to_frame))

        self.stream.wait_stream(torch.cuda.current_stream(device))

    def apply(self, target):
        caller = torch.cuda.current_stream(self.stream.device)
        self.stream.wait_stream(caller)
        self.copy_stream.wait_stream(self.stream)

        with torch.cuda.device(self.stream.device), torch.cuda.stream(self.stream):
            for batch_index, (begin, end, descriptors, tile_to_frame) in enumerate(self.batches):
                slot = batch_index % len(self.buffers)
                # Wait only when reusing this slot so a copy can overlap decoding in the other slot.
                with torch.cuda.stream(self.copy_stream):
                    if batch_index >= len(self.buffers):
                        self.copy_stream.wait_event(self.applied[slot])
                    self.buffers[slot][: end - begin].copy_(self.bank[begin:end], non_blocking=True)
                    self.copied[slot].record(self.copy_stream)

                self.stream.wait_event(self.copied[slot])
                _apply_tiles[(tile_to_frame.numel(),)](
                    self.buffers[slot], descriptors, tile_to_frame, target == self.source, TILE
                )
                self.applied[slot].record(self.stream)

        self.stream.synchronize()
