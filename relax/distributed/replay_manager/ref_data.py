# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Binary reference manifests, token identity and fixed-width row handles."""

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from typing import Any


REFERENCE_FIELD = "rollout_replay_refs"
FIELDS = {"moe": "rollout_routed_experts", "indexer": "rollout_indexer_topk"}


def register_kind(kind: str, field: str) -> None:
    """Register an int32 layered payload in every participating process.

    Custom capture, layout and replay semantics require an explicit adapter.
    Registration is idempotent; existing kind/field mappings cannot be changed.
    """
    if not isinstance(kind, str) or not kind.isidentifier():
        raise ValueError("Replay kind must be a nonempty identifier")
    if not isinstance(field, str) or not field.isidentifier() or field == REFERENCE_FIELD:
        raise ValueError("Replay field must be an identifier distinct from the reference field")
    if kind in FIELDS:
        if FIELDS[kind] != field:
            raise ValueError(f"Replay kind {kind!r} is already registered")
        return
    if field in FIELDS.values():
        raise ValueError(f"Replay field {field!r} is already registered")
    FIELDS[kind] = field


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(tokens)}q", *tokens)).hexdigest()


@dataclass(frozen=True)
class ReplayRef:
    """Small wire descriptor locating immutable chunks through their run's
    registered inventory."""

    version: int
    owner: str
    run: str
    record: str
    kind: str
    rows: int
    layers: int
    topk: int
    chunk_rows: int
    tokens: str

    def __post_init__(self) -> None:
        if self.version not in (1, 2) or self.kind not in FIELDS:
            raise ValueError("Unsupported replay manifest version or kind")
        if min(self.rows, self.layers, self.topk, self.chunk_rows) <= 0:
            raise ValueError("Replay manifest dimensions must be positive")
        if self.version == 2 and (len(self.record) != 32 or len(self.tokens) != 64):
            raise ValueError("Binary replay requires a UUID record and SHA256 token identity")
        for name in ("owner", "run", "record", "tokens"):
            value = getattr(self, name)
            if not value or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid replay identity: {name}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReplayRef":
        return cls(**value)


_MAGIC = b"RPLY"
_HEADER = struct.Struct("<4sHHI")  # magic, wire version, entry count, total bytes
_SHAPE = struct.Struct("<IIII")  # rows, layers, topk, chunk rows
_MAX_WIRE_BYTES = 16 << 20


def _text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > 65535:
        raise ValueError("Replay descriptor string is too large")
    return struct.pack("<H", len(encoded)) + encoded


def encode_refs(value: dict[str, Any]) -> bytes:
    """Encode a sample's kind manifests with a versioned little-endian schema.

    Uniform chunk ranges are implicit in rows/chunk_rows. A record UUID groups
    the layer objects; TQ locators are derived by the transport, not pickled
    here.
    """
    if not value or len(value) > 65535:
        raise ValueError("Invalid replay manifest entry count")
    entries = []
    for kind, item in sorted(value.items()):
        ref = ReplayRef.from_dict(item)
        if kind != ref.kind or ref.version != 2:
            raise ValueError("New replay publication must use a matching v2 manifest")
        entries.append(
            _text(kind)
            + _text(ref.owner)
            + _text(ref.run)
            + bytes.fromhex(ref.record)
            + bytes.fromhex(ref.tokens)
            + _SHAPE.pack(ref.rows, ref.layers, ref.topk, ref.chunk_rows)
        )
    body = b"".join(entries)
    size = _HEADER.size + len(body)
    if size > _MAX_WIRE_BYTES:
        raise ValueError("Replay manifest exceeds wire limit")
    return _HEADER.pack(_MAGIC, 2, len(entries), size) + body


def decode_refs(payload: bytes) -> dict[str, Any]:
    """Decode metadata only; legacy JSON is readable but never newly
    emitted."""
    if not payload or len(payload) > _MAX_WIRE_BYTES:
        raise ValueError("Invalid replay manifest size")
    if payload.startswith(b"{"):
        result = json.loads(payload)
        if not isinstance(result, dict) or not result:
            raise ValueError("Invalid legacy replay manifest")
        for kind, item in result.items():
            if ReplayRef.from_dict(item).kind != kind:
                raise ValueError("Replay kind mismatch")
        return result
    if len(payload) < _HEADER.size:
        raise ValueError("Truncated replay header")
    magic, version, count, size = _HEADER.unpack_from(payload)
    if magic != _MAGIC or version != 2 or size != len(payload) or not count:
        raise ValueError("Invalid replay header/version/length")
    offset = _HEADER.size

    def take(size: int) -> bytes:
        nonlocal offset
        if offset + size > len(payload):
            raise ValueError("Truncated replay manifest")
        value = payload[offset : offset + size]
        offset += size
        return value

    def text() -> str:
        return take(struct.unpack("<H", take(2))[0]).decode("utf-8")

    result = {}
    for _ in range(count):
        kind, owner, run = text(), text(), text()
        record, tokens = take(16).hex(), take(32).hex()
        rows, layers, topk, chunk = _SHAPE.unpack(take(_SHAPE.size))
        ref = ReplayRef(2, owner, run, record, kind, rows, layers, topk, chunk, tokens)
        if kind in result:
            raise ValueError("Duplicate replay kind")
        result[kind] = ref.to_dict()
    if offset != len(payload):
        raise ValueError("Trailing replay manifest bytes")
    return result


class TokenHandle:
    """Six int32 words: UUID[4], row[1], tag[1]; bit-preserving layout only."""

    RECORD = 0
    PAD = 1
    GAP = 2
    WIDTH = 6

    @staticmethod
    def expand(ref: ReplayRef):
        import numpy as np

        if len(ref.record) != 32 or ref.rows >= 2**31:
            raise ValueError("Replay record cannot be represented by token handles")
        result = np.zeros((ref.rows, 6), dtype=np.int32)
        result[:, :4] = np.frombuffer(bytes.fromhex(ref.record), dtype="<i4")
        result[:, 4] = np.arange(ref.rows, dtype=np.int32)
        return result

    @classmethod
    def padding(cls, count: int, *, gap: bool = False):
        import numpy as np

        result = np.zeros((count, 6), dtype=np.int32)
        result[:, 4] = np.arange(count, dtype=np.int32)
        result[:, 5] = cls.GAP if gap else cls.PAD
        return result

    @classmethod
    def coordinates(cls, handles, refs: list[ReplayRef]):
        import numpy as np

        if handles.dtype != np.int32 or handles.ndim != 2 or handles.shape[1] != 6:
            raise ValueError("Replay handles must be int32 [rows, 6]")
        if np.any((handles[:, 5] < cls.RECORD) | (handles[:, 5] > cls.GAP)):
            raise ValueError("Unknown replay handle tag")
        # A gap must be supplied with explicit per-kind semantics; never silently
        # convert a missing real token to padding in the current Relax adapter.
        if np.any(handles[:, 5] == cls.GAP):
            raise ValueError("Replay gap requires explicit adapter handling")
        result = np.stack((np.full(len(handles), -1), handles[:, 4].astype(np.int64)), axis=1)
        found = handles[:, 5] == cls.PAD
        if np.any(found & np.any(handles[:, :4] != 0, axis=1)) or np.any(handles[:, 4] < 0):
            raise ValueError("Malformed replay handle")
        seen = {}
        for i, ref in enumerate(refs):
            if ref.record in seen:
                if seen[ref.record] != ref:
                    raise ValueError("Conflicting manifests for one replay record")
                continue  # Repeated identical samples can resolve to the same body.
            seen[ref.record] = ref
            words = np.frombuffer(bytes.fromhex(ref.record), dtype="<i4")
            selected = np.all(handles[:, :4] == words, axis=1) & (handles[:, 5] == cls.RECORD)
            if np.any(handles[selected, 4] >= ref.rows):
                raise ValueError("Replay handle row outside manifest")
            result[selected, 0] = i
            found |= selected
        if not np.all(found):
            raise ValueError("Replay handle has no manifest")
        return result


def validate_sample(value: dict[str, Any], tokens: list[int]) -> None:
    if not value:
        raise ValueError("Missing replay reference")
    for kind, item in value.items():
        ref = ReplayRef.from_dict(item)
        if kind != ref.kind or ref.rows != len(tokens) - 1 or ref.tokens != token_digest(tokens):
            raise ValueError("Replay reference does not match sample tokens or kind")
