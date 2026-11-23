# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Binary reference manifests and token identity."""

import hashlib
import json
import struct
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

from .codecs import replay_codec


REFERENCE_FIELD = "rollout_replay_refs"
FIELDS = {"moe": "rollout_routed_experts", "mm_embedding": "rollout_mm_embeddings"}


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
    codec: str = "int32"

    def __post_init__(self) -> None:
        replay_codec(self.codec).validate(self.kind, self.topk)
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


def selection_ref(value: dict[str, Any]) -> ReplayRef:
    """Return the immutable capture identity behind a training-token view."""
    if "training_tokens" not in value:
        return ReplayRef.from_dict(value)
    manifest = value.copy()
    digest = manifest.pop("training_tokens")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid normalized replay token digest")
    return ReplayRef.from_dict(manifest)


@dataclass(frozen=True)
class MoERouteSpan:
    """The tail of one capture, at its logical positions in the trajectory."""

    ref: ReplayRef
    start: int
    stop: int
    training_tokens: str | None = None

    @property
    def source_start(self) -> int:
        return self.ref.rows - (self.stop - self.start)


def merge_route_spans(
    previous: tuple[MoERouteSpan, ...], incoming: tuple[MoERouteSpan, ...], expected_rows: int
) -> tuple[MoERouteSpan, ...]:
    """Keep accepted rows and append only the uncovered tail of a resumed
    capture."""
    start = previous[-1].stop if previous else 0
    end = incoming[-1].stop if incoming else start
    if not start <= end == expected_rows or (incoming and incoming[0].start > start):
        raise ValueError("Replay references do not cover the accepted Agentic token rows")
    return previous + tuple(
        MoERouteSpan(span.ref, max(span.start, start), span.stop, span.training_tokens)
        for span in incoming
        if span.stop > start
    )


def selection_route_spans(value: dict[str, Any], kind: str) -> tuple[MoERouteSpan, ...]:
    """Read immutable selection tails at their original document
    coordinates."""
    if kind != "moe":
        raise ValueError("Unsupported selection trajectory kind")
    if "spans" not in value:
        ref = selection_ref(value)
        if ref.kind != kind:
            raise ValueError("Expected a matching selection replay reference")
        return (MoERouteSpan(ref, 0, ref.rows, value.get("training_tokens")),)
    if (
        set(value) != {"tokens", "spans"}
        or not isinstance(value["tokens"], str)
        or len(value["tokens"]) != 64
        or any(c not in "0123456789abcdef" for c in value["tokens"])
        or not isinstance(value["spans"], list)
        or not value["spans"]
    ):
        raise ValueError("Invalid selection replay trajectory")
    spans = []
    stop = 0
    shape = None
    for item in value["spans"]:
        if not isinstance(item, dict) or set(item) != {"ref", "start", "stop"}:
            raise ValueError("Invalid selection replay span")
        ref = selection_ref(item["ref"])
        span = MoERouteSpan(ref, item["start"], item["stop"], item["ref"].get("training_tokens"))
        ref_shape = (ref.run, ref.owner, ref.layers, ref.topk, ref.codec)
        if (
            ref.kind != kind
            or ref.version != 2
            or type(span.start) is not int
            or type(span.stop) is not int
            or not stop == span.start < span.stop
            or not 0 <= span.source_start <= span.start
            or (shape is not None and ref_shape != shape)
        ):
            raise ValueError("Selection replay spans must cover consecutive rows with matching manifests")
        shape = ref_shape
        stop = span.stop
        spans.append(span)
    return tuple(spans)


def moe_route_spans(value: dict[str, Any]) -> tuple[MoERouteSpan, ...]:
    return selection_route_spans(value, "moe")


def selection_route_manifest(spans: tuple[MoERouteSpan, ...], tokens: list[int]) -> dict[str, Any]:
    return {
        "tokens": token_digest(tokens),
        "spans": [
            {
                "ref": span.ref.to_dict()
                | ({"training_tokens": span.training_tokens} if span.training_tokens else {}),
                "start": span.start,
                "stop": span.stop,
            }
            for span in spans
        ],
    }


def moe_route_manifest(spans: tuple[MoERouteSpan, ...], tokens: list[int]) -> dict[str, Any]:
    return selection_route_manifest(spans, tokens)


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
    has_spans = any("spans" in value.get(kind, {}) for kind in ("moe",))
    has_view = any("training_tokens" in item for item in value.values())
    if has_view or has_spans or "mm_embedding" in value:
        list(iter_refs(value))
        body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
        size = _HEADER.size + len(body)
        if size > _MAX_WIRE_BYTES:
            raise ValueError("Replay manifest exceeds wire limit")
        return _HEADER.pack(_MAGIC, 6 if has_view else 5 if has_spans else 4, len(value), size) + body
    wire_version = 3 if any(item.get("codec", "int32") != "int32" for item in value.values()) else 2
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
            + (_text(ref.codec) if wire_version == 3 else b"")
        )
    body = b"".join(entries)
    size = _HEADER.size + len(body)
    if size > _MAX_WIRE_BYTES:
        raise ValueError("Replay manifest exceeds wire limit")
    return _HEADER.pack(_MAGIC, wire_version, len(entries), size) + body


def decode_refs(payload: bytes) -> dict[str, Any]:
    """Decode metadata only; legacy JSON is readable but never newly
    emitted."""
    if not payload or len(payload) > _MAX_WIRE_BYTES:
        raise ValueError("Invalid replay manifest size")
    if payload.startswith(b"{"):
        result = json.loads(payload)
        if not isinstance(result, dict) or not result:
            raise ValueError("Invalid legacy replay manifest")
        list(iter_refs(result))
        return result
    if len(payload) < _HEADER.size:
        raise ValueError("Truncated replay header")
    magic, version, count, size = _HEADER.unpack_from(payload)
    if magic != _MAGIC or version not in (2, 3, 4, 5, 6) or size != len(payload) or not count:
        raise ValueError("Invalid replay header/version/length")
    if version in (4, 5, 6):
        result = json.loads(payload[_HEADER.size :])
        if not isinstance(result, dict) or len(result) != count:
            raise ValueError("Invalid replay envelope")
        if version == 4 and "mm_embedding" not in result:
            raise ValueError("Invalid multimodal replay envelope")
        if version == 5 and "spans" not in result.get("moe", {}):
            raise ValueError("Invalid MoE replay trajectory envelope")
        list(iter_refs(result))
        return result
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
        ref = ReplayRef(
            2, owner, run, record, kind, rows, layers, topk, chunk, tokens, text() if version == 3 else "int32"
        )
        if kind in result:
            raise ValueError("Duplicate replay kind")
        result[kind] = ref.to_dict()
    if offset != len(payload):
        raise ValueError("Trailing replay manifest bytes")
    return result


def validate_sample(value: dict[str, Any], tokens: list[int]) -> None:
    if not value:
        raise ValueError("Missing replay reference")
    for kind, item in value.items():
        if kind == "mm_embedding":
            validate_mm_envelope(item)
            continue
        if kind == "moe" and "spans" in item:
            spans = selection_route_spans(item, kind)
            if spans[-1].stop != len(tokens) - 1 or item["tokens"] != token_digest(tokens):
                raise ValueError("MoE replay trajectory does not match sample tokens")
            if any(
                (span.training_tokens or span.ref.tokens)
                != token_digest(tokens[span.stop - span.ref.rows : span.stop + 1])
                for span in spans
            ):
                raise ValueError("MoE replay span does not match its sample token range")
            continue
        ref = selection_ref(item)
        if (
            kind != ref.kind
            or ref.rows != len(tokens) - 1
            or item.get("training_tokens", ref.tokens) != token_digest(tokens)
        ):
            raise ValueError("Replay reference does not match sample tokens or kind")


def validate_mm_envelope(value: dict[str, Any]) -> None:
    """Validate media-coordinate metadata independently of generated text."""
    if not isinstance(value, dict) or set(value) != {"version", "media"} or value["version"] != 1:
        raise ValueError("Invalid MM replay envelope")
    if not isinstance(value["media"], list):
        raise ValueError("MM replay media must be a list")
    for item in value["media"]:
        if not isinstance(item, dict) or set(item) != {"hash", "modality", "grid", "ref", "lease"}:
            raise ValueError("Invalid MM replay media entry")
        image_hash, grid, lease = item["hash"], item["grid"], item["lease"]
        if (
            not isinstance(image_hash, str)
            or len(image_hash) != 64
            or any(c not in "0123456789abcdef" for c in image_hash)
            or item["modality"] not in ("image", "video")
            or not isinstance(grid, list)
            or len(grid) != 3
            or any(type(n) is not int or n <= 0 for n in grid)
            or grid[1] % 2
            or grid[2] % 2
            or not isinstance(lease, str)
            or len(lease) != 32
            or any(c not in "0123456789abcdef" for c in lease)
        ):
            raise ValueError("Invalid MM replay hash, grid or lease")
        ref = ReplayRef.from_dict(item["ref"])
        if (
            ref.kind != "mm_embedding"
            or ref.codec != "bf16-raw-v1"
            or ref.layers != 1
            or ref.topk not in (2048, 2560)
            or ref.tokens != image_hash
            or ref.rows != grid[0] * grid[1] * grid[2] // 4
        ):
            raise ValueError("MM replay reference differs from its media geometry")


def iter_refs(value: dict[str, Any]) -> Iterator[ReplayRef]:
    """Yield owned bodies from both token selections and media references."""
    for kind, item in value.items():
        if kind == "mm_embedding":
            validate_mm_envelope(item)
            for media in item["media"]:
                yield ReplayRef.from_dict(media["ref"])
        elif kind == "moe" and "spans" in item:
            yield from (span.ref for span in selection_route_spans(item, kind))
        else:
            ref = selection_ref(item)
            if ref.kind != kind:
                raise ValueError("Replay kind mismatch")
            yield ref


def publisher_holds(manifests: Iterable[dict[str, Any]]) -> tuple[list[ReplayRef], list[dict]]:
    """Route publishers own records; media occurrences each own an independent
    lease."""
    refs = []
    media = []
    for manifest in manifests:
        refs.extend(ref for ref in iter_refs(manifest) if ref.kind != "mm_embedding")
        media.extend(manifest.get("mm_embedding", {}).get("media", ()))
    return refs, media
