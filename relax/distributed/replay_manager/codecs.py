# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Wire codec contracts; metadata lookup never imports GPU dependencies."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _QSABitmapReplay:
    bits: Any
    mask: Any
    missing_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        import torch

        if (
            self.bits.dtype != torch.int32
            or self.bits.ndim != 2
            or self.bits.shape[1] < 1
            or not self.bits.is_contiguous()
            or self.mask.dtype != torch.bool
            or self.mask.shape != (self.bits.shape[0],)
            or self.mask.device != self.bits.device
            or not isinstance(self.missing_rows, tuple)
            or any(type(row) is not int or not 0 <= row < self.bits.shape[0] for row in self.missing_rows)
            or tuple(sorted(set(self.missing_rows))) != self.missing_rows
        ):
            raise ValueError("Invalid direct QSA bitmap payload")


@dataclass(frozen=True)
class ReplayCodec:
    name: str
    kind: str | None = None
    topk: int | None = None
    max_encoded_width: int | None = None
    element_bytes: int = 4

    @property
    def encoded(self) -> bool:
        return self.max_encoded_width is not None

    def validate(self, kind: str, topk: int) -> None:
        if (self.kind is not None and kind != self.kind) or (self.topk is not None and topk != self.topk):
            raise ValueError(f"Unsupported replay shape/kind for codec {self.name!r}")

    def validate_width(self, width: int | None) -> None:
        if self.max_encoded_width is None or width is None or not 1 <= width <= self.max_encoded_width:
            raise ValueError(f"Invalid encoded row width for codec {self.name!r}")

    def decode_into(self, data: Any, rows: Any, destinations: Any, output: Any) -> None:
        if not self.encoded:
            raise ValueError("Raw replay does not use a set decoder")
        from ._bitmap import decode_into

        decode_into(data, rows, destinations, output)

    def decode_contiguous_into(
        self, data: Any, source_start: int, destination_start: int, count: int, output: Any
    ) -> None:
        if not self.encoded:
            raise ValueError("Raw replay does not use a set decoder")
        from ._bitmap import decode_contiguous_into

        decode_contiguous_into(data, source_start, destination_start, count, output)


BF16_RAW = ReplayCodec("bf16-raw-v1", kind="mm_embedding", topk=2560, element_bytes=2)
RAW = ReplayCodec("int32")
KEYE_BITMAP = ReplayCodec("keye-bitmap-v1", kind="indexer", topk=2048, max_encoded_width=1025)
QSA_BITMAP = ReplayCodec("qsa-r4-bitmap-v1", kind="indexer", topk=512, max_encoded_width=2049)
_CODECS = {codec.name: codec for codec in (RAW, KEYE_BITMAP, QSA_BITMAP, BF16_RAW)}


def replay_codec(name: str) -> ReplayCodec:
    try:
        return _CODECS[name]
    except KeyError:
        raise ValueError(f"Unsupported replay codec {name!r}") from None
