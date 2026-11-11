# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Wire codec contracts; metadata lookup never imports GPU dependencies."""

from dataclasses import dataclass
from typing import Any


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


RAW = ReplayCodec("int32")
KEYE_BITMAP = ReplayCodec("keye-bitmap-v1", kind="indexer", topk=2048, max_encoded_width=1025)
_CODECS = {codec.name: codec for codec in (RAW, KEYE_BITMAP)}


def replay_codec(name: str) -> ReplayCodec:
    try:
        return _CODECS[name]
    except KeyError:
        raise ValueError(f"Unsupported replay codec {name!r}") from None
