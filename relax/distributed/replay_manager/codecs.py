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
        if self.name == "uint8" and not 0 < topk <= 256:
            raise ValueError("Uint8 MoE replay supports at most 256 selected experts")

    def validate_width(self, width: int | None) -> None:
        if self.max_encoded_width is None or width is None or not 1 <= width <= self.max_encoded_width:
            raise ValueError(f"Invalid encoded row width for codec {self.name!r}")

    def decode_into(self, data: Any, rows: Any, destinations: Any, output: Any) -> None:
        if not self.encoded:
            raise ValueError("Raw replay does not use a set decoder")
        raise NotImplementedError("Encoded replay requires a decoder supplied by its codec adapter")

    def decode_contiguous_into(
        self, data: Any, source_start: int, destination_start: int, count: int, output: Any
    ) -> None:
        if not self.encoded:
            raise ValueError("Raw replay does not use a set decoder")
        raise NotImplementedError("Encoded replay requires a decoder supplied by its codec adapter")


BF16_RAW = ReplayCodec("bf16-raw-v1", kind="mm_embedding", element_bytes=2)
RAW = ReplayCodec("int32")
UINT8 = ReplayCodec("uint8", kind="moe", element_bytes=1)
_CODECS = {codec.name: codec for codec in (RAW, UINT8, BF16_RAW)}


def moe_route_codec(num_experts: int | None) -> str:
    """Narrow only when the model's complete expert ID range fits in a byte."""
    return UINT8.name if num_experts is not None and 0 < num_experts <= 256 else RAW.name


def replay_codec(name: str) -> ReplayCodec:
    try:
        return _CODECS[name]
    except KeyError:
        raise ValueError(f"Unsupported replay codec {name!r}") from None
