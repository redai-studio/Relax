# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Format-aware dispatch for colocated SGLang weight updates."""

from collections.abc import Callable, Sequence
from typing import Any

from relax.backends.sglang.weight_loader import EXPERT_SLICE_LOADER


def dispatch_typed_buckets(
    packets: Sequence[Sequence[tuple[str, str]]],
    *,
    engine: Any,
    version: str | int | None,
    serializer: Any,
    empty_flattened: Callable[[], dict[str, Any]],
    held: list[Any],
    dispatch_errors: list[str] | None = None,
) -> list[Any]:
    """Keep a single request format even with empty or differently packed
    ranks.

    Format tags ride in the existing Gloo gather. Producers retain GPU storage
    until all submitted readers complete, including partial dispatch failures.
    """
    if any(fmt not in ("direct", "flattened_bucket", EXPERT_SLICE_LOADER) for packet in packets for fmt, _ in packet):
        raise ValueError("Unsupported weight IPC format")
    refs, empty = [], {}
    count = max((len(packet) for packet in packets), default=0)
    for index in range(count):
        formats = dict.fromkeys(packet[index][0] for packet in packets if index < len(packet))
        for load_format in formats:
            try:
                serialized = []
                for packet in packets:
                    if index < len(packet) and packet[index][0] == load_format:
                        serialized.append(packet[index][1])
                    else:
                        if load_format not in empty:
                            payload = empty_flattened() if load_format == "flattened_bucket" else []
                            held.append(payload)
                            empty[load_format] = serializer.serialize(payload, output_str=True)
                        serialized.append(empty[load_format])
                kwargs = dict(serialized_named_tensors=serialized, load_format=load_format)
                if version is not None:
                    kwargs["weight_version"] = str(version)
                refs.append(engine.update_weights_from_tensor.remote(**kwargs))
            except Exception as error:
                if dispatch_errors is None:
                    raise
                dispatch_errors.append(str(error))
                return refs
    return refs
