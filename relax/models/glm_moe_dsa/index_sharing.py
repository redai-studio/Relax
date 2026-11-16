# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GLM DSA index ownership and per-microbatch state (also used during
recompute)."""

from collections.abc import Callable
from typing import Any, TypeVar


_T = TypeVar("_T")
_HOLDER_ATTR = "_glm_dsa_topk"


def index_sources(hf_config: Any) -> tuple[int, ...]:
    """Return the one-based computing layer for each decoder layer.

    Match SGLang's DSA schedule and check the explicit HF weight layout when
    present. GLM-5/5.1 default to independent indexers on every layer.
    """
    frequency = getattr(hf_config, "index_topk_freq", None)
    frequency = 1 if frequency is None else frequency
    offset = getattr(hf_config, "index_skip_topk_offset", None)
    offset = 2 if offset is None else offset
    if frequency < 1 or offset < 1:
        raise ValueError("DSA index_topk_freq and index_skip_topk_offset must be positive")
    pattern = getattr(hf_config, "index_topk_pattern", None)
    types = getattr(hf_config, "indexer_types", None)
    if types is not None and len(types) != hf_config.num_hidden_layers:
        raise ValueError("DSA indexer_types must describe every decoder layer")
    sources = []
    for layer_id in range(hf_config.num_hidden_layers):
        shared = (
            layer_id < len(pattern) and pattern[layer_id] == "S"
            if pattern is not None
            else max(layer_id + 1 - offset, 0) % frequency != 0
        )
        if types is not None and types[layer_id] != ("shared" if shared else "full"):
            raise ValueError(f"DSA indexer_types disagrees with the top-k schedule at layer {layer_id}")
        if shared and not sources:
            raise ValueError("The first DSA layer cannot share a previous layer's indices")
        sources.append(sources[-1] if shared else layer_id + 1)
    return tuple(sources)


def validate_stage(sources: tuple[int, ...], layer_offset: int, num_layers: int) -> None:
    """Require each PP/VPP stage to own the indexers its layers consume."""
    if not num_layers:
        return
    if layer_offset < 0 or layer_offset + num_layers > len(sources):
        raise ValueError("DSA pipeline stage is outside the configured decoder layers")
    first_layer = layer_offset + 1
    if sources[layer_offset] != first_layer:
        raise ValueError(
            f"DSA pipeline stage starts at shared layer {first_layer}, whose indexer is on layer "
            f"{sources[layer_offset]}. Cross-PP top-k sharing is unsupported; start each PP/VPP stage on a full layer."
        )


def shared_topk(
    packed_seq_params: Any,
    layer_number: int,
    source_layer: int,
    compute: Callable[[], _T],
) -> _T:
    """Keep each computing layer's indices until its microbatch graph is freed.

    Megatron's checkpoint closure retains the same packed metadata. Keeping all
    sources (rather than just the last layer) makes single-layer recompute
    safe; using this microbatch-owned holder avoids PP 1F1B cross-microbatch
    overwrite. Computing layers refresh their own entry on every
    forward/recompute.
    """
    if packed_seq_params is None:
        raise ValueError("DSA index sharing requires per-microbatch packed_seq_params")
    holder = getattr(packed_seq_params, _HOLDER_ATTR, None)
    if holder is None:
        holder = {}
        setattr(packed_seq_params, _HOLDER_ATTR, holder)
    if layer_number == source_layer:
        holder[source_layer] = compute()
    elif source_layer not in holder:
        raise ValueError(
            f"DSA shared layer {layer_number} needs indices from layer {source_layer} in this microbatch. "
            "The computing layer must run in the same PP/VPP stage first."
        )
    return holder[source_layer]
