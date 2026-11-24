# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep DeepSeek-V4 fusion pairs together during DCS weight updates."""

import re
from collections.abc import Callable
from functools import partial

import torch


def get_dcs_bucket_splitter(
    model_name: str | None,
    *,
    use_bridge: bool,
    has_model: bool,
) -> Callable[..., tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]] | None:
    """Select DSv4's bucket policy for Bridge models."""
    if not (
        use_bridge
        and has_model
        and isinstance(model_name, str)
        and "deepseekv4" in re.sub(r"[-_]", "", model_name.lower())
    ):
        return None

    from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import _fused_pair_key

    return partial(split_fused_weight_pairs, pair_key=_fused_pair_key)


def split_fused_weight_pairs(
    named_tensors: list[tuple[str, torch.Tensor]],
    pair_key: Callable[[str], str | None],
    *,
    require_complete: bool = False,
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    """Retain incomplete SGLang fusion pairs for the next DCS bucket.

    Bridge's pair keys distinguish weight/scale suffixes and compressor
    families. Unrelated tensors can be sent without retaining a whole layer.
    """
    pairs: dict[str, list[int]] = {}
    for index, (name, _) in enumerate(named_tensors):
        key = pair_key(name)
        if key is None:
            continue
        indices = pairs.setdefault(key, [])
        if len(indices) == 2 or any(named_tensors[previous][0] == name for previous in indices):
            raise ValueError(f"Duplicate fused weight in DCS bucket: {name}")
        indices.append(index)
    pending_indices = {indices[0] for indices in pairs.values() if len(indices) == 1}
    pending = [entry for index, entry in enumerate(named_tensors) if index in pending_indices]
    if require_complete and pending:
        raise ValueError(f"DCS weight update ended with unpaired fused weights: {[name for name, _ in pending]}")
    ready = [entry for index, entry in enumerate(named_tensors) if index not in pending_indices]
    return ready, pending
