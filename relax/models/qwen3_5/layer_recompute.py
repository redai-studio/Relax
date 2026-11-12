# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Scope native GDN and expert checkpoint policy to one layer invocation."""

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def layer_context(layer: Any, packed: Any) -> Iterator[None]:
    expert = getattr(layer, "_qwen_expert_checkpoint_state", None)
    gdn = getattr(layer, "_qwen_gdn_checkpoint_state", None)
    if expert is None and gdn is None:
        # MTP and unrelated models preserve their existing checkpoint policy.
        yield
        return
    record = expert if expert is not None else gdn
    assert packed is not None and hasattr(packed, record["mask_key"]), "Frozen packed mask required"
    mask = tuple(getattr(packed, record["mask_key"]))
    assert len(set(mask)) == len(mask) and all(0 <= index < 40 for index in mask)
    if expert is not None:
        assert expert["active"] is None, "Unexpected recursive transformer layer invocation"
    if gdn is not None:
        assert not gdn["entered"], "Recursive layer invocation unsupported"
    if expert is not None:
        expert["active"] = expert["layer"] in mask
    previous = None
    if gdn is not None:
        previous = gdn["module"].recompute_gdn
        gdn["entered"] = True
        gdn["module"].recompute_gdn = gdn["layer"] not in mask
    try:
        yield
    finally:
        if expert is not None:
            expert["active"] = None
        if gdn is not None:
            gdn["module"].recompute_gdn = previous
            gdn["entered"] = False
