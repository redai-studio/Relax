# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Map Megatron timer names onto the coarse stages in the RFC."""

from __future__ import annotations


STAGES: tuple[str, ...] = (
    "fwd",
    "bwd",
    "pp_recv",
    "pp_send",
    "dp_grad_sync",
    "optim",
    "attention",
    "moe",
)

_EXACT = {
    "forward-compute": "fwd",
    "backward-compute": "bwd",
    "forward-recv": "pp_recv",
    "backward-recv": "pp_recv",
    "forward-send": "pp_send",
    "backward-send": "pp_send",
    "forward-send-forward-recv": "pp_recv",
    "forward-send-backward-recv": "pp_recv",
    "backward-send-forward-recv": "pp_recv",
    "backward-send-backward-recv": "pp_recv",
    "all-grads-sync": "dp_grad_sync",
    "layernorm-grads-all-reduce": "dp_grad_sync",
    "embedding-grads-all-reduce": "dp_grad_sync",
    "grads-reduce-scatter": "dp_grad_sync",
    "params-all-gather": "dp_grad_sync",
    "optimizer": "optim",
}


def stage_of(name: str, *, module_stages: bool = False) -> str | None:
    """Return the coarse stage for a Megatron timer name, or None to ignore it.

    ``forward-backward`` is ignored so the outer span is not added on top of
    forward and backward. Attention and MoE names stay ignored unless the
    optional module-stage flag is on.
    """
    if name == "forward-backward":
        return None
    if name in _EXACT:
        return _EXACT[name]
    if "recv" in name:
        return "pp_recv"
    if "send" in name:
        return "pp_send"
    if name.startswith("optimizer"):
        return "optim"
    if "grad" in name and ("sync" in name or "all-reduce" in name or "reduce-scatter" in name):
        return "dp_grad_sync"
    if module_stages and "attention" in name:
        return "attention"
    if module_stages and ("moe" in name or "expert" in name):
        return "moe"
    return None
