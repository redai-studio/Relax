# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compatibility entry point for Megatron patches using the legacy R3 path."""

from typing import Any

from relax.distributed.replay_manager.backend.megatron import (
    RoutingReplay,
    get_routing_replay_compute_topk,
    register_routing_replay,
    set_routing_replay,
)


__all__ = [
    "RoutingReplay",
    "get_routing_replay_compute_topk",
    "register_routing_replay",
    "set_routing_replay",
]


def __getattr__(name: str) -> Any:
    if name == "ROUTING_REPLAY":
        from relax.distributed.replay_manager.backend import megatron

        return megatron.ROUTING_REPLAY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
