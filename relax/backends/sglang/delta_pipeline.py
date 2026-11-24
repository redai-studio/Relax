# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Deliver CUDA IPC handles to the correct SGLang pipeline stage.

SGLang indexes the HTTP list by TP rank on *every* stage. Only deserialize the
local stage's CUDA handles. Other stages' serialized bytes stay on CPU.
"""

from __future__ import annotations

import json
from typing import Any

import torch

from .delta_loader import load_delta_weights, validate_transport_storage


PIPELINE_LOADER = "relax.backends.sglang.delta_pipeline.load_pipeline_weights"


def local_payload(weights: list) -> list:
    from sglang.srt.distributed.parallel_state import get_pp_group, get_tp_group
    from sglang.srt.utils import MultiprocessingSerializer

    if len(weights) != 1 or weights[0][1].numel():
        raise ValueError("Invalid pipeline routing control")
    header = json.loads(weights[0][0])
    pp, tp = get_pp_group(), get_tp_group()
    if pp.world_size != len(header["stages"]) or (tp.world_size, tp.rank_in_group) != (header["tp"], header["rank"]):
        raise ValueError("Delta IPC topology differs from SGLang process groups")
    return MultiprocessingSerializer.deserialize(header["stages"][pp.rank_in_group])


def load_pipeline_weights(model: Any, weights: list) -> None:
    weights = local_payload(weights)
    _install(model, weights)


def _install(model: Any, weights: list) -> None:
    # SGLang returns the HTTP response from stage zero. The trailing IPC byte
    # signals completion on each individual stage, including after a flush.
    if len(weights) == 1:
        if json.loads(weights[0][0])["op"] not in ("begin", "document"):
            raise ValueError("Pipeline IPC acknowledgement missing")
        load_delta_weights(model, weights)
        return
    if len(weights) != 2 or weights[1][0] != "__delta_capability__":
        raise ValueError("Invalid pipeline IPC envelope")
    raw, status = weights[0][1], weights[1][1]
    if (
        type(status) is not torch.Tensor
        or status.dtype != torch.uint8
        or status.ndim != 1
        or not status.numel()
        or not status.is_contiguous()
        or status.device != raw.device
    ):
        raise ValueError("Invalid pipeline completion byte")
    validate_transport_storage(model, (raw, status))
    control = json.loads(weights[0][0])
    if control["op"] == "confirm_flush":
        receiver = getattr(model, "_relax_delta_receiver", None)
        if (
            receiver is None
            or receiver.failed
            or receiver.active is not None
            or receiver.version != control["version"]
            or status.numel() != 1
        ):
            raise ValueError("Pipeline stage has not completed its publication")
    else:
        load_delta_weights(model, [weights[0], (weights[1][0], status[:-1])])
    status[-1].fill_(1)
