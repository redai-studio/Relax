# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HTTP weight-sync control without importing the training tensor runtime."""

import json


EXPERT_IPC_LOADER = "relax.backends.sglang.shard_p2p_weight_loader.control"
TP_IPC_LOADER = "relax.backends.sglang.shard_p2p_weight_loader.control_tp"
COLOCATE_NIXL_LOADER = "relax.backends.sglang.shard_p2p_weight_loader.control_remote"


def normalize_shard_control_response(response: dict | None, load_format: str | None) -> dict | None:
    """DP schedulers can repeat the same full TP-group reply, joined by |."""
    if not (
        load_format in (COLOCATE_NIXL_LOADER, EXPERT_IPC_LOADER, TP_IPC_LOADER)
        and response
        and response.get("success") is True
        and response.get("message", "").lstrip().startswith("{")
    ):
        return response
    decoder = json.JSONDecoder()
    remaining = response["message"].strip()
    reply, end = decoder.raw_decode(remaining)
    while remaining := remaining[end:].strip():
        if not remaining.startswith("|"):
            raise ValueError("Invalid SGLang shard control reply separator")
        remaining = remaining[1:].lstrip()
        duplicate, end = decoder.raw_decode(remaining)
        if duplicate != reply:
            raise ValueError("SGLang DP schedulers disagree on the shard control reply")
    return {**response, "message": json.dumps(reply, separators=(",", ":"))}
