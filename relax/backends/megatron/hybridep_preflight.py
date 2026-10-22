# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fail closed for the audited Qwen3.8 EP8 and Qwen3.6 EP4 intranode
layouts."""


def validate_hybridep_worker_group() -> None:
    import importlib
    import importlib.metadata as metadata
    import json
    import os
    import socket
    from pathlib import Path

    import torch.distributed as dist
    from megatron.core import mpu

    from relax.utils.env import Envs
    from relax.utils.logging_utils import get_logger

    group = mpu.get_expert_tensor_and_model_parallel_group()
    size = dist.get_world_size(group)
    local = {
        "rank": dist.get_global_rank(group, dist.get_rank(group)),
        "hostname": socket.gethostname(),
        "paths": {},
        "errors": [],
    }
    supported = [{"tp": tp, "pp": 4, "cp": 1, "ep": 8, "etp": 1, "world": 32} for tp in (2, 8)]
    supported.append({"tp": 1, "pp": 2, "cp": 4, "ep": 4, "etp": 1, "world": 16})
    actual = {
        "tp": mpu.get_tensor_model_parallel_world_size(),
        "pp": mpu.get_pipeline_model_parallel_world_size(),
        "cp": mpu.get_context_parallel_world_size(),
        "ep": mpu.get_expert_model_parallel_world_size(),
        "etp": mpu.get_expert_tensor_parallel_world_size(),
        "world": dist.get_world_size(group=dist.group.WORLD),
    }
    expected_size = actual["etp"] * actual["ep"]
    if actual not in supported or size != expected_size:
        local["errors"].append(f"Unsupported topology: {actual}, TPxEP size={size}")
    if Envs.NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:
        local["errors"].append("Forced HybridEP accessible-rank override is forbidden")
    hybrid_cpp = None
    try:
        package = metadata.distribution("relax-hybrid-ep")
        revision = os.environ["RELAX_HYBRID_EP_COMMIT"]
        if "+" not in package.version or not revision.startswith(package.version.split("+", 1)[1]):
            raise RuntimeError("HybridEP package does not match the unified image revision")
        root = Path(package.locate_file("")).resolve()
        for name in ("_relax_hybrid_ep", "_relax_hybrid_ep.deep_ep_cpp", "_relax_hybrid_ep.hybrid_ep_cpp"):
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            local["paths"][name] = str(path)
            if not path.is_relative_to(root):
                local["errors"].append(f"{name} loaded outside installed distribution: {path}")
            if name == "_relax_hybrid_ep.hybrid_ep_cpp":
                hybrid_cpp = module
    except Exception as exc:
        local["errors"].append(f"Installed module validation failed: {exc!r}")
    members = [None] * size
    dist.all_gather_object(members, local, group=group)
    ranks = [member["rank"] for member in members]
    if (
        size != expected_size
        or len(set(ranks)) != expected_size
        or len({member["hostname"] for member in members}) != 1
        or any(member["errors"] for member in members)
    ):
        raise RuntimeError(f"HybridEP intranode preflight rejected group: {members}")
    # This detects actual accessible peers; never replace detection with a forced domain size.
    detected = hybrid_cpp.ExtendedMemoryAllocator().detect_accessible_ranks(group)
    detections = [None] * size
    dist.all_gather_object(detections, int(detected), group=group)
    if any(value != expected_size for value in detections):
        raise RuntimeError(
            f"HybridEP accessible peers must all be {expected_size}: ranks={ranks}, detected={detections}"
        )
    if dist.get_rank(group) == 0:
        get_logger(__name__).info(
            "HYBRID_EP_PREFLIGHT %s", json.dumps({"members": members, "detected_ranks": detections})
        )
