# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Seed only observed Triton autotune keys with configurations measured on the
identical Blackwell SM103 stack."""

import hashlib
import importlib
from pathlib import Path


CALIBRATION = {
    "environment": {
        "package_sha256": "30f7feebdbfa87b8e90c143ec62a7fa0781bfd736c856c0ec69452f0c778be3f",
        "torch": "2.13.0+cu130",
        "triton": "3.7.1",
        "multiprocessors": 148,
        "capability": [10, 3],
    },
    "keys": {
        "l2norm_fwd_kernel": [
            {
                "key": [128, 8, "torch.bfloat16", "torch.bfloat16", "torch.float32"],
                "config": {"meta": {"BT": 32}, "warps": 8, "stages": 3, "ctas": 1, "maxnreg": None},
            },
            {
                "key": [128, 16, "torch.bfloat16", "torch.bfloat16", "torch.float32"],
                "config": {"meta": {"BT": 64}, "warps": 16, "stages": 3, "ctas": 1, "maxnreg": None},
            },
            {
                "key": [128, 24, "torch.bfloat16", "torch.bfloat16", "torch.float32"],
                "config": {"meta": {"BT": 64}, "warps": 16, "stages": 3, "ctas": 1, "maxnreg": None},
            },
            {
                "key": [128, 32, "torch.bfloat16", "torch.bfloat16", "torch.float32"],
                "config": {"meta": {"BT": 64}, "warps": 16, "stages": 3, "ctas": 1, "maxnreg": None},
            },
        ],
        "l2norm_bwd_kernel": [
            {
                "key": [128, 8, "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.bfloat16"],
                "config": {"meta": {"BT": 64}, "warps": 16, "stages": 3, "ctas": 1, "maxnreg": None},
            },
            {
                "key": [128, 16, "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.bfloat16"],
                "config": {"meta": {"BT": 16}, "warps": 8, "stages": 3, "ctas": 1, "maxnreg": None},
            },
            {
                "key": [128, 24, "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.bfloat16"],
                "config": {"meta": {"BT": 16}, "warps": 8, "stages": 3, "ctas": 1, "maxnreg": None},
            },
            {
                "key": [128, 32, "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.bfloat16"],
                "config": {"meta": {"BT": 16}, "warps": 8, "stages": 3, "ctas": 1, "maxnreg": None},
            },
        ],
    },
}


def fingerprint(mod):
    import torch
    import triton

    return dict(
        package_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
        torch=torch.__version__,
        triton=triton.__version__,
        multiprocessors=torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
        capability=list(torch.cuda.get_device_capability()),
    )


def fields(cfg):
    return dict(
        meta=cfg.kwargs,
        warps=cfg.num_warps,
        stages=cfg.num_stages,
        ctas=cfg.num_ctas,
        maxnreg=getattr(cfg, "maxnreg", None),
    )


def install():
    from relax.utils.logging_utils import get_logger

    table = CALIBRATION
    module = importlib.import_module("fla.modules.l2norm")
    if fingerprint(module) != table["environment"]:
        return 0
    matched = 0
    for name, entries in table["keys"].items():
        tuner = getattr(module, name)
        for row in entries:
            cfg = [c for c in tuner.configs if fields(c) == row["config"]]
            if len(cfg) != 1:
                continue
            tuner.cache[tuple(row["key"])] = cfg[0]
            matched += 1
    get_logger(__name__).info(
        "Seeded %d measured native FLA Blackwell SM103 forward/backward configuration keys", matched
    )
    return matched
