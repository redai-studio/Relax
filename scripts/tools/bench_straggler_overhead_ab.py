# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Paired off/on wall-time A/B for the non-blocking timer shim (synthetic)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

from relax.utils.straggler.timer_shim import NonBlockingTimers


def once(steps: int, x: torch.Tensor, w: torch.Tensor, timers: NonBlockingTimers | None, repeats: int) -> float:
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(steps):
        if timers is not None:
            timers("forward-compute").start()
        y = x @ w
        for _i in range(repeats):
            y = y @ w
        if timers is not None:
            timers("forward-compute").stop()
            timers("all-grads-sync").start()
        dist.all_reduce(y)
        if timers is not None:
            timers("all-grads-sync").stop()
    torch.cuda.synchronize()
    dist.barrier()
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/root/autodl-tmp/straggler_overhead_ab.json")
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--dim", type=int, default=3072)
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    x = torch.randn(args.dim, args.dim, device=device)
    w = torch.randn(args.dim, args.dim, device=device)
    once(5, x, w, None, args.repeats)

    pairs = []
    for i in range(args.pairs):
        # alternate phase so odd/even cancel run-to-run drift
        if i % 2 == 0:
            off = once(args.steps, x, w, None, args.repeats)
            on = once(args.steps, x, w, NonBlockingTimers(), args.repeats)
        else:
            on = once(args.steps, x, w, NonBlockingTimers(), args.repeats)
            off = once(args.steps, x, w, None, args.repeats)
        pct = (on - off) / off * 100.0 if off > 0 else 0.0
        pairs.append({"off_sec": off, "on_sec": on, "overhead_pct": pct})

    if rank == 0:
        pcts = [item["overhead_pct"] for item in pairs]
        result = {
            "world_size": dist.get_world_size(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(dist.get_world_size())],
            "pairs": pairs,
            "median_overhead_pct": statistics.median(pcts),
            "mean_overhead_pct": statistics.fmean(pcts),
            "note": "synthetic paired A/B, not a Relax recipe",
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
