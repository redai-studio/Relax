# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Distributed parity test for the context-parallel gather/split utilities
(``reconstruct_tensor_cp`` / ``split_tensor_cp``) added to the candidate MCore
pin.

The Qwen4Exp CP port (PLE causal conv + QSA indexer block selection) rests on these two
helpers, so they are the highest-risk new code and the cheapest to verify in isolation.
The check runs on a 2-process (CP=2, TP=1, sequence-parallel off) gloo/CPU group:

1. ``reconstruct_tensor_cp`` gathers every rank's zigzag-local slice into the full
   sequential sequence on every rank — parity with the CP=1 full (so a module that ran on
   the full sequence under CP=1 produces the same output under CP>1).
2. ``split_tensor_cp`` is the round-trip inverse: ``split(reconstruct(x)) == x``.
3. The local autograd graph survives the all-gather: a loss on the split output backprops
   a finite gradient into the local input (the local slot is re-pinned, not detached).

Run::

    torchrun --nproc_per_node=2 \\
        tests/backends/megatron/test_cp_reconstruct_split.py

Use the unified image or select its MCore checkout through PYTHONPATH.
"""

import os

import torch
import torch.distributed as dist


def _init():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=world,
        expert_tensor_parallel_size=1,
    )
    return parallel_state


def main():
    ps = _init()
    from megatron.core.ssm.mamba_context_parallel import (
        _redo_attention_load_balancing,
        reconstruct_tensor_cp,
        split_tensor_cp,
    )

    rank = ps.get_context_parallel_rank()
    cp_size = ps.get_context_parallel_world_size()
    s_local, b, d = 4, 1, 3
    s_full = s_local * cp_size

    # Build a known global "full" (sequential) identically on every rank, then derive each
    # rank's zigzag-local slice the way the model's CP split would. This lets reconstruct be
    # checked against the *true* full (parity), not merely a round-trip.
    torch.manual_seed(123)
    full = torch.randn(s_full, b, d)
    load_balanced = _redo_attention_load_balancing(full, cp_size, None)
    local_x = load_balanced[rank * s_local : (rank + 1) * s_local].clone().requires_grad_(True)

    # 1. reconstruct -> every rank holds the full sequential sequence (parity vs CP=1 full).
    gathered = reconstruct_tensor_cp(local_x, None, dim=0)
    assert gathered.shape[0] == s_full, f"gathered seq {gathered.shape[0]} != {s_full}"
    max_err = (gathered - full).abs().max().item()
    assert max_err < 1e-5, f"reconstruct parity failed on rank {rank}: max_err={max_err}"
    if rank == 0:
        print(f"[reconstruct] parity OK (max_err={max_err:.2e})")

    # 2. split is the round-trip inverse.
    back = split_tensor_cp(gathered, None, dim=0)
    assert back.shape[0] == s_local, f"split seq {back.shape[0]} != {s_local}"
    rt_err = (back - local_x).abs().max().item()
    assert rt_err < 1e-5, f"round-trip failed on rank {rank}: rt_err={rt_err}"
    if rank == 0:
        print(f"[split round-trip] OK (rt_err={rt_err:.2e})")

    # 3. backward reaches the local tensor (autograd preserved through the all-gather).
    loss = back.pow(2).sum()
    loss.backward()
    assert local_x.grad is not None, f"rank {rank}: local_x.grad is None"
    assert torch.isfinite(local_x.grad).all(), f"rank {rank}: local_x.grad not finite"
    if rank == 0:
        print(f"[backward] local_x.grad finite OK (max={local_x.grad.abs().max().item():.2e})")

    if rank == 0:
        print("ALL CHECKS PASSED")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
