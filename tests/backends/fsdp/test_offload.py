# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Sleep/wake offload of an FSDP2-sharded module.

These tests exist because the first implementation reassigned ``param.data``,
which is a silent no-op under ``fully_shard``: ``.data`` on a DTensor parameter
returns a fresh alias whose ``_local_tensor`` is a different object, so nothing
reaches the parameter and the offload freed zero bytes. The colocate memory
time-share depends entirely on this working, so it is asserted in bytes, not in
"the call did not raise".

The naive alternative — rebinding ``_local_tensor`` to a CPU tensor — desyncs
FSDP2's ``FSDPParam._sharded_param_data`` (which aliases the same storage) and
makes the forward read stale weights forever, so parity across a sleep is
asserted too.
"""

from __future__ import annotations

import os

import pytest
import torch


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FSDP2 offload")


@pytest.fixture()
def dist_env():
    """Single-rank NCCL process group, torn down after the test."""
    import torch.distributed as dist

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29617")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("nccl")
        created = True
    torch.cuda.set_device(0)
    yield
    if created:
        dist.destroy_process_group()


class _Block(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def _build_sharded(dim: int, blocks: int, seed: int) -> torch.nn.Module:
    from torch.distributed.fsdp import fully_shard

    torch.manual_seed(seed)
    model = torch.nn.Sequential(*[_Block(dim) for _ in range(blocks)]).cuda()
    for block in model:
        fully_shard(block)
    fully_shard(model)
    return model


@requires_cuda
def test_fsdp_offload_frees_sharded_parameter_memory(dist_env):
    """Offload must release the parameter bytes, not merely run."""
    from relax.backends.fsdp.runtime import offload_module_to_cpu, onload_module_to_device

    dim, blocks = 2048, 4
    expected_bytes = blocks * dim * dim * 4  # fp32 weights dominate; bias is noise
    model = _build_sharded(dim, blocks, seed=0)

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    offload_module_to_cpu(model)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()

    freed = before - after
    assert freed >= expected_bytes * 0.95, f"offload freed only {freed} bytes, expected ~{expected_bytes}"

    onload_module_to_device(model, torch.device("cuda:0"))
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() >= before * 0.95, "onload did not restore the parameter storage"


@requires_cuda
def test_fsdp_offload_roundtrip_preserves_training(dist_env):
    """A sleep/wake cycle mid-training must not perturb the loss trajectory."""
    from relax.backends.fsdp.runtime import offload_module_to_cpu, onload_module_to_device

    dim, blocks, steps, sleep_at = 512, 2, 4, 2
    device = torch.device("cuda:0")
    torch.manual_seed(7)
    batches = [torch.randn(8, dim, device=device) for _ in range(steps)]

    def run(sleep: bool) -> list[float]:
        model = _build_sharded(dim, blocks, seed=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        for i, x in enumerate(batches):
            if sleep and i == sleep_at:
                offload_module_to_cpu(model)
                onload_module_to_device(model, device)
            loss = model(x).square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach()))
        return losses

    assert run(sleep=False) == run(sleep=True)


@requires_cuda
def test_fsdp_offload_is_idempotent(dist_env):
    """Double offload / double onload must not corrupt weights or double-
    copy."""
    from relax.backends.fsdp.runtime import offload_module_to_cpu, onload_module_to_device

    model = _build_sharded(256, 2, seed=3)
    reference = [p._local_tensor.detach().clone() for p in model.parameters()]

    offload_module_to_cpu(model)
    offload_module_to_cpu(model)
    onload_module_to_device(model, torch.device("cuda:0"))
    onload_module_to_device(model, torch.device("cuda:0"))

    for expected, param in zip(reference, model.parameters(), strict=True):
        assert torch.equal(expected, param._local_tensor)
