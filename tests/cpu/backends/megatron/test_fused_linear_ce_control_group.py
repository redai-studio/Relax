# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real Gloo regression for source-only weight-sync group registration."""

from __future__ import annotations

import ast
import importlib.util
import sys
from argparse import Namespace
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.utils.distributed_utils import init_process_group


ROOT = Path(__file__).resolve().parents[4]


def _worker(rank: int, rendezvous: str) -> None:
    dist.new_group = partial(dist.new_group, timeout=timedelta(seconds=10))
    # Load the real control implementation without importing optional model/CUDA
    # dependencies through the backend package's __init__.
    spec = importlib.util.spec_from_file_location("fused_control", ROOT / "relax/backends/megatron/fused_linear_ce.py")
    fused = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fused
    spec.loader.exec_module(fused)
    source = ROOT / "relax/backends/megatron/model.py"
    startup = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "initialize_model_and_optimizer"
    )

    class ModelSetupReached(Exception):
        pass

    def setup(*_args):
        # Stop at model construction: the actual startup body must already have
        # created the group. No model, optimizer or GPU is needed for this race.
        raise ModelSetupReached

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=10))
    try:
        groups = []
        for role in ("actor", "actor_fwd"):
            group = dist.new_group([0, 1], backend="gloo", timeout=timedelta(seconds=10))
            groups.append(group)
            scope = {
                "torch": torch,
                "mpu": SimpleNamespace(
                    get_tensor_model_parallel_world_size=lambda: 2,
                    get_tensor_model_parallel_group=lambda: group,
                ),
                "_tp_control_group": fused._tp_control_group,
                "setup_model_and_optimizer": setup,
            }
            exec(compile(ast.Module(body=[startup], type_ignores=[]), str(source), "exec"), scope)
            with pytest.raises(ModelSetupReached):
                scope["initialize_model_and_optimizer"](Namespace(fused_linear_ce=True), role)

        if rank == 0:
            # The same real helper used by DeviceDirect, with an independent
            # Store and only the source rank participating.
            init_process_group(
                backend="gloo",
                store=dist.HashStore(),
                rank=0,
                world_size=1,
                group_name="source_only_weight_sync",
                timeout=timedelta(seconds=10),
            )
        ready = torch.ones(1)
        dist.all_reduce(ready, group=dist.group.WORLD)

        # Forward must reuse the bootstrap groups. Reject creation rather than
        # let an unpatched test hang on the mismatched rendezvous names.
        def late_creation(*_args, **_kwargs):
            raise AssertionError("TP control group was not created before weight sync")

        dist.new_group = late_creation
        for group in groups:
            for invalid_rank in (None, 0, 1, None):
                reason = "unsupported local shard" if rank == invalid_rank else None
                result = fused._tp_fallback_reason(reason, group)
                assert (result is not None) is (invalid_rank is not None)
                if reason is not None:
                    assert result == reason
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
def test_tp_control_group_precedes_asymmetric_weight_sync(tmp_path: Path) -> None:
    # Two CPU processes suffice: mocks cannot reproduce c10d registry skew.
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
