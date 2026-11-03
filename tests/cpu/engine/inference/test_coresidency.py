# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Approved inline residency through the production planner, ledger and
owner."""

import asyncio
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.distributed.ray.inference_manager import InferenceManager
from relax.distributed.ray.placement_planner import plan_placement
from relax.engine.inference.admission import AdmissionGate
from relax.engine.inference.ownership import OwnershipError, OwnershipLedger


class Handle:
    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        async def remote(*args, **kwargs):
            result = getattr(self.target, name)(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

        return SimpleNamespace(remote=remote)


class Runtime:
    def __init__(self):
        self._admission = AdmissionGate()
        self.resident = False

    def onload(self, tags=None):
        self.resident = True

    def offload(self):
        self.resident = False

    def recover(self):
        assert self.resident


def example_args(deferred):
    # Extract only the layout inputs, without executing the launch script.
    script = Path(__file__).parents[4] / "examples/generate_reward_model/run-qwen3-4B-8xgpu-colocated.sh"
    source = script.read_text()
    resource = json.loads(re.search(r"--resource '([^']+)'", source)[1])
    engine_gpus = int(re.search(r"--genrm-num-gpus-per-engine (\d+)", source)[1])
    assert "--colocate" in source
    return SimpleNamespace(
        resource=resource,
        colocate=True,
        offload_rollout=True,
        _genrm_colocate_with_rollout=True,
        _genrm_instances_resolved={
            "__default__": {"num_gpus": resource["genrm"][1], "num_gpus_per_engine": engine_gpus}
        },
        defer_reward_to_post_process=deferred,
    )


@pytest.mark.parametrize("deferred", [False, True])
async def test_example_weights_rest_generate_train_preserves_approved_residency(deferred):
    args = example_args(deferred)
    plan = plan_placement(args)
    ledger = OwnershipLedger(plan)
    args._inference_authority = Handle(ledger)
    owners = {}
    for claim in plan.claims:
        owner = InferenceManager.__ray_metadata__.modified_class(args, claim.role)
        owner._claims[claim.model] = (claim,)
        owner._locks[claim.model] = asyncio.Lock()
        owner._runtimes[claim.model] = Runtime()
        owners[claim.role] = owner
        ledger.register_model(claim.role, claim.model, Handle(owner), claim)
    training = await ledger.acquire("weights", "weights", ledger.claims_for("actor"), role="actor")
    await ledger.command(training, "weights", "generate", "weights")
    await ledger.command(training, "rest", "generate", "rest")
    assert len(ledger.snapshot()["resident"]) == (1 if deferred else 2)
    await ledger.release(training)

    workload = await ledger.acquire("rollout", "generate", ledger.claims_for("rollout"), role="rollout")
    await ledger.command(workload, "generate", "generate")
    if deferred:
        with pytest.raises(OwnershipError, match="must confirm release"):
            ledger.reserve_activation(workload, "genrm", "__default__")
        assert not ledger.reserve_maintenance("genrm", "__default__")
        await ledger.command(workload, "score", "score")
        assert not owners["rollout"]._runtimes["__default__"].resident
        assert owners["genrm"]._runtimes["__default__"].resident
    else:
        for role, owner in owners.items():
            assert ledger.reserve_maintenance(role, "__default__")
            await owner.maintenance("__default__", "recover")
            assert owner._runtimes["__default__"].resident
    await ledger.release(workload)
    training = await ledger.acquire("train", "train", ledger.claims_for("actor"), role="actor")
    await ledger.command(training, "train", "train")
    assert ledger.snapshot()["resident"] == []
    assert all(not owner._runtimes["__default__"].resident for owner in owners.values())
    assert not ledger.reserve_maintenance("rollout", "__default__")
    assert not ledger.reserve_maintenance("genrm", "__default__")
    await ledger.release(training)
