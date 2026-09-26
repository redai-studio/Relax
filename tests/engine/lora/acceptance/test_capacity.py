# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pinned two-version capacity, GPU drain, one unload and physical slot
reuse."""

import json
from pathlib import Path
from uuid import uuid4

from . import scenarios, support
from .assertions import worker_observations


CHECKS = (
    "native_pin",
    "capacity",
    "last_gpu_use",
    "cancel_finish_race",
    "slot_clear_completion",
    "slot_reuse",
    "slot_reuse_negative_control",
    "c_generation_and_b_continuation",
    "cuda_graph",
)


async def run(ctx):
    ctx.config["capture_prepares"] = True
    support.configure_test(ctx)
    ctx.native_a = await support.native_bindings(ctx)
    initial = await support.request(ctx, "GET", "/lora/versions")
    await support.publish(ctx, ctx.versions["B"])
    ctx.native_b = await support.native_bindings(ctx)
    loaded = await support.request(ctx, "GET", "/lora/versions")
    for name in ("A", "B"):
        for receipt in loaded["versions"][ctx.versions[name]]["ready"].values():
            for observation in worker_observations(receipt).values():
                assert all(observation.get(key) for key in ("pinned", "resident", "cpu_adapter", "config"))
                assert observation["native_lora_id"] == receipt["native_lora_id"]
    ctx.report["checks"]["native_pin"] = {"status": "PASS", "observations": loaded}
    rejected = await ctx.client.post(
        ctx.base + "/lora/publications", json={"version_id": ctx.versions["C"], "request_id": uuid4().hex}
    )
    assert rejected.status_code == 507, rejected.text
    ctx.report["checks"]["capacity"] = {"status": "PASS", "rejection": rejected.json()}
    operation = initial["versions"][ctx.versions["A"]]["operation_id"]
    uid = ctx.native_a[0]["native_lora_id"]
    await scenarios.gpu_drain_race(ctx, operation, uid)
    clear = await support.wait_file(ctx, ctx.root / (uid + ".clear.json"))
    pending = await support.request(ctx, "GET", "/lora/publications/" + operation)
    assert clear["pending"] and pending["state"] == "RETIRING"
    rejected = await ctx.client.post(
        ctx.base + "/lora/publications", json={"version_id": ctx.versions["C"], "request_id": uuid4().hex}
    )
    assert rejected.status_code == 507, rejected.text
    retired = await support.wait_state(ctx, operation, "RETIRED")
    support.configure_test(ctx)
    prepares = [
        json.loads(line) for path in ctx.root.glob("prepares-*.jsonl") for line in path.read_text().splitlines()
    ]
    assert prepares and any(Path(item["path"]).name == ctx.versions["B"] for item in prepares)
    assert not any(Path(item["path"]).name == ctx.versions["C"] for item in prepares), (
        "capacity rejection reached native load"
    )
    ctx.report["checks"]["capacity"]["native_prepares_before_C_admission"] = prepares
    assert len(retired["absence_receipts"]) == 2
    assert all(receipt["actual_unload_count"] == 1 for receipt in retired["absence_receipts"].values())
    ctx.report["checks"]["slot_clear_completion"] = {"status": "PASS", "pending": pending, "retired": retired}
    published = await support.publish(ctx, ctx.versions["C"])
    for engine, receipt in published["ready"].items():
        old = worker_observations(initial["versions"][ctx.versions["A"]]["ready"][engine])
        new = worker_observations(receipt)
        absent = worker_observations(retired["absence_receipts"][engine])
        assert old.keys() == new.keys() == absent.keys()
        for rank in new:
            assert old[rank]["slot"] == new[rank]["slot"]
            assert not any(absent[rank][key] for key in ("resident", "clearing", "pinned", "cpu_adapter", "config"))
            assert absent[rank]["native_executions"] == 0 and absent[rank]["actual_unload_count"] == 1
    assert published["digest"] == loaded["versions"][ctx.versions["B"]]["digest"]
    native_c = await support.native_bindings(ctx)
    await scenarios.mixed_batches(
        ctx, ctx.native_b, native_c, ctx.versions["B"], ctx.versions["B"], "c_generation_and_b_continuation"
    )
    await scenarios.stale_slot_negative_control(ctx, ctx.native_a, native_c)
    ctx.report["checks"]["slot_reuse"] = {"status": "PASS", "retired": retired, "replacement": published}


def test_capacity(auto_acceptance):
    auto_acceptance("capacity")
