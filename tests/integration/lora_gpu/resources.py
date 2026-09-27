# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""GPU capacity, native in-flight ownership, cleanup and replay checks."""

import asyncio
import time
from typing import Any

from tests.integration.lora_gpu.support import events


async def check_resources(
    args: Any,
    report: Any,
    registry: Any,
    shard: Any,
    sessions: Any,
    snapshots: Any,
    publish: Any,
    publisher: Any,
    fanout: Any,
    backend: Any,
    input_ids: Any,
    name_a: Any,
) -> Any:
    from relax.agentic.session.lora_version import LoRAVersionError
    from relax.distributed.checkpoint_service.lora_publication import (
        LoRAPublicationError,
        materialize_adapter_snapshot,
    )

    result = report["resources"] = {"passed": False}
    for label, delta in (("C", 0.001), ("D", 0.002)):
        base = snapshots["B"]
        tensors = dict(base.tensors)
        key = next(iter(tensors))
        tensors[key] = tensors[key] + delta
        snapshots[label] = materialize_adapter_snapshot(base.config, tensors)
    before = len(fanout.calls)
    try:
        await asyncio.to_thread(publish, "C")
    except LoRAVersionError as exc:
        assert exc.code == "CAPACITY_ERROR", exc
        result["capacity_error"] = exc.code
    else:
        raise AssertionError("C must not publish while A references retain the second slot")
    assert len(fanout.calls) == before
    result["capacity_rejection_engine_rpcs"] = 0

    # Begin a real A request before releasing its Session references, and prove
    # native acquire happened before allowing the publisher to reclaim A.
    request = asyncio.create_task(
        backend.generate(
            input_ids=input_ids,
            sampling_params={"max_new_tokens": 512, "temperature": 0.0, "ignore_eos": True},
            session_id="engine0-old",
            request_id="retained-A",
            lora_path=name_a,
        )
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        trace = events(args.output / "engine0.events.jsonl")
        if any(e["ev"] == "req.resolve.exit" and "retained-A" in e["rids"] for e in trace):
            break
        assert not request.done(), "Long request completed before the overlap probe"
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("No native acquire for retained-A")
    for item in sessions.values():
        await shard._release_session_lora_ref(item)
    assert not request.done()
    start = time.monotonic()
    assert await asyncio.to_thread(publisher.reclaim_once)
    output = await request
    assert len(output.new_tokens) == 512
    result["reclaim_wait_s"] = time.monotonic() - start
    result["inflight_completed_tokens"] = len(output.new_tokens)
    assert not await asyncio.to_thread(publisher.reclaim_once)
    for engine in ("engine0", "engine1"):
        trace = events(args.output / f"{engine}.events.jsonl")
        unloads = [e for e in trace if e["ev"] == "manager.unload.exit" and e.get("name") == name_a]
        assert len(unloads) == 1 and unloads[0]["success"] is True, unloads
        result[f"{engine}_A_physical_unloads"] = len(unloads)
    result["native_reclaim_order"] = reclaim_order(events(args.output / "engine0.events.jsonl"), name_a)
    outcome_c = await asyncio.to_thread(publish, "C")
    assert outcome_c.status == "PUBLISHED"
    result["C_after_release"] = outcome_c.status
    fanout.reject_end = True
    try:
        await asyncio.to_thread(publish, "D")
    except LoRAPublicationError as exc:
        assert exc.kind == "RETRYABLE", exc
        result["single_engine_failure"] = exc.kind
    else:
        raise AssertionError("Injected engine1 End conflict must reject publication")
    finally:
        fanout.reject_end = False
    status = await registry.status.remote()
    assert status.default_version == outcome_c.version_id and status.capacity_owning == 1
    result["failure_preserves_default"] = True
    outcome_d = await asyncio.to_thread(publish, "D")
    assert outcome_d.status == "PUBLISHED"
    before = len(fanout.calls)
    replay = await asyncio.to_thread(publish, "D")
    assert replay.status == "NO_OP" and len(fanout.calls) == before
    result.update(retry=outcome_d.status, replay=replay.status, replay_engine_rpcs=0, passed=True)


def reclaim_order(trace: list[dict], name: str) -> dict:
    """Require native terminal release before physical reclamation, not just
    HTTP success."""
    resolved = next(e for e in trace if e["ev"] == "req.resolve.exit" and "retained-A" in e["rids"])
    identity = resolved["lora_id"]
    entered = next(e for e in trace if e["ev"] == "registry.wait_for_unload.enter" and e["id"] == identity)
    assert entered["counts"][identity] == 1
    released = next(
        e
        for e in trace
        if e["ev"] == "registry.release"
        and identity in e["ids"]
        and e["mono"] >= entered["mono"]
        and e["counts"][identity] == 0
    )
    exited = next(e for e in trace if e["ev"] == "registry.wait_for_unload.exit" and e["id"] == identity)
    unloaded = next(e for e in trace if e["ev"] == "manager.unload.exit" and e["name"] == name)
    assert resolved["mono"] <= entered["mono"] <= released["mono"] <= exited["mono"] <= unloaded["mono"]
    return {
        "resolved": resolved,
        "wait_enter": entered,
        "native_release": released,
        "wait_exit": exited,
        "physical_unload": unloaded,
    }
