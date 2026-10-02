# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import json
import math
import time
from uuid import uuid4

from . import support
from .assertions import compare, compare_decode, scores, worker_observations


async def cache_direction(ctx, source, target, label):
    records = []
    ctx.report["checks"][label] = {"status": "INCOMPLETE", "observations": records}
    separation = 0
    for i, prompt in enumerate(ctx.prompts):
        # Separate direction prefixes and diagnostic prefixes from warmup.
        ids = ctx.tokenizer.encode(f"No7 {label} diagnostic {i}: " + prompt, add_special_tokens=True)
        expected = await support.cold_diagnostic(ctx, target[0]["binding"]["version_id"], ids)
        source_score = await support.cold_diagnostic(ctx, source[0]["binding"]["version_id"], ids)
        separation += sum(
            abs(source_score[token] - expected[token]) > 10 * (ctx.atol + ctx.rtol * abs(expected[token]))
            for token in expected
        )
        for warm, probe in zip(source, target, strict=True):
            if warm["engine"] != probe["engine"]:
                raise AssertionError("cache diagnostic must remain on the same engine")
            cache_key = "no7-cache-" + uuid4().hex
            source_output = await support.diagnostic(ctx, warm, ids, extra_key=cache_key)
            record = {"engine": probe["engine"], "prefix": ids, "source": source_output, "cold": expected}
            records.append(record)
            compare(scores(source_output), source_score, atol=ctx.atol, rtol=ctx.rtol)
            output = await support.diagnostic(ctx, probe, ids, extra_key=cache_key)
            record["cross"] = output
            if output["meta_info"].get("cached_tokens") != 0:
                raise AssertionError("first target request hit another adapter's cache namespace")
            error = compare(scores(output), expected, atol=ctx.atol, rtol=ctx.rtol)
            own = await support.diagnostic(ctx, probe, ids, extra_key=cache_key)
            record["same"] = own
            if own["meta_info"].get("cached_tokens", 0) <= 0:
                raise AssertionError("same-version cache positive control did not hit")
            compare(scores(own), expected, atol=ctx.atol, rtol=ctx.rtol)
            record.update(
                {
                    "engine": probe["engine"],
                    "prefix": ids,
                    "error": error,
                    "cross_cached_tokens": output["meta_info"].get("cached_tokens"),
                    "same_cached_tokens": own["meta_info"].get("cached_tokens"),
                    "actual": scores(output),
                    "cold": expected,
                }
            )
    if separation < 8:
        raise AssertionError("A/B fixtures lack the pre-registered numerical separation")
    ctx.report["checks"][label] = {"status": "PASS", "observations": records, "separated_positions": separation}


async def mixed_batches(ctx, left, right, baseline_left, baseline_right, label):
    support.configure_test(ctx, capture_mixed_batches=True, capture_graph=True)
    traces = []
    ctx.report["checks"][label] = {"status": "INCOMPLETE", "observations": traces}
    state = await support.request(ctx, "GET", "/lora/versions")
    expected_graphs = set()
    for engine_index, (a, b) in enumerate(zip(left, right, strict=True)):
        if a["engine"] != b["engine"]:
            raise AssertionError("mixed-batch test needs both adapters on the same engine")
        engine_id = a["engine"]["engine_id"]
        version = state["versions"][a["binding"]["version_id"]]
        ranks = a["execution_workers"]
        if ranks != b["execution_workers"]:
            raise AssertionError("mixed adapters must share the same attention DP route")
        if not set(map(str, ranks)) <= worker_observations(version["ready"][engine_id]).keys():
            raise AssertionError("execution rank is outside the prepared instance")
        members = {a["native_lora_id"], b["native_lora_id"]}
        coverage = {(uid, rank): set() for uid in members for rank in ranks}
        expected_graphs.update((uid, rank) for uid in members for rank in ranks)
        for repetition in range(16):
            ids = ctx.tokenizer.encode(
                f"No7 mixed {label} {engine_index} {repetition}: " + ctx.prompts[repetition] * 8,
                add_special_tokens=True,
            )
            cache_key = "no7-mixed-" + uuid4().hex
            outputs = await asyncio.gather(
                support.diagnostic(ctx, a, ids, tokens=8, extra_key=cache_key),
                support.diagnostic(ctx, b, ids, tokens=8, extra_key=cache_key),
            )
            for bound, output, baseline_version in zip((a, b), outputs, (baseline_left, baseline_right), strict=True):
                generated = output["output_ids"]
                trace = {"engine": bound["engine"], "prefix": ids, "actual": output}
                traces.append(trace)
                if output["meta_info"].get("cached_tokens") != 0:
                    raise AssertionError("mixed-batch decode must start with an isolated cold prefix")
                expected = await support.reference_decode(ctx, baseline_version, ids, tokens=8)
                trace["baseline"] = expected
                proof = {}
                for rank in ranks:
                    rid = output["verification_completion"]["rid"]
                    proof_id = rid if rank == 0 else f"{rid}.rank-{rank}"
                    path = ctx.root / (proof_id + ".mixed.json")
                    proof[rank] = json.loads(path.read_text()) if path.exists() else {}
                    for phase, batch in proof[rank].items():
                        if (
                            set(batch["members"]) != members
                            or batch["native_lora_id"] != bound["native_lora_id"]
                            or batch["worker_rank"] != rank
                        ):
                            raise AssertionError("mixed batch proof contains the wrong native instances/rank")
                        coverage[bound["native_lora_id"], rank].add(phase)
                trace.update(
                    {
                        "engine": bound["engine"],
                        "rid": output["verification_completion"]["rid"],
                        "proof": proof,
                        "prefix": ids,
                        "output_ids": generated,
                    }
                )
                trace["errors"] = compare_decode(output, expected, atol=ctx.atol, rtol=ctx.rtol)
            if all(phases >= {"prefill", "decode"} for phases in coverage.values()):
                break
        else:
            raise AssertionError(f"missing actual mixed-batch coverage: {coverage}")
    ctx.report["checks"][label] = {"status": "PASS", "observations": traces}
    graphs = [json.loads(path.read_text()) for path in ctx.root.glob("*.graph.json")]
    if not expected_graphs <= {(item["native_lora_id"], item["worker_rank"]) for item in graphs if item["replayed"]}:
        raise AssertionError("each execution rank must execute actual decode graph replay")
    ctx.report["checks"]["cuda_graph"] = {"status": "PASS", "workers": graphs}
    support.configure_test(ctx)


async def calibration(ctx):
    ids = ctx.tokenizer.encode("No7 isolated KV calibration: " + ctx.prompts[0], add_special_tokens=True)
    requested = ctx.config["kv_diagnostics"]
    rids = [uuid4().hex, uuid4().hex]
    support.configure_test(ctx, capture_kv={rid: requested for rid in rids})
    values = []
    try:
        for name, rid in zip(("A", "B"), rids, strict=True):
            baseline = ctx.config["baselines"][ctx.versions[name]]
            response = await ctx.client.post(
                baseline["url"].rstrip("/") + "/generate",
                json={
                    "rid": rid,
                    "input_ids": ids,
                    "lora_path": baseline["lora_path"],
                    "return_logprob": True,
                    "logprob_start_len": 0,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                },
            )
            response.raise_for_status()
            path = ctx.root / (rid + ".kv.json")
            deadline = time.monotonic() + 30
            while not path.exists():
                if time.monotonic() > deadline:
                    raise AssertionError("isolated baseline did not produce real native KV capture")
                await asyncio.sleep(0.02)
            values.append(json.loads(path.read_text()))
        differences = 0
        for name in ("k", "v"):
            a, b = values[0][name], values[1][name]
            if len(a) != len(b) or not all(math.isfinite(value) for value in a + b):
                raise AssertionError("invalid native KV capture")
            differences += sum(
                abs(x - y) > 10 * (ctx.config["kv_atol"] + ctx.config["kv_rtol"] * abs(y))
                for x, y in zip(a, b, strict=True)
            )
        if differences < 8:
            raise AssertionError("fixtures do not change enough pre-registered native K/V elements")
        ctx.report["checks"]["kv_sensitivity"] = {"status": "PASS", "captures": values, "differences": differences}
    finally:
        support.configure_test(ctx)


async def wrong_kv_controls(ctx, left, right):
    observations = []
    ctx.report["checks"]["wrong_kv_negative_control"] = {"status": "INCOMPLETE", "observations": observations}
    for label, sources, targets in (("A-to-B", left, right), ("B-to-A", right, left)):
        for index, (source, target) in enumerate(zip(sources, targets, strict=True)):
            ids = ctx.tokenizer.encode(f"No7 wrong-KV {label} {index}: " + ctx.prompts[index], add_special_tokens=True)
            expected = await support.cold_diagnostic(ctx, target["binding"]["version_id"], ids)
            cache_key = "no7-wrong-kv-" + uuid4().hex
            positive = await support.diagnostic(ctx, target, ids, extra_key=cache_key)
            record = {"direction": label, "engine": target["engine"], "positive": positive, "cold": expected}
            observations.append(record)
            compare(scores(positive), expected, atol=ctx.atol, rtol=ctx.rtol)
            warm_positive = await support.diagnostic(ctx, target, ids, extra_key=cache_key)
            record["warm_positive"] = warm_positive
            if warm_positive["meta_info"].get("cached_tokens", 0) <= 0:
                raise AssertionError("wrong-KV normal control did not hit its own cache")
            compare(scores(warm_positive), expected, atol=ctx.atol, rtol=ctx.rtol)
            await support.diagnostic(ctx, source, ids, extra_key=cache_key)
            rid = uuid4().hex
            support.configure_test(
                ctx,
                cache_alias={rid: {"actual_uid": target["native_lora_id"], "cached_uid": source["native_lora_id"]}},
            )
            try:
                output = await support.diagnostic(ctx, target, ids, rid=rid, extra_key=cache_key)
            finally:
                support.configure_test(ctx)
            alias = json.loads((ctx.root / (rid + ".alias.json")).read_text())
            record.update(alias=alias, negative=output)
            if alias["actual_uid"] != target["native_lora_id"] or output["meta_info"].get("cached_tokens", 0) <= 0:
                raise AssertionError("negative control did not reuse the wrong native KV")
            actual = scores(output)
            if actual.keys() != expected.keys():
                raise AssertionError("wrong KV control lost diagnostic token scores")
            try:
                compare(actual, expected, atol=ctx.atol, rtol=ctx.rtol)
            except AssertionError:
                record.update(
                    {
                        "direction": label,
                        "engine": target["engine"],
                        "alias": alias,
                        "actual": scores(output),
                        "cold": expected,
                        "cached_tokens": output["meta_info"]["cached_tokens"],
                    }
                )
            else:
                raise AssertionError("wrong KV was not detected; cache isolation experiment is invalid")
    ctx.report["checks"]["wrong_kv_negative_control"] = {"status": "PASS", "observations": observations}


async def failed_publication(ctx):
    support.configure_test(
        ctx,
        fail_prepare={"engine_boot_id": ctx.native_a[1]["engine"]["boot_id"]},
        replay_old_ack=True,
    )
    try:
        operation = await support.request(
            ctx, "POST", "/lora/publications", {"version_id": ctx.versions["B"], "request_id": uuid4().hex}
        )
        failed = await support.wait_state(ctx, operation["operation_id"], "ABORTED")
        current = await support.request(ctx, "GET", "/lora/versions")
        if current["default"]["version_id"] != ctx.versions["A"] or len(failed["absence_receipts"]) != 2:
            raise AssertionError("failed prepare did not preserve A and clean both targets")
        proof = json.loads((ctx.root / "control.late-ack.json").read_text())
        if not proof["unchanged"] or proof["old"] == proof["active"]:
            raise AssertionError("native control late-ACK injection did not execute")
        ctx.report["checks"]["failure_late_ack"] = {"status": "PASS", "failed": failed, "ack": proof}
        return operation["operation_id"]
    finally:
        support.configure_test(ctx)


async def partial_publication(ctx, old_cases, retry_of):
    barrier = ctx.root / "hold-E2-prepare"
    barrier.touch()
    engine_boot_id = ctx.native_a[1]["engine"]["boot_id"]
    support.configure_test(ctx, hold_prepare={"engine_boot_id": engine_boot_id, "barrier": str(barrier)})
    operation = await support.request(
        ctx,
        "POST",
        "/lora/publications",
        {"version_id": ctx.versions["B"], "request_id": uuid4().hex, "retry_of": retry_of},
    )
    op_id = operation["operation_id"]
    deadline = time.monotonic() + ctx.config["timeout_seconds"]
    try:
        while True:
            state = await support.request(ctx, "GET", "/lora/publications/" + op_id)
            if state["state"] != "PREPARING":
                raise AssertionError("partial readiness barrier failed: " + str(state))
            if len(state["ready"]) == 1:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("E1 never became READY")
            await asyncio.sleep(0.02)
        during = await support.start_case(ctx, "during-partial", 0)
        old_cases.append(during)
        current = await support.request(ctx, "GET", "/lora/versions")
        if current["default"]["version_id"] != ctx.versions["A"]:
            raise AssertionError("partial prepare changed default")
        await support.request(ctx, "POST", "/lora/publications/" + op_id + "/cancel")
        await support.wait_state(ctx, op_id, "ABORTED")
        # The held E2 prepare runs only after retire installed its fence.
        ctx.report["checks"]["partial_readiness"] = {"status": "PASS", "observation": state}
    finally:
        barrier.unlink(missing_ok=True)
        support.configure_test(ctx)
    return op_id


async def gpu_drain_race(ctx, operation_id, clear_uid):
    """Retire A under a real delayed instance event while B keeps
    generating."""
    bound = ctx.native_a[0]
    rid = uuid4().hex
    support.configure_test(
        ctx,
        delay_last_use={
            "rid": rid,
            "cycles": ctx.config["fault_delay_cycles"],
            "worker_rank": min(bound["execution_workers"]),
        },
        delay_clear={"uid": clear_uid, "cycles": ctx.config["fault_delay_cycles"]},
        capture_retired_slots=[item["native_lora_id"] for item in ctx.native_a],
    )
    ids = ctx.tokenizer.encode("No7 physical completion race " + ctx.prompts[0], add_special_tokens=True)
    result = await support.diagnostic(ctx, bound, ids, rid=rid)
    evidence = await support.wait_file(ctx, ctx.root / (rid + ".last-use.json"))
    if not evidence["pending"]:
        raise AssertionError("GPU dependency completed before the experiment could observe it")
    payload = support.attempt_payload(ctx, bound, rid)
    cancellations = []
    for _ in range(2):
        response = await ctx.client.post(bound["engine"]["endpoint"] + "/cancel_lora_attempt", json=payload)
        response.raise_for_status()
        if response.json()["state"] != "REQUEST_FINISHED":
            raise AssertionError("late cancel changed a completed native request")
        cancellations.append(response.json())
    for item in ctx.native_a:
        await support.close_native(ctx, item)
    waiting = await support.wait_file(ctx, ctx.root / (rid + ".retire-wait.json"))
    if waiting["actual_unload_count"] or not waiting["pending"]:
        raise AssertionError("instance retirement did not wait for the last GPU reader")
    before = await support.request(ctx, "GET", "/lora/publications/" + operation_id)
    if before["state"] != "RETIRING":
        raise AssertionError("A must retain its business slot during physical drain")
    same_engine_b = next(item for item in ctx.native_b if item["engine"] == bound["engine"])
    await support.diagnostic(
        ctx, same_engine_b, ctx.tokenizer.encode("B advances while A waits", add_special_tokens=True)
    )
    if (ctx.root / (rid + ".last-use-done.json")).exists():
        raise AssertionError("delay profile too short to prove B progress during A physical drain")
    rejected = await ctx.client.post(
        ctx.base + "/lora/publications", json={"version_id": ctx.versions["C"], "request_id": uuid4().hex}
    )
    if rejected.status_code != 507:
        raise AssertionError("C was admitted before A's last GPU use completed")
    ctx.report["checks"]["last_gpu_use"] = {
        "status": "PASS",
        "injection": evidence,
        "retirement": waiting,
        "manager": before,
        "logical_completion": result["verification_completion"],
    }
    ctx.report["checks"]["cancel_finish_race"] = {"status": "PASS", "cancellations": cancellations}
    # Keep hooks enabled: the caller independently checks the slot-clear event.


async def stale_slot_negative_control(ctx, retired, replacements):
    evidence = []
    ctx.report["checks"]["slot_reuse_negative_control"] = {"status": "INCOMPLETE", "observations": evidence}
    try:
        for index, (old, current) in enumerate(zip(retired, replacements, strict=True)):
            if old["engine"] != current["engine"]:
                raise AssertionError("slot negative control engine mismatch")
            rid = uuid4().hex
            ids = ctx.tokenizer.encode(
                "No7 stale slot negative control " + ctx.prompts[index], add_special_tokens=True
            )
            expected = await support.cold_diagnostic(ctx, ctx.versions["B"], ids)
            positive = await support.diagnostic(ctx, current, ids, extra_key=uuid4().hex)
            record = {"engine": current["engine"], "positive": positive, "baseline": expected}
            evidence.append(record)
            compare(scores(positive), expected, atol=ctx.atol, rtol=ctx.rtol)
            support.configure_test(
                ctx,
                stale_slot={"rid": rid, "actual_uid": current["native_lora_id"], "retired_uid": old["native_lora_id"]},
            )
            output = await support.diagnostic(ctx, current, ids, rid=rid, extra_key=uuid4().hex)
            record["negative"] = output
            ranks = output["verification_completion"]["execution_workers"]
            if not ranks or set(ranks) != set(current["execution_workers"]):
                raise AssertionError("stale slot control lacks execution worker evidence")
            injection = []
            for rank in ranks:
                proof_id = rid if rank == 0 else f"{rid}.rank-{rank}"
                injection.append(await support.wait_file(ctx, ctx.root / (proof_id + ".stale-slot.json")))
            actual = scores(output)
            if actual.keys() != expected.keys():
                raise AssertionError("stale slot control lost diagnostic token scores")
            try:
                compare(actual, expected, atol=ctx.atol, rtol=ctx.rtol)
            except AssertionError:
                record.update(injection=injection, actual=actual)
            else:
                raise AssertionError("fixture did not detect retired A weights in C's slot")
            support.configure_test(ctx)
            restored = await support.diagnostic(ctx, current, ids, extra_key=uuid4().hex)
            record["restored"] = restored
            compare(scores(restored), expected, atol=ctx.atol, rtol=ctx.rtol)
    finally:
        support.configure_test(ctx)
    ctx.report["checks"]["slot_reuse_negative_control"] = {"status": "PASS", "observations": evidence}


async def memory_handoff(ctx, bindings):
    """Exercise actual memory saver outside the no-pause publication window."""
    if not ctx.config.get("memory_handoff", False):
        ctx.report["checks"]["memory_handoff"] = {"status": "NOT_REQUESTED"}
        return
    before = await support.request(ctx, "GET", "/lora/versions")
    for bound in bindings:
        await support.diagnostic(ctx, bound, ctx.tokenizer.encode("LoRA memory handoff warm prefix"), tokens=4)
    # Existing Agentic tool-wait sessions also remain alive across this handoff.
    await ctx.manager.offload.remote()
    suspended = await support.request(ctx, "GET", "/lora/versions")
    if not suspended["suspended"] or suspended["occupied"] != before["occupied"]:
        raise AssertionError("offload lost version occupancy or did not close admission")
    rejected = await ctx.manager.lora_control.remote(
        "bind", {"owner_epoch": ctx.native_owner, "session_id": "suspended-" + uuid4().hex}
    )
    if rejected.get("error_code") != "ENGINE_SUSPENDED":
        raise AssertionError("suspended engine accepted a new Session")
    for version, record in before["versions"].items():
        after = suspended["versions"][version]
        if (record["session_refs"], record["binding"], record["absent"]) != (
            after["session_refs"],
            after["binding"],
            after["absent"],
        ):
            raise AssertionError("offload changed Session references or retired an adapter")
    await ctx.manager.onload_weights.remote()
    partial = await support.request(ctx, "GET", "/lora/versions")
    if not partial["suspended"]:
        raise AssertionError("weights-only resume admitted requests before KV/graphs restored")
    await ctx.manager.onload_kv.remote()
    restored = await support.request(ctx, "GET", "/lora/versions")
    if restored["suspended"] or restored["default"] != before["default"]:
        raise AssertionError("resume failed to restore the same default")
    directory = ctx.root / "memory-graph"
    directory.mkdir()
    support.configure_test(ctx, directory=str(directory), capture_graph=True)
    results = []
    try:
        for index, bound in enumerate(bindings):
            replay = await ctx.manager.lora_control.remote(
                "bind", {"owner_epoch": ctx.native_owner, "session_id": bound["session_id"]}
            )
            if any(replay[field] != bound[field] for field in ("binding", "engine", "native_lora_id")):
                raise AssertionError("resume rebound a retained Session")
            ids = ctx.tokenizer.encode(f"LoRA memory handoff continuation {index}")
            expected = await support.cold_diagnostic(ctx, bound["binding"]["version_id"], ids)
            actual = await support.diagnostic(ctx, bound, ids, tokens=8)
            compare(scores(actual), expected, atol=ctx.atol, rtol=ctx.rtol)
            results.append(actual)
        graphs = [json.loads(path.read_text()) for path in directory.glob("*.graph.json")]
        expected_workers = {
            (key, rank)
            for key, receipt in before["versions"][ctx.versions["B"]]["ready"].items()
            for rank in worker_observations(receipt)
        }
        observed_workers = {(str(item["engine_id"]), str(item["worker_rank"])) for item in graphs if item["replayed"]}
        if not expected_workers <= observed_workers:
            raise AssertionError("no actual CUDA graph replay on every worker after memory restore")
    finally:
        support.configure_test(ctx)
    ctx.report["checks"]["memory_handoff"] = {
        "status": "PASS",
        "before": before,
        "suspended": suspended,
        "restored": restored,
        "numerical": results,
        "graphs": graphs,
    }
