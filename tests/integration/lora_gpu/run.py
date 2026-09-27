# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in, three-GPU joint publication/Session/logprob acceptance experiment.

Run as a script. Each engine has one sequential client, concurrently with the
other engine and publication. Cold and warm baselines are declared separately;
no selection by observed numerical closeness or tolerance adjustment is
allowed.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests.integration.lora_gpu.support import Engines, Fanout, events, init_transport


EPS = 0.002
PROMPTS = {"first": "Count from one:", "followup": "The capital of France is"}
TOKENS = 4


def arguments(baseline: Any = False) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-a", required=True)
    parser.add_argument("--adapter-b", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs=3, type=int, default=[0, 1, 2])
    parser.add_argument("--port", type=int, default=31070)
    parser.add_argument("--aot", action="store_true", help="Opt in to existing A100 activation/RMSNorm AOT workaround")
    parser.add_argument("--deterministic", action="store_true", help="Require one cold baseline across cache regimes")
    parser.add_argument("--max-progress-gap", type=float, default=2.0, help="Maximum publication no-completion gap")
    if baseline:
        parser.add_argument("--baseline-report", type=Path, required=True)
    return parser.parse_args()


def snapshot(path: Any) -> Any:
    from safetensors.torch import load_file

    from relax.distributed.checkpoint_service.lora_publication import materialize_adapter_snapshot

    directory = Path(path)
    return materialize_adapter_snapshot(
        json.loads((directory / "adapter_config.json").read_text()),
        load_file(str(directory / "adapter_model.safetensors")),
    )


def compare(actual: Any, baseline: Any) -> Any:
    same_ids = actual["tokens"] == baseline["tokens"]
    a, b = actual["logprobs"], baseline["logprobs"]
    finite = all(math.isfinite(x) for x in a + b)
    error = max((abs(x - y) for x, y in zip(a, b)), default=math.inf) if finite else math.inf
    return {
        "same_tokens": same_ids,
        "max_abs": error,
        "passed": same_ids and len(a) == len(b) == TOKENS and math.isfinite(error) and error <= EPS,
    }


def adapter(port: Any, tokenizer: Any) -> Any:
    from relax.agentic.pipeline.runtime import SGLangBackendAdapter

    result = object.__new__(SGLangBackendAdapter)
    result._args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=port,
        use_rollout_routing_replay=False,
        sglang_router_policy="cache_aware",
        slime_router_sticky=False,
    )
    result._session_lifecycle = False
    result.tokenizer = tokenizer
    result.compiler = SimpleNamespace(processor=None)
    return result


async def measure(backend: Any, inputs: Any, name: Any, sid: Any, rid: Any) -> Any:
    result = await backend.generate(
        input_ids=inputs,
        sampling_params={"max_new_tokens": TOKENS, "temperature": 0.0, "ignore_eos": True},
        session_id=sid,
        request_id=rid,
        lora_path=name,
    )
    return {
        "tokens": result.new_tokens,
        "logprobs": result.new_log_probs,
        "cached": result.meta_info.get("cached_tokens"),
        "latency": result.elapsed,
    }


async def main(args: Any, report: Any) -> None:
    import ray
    import requests
    import torch
    from transformers import AutoTokenizer

    from relax.agentic.session.lora_version import LoRAVersionRegistry
    from relax.agentic.session.service import AgenticSessionShard, ResidentGroup, _SessionRecord, _SessionResultCell
    from relax.distributed.checkpoint_service.backends.device_direct import bucket_tensor_counts
    from relax.distributed.checkpoint_service.lora_publication import LoRAPublisher, RayLoRAVersionRegistryClient
    from relax.utils.http_utils import init_http_client
    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    init_http_client(
        SimpleNamespace(
            rollout_num_gpus=2, rollout_num_gpus_per_engine=1, sglang_server_concurrency=16, use_distributed_post=False
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = {key: tokenizer.encode(text) for key, text in PROMPTS.items()}
    engines = Engines(args, args.output)
    background = []
    stop = asyncio.Event()
    fanout = None
    group = None
    baseline = {}
    try:
        # Independent clean A-only/B-only processes. Warm references must repeat
        # within the same predeclared tolerance before the joint experiment starts.
        for label, path in (("A", args.adapter_a), ("B", args.adapter_b)):
            port = args.port + 2
            proc = engines.launch(port, args.gpus[0], f"baseline-{label}")
            response = requests.post(
                f"http://127.0.0.1:{port}/load_lora_adapter", json={"lora_name": label, "lora_path": path}, timeout=120
            )
            response.raise_for_status()
            backend = adapter(port, tokenizer)
            baseline[label] = {}
            for case, ids in inputs.items():
                measurements = [
                    await measure(backend, ids, label, "baseline", f"base-{label}-{case}-{i}") for i in range(4)
                ]
                assert measurements[0]["cached"] == 0, measurements
                assert all(m["cached"] == len(ids) - 1 for m in measurements[1:]), measurements
                baseline[label][case] = {"cold": measurements[0], "warm": measurements[1]}
                for other in measurements[2:]:
                    assert compare(other, measurements[1])["passed"], measurements
                if args.deterministic:
                    assert all(compare(other, measurements[0])["passed"] for other in measurements[1:]), measurements
            report["baselines"] = baseline
            engines.stop(proc)
            logger.info("Independent %s cold/warm baselines complete", label)
        for case in inputs:
            for regime in ("cold", "warm"):
                assert not compare(baseline["A"][case][regime], baseline["B"][case][regime])["passed"], (
                    "Fixtures not distinguishable",
                    case,
                    regime,
                )

        ports = {"engine0": args.port, "engine1": args.port + 1}
        for (engine, port), gpu in zip(ports.items(), args.gpus):
            engines.launch(port, gpu, engine)
        os.environ.pop("RAY_ADDRESS", None)
        ray.init(num_cpus=2, num_gpus=0, include_dashboard=False, logging_level="ERROR")
        registry = ray.remote(LoRAVersionRegistry).options(num_cpus=0).remote(logical_capacity=2)
        shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
        shard._lora_registry = registry
        shard._versioned_lora = True
        sessions = {}

        async def session(sid: Any) -> Any:
            if sid not in sessions:
                cell = _SessionResultCell()
                resident = ResidentGroup(rollout_mode="train", group_id=sid, result_cells={sid: cell}, sessions=[])
                sessions[sid] = _SessionRecord(
                    group=resident, session_id=sid, session_sampling_params={}, result_cell=cell
                )
            return await shard._ensure_session_policy_binding(sessions[sid])

        group_name = "lora-joint-acceptance"
        group = init_transport(ports, args.gpus[2], group_name)
        fanout = Fanout(ports)
        snapshots = {"A": snapshot(args.adapter_a), "B": snapshot(args.adapter_b)}
        report["digests"] = {key: value.digest for key, value in snapshots.items()}
        current = {"snapshot": snapshots["A"]}

        def broadcast(names: Any, index: Any) -> None:
            device = torch.device("cuda", args.gpus[2])
            handles = [
                torch.distributed.broadcast(
                    current["snapshot"].tensors[name].to(device), src=0, group=group, async_op=True
                )
                for name in names
            ]
            for handle in handles:
                handle.wait()

        publisher = LoRAPublisher(
            fire=fanout.fire,
            collect=fanout.collect,
            broadcast=broadcast,
            registry=RayLoRAVersionRegistryClient(registry),
            bucket_cap_bytes=512 << 10,
            group_name=group_name,
        )

        def publish(label: Any) -> Any:
            current["snapshot"] = snapshots[label]
            tensors = current["snapshot"].tensors.values()
            return publisher.publish(
                current["snapshot"],
                bucket_tensor_counts([t.numel() * t.element_size() for t in tensors], 512 << 10),
                version_id={"A": 1, "B": 2, "C": 3, "D": 4}[label],
            )

        outcome_a = await asyncio.to_thread(publish, "A")
        names = {"A": outcome_a.lora_name}
        phase = {"name": "before"}
        records = report["requests"] = []
        report["occupancy"] = []

        async def worker(engine: Any, port: Any) -> None:
            backend = adapter(port, tokenizer)
            seen = set()
            index = 0
            while not stop.is_set():
                stage = phase["name"]
                roles = ["old"] if stage in ("before", "publishing") else ["old", "mid"]
                if stage == "after":
                    roles.append("new")
                role = roles[(index // len(inputs)) % len(roles)]
                case = list(inputs)[index % len(inputs)]
                label = "B" if role == "new" else "A"
                sid = f"{engine}-{role}"
                rid = f"{sid}-{index}"
                start = time.monotonic()
                record = {
                    "engine": engine,
                    "role": role,
                    "sid": sid,
                    "rid": rid,
                    "phase": stage,
                    "label": label,
                    "case": case,
                    "start": start,
                }
                try:
                    binding = await session(sid)
                    record["lora_path"] = binding.lora_name
                    assert binding.lora_name == names[label]
                    regime = "warm" if (label, case) in seen else "cold"
                    got = await measure(backend, inputs[case], binding.lora_name, sid, rid)
                    expected_cached = 0 if regime == "cold" else len(inputs[case]) - 1
                    record.update(
                        got,
                        regime=regime,
                        baseline_regime="cold" if args.deterministic else regime,
                        expected_cached=expected_cached,
                        comparison=compare(got, baseline[label][case]["cold" if args.deterministic else regime]),
                    )
                    record["passed"] = record["comparison"]["passed"] and got["cached"] == expected_cached
                    seen.add((label, case))
                except Exception as exc:
                    record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                record["end"] = time.monotonic()
                records.append(record)
                index += 1
                await asyncio.sleep(0.01)

        async def monitor() -> None:
            while not stop.is_set():
                status = await registry.status.remote()
                report["occupancy"].append(
                    {
                        "time": time.monotonic(),
                        "capacity_owning": status.capacity_owning,
                        "default": status.default_version,
                    }
                )
                await asyncio.sleep(0.1)

        background = [asyncio.create_task(worker(engine, port)) for engine, port in ports.items()]
        background.append(asyncio.create_task(monitor()))
        await asyncio.sleep(3)
        assert all(any(r["engine"] == engine and r.get("passed") for r in records) for engine in ports)
        phase["name"] = "publishing"
        fanout.hold_end = True
        fanout.gate.clear()
        start = time.monotonic()
        publication = asyncio.create_task(asyncio.to_thread(publish, "B"))
        background.append(publication)
        assert await asyncio.to_thread(fanout.half_ready.wait, 60), "No engine0 READY_LOCAL"
        half_ready_at = time.monotonic()
        phase["name"] = "half"
        status = await registry.status.remote()
        assert status.default_version == outcome_a.version_id
        await asyncio.sleep(3)
        fanout.gate.set()
        outcome_b = await publication
        end = time.monotonic()
        names["B"] = outcome_b.lora_name
        phase["name"] = "after"
        report["publication"] = {
            "start": start,
            "end": end,
            "duration": end - start,
            "buckets": outcome_b.bucket_count,
            "half_ready_at": half_ready_at,
        }
        await asyncio.sleep(5)
        stop.set()
        await asyncio.gather(*background)
        for record in records:
            record["completed_during_publication"] = start <= record["end"] <= end
        for engine in ports:
            for role, stage in (
                ("old", "half"),
                ("mid", "half"),
                ("old", "after"),
                ("mid", "after"),
                ("new", "after"),
            ):
                for case in inputs:
                    assert any(
                        r["engine"] == engine
                        and r["role"] == role
                        and r["phase"] == stage
                        and r["case"] == case
                        and r.get("passed")
                        for r in records
                    ), (engine, role, stage, case)
            assert any(
                r["engine"] == engine and r["completed_during_publication"] and r.get("passed") for r in records
            )
        assert records and all(r.get("passed") for r in records), "Numerical/request failure; inspect requests"
        assert all(x["capacity_owning"] <= 2 for x in report["occupancy"])
        for engine in ports:
            traces = events(args.output / f"{engine}.events.jsonl")
            assert not any(e["ev"] == "instr.error" or "flush" in e["ev"] for e in traces)
            received = {rid: e["lora"] for e in traces if e["ev"] == "req.resolve.enter" for rid in e["rids"]}
            assert all(received.get(r["rid"]) == r["lora_path"] for r in records if r["engine"] == engine)
            log = (args.output / f"{engine}.log").read_text()
            forbidden = [
                line
                for line in log.splitlines()
                if '"POST /' in line
                and any(
                    path in line
                    for path in (
                        "/pause_generation",
                        "/continue_generation",
                        "/flush_cache",
                        "/abort_request",
                        "/abort_all",
                        "/update_weights",
                    )
                )
            ]
            assert not forbidden, forbidden
        from tests.integration.lora_gpu.resources import check_resources

        fanout.hold_end = False
        await check_resources(
            args,
            report,
            registry,
            shard,
            sessions,
            snapshots,
            publish,
            publisher,
            fanout,
            adapter(ports["engine0"], tokenizer),
            inputs["first"],
            names["A"],
        )
        latencies = sorted(r["latency"] for r in records)
        report["metrics"] = {
            "completed": len(records),
            "failures": sum(not r["passed"] for r in records),
            "max_logprob_error": max(r["comparison"]["max_abs"] for r in records),
            "p50": latencies[int((len(latencies) - 1) * 0.50)],
            "p95": latencies[int((len(latencies) - 1) * 0.95)],
            "p99": latencies[int((len(latencies) - 1) * 0.99)],
        }
        report["progress"] = {}
        for engine in ports:
            entries = [r for r in records if r["engine"] == engine]
            intervals = {
                "transport": (start, half_ready_at),
                "half": (half_ready_at, end),
                "publication": (start, end),
            }
            report["progress"][engine] = {}
            for label, (lo, hi) in intervals.items():
                completed = [r for r in entries if lo <= r["end"] <= hi]
                timestamps = sorted([lo, hi] + [r["end"] for r in completed])
                report["progress"][engine][label] = {
                    "completed": len(completed),
                    "max_no_progress_s": max(b - a for a, b in zip(timestamps, timestamps[1:])),
                    "latencies": [r["latency"] for r in completed],
                }
        assert all(report["progress"][engine]["transport"]["completed"] > 0 for engine in ports)
        assert all(
            report["progress"][engine]["publication"]["max_no_progress_s"] <= args.max_progress_gap for engine in ports
        ), report["progress"]
        report["passed"] = True
        logger.info("Joint GPU acceptance passed: %s", report["metrics"])
    finally:
        stop.set()
        if fanout:
            fanout.gate.set()
        if background:
            await asyncio.gather(*background, return_exceptions=True)
        engines.close()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    plan = {
        "tolerance": EPS,
        "prompts": PROMPTS,
        "generated_tokens": TOKENS,
        "comparison": (
            "One independent cold baseline for all requests; exact tokens and finite logprobs"
            if args.deterministic
            else "Exact output tokens and each output logprob; cold: cached=0, warm: cached=input_length-1"
        ),
        "concurrency": "One sequential client per engine; two engines and NCCL publication concurrently",
        "coverage": "Production Session binding/release and SGLangBackendAdapter, no Agent process/full IR loop",
        "arguments": vars(args),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "working_tree_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
    }
    import sglang

    source_root = Path(sglang.__file__).parent
    files = list(Path(__file__).parent.rglob("*.py")) + [
        Path(args.adapter_a) / "adapter_model.safetensors",
        Path(args.adapter_a) / "adapter_config.json",
        Path(args.adapter_b) / "adapter_model.safetensors",
        Path(args.adapter_b) / "adapter_config.json",
        source_root / "srt/managers/tokenizer_manager.py",
        source_root / "srt/managers/tokenizer_control_mixin.py",
        source_root / "srt/model_executor/model_runner.py",
    ]
    plan["source_sha256"] = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2, default=str))
    report = {"passed": False, "plan": plan}
    try:
        asyncio.run(main(args, report))
    except BaseException as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
