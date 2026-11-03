#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Execute a pinned conditioning-cache plan on an existing dedicated single-
node 8-GPU Ray head.

Default is preparation only. --execute is explicit; this script never starts or
stops Ray, kills a process, downloads a model, or deletes cache files. Failed
or timed-out jobs remain for inspection. Successful jobs permit name-scoped
Serve deletion only after live replica ownership was captured and revalidated.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote

from owned_serve import active_other_jobs, claim_owned_serve, validate_owned_serve
from prepare_experiments import PROTECTED, prepare, require, sha256, source_snapshot


TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}
RAY_PROBE = """
import json,sys,ray
ray.init(address=sys.argv[1],log_to_driver=False)
try:
 print('CONDITIONING_RAY='+json.dumps({'total_gpus':ray.cluster_resources().get('GPU',0),'available_gpus':ray.available_resources().get('GPU',0)}))
finally:ray.shutdown()
"""
CLEANUP = """
import json,sys,urllib.request,ray
sys.path.insert(0,sys.argv[3])
from owned_serve import active_other_jobs,delete_owned_serve
from ray import serve
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get(path):
 with opener.open(sys.argv[2]+path,timeout=15) as r:return json.load(r)
ray.init(address=sys.argv[1],log_to_driver=False)
try:
 if ray.cluster_resources().get('GPU',0)!=8:raise RuntimeError('Cluster GPU count changed')
 def snapshot():
  if active_other_jobs(get('/api/jobs/'),probe_job_id=ray.get_runtime_context().get_job_id()):
   raise RuntimeError('Another job appeared; no Serve deletion')
  return get('/api/serve/applications/')
 retries=[]
 deleted=delete_owned_serve(snapshot,lambda name:serve.delete(name,_blocking=False),json.loads(sys.argv[4]),retry_events=retries)
 print('CONDITIONING_CLEANUP='+json.dumps({'deleted_applications':deleted,'transient_retries':retries}))
finally:ray.shutdown()
"""


def write_json(path: Path, value: Any) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pending.replace(path)


def harness_sources(extra: tuple[str, ...] = ()) -> dict[str, bytes]:
    root = Path(__file__).resolve().parent
    return {
        name: (root / name).read_bytes()
        for name in ("run_experiments.py", "prepare_experiments.py", "owned_serve.py", *extra)
    }


def archive_harness(directory: Path, sources: dict[str, bytes]) -> dict[str, str]:
    target = directory / "harness"
    target.mkdir()
    for name, content in sources.items():
        (target / name).write_bytes(content)
    return {name: sha256(content) for name, content in sources.items()}


def json_output(command: list[str], prefix: str, *, env: dict[str, str] | None = None) -> dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, env=env, check=False)
    require(result.returncode == 0, f"Inspection/owned cleanup failed: {result.stderr[-6000:]}")
    lines = [line for line in result.stdout.splitlines() if line.startswith(prefix)]
    require(len(lines) == 1, f"Expected one {prefix} record")
    return json.loads(lines[0][len(prefix) :])


def read_events(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for file in sorted(directory.glob("events-rank*.jsonl"))
        for line in file.read_text().splitlines()
        if line.strip()
    ]


def has_completed_step(directory: Path, world: int) -> bool:
    ranks: set[int] = set()
    for file in directory.glob("events-rank*.jsonl"):
        # The writer may be appending its last line. This probe grants no acceptance.
        for line in file.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("stage") == "step" and event.get("step") == 0:
                ranks.add(event.get("rank"))
    return ranks == set(range(world))


def subset(expected: Any, actual: Any, label: str) -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"Observed config is not an object at {label}")
        for key, value in expected.items():
            require(key in actual, f"Missing observed config {label}.{key}")
            subset(value, actual[key], label + "." + key)
    else:
        require(expected == actual, f"Observed config differs at {label}: {actual!r} != {expected!r}")


def numerical_gate(report: dict[str, Any]) -> dict[str, Any]:
    require(
        report.get("passed") is True and report.get("cache_branch_verified") is True,
        "Numerical validation did not exercise the actual cache branch",
    )
    require(report.get("relative_tolerance") == 0, "Initial numerical gate requires rtol=0")
    loss = report["loss"]
    require(
        loss.get("finite") is True and loss.get("passed") is True and loss.get("absolute_error") == 0,
        "Loss is not finite/exact",
    )
    counts = {}
    for section in ("gradients", "parameter_updates"):
        values = report[section]
        require(isinstance(values, dict) and bool(values), f"Empty {section}")
        for name, value in values.items():
            require(value.get("passed") is True and value.get("finite") is True, f"Invalid {section}.{name}")
            for field in ("reference_l2", "actual_l2", "error_l2", "max_absolute_error"):
                number = value.get(field)
                require(
                    type(number) in (float, int) and math.isfinite(number) and number >= 0,
                    f"Invalid {section}.{name}.{field}",
                )
            require(value["error_l2"] == value["max_absolute_error"] == 0, f"Nonexact {section}.{name}")
            require(value["reference_l2"] != 0 or value["actual_l2"] == 0, f"Nonzero result for zero reference {name}")
        counts[section] = len(values)
    require(
        any(value["reference_l2"] > 0 for value in report["parameter_updates"].values()),
        "All reference FP32 updates are zero; select verification warmup/state that produces an actual update",
    )
    return {"passed": True, "tensor_counts": counts, "max_absolute_error": 0}


def validate_completed(directory: Path, request: dict[str, Any], w: dict[str, Any]) -> dict[str, Any]:
    repo = Path(w["source"]["repo"])
    script = repo / "scripts/tools/benchmark_condition_cache.py"
    spec = importlib.util.spec_from_file_location("conditioning_public_benchmark", script)
    require(spec is not None and spec.loader is not None, "Cannot load public benchmark validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.summarize_run(directory, cold_steps=request["run"]["cold_steps"])
    require(
        summary["complete"] and summary["job_complete"] and summary["source_artifact_verified"],
        "Incomplete training evidence: " + str(summary["issues"]),
    )
    rows = read_events(directory)
    world, batch = w["training"]["world_size"], w["training"]["global_batch_size"]
    for event in (x for x in rows if x["stage"] == "init"):
        require(event["world_size"] == world and event["global_batch_size"] == batch, "Observed world/batch differs")
        require(
            event["expected_steps"] == request["run"]["steps"] and event["start_step"] == 0,
            "Observed planned steps differ or resumed unexpectedly",
        )
        require(event["model_fingerprint"] == w["model"]["content_sha256"], "Model fingerprint differs")
        require(event["metadata_sha256"] == w["data"]["metadata_sha256"], "Metadata fingerprint differs")
        subset(w["training"]["semantic_config"], event["config"], "config")
    for event in (x for x in rows if x["stage"] == "step"):
        require(
            event.get("cache_errors") == 0 and event.get("cache_guard_disabled") is False,
            "Unexpected cache error or frozen guard invalidation",
        )
        reader = event.get("prefetch_stats", {})
        if reader:
            require(reader.get("reserved_bytes", 0) <= reader["max_bytes"], "Reader budget exceeded")
        cache = event.get("cache_stats", {})
        if cache:
            require(
                cache.get("resident_bytes", 0) + cache.get("bytes_reserved", 0) <= cache["max_bytes"],
                "Payload/reservation budget exceeded",
            )
            require(cache.get("staging_reserved_bytes", 0) <= cache["max_staging_bytes"], "Writer staging exceeded")
    if request["run"]["cache"] and w["cache"]["prefetch_depth"]:
        for event in (x for x in rows if x["stage"] == "close"):
            reader = event.get("prefetch_stats", {})
            require(reader.get("closed") is True and reader.get("drained") is True, "Reader not drained at close")
    if request["run"]["purpose"] == "verify":
        numerical_gate(json.loads((directory / "conditioning-verification.json").read_text()))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    args = parser.parse_args()
    require(args.poll_seconds > 0 and args.timeout_seconds > 0, "Timeouts must be positive")
    raw = args.plan.read_bytes()
    tool_sources = harness_sources()
    plan = json.loads(raw)
    prepared = prepare(plan, check_local=args.execute)
    if not args.execute:
        print(json.dumps({"prepared_runs": len(prepared["requests"]), "submitted": False, "plan_sha256": sha256(raw)}))
        return
    cluster = plan["cluster"]
    require(
        cluster["expected_gpus"] == 8 and all(w["training"]["world_size"] == 8 for w in plan["workloads"].values()),
        "This execution harness is restricted to the authorized 8-GPU node",
    )
    require(all(row["purpose"] != "cancel" for row in plan["runs"]), "Cancellation uses a separate owned observer")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(path: str) -> Any:
        with opener.open(cluster["dashboard"] + path, timeout=15) as response:
            return json.load(response)

    lock_path = Path(cluster["lock_file"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+b")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = Path(plan["runs_root"])
    root.mkdir(parents=True, exist_ok=True)
    progress = (root / "progress.jsonl").open("a", buffering=1)

    def record(event: str, **values: Any) -> None:
        text = json.dumps({"event": event, "unix": time.time(), **values}, allow_nan=False)
        progress.write(text + "\n")
        print(text, flush=True)

    def no_others(job_id: str | None = None) -> None:
        require(not active_other_jobs(get("/api/jobs/"), submission_id=job_id), "Another active job appeared")

    def resources() -> dict[str, Any]:
        ray = json_output([sys.executable, "-c", RAY_PROBE, cluster["ray_address"]], "CONDITIONING_RAY=")
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"], text=True
        )
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True
        )
        memory = {int(x.split(",")[0]): int(x.split(",")[1].strip().split()[0]) for x in gpu.splitlines() if x.strip()}
        require(
            ray["total_gpus"] == ray["available_gpus"] == 8
            and memory == dict.fromkeys(range(8), 0)
            and not processes.strip(),
            "GPUs/Ray resources are not idle",
        )
        shm = os.statvfs("/dev/shm")
        return {
            "ray": ray,
            "gpu_memory_csv_rows": gpu.splitlines(),
            "compute_empty": True,
            "shm_used_bytes": (shm.f_blocks - shm.f_bfree) * shm.f_frsize,
        }

    def namespaces(cache_root: str) -> list[str]:
        managed = Path(cache_root) / "relax-conditioning-v1"
        return sorted(x.name for x in managed.iterdir() if x.is_dir()) if managed.exists() else []

    record("runner_start", plan_sha256=sha256(raw), pid=os.getpid())
    try:
        for request in prepared["requests"]:
            row = request["run"]
            w = plan["workloads"][row["workload"]]
            directory = Path(request["directory"])
            require(not directory.exists(), f"Existing evidence is immutable; use a new name: {directory}")
            no_others()
            require(
                get("/api/serve/applications/").get("applications") == {}, "Unowned Serve apps exist; refusing launch"
            )
            require(not namespaces(w["cache"]["root"]), "Cache namespace already exists; no deletion attempted")
            before = resources()
            require(args.plan.read_bytes() == raw, "Plan changed")
            require(harness_sources() == tool_sources, "Experiment harness changed")
            prepare(plan, check_local=True)
            base, patch, source = source_snapshot(Path(w["source"]["repo"]))
            require(source == request["implementation_sha256"], "Source changed")
            try:
                get("/api/jobs/" + quote(request["job_id"], safe=""))
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
            else:
                raise RuntimeError("Job ID already exists; choose another name")
            directory.mkdir()
            submitted = time.time()
            request.update(
                base_commit=base,
                started_unix=submitted,
                plan_sha256=sha256(raw),
                runner_sha256=sha256(Path(__file__).read_bytes()),
                metadata_sha256=w["data"]["metadata_sha256"],
                idle_resources=before,
                harness_sha256=archive_harness(directory, tool_sources),
            )
            write_json(directory / "request.json", request)
            (directory / "implementation.patch").write_bytes(patch)
            env = {
                key: value
                for key, value in os.environ.items()
                if key not in PROTECTED and not key.startswith(("RELAX_CONDITION_", "FSDP_", "SFT_"))
            }
            env.update(request["env"])
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
            with (directory / "submit.log").open("w") as log:
                result = subprocess.run(
                    request["command"],
                    cwd=w["source"]["repo"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            require(result.returncode == 0, "Submission failed; preserved evidence, no cleanup attempted")
            owned = None
            deadline = time.monotonic() + args.timeout_seconds
            while True:
                no_others(request["job_id"])
                job = get("/api/jobs/" + quote(request["job_id"], safe=""))
                write_json(directory / "job.json", job)
                if owned is None and has_completed_step(directory, 8):
                    snapshot = get("/api/serve/applications/")
                    if snapshot.get("applications"):
                        owned = claim_owned_serve(
                            snapshot, submitted, directory, expected_deployments=cluster["serve_deployments"]
                        )
                        write_json(directory / "owned-serve.json", owned)
                record("job_status", run=row["name"], status=job["status"])
                if job["status"] in TERMINAL:
                    logs = get("/api/jobs/" + quote(request["job_id"], safe="") + "/logs")
                    (directory / "driver.log").write_text(logs.get("logs", ""))
                    break
                require(time.monotonic() < deadline, "Job timeout; left untouched for inspection")
                time.sleep(args.poll_seconds)
            require(job["status"] == "SUCCEEDED", "Job failed/stopped; no implicit cleanup or retry")
            require(
                args.plan.read_bytes() == raw and source_snapshot(Path(w["source"]["repo"]))[2] == source,
                "Plan/source changed during execution",
            )
            require(harness_sources() == tool_sources, "Experiment harness changed during execution")
            validation = validate_completed(directory, request, w)
            write_json(directory / "validation.json", validation)
            no_others()
            apps = get("/api/serve/applications/")
            if apps.get("applications"):
                require(owned is not None, "Live ownership was not captured; refusing cleanup")
                validate_owned_serve(apps, owned)
                cleanup = json_output(
                    [
                        sys.executable,
                        "-c",
                        CLEANUP,
                        cluster["ray_address"],
                        cluster["dashboard"],
                        str(Path(__file__).resolve().parent),
                        json.dumps(owned),
                    ],
                    "CONDITIONING_CLEANUP=",
                    env=env,
                )
                write_json(directory / "owned-cleanup.json", cleanup)
            deadline = time.monotonic() + 120
            while get("/api/serve/applications/").get("applications"):
                no_others()
                require(time.monotonic() < deadline, "Owned Serve release timeout")
                time.sleep(1)
            deadline = time.monotonic() + 120
            while True:
                no_others()
                try:
                    after = resources()
                    break
                except ValueError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(1)
            require(not namespaces(w["cache"]["root"]), "Normal completion leaked a cache namespace")
            after.update(
                unix=time.time(),
                job_id=request["job_id"],
                namespace_count=0,
                payload_file_count=0,
                payload_logical_bytes=0,
                scope="post_job_idle; excluded from Ray job wall",
            )
            write_json(directory / "resource_after.json", after)
            record("run_complete", run=row["name"], job_id=request["job_id"])
        record("runner_complete")
    except BaseException as error:
        record("runner_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        progress.close()
        lock.close()


if __name__ == "__main__":
    main()
