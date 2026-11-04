"""Prepare assets and run fixed online/cache SFT comparisons on selected
GPUs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.performance_summary import cumulative_payback

from runner.common import CANCEL_REQUESTED, digest, execute, write_json


KIT = Path(__file__).resolve().parents[1]
REPO = KIT.parents[1]


def plan(config: dict, prepared: dict, hw: dict, work: Path, python: str) -> list[dict]:
    world = hw["world"]
    cache = config["cpu_cache_total_bytes"] // world
    rows = []

    def add(name: str, model: str, metadata: str, mode: str, cached: bool, trajectory: str | None = None):
        source = prepared["models"][model]
        directory = work / "stages" / name
        argv = [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={world}",
            "-m",
            "experiments.native_runner",
            "--mode",
            mode,
            "--adapter",
            config["models"][model]["adapter"],
            "--model-path",
            source["path"],
            "--metadata",
            prepared["data"][metadata],
            "--dataset-size",
            str({"debug8": 8, "high128": 128, "low1024": 1024}[metadata]),
            "--data-root",
            prepared["data"]["root"],
            "--run-dir",
            str(directory),
            "--initial-state-dir",
            str(work / "initial" / model),
            "--world",
            str(world),
            "--global-batch",
            "8",
            "--lora-rank",
            "32",
            "--seed",
            str(config["seed"]),
            "--max-pixels",
            str(config["max_pixels"]),
            "--attention-implementation",
            config["attention_implementation"],
            "--cache-identity",
            source["identity"],
            "--cache-device",
            "cpu",
            "--cache-bytes",
            str(cache if cached else 0),
            "--warmup",
            str(config["warmup_steps"]),
            "--measure",
            str(config["measure_steps"]),
        ]
        if trajectory:
            argv += ["--trajectory", trajectory]
        rows.append(
            {
                "name": name,
                "model": model,
                "kind": mode,
                "cached": cached,
                "argv": argv,
                "run_dir": str(directory),
                "result": str(directory / "result.json"),
            }
        )

    for model in config["models"]:
        for trajectory in ["a1", "a2", "bcold", "bwarm"]:
            add(
                f"{model}-correctness-{trajectory}",
                model,
                "debug8",
                "correctness",
                trajectory.startswith("b"),
                trajectory,
            )
        paths = {name: work / "stages" / f"{model}-correctness-{name}" for name in ["a1", "a2", "bcold", "bwarm"]}
        output = work / "stages" / f"{model}-correctness-compare.json"
        argv = [python, "-m", "experiments.compare"]
        for name, value in paths.items():
            argv += [f"--{name}", str(value)]
        rows.append(
            {
                "name": f"{model}-correctness-compare",
                "model": model,
                "kind": "compare",
                "argv": argv + ["--output", str(output)],
                "result": str(output),
            }
        )
    for workload in config["workloads"]:
        metadata = Path(workload["metadata"]).stem
        for repeat in range(config["paired_repetitions"]):
            for letter in ["a", "b"] if repeat != 1 else ["b", "a"]:
                add(
                    f"{workload['name']}-pair{repeat + 1}-{letter}",
                    workload["model"],
                    metadata,
                    "performance",
                    letter == "b",
                )
    return rows


def summarize(config: dict, stages: list[dict], work: Path) -> dict:
    import statistics

    numerical = {}
    for model in config["models"]:
        path = work / "stages" / f"{model}-correctness-compare.json"
        numerical[model] = (
            json.loads(path.read_text()) if path.exists() else {"passed": False, "status": "not_completed"}
        )
    workloads = {}
    for row in config["workloads"]:
        pairs = []
        for repeat in range(1, config["paired_repetitions"] + 1):
            paths = [work / "stages" / f"{row['name']}-pair{repeat}-{letter}" / "result.json" for letter in ["a", "b"]]
            if not all(path.exists() for path in paths):
                continue
            a, b = [json.loads(path.read_text()) for path in paths]
            if a.get("native_contract") != b.get("native_contract"):
                raise ValueError("A/B native model/data/optimizer/precision contract mismatch")
            for result in [a, b]:
                if (
                    result.get("status") != "completed"
                    or result["warmup_updates"] != config["warmup_steps"]
                    or result["measured_updates"] != config["measure_steps"]
                ):
                    raise ValueError("Incomplete or modified measurement window")
            online = a["measured_seconds"]
            cached = b["measured_seconds"]
            a_total = a["initialization_plus_total_seconds"]
            b_total = b["initialization_plus_total_seconds"]
            saving_per_update = (online - cached) / config["measure_steps"]
            cold_extra = max(0, b["warmup_seconds"] - a["warmup_seconds"])
            pairs.append(
                {
                    "repeat": repeat,
                    "online_seconds": online,
                    "cached_seconds": cached,
                    "steady_throughput_gain": online / cached - 1,
                    "online_full_stage_seconds": a_total,
                    "cached_full_stage_seconds": b_total,
                    "cold_extra_seconds": cold_extra,
                    "projected_payback_updates_from_warmup": cold_extra / saving_per_update
                    if saving_per_update > 0
                    else None,
                    **cumulative_payback(a, b),
                    "cache_last_snapshot": b["resources_and_cache"][-1]["cache"],
                    "scope": b["scope"],
                }
            )
        gains = [pair["steady_throughput_gain"] for pair in pairs]
        workloads[row["name"]] = {
            "pairs": pairs,
            "completed_pairs": len(pairs),
            "mean_gain": statistics.mean(gains) if gains else None,
            "median_gain": statistics.median(gains) if gains else None,
            "min_gain": min(gains) if gains else None,
            "max_gain": max(gains) if gains else None,
            "sample_stddev_gain": statistics.stdev(gains) if len(gains) > 1 else None,
            "scope": row["scope"],
        }
    complete = all(item.get("passed") for item in numerical.values()) and all(
        v["completed_pairs"] == 3 for v in workloads.values()
    )
    strict_numerical = all(item.get("strict_acceptance") is True for item in numerical.values())
    repeated = [v for k, v in workloads.items() if not k.endswith("low_reuse")]
    speed = any(v["median_gain"] is not None and v["median_gain"] >= 0.05 for v in repeated)
    return {
        "schema": 1,
        "numerical": numerical,
        "performance": workloads,
        "campaign_matrix_completed": complete,
        "hard_numerical_bounds_met": all(item.get("passed") for item in numerical.values()),
        "strict_numerical_acceptance": strict_numerical,
        "numerical_review_required": any(item.get("needs_nondeterminism_review") for item in numerical.values()),
        "technical_thresholds_met": complete and strict_numerical and speed,
        "official_acceptance": "not determined by this example",
        "scope": "Native Relax FSDPSFTTrainer including preparation/fwd/bwd/reduction/Adam/scheduler; excludes Ray/TQ and checkpoint I/O",
        "not_a_converged_model_quality_or_leaderboard_score": True,
    }


def validate_prepared(path: Path, *, verify_bytes: bool = True) -> dict[str, Any]:
    """Verify the public asset lock and the exported image/metadata
    receipts."""
    from assets.prepare import identity

    prepared = json.loads(path.read_text())
    lock_path = KIT / "assets/manifest.json"
    lock = json.loads(lock_path.read_text())
    if prepared.get("state") != "READY" or prepared.get("input_manifest_sha256") != digest(lock_path):
        raise ValueError("Prepared assets do not match assets/manifest.json; run prepare again")

    def check(file: Path, expected: dict) -> None:
        if not file.is_file() or file.stat().st_size != expected["size"]:
            raise ValueError(f"Missing or resized asset: {file}")
        if verify_bytes and digest(file) != expected["sha256"]:
            raise ValueError(f"Asset SHA-256 mismatch: {file}")

    for name, spec in lock["models"].items():
        model = prepared["models"][name]
        if model["identity"] != identity(spec["files"]):
            raise ValueError(f"Model identity mismatch: {name}")
        for item in spec["files"]:
            check(Path(model["path"]) / item["path"], item)
    data = prepared["data"]
    ready = json.loads(Path(data["receipt"]).read_text())
    if ready.get("source_identity") != identity(lock["data"]["files"]):
        raise ValueError("Data receipt belongs to another source revision")
    for subset, count in (("debug8", 8), ("high128", 128), ("low1024", 1024)):
        metadata = Path(data[subset])
        if digest(metadata) != ready["metadata_sha256"][subset] or len(json.loads(metadata.read_text())) != count:
            raise ValueError(f"Changed metadata: {metadata}")
    for item in ready["images"]:
        check(Path(data["root"]) / item["path"], item)
    return prepared


def source_identity() -> str:
    """Bind resume to the experiment and the native model/trainer
    implementation."""
    files = set()
    for tree in (KIT / "experiments", KIT / "runner", REPO / "relax/backends/fsdp", REPO / "relax/models/sensenova"):
        files.update(tree.glob("*.py"))
    files.update(
        REPO / "relax" / relative for relative in ("models/condition_cache.py", "models/qwen_image/sft_adapter.py")
    )
    manifest = {str(path.relative_to(REPO)): digest(path) for path in sorted(files)}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def validate_recipe(config: dict) -> None:
    """Reject config edits that the fixed native benchmark would otherwise
    ignore."""
    fixed = {
        "global_batch": 8,
        "lora_rank": 32,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "seed": 20261030,
        "warmup_steps": 16,
        "measure_steps": 64,
        "paired_repetitions": 3,
        "attention_implementation": "torch",
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"The fixed native benchmark requires {key}={expected!r}")
    acceptance = config["acceptance"]
    if (
        acceptance["relative_loss_grad_delta_l2_max"] != 0.01
        or acceptance["at_least_one_model_steady_speedup"] != 0.05
    ):
        raise ValueError("The numerical 1% and throughput 5% thresholds are fixed")


def run(args: argparse.Namespace) -> int:
    """Run one sequential matrix; resume skips only unchanged, verified
    results."""
    import fcntl

    config = json.loads(args.config.read_text())
    validate_recipe(config)
    prepared = validate_prepared(args.prepared.resolve())
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(devices) < args.world or len(set(devices)) != len(devices):
        raise ValueError("--devices must name enough distinct GPUs for --world")
    if args.stage_timeout_seconds <= 0 or args.wall_hours <= 0:
        raise ValueError("Timeouts must be positive")
    contract = {
        "config": config,
        "prepared": prepared,
        "world": args.world,
        "devices": devices,
        "source_identity": source_identity(),
    }
    state_path = work / "campaign-state.json"
    with (work / ".campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another campaign owns this work directory") from error
        if state_path.exists():
            if not args.resume:
                raise FileExistsError("Existing campaign: use --resume or a new --work")
            state = json.loads(state_path.read_text())
            if state["contract"] != contract:
                raise ValueError("Resume requires the same source, assets, recipe and devices")
        else:
            now = time.time()
            state = {
                "status": "running",
                "started_unix": now,
                "deadline_unix": now + args.wall_hours * 3600,
                "contract": contract,
                "completed": [],
                "result_digests": {},
            }
        write_json(work / "config.json", config)
        stages = plan(config, prepared, {"world": args.world}, work, sys.executable)
        write_json(work / "plan.json", stages)
        write_json(state_path, state)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        env["PYTHONPATH"] = os.pathsep.join([str(KIT), str(REPO), env.get("PYTHONPATH", "")])
        env.setdefault("OMP_NUM_THREADS", "1")
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda _signum, _frame: CANCEL_REQUESTED.set())
        try:
            import torch

            hardware = {
                "world": args.world,
                "devices": devices,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("diffusers", "transformers", "peft", "accelerate")
                },
            }
            query = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import json,torch; print(json.dumps(["
                    "{'name':torch.cuda.get_device_name(i),'total_memory':torch.cuda.get_device_properties(i).total_memory}"
                    " for i in range(torch.cuda.device_count())]))",
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            hardware["visible_gpus"] = json.loads(query.stdout)
            if len(hardware["visible_gpus"]) < args.world:
                raise RuntimeError("Not enough visible CUDA GPUs for the requested world size")
            write_json(work / "environment.json", hardware)
            for stage in stages:
                result = Path(stage["result"])
                if stage["name"] in state["completed"]:
                    if not result.is_file() or digest(result) != state["result_digests"][stage["name"]]:
                        raise ValueError(f"Previously completed evidence changed: {result}")
                    continue
                if CANCEL_REQUESTED.is_set() or time.time() >= state["deadline_unix"]:
                    state["status"] = "interrupted_or_deadline"
                    break
                attempts = len(list((work / "receipts").glob(f"{stage['name']}-attempt*.json")))
                directory = Path(stage.get("run_dir", ""))
                if stage.get("run_dir") and directory.exists():
                    failed = work / "failed" / f"{stage['name']}-attempt{attempts - 1}"
                    failed.parent.mkdir(parents=True, exist_ok=True)
                    directory.rename(failed)
                elif result.exists():
                    failed = work / "failed" / f"{stage['name']}-attempt{attempts - 1}.json"
                    failed.parent.mkdir(parents=True, exist_ok=True)
                    result.rename(failed)
                state.update(status="running", current_stage=stage["name"])
                write_json(state_path, state)
                execute(
                    stage["argv"],
                    KIT,
                    env,
                    work / "receipts" / f"{stage['name']}-attempt{attempts}.json",
                    min(args.stage_timeout_seconds, state["deadline_unix"] - time.time()),
                )
                record = json.loads(result.read_text())
                if not record.get("passed"):
                    raise RuntimeError(f"Stage did not pass: {stage['name']}")
                state["completed"].append(stage["name"])
                state["result_digests"][stage["name"]] = digest(result)
                write_json(state_path, state)
            else:
                state["status"] = "matrix_completed"
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            state["last_updated_unix"] = time.time()
            write_json(state_path, state)
            write_json(work / "summary.json", summarize(config, stages, work))
    return 0 if state["status"] == "matrix_completed" else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Download SHA-locked public models and export real edit pairs")
    prepare.add_argument("--assets", type=Path, required=True)
    prepare.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    validate = commands.add_parser("validate", help="Read back all locked asset sizes and SHA-256 values")
    validate.add_argument("--prepared", type=Path, required=True)
    train = commands.add_parser("run", help="Run both models and all A/B pairs sequentially")
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--work", type=Path, required=True)
    train.add_argument("--devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    train.add_argument("--world", type=int, default=1, choices=(1, 2, 4))
    train.add_argument("--config", type=Path, default=KIT / "configs/campaign.json")
    train.add_argument("--stage-timeout-seconds", type=float, default=10800)
    train.add_argument("--wall-hours", type=float, default=24)
    train.add_argument("--resume", action="store_true")
    aggregate = commands.add_parser("summarize", help="Summarize every recorded pair, including regressions")
    aggregate.add_argument("--work", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        execute(
            [
                sys.executable,
                "-m",
                "assets.prepare",
                "--output",
                str(args.assets.resolve()),
                "--workers",
                str(args.workers),
            ],
            KIT,
            dict(os.environ),
            args.assets.resolve() / "prepare-receipt.json",
            24 * 3600,
        )
    elif args.command == "validate":
        validate_prepared(args.prepared.resolve())
        print("All locked model/data bytes verified")
    elif args.command == "run":
        return run(args)
    else:
        config_file = args.work / "config.json"
        config = json.loads((config_file if config_file.exists() else KIT / "configs/campaign.json").read_text())
        write_json(args.work / "summary.json", summarize(config, [], args.work))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
