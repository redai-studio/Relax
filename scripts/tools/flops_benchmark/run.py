# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run paired experiments on an existing dedicated Ray cluster."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from relax.utils.logging_utils import get_logger
from scripts.tools.flops_benchmark.checkpoint import prepare_view, verify_view
from scripts.tools.flops_benchmark.common import digest, file_digest, write_json
from scripts.tools.flops_benchmark.compare import compare
from scripts.tools.flops_benchmark.summarize import summarize


logger = get_logger(__name__)
MODELS = {"dense": ("Qwen3.5-4B", 4), "moe": ("Qwen3.5-35B-A3B", 8)}


def environment_snapshot() -> dict:
    packages = {}
    for name in (
        "torch",
        "triton",
        "transformer-engine",
        "flash-attn",
        "fla-core",
        "flash-linear-attention",
        "ray",
        "transformers",
        "transferqueue",
        "nvidia-resiliency-ext",
        "numpy",
        "pyarrow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    def command(args: list[str]) -> str:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"Preflight command failed: {args[0]}")
        return result.stdout.strip()

    revision = os.environ.get("RELAX_FLOPS_CODE_REVISION") or command(["git", "rev-parse", "HEAD"])
    sources = {
        str(path): file_digest(path)
        for root in ("relax", "configs", "scripts/models", "scripts/training/sft", "scripts/tools/flops_benchmark")
        for path in sorted(Path(root).rglob("*"))
        if path.is_file() and path.suffix in {".py", ".sh", ".json", ".yaml"}
    }
    megatron_root = Path(os.environ["MEGATRON"])
    megatron_sources = {
        str(path.relative_to(megatron_root)): file_digest(path) for path in sorted(megatron_root.rglob("*.py"))
    }
    if not megatron_sources:
        raise ValueError("MEGATRON must point to the pinned, patched sources")
    return {
        "packages": packages,
        "code_revision": revision,
        "source_files": sources,
        "megatron_source_files": megatron_sources,
        "container_image": os.environ.get("RELAX_FLOPS_IMAGE_DIGEST"),
        "python": platform.python_version(),
        "cpu": {
            row["field"]: row["data"]
            for row in json.loads(command(["lscpu", "--json"]))["lscpu"]
            if "MHz" not in row["field"]
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "runtime_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "NCCL_SOCKET_IFNAME",
                "GLOO_SOCKET_IFNAME",
                "NVTE_FLASH_ATTN",
                "NVTE_FUSED_ATTN",
                "RELAX_SFT_SELECTIVE_LOGITS",
            )
        },
        "gpus": command(["nvidia-smi", "--query-gpu=name,memory.total,uuid,driver_version", "--format=csv,noheader"]),
        "topology": command(["nvidia-smi", "topo", "-m"]),
    }


class Experiments:
    def __init__(self, args: argparse.Namespace) -> None:
        from ray.job_submission import JobSubmissionClient

        self.args = args
        self.model, self.gpus = MODELS[args.model]
        self.tag = self.model.lower() + "-flops" + ("-" + args.name if args.name else "")
        self.model_dir, self.data_dir, self.save_dir = (
            Path(os.environ[key]) for key in ("MODEL_DIR", "DATA_DIR", "SAVE_DIR")
        )
        self.address = os.environ["RAY_JOB_ADDRESS"].rstrip("/")
        self.client = JobSubmissionClient(self.address)
        self.environment = environment_snapshot()
        self.root = self.save_dir / f"{self.tag}-evidence"
        self.root.mkdir(parents=True, exist_ok=True)
        environment_path = self.root / "environment.json"
        if environment_path.exists() and json.loads(environment_path.read_text()) != self.environment:
            raise ValueError("Experiment environment or source files changed; use a fresh --name")
        write_json(environment_path, self.environment)
        self.script = Path(f"scripts/training/sft/run-{self.model.lower()}-{self.gpus}xgpu-flops.sh")
        self.tolerances = Path(__file__).with_name("tolerances.json")
        self.verified_views: set[str] = set()

    def workload(self, recipe: str = "stratified", batch_size: int = 16, capacity: int = 24576) -> tuple[str, Path]:
        relative = f"{self.model}/{recipe}-gbs{batch_size}-cap{capacity}"
        manifest_path = self.data_dir / relative / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if file_digest(manifest_path.parent / manifest["data_file"]) != manifest["data_sha256"]:
            raise ValueError("Frozen workload file was modified")
        return relative, manifest_path

    def job(
        self,
        name: str,
        mode: str,
        arm: str,
        workload: str,
        *,
        layout: str = "default",
        end_step: int,
        load_name: str,
        budget: int | None = None,
    ) -> dict:
        import requests

        experiment = f"{self.tag}-{name}"
        record_path = self.root / f"{name}.json"
        manifest_path = self.data_dir / workload / "manifest.json"
        request = {
            "arm": arm,
            "mode": mode,
            "layout": layout,
            "end_step": end_step,
            "load_name": load_name,
            "workload": workload,
            "budget": budget,
            "script_sha256": file_digest(self.script),
            "manifest_sha256": file_digest(manifest_path),
            "environment_id": digest(self.environment),
        }
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record["status"] == "SUCCEEDED" and all(record.get(key) == value for key, value in request.items()):
                logger.info("Reusing completed experiment %s", name)
                return record
            raise RuntimeError(f"Previous attempt {name} did not succeed; inspect it before choosing a new name")
        if any(
            str(job.status) in {"RUNNING", "PENDING", "JobStatus.RUNNING", "JobStatus.PENDING"}
            for job in self.client.list_jobs()
        ):
            raise RuntimeError("Benchmark requires an idle, dedicated Ray cluster")
        # Only this benchmark's dedicated cluster is accepted above.
        response = requests.delete(f"{self.address}/api/serve/applications/", timeout=30)
        response.raise_for_status()
        import ray

        ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
        cleanup_deadline = time.monotonic() + 60
        while True:
            state = requests.get(f"{self.address}/api/serve/applications/", timeout=10)
            state.raise_for_status()
            if not state.json()["applications"] and ray.available_resources().get("GPU", 0) >= self.gpus:
                break
            if time.monotonic() >= cleanup_deadline:
                raise TimeoutError("Previous benchmark deployments did not release their GPUs")
            time.sleep(2)
        job_id = experiment + "-" + str(time.time_ns())
        env = {
            **os.environ,
            "RELAX_SKIP_CLUSTER_CLEANUP": "1",
            "RAY_NO_WAIT": "1",
            "EXP_NAME": experiment,
            "FLOPS_BENCH_MODE": mode,
            "FLOPS_BENCH_ARM": arm,
            "FLOPS_BENCH_LAYOUT": layout,
            "FLOPS_BENCH_WORKLOAD": workload,
            "FLOPS_BENCH_END_STEP": str(end_step),
            "FLOPS_BENCH_LOAD_NAME": load_name,
            "FLOPS_BENCH_JOB_ID": job_id,
        }
        if budget is not None:
            env["FLOPS_BENCH_BUDGET"] = str(budget)
        else:
            env.pop("FLOPS_BENCH_BUDGET", None)
        submission = subprocess.run(
            ["bash", "scripts/entrypoint/ray-job.sh", str(self.script)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        (self.root / f"{name}.submit.log").write_text(submission.stdout + submission.stderr)
        if submission.returncode:
            raise RuntimeError(f"Ray submission failed: {name}; see submission log")
        deadline = time.monotonic() + self.args.job_timeout
        timed_out = False
        while True:
            status = self.client.get_job_status(job_id)
            if status.is_terminal():
                break
            if time.monotonic() > deadline:
                self.client.stop_job(job_id)
                timed_out = True
                break
            time.sleep(5)
        log = self.root / f"{name}.log"
        log.write_text(self.client.get_job_logs(job_id))
        record = {
            **request,
            "job_id": job_id,
            "status": "TIMED_OUT" if timed_out else status.value,
            "log": str(log),
            "experiment": experiment,
            "workload": workload,
            "budget": budget,
            "script_sha256": file_digest(self.script),
        }
        if timed_out:
            write_json(record_path, record)
            raise TimeoutError(f"Stopped timed-out benchmark job {job_id}; see {log}")
        if status.value != "SUCCEEDED":
            write_json(record_path, record)
            raise RuntimeError(f"Benchmark job {name} ended with {status.value}; see {log}")
        if mode != "warm":
            text = log.read_text().lower()
            for failure in (
                "falling back to the hf checkpoint",
                "falling back to args.hf_checkpoint",
                "use args.hf_checkpoint",
                "loading model weights only",
                "training restarts from scratch",
            ):
                if failure in text:
                    record["status"] = "INVALID_RESTORE"
                    write_json(record_path, record)
                    raise RuntimeError(f"Checkpoint did not restore the requested state: {failure}")
        write_json(record_path, record)
        logger.info("Completed %s", name)
        return record

    def warm(self, layout: str = "default") -> str:
        workload, _ = self.workload()
        name = f"{layout}-warm"
        native_name = f"{self.tag}-{name}-native"
        view = self.save_dir / native_name
        self.job(name, "warm", "baseline", workload, layout=layout, end_step=20, load_name="")
        if not (view / "fingerprint.json").exists():
            prepare_view(self.save_dir / f"{self.tag}-{name}", view)
            self.verified_views.add(native_name)
        elif native_name not in self.verified_views:
            verify_view(view)
            self.verified_views.add(native_name)
        return native_name

    def numerical(self, layout: str = "default", *, controls: bool = True) -> None:
        workload, _ = self.workload()
        load_name = self.warm(layout)
        arms = [("baseline", "baseline", None)]
        if controls:
            arms += [("repeat", "baseline", None), ("regroup", "baseline", 12288)]
        arms.append(("flops", "flops", None))
        reference: dict | None = None
        for label, arm, budget in arms:
            record = self.job(
                f"{layout}-numeric-{label}",
                "capture",
                arm,
                workload,
                layout=layout,
                end_step=21,
                load_name=load_name,
                budget=budget,
            )
            if reference is None:
                reference = record
                continue
            reference_capture = self.save_dir / reference["experiment"] / "capture"
            candidate_capture = self.save_dir / record["experiment"] / "capture"
            report_path = self.root / f"{layout}-numeric-{label}-comparison.json"
            comparison_inputs = {
                "reference_job": digest(reference),
                "candidate_job": digest(record),
                "reference_log": file_digest(Path(reference["log"])),
                "candidate_log": file_digest(Path(record["log"])),
                "tolerances": file_digest(self.tolerances),
                "manifests": {
                    str(path.relative_to(self.save_dir)): file_digest(path)
                    for root in (reference_capture, candidate_capture)
                    for path in sorted(root.glob("rank-*/manifest.json"))
                },
            }
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if report.get("inputs") != comparison_inputs:
                    raise ValueError("Completed numerical comparison inputs changed")
            else:
                report = compare(
                    reference_capture,
                    candidate_capture,
                    Path(reference["log"]),
                    Path(record["log"]),
                    self.tolerances,
                )
                report["inputs"] = comparison_inputs
                write_json(report_path, report)
            if not report["passed"]:
                raise RuntimeError(f"Numerical gate failed: {layout}/{label}")
            # Preserve manifests and reports; only release successfully compared candidates.
            if not (candidate_capture / "released.json").exists():
                for path in candidate_capture.glob("rank-*/manifest.json"):
                    for shard in json.loads(path.read_text())["shards"]:
                        file = path.parent / shard["file"]
                        if file.name != shard["file"]:
                            raise ValueError("Invalid capture file path")
                        file.unlink(missing_ok=True)
                write_json(candidate_capture / "released.json", {"report_sha256": file_digest(report_path)})

    def performance(
        self, recipe: str = "stratified", batch_size: int = 16, capacity: int = 24576, pairs: int = 3, steps: int = 30
    ) -> None:
        workload, manifest = self.workload(recipe, batch_size, capacity)
        load_name = self.warm()
        fingerprint = json.loads((self.save_dir / load_name / "fingerprint.json").read_text())
        configuration_id = digest(
            {
                "environment": self.environment,
                "manifest": file_digest(manifest),
                "initial_state": fingerprint["initial_state_id"],
                "script": file_digest(self.script),
            }
        )
        records = []
        prefix = f"{recipe}-gbs{batch_size}-cap{capacity}"
        for pair in range(pairs):
            order = ("baseline", "flops") if pair % 2 == 0 else ("flops", "baseline")
            for arm in order:
                record = self.job(
                    f"{prefix}-pair{pair}-{arm}", "perf", arm, workload, end_step=20 + steps, load_name=load_name
                )
                records.append(
                    {
                        **record,
                        "pair": pair,
                        "start_step": 20,
                        "steps": steps,
                        "warmup": 10,
                        "configuration_id": configuration_id,
                    }
                )
                write_json(self.root / f"{prefix}-runs.json", records)
        report = summarize(manifest, self.root / f"{prefix}-runs.json")
        write_json(self.root / f"{prefix}-summary.json", report)

    def audit(self) -> None:
        workload, _ = self.workload()
        load_name = self.warm()
        for arm in ("baseline", "flops"):
            self.job(f"audit-{arm}", "audit", arm, workload, end_step=23, load_name=load_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--suite", choices=("full", "numeric", "performance", "coverage", "audit"), default="full")
    parser.add_argument("--job-timeout", type=int, default=1800)
    parser.add_argument("--name", default="", help="Stable suffix for a fresh experiment, e.g. retry1")
    args = parser.parse_args()
    experiment = Experiments(args)
    if args.suite in {"full", "numeric", "performance", "coverage"}:
        experiment.numerical()
    if args.suite in {"full", "performance"}:
        experiment.performance()
        if args.model == "dense":
            experiment.performance("natural", pairs=1, steps=60)
            experiment.performance("natural", batch_size=128, capacity=8192, pairs=1)
    if args.suite in {"full", "audit"}:
        experiment.audit()
    if args.suite in {"full", "coverage"} and args.model == "dense":
        for layout in ("tp", "pp", "vpp", "cp"):
            experiment.numerical(layout, controls=False)


if __name__ == "__main__":
    main()
