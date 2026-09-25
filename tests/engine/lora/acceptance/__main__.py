# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One-command, explicitly selected two-GPU immutable LoRA acceptance."""

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
from uuid import uuid4

from relax.utils.logging_utils import get_logger

from .assertions import audit_evidence
from .deployment import TASKS
from .prepare import prepare
from .processes import owned_process, select_gpus


logger = get_logger(__name__)


def wait_child(process, timeout: float, label: str, log: Path) -> None:
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{label} timed out; log: {log}")
        time.sleep(1)
    if process.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"{label} failed (exit {process.returncode}); {log}\n{tail}")


def run_suite(args) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "FAIL",
        "full_acceptance": "INCOMPLETE",
        "checks": {},
        "tasks": {task: {"status": "NOT_RUN"} for task in args.tasks},
        "processes": [],
        "requested_tasks": args.tasks,
        "inference_profile": "deterministic_triton" if getattr(args, "deterministic", False) else "default",
    }
    try:
        gpus = select_gpus(args.gpus)
        report["gpus"] = gpus
        model = args.model.resolve(strict=True)
        artifacts = output / "artifacts"
        prepare(model, artifacts)
        profile = json.loads((artifacts / "verification.json").read_text())
        repo = Path(__file__).resolve().parents[4]
        packages = {}
        for name in ("torch", "sglang", "ray", "transformers", "safetensors", "peft"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "NOT_INSTALLED"
        report["software"] = {
            "python": sys.version,
            "packages": packages,
            "relax_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "relax_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
            "sglang_patch_sha256": hashlib.sha256(
                (repo / "docker/patch/sglang/v0.5.17.patch").read_bytes()
            ).hexdigest(),
        }
        profile.update(
            model_path=str(model),
            publication_config=str(artifacts / "publication.yaml"),
            artifacts_file=str(artifacts / "artifacts.json"),
            gpus=gpus,
            mem_fraction=0.2,
            agent_command=f"{sys.executable} -m tests.engine.lora.acceptance.agent",
            timeout_seconds=1800,
            startup_timeout_seconds=1200,
            deterministic=getattr(args, "deterministic", False),
        )
        if getattr(args, "overhead_profile", None):
            profile.update(args.overhead_profile)
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(gpu["uuid"] for gpu in gpus),
            PYTHONPATH=os.pathsep.join((str(repo / "tests/backends/sglang/lora_test_hooks"), str(repo))),
            OMP_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
        )
        env.pop("RAY_ADDRESS", None)
        env.pop("RELAX_LORA_NATIVE_CONFIG", None)
        env.pop("NO7_TEST_CONTROL_FILE", None)
        # Every endpoint in this isolated deployment is local. Inherited HTTP
        # proxies can intercept the engine's requests-based readiness probe
        # even while direct /health calls succeed, leaving init pending forever.
        for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env.pop(name, None)
        env.update(no_proxy="*", NO_PROXY="*")
        if "export" in args.tasks:
            training = output / "training-export"
            log = output / "training-export.log"
            command = [
                sys.executable,
                "-m",
                f"{__package__}.training_export",
                "--model",
                str(model),
                "--store",
                str(artifacts / "store"),
                "--output",
                str(training),
            ]
            logger.info("Running a real optimizer step and existing checkpoint export; %s", log)
            with owned_process(
                command, dict(env, CUDA_VISIBLE_DEVICES=gpus[0]["uuid"]), log, report["processes"]
            ) as proc:
                wait_child(proc, 1200, "training export", log)
            profile["export_evidence"] = str(training / "export.json")
        for task in args.tasks:
            # Detect leaked processes or newly started jobs before each isolated task.
            select_gpus(args.gpus)
            root = output / task
            root.mkdir()
            barrier = root / "evidence"
            barrier.mkdir()
            control = root / "control.json"
            control.write_text("{}\n")
            config = dict(
                profile, task=task, output=str(root), barrier_directory=str(barrier), test_control_file=str(control)
            )
            path = root / "config.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            log = root / "deployment.log"
            logger.info("Starting task %s; live logs: %s", task, log)
            try:
                with owned_process(
                    [sys.executable, "-m", __package__, "--worker", str(path)],
                    dict(env, NO7_TEST_CONTROL_FILE=str(control)),
                    log,
                    report["processes"],
                ) as proc:
                    wait_child(proc, max(5400, config["timeout_seconds"] + 2400), task, log)
            finally:
                evidence = root / "report.json"
                result = (
                    json.loads(evidence.read_text())
                    if evidence.exists()
                    else {"status": "FAIL", "error": "worker did not produce a report", "checks": {}}
                )
                report["tasks"][task] = result
                for name, value in result["checks"].items():
                    if value["status"] == "PASS" or name not in report["checks"]:
                        report["checks"][name] = value
            if result["status"] != "PASS":
                raise AssertionError(f"{task} did not pass")
            logger.info("Task %s PASS", task)
        report["missing_evidence"] = audit_evidence(report)
        report["full_acceptance"] = (
            "PASS" if not report["missing_evidence"] and set(args.tasks) == set(TASKS) else "INCOMPLETE"
        )
        if set(args.tasks) == set(TASKS) and report["missing_evidence"]:
            raise AssertionError(f"missing full-suite evidence: {report['missing_evidence']}")
        report["status"] = "PASS"
        if getattr(args, "overhead_profile", None):
            report["status"] = "COMPLETE"
            report["overhead_verdict"] = report["checks"]["steady_state_overhead"]["analysis"]["verdict"]
        return report
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        logger.info("Acceptance %s; report: %s", report["status"], output / "report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Local complete HF dense Qwen2/Qwen3/Llama model")
    parser.add_argument("--gpus", nargs=2, help="Two physical nvidia-smi indices; no implicit GPU selection")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic inference and Triton attention on managed and reference engines; retain graphs and cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test-results") / f"lora-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is None and (args.model is None or args.gpus is None):
        parser.error("--model and --gpus are required; services are deployed automatically")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must not contain duplicates")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.worker is not None:
            from .deployment import worker

            worker(json.loads(args.worker.read_text()))
        else:
            run_suite(args)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
