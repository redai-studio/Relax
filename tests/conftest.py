# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep resource evidence beside JUnit without changing test selection."""

import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


def _resources() -> dict:
    report = {
        "pid": os.getpid(),
        "container_hostname": os.uname().nodename,
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "max_rss_unit": "bytes" if sys.platform == "darwin" else "KiB",
        "threads_env": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }
    for name in (
        "cpu.max",
        "cpuset.cpus.effective",
        "memory.max",
        "memory.current",
        "memory.peak",
        "cpu.stat",
        "pids.max",
        "pids.current",
    ):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            report[name] = path.read_text().strip()
    for controller, names in {
        "cpu": ("cpu.cfs_quota_us", "cpu.cfs_period_us", "cpu.shares", "cpu.stat"),
        "cpu,cpuacct": ("cpu.cfs_quota_us", "cpu.cfs_period_us", "cpu.shares", "cpu.stat"),
        "cpuset": ("cpuset.cpus", "cpuset.effective_cpus"),
        "memory": ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes"),
        "pids": ("pids.max", "pids.current"),
    }.items():
        for name in names:
            path = Path("/sys/fs/cgroup") / controller / name
            if path.is_file():
                report[f"cgroup_v1/{controller}/{name}"] = path.read_text().strip()
    for name in ("cgroup", "mountinfo"):
        path = Path("/proc/self") / name
        if path.is_file():
            report[f"proc_self_{name}"] = "\n".join(
                line for line in path.read_text().splitlines() if name == "cgroup" or "cgroup" in line
            )
    auto_init_hook = sys.modules.get("ray._private.auto_init_hook")
    report["ray_auto_init_enabled"] = getattr(auto_init_hook, "enable_auto_connect", None)
    try:
        import psutil
    except ImportError:
        report["tcp_connections"] = "psutil unavailable"
    else:
        try:
            report["tcp_connections"] = [
                {"pid": item.pid, "local": str(item.laddr), "remote": str(item.raddr), "status": item.status}
                for item in psutil.net_connections(kind="tcp")
            ]
        except psutil.AccessDenied:
            report["tcp_connections"] = "access denied"
    torch = sys.modules.get("torch")
    if torch is not None:
        report["torch_threads"] = torch.get_num_threads()
        report["torch_interop_threads"] = torch.get_num_interop_threads()
    return report


def _sample(stop: threading.Event, path: Path) -> None:
    try:
        import psutil
    except ImportError:
        psutil = None
    nvidia_smi = shutil.which("nvidia-smi")
    with path.open("w") as output:
        while not stop.is_set():
            sample = {"time": time.time()}
            if psutil is not None:
                processes = []
                for process in [psutil.Process(), *psutil.Process().children(recursive=True)]:
                    try:
                        processes.append(
                            {
                                "pid": process.pid,
                                "created": process.create_time(),
                                "name": process.name(),
                                "rss": process.memory_info().rss,
                                "threads": process.num_threads(),
                                "cpu_seconds": sum(process.cpu_times()[:2]),
                            }
                        )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                sample["processes"] = processes
            if nvidia_smi:
                for label, query in (
                    ("gpu", "--query-gpu=uuid,memory.used,memory.total,utilization.gpu"),
                    ("gpu_processes", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory"),
                ):
                    try:
                        result = subprocess.run(
                            [nvidia_smi, query, "--format=csv,noheader,nounits"],
                            capture_output=True,
                            text=True,
                            timeout=3,
                        )
                        sample[label] = {"status": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        sample[label] = {"error": str(exc)}
            output.write(json.dumps(sample) + "\n")
            output.flush()
            stop.wait(1)


def pytest_sessionstart(session: pytest.Session) -> None:
    config = session.config
    xmlpath = config.getoption("xmlpath", default=None)
    if not xmlpath or hasattr(config, "workerinput"):
        return
    directory = Path(xmlpath).parent
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "resources-controller-start.json").write_text(json.dumps(_resources(), indent=2) + "\n")
    stop = threading.Event()
    sampler = threading.Thread(target=_sample, args=(stop, directory / "resource-samples.jsonl"), daemon=True)
    config._relax_resource_sampler = (stop, sampler)
    sampler.start()


def pytest_sessionfinish(session: pytest.Session) -> None:
    config = session.config
    xmlpath = config.getoption("xmlpath", default=None)
    if not xmlpath:
        return
    sampler = getattr(config, "_relax_resource_sampler", None)
    if sampler is not None:
        stop, thread = sampler
        stop.set()
        thread.join(timeout=8)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    Path(xmlpath).parent.joinpath(f"resources-{worker}-finish.json").write_text(
        json.dumps(_resources(), indent=2) + "\n"
    )
