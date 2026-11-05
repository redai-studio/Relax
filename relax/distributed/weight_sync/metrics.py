# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Per-update timings, application byte counters and process-tree memory."""

import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

from relax.utils.logging_utils import get_logger

from .diagnostics import save_diagnostic
from .store import canonical
from .traffic import IOObservation, capture


logger = get_logger(__name__)
_memory_slot = threading.BoundedSemaphore(1)


_stage_id = ContextVar("weight_sync_stage_id", default=None)


class SyncMetrics:
    def __init__(self, update_id: str) -> None:
        self.update_id = update_id
        self.seconds: dict[str, float] = {}
        self.bytes: dict[str, int] = dict.fromkeys(("G", "W", "X", "T", "C"), 0)
        self.observed: set[str] = set()
        self.details: list[dict] = []
        self.observations: list[dict] = []
        self.related_reports: list[dict] = []
        self.gaps: list[dict] = []
        self._lock = threading.Lock()
        self.operation_id = uuid.uuid4().hex
        self.source_participant = None

    @contextmanager
    def stage(self, name: str, *, path=None) -> Iterator[None]:
        start = time.monotonic()
        parent = _stage_id.get()
        stage_id = uuid.uuid4().hex
        token = _stage_id.set(stage_id)
        observation = IOObservation(
            path=path, interfaces=tuple(filter(None, os.environ.get("RELAX_WEIGHT_METRIC_INTERFACES", "").split(",")))
        )
        try:
            with observation:
                yield
        finally:
            _stage_id.reset(token)
            with self._lock:
                self.seconds[name] = self.seconds.get(name, 0.0) + time.monotonic() - start
                self.observations.append(
                    {
                        "stage": name,
                        "stage_id": stage_id,
                        "parent_stage_id": parent,
                        "path": str(path) if path is not None else None,
                        "io_attribution": "process_inclusive_of_nested_stages",
                        **observation.report,
                    }
                )

    def related(self, report: dict) -> None:
        """Keep separately attributed work without adding it to parent
        bytes."""
        with self._lock:
            self.related_reports.append(report)

    def unmeasured(self, stage: str, reason: str) -> None:
        with self._lock:
            self.gaps.append({"stage": stage, "reason": reason, "monotonic_ns": time.monotonic_ns()})

    def transferred(self, stage: str, count: int, *, link: str, scope: str = "application", attempt: int = 1) -> None:
        if stage not in self.bytes or type(count) is not int or count < 0:
            raise ValueError("Invalid communication counter")
        with self._lock:
            self.bytes[stage] += count
            self.observed.add(stage)
            self.details.append(
                {
                    "event_id": f"{self.operation_id}:{len(self.details)}",
                    "operation_id": self.operation_id,
                    "attempt": attempt,
                    "monotonic_ns": time.monotonic_ns(),
                    "stage": stage,
                    "bytes": count,
                    "link": link,
                    "scope": scope,
                }
            )

    def report(self) -> dict:
        with self._lock:
            return {
                "update_id": self.update_id,
                "operation_id": self.operation_id,
                "source_participant": self.source_participant,
                "seconds": dict(self.seconds),
                "bytes": {key: value if key in self.observed else None for key, value in self.bytes.items()},
                "counter_scope": "application"
                if all(item["scope"] == "application" for item in self.details)
                else "separate_views",
                "bytes_by_scope": {
                    scope: {
                        stage: sum(
                            item["bytes"] for item in self.details if item["scope"] == scope and item["stage"] == stage
                        )
                        for stage in self.bytes
                    }
                    for scope in {item["scope"] for item in self.details}
                },
                "counter_coverage_complete": False,
                "application_stage_coverage_complete": self.observed == set(self.bytes) and not self.gaps,
                "event_coverage": sorted(self.observed),
                "physical_link_coverage_complete": False,
                "measurement_level": "application_events",
                "links": list(self.details),
                "observations": list(self.observations),
                "related_reports": list(self.related_reports),
                "unmeasured": list(self.gaps),
            }

    def save(self, path: str | Path) -> None:
        with capture(None):
            save_diagnostic(Path(path), canonical(self.report()))


def process_memory(pid: int) -> dict:
    """Read Linux process-tree PSS, including file-backed pages exactly
    once."""
    pending, seen = [pid], set()
    totals = {"rss_bytes": 0, "pss_bytes": 0, "anonymous_bytes": 0, "swap_bytes": 0, "processes": 0}
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            children = Path(f"/proc/{current}/task/{current}/children").read_text().split()
            pending.extend(int(child) for child in children)
            lines = Path(f"/proc/{current}/smaps_rollup").read_text().splitlines()
        except (OSError, ValueError):
            continue
        fields = {}
        for line in lines:
            parts = line.split()
            if len(parts) == 3 and parts[2] == "kB":
                fields[parts[0].rstrip(":")] = int(parts[1]) * 1024
        for output, key in (
            ("rss_bytes", "Rss"),
            ("pss_bytes", "Pss"),
            ("anonymous_bytes", "Anonymous"),
            ("swap_bytes", "Swap"),
        ):
            totals[output] += fields.get(key, 0)
        totals["processes"] += 1
    return totals


def cgroup_memory() -> dict:
    """Report the current Linux cgroup, including unmapped file-cache pages."""
    try:
        line = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
        root = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
        fields = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
        return {
            "cgroup_current_bytes": int((root / "memory.current").read_text()),
            "cgroup_anon_bytes": int(fields.get("anon", 0)),
            "cgroup_file_cache_bytes": int(fields.get("file", 0)),
            "cgroup_file_mapped_bytes": int(fields.get("file_mapped", 0)),
            "cgroup_inactive_file_bytes": int(fields.get("inactive_file", 0)),
            "cgroup_shmem_bytes": int(fields.get("shmem", 0)),
            "cgroup_file_dirty_bytes": int(fields.get("file_dirty", 0)),
            "cgroup_file_writeback_bytes": int(fields.get("file_writeback", 0)),
        }
    except (OSError, ValueError, StopIteration):
        return {}


class MemorySampler:
    """Sample memory with a five-percent periodic scan budget.

    ``interval`` is the minimum pause. Expensive process-tree PSS walks extend
    that pause. Entry and exit waits are bounded; coverage gaps are reported.
    Reported peaks are sampled lower bounds, not exact allocation peaks.
    """

    def __init__(
        self,
        interval: float = 0.1,
        *,
        track_gpu: bool = False,
        trace_path: str | Path | None = None,
        wait_timeout: float = 0.2,
    ) -> None:
        if not math.isfinite(interval) or interval < 0.01:
            raise ValueError("Sampling interval must be at least 10 ms")
        if not math.isfinite(wait_timeout) or wait_timeout <= 0:
            raise ValueError("Sampling wait must be positive and finite")
        self.wait_timeout = wait_timeout
        self.interval = interval
        self._next_wait_seconds = interval
        self.sampling = {
            "minimum_interval_seconds": interval,
            "periodic_scan_budget_fraction": 0.05,
            "boundary_samples_outside_budget": True,
            "sampled_peaks_are_lower_bounds": True,
            "samples": 0,
            "status": "not_started",
            "errors": [],
            "boundary_wait_seconds": wait_timeout,
            "total_scan_seconds": 0.0,
            "maximum_scan_seconds": 0.0,
        }
        self.peak: dict[str, int] = {}
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self._trace = None
        self.trace_error = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._started = False
        self._abandoned = False
        self.track_gpu = track_gpu
        self._thread = threading.Thread(target=self._run, name="weight-sync-memory", daemon=True)
        self._nvml = None
        self._gpu_baseline = None
        self.gpu_status = "not_requested"

    def _initialize_gpu(self) -> None:
        if self.track_gpu:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
                self.gpu_status = "sampled_nvml_process_footprint"
            except Exception as error:
                self.gpu_status = f"unavailable:{type(error).__name__}"

    def _warning(self, reason: str) -> None:
        self.sampling["status"] = "partial"
        self.sampling["errors"].append(reason)
        logger.warning("Memory observation incomplete: %s; weight update continues.", reason)

    def _close_trace(self) -> None:
        trace, self._trace = self._trace, None
        if trace is not None:
            try:
                trace.close()
            except OSError as error:
                self.trace_error = str(error)

    def _gpu_memory(self) -> dict:
        if self._nvml is None:
            return {}
        pending, pids = [os.getpid()], set()
        while pending:
            pid = pending.pop()
            if pid in pids:
                continue
            pids.add(pid)
            try:
                pending.extend(int(child) for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
            except (OSError, ValueError):
                pass
        try:
            used = 0
            seen_gpu_processes = matched_gpu_processes = 0
            for index in range(self._nvml.nvmlDeviceGetCount()):
                device = self._nvml.nvmlDeviceGetHandleByIndex(index)
                for process in self._nvml.nvmlDeviceGetComputeRunningProcesses(device):
                    seen_gpu_processes += 1
                    if process.pid in pids and 0 <= process.usedGpuMemory < (1 << 63):
                        matched_gpu_processes += 1
                        used += process.usedGpuMemory
            if seen_gpu_processes and not matched_gpu_processes:
                # NVML can expose host PIDs while /proc exposes container PIDs.
                # An unmatched process set is not evidence of zero GPU memory.
                self.gpu_status = "unattributed_nvml_pids; use_worker_or_allocated_device_memory"
                return {}
            self.gpu_status = "sampled_nvml_process_footprint"
            if self._gpu_baseline is None:
                self._gpu_baseline = used
            return {"gpu_process_used_bytes": used, "gpu_above_start_bytes": max(0, used - self._gpu_baseline)}
        except self._nvml.NVMLError as error:
            self.gpu_status = f"partial:{type(error).__name__}"
            return {}

    def _sample(self) -> None:
        started = time.monotonic()
        sample = {**process_memory(os.getpid()), **cgroup_memory(), **self._gpu_memory()}
        if self._abandoned:
            return
        for key, value in sample.items():
            self.peak[key] = max(self.peak.get(key, 0), value)
        if self._trace is not None:
            try:
                self._trace.write(
                    canonical(
                        {
                            "unix_ns": time.time_ns(),
                            "monotonic_ns": time.monotonic_ns(),
                            "collection_seconds": time.monotonic() - started,
                            "periodic_scan_budget_fraction": self.sampling["periodic_scan_budget_fraction"],
                            **sample,
                        }
                    )
                    + b"\n"
                )
                self._trace.flush()
            except OSError as error:
                self.trace_error = str(error)
                self._close_trace()
                self._warning(f"trace write: {error}")
        elapsed = time.monotonic() - started
        self.sampling["samples"] += 1
        self.sampling["total_scan_seconds"] += elapsed
        self.sampling["maximum_scan_seconds"] = max(self.sampling["maximum_scan_seconds"], elapsed)
        self._next_wait_seconds = max(
            self.interval, elapsed * (1 / self.sampling["periodic_scan_budget_fraction"] - 1)
        )

    def _run(self) -> None:
        try:
            if self.trace_path is not None:
                try:
                    self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                    self._trace = self.trace_path.open("xb")
                except OSError as error:
                    self.trace_error = str(error)
                    self._warning(f"trace open: {error}")
            self._initialize_gpu()
            self._sample()
            self._ready.set()
            while not self._stop.wait(self._next_wait_seconds):
                self._sample()
            if not self._abandoned:
                self._sample()
        except Exception as error:
            self._warning(f"sampling stopped: {type(error).__name__}: {error}")
        finally:
            try:
                self._close_trace()
                if self._nvml is not None:
                    self._nvml.nvmlShutdown()
            except Exception as error:
                self._warning(f"sampler cleanup: {type(error).__name__}: {error}")
            finally:
                _memory_slot.release()
                self._ready.set()

    def __enter__(self) -> "MemorySampler":
        if not _memory_slot.acquire(blocking=False):
            self._warning("another memory sampler is active; this operation has no memory samples")
            return self
        try:
            self.sampling["status"] = "sampling"
            self._thread.start()
            self._started = True
        except Exception as error:
            _memory_slot.release()
            self._warning(f"sampler start: {type(error).__name__}: {error}")
            return self
        if not self._ready.wait(self.wait_timeout):
            self._warning("initial sample is pending")
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if not self._started:
            return
        self._thread.join(timeout=self.wait_timeout)
        if self._thread.is_alive():
            self._abandoned = True
            self._warning("sampler exit is pending; available peaks cover completed samples only")
        elif not self.sampling["errors"]:
            self.sampling["status"] = "complete"
