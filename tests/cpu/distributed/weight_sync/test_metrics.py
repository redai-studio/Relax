import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from relax.distributed.weight_sync import metrics as metrics_module
from relax.distributed.weight_sync.metrics import MemorySampler, SyncMetrics
from relax.distributed.weight_sync.store import load_json


class MetricsTests(unittest.TestCase):
    def test_unmatched_nvml_namespace_is_not_reported_as_zero(self):
        sampler = MemorySampler()
        process = SimpleNamespace(pid=2**30, usedGpuMemory=8 << 30)
        sampler._nvml = SimpleNamespace(
            nvmlDeviceGetCount=lambda: 1,
            nvmlDeviceGetHandleByIndex=lambda index: index,
            nvmlDeviceGetComputeRunningProcesses=lambda device: [process],
            NVMLError=RuntimeError,
        )
        self.assertEqual(sampler._gpu_memory(), {})
        self.assertIn("unattributed", sampler.gpu_status)
        process.pid = os.getpid()
        self.assertEqual(sampler._gpu_memory()["gpu_process_used_bytes"], 8 << 30)
        self.assertEqual(sampler.gpu_status, "sampled_nvml_process_footprint")

    def test_application_counters_keep_their_measurement_scope(self):
        metrics = SyncMetrics("update")
        metrics.transferred("X", 42, link="sharedfs")
        report = metrics.report()
        self.assertIsNone(report["bytes"]["G"])
        self.assertFalse(report["counter_coverage_complete"])
        for stage in ("G", "W", "T", "C"):
            metrics.transferred(stage, 0, link="local")
        report = metrics.report()
        self.assertFalse(report["counter_coverage_complete"])
        self.assertTrue(report["application_stage_coverage_complete"])
        self.assertEqual(report["counter_scope"], "application")

    def test_memory_samples_preserve_aligned_values_and_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            with MemorySampler(trace_path=path) as sampler:
                pass
            records = [load_json(line) for line in path.read_bytes().splitlines()]
            self.assertGreaterEqual(len(records), 2)
            self.assertEqual(sorted(row["monotonic_ns"] for row in records), [row["monotonic_ns"] for row in records])
            for field, maximum in sampler.peak.items():
                self.assertEqual(maximum, max(row.get(field, 0) for row in records))
            self.assertIsNone(sampler.trace_error)

    def test_expensive_scans_back_off_without_changing_measured_peaks(self):
        sampler = MemorySampler()
        with (
            patch.object(metrics_module, "process_memory", return_value={"pss_bytes": 4096}),
            patch.object(metrics_module, "cgroup_memory", return_value={}),
            patch.object(metrics_module.time, "monotonic", side_effect=[10.0, 10.4]),
        ):
            sampler._sample()
        self.assertEqual(sampler.peak, {"pss_bytes": 4096})
        self.assertAlmostEqual(sampler._next_wait_seconds, 7.6)
        self.assertAlmostEqual(sampler.sampling["total_scan_seconds"], 0.4)
        self.assertAlmostEqual(0.4 / (0.4 + sampler._next_wait_seconds), 0.05)

    def test_cheap_scans_retain_the_requested_minimum_pause(self):
        sampler = MemorySampler(interval=0.2)
        with (
            patch.object(metrics_module, "process_memory", return_value={"rss_bytes": 1024}),
            patch.object(metrics_module, "cgroup_memory", return_value={}),
            patch.object(metrics_module.time, "monotonic", side_effect=[10.0, 10.001]),
        ):
            sampler._sample()
        self.assertEqual(sampler._next_wait_seconds, 0.2)
        self.assertEqual(sampler.sampling["samples"], 1)

    def test_no_duplicate_entry_scan_and_stop_interrupts_long_cooldown(self):
        waiting, stopped = threading.Event(), threading.Event()
        pauses = []

        class Stop:
            def wait(self, seconds):
                pauses.append(seconds)
                waiting.set()
                return stopped.wait(seconds)

            def is_set(self):
                return stopped.is_set()

            def set(self):
                stopped.set()

        sampler = MemorySampler(interval=60)
        sampler._stop = Stop()
        with (
            patch.object(
                metrics_module, "process_memory", side_effect=[{"pss_bytes": 100}, {"pss_bytes": 200}]
            ) as scan,
            patch.object(metrics_module, "cgroup_memory", return_value={}),
        ):
            started = time.monotonic()
            with sampler:
                self.assertTrue(waiting.wait(timeout=2))
                self.assertEqual(scan.call_count, 1)
            self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(pauses, [60])
        self.assertEqual(sampler.sampling["samples"], 2)
        self.assertEqual(sampler.peak["pss_bytes"], 200)
        self.assertTrue(sampler.sampling["boundary_samples_outside_budget"])
        self.assertTrue(sampler.sampling["sampled_peaks_are_lower_bounds"])
