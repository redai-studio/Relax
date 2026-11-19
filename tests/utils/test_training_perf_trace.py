# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rank-local diagnostics must preserve training semantics and async
execution."""

import json
from types import SimpleNamespace

import pytest
import torch

from relax.utils.training import perf_trace as trace


@pytest.fixture
def cpu_trace(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_TRAIN_PERF_DIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trace, "_topology", lambda args: {"rank": 2, "dp": 1, "tp": 0, "pp": 0, "cp": 0})
    return tmp_path


def test_disabled_trace_does_not_query_cuda_or_write(monkeypatch):
    monkeypatch.delenv("RELAX_TRAIN_PERF_DIR", raising=False)
    monkeypatch.setattr(trace, "_memory", lambda: pytest.fail("disabled path queried CUDA"))

    @trace.trace_training_step
    def train(actor, rollout_id):
        with trace.phase("forward_backward"):
            return rollout_id + 7

    assert train(object(), 3) == 10
    assert not trace.active()


def test_workload_dummy_and_vpp_not_double_counted(cpu_trace):
    data = {
        "total_lengths": [10, 30, 20],
        "response_lengths": [2, 3, 4],
        "sample_indices": [1, 1, 2],
        "multimodal_train_inputs": [
            {"image_grid_thw": torch.empty(2, 3), "pixel_values": torch.empty(12, 4)},
            None,
            None,
        ],
    }
    iterator = SimpleNamespace(micro_batch_indices=[[0, 2], [1], [1]], dummy_micro_batch_offsets={2})

    @trace.trace_training_step
    def train(actor, rollout_id):
        trace.record_workload(data, [iterator, iterator], [3], None)
        with trace.phase("forward_backward"):
            pass
        return "unchanged"

    assert train(SimpleNamespace(args=SimpleNamespace(), _rank=2), 3) == "unchanged"
    row = json.loads(next(cpu_trace.glob("*.jsonl")).read_text())
    workload = row["workloads"][0]
    assert workload["tokens"] == 60
    assert workload["logical_trajectories"] == 2
    assert workload["rows"] == 3
    assert workload["microbatch_tokens"] == [30, 30, 30]
    assert workload["dummy_tokens"] == 30
    assert workload["scheduled_tokens_including_dummy"] == 90
    assert workload["image_refs"] == 2 and workload["vision_patch_refs"] == 12
    assert row["status"] == "ok"
    assert row["topology"]["dp"] == 1
    assert {p["name"] for p in row["phases"]} == {"forward_backward", "train_rpc"}
    assert not trace.active()


def test_exception_is_preserved_and_evidence_written(cpu_trace):
    @trace.trace_training_step
    def train(actor, rollout_id):
        with trace.phase("optimizer_step"):
            raise RuntimeError("original error")

    with pytest.raises(RuntimeError, match="original error"):
        train(SimpleNamespace(args=SimpleNamespace(), _rank=2), 0)
    row = json.loads(next(cpu_trace.glob("*.jsonl")).read_text())
    assert row["status"] == "error"
    assert row["topology"] == {"rank": 2}  # Failure before groups are reloaded is supported.
    assert not trace.active()


def test_allocator_peak_resets_once_and_never_synchronizes(cpu_trace, monkeypatch):
    resets = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="GPU-test"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: resets.append(1))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: pytest.fail("must not synchronize"))
    monkeypatch.setattr(trace, "_memory", lambda: {"step_peak_allocated_bytes": 123})

    @trace.trace_training_step
    def train(actor, rollout_id):
        with trace.phase("one"), trace.phase("two"):
            pass

    train(SimpleNamespace(args=SimpleNamespace(), _rank=2), 1)
    row = json.loads(next(cpu_trace.glob("*.jsonl")).read_text())
    assert resets == [1]
    assert row["gpu_uuid"] == "GPU-test"
    assert row["memory_end"]["step_peak_allocated_bytes"] == 123


def test_profiler_metadata_roundtrip(cpu_trace):
    @trace.trace_training_step
    def train(actor, rollout_id):
        assert trace.profiler_metadata()["rollout_id_at_export"] == 4

    train(SimpleNamespace(args=SimpleNamespace(), _rank=2), 4)
    assert trace.profiler_metadata() == {}


def test_topology_reads_reloaded_groups_without_new_collectives(monkeypatch):
    import sys

    import torch.distributed as dist

    group = SimpleNamespace(group_name="before", group_desc="test", ranks=[0, 1])
    wrapper = SimpleNamespace(group=group)
    mpu = SimpleNamespace()
    for kind in ("tensor_model", "data", "pipeline_model", "context", "model"):
        setattr(mpu, f"get_{kind}_parallel_group", lambda **kwargs: wrapper)
    for kind in ("tensor_model", "data", "pipeline_model", "context"):
        setattr(mpu, f"get_{kind}_parallel_rank", lambda **kwargs: 0)
        setattr(mpu, f"get_{kind}_parallel_world_size", lambda **kwargs: 2)
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: pytest.fail("no new collective"))
    before = trace._topology(SimpleNamespace(sequence_parallel=True))
    wrapper.group = SimpleNamespace(group_name="after", group_desc="test", ranks=[0, 1])
    after = trace._topology(SimpleNamespace(sequence_parallel=True))
    assert before["groups"][0]["name"] == "before"
    assert after["groups"][0]["name"] == "after"
    assert after["sequence_parallel"] is True
    assert {g["label"] for g in after["groups"]} == {"TP", "DP", "DP_CP", "PP", "CP", "MODEL"}


def test_timer_ranges_close_and_wait_does_not_leak(cpu_trace):
    from relax.utils.timer import Timer

    @trace.trace_training_step
    def train(actor, rollout_id):
        timer = Timer()
        with timer.context("perf_test_phase"):
            pass
        timer.start("train_wait")
        assert "train_wait" not in timer._perf_ranges
        timer.end("train_wait")
        assert not timer._perf_ranges

    train(SimpleNamespace(args=SimpleNamespace(), _rank=2), 0)
    row = json.loads(next(cpu_trace.glob("*.jsonl")).read_text())
    assert "perf_test_phase" in {p["name"] for p in row["phases"]}


def test_native_cpu_profiler_exports_rank_and_phase_metadata(cpu_trace):
    from relax.utils.profile_utils import TrainProfiler, _create_torch_profiler

    args = SimpleNamespace(
        tb_experiment_name="test",
        profile_step_start=0,
        profile_step_end=0,
        profile_with_stack=False,
        profile_with_memory=False,
        profile_with_flops=False,
    )
    from unittest.mock import patch

    @trace.trace_training_step
    def train(actor, rollout_id):
        with patch("relax.utils.profile_utils._get_rank_tag", return_value="rank2_dp1_tp0_pp0"):
            prof = _create_torch_profiler(args, "train_overall")
        with prof:
            with trace.phase("forward_backward"):
                torch.ones(3) + 2
            holder = SimpleNamespace(_torch_profiler_overall=prof, _memory_profiler_overall=None)
            TrainProfiler.step(holder, rollout_id)

    train(SimpleNamespace(args=args, _rank=2), 0)
    import gzip

    with gzip.open(next((cpu_trace / "traces").glob("*.gz")), "rt") as handle:
        data = json.load(handle)
    assert data["relax"]["rollout_id_at_export"] == 0
    assert data["relax"]["topology"]["rank"] == 2
    assert any(event["name"] == "relax.phase/forward_backward" for event in data["traceEvents"])
