# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import gzip
import json
from argparse import Namespace
from unittest.mock import Mock

import pytest

from relax.utils import profile_utils


@pytest.mark.parametrize("rank,selected", [(0, True), (48, True), (1, False)])
def test_train_profiler_honors_rank_selection(monkeypatch, rank, selected):
    args = Namespace(
        profile_ranks=[0, 48],
        use_pytorch_profiler=True,
        record_memory_history=True,
        profile_target=["train_overall", "train_actor", "train_log_probs"],
        tb_experiment_name="rank-selection",
        memory_snapshot_dir=None,
        memory_snapshot_num_steps=1,
    )
    monkeypatch.setattr(profile_utils.torch.distributed, "get_rank", lambda group: rank)
    create_torch = Mock()
    create_memory = Mock()
    monkeypatch.setattr(profile_utils, "_create_torch_profiler", create_torch)
    monkeypatch.setattr(profile_utils._BaseMemoryProfiler, "create", create_memory)
    profiler = profile_utils.TrainProfiler(args)
    profiler.on_init_end()
    profiler.step(0)
    assert list(profiler.iterate_train_actor(iter([10, 20]))) == [10, 20]
    assert list(profiler.iterate_train_log_probs(iter([30]))) == [30]
    assert create_torch.call_count == (3 if selected else 0)
    assert create_memory.call_count == int(selected)
    if selected:
        assert create_torch.return_value.start.call_count == 3
        assert create_torch.return_value.step.call_count == 4
        create_memory.return_value.start.assert_called_once()
        create_memory.return_value.stop.assert_called_once()


def test_train_profiler_without_rank_filter_preserves_legacy_behavior(monkeypatch):
    get_rank = Mock(side_effect=AssertionError("No distributed state is needed without a rank filter"))
    monkeypatch.setattr(profile_utils.torch.distributed, "get_rank", get_rank)
    assert profile_utils._should_profile_train_rank(Namespace())
    get_rank.assert_not_called()


def _python_stack_probe_leaf(tensor):
    return tensor.sin().square()


@pytest.mark.parametrize("compression", [None, "1"])
def test_train_profiler_exports_python_operator_call_stacks(monkeypatch, tmp_path, compression):
    """with_stack alone can omit operator Call stack fields from Chrome
    traces."""
    torch = profile_utils.torch
    if compression is None:
        monkeypatch.delenv("RELAX_PROFILE_GZIP_LEVEL", raising=False)
    else:
        monkeypatch.setenv("RELAX_PROFILE_GZIP_LEVEL", compression)
    monkeypatch.setattr(profile_utils.tempfile, "tempdir", str(tmp_path))
    original_profile = torch.profiler.profile

    def cpu_profile(**kwargs):
        return original_profile(activities=[torch.profiler.ProfilerActivity.CPU], **kwargs)

    monkeypatch.setattr(torch.profiler, "profile", cpu_profile)
    monkeypatch.setattr(profile_utils, "_get_rank_tag", lambda: "rank0_dp0_tp0_pp0")
    monkeypatch.setattr(profile_utils, "_get_train_trace_dir", lambda args: str(tmp_path))
    args = Namespace(
        profile_step_start=0,
        profile_step_end=0,
        profile_with_stack=True,
        profile_with_memory=False,
        profile_with_flops=False,
    )
    tensor = torch.randn(16)
    with profile_utils._create_torch_profiler(args, "stack-test") as profiler:
        _python_stack_probe_leaf(tensor)
        profiler.step()
    files = list(tmp_path.glob("*.pt.trace.json.gz"))
    assert len(files) == 1
    with gzip.open(files[0], "rt") as stream:
        events = json.load(stream)["traceEvents"]
    assert any(
        event.get("cat") == "python_function" and "_python_stack_probe_leaf" in event["name"] for event in events
    )
    assert any(
        event.get("cat") == "cpu_op" and "_python_stack_probe_leaf" in event.get("args", {}).get("Call stack", "")
        for event in events
    )
    assert not list(tmp_path.glob("relax-*.json"))
    assert not list(tmp_path.glob("*.partial"))


def test_failed_trace_export_preserves_local_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_PROFILE_GZIP_LEVEL", "1")
    monkeypatch.setattr(profile_utils.tempfile, "tempdir", str(tmp_path))
    output = tmp_path / "published"

    def interrupted_export(path):
        profile_utils.Path(path).write_text('{"traceEvents": [')
        raise RuntimeError("simulated export interruption")

    profiler = Mock()
    profiler.export_chrome_trace.side_effect = interrupted_export
    with pytest.raises(RuntimeError, match="simulated export interruption"):
        profile_utils._create_train_trace_handler(str(output), "rank0")(profiler)
    retained = list(tmp_path.glob("relax-rank0-*.json"))
    assert len(retained) == 1
    assert retained[0].read_text() == '{"traceEvents": ['
    assert not list(output.iterdir())
