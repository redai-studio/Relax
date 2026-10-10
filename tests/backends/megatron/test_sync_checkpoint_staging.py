# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Round-trip the deployment patch against the installed Megatron/PyTorch
writer."""

import importlib.util
import queue
import shutil
import subprocess
import sys
import time
import weakref
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")
ROOT = Path(__file__).resolve().parents[3]
PATCH = ROOT / "docker/patch/megatron/sync-save-bounded-staging.patch"


@pytest.fixture
def writer_module(tmp_path, monkeypatch):
    import megatron.core.dist_checkpointing.strategies.filesystem_async as original

    relative = Path("megatron/core/dist_checkpointing/strategies")
    destination = tmp_path / relative
    destination.mkdir(parents=True)
    for name in ("filesystem_async.py", "torch.py"):
        shutil.copy2(Path(original.__file__).parent / name, destination / name)
    check = subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=tmp_path, capture_output=True)
    if check.returncode == 0:
        subprocess.run(["git", "apply", str(PATCH)], cwd=tmp_path, check=True)
    else:
        subprocess.run(["git", "apply", "--reverse", "--check", str(PATCH)], cwd=tmp_path, check=True)
    name = original.__name__
    spec = importlib.util.spec_from_file_location(name, destination / "filesystem_async.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def prepare(module, path, state, synchronous=True, threads=1, hint=None):
    from torch.distributed.checkpoint.default_planner import DefaultSavePlanner

    writer = module.FileSystemWriterAsync(path, synchronous=synchronous, thread_count=threads, separation_hint=hint)
    planner = DefaultSavePlanner()
    planner.set_up_planner(state, is_coordinator=True)
    writer.set_up_storage_writer(True)
    local = writer.prepare_local_plan(planner.create_local_plan())
    plans, metadata = planner.create_global_plan([local])
    plans = writer.prepare_global_plan(plans)
    plan = planner.finish_plan(plans[0])
    writer.prepare_write_data(plan, planner)
    return writer, metadata


def write(module, writer, metadata, budget=0):
    module.FileSystemWriterAsync.write_data_synchronously(
        [writer.transforms] if hasattr(writer, "transforms") else [],
        False,
        0,
        writer.write_buckets,
        writer.results_queue,
        budget,
    )
    results = writer.retrieve_write_results()
    assert isinstance(results, list)
    writer.finish(metadata, [results])


def test_sync_staging_roundtrip_compacts_views_and_releases_each_item(writer_module, tmp_path, monkeypatch):
    from torch.distributed.checkpoint import load

    backing = torch.arange(1_000_000, dtype=torch.float32)
    state = {
        "model": backing[10:110],
        "optimizer.master": backing[20:120],
        "optimizer.exp_avg": backing[30:130],
        "optimizer.exp_avg_sq": backing[40:140],
        "optimizer.step": torch.tensor(2),
        "noncontiguous": backing[:200].view(10, 20).t(),
        "extra": {"iteration": 1},
    }
    writer, metadata = prepare(writer_module, tmp_path / "checkpoint", state, threads=2, hint="optimizer")
    tensors = [tensor for bucket in writer.write_buckets for _, tensor in bucket[2][1]]
    assert any(t.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr() for t in tensors)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    assert writer.get_save_function_and_args()[1] is None
    original = writer_module.FileSystemWriterAsync._stage_sync_tensor
    references = []

    def stage(tensor):
        assert all(ref() is None for ref in references), "previous staging allocation is still live"
        result = original(tensor)
        if result.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr():
            references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(writer_module.FileSystemWriterAsync, "_stage_sync_tensor", staticmethod(stage))
    write(writer_module, writer, metadata)
    assert all(ref() is None for ref in references)
    assert sum(p.stat().st_size for p in (tmp_path / "checkpoint").glob("*.distcp")) < 100_000
    restored = {k: torch.empty_like(v) if isinstance(v, torch.Tensor) else {"iteration": 0} for k, v in state.items()}
    load(restored, checkpoint_id=tmp_path / "checkpoint")
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(restored[key], value)
        else:
            assert restored[key] == value


def test_sync_write_failure_reaches_finalization(writer_module, tmp_path, monkeypatch):
    writer, _ = prepare(writer_module, tmp_path / "failed", {"model": torch.ones(8)})

    def fail(*args, **kwargs):
        raise OSError("injected disk failure")

    monkeypatch.setattr(writer_module, "_write_item", fail)
    writer_module.FileSystemWriterAsync.write_data_synchronously(
        [writer.transforms] if hasattr(writer, "transforms") else [],
        False,
        0,
        writer.write_buckets,
        writer.results_queue,
    )
    result = writer.retrieve_write_results()
    assert not isinstance(result, list)
    assert "injected disk failure" in str(result)
    assert not (tmp_path / "failed" / ".metadata").exists()


def _record_stagers(module, monkeypatch):
    stagers = []
    original = module._BoundedSyncStager

    def factory(max_bytes):
        stager = original(max_bytes)
        stagers.append(stager)
        return stager

    monkeypatch.setattr(module, "_BoundedSyncStager", factory)
    return stagers


def test_sync_windowed_staging_respects_byte_budget(writer_module, tmp_path, monkeypatch):
    from torch.distributed.checkpoint import load

    state = {f"tensor-{i}": torch.full((100,), float(i)) for i in range(6)}  # 400 bytes each
    writer, metadata = prepare(writer_module, tmp_path / "windowed", state)
    stagers = _record_stagers(writer_module, monkeypatch)
    real_write_item = writer_module._write_item

    def slow_write_item(*args, serialization_format=None, **kwargs):
        time.sleep(0.05)
        return real_write_item(*args, serialization_format=serialization_format, **kwargs)

    monkeypatch.setattr(writer_module, "_write_item", slow_write_item)
    write(writer_module, writer, metadata, budget=1000)
    assert len(stagers) == 1
    # Two 400-byte items fit under the budget; the third waits for a take.
    assert stagers[0].peak_staged_bytes == 800
    restored = {key: torch.empty_like(value) for key, value in state.items()}
    load(restored, checkpoint_id=tmp_path / "windowed")
    for key, value in state.items():
        torch.testing.assert_close(restored[key], value)


def test_sync_windowed_staging_allows_single_oversized_item(writer_module, tmp_path, monkeypatch):
    from torch.distributed.checkpoint import load

    state = {"large": torch.full((100,), 1.0), "small": torch.full((100,), 2.0)}  # 400 bytes each
    writer, metadata = prepare(writer_module, tmp_path / "oversize", state)
    stagers = _record_stagers(writer_module, monkeypatch)
    write(writer_module, writer, metadata, budget=100)  # smaller than one item
    assert len(stagers) == 1
    # An item larger than the budget is staged only once the window is empty.
    assert stagers[0].peak_staged_bytes == 400
    restored = {key: torch.empty_like(value) for key, value in state.items()}
    load(restored, checkpoint_id=tmp_path / "oversize")
    for key, value in state.items():
        torch.testing.assert_close(restored[key], value)


def test_sync_windowed_staging_stage_failure_reaches_finalization(writer_module, tmp_path, monkeypatch):
    writer, _ = prepare(writer_module, tmp_path / "stage-fail", {"model": torch.ones(8), "model2": torch.ones(8)})

    def fail_stage(tensor):
        raise RuntimeError("injected staging failure")

    monkeypatch.setattr(writer_module._BoundedSyncStager, "_stage", staticmethod(fail_stage))
    writer_module.FileSystemWriterAsync.write_data_synchronously(
        [writer.transforms] if hasattr(writer, "transforms") else [],
        False,
        0,
        writer.write_buckets,
        writer.results_queue,
        1024,
    )
    result = writer.retrieve_write_results()
    assert not isinstance(result, list)
    assert "injected staging failure" in str(result)


def test_sync_stage_budget_from_environment(writer_module, tmp_path, monkeypatch):
    monkeypatch.setenv("MEGATRON_SYNC_SAVE_STAGE_BYTES", "123456")
    writer, _ = prepare(writer_module, tmp_path / "env", {"model": torch.ones(4)})
    assert writer.sync_stage_max_bytes == 123456
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    assert writer.get_save_function_and_args()[2][-1] == 123456


def test_async_preparation_still_compacts_views_and_preloads(writer_module, tmp_path, monkeypatch):
    monkeypatch.setattr(writer_module, "get_write_results_queue", queue.Queue)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    backing = torch.arange(1000)
    writer, _ = prepare(writer_module, tmp_path / "async", {"model": backing[:10]}, synchronous=False)
    tensor = writer.write_buckets[0][2][1][0][1]
    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    assert tensor.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()
    assert writer.get_save_function_and_args()[1] is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for synchronous D2H checkpoint validation")
def test_sync_cuda_roundtrip(writer_module, tmp_path):
    from torch.distributed.checkpoint import load

    source = torch.randn(32, 64, device="cuda").t()
    writer, metadata = prepare(writer_module, tmp_path / "gpu", {"model": source})
    assert writer.write_buckets[0][2][1][0][1].is_cuda
    write(writer_module, writer, metadata)
    restored = {"model": torch.empty_like(source, device="cpu")}
    load(restored, checkpoint_id=tmp_path / "gpu")
    torch.testing.assert_close(restored["model"], source.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for windowed D2H staging")
def test_sync_cuda_windowed_roundtrip(writer_module, tmp_path):
    from torch.distributed.checkpoint import load

    source = torch.randn(32, 64, device="cuda").t()  # noncontiguous, 8192 bytes
    big = torch.randn(1024, 1024, device="cuda")  # 4 MiB, exceeds the budget alone
    writer, metadata = prepare(writer_module, tmp_path / "gpu-window", {"a": source, "b": big})
    assert writer.write_buckets[0][2][1][0][1].is_cuda
    write(writer_module, writer, metadata, budget=8192)
    restored = {"a": torch.empty_like(source, device="cpu"), "b": torch.empty_like(big, device="cpu")}
    load(restored, checkpoint_id=tmp_path / "gpu-window")
    torch.testing.assert_close(restored["a"], source.cpu())
    torch.testing.assert_close(restored["b"], big.cpu())


def test_sync_optimizer_resume_and_next_update(writer_module, tmp_path):
    from torch.distributed.checkpoint import load

    parameter = torch.nn.Parameter(torch.arange(12, dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    for _ in range(2):
        parameter.grad = parameter.detach().square() + 0.1
        optimizer.step()
    state = {"model": parameter.detach(), "optimizer": optimizer.state_dict()}
    writer, metadata = prepare(writer_module, tmp_path / "resume", state)
    write(writer_module, writer, metadata)
    resumed = torch.nn.Parameter(torch.zeros_like(parameter))
    resumed_optimizer = torch.optim.Adam([resumed], lr=0.01)
    resumed.grad = torch.ones_like(resumed)
    resumed_optimizer.step()  # Allocate the state skeleton expected by DCP.
    restored = {"model": resumed.detach(), "optimizer": resumed_optimizer.state_dict()}
    load(restored, checkpoint_id=tmp_path / "resume")
    resumed_optimizer.load_state_dict(restored["optimizer"])
    for param, opt in [(parameter, optimizer), (resumed, resumed_optimizer)]:
        param.grad = param.detach().square() + 0.1
        opt.step()
    torch.testing.assert_close(resumed, parameter)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(resumed_optimizer.state[resumed][key], optimizer.state[parameter][key])


@pytest.mark.parametrize("setting", [None, "1", "0"])
def test_strategy_selects_writer_and_reuses_plan(writer_module, tmp_path, monkeypatch, setting):
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    if setting is None:
        monkeypatch.delenv("MEGATRON_SYNC_SAVE_BOUNDED_STAGING", raising=False)
    else:
        monkeypatch.setenv("MEGATRON_SYNC_SAVE_BOUNDED_STAGING", setting)
    if setting != "1":
        # Rollback must ignore the new budget entirely.
        monkeypatch.setenv("MEGATRON_SYNC_SAVE_STAGE_BYTES", "invalid-but-unused")
    # MCore hardcodes CUDA for its failure-flag collective; use CPU with Gloo here.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    module_name = "megatron.core.dist_checkpointing.strategies.torch"
    spec = importlib.util.spec_from_file_location(module_name, Path(writer_module.__file__).with_name("torch.py"))
    strategy_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, strategy_module)
    spec.loader.exec_module(strategy_module)
    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        original = writer_module.FileSystemWriterAsync.prepare_write_data
        modes = []
        preloads = []
        original_preload = writer_module.FileSystemWriterAsync.preload_tensors

        def preload_spy(*args, **kwargs):
            preloads.append(True)
            return original_preload(*args, **kwargs)

        monkeypatch.setattr(writer_module.FileSystemWriterAsync, "preload_tensors", staticmethod(preload_spy))

        def prepare_spy(self, *args, **kwargs):
            modes.append(self.synchronous)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(writer_module.FileSystemWriterAsync, "prepare_write_data", prepare_spy)
        strategy = strategy_module.TorchDistSaveShardedStrategy("torch_dist", 1, cached_metadata=True)
        source = torch.arange(128, dtype=torch.float32)
        for iteration in range(2):
            destination = tmp_path / f"iteration-{iteration}"
            destination.mkdir()
            strategy.save({"model": ShardedTensor.from_rank_offsets("model", source)}, destination)
            restored = {"model": torch.empty_like(source, device="cpu")}
            torch.distributed.checkpoint.load(restored, checkpoint_id=destination)
            torch.testing.assert_close(restored["model"], source.cpu())
        assert modes == [setting == "1"] * 2
        assert len(preloads) == (0 if setting == "1" else 2)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("budget", [100, 800])
def test_sync_window_budget_includes_writer_and_releases_producer_reference(
    writer_module, tmp_path, monkeypatch, budget
):
    # Each view needs its own 400-byte compaction allocation. Track real live
    # tensors, rather than trusting the stager's own accounting counter.
    backing = torch.arange(1000, dtype=torch.float32)
    state = {f"tensor-{i}": backing[i : i + 100] for i in range(6)}
    writer, metadata = prepare(writer_module, tmp_path / "live-budget", state)
    original_stage = writer_module._BoundedSyncStager._stage
    references = []
    peaks = []

    def tracked_stage(tensor):
        result, event = original_stage(tensor)
        references.append(weakref.ref(result))
        live_bytes = sum(ref().numel() * ref().element_size() for ref in references if ref() is not None)
        peaks.append(live_bytes)
        assert live_bytes <= max(budget, 400)
        return result, event

    original_write = writer_module._write_item

    def slow_write(*args, serialization_format=None, **kwargs):
        time.sleep(0.02)
        return original_write(*args, serialization_format=serialization_format, **kwargs)

    monkeypatch.setattr(writer_module._BoundedSyncStager, "_stage", staticmethod(tracked_stage))
    monkeypatch.setattr(writer_module, "_write_item", slow_write)
    write(writer_module, writer, metadata, budget=budget)
    assert max(peaks) == (400 if budget < 400 else 800)
    assert all(ref() is None for ref in references)


def test_sync_window_write_failure_closes_blocked_producer(writer_module, tmp_path, monkeypatch):
    backing = torch.ones(1000)
    writer, _ = prepare(writer_module, tmp_path / "window-failure", {str(i): backing[:100] for i in range(4)})
    stagers = _record_stagers(writer_module, monkeypatch)

    def fail(*args, **kwargs):
        raise OSError("window write failure")

    monkeypatch.setattr(writer_module, "_write_item", fail)
    writer_module.FileSystemWriterAsync.write_data_synchronously(
        [], False, 0, writer.write_buckets, writer.results_queue, 400
    )
    assert "window write failure" in str(writer.retrieve_write_results())
    assert not stagers[0]._thread.is_alive()
    assert not stagers[0]._window


def test_sync_negative_budget_rejected(writer_module, tmp_path, monkeypatch):
    monkeypatch.setenv("MEGATRON_SYNC_SAVE_STAGE_BYTES", "-1")
    with pytest.raises(ValueError, match="must be nonnegative"):
        prepare(writer_module, tmp_path / "negative", {"model": torch.ones(4)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for caller stream ordering")
def test_sync_window_waits_for_caller_cuda_stream(writer_module):
    source = torch.zeros(128, device="cuda")
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stager = writer_module._BoundedSyncStager(1024)
    try:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
            source.fill_(7)
            stager.start([source])
        restored = stager.take()
        torch.testing.assert_close(restored, torch.full((128,), 7.0))
        del restored
        stager.release()
    finally:
        stager.close()


@pytest.mark.parametrize("phase", ["start", "close"])
def test_sync_stager_lifecycle_failure_reaches_finalization(writer_module, tmp_path, monkeypatch, phase):
    writer, _ = prepare(writer_module, tmp_path / phase, {"model": torch.ones(8)})
    original = getattr(writer_module._BoundedSyncStager, phase)

    def fail(self, *args):
        if phase == "close":
            original(self, *args)
        raise RuntimeError(f"injected {phase} failure")

    monkeypatch.setattr(writer_module._BoundedSyncStager, phase, fail)
    writer_module.FileSystemWriterAsync.write_data_synchronously(
        [writer.transforms] if hasattr(writer, "transforms") else [],
        False,
        0,
        writer.write_buckets,
        writer.results_queue,
        128,
    )
    assert f"injected {phase} failure" in str(writer.retrieve_write_results())
    assert not (tmp_path / phase / ".metadata").exists()


def test_sync_producer_base_exception_reaches_finalization(writer_module, tmp_path, monkeypatch):
    writer, _ = prepare(writer_module, tmp_path / "producer-exit", {"model": torch.ones(8)})

    def fail(tensor):
        raise SystemExit("injected producer exit")

    monkeypatch.setattr(writer_module._BoundedSyncStager, "_stage", staticmethod(fail))
    writer_module.FileSystemWriterAsync.write_data_synchronously(
        [writer.transforms] if hasattr(writer, "transforms") else [],
        False,
        0,
        writer.write_buckets,
        writer.results_queue,
        128,
    )
    assert "injected producer exit" in str(writer.retrieve_write_results())


def test_sync_write_failure_preserved_if_cleanup_also_fails(writer_module, tmp_path, monkeypatch):
    writer, _ = prepare(writer_module, tmp_path / "two-errors", {"model": torch.ones(8)})
    original_close = writer_module._BoundedSyncStager.close

    def fail_write(*args, **kwargs):
        raise OSError("original write failure")

    def fail_close(self):
        original_close(self)
        raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(writer_module, "_write_item", fail_write)
    monkeypatch.setattr(writer_module._BoundedSyncStager, "close", fail_close)
    writer_module.FileSystemWriterAsync.write_data_synchronously(
        [], False, 0, writer.write_buckets, writer.results_queue, 128
    )
    assert "original write failure" in str(writer.retrieve_write_results())
