# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the synchronous checkpoint writer's bounded staging."""

import time
import weakref

import pytest
import torch
from torch.distributed import checkpoint


bounded_staging = pytest.importorskip(
    "megatron.core.dist_checkpointing.strategies.bounded_staging", exc_type=ImportError
)


def save_checkpoint(state, path, budget=0):
    writer = bounded_staging.make_bounded_writer(checkpoint.FileSystemWriter, path, budget)
    checkpoint.save(state, storage_writer=writer)


def test_sync_staging_roundtrip_compacts_views_and_releases_each_item(tmp_path, monkeypatch):
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
    original = bounded_staging._stage_sync_tensor
    references = []

    def stage(tensor):
        assert all(ref() is None for ref in references)
        result = original(tensor)
        if result.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr():
            references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(bounded_staging, "_stage_sync_tensor", stage)
    path = tmp_path / "checkpoint"
    save_checkpoint(state, path)

    assert references and all(ref() is None for ref in references)
    assert sum(file.stat().st_size for file in path.glob("*.distcp")) < 100_000
    restored = {
        key: torch.empty_like(value) if torch.is_tensor(value) else {"iteration": 0} for key, value in state.items()
    }
    checkpoint.load(restored, checkpoint_id=path)
    torch.testing.assert_close(restored, state)


@pytest.mark.parametrize("budget,expected_peak", [(100, 400), (800, 800), (1000, 800)])
def test_sync_window_keeps_only_active_staging_allocations(tmp_path, monkeypatch, budget, expected_peak):
    backing = torch.arange(1000, dtype=torch.float32)
    state = {f"tensor-{i}": backing[i : i + 100] for i in range(6)}
    original_stage = bounded_staging._BoundedSyncStager._stage
    original_write = bounded_staging._write_item
    references = []
    peaks = []

    def tracked_stage(tensor):
        result, event = original_stage(tensor)
        references.append(weakref.ref(result))
        peaks.append(sum(value.nbytes for ref in references if (value := ref()) is not None))
        return result, event

    def slow_write(*args, **kwargs):
        time.sleep(0.02)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(bounded_staging._BoundedSyncStager, "_stage", staticmethod(tracked_stage))
    monkeypatch.setattr(bounded_staging, "_write_item", slow_write)

    save_checkpoint(state, tmp_path, budget)

    assert max(peaks) == expected_peak
    assert all(ref() is None for ref in references)
    restored = {key: torch.empty_like(value) for key, value in state.items()}
    checkpoint.load(restored, checkpoint_id=tmp_path)
    torch.testing.assert_close(restored, state)


@pytest.mark.parametrize("budget", [0, 400])
def test_sync_write_failure_reaches_finalization_and_closes_producer(tmp_path, monkeypatch, budget):
    real_stager = bounded_staging._BoundedSyncStager
    stagers = []

    def record_stager(max_bytes):
        stager = real_stager(max_bytes)
        stagers.append(stager)
        return stager

    def fail(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(bounded_staging, "_BoundedSyncStager", record_stager)
    monkeypatch.setattr(bounded_staging, "_write_item", fail)
    backing = torch.ones(1000)
    with pytest.raises(checkpoint.CheckpointException, match="injected write failure"):
        save_checkpoint({str(i): backing[:100] for i in range(4)}, tmp_path, budget)

    assert not (tmp_path / ".metadata").exists()
    assert len(stagers) == int(budget > 0)
    for stager in stagers:
        assert not stager._thread.is_alive()
        assert not stager._window


@pytest.mark.parametrize("error", [RuntimeError, SystemExit])
def test_sync_producer_failure_reaches_finalization(tmp_path, monkeypatch, error):
    def fail(tensor):
        raise error("injected staging failure")

    monkeypatch.setattr(bounded_staging._BoundedSyncStager, "_stage", staticmethod(fail))
    with pytest.raises(checkpoint.CheckpointException, match="injected staging failure"):
        save_checkpoint({"model": torch.ones(8)}, tmp_path, 128)
    assert not (tmp_path / ".metadata").exists()


@pytest.mark.parametrize("phase", ["start", "close"])
def test_sync_stager_lifecycle_failure_reaches_finalization(tmp_path, monkeypatch, phase):
    original = getattr(bounded_staging._BoundedSyncStager, phase)

    def fail(self, *args):
        if phase == "close":
            original(self, *args)
        raise RuntimeError(f"injected {phase} failure")

    monkeypatch.setattr(bounded_staging._BoundedSyncStager, phase, fail)
    with pytest.raises(checkpoint.CheckpointException, match=f"injected {phase} failure"):
        save_checkpoint({"model": torch.ones(8)}, tmp_path, 128)
    assert not (tmp_path / ".metadata").exists()


def test_sync_write_failure_preserved_if_cleanup_also_fails(tmp_path, monkeypatch):
    original_close = bounded_staging._BoundedSyncStager.close

    def fail_write(*args, **kwargs):
        raise OSError("original write failure")

    def fail_close(self):
        original_close(self)
        raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(bounded_staging, "_write_item", fail_write)
    monkeypatch.setattr(bounded_staging._BoundedSyncStager, "close", fail_close)
    with pytest.raises(checkpoint.CheckpointException, match="original write failure"):
        save_checkpoint({"model": torch.ones(8)}, tmp_path, 128)
    assert not (tmp_path / ".metadata").exists()


def test_sync_optimizer_resume_and_next_update(tmp_path):
    parameter = torch.nn.Parameter(torch.arange(12, dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    for _ in range(2):
        parameter.grad = parameter.detach().square() + 0.1
        optimizer.step()
    save_checkpoint({"model": parameter.detach(), "optimizer": optimizer.state_dict()}, tmp_path, 128)

    resumed = torch.nn.Parameter(torch.zeros_like(parameter))
    resumed_optimizer = torch.optim.Adam([resumed], lr=0.01)
    resumed.grad = torch.ones_like(resumed)
    resumed_optimizer.step()  # Allocate the state skeleton expected by DCP.
    restored = {"model": resumed.detach(), "optimizer": resumed_optimizer.state_dict()}
    checkpoint.load(restored, checkpoint_id=tmp_path)
    resumed_optimizer.load_state_dict(restored["optimizer"])
    for param, opt in [(parameter, optimizer), (resumed, resumed_optimizer)]:
        param.grad = param.detach().square() + 0.1
        opt.step()
    torch.testing.assert_close(resumed, parameter)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(resumed_optimizer.state[resumed][key], optimizer.state[parameter][key])


@pytest.mark.parametrize(
    "setting,budget,expected",
    [
        (None, "unused", None),
        ("0", "unused", None),
        ("1", None, 1024**3),
        ("1", "123456", 123456),
        ("1", "0", 0),
        ("1", "-1", None),
    ],
)
def test_sync_strategy_uses_the_configured_budget(tmp_path, monkeypatch, setting, budget, expected):
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy

    for key, value in (
        ("MEGATRON_SYNC_SAVE_BOUNDED_STAGING", setting),
        ("MEGATRON_SYNC_SAVE_STAGE_BYTES", budget),
    ):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    observed = []
    original = bounded_staging.make_bounded_writer

    def record_writer(writer_type, path, stage_bytes):
        observed.append(stage_bytes)
        return original(writer_type, path, stage_bytes)

    monkeypatch.setattr(bounded_staging, "make_bounded_writer", record_writer)
    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        strategy = TorchDistSaveShardedStrategy("torch_dist", 1)
        source = torch.arange(128, dtype=torch.float32)
        state = {"model": ShardedTensor.from_rank_offsets("model", source)}
        destination = tmp_path / "checkpoint"
        if budget == "-1":
            with pytest.raises(ValueError, match="MEGATRON_SYNC_SAVE_STAGE_BYTES must be nonnegative"):
                strategy.save(state, destination)
            assert not observed
        else:
            strategy.save(state, destination)
            restored = {"model": torch.empty_like(source)}
            checkpoint.load(restored, checkpoint_id=destination)
            torch.testing.assert_close(restored["model"], source)
            assert observed == ([] if expected is None else [expected])
    finally:
        torch.distributed.destroy_process_group()
