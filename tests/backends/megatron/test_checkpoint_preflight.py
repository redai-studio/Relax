# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Checkpoint selection, real DCP metadata and two-rank failure agreement."""

import multiprocessing
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.checkpoint import save

from relax.backends.megatron import checkpoint, checkpoint_metadata


@pytest.mark.parametrize(
    "tracker,step,expected",
    [
        ("7", None, "iter_0000007"),
        ("7", 0, "iter_0000007"),
        ("7", 9, "iter_0000009"),
        ("release", None, "release"),
        ("release", 9, "release"),
        (None, 9, "iter_0000009"),
        (None, 0, None),
    ],
)
def test_checkpoint_iteration_selection(tmp_path, tracker, step, expected):
    if tracker is not None:
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(tracker)
    result = checkpoint._resolve_checkpoint_iteration_dir(tmp_path, step)
    assert result == (tmp_path / expected if expected else None)


def test_checkpoint_invalid_tracker_fails(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("unfinished")
    with pytest.raises(ValueError, match="Invalid checkpoint iteration"):
        checkpoint._resolve_checkpoint_iteration_dir(tmp_path)


@pytest.mark.parametrize("optimizer", [False, True])
def test_checkpoint_real_dcp_optimizer_metadata(tmp_path, optimizer):
    state = {"model": {"weight": torch.zeros(2)}}
    if optimizer:
        state["optimizer"] = {"exp_avg": torch.ones(2)}
    save(state, checkpoint_id=tmp_path)
    assert (
        checkpoint_metadata._checkpoint_has_optimizer_state(tmp_path, Namespace(ckpt_format="torch_dist")) is optimizer
    )


@pytest.mark.parametrize(
    "key,expected",
    [
        ("optimizer.state.exp_avg.weight", True),
        ("chained_0.optimizer.state.exp_avg.weight", True),
        ("chained_1.optimizer.state.exp_avg_sq.weight", True),
        ("chained_12.optimizer.distributed.dp_group_idx_0", True),
        ("optimizer/state/shard_0", True),
        ("model.optimizer.weight", False),
        ("optimizer_statistics", False),
        ("chained_0.model.weight", False),
        ("chained_invalid.optimizer.state", False),
    ],
)
def test_checkpoint_optimizer_shard_namespaces(tmp_path, key, expected):
    save({key: torch.ones(2)}, checkpoint_id=tmp_path)
    assert (
        checkpoint_metadata._checkpoint_has_optimizer_state(tmp_path, Namespace(ckpt_format="torch_dist")) is expected
    )


def _stub_loaders(monkeypatch, args):
    calls = []
    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(
        checkpoint, "_load_checkpoint_hf", lambda **kw: calls.append(("hf", kw["load_path"])) or (0, 0)
    )
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **kw: calls.append(("dcp",)) or (7, 0))
    monkeypatch.setattr(checkpoint, "_load_checkpoint_metadata", lambda *a, **kw: {})
    return calls


def test_checkpoint_missing_iteration_falls_back_to_hf(monkeypatch, tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
    calls = _stub_loaders(monkeypatch, Namespace(load=str(tmp_path)))
    assert checkpoint.load_checkpoint(None, None, None, {}, False) == (0, 0)
    assert calls == [("hf", None)]


@pytest.mark.parametrize(
    "no_load_optim,finetune,expected", [(False, False, True), (True, False, True), (False, True, False)]
)
def test_checkpoint_model_only_sets_optimizer_policy(monkeypatch, tmp_path, no_load_optim, finetune, expected):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
    save({"model": torch.zeros(2)}, checkpoint_id=tmp_path / "iter_0000007")
    args = Namespace(load=str(tmp_path), ckpt_format="torch_dist", no_load_optim=no_load_optim, finetune=finetune)
    calls = _stub_loaders(monkeypatch, args)
    checkpoint.load_checkpoint(None, object(), None, {}, False)
    assert calls == [("dcp",)]
    assert args.no_load_optim is expected


@pytest.mark.parametrize("missing", ["iteration", "optimizer"])
def test_reward_model_resume_rejects_incomplete_checkpoint(monkeypatch, tmp_path, missing):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
    args = Namespace(load=str(tmp_path), loss_type="rm", ckpt_format="torch_dist")
    if missing == "optimizer":
        save({"model": torch.zeros(2)}, checkpoint_id=tmp_path / "iter_0000007")
    calls = _stub_loaders(monkeypatch, args)
    with pytest.raises(RuntimeError, match="RM resume requires"):
        checkpoint.load_checkpoint([Namespace(role="actor")], object(), None, {}, False)
    assert calls == []


def test_checkpoint_explicit_step_without_tracker_is_megatron(tmp_path):
    (tmp_path / "iter_0000009").mkdir()
    state = checkpoint._checkpoint_load_state(str(tmp_path), Namespace(ckpt_step=9), False)
    assert state["megatron"] and state["valid_iteration"]
    assert state["iteration_dir"] == str(tmp_path / "iter_0000009")


def test_checkpoint_lora_metadata_uses_explicit_step(monkeypatch, tmp_path):
    from megatron.core import dist_checkpointing

    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
    (tmp_path / "iter_0000009").mkdir()
    seen = []
    marker = {"format_version": 1}
    args = Namespace(**{checkpoint._LORA_CHECKPOINT_METADATA_ATTR: marker})
    monkeypatch.setattr(dist_checkpointing, "load_common_state_dict", lambda path: seen.append(path) or {"args": args})
    monkeypatch.setattr(dist_checkpointing, "check_is_distributed_checkpoint", lambda path: True)
    directory = checkpoint._resolve_checkpoint_iteration_dir(tmp_path, ckpt_step=9)
    common = checkpoint._load_checkpoint_metadata(args, [Namespace(role="actor")], directory)
    assert checkpoint._metadata_value(common["args"], checkpoint._LORA_CHECKPOINT_METADATA_ATTR) == marker
    assert seen == [tmp_path / "iter_0000009"]


def _probe_worker(rank, rendezvous, output):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    checkpoint_metadata.get_gloo_group = lambda: dist.group.WORLD
    outcomes = []
    try:
        for mode in ("valid", "missing", "permission", "metadata", "tracker", "optimizer"):
            calls = []
            checkpoint.get_args = lambda: Namespace(load="/shared/checkpoint")
            checkpoint._load_checkpoint_hf = lambda **kw: calls.append("hf")
            checkpoint._load_checkpoint_megatron = lambda **kw: calls.append("dcp")
            checkpoint._load_checkpoint_metadata = lambda *a, **kw: {}

            def probe(*args):
                if rank == 1 and mode in ("permission", "metadata"):
                    raise OSError(mode)
                return {
                    "exists": not (rank == 1 and mode == "missing"),
                    "megatron": True,
                    "valid_iteration": True,
                    "iteration_dir": "iter_0000009" if rank == 1 and mode == "tracker" else "iter_0000007",
                    "has_optimizer": not (rank == 1 and mode == "optimizer"),
                    "hf": False,
                }

            checkpoint._checkpoint_load_state = probe
            try:
                checkpoint.load_checkpoint(None, object(), None, {}, False)
                outcomes.append((mode, "ok", calls))
            except RuntimeError:
                outcomes.append((mode, "error", calls))
        Path(output, f"rank-{rank}.txt").write_text(repr(outcomes))
    finally:
        dist.destroy_process_group()


def test_checkpoint_two_rank_probe_errors_never_enter_loader(tmp_path):
    import ast

    ctx = multiprocessing.get_context("spawn")
    workers = [
        ctx.Process(target=_probe_worker, args=(rank, str(tmp_path / "rendezvous"), str(tmp_path)))
        for rank in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)
        assert all(not worker.is_alive() and worker.exitcode == 0 for worker in workers)
        results = [ast.literal_eval((tmp_path / f"rank-{rank}.txt").read_text()) for rank in range(2)]
        assert results[0] == results[1]
        assert results[0][0] == ("valid", "ok", ["dcp"])
        assert all(status == "error" and calls == [] for _, status, calls in results[0][1:])
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
