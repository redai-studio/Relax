# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Checkpoint selection, real DCP metadata and two-rank failure agreement."""

import ast
import multiprocessing
import os
import re
from argparse import Namespace
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.distributed as dist
from torch.distributed.checkpoint import save

from relax.backends.megatron import checkpoint_metadata, compat
from relax.engine.sft.runtime import is_rm_mode
from relax.utils.logging_utils import get_logger
from relax.utils.model_source import is_model_source_alias


def _load_checkpoint_module():
    """Load real checkpoint routing without optional backend imports.

    Tests replace the heavyweight loaders at their existing call boundary.
    Spawned Gloo workers run this loader too, without inheriting pytest mocks.
    """
    source = Path(__file__).resolve().parents[4] / "relax/backends/megatron/checkpoint.py"
    names = {
        "_resolve_checkpoint_iteration_dir",
        "_checkpoint_load_state",
        "load_checkpoint",
        "is_megatron_checkpoint",
        "_is_hf_checkpoint",
        "_is_dir_nonempty",
        "_metadata_value",
        "_load_checkpoint_metadata",
        "_validate_reward_model_tensor_metadata",
        "_alias_renamed_transfer_queue_enum",
        "_format_opt_param_scheduler_error",
        "_select_hf_load_source",
    }
    definitions = [
        node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ModuleType("checkpoint_routing_under_test")
    module.__dict__.update(
        Namespace=Namespace,
        Path=Path,
        os=os,
        re=re,
        nullcontext=nullcontext,
        logger=get_logger(__name__),
        is_model_source_alias=is_model_source_alias,
        is_rm_mode=is_rm_mode,
        _collective_checkpoint_probe=checkpoint_metadata._collective_checkpoint_probe,
        _checkpoint_has_optimizer_state=checkpoint_metadata._checkpoint_has_optimizer_state,
        validate_native_checkpoint=checkpoint_metadata.validate_native_checkpoint,
        patch_hybrid_optimizer_native_fp32_checkpoint_load=compat.patch_hybrid_optimizer_native_fp32_checkpoint_load,
        preserve_hdo_dp_reshardable_steps_on_load=compat.preserve_hdo_dp_reshardable_steps_on_load,
        REWARD_MODEL_HEAD_TYPE="reward_model_terminal_v1",
        _LORA_CHECKPOINT_METADATA_ATTR="relax_lora_checkpoint_metadata",
        get_args=None,
        _load_checkpoint_hf=None,
        _load_checkpoint_megatron=None,
    )
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), module.__dict__)
    return module


checkpoint = _load_checkpoint_module()


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
    monkeypatch.setattr(
        checkpoint,
        "_load_checkpoint_metadata",
        lambda *a, **kw: {"iteration": 7, "opt_param_scheduler": {}, "rng_state": []},
    )
    return calls


def test_checkpoint_missing_iteration_never_restarts_from_hf(monkeypatch, tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
    calls = _stub_loaders(monkeypatch, Namespace(load=str(tmp_path)))
    with pytest.raises(RuntimeError, match="iteration directory.*does not exist"):
        checkpoint.load_checkpoint(None, None, None, {}, False)
    assert calls == []


@pytest.mark.parametrize("no_load_optim,finetune", [(True, False), (False, True)])
def test_checkpoint_explicit_model_only_load(monkeypatch, tmp_path, no_load_optim, finetune):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
    save({"model": torch.zeros(2)}, checkpoint_id=tmp_path / "iter_0000007")
    args = Namespace(load=str(tmp_path), ckpt_format="torch_dist", no_load_optim=no_load_optim, finetune=finetune)
    calls = _stub_loaders(monkeypatch, args)
    checkpoint.load_checkpoint(None, object(), None, {}, False)
    assert calls == [("dcp",)]
    assert args.no_load_optim is no_load_optim


def test_checkpoint_missing_optimizer_requires_explicit_reset(monkeypatch, tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
    save({"model": torch.zeros(2)}, checkpoint_id=tmp_path / "iter_0000007")
    args = Namespace(load=str(tmp_path), ckpt_format="torch_dist")
    calls = _stub_loaders(monkeypatch, args)
    with pytest.raises(RuntimeError, match="no optimizer state"):
        checkpoint.load_checkpoint(None, object(), None, {}, False)
    assert calls == []


def test_checkpoint_release_remains_an_explicit_weights_only_source(monkeypatch, tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
    save({"model": torch.zeros(2)}, checkpoint_id=tmp_path / "release")
    args = Namespace(load=str(tmp_path), ckpt_format="torch_dist")
    calls = _stub_loaders(monkeypatch, args)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_metadata", lambda *a: {"iteration": 0})
    checkpoint.load_checkpoint(None, object(), None, {}, False)
    assert calls == [("dcp",)]


@pytest.mark.parametrize("damage", ["missing", "truncated", "outside"])
def test_checkpoint_shard_damage_is_detected_before_loading(monkeypatch, tmp_path, damage):
    from torch.distributed.checkpoint import FileSystemReader

    directory = tmp_path / "iter_0000007"
    save({"model": torch.zeros(2), "optimizer": torch.ones(2)}, checkpoint_id=directory)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
    metadata = FileSystemReader(directory).read_metadata()
    storage = next(iter(metadata.storage_data.values()))
    shard = directory / storage.relative_path
    if damage == "missing":
        shard.unlink()
    elif damage == "truncated":
        shard.write_bytes(b"partial")
    else:
        # A symlink must not let a snapshot depend on storage outside its root.
        outside = tmp_path / "external-shard"
        shard.rename(outside)
        shard.symlink_to(outside)
    args = Namespace(load=str(tmp_path), ckpt_format="torch_dist")
    calls = _stub_loaders(monkeypatch, args)
    with pytest.raises(RuntimeError, match="missing|truncated|outside"):
        checkpoint.load_checkpoint(None, object(), None, {}, False)
    assert calls == []


@pytest.mark.parametrize("missing", [None, "optimizer", "scheduler", "rng", "iteration", "legacy_scheduler"])
def test_checkpoint_resume_state_contract(tmp_path, missing):
    directory = tmp_path / "iter_0000007"
    state = {"model": torch.zeros(2)}
    if missing != "optimizer":
        state["optimizer"] = torch.ones(2)
    save(state, checkpoint_id=directory)
    common = {"iteration": 7, "opt_param_scheduler": {}, "rng_state": []}
    if missing == "scheduler":
        del common["opt_param_scheduler"]
    if missing == "rng":
        del common["rng_state"]
    if missing == "iteration":
        common["iteration"] = 6
    if missing == "legacy_scheduler":
        common["lr_scheduler"] = common.pop("opt_param_scheduler")
    if missing in {None, "legacy_scheduler"}:
        assert checkpoint_metadata.validate_native_checkpoint(directory, common_state=common) == common
    else:
        with pytest.raises(RuntimeError, match="optimizer|scheduler|RNG|iteration"):
            checkpoint_metadata.validate_native_checkpoint(directory, common_state=common)
    if missing in {"optimizer", "scheduler", "rng"}:
        assert (
            checkpoint_metadata.validate_native_checkpoint(
                directory, require_optimizer=False, require_rng=False, common_state=common
            )
            == common
        )


def test_checkpoint_recipe_preflight_reads_native_common_state(tmp_path):
    from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config

    directory = tmp_path / "iter_0000007"
    save({"model": torch.zeros(2), "optimizer": torch.ones(2)}, checkpoint_id=directory)
    common = {"iteration": 7, "opt_param_scheduler": {}, "rng_state": []}
    torch.save(common, directory / "common.pt")
    save_config(CheckpointingConfig(sharded_backend="torch_dist"), str(directory))
    assert checkpoint_metadata.validate_native_checkpoint(directory) == common


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
    dist_checkpointing = pytest.importorskip("megatron.core.dist_checkpointing")

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
