# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real chained AdamW state and scheduler round trips through CPU DCP.

Use PyTorch's synchronous writer in place of Core's CUDA-only save finalizer.
Core still generates shard keys and common state and performs the actual load.
CUDA optimizer construction and distributed GPU save are not exercised here.
"""

import copy
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp


pytest.importorskip("megatron.core", exc_type=ImportError)

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
    TorchDistSaveShardedStrategy,
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.optimizer import ChainedOptimizer, FP32Optimizer, MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.training import checkpointing as mcore_checkpointing

from relax.backends.megatron import checkpoint, checkpoint_metadata


class _CPUSaveStrategy(TorchDistSaveShardedStrategy):
    def save(self, sharded_state_dict, checkpoint_dir):
        shards, _, _ = _replace_state_dict_keys_with_sharded_keys(sharded_state_dict, True)
        dcp.save(
            mcore_to_pyt_state_dict(shards, False),
            storage_writer=dcp.FileSystemWriter(checkpoint_dir),
            planner=MCoreSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        )


def _cpu_chain():
    config = OptimizerConfig(lr=0.01, clip_grad=0.0)
    parameters = [torch.nn.Parameter(torch.tensor([0.5, -1.0]) + i) for i in range(2)]
    children = []
    for index, parameter in enumerate(parameters):
        adam = torch.optim.AdamW(
            [
                {
                    "params": [parameter],
                    "wd_mult": 1.0,
                    "lr_mult": 1.0,
                    "is_expert_parallel": index == 1,
                    "is_decoupled_lr": False,
                }
            ],
            lr=0.01,
        )
        # FP32Optimizer.__init__ only adds a CUDA scale tensor after the base
        # initialization; keep its real checkpoint methods with a CPU scale.
        child = object.__new__(FP32Optimizer)
        MegatronOptimizer.__init__(child, adam, config, lambda *_: None)
        child._scale = torch.ones(1)
        child.is_stub_optimizer = False
        children.append(child)
    chain = ChainedOptimizer(children)
    scheduler = OptimizerParamScheduler(
        chain,
        init_lr=0.001,
        max_lr=0.01,
        min_lr=0.001,
        lr_warmup_steps=1,
        lr_decay_steps=10,
        lr_decay_style="linear",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=10,
        wd_incr_style="linear",
    )
    return chain, scheduler, parameters


def _step(chain, scheduler, step):
    for index, child in enumerate(chain.chained_optimizers):
        for parameter in child.optimizer.param_groups[0]["params"]:
            parameter.grad = torch.tensor([0.2 * (step + 1), -0.3 * (index + 1)])
        child.optimizer.step()
    scheduler.step(1)


def _state(chain, scheduler, parameters, metadata, *, loading=False):
    model = {name: ShardedTensor.from_rank_offsets(name, p) for name, p in zip(("dense", "expert"), parameters)}
    return {
        "model": model,
        "optimizer": chain.sharded_state_dict(model, is_loading=loading, metadata=metadata),
        "opt_param_scheduler": scheduler.state_dict(),
        "iteration": 3,
    }


def _assert_same(reference, restored):
    for expected, actual in zip(reference[0].chained_optimizers, restored[0].chained_optimizers, strict=True):
        expected_p = expected.optimizer.param_groups[0]["params"][0]
        actual_p = actual.optimizer.param_groups[0]["params"][0]
        torch.testing.assert_close(actual_p, expected_p, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                actual.optimizer.state[actual_p][key], expected.optimizer.state[expected_p][key], rtol=0, atol=0
            )
        for key in ("lr", "weight_decay"):
            assert actual.optimizer.param_groups[0][key] == expected.optimizer.param_groups[0][key]
    assert restored[1].state_dict() == reference[1].state_dict()


@pytest.mark.parametrize(
    "sharding_type,avoid_prefix,prefixed",
    [("dp_reshardable", False, True), ("dp_reshardable", True, True), ("fully_reshardable", True, False)],
)
def test_checkpoint_chained_optimizer_resumes_adam_and_scheduler(
    tmp_path, monkeypatch, sharding_type, avoid_prefix, prefixed
):
    owned_group = not dist.is_initialized()
    if owned_group:
        dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    monkeypatch.setattr(checkpoint_metadata, "get_gloo_group", lambda: dist.group.WORLD)
    try:
        reference = _cpu_chain()
        for step in range(3):
            _step(*reference[:2], step)
        metadata = {"distrib_optim_sharding_type": sharding_type, "chained_optim_avoid_prefix": avoid_prefix}
        path = tmp_path / "iter_0000003"
        path.mkdir()
        # This is the real ChainedOptimizer.sharded_state_dict and Megatron
        # serializer, with only the synchronous CPU writer substituted.
        dist_checkpointing.save(_state(*reference, metadata), str(path), sharded_strategy=_CPUSaveStrategy())
        keys = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
        for index in range(2):
            prefix = f"chained_{index}.optimizer." if prefixed else "optimizer."
            assert any(key.startswith(prefix) for key in keys)
        # This optimizer-only fixture saves no RNG snapshot and explicitly
        # resets RNG; full-resume RNG validation has separate coverage.
        args = Namespace(
            load=str(tmp_path), ckpt_format="torch_dist", no_load_optim=False, no_load_rng=True, finetune=False
        )
        assert checkpoint_metadata._checkpoint_has_optimizer_state(path, args)
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("3")
        restored = _cpu_chain()
        # Allocate Adam's loading template without advancing the fresh scheduler.
        for child in restored[0].chained_optimizers:
            for parameter in child.optimizer.param_groups[0]["params"]:
                parameter.grad = torch.zeros_like(parameter)
            child.optimizer.step()
        scheduler_before = copy.deepcopy(restored[1].state_dict())

        def load_on_cpu(**kwargs):
            assert not args.no_load_optim
            assert kwargs["optimizer"] is restored[0]
            assert kwargs["opt_param_scheduler"] is restored[1]
            loaded = dist_checkpointing.load(_state(*restored, metadata, loading=True), str(path))
            with torch.no_grad():
                for name, parameter in zip(("dense", "expert"), restored[2]):
                    parameter.copy_(loaded["model"][name])
            restored[0].load_state_dict(loaded["optimizer"])
            restored[1].load_state_dict(loaded["opt_param_scheduler"])
            return loaded["iteration"], 0

        monkeypatch.setattr(checkpoint, "get_args", lambda: args)
        monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", load_on_cpu)
        assert checkpoint.load_checkpoint([Namespace(role="actor")], restored[0], restored[1], {}, False) == (3, 0)
        assert not args.no_load_optim
        assert restored[1].state_dict() != scheduler_before
        _assert_same(reference, restored)
        for step in range(3, 6):
            _step(*reference[:2], step)
            _step(*restored[:2], step)
            _assert_same(reference, restored)
    finally:
        if owned_group:
            dist.destroy_process_group()


@pytest.fixture
def checkpoint_cpu_group(tmp_path, monkeypatch):
    owned_group = not dist.is_initialized()
    if owned_group:
        dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    monkeypatch.setattr(checkpoint_metadata, "get_gloo_group", lambda: dist.group.WORLD)
    try:
        yield
    finally:
        if owned_group:
            dist.destroy_process_group()


@pytest.mark.parametrize(
    "mismatch,selection",
    [
        ("bucket_shape", "tracked"),
        ("model_shape", "tracked"),
        ("bucket_shape", "non_persistent"),
        ("bucket_shape", "explicit_step"),
        ("bucket_assertion", "explicit_step"),
    ],
)
def test_checkpoint_optimizer_bucket_planning_errors(checkpoint_cpu_group, monkeypatch, tmp_path, mismatch, selection):
    source_root = tmp_path / "non_persistent" if selection == "non_persistent" else tmp_path
    selected_step = 5 if selection == "non_persistent" else 3
    tracked_step = 5 if selection == "explicit_step" else 3
    path = source_root / f"iter_{selected_step:07d}"
    (tmp_path / "latest_checkpointed_iteration.txt").write_text(str(tracked_step))
    if selection == "non_persistent":
        source_root.mkdir()
        (source_root / "latest_checkpointed_iteration.txt").write_text(str(selected_step))

    weight_key = "language_model.decoder.layers.mlp.router.weight"
    bucket_prefix = "chained_0.optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_(torch.bfloat16, torch.float32)"
    saved_bucket = f"{bucket_prefix}.bucket_idx_0.exp_avg"
    weight = torch.tensor([0.5, -1.0], dtype=torch.bfloat16)
    for root, step in dict.fromkeys(((tmp_path, tracked_step), (source_root, selected_step))):
        iteration_dir = root / f"iter_{step:07d}"
        iteration_dir.mkdir()
        # Keep the other iteration compatible so loading it would not raise.
        bucket_size = 3 if iteration_dir != path and mismatch == "bucket_shape" else 2
        dist_checkpointing.save(
            {
                "model": {"weight": ShardedTensor.from_rank_offsets(weight_key, weight)},
                "optimizer": {"exp_avg": ShardedTensor.from_rank_offsets(saved_bucket, torch.zeros(bucket_size))},
                "iteration": step,
                "opt_param_scheduler": {"steps": step},
            },
            str(iteration_dir),
            sharded_strategy=_CPUSaveStrategy(),
        )
    args = Namespace(
        load=str(tmp_path),
        ckpt_format="torch_dist",
        ckpt_step=selected_step if selection == "explicit_step" else None,
        no_load_rng=True,
        finetune=False,
        non_persistent_ckpt_type="global" if selection == "non_persistent" else None,
        non_persistent_global_ckpt_dir=None,
        ckpt_assume_constant_structure=False,
        ckpt_fully_parallel_load=False,
        ckpt_load_validate_sharding_integrity=False,
        dist_ckpt_strictness="assume_ok_unexpected",
        verify_integrity=False,
    )
    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    # MCore's tracker reader uses a CUDA reduction. Keep its directory
    # selection and production DCP loader while reading the CPU fixture's tracker.
    monkeypatch.setattr(mcore_checkpointing, "read_metadata", lambda tracker: (int(Path(tracker).read_text()), False))
    original = AssertionError(
        "Number of unpadded elements in each bucket need to be the same in current run ([2, 1]) and checkpoint ([3])"
    )

    def load_on_cpu(**kwargs):
        model_size = 3 if mismatch == "model_shape" else 2
        bucket_size = 3 if mismatch == "bucket_shape" else 2
        load_template = {
            "model": {"weight": ShardedTensor.from_rank_offsets(weight_key, torch.empty(model_size))},
            "optimizer": {"exp_avg": ShardedTensor.from_rank_offsets(saved_bucket, torch.empty(bucket_size))},
        }
        result = mcore_checkpointing._load_base_checkpoint(args.load, args, sharded_state_dict=load_template)
        if mismatch == "bucket_assertion":
            assert result[0]["iteration"] == selected_step
            raise original
        return result

    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", load_on_cpu)
    model = [Namespace(role="actor")]
    optimizer = object()
    if mismatch == "model_shape":
        with pytest.raises(dcp.CheckpointException, match="Global shape mismatch"):
            checkpoint.load_checkpoint(model, optimizer, None, {}, False)
        return

    with pytest.raises(RuntimeError, match="--finetune --save <new-directory>") as error:
        checkpoint.load_checkpoint(model, optimizer, None, {}, False)
    message = str(error.value)
    assert f"--load {tmp_path}" in message
    assert ("--ckpt-step 3" in message) == (selection == "explicit_step")
    assert "original parameter precision and DDP bucket settings" in message
    if mismatch == "bucket_assertion":
        assert error.value.__cause__ is original
    else:
        assert isinstance(error.value.__cause__, dcp.CheckpointException)
        assert saved_bucket in str(error.value.__cause__)

    model_state = dist_checkpointing.load(
        {"model": {"weight": ShardedTensor.from_rank_offsets(weight_key, torch.empty(2))}}, str(path)
    )
    torch.testing.assert_close(model_state["model"]["weight"], weight.float(), rtol=0, atol=0)
