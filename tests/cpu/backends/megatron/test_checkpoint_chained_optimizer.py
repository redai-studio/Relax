# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real chained AdamW state and scheduler round trips through CPU DCP.

Use PyTorch's synchronous writer in place of Core's CUDA-only save finalizer.
Core still generates shard keys and common state and performs the actual load.
CUDA optimizer construction and distributed GPU save are not exercised here.
"""

import copy
from argparse import Namespace

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
        args = Namespace(load=str(tmp_path), ckpt_format="torch_dist", no_load_optim=False, finetune=False)
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
