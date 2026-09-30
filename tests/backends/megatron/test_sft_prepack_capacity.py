# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run production partitioning on CPU without importing Megatron or PyTorch."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


_ROOT = Path(__file__).resolve().parents[3]


def _tree(path):
    return ast.parse((_ROOT / path).read_text())


def _execute(nodes, namespace):
    module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *nodes], type_ignores=[])
    exec(compile(module, "<production partitioning>", "exec"), namespace)


@pytest.fixture
def partitioning():
    namespace = {
        "logger": Mock(),
        "mpu": SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: 2,
            get_context_parallel_world_size=lambda: 1,
        ),
    }
    _execute(_tree("relax/utils/data/seqlen_balancing.py").body, namespace)
    names = {
        "get_minimum_num_micro_batch_size",
        "_get_micro_batch_token_capacity",
        "_get_first_fit_partitions",
        "_partitions_fit_capacity_or_singleton_oversize",
        "_get_capacity_safe_balanced_partitions",
    }
    for path in ("relax/utils/data/data.py", "relax/backends/megatron/data.py"):
        _execute([node for node in _tree(path).body if getattr(node, "name", None) in names], namespace)
    return namespace


@pytest.mark.parametrize("repack", [False, True])
@pytest.mark.parametrize("allgather_cp", [False, True])
def test_prepack_partitions_respect_token_capacity(partitioning, repack, allgather_cp):
    # The original KK-only partition produces [11, 10, 9, 9, 9] with K=5.
    lengths = [3, 9, 7, 6, 2, 5, 2, 5, 9]
    args = SimpleNamespace(max_tokens_per_gpu=10, allgather_cp=allgather_cp, data_pad_size_multiplier=2)
    capacity = 8 if allgather_cp else 10
    local_k = partitioning["get_minimum_num_micro_batch_size"](lengths, capacity)
    partitioning.update(
        self=SimpleNamespace(args=args),
        samples=lengths,
        cp_size=1,
        window=SimpleNamespace(rollout_data={"total_lengths": lengths}),
        max_k=local_k + 1,
    )
    actor = next(
        node
        for node in _tree("relax/backends/megatron/actor.py").body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
    )
    if repack:
        condition = next(
            node
            for node in ast.walk(actor)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "local_k < max_k"
        )
        nodes = condition.body[0].body
        names = {"capacity", "micro_batch_indices"}
    else:
        nodes = next(node for node in actor.body if getattr(node, "name", None) == "_pack_sft_prepack_window").body
        names = {"max_tokens", "k_local", "micro_batch_indices"}
    # Execute the actor's real partition call sites, stopping before tensor packing.
    _execute(
        [
            node
            for node in nodes
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        ],
        partitioning,
    )
    partitions = partitioning["micro_batch_indices"]
    assert len(partitions) == local_k + int(repack)
    assert sorted(index for partition in partitions for index in partition) == list(range(len(lengths)))
    assert all(
        sum(lengths[index] for index in partition) <= capacity or len(partition) == 1 for partition in partitions
    )
    if allgather_cp:
        # An indivisible oversized sample must remain alone.
        assert all(
            len(partition) == 1 for partition in partitions if any(lengths[index] > capacity for index in partition)
        )


def test_allgather_capacity_rejects_budget_smaller_than_padding(partitioning):
    args = SimpleNamespace(allgather_cp=True, data_pad_size_multiplier=8)
    with pytest.raises(ValueError, match="padding must fit"):
        partitioning["_get_micro_batch_token_capacity"](args, 10, 1)
