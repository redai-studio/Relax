# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run production partitioning on CPU without importing Megatron or PyTorch."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


_ROOT = Path(__file__).resolve().parents[4]


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
    _execute(_tree("relax/utils/data/packing.py").body, namespace)
    names = {
        "get_first_fit_partitions",
        "_get_first_fit_partitions",
        "_partitions_fit_capacity_or_singleton_oversize",
        "get_minimum_num_micro_batch_size",
        "_get_micro_batch_token_capacity",
        "_get_capacity_safe_balanced_partitions",
    }
    for path in ("relax/utils/data/data.py", "relax/backends/megatron/data.py"):
        _execute([node for node in _tree(path).body if getattr(node, "name", None) in names], namespace)
    return namespace


def _run_actor_partitions(namespace, repack):
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
        nodes, names = condition.body[0].body, {"capacity", "micro_batch_indices"}
    else:
        nodes = next(node for node in actor.body if getattr(node, "name", None) == "_pack_sft_prepack_window").body
        names = {"max_tokens", "k_local", "micro_batch_indices"}
    _execute(
        [
            node
            for node in nodes
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
            )
            or (isinstance(node, ast.If) and ast.unparse(node.test) == "getattr(self.args, 'balance_by_flops', False)")
        ],
        namespace,
    )
    return namespace["micro_batch_indices"]


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
        rollout_data={"total_lengths": lengths},
        cp_size=1,
        window=SimpleNamespace(rollout_data={"total_lengths": lengths}),
        max_k=local_k + 1,
    )
    partitions = _run_actor_partitions(partitioning, repack)
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


@pytest.mark.parametrize("rank", [0, 1])
def test_prepack_executes_central_plan_with_variable_local_rows(partitioning, rank):
    import time
    from contextlib import nullcontext

    import torch

    from relax.utils.data.flops_balancing import STEP_PLANS_FIELD, encode_schedule, local_schedule, restore_schedule

    geometry = dict(version=4, capacity=5, microbatch_group_size=1, context_parallel_size=1, pipeline_stages=1)
    metadata = {i: dict(total_lengths=1, flops_cost=9.0 if i == 5 else 1.0) for i in range(6)}
    assignments, updates = encode_schedule(list(range(6)), metadata, geometry, 2, 6, ("p", "sft_train", 0))
    rows = assignments[rank]
    batch = dict(total_lengths=[1] * len(rows), ids=rows)
    restore_schedule(
        SimpleNamespace(global_indexes=rows, get_all_custom_meta=lambda: [updates[i] for i in rows]), batch
    )
    actor = next(
        n
        for n in _tree("relax/backends/megatron/actor.py").body
        if isinstance(n, ast.ClassDef) and n.name == "MegatronTrainRayActor"
    )
    method = next(n for n in actor.body if getattr(n, "name", None) == "_pack_sft_prepack_window")
    select = next(
        n
        for n in _tree("relax/backends/megatron/actor.py").body
        if getattr(n, "name", None) == "_select_rollout_samples"
    )
    captured = []

    def pack(_args, selected):
        assert STEP_PLANS_FIELD not in selected
        captured.append(selected["ids"])
        return dict(ids=selected["ids"])

    partitioning.update(
        torch=torch,
        time=time,
        STEP_PLANS_FIELD=STEP_PLANS_FIELD,
        local_schedule=local_schedule,
        PrefetchedSFTWindow=SimpleNamespace,
        PrepackedBatch=dict,
        prepack_sft_micro_batch_cpu=pack,
        move_tensors_to_device=lambda value, *_, **__: value,
        device_module=SimpleNamespace(
            set_device=lambda *_: None,
            stream_context=lambda *_: nullcontext(),
            Event=lambda: SimpleNamespace(record=lambda *_: None),
        ),
    )
    partitioning["mpu"].get_data_parallel_rank = lambda **_: rank
    _execute([select, method], partitioning)
    owner = SimpleNamespace(
        args=SimpleNamespace(
            balance_by_flops=True, use_dynamic_batch_size=True, loss_type="sft", max_tokens_per_gpu=5
        ),
        _sft_prepack_local_batch_size=lambda: 3,
        _sft_prefetch_stream=object(),
        _sft_device="cpu",
    )
    result = partitioning["_pack_sft_prepack_window"](owner, 0, batch, 0.0, 0.0)
    assert captured == [rows]
    assert len(result.packed_micro_batches) == 1
    assert len(rows) == (1 if rank == 0 else 5)
