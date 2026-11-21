# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise current production iterator bodies; only environment I/O is
injected."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from tests.cpu.no16.harness import EvidenceLog, production_partitioner


def iterator(
    lengths,
    context=None,
    *,
    dp=1,
    cp=1,
    vpp=1,
    group_size=1,
    counts=None,
    agree=None,
    physical=False,
    real_torch=None,
    process_group=None,
    global_batch_size=None,
    capacity=10,
    balance_data=False,
):
    production = production_partitioner()
    reductions = []
    group_token = "test_dp" if process_group is None else process_group

    def reduce(values, *, op, group):
        assert group == group_token
        reductions.append(values.tolist())
        if real_torch is not None:
            real_torch.distributed.all_reduce(values, op=real_torch.distributed.ReduceOp.MAX, group=process_group)
        if agree:
            agree(values)

    args = SimpleNamespace(
        global_batch_size=global_batch_size or (len(lengths) if counts is None else counts[0]) * dp,
        use_dynamic_batch_size=True,
        micro_batch_size=1,
        max_tokens_per_gpu=capacity,
        balance_data=balance_data,
        sft_packing_strategy="balanced",
        data_pad_size_multiplier=1,
        allgather_cp=False,
        qkv_format="thd",
        dynamic_context_parallel=False,
        uses_unsplit_forward=physical,
    )
    logger = EvidenceLog()
    deps = dict(
        np=np,
        logger=logger,
        torch=real_torch
        or SimpleNamespace(
            tensor=lambda values, **kw: np.array(values, dtype=int),
            int=int,
            clamp=lambda values, min: np.maximum(values, min),
        ),
        mpu=SimpleNamespace(
            get_data_parallel_world_size=lambda **kw: dp,
            get_data_parallel_group=lambda: group_token,
            get_virtual_pipeline_model_parallel_world_size=lambda: vpp,
            get_context_parallel_world_size=lambda: cp,
            get_tensor_model_parallel_world_size=lambda: 1,
        ),
        dist=SimpleNamespace(all_reduce=reduce, ReduceOp=SimpleNamespace(MAX="MAX")),
        device_utils=SimpleNamespace(make_current_torch_device=lambda: "cpu"),
        get_model_config=lambda model: SimpleNamespace(microbatch_group_size_per_vp_stage=group_size),
        is_preference_mode=lambda args: False,
        ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY="rollout_mini_local_sample_counts",
        ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY="rollout_mini_global_sample_counts",
    )
    for value in vars(production).values():
        if callable(value) and hasattr(value, "__globals__"):
            value.__globals__.update(deps)
    payload = dict(
        total_lengths=list(lengths),
        ids=list(range(len(lengths))),
        rollout_mini_local_sample_counts=counts or [len(lengths)],
    )
    active = None if context is None else replace(context, control_group=group_token)
    iterators, k = production.get_data_iterator(args, [None], payload, flops_context=active)
    batches = [it.get_next(["ids"]) for it in iterators for _ in range(sum(k))]
    return batches, k, reductions, iterators
