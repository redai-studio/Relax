# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Current source bodies, fixed TQ samplers and explicit CPU environment
seams."""

from __future__ import annotations

import copy
import json
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
from no16.harness import EvidenceLog, Partition, production_partitioner
from no16.source import ROOT, load, samplers

from relax.utils.data.flops_scheduling import runtime_context
from relax.utils.data.global_flops_pack_planner import ExecutionContract
from relax.utils.data.global_flops_pack_protocol import REQUEST_CONTRACT, checkpoint_identity, restore_plan


# Reuse the exact P2 fixture constructor without importing pytest in the existing Torch environment.
tiny_config = load(ROOT / "tests/cpu/no16_p2/test_estimator.py", names={"tiny_config"}).tiny_config


def setup(tmp_path, kind="sft", *, dp=2, capacity=3, cp=1, tp=1, pp=1, vpp=1, mb_group=1, global_batch_size=4):
    # Gloo workers share a model path. Never expose a truncated JSON during concurrent setup.
    temporary = tmp_path / (".config-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(tiny_config()), encoding="utf-8")
    temporary.replace(tmp_path / "config.json")
    args = SimpleNamespace(
        flops_scheduling="both",
        flops_scheduling_order="pack-first",
        loss_type="sft" if kind == "sft" else "policy_loss",
        enable_mtp_training=False,
        use_dynamic_batch_size=True,
        balance_data=kind == "rl",
        n_samples_per_prompt=1,
        hf_checkpoint=str(tmp_path),
        qkv_format="thd",
        global_batch_size=global_batch_size,
        max_tokens_per_gpu=capacity,
        data_pad_size_multiplier=1,
        allgather_cp=False,
        dynamic_context_parallel=False,
        micro_batch_size=1,
        sft_packing_strategy="balanced",
    )
    context = runtime_context(args)
    contract = ExecutionContract(
        dp, tp, pp, cp, vpp, mb_group, capacity, checkpoint_signature=checkpoint_identity(args, context.estimator, dp)
    )
    refs = samplers()
    pack = load(
        ROOT / "relax/utils/data/global_flops_pack_sampler.py",
        omitted_imports=("transfer_queue", "relax.utils.data.identity_window_sampler"),
        injected={"BaseSampler": refs["base"], "IdentityWindowSampler": refs["rl"]},
    )
    factory = load(
        ROOT / "relax/utils/data/flops_sampler.py",
        omitted_imports=(
            "transfer_queue",
            "relax.utils.data.global_flops_pack_sampler",
            "relax.utils.data.identity_window_sampler",
        ),
        injected={
            "BaseSampler": refs["base"],
            "IdentityWindowSampler": refs["rl"],
            "SeqlenBalancedSampler": refs["sft"],
            "GlobalPackSampler": pack.GlobalPackSampler,
        },
    )
    original = (
        refs["sft"](dp_size=dp, n_samples_per_prompt=1)
        if kind == "sft"
        else refs["rl"](dp_size=dp, placement="balanced")
    )
    wrapper = factory.configure_sampler(args, original, dp)
    refs.update(factory=factory, original=original)
    return args, context, contract, wrapper, refs


def partition(lengths, identities=None):
    identities = list(range(len(lengths))) if identities is None else identities
    return Partition(
        [
            dict(id=str(i), row=100 + 7 * i, rollout_id=identity, length=length)
            for i, (identity, length) in enumerate(zip(identities, lengths, strict=True))
        ]
    )


class Meta:
    def __init__(self, partition, rows):
        self.global_indexes = list(rows)
        self._metadata = copy.deepcopy([partition.metadata[row] for row in rows])
        self.size = len(rows)

    def get_all_custom_meta(self):
        return self._metadata


def pull(wrapper, part, contract, rank, *, mini=0, task="sft_train", quota=2, partition_id="p", ready=None):
    rows, consumed = wrapper.sample(
        sorted(part.global_indexes) if ready is None else ready,
        quota,
        task_name=task,
        partition_id=partition_id,
        partition=part,
        dp_rank=rank,
        batch_index=mini,
        **{REQUEST_CONTRACT: asdict(contract)},
    )
    assert rows == consumed
    return rows, Meta(part, rows)


def payload_from(part, rows):
    lengths = [part.metadata[row]["total_lengths"] for row in rows]
    return dict(
        ids=list(rows),
        total_lengths=lengths,
        tokens=[[row] * length for row, length in zip(rows, lengths, strict=True)],
        loss_masks=[[1] * length for length in lengths],
        sample_indices=[part.metadata[row]["sample_index"] for row in rows],
    )


def restore(part, rows, meta, context, contract, rank, *, mini=0, task="sft_train", partition_id="p"):
    payload = payload_from(part, rows)
    restore_plan(
        meta,
        payload,
        task=task,
        partition=partition_id,
        mini=mini,
        rank=rank,
        contract=contract,
        estimator=context.estimator,
    )
    return payload


def backend(args, context, contract, rank, payload, *, group="cpu-test-group", torch=None, training=True):
    module = production_partitioner()
    deps = dict(
        np=np,
        torch=SimpleNamespace(
            tensor=lambda values, **kw: np.array(values, dtype=int),
            int=int,
            clamp=lambda values, min: np.maximum(values, min),
        ),
        dist=SimpleNamespace(all_reduce=lambda *a, **kw: None, ReduceOp=SimpleNamespace(MAX="MAX")),
        device_utils=SimpleNamespace(make_current_torch_device=lambda: "cpu"),
        ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY="rollout_mini_local_sample_counts",
        ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY="rollout_mini_global_sample_counts",
        logger=EvidenceLog(),
        is_preference_mode=lambda a: False,
        mpu=SimpleNamespace(
            get_data_parallel_world_size=lambda **kw: contract.dp,
            get_data_parallel_rank=lambda **kw: rank,
            get_data_parallel_group=lambda: group,
            get_virtual_pipeline_model_parallel_world_size=lambda: contract.vpp,
            get_context_parallel_world_size=lambda: contract.cp,
            get_tensor_model_parallel_world_size=lambda: contract.tp,
            get_pipeline_model_parallel_world_size=lambda: contract.pp,
        ),
        get_model_config=lambda model: SimpleNamespace(microbatch_group_size_per_vp_stage=contract.mb_group),
    )
    if torch is not None:
        deps["torch"] = torch
    for value in vars(module).values():
        if callable(value) and hasattr(value, "__globals__"):
            value.__globals__.update(deps)
    active = replace(context, control_group=group)
    delegate = load(
        ROOT / "relax/backends/megatron/actor.py",
        class_name="MegatronTrainRayActor",
        names={"_get_data_iterator"},
        injected={"get_data_iterator": module.get_data_iterator},
    )
    actor = SimpleNamespace(args=args, model=[None], _flops_context=active, _pack_first_contract=contract)
    iterators, counts = delegate._get_data_iterator(actor, payload, training=training)
    batches = [it.get_next(["ids", "tokens", "loss_masks"]) for it in iterators for _ in range(sum(counts))]
    return batches, counts, iterators


def concat(batches):
    module = load(
        ROOT / "relax/backends/megatron/data.py",
        names={"concat_rollout_batches", "_same_scalar_value"},
        injected={"torch": SimpleNamespace(Tensor=type("UnusedTensor", (), {}))},
    )
    return module.concat_rollout_batches(batches)


class SourceContractClient:
    """In-process fixed TQ sampler seam; explicitly not an installed TQ
    service."""

    def __init__(self, sampler, part):
        self.sampler, self.partition = sampler, part
        self.requests = []

    def get_meta(self, *, data_fields, batch_size, partition_id, sampling_config, task_name):
        self.requests.append(copy.deepcopy(sampling_config))
        options = {k: v for k, v in sampling_config.items() if k not in ("partition_id", "task_name")}
        rows, _ = self.sampler.sample(
            sorted(self.partition.global_indexes),
            batch_size,
            task_name=task_name,
            partition_id=partition_id,
            partition=self.partition,
            **options,
        )
        return Meta(self.partition, rows)

    def get_data(self, meta):
        return payload_from(self.partition, meta.global_indexes)


def fetch_through_actor(
    args, context, contract, sampler, part, rank, *, mini=0, task="sft_train", client=None, is_drained=None
):
    """Actual Actor -> fetch helper -> sampler -> restore; GPU post-processing
    is a CPU identity seam."""
    topology = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 0,
        get_pipeline_model_parallel_rank=lambda: 0,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_pipeline_model_parallel_world_size=lambda: 1,
        get_context_parallel_world_size=lambda: 1,
        get_data_parallel_rank=lambda **kw: rank,
    )
    stream = load(
        ROOT / "relax/utils/data/stream_dataloader.py",
        names={"get_data_from_transfer_queue", "fetch_data_from_transfer_queue"},
        injected=dict(
            time=time,
            mpu=topology,
            timer=lambda name: nullcontext(),
            TensorDict=type("UnusedTensorDict", (), {}),
            device_utils=SimpleNamespace(make_current_torch_device=lambda: "cpu"),
            _maybe_log_tgd_pickle_diag=lambda *a: None,
            _maybe_log_per_rank_fetch_diag=lambda *a: None,
            post_process_rollout_data=lambda *a: None,
        ),
    )
    actor_module = load(
        ROOT / "relax/backends/megatron/actor.py",
        class_name="MegatronTrainRayActor",
        names={"_get_data_from_transfer_queue"},
        injected=dict(mpu=topology, get_data_from_transfer_queue=stream.get_data_from_transfer_queue),
    )
    args.per_rank_fetch = False
    args.sft_tq_timeout_minutes = None  # Real RL parser retains None; distributed timeout is used.
    args.distributed_timeout_minutes = 2
    actor = SimpleNamespace(
        args=args,
        _flops_context=context,
        _pack_first_contract=contract,
        data_system_client=client or SourceContractClient(sampler, part),
        _sft_image_preprocess_on_rank=False,
        all_consumed=lambda *a: False if is_drained is None else is_drained(),
        _pack_first_local_consumed=lambda *a, **kw: False if is_drained is None else is_drained(),
    )
    return actor_module._get_data_from_transfer_queue(
        actor, task, 0, ["tokens", "loss_masks"], 2, mini, partition_id="p"
    )
