# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Assign MFSDP parameter all-gathers to communicators independent of gradient
reduction."""

import copy
from datetime import timedelta

import torch.distributed as dist


def create_collection(timeout_minutes=10):
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    pg = ProcessGroupCollection.use_mpu_process_groups()
    pg.dp_cp_ag, pg.expt_dp_ag = parallel_state.create_all_gather_groups(
        for_expert_parallelism=True, timeout=timedelta(minutes=timeout_minutes), nccl_comm_cfgs={}
    )
    for base, ag in ((pg.dp_cp, pg.dp_cp_ag), (pg.expt_dp, pg.expt_dp_ag)):
        assert ag is not None and ag is not base
        assert dist.get_process_group_ranks(ag) == dist.get_process_group_ranks(base)
    return pg


def check_buffers(owner, pg):
    assert not owner.dist_index.use_hybrid_fsdp
    counts = {"dense": 0, "expert": 0}
    for bucket in owner.param_and_grad_buffer.parameter_groups:
        expert = bucket.is_expert_param
        ag, rs = (pg.expt_dp_ag, pg.expt_dp) if expert else (pg.dp_cp_ag, pg.dp_cp)
        assert bucket.model_weight_buffer.data_parallel_group is ag
        for buffer in (bucket.main_grad_buffer, bucket.main_weight_buffer):
            if buffer is not None:
                assert buffer.data_parallel_group is rs
        counts["expert" if expert else "dense"] += 1
    return dict(
        counts,
        rank=dist.get_rank(group=dist.group.WORLD),
        dense_ag_ranks=dist.get_process_group_ranks(pg.dp_cp_ag),
        expert_ag_ranks=dist.get_process_group_ranks(pg.expt_dp_ag),
        ag_prefetch_elements=owner.suggested_AG_prefetch_size,
        rs_queue_elements=owner.suggested_RS_queue_capacity,
        all_buffer_assignments_verified=True,
    )


def initialize(args):
    import relax.backends.megatron.model_provider as provider_api
    from relax.utils.logging_utils import get_logger

    assert args.use_megatron_fsdp and (args.tensor_model_parallel_size, args.pipeline_model_parallel_size) == (2, 2)
    assert (args.context_parallel_size, args.expert_model_parallel_size, args.expert_tensor_parallel_size) == (1, 8, 1)
    assert args.num_distributed_optimizer_instances == 1
    assert args.data_parallel_sharding_strategy == "optim_grads_params"
    assert not args.nccl_communicator_config_path
    groups = create_collection(args.distributed_timeout_minutes)
    native_configure = provider_api._configure_bridge_provider

    def configure(*positional, **kwargs):
        provider = native_configure(*positional, **kwargs)
        native_provide = provider.provide_distributed_model

        def provide(*positional, **kwargs):
            assert not positional and kwargs["use_megatron_fsdp"] and kwargs["wrap_with_ddp"]
            pg = copy.copy(kwargs.get("pg_collection") or groups)
            for supplied, expected in ((pg.dp_cp, groups.dp_cp), (pg.expt_dp, groups.expt_dp)):
                assert dist.get_process_group_ranks(supplied) == dist.get_process_group_ranks(expected)
            pg.dp_cp_ag, pg.expt_dp_ag = groups.dp_cp_ag, groups.expt_dp_ag
            kwargs["pg_collection"] = pg
            chunks = native_provide(**kwargs)
            for chunk in chunks:
                report = check_buffers(chunk.module, pg)
                get_logger(__name__).info("MFSDP independent AG groups: %s", report)
            return chunks

        provider.provide_distributed_model = provide
        return provider

    provider_api._configure_bridge_provider = configure
