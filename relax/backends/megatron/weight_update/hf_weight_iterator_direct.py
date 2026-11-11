# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import dataclasses
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from tqdm import tqdm

from relax.distributed.weight_sync import traffic
from relax.utils import device as device_utils
from relax.utils.device import device_module
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.types import ParamInfo

from ..sglang import monkey_patch_torch_reductions
from ..weight_conversion import convert_to_hf
from .common import all_gather_params_async, named_params_and_buffers
from .hf_weight_iterator_base import HfWeightIteratorBase


_NON_BLOCKING = device_utils.use_non_blocking_copy()


class HfWeightIteratorDirect(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        agreement = kwargs.pop("publication_agreement", None)
        super().__init__(*args, **kwargs)
        self.megatron_local_param_info_buckets = _get_megatron_local_param_info_buckets(
            self.args, self.model, agreement=agreement
        )

    def get_hf_weight_chunks(self, megatron_local_weights):
        rank = dist.get_rank()

        for megatron_local_param_infos in tqdm(
            self.megatron_local_param_info_buckets, disable=rank != 0, desc="Update weights"
        ):
            megatron_full_params = _get_megatron_full_params(megatron_local_param_infos, megatron_local_weights)
            hf_named_tensors = self._convert_to_hf_named_tensors(megatron_full_params, megatron_local_param_infos)
            yield hf_named_tensors
            del megatron_full_params

    def get_hf_weight_chunks_checked(self, megatron_local_weights, agreement):
        """Agree on local failures between complete collective boundaries."""
        for infos in self.megatron_local_param_info_buckets:
            complete = _get_megatron_full_params(infos, megatron_local_weights, agreement=agreement)
            converted, error = None, None
            try:
                converted = self._convert_to_hf_named_tensors(complete, infos)
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
            agreement(error)
            yield converted
            del complete, converted

    def _convert_to_hf_named_tensors(self, megatron_full_params: Sequence[torch.Tensor], param_infos: list[ParamInfo]):
        hf_named_tensors = []
        for info, param in zip(param_infos, megatron_full_params, strict=False):
            hf_named_tensors.extend(
                convert_to_hf(self.args, self.model_name, info.name, param, self.quantization_config)
            )
        return hf_named_tensors


def _get_megatron_full_params(
    megatron_local_param_infos: Sequence[ParamInfo],
    megatron_local_weights,
    *,
    agreement=None,
) -> Sequence[torch.Tensor]:
    monkey_patch_torch_reductions()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()
    rank = dist.get_rank()
    # init params:
    params = []
    error = None
    try:
        for info in megatron_local_param_infos:
            if dist.get_rank() == info.src_rank:
                params.append(
                    torch.nn.Parameter(
                        megatron_local_weights[info.name].to(
                            device=device_utils.make_current_torch_device(), non_blocking=_NON_BLOCKING
                        ),
                        requires_grad=False,
                    )
                )
            else:
                params.append(
                    torch.empty(info.shape, dtype=info.dtype, device=device_utils.make_current_torch_device())
                )
        device_module.synchronize()
        for info, param in zip(megatron_local_param_infos, params, strict=True):
            for key, value in info.attrs.items():
                setattr(param, key, value)
    except Exception as caught:
        if agreement is None:
            raise
        error = f"{type(caught).__name__}: {caught}"
    if agreement is not None:
        agreement(error)

    # broadcast params across pp ranks
    if pp_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if info.src_rank in dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()):
                handles.append(
                    traffic.broadcast(
                        param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group(), async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    # broadcast params across ep ranks
    if ep_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if ".experts." in info.name:
                src_rank = (
                    info.src_rank
                    if info.src_rank in dist.get_process_group_ranks(mpu.get_expert_model_parallel_group())
                    else rank
                )
                handles.append(
                    traffic.broadcast(param, src=src_rank, group=mpu.get_expert_model_parallel_group(), async_op=True)
                )
        for handle in handles:
            handle.wait()

    # Set tp attrs for all params
    for info, param in zip(megatron_local_param_infos, params, strict=False):
        for key, value in info.attrs.items():
            setattr(param, key, value)

    # Batch async all_gather for all parameters
    gathered_params = all_gather_params_async(
        list(zip(megatron_local_param_infos, params, strict=False)), agreement=agreement
    )

    return gathered_params


def _get_megatron_local_param_info_buckets(
    args: Namespace, model: Sequence[torch.nn.Module], *, agreement=None
) -> list[list[ParamInfo]]:
    """Partition params into buckets ≤ update_weight_buffer_size (with TP
    replication)."""
    param_infos = _get_megatron_local_param_infos(args, model, agreement=agreement)
    param_info_buckets = [[]]  # Start with one empty bucket
    buffer_size = 0  # Track current bucket size in bytes

    for info in param_infos:
        # Expert params use expert-TP size, others use regular-TP size
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()

        # Full param size = shard size × TP replicas (all-gather will reconstruct full param)
        param_size = info.size * tp_size

        # If adding this param exceeds limit AND current bucket has params: start new bucket
        if buffer_size + param_size > args.update_weight_buffer_size and len(param_info_buckets[-1]) > 0:
            param_info_buckets.append([])
            buffer_size = 0

        # Add param to current bucket and update size
        param_info_buckets[-1].append(info)
        buffer_size += param_size

    return param_info_buckets


def _get_megatron_local_param_infos(
    args: Namespace, model: Sequence[torch.nn.Module], *, agreement=None
) -> list[ParamInfo]:
    """Build global param metadata: collect → exchange PP/EP → resolve
    duplicates (MTP virtual PP) by min src_rank → validate.

    When supplied, ``agreement`` synchronizes rank-local failures before each
    subsequent collective. Returns sorted ParamInfo identical across all ranks.
    """

    def local_stage(stage, operation):
        if agreement is None:
            return operation()
        result, error = None, None
        try:
            result = operation()
        except Exception as caught:
            error = f"{stage}: {type(caught).__name__}: {caught}"
        agreement(error)
        return result

    def collect_local():
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        ep_size = mpu.get_expert_model_parallel_world_size()
        rank = dist.get_rank()
        param_infos = {}
        for name, param in named_params_and_buffers(args, model):
            param_infos[name] = ParamInfo(
                name=name,
                dtype=param.dtype,
                shape=param.shape,
                attrs={
                    "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                    "partition_dim": getattr(param, "partition_dim", -1),
                    "partition_stride": getattr(param, "partition_stride", 1),
                    "parallel_mode": getattr(param, "parallel_mode", None),
                },
                size=param.numel() * param.element_size(),
                src_rank=rank,
            )
        return pp_size, ep_size, rank, param_infos

    pp_size, ep_size, rank, param_infos = local_stage("local parameter metadata", collect_local)

    if pp_size > 1:
        param_infos_list = [None] * pp_size
        traffic.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_pipeline_model_parallel_group()
        )

        def merge_pipeline_metadata():
            for src_rank, infos in param_infos_list:
                if src_rank == rank:
                    continue
                for name, info in infos.items():
                    if name in param_infos:
                        assert args.mtp_num_layers is not None
                        old_info = param_infos[name]
                        if old_info.src_rank > src_rank:
                            param_infos[name] = info
                    else:
                        param_infos[name] = info

        local_stage("pipeline metadata merge", merge_pipeline_metadata)

    if ep_size > 1:
        param_infos_list = [None] * ep_size
        traffic.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_expert_model_parallel_group()
        )

        def merge_expert_metadata():
            for src_rank, infos in param_infos_list:
                for name, info in infos.items():
                    if name not in param_infos:
                        # src_rank is the rank within the expert model parallel group.
                        param_infos[name] = dataclasses.replace(info, src_rank=src_rank)

        local_stage("expert metadata merge", merge_expert_metadata)

    def sort_metadata():
        return sorted(param_infos.values(), key=lambda info: info.name)

    param_infos = local_stage("parameter metadata sorting", sort_metadata)

    # Check all ranks have the same parameter info.
    all_param_info_list = [None] * dist.get_world_size()
    traffic.all_gather_object(
        obj=param_infos,
        object_list=all_param_info_list,
        group=get_gloo_group(),
    )

    def validate_metadata():
        for rank_index, infos in enumerate(all_param_info_list):
            if len(infos) != len(param_infos):
                raise ValueError(f"Parameter count mismatch on rank {rank_index}: {len(infos)} != {len(param_infos)}")
            for i, param_info in enumerate(param_infos):
                other = infos[i]
                if other.name != param_info.name:
                    raise ValueError(f"Parameter name mismatch: {other.name} != {param_info.name}")
                if other.shape != param_info.shape:
                    raise ValueError(
                        f"Parameter shape mismatch for {param_info.name}: {other.shape} != {param_info.shape}"
                    )
                if other.dtype != param_info.dtype:
                    raise ValueError(
                        f"Parameter dtype mismatch for {param_info.name}: {other.dtype} != {param_info.dtype}"
                    )
                if other.size != param_info.size:
                    raise ValueError(
                        f"Parameter byte-size mismatch for {param_info.name}: {other.size} != {param_info.size}"
                    )
                if other.attrs != param_info.attrs:
                    raise ValueError(
                        f"Parameter parallel-layout metadata mismatch for {param_info.name}: "
                        f"{other.attrs} != {param_info.attrs}"
                    )

    local_stage("parameter metadata validation", validate_metadata)
    return param_infos
