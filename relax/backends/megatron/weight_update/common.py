# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import inspect
import re
from argparse import Namespace
from collections.abc import Iterator, Mapping, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from relax.backends.megatron.misc_utils import strip_param_name_prefix
from relax.backends.megatron.weight_update.train_offload import torch_memory_saver_preloaded
from relax.utils.device import is_npu_available
from relax.utils.misc import get_hf_config
from relax.utils.types import ParamInfo


def gdn_component_sizes(config, tp_size: int, name: str) -> tuple[int, ...] | None:
    """Component-major GDN layout; unrelated parameters use ordinary TP
    metadata."""
    if not any(key in name for key in ("self_attention.conv1d.weight", "self_attention.in_proj.weight")):
        return None
    config = getattr(config, "text_config", None) or config
    if not all(
        hasattr(config, key)
        for key in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim")
    ):
        return None
    if config.linear_num_key_heads % tp_size or config.linear_num_value_heads % tp_size:
        raise ValueError("GDN head counts must be divisible by TP size")
    qk = config.linear_key_head_dim * (config.linear_num_key_heads // tp_size)
    value = config.linear_value_head_dim * (config.linear_num_value_heads // tp_size)
    heads = config.linear_num_value_heads // tp_size
    return (qk, qk, value) if "conv1d" in name else (qk, qk, value, value, heads, heads)


def _restore_tp_weight(name, partitions, partition_dim, split_sizes=None):
    if split_sizes is not None:
        assert partition_dim == 0, "GDN components must be sharded along rows"
        components = [part.split(split_sizes, dim=0) for part in partitions]
        return torch.cat([parts[index] for index in range(len(split_sizes)) for parts in components], dim=0)
    if "linear_fc1.weight" in name and "vision_model" not in name:
        components = [part.chunk(2, dim=0) for part in partitions]
        partitions = [part[0] for part in components] + [part[1] for part in components]
        if is_npu_available:
            partition_dim = 0
    if "linear_fc2.weight" in name and "vision_model" not in name and partition_dim == 0:
        partition_dim = 1
    return torch.cat(partitions, dim=partition_dim)


def _gather_tp_weights(named_params, *, split_sizes, async_op):
    tasks, handles = [], []
    for name, param in named_params:
        name = name.replace(".to_wrap.", ".")
        if "expert_bias" in name:
            tasks.append((name, param, None, None))
            continue
        if not hasattr(param, "tensor_model_parallel"):
            multimodal = any(part in name for part in (".audio_model.", ".vision_model.", ".video_model."))
            assert multimodal or not isinstance(param, torch.nn.Parameter), f"{name} lacks TP metadata"
            tasks.append((name, param, None, None))
            continue
        if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
            tasks.append((name, param.data, None, None))
            continue
        if ".experts." in name:
            size, group = mpu.get_expert_tensor_parallel_world_size(), mpu.get_expert_tensor_parallel_group()
        else:
            size, group = mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()
        if "linear_fc1" not in name:
            assert getattr(param, "partition_stride", 1) == 1, f"Unsupported partition stride for {name}"
        partitions = [torch.empty_like(param.data) for _ in range(size)]
        handle = dist.all_gather(partitions, param.data, group=group, async_op=async_op)
        if handle is not None:
            handles.append(handle)
        tasks.append((name, None, partitions, param.partition_dim))
    # All async collectives launch before any wait. Results remain in input order.
    for handle in handles:
        handle.wait()
    return [
        direct if partitions is None else _restore_tp_weight(name, partitions, dim, split_sizes.get(name))
        for name, direct, partitions, dim in tasks
    ]


def all_gather_param(args, name: str, param: torch.Tensor) -> torch.Tensor:
    """Gather one parameter using the same layout rules as batched export."""
    name = name.replace(".to_wrap.", ".")
    split_sizes = {}
    if (
        getattr(param, "tensor_model_parallel", False)
        and getattr(param, "parallel_mode", None) != "duplicated"
        and any(key in name for key in ("self_attention.conv1d.weight", "self_attention.in_proj.weight"))
    ):
        split_sizes[name] = gdn_component_sizes(
            get_hf_config(args.hf_checkpoint), mpu.get_tensor_model_parallel_world_size(), name
        )
    return _gather_tp_weights([(name, param)], split_sizes=split_sizes, async_op=False)[0]


def all_gather_params_async(
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
    *,
    gdn_split_sizes: Mapping[str, tuple[int, ...]] | None = None,
) -> list[torch.Tensor]:
    """Launch every parameter gather before waiting, then restore shared TP
    layouts."""
    return _gather_tp_weights(
        [(info.name, param) for info, param in param_infos_and_params],
        split_sizes=gdn_split_sizes or {},
        async_op=True,
    )


def named_params_and_buffers(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
    include_persistent_buffers: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    if convert_to_global_name:
        ans = _named_params_and_buffers_global(args, model, include_persistent_buffers=include_persistent_buffers)
    else:
        ans = _named_params_and_buffers_vanilla(model, include_persistent_buffers=include_persistent_buffers)

    if translate_gpu_to_cpu:
        ans = ((name, _maybe_get_cpu_backup(tensor)) for name, tensor in ans)

    return ans


def _maybe_get_cpu_backup(x: torch.Tensor) -> torch.Tensor:
    # Selective offload (weight_update/train_offload.py): when the offloader
    # frees a param's GPU storage via storage().resize_(0), it stashes the CPU copy
    # on the tensor as ``_relax_cpu_offload_data``. If the GPU storage is empty, read
    # from that CPU copy instead of touching the now-invalid CUDA storage.
    if getattr(x, "_relax_cpu_offload_data", None) is not None and x.storage().size() == 0:
        return x._relax_cpu_offload_data

    # torch_memory_saver path: only usable when its LD_PRELOAD hook is active;
    # otherwise get_cpu_backup() would assert on an uninitialized saver.
    # NOTE: with --selective-offload, actor_group.py does NOT LD_PRELOAD the TMS hook,
    # so torch_memory_saver_preloaded() is False here and this branch is skipped —
    # i.e. LD_PRELOAD is the single source of truth for "TMS is the active mechanism".
    if torch_memory_saver_preloaded():
        from torch_memory_saver import torch_memory_saver

        if (cpu_tensor := torch_memory_saver.get_cpu_backup(x)) is not None:
            return cpu_tensor

    return x


def _named_buffers_for_weight_update(
    model_module: torch.nn.Module, include_persistent_buffers: bool
) -> Iterator[tuple[str, torch.Tensor]]:
    if include_persistent_buffers:
        # Match Megatron Bridge: checkpoint state includes persistent buffers,
        # but excludes runtime caches and counters registered as non-persistent.
        for module_prefix, module in model_module.named_modules():
            non_persistent = getattr(module, "_non_persistent_buffers_set", ())
            for local_name, buffer in module.named_buffers(recurse=False):
                if local_name in non_persistent:
                    continue
                name = f"{module_prefix}.{local_name}" if module_prefix else local_name
                yield name, buffer
        return

    for name, buffer in model_module.named_buffers():
        if "expert_bias" in name:
            yield name, buffer


def _named_params_and_buffers_vanilla(
    model: Sequence[torch.nn.Module], include_persistent_buffers: bool = False
) -> Iterator[tuple[str, torch.Tensor]]:
    for vp_stage, model_module in enumerate(model):

        def _compute_fqn(name, vp_stage=vp_stage):
            return f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}"

        for name, param in model_module.named_parameters():
            yield _compute_fqn(name), param

        for name, buffer in _named_buffers_for_weight_update(model_module, include_persistent_buffers):
            yield _compute_fqn(name), buffer


def _named_params_and_buffers_global(
    args: Namespace, model: Sequence[torch.nn.Module], include_persistent_buffers: bool = False
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (global_name, param/buffer) with consistent names across PP/EP.

    Adjusts indices for virtual PP + EP offsets. Handles decoder.layers,
    mtp.layers (Multi-Token Prediction), expert_bias.
    """
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    if args.num_experts:
        expert_offset = ep_rank * args.num_experts // ep_size

    sig = inspect.signature(get_transformer_layer_offset)
    need_vp_stage = "vp_stage" in sig.parameters

    for vp_stage, model_module in enumerate(model):
        if need_vp_stage:
            layer_offset = get_transformer_layer_offset(model_module.config, vp_stage)
        else:
            layer_offset = get_transformer_layer_offset(model_module.config)
        for name, param in model_module.named_parameters():
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.(?:(.*)\.)?decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                # MTP (Multi-Token Prediction) layers for speculative decoding
                mtp_layers_pattern = r"module\.module\.(?:(.*)\.)?mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    yield name, param
                    continue

                # MTP layer indices start from 0
                middle_path, layer_idx, rest = match.groups()
                # GPT MTP uses transformer_layer; hybrid MTP (e.g. Qwen3.8)
                # nests its transformer layer inside mtp_model_layer.layers.
                expert_pattern = (
                    r"((?:transformer_layer|mtp_model_layer\.layers\.\d+)\.mlp\.experts\..+)\.weight(\d+)$"
                )
                match = re.match(expert_pattern, rest)
                if not match:
                    yield name, param
                    continue

                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                if middle_path is None:
                    middle_path = ""
                else:
                    middle_path += "."
                yield (
                    f"module.module.{middle_path}mtp.layers.{layer_idx}.{rest}.weight{expert_idx}",
                    param,
                )
                continue

            middle_path, layer_idx, rest = match.groups()
            if middle_path is None:
                middle_path = ""
            else:
                middle_path += "."
            layer_idx = int(layer_idx) + layer_offset

            # this is hardcoded for te grouped matmul
            expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield (
                    f"module.module.{middle_path}decoder.layers.{layer_idx}.mlp.experts.{rest}.weight{expert_idx}",
                    param,
                )
            else:
                yield f"module.module.{middle_path}decoder.layers.{layer_idx}.{rest}", param

        # Treat selected model-state buffers needed by rollout as normal parameters.
        for name, buffer in _named_buffers_for_weight_update(model_module, include_persistent_buffers):
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.(?:(.*)\.)?decoder\.layers\.(\d+)\.(.+)"

            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                middle_path, layer_idx, rest = match.groups()
                if middle_path is None:
                    middle_path = ""
                else:
                    middle_path += "."
                layer_idx = int(layer_idx) + layer_offset
                yield f"module.module.{middle_path}decoder.layers.{layer_idx}.{rest}", buffer
