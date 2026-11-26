# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Read the native expert loader's actual ownership without reading weights."""

from functools import partial

import torch


def describe_native_fp8_experts(model: torch.nn.Module) -> list[dict] | None:
    try:
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.runtime_context import get_exec, get_parallel
    except ImportError:
        return None

    moe, parallel = get_exec().moe, get_parallel()
    required = dict(
        enable_eplb=False,
        ep_num_redundant_experts=0,
        init_expert_location="trivial",
        elastic_ep_backend=None,
        ep_join_mode=None,
        enable_waterfill=False,
    )
    if any(not hasattr(moe, key) or getattr(moe, key) != value for key, value in required.items()):
        return None
    if parallel.elastic_ep_initial_size is not None or parallel.ep_join_rank_offset != 0:
        return None

    # The native loader consults this mapping before its local expert map.
    # A global metadata object also exists for the trivial identity layout.
    metadata = get_global_expert_location_metadata()
    result = []
    for name, module in model.named_modules():
        if not isinstance(module, FusedMoE):
            continue
        fields = (
            "layer_id",
            "moe_ep_rank",
            "moe_ep_size",
            "moe_tp_size",
            "_expert_storage_rank",
            "_num_global_routed",
            "_num_local_routed",
            "num_local_experts",
            "num_fused_shared_experts",
        )
        values = {field: getattr(module, field, None) for field in fields}
        if any(type(value) is not int for value in values.values()):
            return None
        count, local = values["_num_global_routed"], values["_num_local_routed"]
        if (
            count <= 0
            or local <= 0
            or values["moe_ep_size"] * local != count
            or values["moe_tp_size"] != 1
            or values["num_local_experts"] != local
            or values["num_fused_shared_experts"] != 0
        ):
            return None
        parameters = dict(module.named_parameters(recurse=False))
        if not all(key in parameters for key in ("w13_weight", "w2_weight")):
            return None
        for parameter in parameters.values():
            if getattr(parameter, "_sglang_require_global_experts", False) is not False:
                return None
            loader = getattr(parameter, "weight_loader", None)
            while isinstance(loader, partial):
                loader = loader.func
            if (
                getattr(loader, "__self__", None) is not module
                or getattr(loader, "__func__", None) is not FusedMoE.weight_loader
            ):
                return None
        if any(
            parameters[key].ndim < 2
            or parameters[key].shape[0] != local
            or parameters[key].dtype != torch.float8_e4m3fn
            for key in ("w13_weight", "w2_weight")
        ):
            return None
        if metadata is not None:
            cpu_map = metadata.logical_to_all_physical_map_cpu
            if (
                cpu_map.device.type != "cpu"
                or cpu_map.ndim != 3
                or not 0 <= values["layer_id"] < cpu_map.shape[0]
                or metadata.num_logical_experts != count
                or metadata.num_physical_experts != count
                or metadata.ep_size != values["moe_ep_size"]
                or any(
                    metadata.logical_to_all_physical(values["layer_id"], expert, False) != [expert]
                    for expert in range(count)
                )
            ):
                return None
        indices = [module._map_global_expert_id_to_local_expert_id(expert) for expert in range(count)]
        start = values["_expert_storage_rank"] * local
        if indices != [expert - start if start <= expert < start + local else -1 for expert in range(count)]:
            return None
        result.append(
            dict(
                module=name,
                **values,
                expert_begin=start,
                expert_end=start + local,
                parameters={
                    key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in parameters.items()
                },
            )
        )
    return result or None
