# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reviewed ordinary-loader capabilities, owned by the SGLang adapter.

Unknown functions, caches, quantization and storage transforms retain ordinary
FULL input. This is not a model selector or another synchronization protocol.
Inherited reviewed loaders reuse the same capability without size/name flags.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from typing import Any

import torch

from relax.utils.sparse_tensor import storage_identity


def full_loader_inputs(model: Any) -> frozenset[str]:
    """Aliased named inputs must preserve ordinary FULL's last-writer order."""
    aliases = {}
    # The reviewed SG loaders build params_dict from named_parameters(), which
    # omits duplicate names of the exact same Parameter. Distinct Parameters
    # sharing storage remain separate ordinary loader inputs.
    for name, parameter in model.named_parameters():
        aliases.setdefault(storage_identity(parameter), set()).add(name)
    return frozenset(name for names in aliases.values() if len(names) > 1 for name in names)


def _layout(value: Any) -> Any:
    """Freeze loader geometry without reading GPU values on the hot path."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_layout(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _layout(item)) for key, item in value.items()))
    if isinstance(value, torch.Tensor):
        return id(value), value._version, tuple(value.shape), value.dtype, tuple(value.stride())
    module = sys.modules.get("sglang.srt.layers.vocab_parallel_embedding")
    shard_type = getattr(module, "VocabParallelEmbeddingShardIndices", None)
    if shard_type is not None and type(value) is shard_type:
        return tuple((field.name, _layout(getattr(value, field.name))) for field in fields(value))
    # Unknown geometry is not a reviewed copy-only loader contract.
    raise TypeError("Unknown SGLang loader geometry")


def _geometry(owner: Any) -> tuple:
    return tuple(
        (name, _layout(getattr(owner, name, None)))
        for name in (
            "tp_rank",
            "tp_size",
            "kv_tp_rank",
            "kv_tp_size",
            "use_presharded_weights",
            "output_size",
            "output_sizes",
            "num_heads",
            "num_kv_heads",
            "head_size",
            "v_head_size",
            "num_kv_head_replicas",
            "total_num_heads",
            "total_num_kv_heads",
            "shard_indices",
            "org_vocab_size",
            "moe_tp_rank",
            "moe_tp_size",
            "moe_ep_rank",
            "_expert_storage_rank",
            "moe_ep_size",
            "num_local_experts",
            "num_fused_shared_experts",
            "layer_id",
            "_num_local_routed",
            "_num_global_routed",
            "_has_fused_shared",
            "intermediate_size_per_partition",
            "use_padded_loading",
            "expert_map",
            "local_expert_mapping",
            "use_triton_kernels",
        )
    )


def sparse_loader_identity(model: Any) -> tuple | None:
    loader = getattr(model, "load_weights", None)
    function = getattr(loader, "__func__", None)
    if function is None or getattr(loader, "__self__", None) is not model:
        return None
    # Exact function identities protect the actual code path, including
    # subclass or instance overrides. Never execute an unknown loader to probe
    # its effects on live parameters, buffers or Python tensor caches.
    known = []
    for module_name, class_name in (
        ("sglang.srt.models.qwen3", "Qwen3ForCausalLM"),
        ("sglang.srt.models.qwen3_moe", "Qwen3MoeForCausalLM"),
    ):
        module = sys.modules.get(module_name)
        cls = getattr(module, class_name, None)
        if cls is not None:
            known.append(cls.load_weights)
    if function not in known or getattr(model, "quant_config", None) is not None:
        return None
    aliased = full_loader_inputs(model)
    if aliased - {"model.embed_tokens.weight", "lm_head.weight"}:
        # Unknown aliases may name fused destinations rather than HF inputs.
        # Only ordinary direct embedding/head inputs have a reviewed mapping.
        return None
    from sglang.srt.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.unquant import (
        UnquantizedEmbeddingMethod,
        UnquantizedFusedMoEMethod,
        UnquantizedLinearMethod,
    )
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    if torch.version.hip is not None:
        return None  # AITER post-load shuffling is a different storage contract.
    methods = (UnquantizedEmbeddingMethod, UnquantizedLinearMethod, UnquantizedFusedMoEMethod)
    identity = [function, getattr(getattr(model, "config", None), "tie_word_embeddings", None)]
    for name, module in model.named_modules():
        if any(callable(value) and callable(getattr(type(module), key, None)) for key, value in vars(module).items()):
            return None  # An instance helper override changes the reviewed loader call graph.
        method = getattr(module, "quant_method", None)
        if (method is not None and type(method) not in methods) or getattr(module, "scheme", None) is not None:
            return None
        if method is not None and any(
            callable(value) and callable(getattr(type(method), key, None)) for key, value in vars(method).items()
        ):
            return None
        if isinstance(module, FusedMoE):
            if (
                type(module) is not FusedMoE
                or getattr(module, "quant_config", None) is not None
                or getattr(method, "use_flashinfer_trtllm_moe", False)
                or getattr(method, "use_flashinfer_cutlass", False)
                or getattr(method, "use_deep_gemm", False)
                or getattr(method, "scheme", None) is not None
                or getattr(method, "w13_swiglu_interleaved", False)
                or getattr(method, "use_triton_kernels", False) != module.use_triton_kernels
                or next(module.parameters()).device.type != "cuda"
            ):
                return None
            from sglang.srt.layers.moe.fused_moe_triton.layer import get_global_expert_location_metadata

            if get_global_expert_location_metadata() is not None:
                return None  # Dynamic physical-expert remapping needs a separately reviewed layout contract.
        try:
            geometry = _geometry(module)
        except TypeError:
            return None
        identity.append(
            (
                name,
                type(method),
                geometry,
                getattr(getattr(module, "moe_runner_config", None), "is_gated", None),
                getattr(method, "load_up_proj_weight_first", False),
            )
        )
    allowed = {
        cls.weight_loader: cls
        for cls in (
            ColumnParallelLinear,
            MergedColumnParallelLinear,
            QKVParallelLinear,
            ReplicatedLinear,
            RowParallelLinear,
            FusedMoE,
            VocabParallelEmbedding,
        )
    }
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if type(parameter) is not torch.nn.Parameter or parameter.dtype not in (torch.bfloat16, torch.float32):
            return None
        if parameter.numel() == 1:
            return None  # default_weight_loader uses scalar item()/fill_, not a masked copy.
        if any(
            getattr(parameter, name, False)
            for name in (
                "is_gguf_weight",
                "is_gguf_weight_type",
                "is_metadata",
                "needs_scalar_to_array",
                "use_bitsandbytes_4bit",
            )
        ) or any(
            getattr(parameter, key, None) is not None
            for key in ("packed_dim", "pack_factor", "packed_factor", "marlin_tile_size")
        ):
            return None  # These actual SG branches retain/transform input or read scalar values.
        bound = getattr(parameter, "weight_loader", default_weight_loader)
        operation = getattr(bound, "__func__", bound)
        owner = getattr(bound, "__self__", None)
        if operation is not default_weight_loader:
            expected = allowed.get(operation)
            if expected is None or (
                type(owner) is not expected
                and not (expected is VocabParallelEmbedding and type(owner) is ParallelLMHead)
            ):
                return None
        identity.append(
            (
                name,
                operation,
                id(owner),
                parameter.dtype,
                tuple(parameter.shape),
                tuple(parameter.stride()),
                storage_identity(parameter),
                tuple(
                    (key, getattr(parameter, key, None))
                    for key in ("input_dim", "output_dim", "is_transposed", "_sglang_require_global_experts")
                ),
            )
        )
    return tuple(identity)
