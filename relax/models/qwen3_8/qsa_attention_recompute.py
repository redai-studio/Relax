# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in activation retention for Qwen3.8 attention or full layers.

Retaining attention also keeps its input norm, PLE, mHC and BDA activations.
"""

from __future__ import annotations

import inspect
import json
from contextvars import ContextVar
from typing import Any, Callable

from relax.utils.env import Envs


# Validated TP, PP, and the last-stage attention layer. The last-stage
# policies also retain decoder layer 47 and the full MTP inner layer.
_RETENTION_POLICIES = {
    "attention": (8, 4, None),
    "last_stage_l44": (8, 8, 44),
    "last_stage_l48": (4, 8, 48),
}


def _retention_policy() -> str:
    policy = Envs.RELAX_QWEN38_QSA_RETENTION_POLICY
    if policy is not None:
        return policy
    # The original 128K recipe passed these settings to the actor.
    if (Envs.AIO_TUNE_PP7_RETAIN, Envs.AIO_TUNE_PP7_FULL_LAYER, Envs.AIO_TUNE_PP7_FULL_SCOPE) == ("1", "47", "mtp"):
        return "last_stage_l44"
    return "attention"


def validate_args(args: Any, policy: str = "attention") -> None:
    if policy not in _RETENTION_POLICIES:
        raise ValueError(
            f"Unknown RELAX_QWEN38_QSA_RETENTION_POLICY={policy!r}. Choose one of {tuple(_RETENTION_POLICIES)}"
        )
    tp_size, pp_size, _ = _RETENTION_POLICIES[policy]
    expected = {
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
        "tensor_model_parallel_size": tp_size,
        "pipeline_model_parallel_size": pp_size,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
    }
    for name, value in expected.items():
        actual = getattr(args, name, None)
        if actual != value:
            raise RuntimeError(f"QSA retention policy {policy!r} requires {name}={value!r}, got {actual!r}")
    if not getattr(args, "bf16", False) or getattr(args, "fp8", None) or getattr(args, "fp4", None):
        raise RuntimeError("QSA retained attention requires BF16 without FP8/FP4")
    if getattr(args, "cpu_offloading", False) or getattr(args, "distribute_saved_activations", False):
        raise RuntimeError("QSA retained attention excludes activation offload/distributed saved activations")
    if getattr(args, "virtual_pipeline_model_parallel_size", None):
        raise RuntimeError("QSA retained attention does not support virtual pipeline")
    if getattr(args, "cuda_graph_impl", "none") not in (None, "none"):
        raise RuntimeError("QSA retained attention does not support CUDA graphs")


_active_retained_mlp: ContextVar[frozenset[int]] = ContextVar("qsa_retained_mlp", default=frozenset())


def retained_mlp(
    layer: Any,
    hidden_states: Any,
    inference_context: Any = None,
    padding_mask: Any = None,
    input_ids: Any = None,
    packed_seq_params: Any = None,
    mhc_recompute_manager: Any = None,
    *,
    native_forward: Callable | None = None,
) -> Any:
    """Checkpoint one complete MLP, including subclasses that call
    ``super()``."""
    from megatron.core import tensor_parallel

    native = layer._relax_native_forward_mlp if native_forward is None else native_forward.__get__(layer)
    kwargs = dict(input_ids=input_ids, packed_seq_params=packed_seq_params)
    if mhc_recompute_manager is not None:
        kwargs["mhc_recompute_manager"] = mhc_recompute_manager
    if (
        not layer.training
        or id(layer) in _active_retained_mlp.get()
        or getattr(layer, "_relax_qsa_full_layer_retained", False)
    ):
        return native(hidden_states, inference_context=inference_context, padding_mask=padding_mask, **kwargs)
    if inference_context is not None:
        raise RuntimeError("QSA retained attention is training-only")

    def run(hidden, padding):
        token = _active_retained_mlp.set(_active_retained_mlp.get() | {id(layer)})
        try:
            return native(hidden, inference_context=None, padding_mask=padding, **kwargs)
        finally:
            _active_retained_mlp.reset(token)

    return tensor_parallel.checkpoint(run, layer.config.distribute_saved_activations, hidden_states, padding_mask)


def configure(args: Any, chunks: Any) -> list[dict[str, Any]]:
    if not Envs.RELAX_QWEN38_RETAIN_QSA_ATTENTION:
        return []
    policy = _retention_policy()
    validate_args(args, policy)
    _, _, attention_layer_number = _RETENTION_POLICIES[policy]
    if attention_layer_number is not None:
        from megatron.core import parallel_state

        if not parallel_state.is_pipeline_last_stage():
            return []
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import Qwen3VLTransformerBlock
    from megatron.core import recompute
    from megatron.core.models.hybrid import hybrid_block
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
    from megatron.core.transformer.transformer_layer import TransformerLayer

    modules = [
        (f"chunk{i}.{name}", module) for i, chunk in enumerate(chunks) for name, module in chunk.named_modules()
    ]
    layers = []
    stacks = []
    bridge_stacks = []
    # Complete validation before changing any object or module binding.
    for path, module in modules:
        if isinstance(module, MultiTokenPredictionLayer) and module.mtp_layer_pattern is None:
            raise RuntimeError("Legacy MTP outer checkpoint is unsupported")
        if isinstance(module, (hybrid_block.HybridStack, Qwen3VLTransformerBlock)):
            config = module.config
            if (config.recompute_granularity, config.recompute_method, config.recompute_num_layers) != (
                "full",
                "uniform",
                1,
            ):
                raise RuntimeError("HybridStack effective recompute config differs from guarded args")
            if getattr(config, "cpu_offloading", False) or getattr(config, "distribute_saved_activations", False):
                raise RuntimeError("HybridStack offload/distributed saved activations are unsupported")
            if getattr(config, "fp8", None) or getattr(config, "fp4", None):
                raise RuntimeError("Retained attention excludes quantized stack configurations")
            if getattr(config, "cuda_graph_impl", "none") not in (None, "none"):
                raise RuntimeError("Retained attention excludes stack CUDA graphs")
            if isinstance(module, Qwen3VLTransformerBlock):
                if not callable(getattr(module, "_relax_native_checkpointed_forward", None)):
                    raise RuntimeError("Bridge is missing the build-time QSA recompute integration")
                if "_checkpointed_forward" in module.__dict__:
                    raise RuntimeError("Bridge instance checkpoint already overridden")
                if inspect.getsourcefile(type(module)._checkpointed_forward) != inspect.getsourcefile(
                    Qwen3VLTransformerBlock
                ):
                    raise RuntimeError("Unexpected Bridge checkpoint override")
                bridge_stacks.append(module)
            else:
                stacks.append(module)
        attention = getattr(module, "self_attention", None)
        if type(attention).__name__ != "QwenSparseSelfAttention":
            continue
        if not isinstance(module, TransformerLayer) or not callable(getattr(module, "_forward_mlp", None)):
            raise RuntimeError("Unsupported QSA owner")
        if not callable(getattr(module, "_relax_native_forward_mlp", None)):
            raise RuntimeError("TransformerLayer is missing the build-time QSA recompute integration")
        if getattr(module, "_relax_qsa_attention_retained", False):
            raise RuntimeError("QSA retained attention already installed")
        if attention.config.qsa_indexer_loss_coeff != 0 or any(
            p.requires_grad for p in attention.indexer.parameters()
        ):
            raise RuntimeError("Only frozen Stage-1 QSA indexers are supported")
        if any(
            getattr(module, flag, False)
            for flag in ("recompute_mlp", "recompute_input_layernorm", "recompute_pre_mlp_layernorm")
        ) or getattr(module.mlp, "moe_layer_recompute", False):
            raise RuntimeError("Unexpected inner MLP recomputation")
        layers.append((path, module))
    owned = {id(layer) for stack in stacks + bridge_stacks for layer in stack.layers}
    if not layers or any(id(layer) not in owned for _, layer in layers):
        raise RuntimeError("All QSA owners must be direct children of HybridStack or guarded Bridge block")
    if hybrid_block.checkpointed_forward is not recompute.checkpointed_forward:
        raise RuntimeError("HybridStack checkpoint entry already changed; review patch composition")
    full_decoder, full_mtp = None, None
    if attention_layer_number is not None:
        # Resolve global decoder numbers within the final Bridge block so
        # module traversal order cannot change the selected layers.
        if len(bridge_stacks) != 1:
            raise RuntimeError("Last-stage QSA retention requires one final-stage Bridge block")
        decoder = {layer.layer_number: layer for layer in bridge_stacks[0].layers}
        attention, full_decoder = decoder.get(attention_layer_number), decoder.get(47)
        mtp_stacks = [stack for stack in stacks if getattr(stack, "is_mtp_layer", False)]
        if full_decoder is None or len(mtp_stacks) != 1 or len(mtp_stacks[0].layers) != 1:
            raise RuntimeError("Last-stage QSA retention requires decoder layer 47 and a one-layer MTP stack")
        full_mtp = mtp_stacks[0].layers[0]
        qsa_owners = {id(layer) for _, layer in layers}
        if id(attention) not in qsa_owners or id(full_mtp) not in qsa_owners:
            raise RuntimeError(
                f"Last-stage QSA retention requires QSA in decoder layer {attention_layer_number} and the MTP layer"
            )
        layers = [(path, layer) for path, layer in layers if layer is attention]

    manifest = [
        {
            "path": path,
            "layer_number": layer.layer_number,
            "attention": "retain _forward_attention including norm/PLE/mHC/BDA",
            "mlp": "checkpoint original _forward_mlp",
        }
        for path, layer in layers
    ]
    for _, layer in layers:
        layer._relax_qsa_attention_retained = True
    if full_decoder is not None:
        full_decoder._relax_qsa_full_layer_retained = True
        # HybridStack uses the attention marker to select the split forward.
        full_mtp._relax_qsa_attention_retained = True
        full_mtp._relax_qsa_full_layer_retained = True
        manifest.extend(
            (
                {"layer_number": full_decoder.layer_number, "scope": "full decoder layer"},
                {"layer_number": full_mtp.layer_number, "scope": "full MTP layer"},
            )
        )
    from relax.utils.logging_utils import get_logger

    get_logger(__name__).info("QSA attention-retention manifest: %s", json.dumps(manifest))
    return manifest
