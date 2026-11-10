# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import itertools
import os
from collections.abc import Callable

import torch
from torch import nn

from examples.chunkflow.adapters import (
    AttentionAdapter,
    DenseAttentionCall,
    attention_modules,
    ple,
    resolve_attention_adapter,
)
from examples.chunkflow.metadata import int32_metadata
from examples.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_ADAPTER_ATTR = "_chunkflow_attention_adapter"
_TRANSPOSE_CACHE_ATTR = "_chunkflow_original_transpose_cache"
_ROPE_FUSION_ATTR = "_chunkflow_original_rope_fusion"


class ChunkedCoreAttention(nn.Module):
    """Connect chunk history to a native executor; no attention math lives
    here."""

    def __init__(
        self,
        config,
        layer_number: int,
        softmax_scale=None,
        runtime=None,
        executor: Callable[[DenseAttentionCall], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.softmax_scale = softmax_scale
        self.runtime = runtime
        if executor is None:
            raise ValueError("ChunkFlow requires an explicit native attention executor")
        self.executor = executor

    def _forward_unit(self, query, key, value, packed_seq_params) -> torch.Tensor:
        unit = getattr(packed_seq_params, UNIT_ATTR, None)
        recompute = getattr(packed_seq_params, RECOMPUTE_ATTR, False)
        cu = packed_seq_params.cu_seqlens_q_cpu
        meta = [(c.group, c.group_idx) for c in unit.chunks for _ in c.segments]

        ks, vs, kv_lengths = [], [], []
        for i, (group, group_idx) in enumerate(meta):
            start, end = cu[i], cu[i + 1]
            k_own, v_own = key[start:end], value[start:end]
            if group is None:
                ks.append(k_own)
                vs.append(v_own)
                kv_lengths.append(end - start)
                continue

            states = self.runtime.history(self.layer_number, group, group_idx, recompute=recompute)
            current = states[-1]
            fp32 = os.environ.get("RELAX_CHUNKFLOW_KV_GRAD_FP32") == "1"
            current.link_tensor("k", k_own, accumulate_fp32=fp32)
            current.link_tensor("v", v_own, accumulate_fp32=fp32)

            earlier = states[:-1]
            ks.extend([*(s.tensors["k"] for s in earlier), k_own])
            vs.extend([*(s.tensors["v"] for s in earlier), v_own])
            kv_lengths.append(sum(s.tensors["k"].shape[0] for s in earlier) + end - start)

        query = query[: cu[-1]]
        metadata = getattr(packed_seq_params, "_chunkflow_attention_metadata", None)
        if metadata is None:
            cu_q = getattr(packed_seq_params, "cu_seqlens_q", None)
            if cu_q is None:
                cu_q = int32_metadata(cu, query.device)
            q_lengths = [end - start for start, end in zip(cu[:-1], cu[1:])]
            cu_kv = (
                cu_q
                if kv_lengths == q_lengths
                else int32_metadata([0, *itertools.accumulate(kv_lengths)], query.device)
            )
            metadata = (cu_q, cu_kv, (max(q_lengths), max(kv_lengths)))
            packed_seq_params._chunkflow_attention_metadata = metadata
        cu_q, cu_kv, max_seqlens = metadata
        scale = self.softmax_scale if self.softmax_scale is not None else query.shape[-1] ** -0.5
        return self.executor(
            DenseAttentionCall(
                query,
                tuple(ks),
                tuple(vs),
                cu_q,
                cu_kv,
                tuple(end - start for start, end in zip(cu[:-1], cu[1:])),
                tuple(kv_lengths),
                max_seqlens,
                scale,
            )
        )

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None, packed_seq_params=None, **kwargs):
        if attention_mask is not None or any(value is not None for value in kwargs.values()):
            raise NotImplementedError(
                "ChunkFlow dense KV adapter does not accept external masks, biases or extra inputs"
            )
        mask_type = getattr(attn_mask_type, "name", attn_mask_type)
        if mask_type not in (None, "causal", "padding_causal", "causal_bottom_right", "padding_causal_bottom_right"):
            raise NotImplementedError(f"ChunkFlow dense KV adapter requires causal attention, got {mask_type}")
        cp_size = getattr(self.config, "context_parallel_size", 1) or 1
        if cp_size > 1:
            raise NotImplementedError(
                "ChunkedCoreAttention does not implement context parallelism: chunkflow's "
                "StateStore is local and CP would split the very sequence it caches. "
                "Run with --context-parallel-size 1."
            )
        if query.dim() != 3:
            raise NotImplementedError(
                f"ChunkedCoreAttention expects THD [T, H, D]; got {query.dim()}D. "
                "SBHD would need its own causal-mask handling before it is safe to enable."
            )

        if getattr(packed_seq_params, UNIT_ATTR, None) is None:
            raise ValueError("ChunkFlow requires a scheduled unit; full calls must use the original forward")
        padded_rows = query.shape[0]
        out = self._forward_unit(query, key, value, packed_seq_params).reshape(-1, query.shape[1] * query.shape[2])
        if out.shape[0] < padded_rows:
            out = torch.nn.functional.pad(out, (0, 0, 0, padded_rows - out.shape[0]))
        return out


def _adaptation_root(model: nn.Module) -> nn.Module:
    # Bridge uses the same attention class for its causal text decoder and
    # bidirectional vision encoder. Only the text subtree is chunk-scheduled.
    model_type = f"{type(model).__module__}.{type(model).__qualname__}"
    if model_type == "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model.Qwen3VLModel":
        language_model = getattr(model, "language_model", None)
        if not isinstance(language_model, nn.Module):
            raise NotImplementedError("ChunkFlow requires a language_model on this Qwen3VL pipeline stage")
        return language_model
    return model


def install(model, chunk_size: int) -> int:
    from examples.chunkflow import rope as chunkflow_rope
    from examples.chunkflow.runtime import RUNTIME_ATTR, attach_runtime

    if chunk_size <= 0:
        raise ValueError("ChunkFlow chunk_size must be positive")
    if os.environ.get("RELAX_CHUNKFLOW_FLASH_BACKEND"):
        raise ValueError(
            "Remove RELAX_CHUNKFLOW_FLASH_BACKEND: ChunkFlow now delegates kernel selection to native Megatron/TE"
        )
    existing_runtime = getattr(model, RUNTIME_ATTR, None)
    if existing_runtime is not None and existing_runtime.chunk_size != chunk_size:
        raise ValueError("ChunkFlow is already configured with a different chunk size")

    # Validate the complete text subtree before replacing even its first layer.
    backbone = _adaptation_root(model)
    for module in backbone.modules():
        get_aux_coeff = getattr(module, "get_aux_loss_coeff", None)
        if callable(get_aux_coeff) and not getattr(module, "is_hash_layer", False):
            if any(get_aux_coeff(mode) > 0 for mode in ("aux_loss", "seq_aux_loss", "global_aux_loss")):
                raise NotImplementedError(
                    "ChunkFlow requires zero MoE load-balancing auxiliary loss: "
                    "chunk-local routing statistics change the full-batch objective."
                )
    ple_layers = {
        id(layer): layer
        for module in backbone.modules()
        if (layer := getattr(module, "per_layer_embedding", None)) is not None
    }
    local_layers = getattr(getattr(backbone, "decoder", None), "layers", None)
    if local_layers is not None:
        for layer in local_layers:
            expected = getattr(getattr(layer, "config", None), "ple_layer_ids", None) or []
            if layer.layer_number in expected and getattr(layer, "per_layer_embedding", None) is None:
                raise NotImplementedError(
                    f"ChunkFlow requires native per_layer_embedding on local layer {layer.layer_number}"
                )
    elif not ple_layers and any(
        getattr(getattr(module, "config", None), "ple_layer_ids", None) for module in backbone.modules()
    ):
        raise NotImplementedError("ChunkFlow PLE configuration requires native per_layer_embedding modules")
    plan = []
    for attention in attention_modules(backbone):
        if hasattr(attention, _ADAPTER_ATTR):
            continue
        adapter = resolve_attention_adapter(attention)
        adapter.validate(attention)
        plan.append((attention, adapter))
    for layer in ple_layers.values():
        if not hasattr(layer, _ADAPTER_ATTR):
            ple.validate(layer)
            plan.append((layer, AttentionAdapter("megatron-native-ple-prefix", ple.validate, ple.install)))
    runtime = attach_runtime(model, chunk_size)
    installed = []
    new_context = bool(ple_layers) and not hasattr(model, ple.CONTEXT_ATTR)
    try:
        if new_context:
            setattr(model, ple.CONTEXT_ATTR, ple.install_context(model))
        for attention, adapter in plan:
            undo = adapter.install(attention, chunk_size, runtime)
            setattr(attention, _ADAPTER_ATTR, undo)
            installed.append(attention)
            logger.info("chunkflow: layer %s uses adapter %s", attention.layer_number, adapter.name)
    except Exception:
        for attention in reversed(installed):
            getattr(attention, _ADAPTER_ATTR)()
            delattr(attention, _ADAPTER_ATTR)
        if new_context and hasattr(model, ple.CONTEXT_ATTR):
            getattr(model, ple.CONTEXT_ATTR)()
            delattr(model, ple.CONTEXT_ATTR)
        if existing_runtime is None:
            delattr(model, RUNTIME_ATTR)
        raise
    chunkflow_rope.install()

    # TE's first-forward flag can overwrite gradients after a later chunk's backward.
    for module in backbone.modules():
        if hasattr(module, "disable_parameter_transpose_cache") and hasattr(module, "is_first_microbatch"):
            if not hasattr(module, _TRANSPOSE_CACHE_ATTR):
                setattr(module, _TRANSPOSE_CACHE_ATTR, module.disable_parameter_transpose_cache)
            module.disable_parameter_transpose_cache = True

    # Chunk-local offsets require unfused RoPE.
    n = len(installed)
    if n:
        seen_configs: set[int] = set()
        for module in backbone.modules():
            cfg = getattr(module, "config", None)
            if cfg is not None and hasattr(cfg, "apply_rope_fusion") and id(cfg) not in seen_configs:
                seen_configs.add(id(cfg))
                if not hasattr(cfg, _ROPE_FUSION_ATTR):
                    setattr(cfg, _ROPE_FUSION_ATTR, cfg.apply_rope_fusion)
                cfg.apply_rope_fusion = False
        logger.info("chunkflow: apply_rope_fusion disabled (the offset path needs the unfused RoPE branch)")
    logger.info("chunkflow: installed %s native module adapters, chunk_size=%s", n, chunk_size)
    return n


def uninstall(model) -> int:
    from examples.chunkflow.runtime import RUNTIME_ATTR

    runtime = getattr(model, RUNTIME_ATTR, None)
    if runtime is not None:
        runtime.assert_empty("uninstall")

    for module in model.modules():
        if hasattr(module, _TRANSPOSE_CACHE_ATTR):
            module.disable_parameter_transpose_cache = getattr(module, _TRANSPOSE_CACHE_ATTR)
            delattr(module, _TRANSPOSE_CACHE_ATTR)
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, _ROPE_FUSION_ATTR):
            config.apply_rope_fusion = getattr(config, _ROPE_FUSION_ATTR)
            delattr(config, _ROPE_FUSION_ATTR)
    n = 0
    for attn in list(model.modules()):
        undo = getattr(attn, _ADAPTER_ATTR, None)
        if undo is None:
            continue
        undo()
        delattr(attn, _ADAPTER_ATTR)
        n += 1
    if runtime is not None:
        delattr(model, RUNTIME_ATTR)
    if hasattr(model, ple.CONTEXT_ATTR):
        getattr(model, ple.CONTEXT_ATTR)()
        delattr(model, ple.CONTEXT_ATTR)
    return n
