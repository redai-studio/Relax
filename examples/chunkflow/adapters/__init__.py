# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Explicit attention adapters; unknown implementations never inherit
support."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any

import torch
from torch import nn

from examples.chunkflow.adapters import gdn, qsa
from examples.chunkflow.runtime import UNIT_ATTR, ChunkflowRuntime


@dataclass(frozen=True)
class DenseAttentionCall:
    """Packed THD queries and per-sequence KV fragments, including history.

    Each query sequence is the suffix of its KV sequence: causal masking must
    align bottom-right. These are physical attention boundaries, not RoPE
    offsets. Executors return [T, H, D] or [T, H*D], with gradients to every KV
    fragment and any native parameters. No original packed metadata is reused.
    """

    query: torch.Tensor
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    cu_q: torch.Tensor
    cu_kv: torch.Tensor
    q_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]
    max_seqlens: tuple[int, int]
    softmax_scale: float

    def joined_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.keys[0] if len(self.keys) == 1 else torch.cat(self.keys, dim=0)
        value = self.values[0] if len(self.values) == 1 else torch.cat(self.values, dim=0)
        return key, value


NativeExecutor = Callable[[Callable[..., torch.Tensor], DenseAttentionCall], torch.Tensor]


@dataclass(frozen=True)
class AttentionAdapter:
    """Validate without mutation; install reversibly and roll back on
    failure."""

    name: str
    validate: Callable[[nn.Module], None]
    install: Callable[[nn.Module, int, ChunkflowRuntime], Callable[[], None]]


_ADAPTERS: dict[tuple[str, str | None], AttentionAdapter] = {}


def _type_name(value: type | str) -> str:
    return value if isinstance(value, str) else f"{value.__module__}.{value.__qualname__}"


def register_attention_adapter(
    attention_type: type | str, core_type: type | str | None, adapter: AttentionAdapter
) -> None:
    """Register before model installation, in each training worker.

    Use core_type=None for a full-layer adapter, including stateful native
    modules without core_attention. Classes match exactly; subclasses need
    their own registration. Duplicate entries are errors.
    """
    key = (_type_name(attention_type), None if core_type is None else _type_name(core_type))
    if key in _ADAPTERS:
        raise ValueError(f"ChunkFlow attention adapter already registered for {key}")
    _ADAPTERS[key] = adapter


def resolve_attention_adapter(attention: nn.Module) -> AttentionAdapter:
    layer_type = _type_name(type(attention))
    core = getattr(attention, "core_attention", None)
    key = (layer_type, _type_name(type(core)) if core is not None else None)
    adapter = _ADAPTERS.get((layer_type, None)) or _ADAPTERS.get(key)
    if adapter is None:
        raise NotImplementedError(
            f"ChunkFlow has no attention adapter for {key[0]} with {key[1]}. "
            "Register an exact-type AttentionAdapter that preserves history, positions, gradients and recomputation. "
            "Sparse/indexer attention cannot use the dense KV adapter without additional state support."
        )
    return adapter


def attention_modules(model: nn.Module) -> Iterator[nn.Module]:
    """Find registered layers and reject unregistered attention/SSM slots."""
    registered = {layer_type for layer_type, _ in _ADAPTERS}
    for name, module in model.named_modules():
        if _type_name(type(module)) in registered or (
            hasattr(module, "layer_number")
            and (
                getattr(module, "core_attention", None) is not None
                or hasattr(module, "gated_delta_rule")
                or name.rsplit(".", 1)[-1] == "self_attention"
            )
        ):
            yield module


def _name(value: Any) -> Any:
    return getattr(value, "name", value)


def validate_dense_attention(attention: nn.Module) -> None:
    """Validate the semantics implemented by the built-in dense executor."""
    core = attention.core_attention
    for obj in (attention, core, attention.config, getattr(core, "config", None)):
        if obj is None:
            continue
        if getattr(obj, "context_parallel_size", 1) != 1:
            raise NotImplementedError("ChunkFlow dense attention requires CP=1")
        if getattr(obj, "_relax_qwen35_optimized", False):
            raise NotImplementedError(
                "The Qwen3.5 optimized recipe bypasses native core forward; use a separate adapter"
            )
        if getattr(obj, "checkpoint_core_attention", False) or getattr(obj, "recompute_granularity", None) == "full":
            raise NotImplementedError(
                "Use ChunkFlow keep_last_k instead of Megatron core-attention/full recomputation"
            )
        if getattr(obj, "recompute_granularity", None) == "selective" and "core_attn" in (
            getattr(obj, "recompute_modules", None) or []
        ):
            raise NotImplementedError("Use ChunkFlow keep_last_k instead of Megatron core-attention recomputation")
        if getattr(obj, "hidden_dropout", 0):
            raise NotImplementedError("ChunkFlow requires hidden dropout=0 for chunk recomputation")
        if getattr(obj, "attention_type", "self") != "self":
            raise NotImplementedError("ChunkFlow dense attention requires self attention")
        mask = _name(getattr(obj, "attn_mask_type", None))
        if mask not in (None, "causal", "padding_causal", "causal_bottom_right", "padding_causal_bottom_right"):
            raise NotImplementedError(f"ChunkFlow dense attention does not support mask type {mask}")
        for field in ("attention_dropout", "dropout_p"):
            dropout = getattr(obj, field, 0)
            if isinstance(dropout, nn.Dropout):
                dropout = dropout.p
            if dropout:
                raise NotImplementedError("ChunkFlow dense attention requires attention dropout=0")
        window = getattr(obj, "window_size", None)
        if window is not None and (isinstance(window, int) or tuple(window) not in ((-1, -1), (-1, 0))):
            raise NotImplementedError("ChunkFlow dense attention needs a separate adapter for sliding windows")
        if _name(getattr(obj, "softmax_type", "vanilla")) != "vanilla":
            raise NotImplementedError("ChunkFlow dense attention requires vanilla softmax")
        for field in ("attention_softcap", "attn_logit_softcapping", "qk_clip", "log_max_attention_logit"):
            if getattr(obj, field, None):
                raise NotImplementedError(f"ChunkFlow dense attention does not implement {field}")


def dense_attention_adapter(
    *,
    name: str,
    execute: NativeExecutor,
    validate: Callable[[nn.Module], None] = validate_dense_attention,
) -> AttentionAdapter:
    """Reuse dense KV state/gradient management around a native executor.

    A native executor receives the original bound forward and an explicit
    DenseAttentionCall. It must implement bottom-right causality and physical
    packed boundaries. Ordinary autograd may retain concatenated history per
    chunk. Kernel selection and attention backward belong to the native module.
    """

    def install(attention: nn.Module, chunk_size: int, runtime: ChunkflowRuntime) -> Callable[[], None]:
        from examples.chunkflow.integration import ChunkedCoreAttention

        core = attention.core_attention
        original = core.forward
        signature = inspect.signature(original)
        had_forward = "forward" in core.__dict__
        previous_forward = core.__dict__.get("forward")
        executor = lambda call: execute(original, call)
        wrapper = ChunkedCoreAttention(
            config=getattr(core, "config", None) or attention.config,
            layer_number=attention.layer_number,
            softmax_scale=getattr(core, "softmax_scale", None),
            runtime=runtime,
            executor=executor,
        )

        # Keep module identity, parameters, hooks and checkpoint paths intact.
        @wraps(original)
        def forward(*args, **kwargs):
            packed = kwargs.get("packed_seq_params")
            if len(args) > 3:
                bound = signature.bind(*args, **kwargs)
                packed = bound.arguments.get("packed_seq_params")
                if getattr(packed, UNIT_ATTR, None) is not None:
                    return wrapper(**bound.arguments)
            if getattr(packed, UNIT_ATTR, None) is None:
                return original(*args, **kwargs)
            return wrapper(*args, **kwargs)

        core.forward = forward

        def undo() -> None:
            if had_forward:
                core.forward = previous_forward
            else:
                del core.forward

        return undo

    return AttentionAdapter(name, validate, install)


class _PackedCausalMask(Enum):
    # Pinned MCore 4a4de865 forwards mask.name to TE but lacks this enum member.
    # TE 2.18 accepts this name; ordinary causal would align unequal Q/KV wrongly.
    padding_causal_bottom_right = "padding_causal_bottom_right"


def megatron_native_attention(original: Callable[..., torch.Tensor], call: DenseAttentionCall) -> torch.Tensor:
    """Call the saved MCore forward; MCore/TE own all attention computation."""
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.enums import AttnMaskType

    key, value = call.joined_kv()
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=call.cu_q,
        cu_seqlens_kv=call.cu_kv,
        max_seqlen_q=call.max_seqlens[0],
        max_seqlen_kv=call.max_seqlens[1],
    )
    # Never copy RoPE offsets into TE's physical padded boundaries.
    mask = getattr(AttnMaskType, "padding_causal_bottom_right", _PackedCausalMask.padding_causal_bottom_right)
    return original(
        call.query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attention_mask=None,
        attn_mask_type=mask,
        packed_seq_params=packed,
    )


def _validate_megatron_attention(attention: nn.Module) -> None:
    validate_dense_attention(attention)
    core = attention.core_attention
    if not getattr(core, "te_forward_mask_type", False):
        raise NotImplementedError("ChunkFlow requires native TE per-call mask selection for bottom-right causality")
    if "forward" in attention.__dict__ or "forward" in core.__dict__:
        raise NotImplementedError("Patched attention forward requires an explicit native adapter")
    groups = getattr(core, "cp_group", None)
    if groups is not None:
        import torch.distributed as dist

        for group in groups if isinstance(groups, (list, tuple)) else (groups,):
            if group is not None and dist.get_world_size(group) != 1:
                raise NotImplementedError("ChunkFlow requires native TE CP groups of size 1")


register_attention_adapter(
    "megatron.core.transformer.attention.SelfAttention",
    "megatron.core.extensions.transformer_engine.TEDotProductAttention",
    dense_attention_adapter(
        name="megatron-native-te", execute=megatron_native_attention, validate=_validate_megatron_attention
    ),
)

# Qwen3.5/3.6 text layers use Bridge's absolute-RoPE attention class. Its
# native QKV, RoPE and output gate surround the same TE core call.
register_attention_adapter(
    "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention.Qwen3VLSelfAttention",
    "megatron.core.extensions.transformer_engine.TEDotProductAttention",
    dense_attention_adapter(
        name="megatron-native-qwen3vl-te", execute=megatron_native_attention, validate=_validate_megatron_attention
    ),
)


for gdn_type in (
    "megatron.core.ssm.gated_delta_net.GatedDeltaNet",
    "megatron.core.ssm.gated_delta_net.gdn.GatedDeltaNet",
):
    register_attention_adapter(gdn_type, None, AttentionAdapter("megatron-native-gdn", gdn.validate, gdn.install))

register_attention_adapter(
    "megatron.core.transformer.experimental_attention_variant.qsa.QwenSparseSelfAttention",
    None,
    AttentionAdapter("megatron-native-qsa-prefix", qsa.validate, qsa.install),
)
