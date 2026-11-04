# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import itertools
import os

import torch
from torch import nn

from relax.backends.megatron.chunkflow.attention_ops import chunk_attention_forward
from relax.backends.megatron.chunkflow.flash_backends import flash3_varlen_ops, flash_backend_name
from relax.backends.megatron.chunkflow.metadata import execution_optimizations_enabled, int32_metadata
from relax.backends.megatron.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR
from relax.backends.megatron.chunkflow.state_store import ChunkState, StateStore
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_ORIG_ATTR = "_chunkflow_original_core_attention"
_TRANSPOSE_CACHE_ATTR = "_chunkflow_original_transpose_cache"


def split_sizes(total: int, chunk_size: int) -> list[int]:
    if chunk_size <= 0 or chunk_size >= total:
        return [total]
    sizes = [chunk_size] * (total // chunk_size)
    tail = total - sum(sizes)
    if tail:
        sizes.append(tail)
    return sizes


class ChunkedCoreAttention(nn.Module):
    def __init__(self, config, layer_number: int, chunk_size: int, softmax_scale=None, runtime=None, **kwargs):
        super().__init__()
        del kwargs
        self.config = config
        self.layer_number = layer_number
        self.chunk_size = chunk_size
        self.softmax_scale = softmax_scale
        self.runtime = runtime

    def _forward_unit(self, query, key, value, packed_seq_params) -> torch.Tensor:
        from relax.backends.megatron.chunkflow.attention_ops import UnitCacheAttention

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

            store = self.runtime.store(self.layer_number, group)
            if not recompute:
                store.put(
                    group_idx,
                    ChunkState({"k": _cache_leaf(k_own), "v": _cache_leaf(v_own)}),
                )
            states = store.lookup(group_idx)
            _fold_leaf_grad_into(k_own, states[-1].tensors["k"])
            _fold_leaf_grad_into(v_own, states[-1].tensors["v"])

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
        out, _lse = UnitCacheAttention.apply(query, True, scale, cu_q, cu_kv, *ks, *vs, max_seqlens)
        return out

    def _attend_one_sequence(self, q, k, v) -> torch.Tensor:
        store = StateStore()
        outs = []
        offset = 0
        for chunk_id, size in enumerate(split_sizes(q.shape[0], self.chunk_size)):
            sl = slice(offset, offset + size)
            out, _lse = chunk_attention_forward(
                chunk_id, q[sl], k[sl], v[sl], store, causal=True, softmax_scale=self.softmax_scale
            )
            outs.append(out)
            offset += size
        return torch.cat(outs, dim=0)

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None, packed_seq_params=None, **kwargs):
        del attention_mask, attn_mask_type, kwargs
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

        if packed_seq_params is not None and getattr(packed_seq_params, UNIT_ATTR, None) is not None:
            padded_rows = query.shape[0]
            out = self._forward_unit(query, key, value, packed_seq_params).reshape(-1, query.shape[1] * query.shape[2])
            if out.shape[0] < padded_rows:
                out = torch.nn.functional.pad(out, (0, 0, 0, padded_rows - out.shape[0]))
            return out

        cu = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
        if cu is None:
            out = self._attend_one_sequence(query, key, value)
        else:
            bounds = getattr(packed_seq_params, "cu_seqlens_q_cpu", None)
            if bounds is None:
                bounds = cu.detach().cpu().tolist()
            outs = []
            for start, end in zip(bounds[:-1], bounds[1:]):
                if end > start:
                    outs.append(self._attend_one_sequence(query[start:end], key[start:end], value[start:end]))
            out = torch.cat(outs, dim=0)
        return out.reshape(out.shape[0], -1)


def _cache_leaf(tensor: torch.Tensor) -> torch.Tensor:
    leaf = tensor.detach().requires_grad_(True)
    if os.environ.get("RELAX_CHUNKFLOW_KV_GRAD_FP32") == "1":
        leaf._chunkflow_grad_sum = None

        def accumulate(value: torch.Tensor) -> None:
            grad = value.grad
            dtype = torch.float32 if grad.dtype in (torch.float16, torch.bfloat16) else grad.dtype
            if value._chunkflow_grad_sum is None:
                value._chunkflow_grad_sum = grad.to(dtype).clone()
            else:
                value._chunkflow_grad_sum.add_(grad)
            value.grad = None

        leaf.register_post_accumulate_grad_hook(accumulate)
    return leaf


def _fold_leaf_grad_into(live: torch.Tensor, leaf: torch.Tensor) -> None:
    # Descending backward folds later chunks' leaf gradients into this chunk.
    def hook(grad: torch.Tensor) -> torch.Tensor:
        if hasattr(leaf, "_chunkflow_grad_sum"):
            cached = leaf._chunkflow_grad_sum
            return grad if cached is None else (grad.to(cached.dtype) + cached).to(grad.dtype)
        return grad if leaf.grad is None else grad + leaf.grad

    # Fp has no graph; Fr attaches this hook to recomputed K/V.
    if live.requires_grad:
        live.register_hook(hook)


def _attention_modules(model):
    for module in model.modules():
        core = getattr(module, "core_attention", None)
        if core is not None and hasattr(module, "layer_number"):
            yield module


def install(model, chunk_size: int) -> int:
    from relax.backends.megatron.chunkflow.runtime import attach_runtime

    backend = flash_backend_name()
    if backend == "fa3":
        flash3_varlen_ops()
    logger.info(f"chunkflow: flash attention backend={backend}")
    runtime = attach_runtime(model, chunk_size)
    from relax.backends.megatron.chunkflow import gdn
    from relax.backends.megatron.chunkflow import rope as chunkflow_rope

    gdn_layers = gdn.install(model, runtime)
    if gdn_layers:
        logger.info("chunkflow: stateful GDN installed on %d layer(s)", gdn_layers)
    chunkflow_rope.install()

    # TE's first-forward flag can overwrite gradients after a later chunk's backward.
    for module in model.modules():
        if hasattr(module, "disable_parameter_transpose_cache") and hasattr(module, "is_first_microbatch"):
            if not hasattr(module, _TRANSPOSE_CACHE_ATTR):
                setattr(module, _TRANSPOSE_CACHE_ATTR, module.disable_parameter_transpose_cache)
            module.disable_parameter_transpose_cache = True

    # Chunk-local offsets require unfused RoPE.
    n = 0
    for attn in _attention_modules(model):
        if hasattr(attn, _ORIG_ATTR):
            continue
        original = attn.core_attention
        setattr(attn, _ORIG_ATTR, original)
        attn.core_attention = ChunkedCoreAttention(
            config=getattr(original, "config", None) or attn.config,
            layer_number=attn.layer_number,
            chunk_size=chunk_size,
            softmax_scale=getattr(original, "softmax_scale", None),
            runtime=runtime,
        )
        n += 1
    if n:
        seen_configs: set[int] = set()
        for module in model.modules():
            cfg = getattr(module, "config", None)
            if cfg is not None and hasattr(cfg, "apply_rope_fusion") and id(cfg) not in seen_configs:
                seen_configs.add(id(cfg))
                cfg.apply_rope_fusion = False
        logger.info("chunkflow: apply_rope_fusion disabled (the offset path needs the unfused RoPE branch)")
    logger.info(f"chunkflow: chunked attention installed on {n} layer(s), chunk_size={chunk_size}")
    logger.info("chunkflow: execution optimization enabled=%s", execution_optimizations_enabled())
    return n


def uninstall(model) -> int:
    from relax.backends.megatron.chunkflow import gdn

    gdn.uninstall(model)
    for module in model.modules():
        if hasattr(module, _TRANSPOSE_CACHE_ATTR):
            module.disable_parameter_transpose_cache = getattr(module, _TRANSPOSE_CACHE_ATTR)
            delattr(module, _TRANSPOSE_CACHE_ATTR)
    n = 0
    for attn in _attention_modules(model):
        original = getattr(attn, _ORIG_ATTR, None)
        if original is None:
            continue
        attn.core_attention = original
        delattr(attn, _ORIG_ATTR)
        n += 1
    return n
