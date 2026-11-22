# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Remove routes of the independent physical padding document before expanded
DeepEP dispatch."""

from functools import wraps

import torch


def local_tail_mask(packed, hidden, tp_rank, tp_size):
    if packed is None or getattr(packed, "qkv_format", None) != "thd" or hidden.ndim != 3 or hidden.shape[1] != 1:
        return None
    bounds = getattr(packed, "cu_seqlens_q_cpu", None)
    lengths = getattr(packed, "valid_seqlens_q_cpu", None)
    if bounds is None or lengths is None:
        return None
    widths = [b - a for a, b in zip(bounds, bounds[1:])]
    valid_count = getattr(packed, "num_valid_tokens", None)
    if (
        len(widths) != len(lengths)
        or bounds[0] != 0
        or bounds[-1] != tp_size * hidden.shape[0]
        or any(v != w for v, w in zip(lengths[:-1], widths[:-1]))
        or lengths[-1] not in (0, widths[-1])
        or sum(lengths) != valid_count
    ):
        return None
    n = hidden.shape[0]
    remaining = max(0, min(n, valid_count - tp_rank * n))
    if remaining == n:
        return None
    cache = getattr(packed, "_q38_hybrid_tail_mask", {})
    key = (n, tp_rank, tp_size, remaining, hidden.device)
    if key not in cache:
        cache[key] = (torch.arange(n, device=hidden.device) >= remaining).reshape(n, 1)
        packed._q38_hybrid_tail_mask = cache
    return cache[key]


def adapt(native):
    from relax.utils.logging_utils import get_logger

    reported = False

    @wraps(native)
    def route(layer, hidden_states, padding_mask=None, input_ids=None, packed_seq_params=None):
        nonlocal reported
        probs, indices = native(layer, hidden_states, padding_mask, input_ids, packed_seq_params)
        group = layer.router.tp_group
        tail = local_tail_mask(packed_seq_params, hidden_states, group.rank(), group.size())
        if tail is None:
            return probs, indices
        if indices.ndim != 2 or indices.shape[0] != tail.numel():
            raise RuntimeError("Unexpected token-major router map")
        if indices.dtype == torch.bool:
            indices = indices & ~tail.reshape(-1, 1)
        else:
            indices = indices.masked_fill(tail.reshape(-1, 1), -1)
        if not reported:
            get_logger(__name__).info(
                "Expanded DeepEP omits only the independent THD padding tail; route dtype %s", indices.dtype
            )
            reported = True
        return probs, indices

    return route


def install(args):
    from megatron.core.transformer.moe.moe_layer import MoELayer

    coeff = args.moe_aux_loss_coeff
    coeff = coeff if isinstance(coeff, (tuple, list)) else [coeff]
    if (
        args.moe_flex_dispatcher_backend != "deepepv2"
        or args.context_parallel_size != 1
        or not args.sequence_parallel
        or any(coeff)
        or getattr(args, "moe_router_enable_expert_bias", False)
        or getattr(args, "moe_z_loss_coeff", None)
        or getattr(args, "moe_expert_capacity_factor", None) is not None
        or getattr(args, "moe_expert_rank_capacity_factor", None) is not None
    ):
        raise RuntimeError("Expected dropless CP1/SP expanded routing without a batch-dependent router objective")
    MoELayer.route = adapt(MoELayer.route)
