# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded cache for already-token-aligned Bridge absolute frequencies."""

import weakref
from collections import OrderedDict

import torch

from relax.models.qwen3_5.ops.rotary_embedding import Fused


class Cache:
    def __init__(self, capacity=2):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = 0
        self.misses = 0

    def apply(self, t, freqs):
        assert t.dtype == torch.bfloat16 and t.ndim == 4 and t.shape[1] == 1
        assert t.shape[0] == freqs.shape[0] and not freqs.requires_grad
        key = (
            id(freqs),
            freqs._version,
            tuple(freqs.shape),
            tuple(freqs.stride()),
            freqs.storage_offset(),
            freqs.device,
            freqs.dtype,
            t.device,
            t.dtype,
            torch.cuda.current_stream(t.device).cuda_stream,
        )
        entry = self.entries.get(key)
        if entry is not None and entry[0]() is freqs:
            self.hits += 1
            self.entries.move_to_end(key)
            c, s = entry[1:]
        else:
            self.misses += 1
            c = torch.cos(freqs).to(t.dtype).contiguous()
            s = torch.sin(freqs).to(t.dtype).contiguous()
            self.entries[key] = (weakref.ref(freqs), c, s)
            self.entries.move_to_end(key)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        return Fused.apply(t, c, s)


def make_wrapper(original, cache, stats):
    def wrapped(
        t,
        freqs,
        rotary_interleaved=False,
        mla_rotary_interleaved=False,
        mscale=1.0,
        inverse=False,
        mla_output_remove_interleaving=False,
        multi_latent_attention=None,
    ):
        supported = (
            multi_latent_attention is None
            and not rotary_interleaved
            and not mla_rotary_interleaved
            and mscale == 1.0
            and not inverse
            and not mla_output_remove_interleaving
            and t.is_cuda
            and t.dtype == torch.bfloat16
            and t.ndim == 4
            and t.shape[1] == 1
            and t.shape[-1] == 256
            and freqs.ndim == 4
            and freqs.shape == (t.shape[0], 1, 1, 64)
            and t.device == freqs.device
            and not freqs.requires_grad
            and not freqs.is_inference()
        )
        if not supported:
            stats["fallbacks"] += 1
            return original(
                t,
                freqs,
                rotary_interleaved=rotary_interleaved,
                mla_rotary_interleaved=mla_rotary_interleaved,
                mscale=mscale,
                inverse=inverse,
                mla_output_remove_interleaving=mla_output_remove_interleaving,
                multi_latent_attention=multi_latent_attention,
            )
        stats["calls"] += 1
        if "first_shape" not in stats:
            stats.update(first_shape=list(t.shape), first_freq_shape=list(freqs.shape))
        return cache.apply(t, freqs)

    return wrapped
