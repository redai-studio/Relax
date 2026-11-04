# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Callable


FlashOps = tuple[Callable[..., Any], Callable[..., Any]]


def flash_backend_name() -> str:
    backend = os.environ.get("RELAX_CHUNKFLOW_FLASH_BACKEND", "fa2")
    if backend not in ("fa2", "fa3"):
        raise ValueError("RELAX_CHUNKFLOW_FLASH_BACKEND must be fa2 or fa3")
    return backend


@lru_cache(maxsize=1)
def flash3_varlen_ops() -> FlashOps:
    # Use TE's FA3 namespace to avoid duplicate custom-op registration.
    try:
        from flash_attn_3.flash_attn_interface import _flash_attn_backward, _flash_attn_forward
    except ImportError as exc:
        raise RuntimeError(
            "ChunkFlow explicitly requested fa3, but flash_attn_3.flash_attn_interface is unavailable"
        ) from exc

    def forward(
        q: Any,
        k: Any,
        v: Any,
        cu_q: Any,
        cu_kv: Any,
        max_q: int,
        max_kv: int,
        dropout: float,
        scale: float,
        causal: bool,
    ) -> tuple[Any, Any, None, None]:
        if dropout != 0:
            raise ValueError("ChunkFlow's FA3 adapter requires dropout=0")
        out, lse, *_rest = _flash_attn_forward(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv,
            max_seqlen_q=max_q,
            max_seqlen_k=max_kv,
            softmax_scale=scale,
            causal=causal,
            window_size_left=-1,
            window_size_right=-1,
        )
        return out, lse, None, None

    def backward(
        dout: Any,
        q: Any,
        k: Any,
        v: Any,
        out: Any,
        lse: Any,
        dq: Any,
        dk: Any,
        dv: Any,
        cu_q: Any,
        cu_kv: Any,
        max_q: int,
        max_kv: int,
        dropout: float,
        scale: float,
        causal: bool,
        window_left: int,
        window_right: int,
        softcap: float,
        alibi: Any,
        deterministic: bool,
    ) -> Any:
        if dropout != 0 or alibi is not None:
            raise ValueError("ChunkFlow's FA3 adapter requires dropout=0 and no ALiBi")
        return _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv,
            max_seqlen_q=max_q,
            max_seqlen_k=max_kv,
            dq=dq,
            dk=dk,
            dv=dv,
            softmax_scale=scale,
            is_causal=causal,
            window_size_left=window_left,
            window_size_right=window_right,
            softcap=softcap,
            deterministic=deterministic,
        )

    return forward, backward
