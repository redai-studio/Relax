# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local geometry contracts shared by mixed QSA entry points and adapters."""


def validate_geometry(backend: str, q_heads: int, kv_heads: int, q_dim: int, k_dim: int) -> None:
    """Check the selected backend before document routing or kernel
    execution."""
    if backend not in ("grouped", "cudnn_pr1765"):
        raise ValueError(f"Unknown mixed QSA long backend: {backend}")
    actual = f"{q_heads}Q/{kv_heads}KV, Q D{q_dim}/K D{k_dim}"
    if backend == "cudnn_pr1765":
        # Match the local head pairs in the pinned NVIDIA PR1765 contract.
        if (q_heads, kv_heads) not in ((12, 1), (24, 2)) or (q_dim, k_dim) != (256, 256):
            raise NotImplementedError(
                f"Mixed QSA backend={backend} received {actual}. "
                "The Relax adapter supports 12Q/1KV/D256 and 24Q/2KV/D256. "
                "For other positive integral GQA layouts with D256, select "
                "RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped."
            )
    elif q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads or q_dim != 256 or k_dim != 256:
        raise NotImplementedError(
            f"Mixed QSA backend={backend} received {actual}; requires positive Q/KV heads, "
            "integral local GQA groups (Q heads divisible by KV heads), and D256 for Q and K."
        )
