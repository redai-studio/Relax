# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in K3 operator installation before constructing the language model.

The switches apply to K3 models in the worker process and must be set before
model construction. Checkpoint parameter names and the KDA path are unchanged.
"""

from functools import wraps

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


def configure_training_ops() -> None:
    """Install each requested operator once, including Bridge's imported
    alias."""
    use_situ = Envs.RELAX_KIMI_K3_FUSED_SITU
    use_attn_res = Envs.RELAX_KIMI_K3_FUSED_ATTN_RES
    use_expert_split = Envs.RELAX_KIMI_K3_EXPERT_SPLIT
    if use_expert_split:
        from .expert_split import install_expert_split

        install_expert_split()
        get_logger(__name__).info("Enabled Kimi K3 grouped-expert LoRA split backward.")
    if not (use_situ or use_attn_res):
        return

    import torch
    from megatron.bridge.models.kimi import kimi_k3_layers, kimi_k3_ops

    if use_situ and not getattr(kimi_k3_ops.situ_and_mul, "_relax_fused", False):
        from .fused_situ import fused_situ_and_mul

        eager_situ = kimi_k3_ops.situ_and_mul

        @wraps(eager_situ)
        def situ(inputs: torch.Tensor, beta: float = 4.0, linear_beta: float = 25.0) -> torch.Tensor:
            if (
                inputs.is_cuda
                and inputs.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and beta > 0
                and linear_beta > 0
            ):
                return fused_situ_and_mul(inputs, beta=beta, linear_beta=linear_beta)
            return eager_situ(inputs, beta=beta, linear_beta=linear_beta)

        situ._relax_fused = True
        kimi_k3_ops.situ_and_mul = situ
        get_logger(__name__).info("Enabled Kimi K3 fused SiTU (precise tanh).")

    if use_attn_res and not getattr(kimi_k3_ops.attn_res_aggregate, "_relax_fused", False):
        from .fused_attn_res import fused_attn_res_aggregate

        # K3 layers import this function by value; updating ops alone misses
        # every training-layer call. The wrapper owns unsupported-input fallback.
        fused_attn_res_aggregate._relax_fused = True
        kimi_k3_ops.attn_res_aggregate = fused_attn_res_aggregate
        kimi_k3_layers.attn_res_aggregate = fused_attn_res_aggregate
        get_logger(__name__).info("Enabled Kimi K3 fused AttnRes aggregation.")
