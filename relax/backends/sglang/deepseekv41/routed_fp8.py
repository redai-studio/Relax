# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Stream V4.1 BF16 routed experts into SGLang's native Triton block-FP8 layout."""

from contextvars import ContextVar
from functools import wraps

import torch
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8


class RoutedFp8MoE(FusedMoE):
    """Quantize each routed expert projection as it arrives from HF or Megatron."""

    def __init__(self, *, quant_config=None, num_fused_shared_experts=0, **kwargs):
        backend = get_moe_runner_backend()
        if not get_moe_a2a_backend().is_none() or not (backend.is_triton() or backend.is_auto()):
            raise ValueError("V4.1 BF16-to-FP8 routed experts require the Triton MoE runner and a2a=none.")
        if quant_config is not None or num_fused_shared_experts or kwargs.get("use_weight_loader_fused"):
            raise ValueError("V4.1 routed FP8 requires separate BF16 expert projections and unfused shared experts.")

        # Request SGLang's FP8 destination buffers from the outset. The source
        # tensors remain BF16. In this backend the post-load hook keeps the native
        # block layout and both Parameter objects and their addresses are stable.
        config = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        super().__init__(
            **kwargs,
            quant_config=config,
            quant_method=Fp8MoEMethod(config),
            num_fused_shared_experts=0,
        )

    @torch.no_grad()
    def _weight_loader_impl(self, param, loaded_weight, weight_name, shard_id, expert_id):
        # FusedMoE.weight_loader already dropped non-owned experts and translated
        # the global id to a local physical slot before reaching this method.
        if shard_id == "w2" and param is self.w2_weight:
            scales = self.w2_weight_scale_inv
        elif shard_id in ("w1", "w3") and param is self.w13_weight:
            scales = self.w13_weight_scale_inv
        else:
            raise ValueError(f"Unexpected routed expert projection {weight_name} ({shard_id})")
        if loaded_weight.dtype != torch.bfloat16 or loaded_weight.ndim != 2:
            raise ValueError(f"{weight_name} needs a BF16 matrix, got {loaded_weight.dtype} {loaded_weight.shape}")
        width = self.intermediate_size_per_partition
        if not self.use_presharded_weights:
            width *= self.moe_tp_size
        shape = (self.hidden_size, width) if shard_id == "w2" else (width, self.hidden_size)
        if tuple(loaded_weight.shape) != shape:
            raise ValueError(f"{weight_name} ({shard_id}) expects {shape}, got {tuple(loaded_weight.shape)}")

        with torch.cuda.device(param.device):
            qweight, scale = per_block_cast_to_fp8(loaded_weight.to(param.device))
            # Each 128x128 block has its own scale, including the two W13
            # projections. Updating one projection never requantizes its sibling.
            native_load = super()._weight_loader_impl
            native_load(param, qweight, weight_name, shard_id, expert_id)
            native_load(scales, scale, weight_name + "_scale_inv", shard_id, expert_id)


def apply_routed_fp8_patch() -> None:
    """Install in the scheduler process before building the opt-in BF16 V4.1 model."""
    from sglang.srt.models import deepseek_v2

    original_init = deepseek_v2.DeepseekV2MoE.__init__
    if getattr(original_init, "_relax_v41_routed_fp8", False):
        return
    original_factory = deepseek_v2.get_moe_impl_class
    v41 = ContextVar("relax_v41_routed_fp8", default=False)

    @wraps(original_init)
    def init_moe(self, config, *args, **kwargs):
        token = v41.set(getattr(config, "model_type", None) == "deepseek_v41")
        try:
            original_init(self, config, *args, **kwargs)
        finally:
            v41.reset(token)

    @wraps(original_factory)
    def get_moe_impl_class(quant_config):
        cls = original_factory(quant_config)
        if not v41.get():
            return cls
        if cls is not FusedMoE:
            raise ValueError("DeepSeek V4.1 routed FP8 requires the standard FusedMoE implementation")
        return RoutedFp8MoE

    init_moe._relax_v41_routed_fp8 = True
    deepseek_v2.get_moe_impl_class = get_moe_impl_class
    deepseek_v2.DeepseekV2MoE.__init__ = init_moe
