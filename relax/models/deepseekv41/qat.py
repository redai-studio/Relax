# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

from megatron.core.models.gpt import GPTModel

from relax.utils.logging_utils import get_logger

from .provider import DeepSeekV41ModelProvider


class DeepSeekV41QATModelProvider(DeepSeekV41ModelProvider):
    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> GPTModel:
        from relax.models.deepseek_v4.master_weights import install_master_weight_hooks
        from relax.models.deepseek_v4.qat import install_mxfp4_qat

        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        selected = install_mxfp4_qat(model, compute="global_bf16")
        install_master_weight_hooks()
        get_logger(__name__).info("1017 routed MXFP4 QAT enabled for %d expert linears (BF16 GEMM)", len(selected))
        return model


def enable_routed_mxfp4_qat(args: Namespace) -> None:
    """Select the routed-expert provider in this QAT SFT worker before model construction."""
    from .bridge import DeepSeekV41Bridge

    if args.loss_type != "sft" or args.megatron_to_hf_mode != "bridge" or not args.bf16:
        raise ValueError("1017 routed MXFP4 QAT requires BF16 SFT using the Bridge provider")
    DeepSeekV41Bridge.PROVIDER_CLASS = DeepSeekV41QATModelProvider
