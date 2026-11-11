# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Validate the recipe's construction-time LCE backward configuration."""


def get_bwd_config():
    try:
        from megatron.core.fusions.linear_cross_entropy.blackwell import entry
    except ModuleNotFoundError as error:
        if error.name != "megatron.core.fusions.linear_cross_entropy":
            raise
        from relax.backends.megatron.kernels import linear_cross_entropy as entry

    return entry._get_bwd_config()


def install(args, vocab_split=8192):
    assert vocab_split in (8192, 16384)
    assert args.cross_entropy_fusion_impl == "linear"
    assert get_bwd_config()._vocab_per_split == vocab_split
