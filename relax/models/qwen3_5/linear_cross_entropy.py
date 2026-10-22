# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Validate the recipe's construction-time LCE backward configuration."""


def install(args, vocab_split=8192):
    from megatron.core.fusions.linear_cross_entropy.blackwell import entry

    assert vocab_split in (8192, 16384)
    assert args.cross_entropy_fusion_impl == "linear"
    assert entry._get_bwd_config()._vocab_per_split == vocab_split
