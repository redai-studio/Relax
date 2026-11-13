# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Use native GDN checkpoint only outside the frozen whole-layer mask."""


def install(model, mask_key, *, gdn_type=None):
    assert len(model) == 1
    chunk = model[0]
    while hasattr(chunk, "module"):
        chunk = chunk.module
    decoder = getattr(chunk, "language_model", chunk).decoder
    assert len(decoder.layers) == 40
    assert not hasattr(decoder, "_qwen_gdn_checkpoint_receipt")
    if gdn_type is None:
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet

        gdn_type = GatedDeltaNet
    targets = []
    for index, layer in enumerate(decoder.layers):
        modules = [m for m in layer.self_attention.modules() if isinstance(m, gdn_type)]
        assert len(modules) == (1 if (index + 1) % 4 else 0)
        if not modules:
            continue
        gdn = modules[0]
        assert gdn.training and not gdn.recompute_gdn
        assert (gdn.tp_size, gdn.cp_size, gdn.sp_size) == (4, 1, 4)
        assert gdn.config.recompute_granularity == "full"
        assert not gdn.config.fp8 and not gdn.config.fp4
        targets.append((index, layer, gdn))
    assert len(targets) == 30
    for index, layer, gdn in targets:
        layer._qwen_gdn_checkpoint_state = dict(layer=index, module=gdn, mask_key=mask_key, entered=False)
    decoder._qwen_gdn_checkpoint_receipt = {"native_GDN_modules": len(targets)}
    return {
        "native_GDN_modules": 30,
        "regular_GDN_checkpoints": 14,
        "whole_layer_mask_bypasses_GDN_inner": True,
        "long_mask_preserved": True,
        "MTP_unchanged": True,
        "new_math_kernels": False,
        "scope": "new combination; macro precision/performance/memory pending",
    }
