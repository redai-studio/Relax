# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Accumulate BF16 expert weight gradients directly into the MFSDP main
gradient buffer."""

from collections import Counter

import torch


def enable_grouped_fusion(owner):
    from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    assert isinstance(owner, MegatronFSDP)
    assert not owner.dist_index.use_hybrid_fsdp
    assert not getattr(owner, "_relax_expert_wgrad_installed", False)
    assert owner.data_parallel_sharding_strategy in ("optim", "optim_grads", "optim_grads_params")
    aliases = Counter(
        id(getattr(p, "orig_param", p)) for _, p in owner.module.named_parameters(remove_duplicate=False)
    )
    modules, selected = [], set()
    for name, module in owner.module.named_modules():
        if not isinstance(module, TEGroupedLinear):
            continue
        config = module.config
        for key in (
            "fp8",
            "fp4",
            "cpu_offloading",
            "moe_single_grouped_weight",
            "moe_use_grouped_tensor",
            "delay_wgrad_compute",
            "use_transformer_engine_op_fuser",
        ):
            assert not getattr(config, key, False), (name, key)
        assert not getattr(module, "delay_wgrad_compute", False)
        assert not getattr(module, "single_grouped_weight", False)
        assert not module.use_bias
        weights = [owner.raw_param[f"{name}.weight{i}"] for i in range(module.num_gemms)]
        for i, raw in enumerate(weights):
            published = getattr(module, f"weight{i}")
            assert getattr(published, "orig_param", published) is raw
        assert len({p._gbuf.is_data_distributed for p in weights}) == 1
        for p in weights:
            assert type(p) is torch.nn.Parameter and p.dtype == torch.bfloat16
            assert p.requires_grad and callable(p.get_main_grad) and aliases[id(p)] == 1
            assert not getattr(p, "allreduce", True) and not getattr(p, "_is_shared", False)
            assert p._gbuf.dtype == torch.float32
            assert id(p) not in selected
            selected.add(id(p))
        modules.append((name, module, weights))
    assert modules, "No ordinary BF16 expert grouped modules found"
    # All checks precede changing the real model. Current TEGroupedMLP calls
    # each fc1/fc2 once in a backward graph; shared/repeated experts are excluded.
    for _, module, weights in modules:
        module.fuse_wgrad_accumulation = True
        module.disable_parameter_transpose_cache = True
        for p in weights:
            p.overwrite_main_grad = p._gbuf.is_data_distributed
    native = owner.all_gather_and_wait_parameters_ready

    def wait_and_set_overwrite(params, *args, **kwargs):
        params = list(params)
        result = native(params, *args, **kwargs)
        for p in params:
            if id(p) in selected:
                p.overwrite_main_grad = p._gbuf.is_data_distributed
        return result

    owner.all_gather_and_wait_parameters_ready = wait_and_set_overwrite
    owner._relax_expert_wgrad_installed = True
    return {
        "modules": len(modules),
        "weights": len(selected),
        "sharded_modules": sum(weights[0]._gbuf.is_data_distributed for _, _, weights in modules),
    }


def initialize(args):
    import relax.backends.megatron.model as model_api
    from relax.utils.logging_utils import get_logger

    assert args.use_megatron_fsdp and args.bf16 and not args.gradient_accumulation_fusion
    native = model_api.get_bridge_distributed_model

    def build(*positional, **kwargs):
        chunks = native(*positional, **kwargs)
        for chunk in chunks:
            report = enable_grouped_fusion(chunk.module)
            get_logger(__name__).info("MFSDP expert-only gradient fusion: %s", report)
        return chunks

    model_api.get_bridge_distributed_model = build
