# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Overlap TP projection communication using exact-shape buffers."""

_INSTALLED = None


def install(args, model):
    global _INSTALLED
    identity = tuple(id(chunk) for chunk in model)
    if _INSTALLED is not None:
        assert identity == _INSTALLED
        return None
    import torch.distributed as dist
    import transformer_engine.pytorch as te
    from megatron.core import mpu
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.transformer.attention import SelfAttention
    from transformer_engine.pytorch.module import base

    from relax.models.qwen3_5.projection_hooks import FIELDS, enable_projection
    from relax.models.qwen3_5.userbuffers import initialize

    assert len(model) == 1
    assert (
        args.tensor_model_parallel_size,
        args.context_parallel_size,
        args.pipeline_model_parallel_size,
        args.expert_model_parallel_size,
        args.expert_tensor_parallel_size,
    ) == (4, 1, 1, 8, 1)
    assert args.max_tokens_per_gpu == 262144 and args.sequence_parallel
    assert args.tp_comm_overlap is False
    assert not args.fp8 and not args.fp4
    root = model[0]
    while hasattr(root, "module"):
        root = root.module
    root = getattr(root, "language_model", root)
    assert len(root.decoder.layers) == 40
    main_ids = {id(layer.self_attention) for layer in root.decoder.layers}
    targets, counts = {}, {"GDN": 0, "FA": 0}
    names = {id(module): name for name, module in root.named_modules()}
    for name, module in root.named_modules():
        if isinstance(module, GatedDeltaNet):
            assert id(module) in main_ids
            counts["GDN"] += 1
            pairs = [("fc1", module.in_proj), ("fc2", module.out_proj)]
        elif isinstance(module, SelfAttention):
            counts["FA"] += 1
            assert id(module) in main_ids or name.startswith("mtp.")
            pairs = [("qkv", module.linear_qkv), ("proj", module.linear_proj)]
        else:
            continue
        for ub_name, projection in pairs:
            assert id(projection) not in targets
            assert isinstance(projection, (te.Linear, te.LayerNormLinear))
            targets[id(projection)] = (names[id(projection)], ub_name, projection)
    assert counts == {"GDN": 30, "FA": 11} and len(targets) == 82
    all_te = {id(module): module for module in root.modules() if isinstance(module, (te.Linear, te.LayerNormLinear))}
    before = {key: (tuple(getattr(module, f) for f in FIELDS), module.ub_name) for key, module in all_te.items()}
    assert all(not any(value[0]) for value in before.values())
    tp = mpu.get_tensor_model_parallel_group()
    rank = dist.get_rank(dist.group.WORLD)
    ranks = dist.get_process_group_ranks(tp)
    # TE splits WORLD into contiguous TP domains; never infer this without checking.
    assert dist.get_world_size(dist.group.WORLD) == 64
    assert ranks == list(range(rank // 4 * 4, rank // 4 * 4 + 4))
    assert not base.is_ub_initialized()
    init_receipt = initialize()
    assert init_receipt["tp_ranks"] == ranks
    receipts = []
    for path, name, module in targets.values():
        flags = enable_projection(module, name, tp)
        # The static call helper owns exact-shape/fallback selection.
        assert hasattr(module, "_qwen_exact_ub_state")
        receipts.append(
            dict(
                path=path,
                name=name,
                flags=flags,
                weight_shape=list(module.weight.shape),
                delayed_wgrad=module.wgrad_store.delay_wgrad_compute(),
            )
        )
    for key, module in all_te.items():
        if key not in targets:
            assert (tuple(getattr(module, f) for f in FIELDS), module.ub_name) == before[key]
    assert all(module.config.tp_comm_overlap is False for _, _, module in targets.values())
    _INSTALLED = identity
    return dict(
        targets=receipts,
        unchanged_other_TE=len(all_te) - len(targets),
        tp_ranks=ranks,
        capacity=262144,
        multishape=init_receipt,
        bulk=False,
        shared_config_overlap=False,
        nonmatching_rows="exact buffers253952/258048/262144; other rows native NCCL; no padding",
    )
