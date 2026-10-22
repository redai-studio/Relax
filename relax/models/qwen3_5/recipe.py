# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Install the Qwen TP4 optimization recipe."""

from pathlib import Path
from typing import Any

from relax.utils.data import wholepack_policy


ENTRY = "relax.models.qwen3_5.recipe.before_step"


def enabled(args: Any) -> bool:
    return wholepack_policy.enabled(args)


def _check_args(args: Any) -> None:
    assert enabled(args)
    wholepack_policy.config_contract(args)
    assert args.tensor_model_parallel_size == 4
    assert args.hidden_size == 2048 and args.num_experts == 256
    assert args.bf16 and args.use_distributed_optimizer and args.sequence_parallel
    assert not getattr(args, "use_megatron_fsdp", False)
    assert getattr(args, "use_precision_aware_optimizer", False)
    assert getattr(args, "optimizer_cpu_offload", False)
    assert args.optimizer_offload_fraction == 0.0


_INITIALIZED = False
_ACTIVE = False
_TEXT_ONLY_CPT = False


def active() -> bool:
    return _ACTIVE


def before_step(args: Any, rollout_id: int, step_id: int, model: list, optimizer: Any, scheduler: Any) -> None:
    """Install the final recipe once; refresh only per-step CPU checkpoint
    metadata."""
    global _INITIALIZED, _ACTIVE, _TEXT_ONLY_CPT
    from relax.models.qwen3_5 import recompute as recompute

    if not _INITIALIZED:
        from relax.models.qwen3_5.dependency_checks import verify_megatron_sources

        verify_megatron_sources()
    recompute.before_step(args, rollout_id, step_id, model, optimizer, scheduler)
    if _INITIALIZED:
        return
    import transformer_engine_torch as tex
    from megatron.core import mpu, tensor_parallel
    from megatron.core.transformer.attention import SelfAttention

    import relax.models.qwen3_5.expert_recompute as expert_recompute
    import relax.models.qwen3_5.gdn_preprocessing as gdn_preprocessing
    import relax.models.qwen3_5.gdn_recompute as gdn_recompute
    import relax.models.qwen3_5.grouped_mlp as grouped_mlp
    import relax.models.qwen3_5.kv_parallel as kv_parallel
    import relax.models.qwen3_5.linear_cross_entropy as linear_cross_entropy
    import relax.models.qwen3_5.projection_overlap as projection_overlap
    import relax.models.qwen3_5.qkv_projection as qkv_projection
    import relax.models.qwen3_5.rotary_embedding as rotary_embedding
    from relax.models.qwen3_5 import attention as attention
    from relax.models.qwen3_5 import attention_gate as attention_gate
    from relax.models.qwen3_5 import attention_value as attention_value
    from relax.models.qwen3_5 import shared_expert as shared_expert
    from relax.models.qwen3_5.gdn_backend import install_gdn
    from relax.models.qwen3_5.grad_norm import install_norm_guard
    from relax.models.qwen3_5.loss_config import VOCAB_PER_SPLIT
    from relax.models.qwen3_5.mtp_loss import install_ce_launches
    from relax.models.qwen3_5.ops.shared_gate import fused_gate
    from relax.utils.env import Envs

    _check_args(args)
    _TEXT_ONLY_CPT = getattr(args, "sft_training_mode", None) == "cpt"
    assert args.max_tokens_per_gpu == 262144
    assert mpu.get_data_parallel_world_size(with_context_parallel=False) == 16
    assert tex.get_cublasLt_version() == 130300
    assert args.qkv_format == "thd" and not getattr(args, "dynamic_context_parallel", False)
    assert not getattr(args, "overlap_moe_expert_parallel_comm", False)
    assert args.mtp_num_layers == 1 and args.cross_entropy_fusion_impl == "linear"
    assert Envs.RELAX_SFT_SELECTIVE_LOGITS
    assert not any(chunk.ddp_config.check_for_nan_in_grad for chunk in model)
    args._selective_loss_head = args._qwen_fused_head = True
    prime_expert_communicators(args, model)
    install_norm_guard(args, optimizer)
    _ACTIVE = True
    install_ce_launches()
    import importlib.util

    install_gdn(model, Path(importlib.util.find_spec("flash_qla").origin).parent.parent)
    attention.before_step(args, rollout_id, step_id, model, optimizer, scheduler)
    attention_value.install(model, SelfAttention)
    attention_gate.install(model, SelfAttention)
    shared_expert.install(model, fused_gate)
    kv_parallel.install(args, model)
    grouped_mlp.install(args, model, optimizer)
    expert_recompute.install(model, tensor_parallel.checkpoint, recompute.MASK_KEY)
    gdn_recompute.install(model, recompute.MASK_KEY)
    projection_overlap.install(args, model)
    gdn_preprocessing.install(model)
    rotary_embedding.install(args, model)
    qkv_projection.install(args, model)
    linear_cross_entropy.install(args, VOCAB_PER_SPLIT)
    _INITIALIZED = True


def prime_expert_communicators(args: Any, model: list) -> None:
    from megatron.core.transformer.moe import fused_a2a as fa
    from megatron.core.transformer.moe import token_dispatcher as td

    assert fa.HAVE_DEEP_EP_V2 and callable(fa.deepepv2_dispatch)
    managers = []
    for chunk in model:
        for module in chunk.modules():
            dispatcher = getattr(module, "token_dispatcher", None)
            manager = getattr(dispatcher, "_comm_manager", None)
            if manager is not None:
                assert not isinstance(manager, td._HybridEPManager)
                if isinstance(manager, td._DeepepV2Manager):
                    managers.append(manager)
    assert len({id(m) for m in managers}) == 41
    assert args.moe_flex_dispatcher_num_sms in (64, 80)
    assert all(
        type(m).__name__ == "ExpandedManager"
        and m.num_sms == m.config.moe_flex_dispatcher_num_sms == args.moe_flex_dispatcher_num_sms
        for m in managers
    )
    # Prime the ACTUAL EP communicators once, before DeepEP can cache a null
    # pointer. A WORLD collective does not initialize an EP subgroup comm.
    import torch
    import torch.distributed as dist

    groups = {id(m.group): m.group for m in managers}
    ordered = sorted(groups.values(), key=lambda g: tuple(dist.get_process_group_ranks(g)))
    device = torch.device("cuda", torch.cuda.current_device())
    for group in ordered:
        assert group.size() == 8
        scalar = torch.ones((), device=device, dtype=torch.float32)
        dist.all_reduce(scalar, group=group)
        torch.cuda.synchronize(device)
        backend = group._get_backend(device)
        assert backend._comm_ptr() != 0, "EP communicator is null after explicit priming"
