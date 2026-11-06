# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Cold qualification for fused offload and native colocated weight sync."""

from relax.backends.sglang.weight_loader import NATIVE_SYNC_FENCE_LOADER
from relax.engine.sft.runtime import is_offline_mode
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def fused_colocate_reason(args, *, role: str, with_ref: bool, with_teacher: bool, uses_tms: bool) -> str | None:
    """Return an unsupported capability, without changing the offload
    policy."""
    if not Envs.RELAX_COLOCATE_FUSED_WEIGHT_SYNC:
        return "disabled by configuration"
    if is_offline_mode(args):
        return "offline training does not synchronize after every train step"
    if role != "actor" or with_ref or with_teacher:
        return "requires a single actor snapshot"
    if uses_tms:
        return "TMS has no fused offload lifecycle"
    for name in ("colocate", "offload_train", "offload_rollout", "enable_weights_backuper", "bf16"):
        if not getattr(args, name, False):
            return f"requires {name}"
    for name in (
        "fully_async",
        "hybrid",
        "keep_old_actor",
        "use_critic",
        "use_opd",
        "rollout_external",
        "lora_rank",
        "dynamic_context_parallel",
        "custom_megatron_before_train_step_hook_path",
        "rollout_data_postprocess_path",
        "nccl_ub",
        "external_cuda_graph",
        "colocate_expert_weight_routing",
        "mtp_num_layers",
        "chunked_optimizer_state_offload",
    ):
        if getattr(args, name, False):
            return f"unsupported {name}"
    if getattr(args, "cuda_graph_impl", "none") != "none":
        return "training CUDA graphs require the original lifecycle"
    if getattr(args, "megatron_to_hf_mode", None) != "bridge" or getattr(args, "update_weights_interval", None) != 1:
        return "requires Bridge and per-step weight sync"
    for name in ("pipeline_model_parallel_size", "context_parallel_size", "expert_tensor_parallel_size"):
        if getattr(args, name, None) != 1:
            return f"requires {name}=1"
    if getattr(args, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        return "virtual pipeline stages are not qualified"
    for name in ("tensor_model_parallel_size", "expert_model_parallel_size", "rollout_num_gpus_per_engine"):
        value = getattr(args, name, None)
        if type(value) is not int or value <= 0:
            return f"invalid {name}"
    if getattr(args, "optimizer_cpu_offload", False):
        if getattr(args, "optimizer_offload_fraction", None) != 1.0:
            return "partial CPU optimizer offload is not qualified"
        if not getattr(args, "use_precision_aware_optimizer", False):
            return "CPU optimizer requires the qualified HybridDeviceOptimizer configuration"
    elif (
        getattr(args, "optimizer", None) != "adam"
        or not getattr(args, "use_distributed_optimizer", False)
        or getattr(args, "use_precision_aware_optimizer", False)
    ):
        return "GPU optimizer requires standard distributed Adam"
    return None


def agree_fused_colocate(reason: str | None, group) -> bool:
    """Select the same lifecycle on every Actor before any state is
    released."""
    import torch.distributed as dist

    mode = (
        Envs.RELAX_COLOCATE_FUSED_WEIGHT_SYNC,
        Envs.RELAX_COLOCATE_KEEP_PROCESS_GROUPS,
        Envs.RELAX_COLOCATE_LIGHT_CLEANUP,
    )
    records = [None] * dist.get_world_size(group)
    dist.all_gather_object(records, (mode, reason), group=group)
    if any(record[0] != mode for record in records):
        raise RuntimeError("Actor ranks disagree on fused lifecycle switches")
    reasons = sorted({record[1] for record in records if record[1] is not None})
    if reasons and dist.get_rank(group) == 0:
        logger.info("Colocate fusion fallback: %s", "; ".join(reasons))
    return not reasons


def receiver_fallback_reason(actor_nodes, engine_nodes, counts, offsets, receivers) -> str | None:
    """Qualification only; existing native routing handles rejected layouts."""
    world = len(actor_nodes)
    if (
        not world
        or not actor_nodes[0]
        or len(set(actor_nodes)) != 1
        or not counts
        or len(engine_nodes) != len(counts)
        or len(offsets) != len(counts)
        or any(type(count) is not int or count <= 0 for count in counts)
        or len(set(counts)) != 1
        or sum(counts) != world
        or list(offsets) != list(range(0, world, counts[0]))
        or any(node != actor_nodes[0] for node in engine_nodes)
    ):
        return "requires dense single-node colocated engines covering the Actor world"
    if len(receivers) != len(counts):
        return "receiver capability inventory is incomplete"
    for receiver, count in zip(receivers, counts, strict=True):
        if (
            receiver.get("device") != "cuda"
            or receiver.get("dtype") not in ("bfloat16", "torch.bfloat16")
            or receiver.get("quantization") is not None
            or receiver.get("tp_size") != count
            or receiver.get("pp_size", 1) != 1
            or receiver.get("enable_lora", False)
            or receiver.get("speculative_algorithm") is not None
            or NATIVE_SYNC_FENCE_LOADER not in (receiver.get("custom_weight_loader") or [])
        ):
            return "receiver lacks the native BF16 completion-fence contract"
    return None


def run_fused_phase(action, group):
    """Converge local failures without releasing sources or falling back."""
    import torch.distributed as dist

    result, error = None, None
    try:
        result = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    if any(item is not None for item in errors):
        raise RuntimeError(f"Fused lifecycle failed: {errors}")
    return result


def run_fused_root_calls(calls, group) -> None:
    """Drain root RPCs and agree before peers enter another collective."""
    import ray
    import torch.distributed as dist

    def invoke():
        if dist.get_rank(group) != 0:
            return
        refs = []
        try:
            for call in calls:
                refs.append(call())
            ray.get(refs)
        except Exception:
            for ref in refs:
                try:
                    ray.get(ref)
                except Exception:
                    pass
            raise

    run_fused_phase(invoke, group)
