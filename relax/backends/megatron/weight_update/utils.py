# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check Actor/SGLang sync support and coordinate errors across Actor ranks."""

import torch

from relax.backends.sglang.weight_loader import NATIVE_SYNC_FENCE_LOADER
from relax.engine.sft.runtime import is_offline_mode
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def validate_actor_weight_retention(
    args, *, role: str, with_ref: bool, with_teacher: bool, requested: bool | None = None
) -> str | None:
    """Check whether Actor GPU weights can be kept until weight sync finishes.

    After training, keep these weights unchanged until both the CPU backup
    and SGLang's reads have completed, then release their GPU storage. The
    backup and transfer may run serially or overlap.

    Return None if supported, otherwise a reason to use the normal offload
    and sync path. This check does not enable retention or change offloading.
    """
    if requested is None:
        requested = Envs.RELAX_COLOCATE_OVERLAP_WEIGHT_BACKUP
    if not requested:
        return "disabled by configuration"
    # Required lifecycle: one Actor snapshot, consumed by synchronous rollout.
    if is_offline_mode(args) or getattr(args, "fully_async", False) or getattr(args, "hybrid", False):
        return "requires the synchronous Actor-to-rollout lifecycle"
    if role != "actor" or with_ref or with_teacher or getattr(args, "keep_old_actor", False):
        return "requires a single actor snapshot"
    if getattr(args, "optimizer", "adam") != "adam":
        return "requires optimizer=adam (including its AdamW mode)"
    for name in ("colocate", "offload_train", "offload_rollout", "enable_weights_backuper"):
        if not getattr(args, name, False):
            return f"requires {name}"

    resource = getattr(args, "resource", {})
    actor_gpus = resource.get("actor", (0, 0))[1]
    rollout_gpus = resource.get("rollout", (0, 0))[1]
    engine_size = getattr(args, "rollout_num_gpus_per_engine", None)
    if not actor_gpus or not 0 < rollout_gpus <= actor_gpus:
        return "requires rollout GPUs within the Actor world"
    if type(engine_size) is not int or engine_size <= 0 or rollout_gpus % engine_size:
        return "requires whole rollout engines within the Actor world"

    # Current integration gaps; keep the original path until these are connected.
    from relax.utils.megatron_peft_utils import is_lora_adapter_mode, is_lora_enabled

    if getattr(args, "megatron_to_hf_mode", None) != "bridge":
        return "raw export uses different snapshot keys from the retained-weight getter"
    if (is_lora_enabled(args) and is_lora_adapter_mode(args)) or getattr(
        args, "colocate_expert_weight_routing", False
    ):
        return "adapter and expert-routing protocols resume generation before Actor weight release"
    if getattr(args, "rollout_external", False):
        return "external rollout is outside the managed offload and completion protocol"
    if (
        getattr(args, "nccl_ub", False)
        or getattr(args, "external_cuda_graph", False)
        or getattr(args, "cuda_graph_impl", "none") != "none"
    ):
        return "graph and userbuffer storage lifetimes are not connected to deferred release"

    # TODO: GPU-validate other dtypes, PP/CP/ETP/VPP, MTP, dynamic CP, LoRA merge,
    # Critic/remote OPD, custom hooks, update intervals and partial/chunked optimizer
    # offload. These are validation gaps, not extra requirements on native behavior.
    return None


def agree_actor_weight_retention(
    reason: str | None,
    group,
    *,
    requested: bool,
    active: bool = False,
    records: list | None = None,
    shard_p2p: bool = False,
) -> bool:
    """Agree across Actor ranks whether to keep GPU weights for sync.

    Return False if any rank rejects the configuration before retention starts.
    Raise if settings disagree or support is lost after retention was enabled.
    All ranks must call this; records may reuse an earlier all-gather result.
    """
    import torch.distributed as dist

    mode = (
        requested,
        Envs.RELAX_COLOCATE_OVERLAP_WEIGHT_BACKUP,
        shard_p2p,
    )
    if records is None:
        records = [None] * dist.get_world_size(group)
        dist.all_gather_object(records, (mode, reason, active), group=group)
    if any(record[0] != mode for record in records):
        raise RuntimeError("Actor ranks disagree on weight retention settings")
    reasons = sorted({record[1] for record in records if record[1] is not None})
    if reasons and dist.get_rank(group) == 0:
        logger.info("Actor GPU weight retention disabled: %s", "; ".join(reasons))
    if reasons and any(record[2] for record in records):
        raise RuntimeError("A receiver lost its capability after Actor GPU weight retention was enabled")
    return not reasons


def validate_sglang_weight_sync_support(
    actor_nodes, engine_nodes, counts, offsets, receivers, engine_size: int
) -> str | None:
    """Check local IPC placement and completion support for Actor retention.

    Require colocated engines within the Actor world and a completion hook that
    waits for CUDA reads. Return None if supported, else a fallback reason.
    """
    world = len(actor_nodes)
    # Required: every Actor participates in the same packing/ACK collectives,
    # and each IPC receiver reads only parameters from its own node.
    if (
        not world
        or type(engine_size) is not int
        or engine_size <= 0
        or len(engine_nodes) != len(counts)
        or len(offsets) != len(counts)
        or any(type(count) is not int or count <= 0 for count in counts)
        or not counts
        or sum(counts) > world
    ):
        return "requires IPC engines within the Actor world"
    # Current gap: the completion RPC still sends engine_size entries per engine.
    if any(count != engine_size for count in counts) or list(offsets) != list(range(0, sum(counts), engine_size)):
        return "completion fences require contiguous engines with the configured GPU count"
    if any(
        not node or any(actor != node for actor in actor_nodes[offset : offset + count])
        for node, count, offset in zip(engine_nodes, counts, offsets, strict=True)
    ):
        return "CUDA IPC requires each engine and its Actor ranks on the same node"
    if len(receivers) != len(counts):
        return "receiver capability inventory is incomplete"
    for receiver, count in zip(receivers, counts, strict=True):
        if (
            receiver.get("device") != "cuda"
            or receiver.get("tp_size") != count
            or NATIVE_SYNC_FENCE_LOADER not in (receiver.get("custom_weight_loader") or [])
        ):
            return "receiver lacks the CUDA IPC completion-fence contract"
        # Current gaps: these receiver protocols have separate load/post-process paths.
        if (
            receiver.get("quantization") is not None
            or receiver.get("pp_size", 1) != 1
            or receiver.get("enable_lora", False)
            or receiver.get("speculative_algorithm") is not None
        ):
            return "receiver-specific loading is not connected to the completion protocol"
    # TODO: GPU-validate non-BF16 weights and multiple nodes with node-local engines.
    return None


def run_with_distributed_error_check(action, group):
    """Run action locally, then share any error with every rank in group.

    Every rank must call this. If any action fails, all ranks raise; otherwise
    each returns its local result. This helper does not undo action's effects.
    """
    import torch.distributed as dist

    result, error = None, None
    try:
        result = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    if any(item is not None for item in errors):
        raise RuntimeError(f"Weight sync failed: {errors}")
    return result


def run_rank0_rpc_calls(calls, group, *, append_handles=None) -> None:
    """Submit and wait for Ray RPCs on group rank 0; all ranks check errors.

    Every rank must call this. Each callable submits one RPC and returns its
    ObjectRef. On failure, rank 0 waits for all already submitted RPCs before
    propagating the error to peers. The caller owns buffer retention and CUDA
    completion checks. append_handles can submit additional RPCs into the same
    list, so their handles are also drained if submission or completion fails.
    """
    import ray
    import torch.distributed as dist

    def invoke():
        if dist.get_rank(group) != 0:
            return
        refs = []
        try:
            for call in calls:
                refs.append(call())
            if append_handles is not None:
                append_handles(refs)
            ray.get(refs)
        except Exception:
            # A failed RPC does not imply the other receivers finished reading.
            for ref in refs:
                try:
                    ray.get(ref)
                except Exception:
                    pass
            raise

    run_with_distributed_error_check(invoke, group)


def snapshot_digests(weights: dict[str, torch.Tensor]) -> dict[str, str]:
    """Byte digests for cold qualification of source replicas."""
    import hashlib

    return {
        name: hashlib.sha256(
            tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
        for name, tensor in weights.items()
    }


def validate_native_colocate_devices(actors: list[dict], engines: list[list[dict]]) -> None:
    """Prove physical GPU reuse from actual CUDA worker incarnations."""
    from relax.distributed.checkpoint_service.backends.shard_p2p.common import validate_worker_identity

    if not actors or not engines or not engines[0]:
        raise ValueError("Colocated device coverage requires complete worker inventories")
    width = len(engines[0])
    if len(engines) * width > len(actors) or any(len(engine) != width for engine in engines):
        raise ValueError("Colocated device coverage exceeds the Actor world or has unequal engines")
    for endpoint in [*actors, *(endpoint for engine in engines for endpoint in engine)]:
        validate_worker_identity(endpoint)

    def physical(endpoint: dict) -> tuple[str, str]:
        return endpoint["boot_id"], endpoint["gpu_uuid"]

    actor_devices = [physical(endpoint) for endpoint in actors]
    receiver_devices = [physical(endpoint) for engine in engines for endpoint in engine]
    if len(set(actor_devices)) != len(actors) or len(set(receiver_devices)) != len(receiver_devices):
        raise ValueError("Colocated device coverage reuses a physical GPU")
    for index, engine in enumerate(engines):
        block = actors[index * width : (index + 1) * width]
        if len({endpoint["boot_id"] for endpoint in engine}) != 1 or {physical(endpoint) for endpoint in engine} != {
            physical(endpoint) for endpoint in block
        }:
            raise ValueError("Colocated receiver devices differ from their Actor block")
