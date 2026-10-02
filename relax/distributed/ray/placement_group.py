# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import socket

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from relax.core.node_group_affinity import (
    require_control_plane_resource_on_node,
    with_control_plane_affinity,
)
from relax.utils.device import ray_get_device_ids
from relax.utils.env import Envs
from relax.utils.http_utils import get_host_info
from relax.utils.logging_utils import get_logger

from .actor_group import RayTrainGroup


logger = get_logger(__name__)


def _get_head_node_id():
    """Get the head node ID based on the head node IP.

    The head node IP is determined from environment variable SLIME_HOST_IP or
    from get_host_info(). Returns the Ray NodeID (hex string) for use with
    NodeAffinitySchedulingStrategy.
    """
    # Get the target head IP from environment or auto-detect
    head_ip = Envs.SLIME_HOST_IP
    if not head_ip:
        _, head_ip = get_host_info()

    # Find the node ID that matches the head IP
    nodes = ray.nodes()
    for node in nodes:
        if node.get("Alive", False):
            node_ip = node.get("NodeManagerAddress", "")
            if node_ip == head_ip:
                node_id = node["NodeID"]
                logger.info(f"Found head node: IP={head_ip}, NodeID={node_id}")
                return node_id

    # Fallback to current node if no match found
    logger.warning(f"Could not find node with IP {head_ip} in ray.nodes(), falling back to current node")
    return ray.get_runtime_context().get_node_id()


@ray.remote
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray_get_device_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, int(gpu_id))


def allocate_train_group(args, num_gpus, pg, runtime_env=None, role="actor"):
    return RayTrainGroup(
        args=args,
        num_gpus=num_gpus,
        pg=pg,
        num_gpus_per_actor=0.4,
        role=role,
        runtime_env=runtime_env,
    )


def create_rollout_worker(args, pg, data_source=None, runtime_env=None, inference_manager_handle=None):
    from .rollout_worker import RolloutWorker

    if inference_manager_handle is None:
        raise ValueError("RolloutWorker requires the task inference manager handle")
    # Validate the CPU host before starting any rollout engines.
    head_node_id = _get_head_node_id()
    logger.info(f"Scheduling RolloutWorker on head node: {head_node_id}")
    require_control_plane_resource_on_node(args, head_node_id)

    router = ray.get(inference_manager_handle.create_rollout_role.remote(args, pg))
    if router["router_ip"] is not None:
        args.sglang_router_ip = router["router_ip"]
        args.sglang_router_port = router["router_port"]

    rollout_worker = RolloutWorker.options(
        **with_control_plane_affinity(
            args,
            {
                "num_cpus": 1,
                "num_gpus": 0,
                "runtime_env": runtime_env,
                "scheduling_strategy": NodeAffinitySchedulingStrategy(
                    node_id=head_node_id,
                    soft=False,  # Hard constraint: must run on the specified node
                ),
            },
        )
    ).remote(args, data_source=data_source, inference_manager_handle=inference_manager_handle)

    # Resolve num_rollout. Semantics:
    #   - both set         -> min(num_rollout, num_epoch * rollout_per_epoch)
    #   - only num_epoch   -> num_epoch * rollout_per_epoch
    #   - only num_rollout -> use as-is
    # SFT pre-resolves both num_rollout and num_rollout_per_epoch from the SFT
    # dataset in controller.py before any role is launched; the injected Rollout
    # has no RL global dataset, so trust those values and skip the RL-side
    # computation (which would assert on rollout_global_dataset).
    if getattr(args, "loss_type", None) == "sft":
        num_rollout_per_epoch = getattr(args, "num_rollout_per_epoch", None)
        logger.info(
            f"RolloutWorker initialized successfully (SFT mode). "
            f"num_rollout_per_epoch={num_rollout_per_epoch} (pre-resolved by controller)."
        )
    else:
        num_rollout_per_epoch = ray.get(
            rollout_worker.get_num_rollout_per_epoch.remote(),
        )
        logger.info(f"RolloutWorker initialized successfully. num_rollout_per_epoch: {num_rollout_per_epoch}")

        args.num_rollout_per_epoch = num_rollout_per_epoch
        if args.num_epoch is not None:
            epoch_rollout = num_rollout_per_epoch * args.num_epoch
            args.num_rollout = min(args.num_rollout, epoch_rollout) if args.num_rollout is not None else epoch_rollout
        assert args.num_rollout is not None and args.num_rollout > 0, (
            f"num_rollout resolved to {args.num_rollout}; "
            f"num_rollout_per_epoch={num_rollout_per_epoch}, num_epoch={args.num_epoch}"
        )

    if args.check_weight_update_equal:
        ray.get(inference_manager_handle.rollout_operation.remote("check_weights", action="snapshot"))
        ray.get(inference_manager_handle.rollout_operation.remote("check_weights", action="reset_tensors"))

    if args.offload_rollout:
        from relax.engine.inference.types import Role

        ray.get(inference_manager_handle.deactivate.remote(Role.ROLLOUT))

    return rollout_worker, num_rollout_per_epoch


def create_genrm_role(args, pg, inference_manager_handle) -> tuple[str, ...]:
    """Start every GenRM instance declared by
    ``args._genrm_instances_resolved`` as one model of the GenRM role on the
    task's inference manager.

    Returns:
        The GenRM model IDs (route keys), in configuration order.
    """
    from relax.engine.inference.config_adapters import genrm_role_models
    from relax.engine.inference.types import Role

    model_ids = tuple(ray.get(inference_manager_handle.create_role.remote(Role.GENRM, genrm_role_models(args, pg))))
    if getattr(args, "offload_rollout", False):
        ray.get([inference_manager_handle.call.remote(Role.GENRM, key, "deactivate") for key in model_ids])
    logger.info(f"GenRM models initialized successfully: instances={list(model_ids)}")
    return model_ids
