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


def create_rollout_manager(args, pg, data_source=None, runtime_env=None):
    from .rollout import RolloutManager

    # Get the head node ID to ensure RolloutManager runs on the head node
    # This is critical because the Router binds to the SLIME_HOST_IP address,
    # and other components expect the router to be accessible at the head node's IP
    head_node_id = _get_head_node_id()
    logger.info(f"Scheduling RolloutManager on head node: {head_node_id}")
    require_control_plane_resource_on_node(args, head_node_id)

    rollout_manager = RolloutManager.options(
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
    ).remote(args, pg, data_source=data_source)

    # Add timeout protection to prevent indefinite blocking during initialization
    # The timeout is set to 120 seconds to allow sufficient time for:

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
            f"RolloutManager initialized successfully (SFT mode). "
            f"num_rollout_per_epoch={num_rollout_per_epoch} (pre-resolved by controller)."
        )
    else:
        num_rollout_per_epoch = ray.get(
            rollout_manager.get_num_rollout_per_epoch.remote(),
        )
        logger.info(f"RolloutManager initialized successfully. num_rollout_per_epoch: {num_rollout_per_epoch}")

        args.num_rollout_per_epoch = num_rollout_per_epoch
        if args.num_epoch is not None:
            epoch_rollout = num_rollout_per_epoch * args.num_epoch
            args.num_rollout = min(args.num_rollout, epoch_rollout) if args.num_rollout is not None else epoch_rollout
        assert args.num_rollout is not None and args.num_rollout > 0, (
            f"num_rollout resolved to {args.num_rollout}; "
            f"num_rollout_per_epoch={num_rollout_per_epoch}, num_epoch={args.num_epoch}"
        )

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch


def create_genrm_manager(args, pg, runtime_env=None):
    """Create and initialize a single GenRM manager (legacy single-instance
    path).

    Args:
        args: Argument namespace containing genRM configuration
        pg: Placement group for resource allocation
        runtime_env: Optional runtime environment configuration

    Returns:
        Initialized GenRM manager
    """
    from .genrm import GenRMManager

    # `name` (with no explicit namespace) lets user code inside other actors of
    # the same Ray job look this up via ray.get_actor("relax_genrm_manager").
    # Used by custom_reward_post_process_path when GenRM lifecycle is managed
    # from userland. Only safe as a fixed, well-known name because this path
    # is exclusively for the single-instance case (see create_genrm_managers).
    genrm_manager = GenRMManager.options(
        **with_control_plane_affinity(
            args,
            {
                "name": "relax_genrm_manager",
                "num_cpus": 1,
                "num_gpus": 0,
                "runtime_env": runtime_env,
            },
        )
    ).remote(args, pg)

    logger.info("GenRMManager initialized successfully")

    # Offload if requested (for colocated mode)
    if args.offload_rollout:
        logger.info("Offloading GenRM engines (colocated mode)")
        ray.get(genrm_manager.offload.remote())

    return genrm_manager


def create_genrm_managers(args, pg, runtime_env=None):
    """Create and initialize the GenRM manager(s) declared by
    ``args._genrm_instances_resolved``.

    Single-instance configs (the legacy --genrm-model-path path, normalized to
    the sentinel key "__default__") go through ``create_genrm_manager`` so the
    well-known Ray actor name "relax_genrm_manager" keeps working for userland
    lookups. Multi-instance configs (--genrm-instances) launch one manager per
    route key, sharing ``pg`` at non-overlapping GPU offsets, named
    "relax_genrm_manager_{key}" -- there is no single well-known name to
    preserve once more than one instance exists.

    Returns:
        ``{route_key: manager_handle}``.
    """
    import copy

    from .genrm import GenRMManager
    from .multi_instance_orchestrator import start_multi_instance_managers

    instance_specs = args._genrm_instances_resolved
    if list(instance_specs.keys()) == ["__default__"]:
        return {"__default__": create_genrm_manager(args, pg, runtime_env=runtime_env)}
    port_window_indices = {key: index for index, key in enumerate(instance_specs)}

    def _build_genrm_manager_args(base_args, key, spec):
        instance_args = copy.copy(base_args)
        instance_args.genrm_model_path = spec["model_path"]
        instance_args.genrm_num_gpus = spec["num_gpus"]
        instance_args.genrm_num_gpus_per_engine = spec["num_gpus_per_engine"]
        instance_args.genrm_engine_config = spec["engine_config"]
        instance_args.genrm_sampling_config = spec["sampling_config"]
        return instance_args

    def _spawn_genrm_manager(key, instance_args, bundle_offset, spec):
        return GenRMManager.options(
            **with_control_plane_affinity(
                instance_args,
                {
                    "name": f"relax_genrm_manager_{key}",
                    "num_cpus": 1,
                    "num_gpus": 0,
                    "runtime_env": runtime_env,
                },
            )
        ).remote(
            instance_args,
            pg,
            bundle_offset=bundle_offset,
            port_window_index=port_window_indices[key],
        )

    managers = start_multi_instance_managers(
        args=args,
        instance_specs=instance_specs,
        build_manager_args=_build_genrm_manager_args,
        spawn_manager=_spawn_genrm_manager,
        region_offset=0,
    )
    logger.info(f"GenRM managers initialized successfully: instances={list(managers.keys())}")
    return managers
