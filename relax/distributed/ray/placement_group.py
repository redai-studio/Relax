# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import socket
import time

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from relax.core.node_group_affinity import (
    require_control_plane_resource_on_node,
    with_control_plane_affinity,
)
from relax.engine.inference.placement import PlacementBinding
from relax.engine.sft.runtime import is_offline_mode
from relax.utils import device as device_utils
from relax.utils.device import ray_get_device_ids, visible_device_index
from relax.utils.env import Envs
from relax.utils.http_utils import get_host_info
from relax.utils.logging_utils import get_logger
from relax.utils.utils import get_ray_accelerator_kwargs

from .actor_group import RayTrainGroup
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST


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

    def get_ip_gpu_and_node_id(self):
        """Where this bundle landed: node IP (for ordering), GPU id, Ray node
        id (for identity), and the local device index an engine on this node
        addresses that GPU by.

        Run with :data:`INFO_ACTOR_RUNTIME_ENV`, this actor sees the visible
        devices an engine inherits, so the index is the engine's own mapping.
        """
        gpu_id = ray_get_device_ids()[0]
        node_id = ray.get_runtime_context().get_node_id()
        return ray.util.get_node_ip_address(), gpu_id, node_id, visible_device_index(int(gpu_id))


# Engines keep the node's visible devices (Ray does not narrow them); the probe
# must see the same ones to report the indices they address.
INFO_ACTOR_RUNTIME_ENV = {"env_vars": dict.fromkeys(NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, "1")}


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


def build_placement_binding(pg, probes, sort_key=sort_key) -> PlacementBinding:
    """Order a placement group's bundles by node and GPU from per-bundle
    probes.

    ``probes[i]`` is ``InfoActor.get_ip_gpu_and_node_id()`` run on bundle
    ``i``. The IP only decides the order; node identity is the Ray node id. A
    probe without the local device index leaves ``local_gpu_ids`` unknown.
    """
    ordered = sorted(
        ((index, probe[0], probe[1]) for index, probe in enumerate(probes)),
        key=sort_key,
    )
    local_known = all(len(probe) > 3 for probe in probes)
    return PlacementBinding(
        pg,
        tuple(index for index, _, _ in ordered),
        tuple(probes[index][1] for index, _, _ in ordered),
        tuple(probes[index][2] for index, _, _ in ordered),
        tuple(int(probes[index][3]) for index, _, _ in ordered) if local_known else None,
    )


def _require_node_group_markers(node_group: str, retries: int = 3, retry_delay: float = 2.0) -> None:
    """Raise if the node-group marker resources remain unavailable."""
    required = {f"{node_group}_gpu", f"{node_group}_cpu"}
    last_available: dict = {}
    for attempt in range(retries):
        cluster_resources = ray.cluster_resources()
        last_available = cluster_resources
        missing = {r for r in required if cluster_resources.get(r, 0) <= 0}
        if not missing:
            return
        if attempt < retries - 1:
            logger.warning(
                f"[Affinity] Node-group marker resources {sorted(missing)} not yet present "
                f"(attempt {attempt + 1}/{retries}); retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)

    missing = sorted(r for r in required if last_available.get(r, 0) <= 0)
    raise RuntimeError(
        f"Node-group affinity is enabled (RELAX_INITIAL_NODE_GROUP='{node_group}'), but the "
        f"required custom marker resources {missing} are not declared anywhere in this cluster. "
        f"Baseline placement groups would hang forever waiting for them. This usually means the "
        f"cluster has no '{node_group}' worker-group declaring '{node_group}_gpu'/'{node_group}_cpu' "
        f"custom resources. Please check your cluster environment; if this cluster intentionally has "
        f"no such marker resources, disable affinity by passing --no-enable-affinity."
    )


def create_placement_group(num_gpus: int, node_group_affinity: bool = True) -> PlacementBinding:
    """Create a packed GPU placement group with optional node-group affinity,
    returned as a :class:`PlacementBinding` probed per bundle."""
    accel_resource = device_utils.get_ray_accelerator_name()
    base_bundle = {accel_resource: 1, "CPU": 1}
    node_group = os.environ.get("RELAX_INITIAL_NODE_GROUP", "").strip()
    if node_group_affinity and node_group:
        _require_node_group_markers(node_group)
        base_bundle = {**base_bundle, f"{node_group}_gpu": 1, f"{node_group}_cpu": 1}
    bundles = [dict(base_bundle) for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)
    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    accelerator_kwargs = get_ray_accelerator_kwargs(1)
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
                runtime_env=INFO_ACTOR_RUNTIME_ENV,
                **accelerator_kwargs,
            ).remote()
        )
    probes = ray.get([actor.get_ip_gpu_and_node_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    binding = build_placement_binding(pg, probes)
    for i, (actual_bundle_index, gpu_id, node_id) in enumerate(
        zip(binding.bundle_indices, binding.gpu_ids, binding.node_ids, strict=True)
    ):
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {probes[actual_bundle_index][0]} ({node_id}), gpu: {gpu_id}"
        )

    return binding


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
    # Offline training pre-resolves num_rollout and num_rollout_per_epoch from its
    # dataset in controller.py before any role is launched; the injected Rollout
    # has no RL global dataset, so trust those values and skip the RL-side
    # computation (which would assert on rollout_global_dataset).
    if is_offline_mode(args):
        num_rollout_per_epoch = getattr(args, "num_rollout_per_epoch", None)
        logger.info(
            f"RolloutWorker initialized successfully (offline mode). "
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
