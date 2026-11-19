# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import socket
import time

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from relax.engine.inference.placement import ProbedPlacementGroup
from relax.utils import device as device_utils
from relax.utils.device import ray_get_device_ids
from relax.utils.env import Envs
from relax.utils.http_utils import get_host_info
from relax.utils.logging_utils import get_logger
from relax.utils.utils import get_ray_accelerator_kwargs

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

    def get_binding(self):
        # Ray records the node worker's visibility before narrowing it to this
        # actor's allocation. Engine actors disable that narrowing and inherit
        # this original list, so their local index must use the same mapping.
        from ray._private.worker import global_worker

        physical = int(ray_get_device_ids()[0])
        resource = device_utils.get_ray_accelerator_name()
        visible = global_worker.original_visible_accelerator_ids.get(resource)
        local = [int(device) for device in visible].index(physical) if visible is not None else physical
        return ray.get_runtime_context().get_node_id(), physical, local


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


def create_placement_group(
    num_gpus: int, node_group_affinity: bool = True
) -> tuple[PlacementGroup, list[int], list[int]]:
    """Create a packed GPU placement group with optional node-group
    affinity."""
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
                **accelerator_kwargs,
            ).remote()
        )
    try:
        gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
        bindings = ray.get([actor.get_binding.remote() for actor in info_actors])
    finally:
        for actor in info_actors:
            ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return ProbedPlacementGroup(
        pg,
        pg_reordered_bundle_indices,
        pg_reordered_gpu_ids,
        tuple(bindings[index][0] for index in pg_reordered_bundle_indices),
        tuple(bindings[index][2] for index in pg_reordered_bundle_indices),
    )


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
    """Deploy a CPU workload worker attached to the task engine owner."""
    from .inference_manager import create_managed_rollout

    return create_managed_rollout(args, pg, data_source, runtime_env)


def create_genrm_manager(args, pg, runtime_env=None):
    """Return the task-owned default model routing handle."""
    managers = create_genrm_managers(args, pg, runtime_env)
    if len(managers) != 1:
        raise ValueError("A single GenRM handle requires exactly one model")
    return next(iter(managers.values()))


def create_genrm_managers(args, pg, runtime_env=None):
    """Services consume role routes; only Controller creates task engine
    pools."""
    prepared = getattr(args, "_genrm_managers", None)
    if prepared is None:
        raise RuntimeError("GenRM models must be registered in the task InferenceManager before deployment")
    return prepared
