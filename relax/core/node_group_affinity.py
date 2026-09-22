# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Node-group affinity helpers for the training entrypoint."""

import os
import time
from typing import Any, Optional

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


_CONTROL_PLANE_NODE_GROUP_ATTR = "_relax_control_plane_node_group"


def _autoscaler_enabled(config_path: str) -> bool:
    """Return whether the autoscaler config is enabled."""
    try:
        from relax.utils.autoscaler.config import AutoscalerConfig

        return bool(AutoscalerConfig.from_yaml(config_path).enabled)
    except Exception as e:
        logger.warning(
            f"Could not read autoscaler config '{config_path}' to decide baseline pinning "
            f"({e}); treating as not enabled and leaving RELAX_INITIAL_NODE_GROUP unset."
        )
        return False


def maybe_pin_baseline_to_stable(args) -> None:
    """Pin an elastic job's baseline roles to the stable worker group."""
    config_path = getattr(args, "autoscaler_config", None)
    autoscaler_enabled = bool(config_path and _autoscaler_enabled(config_path))
    if autoscaler_enabled and not os.environ.get("RELAX_INITIAL_NODE_GROUP", "").strip():
        os.environ["RELAX_INITIAL_NODE_GROUP"] = "stable"

    node_group = None
    if autoscaler_enabled and getattr(args, "enable_affinity", True):
        node_group = os.environ.get("RELAX_INITIAL_NODE_GROUP", "").strip() or "stable"
    setattr(args, _CONTROL_PLANE_NODE_GROUP_ATTR, node_group)


def _get_elastic_node_group(args: Any) -> Optional[str]:
    """Return the baseline node group for an enabled Relax autoscaler job."""
    if args is None or not getattr(args, "enable_affinity", True):
        return None

    if hasattr(args, _CONTROL_PLANE_NODE_GROUP_ATTR):
        return getattr(args, _CONTROL_PLANE_NODE_GROUP_ATTR) or None

    config_path = getattr(args, "autoscaler_config", None)
    if not config_path or not _autoscaler_enabled(config_path):
        return None

    return os.environ.get("RELAX_INITIAL_NODE_GROUP", "").strip() or "stable"


def with_control_plane_affinity(args: Any, actor_options: Optional[dict] = None) -> dict:
    """Copy actor options and pin an elastic job's control plane to its stable
    CPU group."""
    options = dict(actor_options or {})
    node_group = _get_elastic_node_group(args)
    if node_group is None:
        return options

    resources = dict(options.get("resources") or {})
    resources[f"{node_group}_cpu"] = 1
    options["resources"] = resources
    return options


def require_control_plane_resource(args: Any, retries: int = 3, retry_delay: float = 2.0) -> None:
    """Fail fast when an elastic job's stable CPU marker is not declared."""
    node_group = _get_elastic_node_group(args)
    if node_group is None:
        return

    import ray

    resource_name = f"{node_group}_cpu"
    for attempt in range(retries):
        if float(ray.cluster_resources().get(resource_name, 0)) >= 1:
            return
        if attempt < retries - 1:
            logger.warning(
                f"[Affinity] Control-plane resource '{resource_name}' is not yet present "
                f"(attempt {attempt + 1}/{retries}); retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)

    raise RuntimeError(
        f"Control-plane affinity requires Ray custom resource '{resource_name}', but the cluster does not declare it"
    )


def require_control_plane_resource_on_node(args: Any, node_id: str) -> None:
    """Fail fast when a hard node affinity conflicts with the stable CPU
    marker."""
    node_group = _get_elastic_node_group(args)
    if node_group is None:
        return

    import ray

    resource_name = f"{node_group}_cpu"
    for node in ray.nodes():
        if node.get("NodeID") != node_id:
            continue
        if not node.get("Alive", False):
            raise RuntimeError(f"Target node {node_id} is not alive for control-plane affinity")
        if float((node.get("Resources") or {}).get(resource_name, 0)) < 1:
            raise RuntimeError(
                f"Target node {node_id} does not declare required control-plane resource "
                f"'{resource_name}', which conflicts with its hard node affinity"
            )
        return

    raise RuntimeError(f"Target node {node_id} was not found for control-plane affinity")
