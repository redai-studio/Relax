# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Checks of a placement plan against the placement group Ray actually gave.

``placement_planner`` works on logical bundle indices and cannot know where
the bundles land. An engine is started with the GPU id of its first bundle and
takes the GPUs that follow it on that node, so its bundles must sit on one node
and have contiguous GPU ids. A ``PACK`` placement group makes that likely, not
certain: on a partly occupied cluster it may spread or skip GPUs. These checks
run once the placement group is ready and before any engine is created.

Only GenRM and teacher engines are checked. Rollout's engine groups come from
``--sglang-config`` and keep their own validation.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from relax.distributed.ray.placement_planner import PlacementError, PlacementPlan
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Default of ``bundle_nodes``: ask Ray. Distinct from ``None``, which means Ray
# was asked and does not know.
_ASK_RAY: Any = object()


def bundle_nodes_of(pg: Any) -> Optional[dict[int, str]]:
    """``{bundle index: node id}`` of a ready placement group, or ``None`` when
    Ray does not report it."""
    from ray.util import placement_group_table

    try:
        table = placement_group_table(pg)
    except Exception as exc:
        logger.warning(f"Could not read the placement group table: {exc}")
        return None
    nodes = (table or {}).get("bundles_to_node_id")
    if not nodes:
        return None
    return {int(index): node for index, node in nodes.items()}


def validate_engine_bundles(
    pg_tuple: tuple,
    start: int,
    num_gpus: int,
    *,
    label: str,
    bundle_nodes: Optional[Mapping[int, str]] = _ASK_RAY,
) -> None:
    """Check the ``num_gpus`` bundles one engine occupies on one node.

    Args:
        pg_tuple: ``(pg, reordered_bundle_indices, reordered_gpu_ids)`` as
            returned by ``create_placement_group``.
        start: Logical index of the engine's first bundle, i.e. its position
            in the two reordered lists.
        num_gpus: Number of bundles the engine takes on this node.
        label: Names the engine in the error message.
        bundle_nodes: ``{bundle index: node id}``; read from Ray when omitted.
            ``None`` means the nodes are unknown: only GPU ids are checked.

    Raises:
        PlacementError: the bundles are on more than one node, or their GPU
            ids are not contiguous.
    """
    pg, reordered_bundle_indices, reordered_gpu_ids = pg_tuple
    if bundle_nodes is _ASK_RAY:
        bundle_nodes = bundle_nodes_of(pg)

    logical = range(start, start + num_gpus)
    gpu_ids = [int(reordered_gpu_ids[index]) for index in logical]

    if bundle_nodes is None:
        logger.warning(f"{label}: node of each bundle is unknown; only checking that GPU ids are contiguous")
        layout = f"GPU ids {gpu_ids}"
    else:
        by_node: dict[str, list[int]] = {}
        for index, gpu_id in zip(logical, gpu_ids, strict=True):
            by_node.setdefault(str(bundle_nodes.get(int(reordered_bundle_indices[index]))), []).append(gpu_id)
        layout = ", ".join(f"node {node[:12]}: GPU ids {ids}" for node, ids in by_node.items())
        if len(by_node) > 1:
            raise PlacementError(
                f"Physical placement mismatch: {label} needs {num_gpus} GPU(s) on one node, but its bundles "
                f"[{start}, {start + num_gpus}) cross a node boundary ({layout}). Make the per-node GPU count "
                f"a multiple of the engine's GPU count, or free enough GPUs on one node."
            )

    if any(later - earlier != 1 for earlier, later in zip(gpu_ids, gpu_ids[1:])):
        raise PlacementError(
            f"Physical placement mismatch: {label} needs {num_gpus} GPU(s) with contiguous GPU ids, but its "
            f"bundles [{start}, {start + num_gpus}) are not contiguous ({layout}). The engine is started on its "
            f"first GPU and takes the ones that follow, so it would use GPUs it was not given."
        )


def validate_physical_placement(
    plan: PlacementPlan,
    pool: str,
    pg_tuple: tuple,
    *,
    num_gpus_per_node: Optional[int] = None,
    bundle_nodes: Optional[Mapping[int, str]] = _ASK_RAY,
) -> None:
    """Check every GenRM and teacher engine planned in ``pool`` against the
    placement group behind it.

    Args:
        plan: Placement plan of the run.
        pool: Name of the pool ``pg_tuple`` is the placement group of.
        pg_tuple: ``(pg, reordered_bundle_indices, reordered_gpu_ids)``.
        num_gpus_per_node: GPUs per node. An engine larger than that spans
            nodes and is checked node by node.
        bundle_nodes: ``{bundle index: node id}``; read from Ray when omitted.

    Raises:
        PlacementError: an engine's bundles cross a node boundary or have
            non-contiguous GPU ids. The message names the model, the engine
            and where its bundles really are.
    """
    engine_sizes = {claim: claim.gpus_per_engine for claim in plan.claims if claim.pool == pool}
    if not any(engine_sizes.values()):
        return
    if bundle_nodes is _ASK_RAY:
        bundle_nodes = bundle_nodes_of(pg_tuple[0])

    for claim, gpus_per_engine in engine_sizes.items():
        if not gpus_per_engine:
            continue
        per_node = min(gpus_per_engine, num_gpus_per_node or gpus_per_engine)
        for start in range(claim.start, claim.stop, per_node):
            engine = (start - claim.start) // gpus_per_engine
            validate_engine_bundles(
                pg_tuple,
                start,
                min(per_node, claim.stop - start),
                label=f"{claim.role}/{claim.model} engine {engine}",
                bundle_nodes=bundle_nodes,
            )
