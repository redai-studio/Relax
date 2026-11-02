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

import time
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
        nodes = (table or {}).get("bundles_to_node_id")
        return {int(index): node for index, node in nodes.items()} if nodes else None
    except Exception as exc:
        logger.warning(f"Could not read the placement group table: {exc}")
        return None


def _bundle_indices(pg_tuple: tuple, start: int, count: int, label: str) -> list[int]:
    _, indices, gpu_ids = pg_tuple
    if start < 0 or count <= 0 or start + count > min(len(indices), len(gpu_ids)):
        raise PlacementError(
            f"Physical placement mismatch: {label} has an invalid bundle slice [{start}, {start + count})"
        )
    selected = [int(index) for index in indices[start : start + count]]
    if len(set(selected)) != count:
        raise PlacementError(f"Physical placement mismatch: {label} has duplicate bundle indices")
    return selected


def _require_nodes(pg: Any, indices: list[int], nodes: Any, label: str) -> Mapping[int, str]:
    for attempt in range(3):
        if nodes is _ASK_RAY or attempt:
            nodes = bundle_nodes_of(pg)
        if nodes is not None and all(isinstance(nodes.get(i), str) and nodes[i].strip() for i in indices):
            return nodes
        if attempt < 2:
            time.sleep(1.0)
    raise PlacementError(
        f"Cannot verify physical placement for {label}: missing node bindings for bundles {indices} "
        "after 3 attempts. Check Ray/GCS placement_group_table and retry once all bindings are available."
    )


def validate_engine_bundles(
    pg_tuple: tuple,
    start: int,
    num_gpus: int,
    *,
    label: str,
    num_gpus_per_node: Optional[int] = None,
    bundle_nodes: Optional[Mapping[int, str]] = _ASK_RAY,
) -> None:
    """Check every node segment of the ``num_gpus`` bundles one engine
    occupies.

    Args:
        pg_tuple: ``(pg, reordered_bundle_indices, reordered_gpu_ids)`` as
            returned by ``create_placement_group``.
        start: Logical index of the engine's first bundle, i.e. its position
            in the two reordered lists.
        num_gpus: Total number of bundles the engine takes.
        label: Names the engine in the error message.
        num_gpus_per_node: Segment size for a multi-node engine.
        bundle_nodes: ``{bundle index: node id}``; read from Ray when omitted.
            Missing evidence is retried, then rejected.

    Raises:
        PlacementError: the bundles are on more than one node, or their GPU
            ids are not contiguous.
    """
    pg, reordered_bundle_indices, reordered_gpu_ids = pg_tuple
    indices = _bundle_indices(pg_tuple, start, num_gpus, label)
    per_node = min(num_gpus, num_gpus_per_node if num_gpus_per_node is not None else num_gpus)
    if per_node <= 0 or num_gpus % per_node:
        raise PlacementError(f"Physical placement mismatch: {label} needs complete {per_node}-GPU node segments")
    bundle_nodes = _require_nodes(pg, indices, bundle_nodes, label)
    seen_nodes: set[str] = set()
    for segment in range(start, start + num_gpus, per_node):
        logical = range(segment, segment + per_node)
        gpu_ids = [int(reordered_gpu_ids[index]) for index in logical]
        by_node: dict[str, list[int]] = {}
        for index, gpu_id in zip(logical, gpu_ids, strict=True):
            by_node.setdefault(bundle_nodes[int(reordered_bundle_indices[index])], []).append(gpu_id)
        layout = ", ".join(f"node {node[:12]}: GPU ids {ids}" for node, ids in by_node.items())
        if len(by_node) > 1:
            raise PlacementError(
                f"Physical placement mismatch: {label} needs {per_node} GPU(s) on one node, but its bundles "
                f"[{segment}, {segment + per_node}) cross a node boundary ({layout}). Make the per-node GPU count "
                f"a multiple of the engine's GPU count, or free enough GPUs on one node."
            )

        node = next(iter(by_node))
        if node in seen_nodes:
            raise PlacementError(
                f"Physical placement mismatch: {label} needs distinct nodes for its segments ({layout})"
            )
        seen_nodes.add(node)
        if any(later - earlier != 1 for earlier, later in zip(gpu_ids, gpu_ids[1:])):
            raise PlacementError(
                f"Physical placement mismatch: {label} needs {per_node} GPU(s) with contiguous GPU ids, but its "
                f"bundles [{segment}, {segment + per_node}) are not contiguous ({layout}). The engine is started on its "
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
    indices = [
        index
        for claim, size in engine_sizes.items()
        if size
        for index in _bundle_indices(pg_tuple, claim.start, claim.stop - claim.start, f"{claim.role}/{claim.model}")
    ]
    bundle_nodes = _require_nodes(pg_tuple[0], indices, bundle_nodes, pool)

    for claim, gpus_per_engine in engine_sizes.items():
        if not gpus_per_engine:
            continue
        if (claim.stop - claim.start) % gpus_per_engine:
            raise PlacementError(f"Physical placement mismatch: {claim.role}/{claim.model} has an incomplete engine")
        for start in range(claim.start, claim.stop, gpus_per_engine):
            engine = (start - claim.start) // gpus_per_engine
            validate_engine_bundles(
                pg_tuple,
                start,
                gpus_per_engine,
                label=f"{claim.role}/{claim.model} engine {engine}",
                num_gpus_per_node=num_gpus_per_node,
                bundle_nodes=bundle_nodes,
            )
