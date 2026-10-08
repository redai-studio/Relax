# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Physical placement checks: a planned engine against the nodes and GPU ids
its bundles really got."""

from __future__ import annotations

from argparse import Namespace

import pytest

from relax.distributed.ray import placement_physical
from relax.distributed.ray.placement_physical import validate_engine_bundles, validate_physical_placement
from relax.distributed.ray.placement_planner import PlacementError, plan_placement


NODE_A, NODE_B = "aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"


def _pg(gpus: list[tuple[str, int]]):
    """A placement group tuple whose logical bundle ``i`` is ``gpus[i]``.

    Returns ``(pg_tuple, bundle_nodes)``. The real bundle indices are reversed
    on purpose, so a check that confuses the two orders fails.
    """
    count = len(gpus)
    reordered_bundle_indices = [count - 1 - i for i in range(count)]
    bundle_nodes = {reordered_bundle_indices[i]: node for i, (node, _gpu) in enumerate(gpus)}
    return ("pg", reordered_bundle_indices, [gpu for _node, gpu in gpus]), bundle_nodes


def _colocate_plan(*, genrm_gpus_per_engine: int = 2, teacher_gpus_per_engine: int | None = None, **overrides):
    values = dict(
        colocate=True,
        hybrid=False,
        rollout_num_gpus=4,
        resource={"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]},
        _genrm_instances_resolved={
            "judge": {"num_gpus": 4, "num_gpus_per_engine": genrm_gpus_per_engine},
        },
    )
    if teacher_gpus_per_engine is not None:
        values.update(
            resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
            _genrm_instances_resolved={},
            use_opd=True,
            opd_type="sglang",
            teacher_hf_checkpoint="/teacher",
            teacher_num_gpus_per_engine=teacher_gpus_per_engine,
        )
    values.update(overrides)
    return plan_placement(Namespace(**values))


def _one_node(count: int = 8) -> list[tuple[str, int]]:
    return [(NODE_A, gpu) for gpu in range(count)]


def test_placement_physical_accepts_engines_on_one_node_with_contiguous_gpus():
    pg_tuple, bundle_nodes = _pg(_one_node())

    validate_physical_placement(_colocate_plan(), "actor", pg_tuple, bundle_nodes=bundle_nodes)
    validate_physical_placement(
        _colocate_plan(teacher_gpus_per_engine=2), "actor", pg_tuple, bundle_nodes=bundle_nodes
    )


def test_placement_physical_rejects_non_contiguous_gpu_ids():
    # The teacher's second 2-GPU engine got GPU 6 and GPU 8 of the node: 7 belongs to someone else.
    gpus = [(NODE_A, gpu) for gpu in (0, 1, 2, 3, 4, 5, 6, 8)]
    pg_tuple, bundle_nodes = _pg(gpus)

    with pytest.raises(PlacementError) as excinfo:
        validate_physical_placement(
            _colocate_plan(teacher_gpus_per_engine=2), "actor", pg_tuple, bundle_nodes=bundle_nodes
        )

    message = str(excinfo.value)
    # Names the model, the engine, and where its bundles really are.
    assert "teacher/__default__ engine 1" in message
    assert "not contiguous" in message
    assert "GPU ids [6, 8]" in message


def test_placement_physical_rejects_an_engine_across_a_node_boundary():
    # Six bundles on one node, two on the next: the second GenRM engine [6, 8) is fine,
    # but with five and three the first one, [4, 6), straddles the boundary.
    gpus = [(NODE_A, gpu) for gpu in range(5)] + [(NODE_B, gpu) for gpu in range(3)]
    pg_tuple, bundle_nodes = _pg(gpus)

    with pytest.raises(PlacementError) as excinfo:
        validate_physical_placement(_colocate_plan(), "actor", pg_tuple, bundle_nodes=bundle_nodes)

    message = str(excinfo.value)
    assert "genrm/judge engine 0" in message
    assert "cross a node boundary" in message
    assert f"node {NODE_A[:12]}: GPU ids [4]" in message and f"node {NODE_B[:12]}: GPU ids [0]" in message


def test_placement_physical_does_not_check_rollout():
    """A rollout-only run has nothing to check: its engine groups keep their
    own validation, even on a layout this check would reject."""
    plan = plan_placement(
        Namespace(colocate=True, hybrid=False, rollout_num_gpus=4, resource={"actor": [1, 8], "rollout": [1, 4]})
    )
    scattered = [(NODE_A, 0), (NODE_B, 5), (NODE_A, 3), (NODE_B, 1)] + _one_node(4)
    pg_tuple, bundle_nodes = _pg(scattered)

    validate_physical_placement(plan, "actor", pg_tuple, bundle_nodes=bundle_nodes)


def test_placement_physical_only_checks_the_given_pool():
    pg_tuple, bundle_nodes = _pg([(NODE_A, 0), (NODE_B, 0)] * 4)

    # The plan's engines live in the actor pool; this is some other placement group.
    validate_physical_placement(_colocate_plan(), "genrm", pg_tuple, bundle_nodes=bundle_nodes)


def test_placement_physical_checks_a_multi_node_engine_node_by_node():
    # One 16-GPU GenRM engine on two 8-GPU nodes.
    plan = plan_placement(
        Namespace(
            colocate=False,
            hybrid=False,
            rollout_num_gpus=4,
            resource={"rollout": [1, 4], "genrm": [1, 16]},
            _genrm_instances_resolved={"judge": {"num_gpus": 16, "num_gpus_per_engine": 16}},
        )
    )
    two_nodes = [(NODE_A, gpu) for gpu in range(8)] + [(NODE_B, gpu) for gpu in range(8)]
    pg_tuple, bundle_nodes = _pg(two_nodes)
    validate_physical_placement(plan, "genrm", pg_tuple, num_gpus_per_node=8, bundle_nodes=bundle_nodes)

    # Nine bundles on the first node: its half of the engine spills over.
    uneven = [(NODE_A, gpu) for gpu in range(9)] + [(NODE_B, gpu) for gpu in range(7)]
    pg_tuple, bundle_nodes = _pg(uneven)
    with pytest.raises(PlacementError, match="genrm/judge engine 0.*cross a node boundary"):
        validate_physical_placement(plan, "genrm", pg_tuple, num_gpus_per_node=8, bundle_nodes=bundle_nodes)


def test_placement_physical_still_checks_gpu_ids_when_nodes_are_unknown():
    pg_tuple, _bundle_nodes = _pg(_one_node())
    validate_engine_bundles(pg_tuple, 0, 2, label="teacher replica 0", bundle_nodes=None)

    pg_tuple, _bundle_nodes = _pg([(NODE_A, 3), (NODE_A, 5)])
    with pytest.raises(PlacementError, match="teacher replica 0.*not contiguous"):
        validate_engine_bundles(pg_tuple, 0, 2, label="teacher replica 0", bundle_nodes=None)


def test_placement_physical_asks_ray_for_the_nodes_once_per_check(monkeypatch):
    pg_tuple, bundle_nodes = _pg(_one_node())
    asked: list = []

    def fake_bundle_nodes_of(pg):
        asked.append(pg)
        return bundle_nodes

    monkeypatch.setattr(placement_physical, "bundle_nodes_of", fake_bundle_nodes_of)

    # Two GenRM engines, one question.
    validate_physical_placement(_colocate_plan(), "actor", pg_tuple)

    assert asked == ["pg"]
