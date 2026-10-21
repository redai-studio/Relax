# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Engine groups consume the planner's slice instead of re-deriving offsets.

``start_engines`` must refuse to run without a slice, and the per-engine GPU
offsets handed to colocated weight sync must come from that slice while
matching what the old offset arithmetic produced for every supported layout.
"""

import pytest


try:
    from relax.distributed.ray.rollout import EngineGroup, RolloutServer
    from relax.engine.inference.placement import (
        PlacementGroupView,
        PlacementOwner,
        PlacementPlanner,
        PlacementRequest,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from conftest import make_engine_group, make_mock_args


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def _planned_groups(args, layout, start=4):
    """Plan ``layout`` -- ``(worker_type, num_gpus, num_gpus_per_engine)``
    groups back to back from ``start`` -- the way ``start_servers`` does."""
    size = start + sum(num_gpus for _, num_gpus, _ in layout)
    view = PlacementGroupView(tuple(range(size)), tuple(range(size)), PlacementOwner.CONTROLLER, identity="pg")
    requests, offset = [], start
    for index, (worker_type, num_gpus, per_engine) in enumerate(layout):
        requests.append(
            PlacementRequest(
                f"rollout/m/group-{index}",
                worker_type,
                num_gpus,
                per_engine,
                args.num_gpus_per_node,
                bundle_offset=offset,
                active=worker_type != "placeholder",
            )
        )
        offset += num_gpus
    groups = []
    for (worker_type, num_gpus, per_engine), planned in zip(
        layout, PlacementPlanner().plan(requests, view), strict=True
    ):
        num_engines = num_gpus // min(per_engine, args.num_gpus_per_node)
        groups.append(
            EngineGroup(
                args=args,
                pg=None,
                all_engines=[object()] * num_engines if worker_type != "placeholder" else [],
                num_gpus_per_engine=per_engine,
                num_new_engines=0,
                worker_type=worker_type,
                gpu_offset=planned.reserved_offset,
                placement=planned,
                pg_owner=PlacementOwner.CONTROLLER,
            )
        )
    return groups


@pytest.mark.parametrize(
    ("layout", "start"),
    [
        ([("regular", 4, 1)], 4),
        ([("regular", 8, 2)], 4),
        ([("regular", 8, 4)], 4),
        # A multi-node engine starts on a node boundary.
        ([("regular", 32, 16)], 8),
        ([("prefill", 4, 2), ("decode", 8, 4)], 4),
        ([("regular", 4, 2), ("placeholder", 4, 4), ("regular", 8, 4)], 4),
    ],
)
def test_engine_group_offsets_come_from_the_slice_and_match_the_old_arithmetic(layout, start):
    args = make_mock_args(num_gpus_per_node=8)
    groups = _planned_groups(args, layout, start)
    server = RolloutServer(engine_groups=groups, router_ip="127.0.0.1", router_port=3000)

    legacy = [g.gpu_offset + j * g.num_gpus_per_engine for g in groups for j in range(len(g.engines))]

    assert server.engine_gpu_offsets == legacy
    # Each offset is the first slot of an engine in its slice.
    assert server.engine_gpu_offsets == [
        g.placement.referenced_offsets[j * g.nodes_per_engine] for g in groups for j in range(len(g.engines))
    ]


def test_engine_group_offsets_follow_the_slice_not_the_offset_field():
    args = make_mock_args(num_gpus_per_node=8)
    (group,) = _planned_groups(args, [("regular", 8, 2)])
    # A stale ``gpu_offset`` no longer decides where weight sync looks.
    group.gpu_offset = 0
    server = RolloutServer(engine_groups=[group], router_ip="127.0.0.1", router_port=3000)

    assert server.engine_gpu_offsets == [4, 6, 8, 10]


def test_engine_group_without_a_slice_starts_no_engine(monkeypatch):
    from relax.distributed.ray import rollout

    created = []
    monkeypatch.setattr(rollout.ray, "remote", lambda cls: created.append(cls))
    group = make_engine_group(engines=[None], num_gpus_per_engine=1)
    group.pg = (object(), [0], [0])

    with pytest.raises(ValueError, match="no planned placement slice"):
        group.start_engines()
    assert created == []
