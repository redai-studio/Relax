# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Contract tests for the unified placement planner.

The planner is the only place that resolves bundle/GPU slices, so these cover
what it must reject before any GPU engine starts, how a retried request
replays, how phases share GPUs, and who is allowed to remove a placement group.
"""

import pytest

from relax.engine.inference.placement import (
    PlacementConflictError,
    PlacementGroupView,
    PlacementOwner,
    PlacementPlanner,
    PlacementRequest,
)


class _FakePlacementGroupId:
    def __init__(self, value: str) -> None:
        self._value = value

    def hex(self) -> str:
        return self._value


class _FakePlacementGroup:
    """Stands in for a Ray placement group, which carries a hex identity that
    is identical in every process holding the group."""

    def __init__(self, value: str) -> None:
        self.id = _FakePlacementGroupId(value)


def _view(size: int = 8, owner: PlacementOwner = PlacementOwner.CONTROLLER, identity: object = None):
    return PlacementGroupView(
        tuple(range(size)),
        tuple(range(100, 100 + size)),
        owner,
        identity=identity if identity is not None else _FakePlacementGroup("group-a"),
    )


def _request(group_id: str = "rollout/group-0", **overrides):
    defaults = dict(
        group_id=group_id,
        worker_type="regular",
        num_gpus=4,
        num_gpus_per_engine=2,
        num_gpus_per_node=4,
    )
    defaults.update(overrides)
    return PlacementRequest(**defaults)


# ---------------------------------------------------------------------------
# Resolution.
# ---------------------------------------------------------------------------
def test_placement_planner_resolves_slots_from_the_parallel_layout():
    planner = PlacementPlanner()

    (slice_,) = planner.plan((_request(bundle_offset=4),), _view())

    assert slice_.reserved_offset == 4
    assert slice_.reserved_size == 4
    # One slot per single-node engine: TP2 over 4 GPUs is two engines.
    assert slice_.referenced_offsets == (4, 6)
    assert slice_.bundle_indices == (4, 6)
    assert slice_.gpu_ids == (104, 106)
    assert slice_.gpus_per_slot == 2
    assert slice_.owner is PlacementOwner.CONTROLLER


def test_placement_planner_resolves_one_slot_per_node_for_a_multi_node_engine():
    planner = PlacementPlanner()

    (slice_,) = planner.plan(
        (_request(num_gpus=8, num_gpus_per_engine=8, num_gpus_per_node=4, bundle_offset=0),),
        _view(),
    )

    assert slice_.gpus_per_slot == 4
    assert slice_.referenced_offsets == (0, 4)


def test_placement_planner_auto_offset_appends_within_the_same_phase_only():
    planner = PlacementPlanner()
    view = _view()

    planner.plan((_request("rollout/group-0", num_gpus=4),), view)
    # A different phase reuses the same GPUs at a different time, so it must
    # not be pushed behind the rollout region.
    (genrm,) = planner.plan((_request("genrm-0", num_gpus=2, phase="genrm"),), view)
    (second,) = planner.plan((_request("rollout/group-1", num_gpus=2),), view)

    assert genrm.reserved_offset == 0
    assert second.reserved_offset == 4


def test_placement_planner_reserves_an_inactive_region_without_slots():
    planner = PlacementPlanner()

    (reserved,) = planner.plan((_request("rollout/placeholder", active=False, bundle_offset=4),), _view())

    assert reserved.reserved_offset == 4
    assert reserved.reserved_size == 4
    assert reserved.referenced_offsets == ()
    assert reserved.active is False


def test_placement_planner_reserves_a_split_region_that_holds_no_whole_engine():
    planner = PlacementPlanner()

    # A split layout parks the static pools behind the rollout region. That
    # reservation runs no engine, so it need not contain whole replicas.
    (reserved,) = planner.plan(
        (_request("rollout/placeholder", num_gpus=3, num_gpus_per_engine=2, active=False, bundle_offset=4),),
        _view(),
    )

    assert reserved.reserved_offset == 4
    assert reserved.reserved_size == 3
    assert reserved.referenced_offsets == ()


# ---------------------------------------------------------------------------
# Validation before any engine starts.
# ---------------------------------------------------------------------------
def test_placement_planner_rejects_a_layout_larger_than_the_group():
    planner = PlacementPlanner()

    with pytest.raises(ValueError, match="Invalid placement size"):
        planner.plan((_request(num_gpus=16),), _view())


def test_placement_planner_rejects_a_slice_outside_the_bundle_range():
    planner = PlacementPlanner()

    with pytest.raises(ValueError, match="too small"):
        planner.plan((_request(num_gpus=4, bundle_offset=6),), _view())


def test_placement_planner_rejects_a_size_that_does_not_fit_the_parallel_layout():
    planner = PlacementPlanner()

    with pytest.raises(ValueError, match="parallel layout"):
        planner.plan((_request(num_gpus=3, num_gpus_per_engine=2),), _view())


def test_placement_planner_rejects_an_engine_spanning_a_node_boundary():
    planner = PlacementPlanner()

    with pytest.raises(ValueError, match="node boundary"):
        planner.plan((_request(num_gpus=2, num_gpus_per_engine=2, num_gpus_per_node=4, bundle_offset=3),), _view())


def test_placement_planner_requires_whole_nodes_for_a_multi_node_engine():
    planner = PlacementPlanner()

    with pytest.raises(ValueError, match="whole nodes"):
        planner.plan((_request(num_gpus=6, num_gpus_per_engine=6, num_gpus_per_node=4),), _view())


def test_placement_planner_requires_a_multi_node_engine_to_start_on_a_node():
    planner = PlacementPlanner()

    with pytest.raises(ValueError, match="node boundary"):
        planner.plan(
            (_request(num_gpus=8, num_gpus_per_engine=8, num_gpus_per_node=4, bundle_offset=2),),
            _view(size=16),
        )


def test_placement_planner_rejects_an_inconsistent_group_view():
    planner = PlacementPlanner()
    view = PlacementGroupView((0, 1, 2), (100, 101), PlacementOwner.CONTROLLER, identity=_FakePlacementGroup("a"))

    with pytest.raises(ValueError, match="inconsistent"):
        planner.plan((_request(num_gpus=2, num_gpus_per_engine=2, num_gpus_per_node=2),), view)


# ---------------------------------------------------------------------------
# Same-phase exclusion and cross-phase sharing.
# ---------------------------------------------------------------------------
def test_placement_planner_rejects_co_resident_slices_in_one_phase():
    planner = PlacementPlanner()
    view = _view()

    planner.plan((_request("rollout/group-0", num_gpus=4, bundle_offset=0),), view)

    with pytest.raises(PlacementConflictError, match="overlap"):
        planner.plan((_request("rollout/group-1", num_gpus=4, bundle_offset=2),), view)


def test_placement_planner_lets_a_deferred_phase_reuse_the_same_slice():
    planner = PlacementPlanner()
    view = _view()

    (student,) = planner.plan((_request("rollout/group-0", num_gpus=4, bundle_offset=0),), view)
    (teacher,) = planner.plan((_request("teacher-0", num_gpus=4, bundle_offset=0, phase="teacher_score"),), view)

    assert student.gpu_ids == teacher.gpu_ids


def test_placement_planner_replays_an_identical_request():
    planner = PlacementPlanner()
    view = _view()
    request = _request(num_gpus=4, bundle_offset=0)

    first = planner.plan((request,), view)
    second = planner.plan((request,), view)

    assert first == second
    assert len(planner.allocations(view)) == 1


def test_placement_planner_rejects_a_changed_request_for_a_recorded_group():
    planner = PlacementPlanner()
    view = _view()
    planner.plan((_request("rollout/group-0", num_gpus=4, bundle_offset=0),), view)

    with pytest.raises(PlacementConflictError, match="conflict"):
        planner.plan((_request("rollout/group-0", num_gpus=2, bundle_offset=0),), view)


def test_placement_planner_keys_the_ledger_by_a_stable_group_identity():
    planner = PlacementPlanner()
    # Two distinct Python objects for one placement group, as a view that
    # crossed an RPC boundary produces.
    first = _view(identity=_FakePlacementGroup("group-a"))
    second = _view(identity=_FakePlacementGroup("group-a"))

    planner.plan((_request("rollout/group-0", num_gpus=4, bundle_offset=0),), first)

    with pytest.raises(PlacementConflictError, match="overlap"):
        planner.plan((_request("rollout/group-1", num_gpus=4, bundle_offset=2),), second)


def test_placement_planner_isolates_distinct_placement_groups():
    planner = PlacementPlanner()
    planner.plan((_request("teacher-0", num_gpus=4, bundle_offset=0),), _view(identity=_FakePlacementGroup("a")))

    (other,) = planner.plan(
        (_request("teacher-0", num_gpus=4, bundle_offset=0),), _view(identity=_FakePlacementGroup("b"))
    )

    assert other.reserved_offset == 0
    assert len(planner.allocations()) == 2


def test_placement_planner_instances_do_not_share_allocations():
    view = _view()
    PlacementPlanner().plan((_request("rollout/group-0", num_gpus=4, bundle_offset=0),), view)

    # A second control domain must not discover the first one's reservation
    # through class state.
    assert PlacementPlanner().allocations(view) == ()


def test_placement_planner_dry_run_validates_without_reserving():
    planner = PlacementPlanner()
    view = _view()

    planned = planner.plan((_request("rollout/group-0", num_gpus=4, bundle_offset=0),), view, dry_run=True)

    assert planned[0].reserved_offset == 0
    assert planner.allocations(view) == ()


def test_placement_planner_dry_run_still_rejects_an_overlap():
    planner = PlacementPlanner()
    view = _view()
    planner.plan((_request("rollout/group-0", num_gpus=4, bundle_offset=0),), view)

    with pytest.raises(PlacementConflictError, match="overlap"):
        planner.plan((_request("rollout/group-1", num_gpus=4, bundle_offset=2),), view, dry_run=True)


# ---------------------------------------------------------------------------
# Release and ownership.
# ---------------------------------------------------------------------------
def test_placement_planner_release_authorizes_removing_an_owned_group():
    planner = PlacementPlanner()
    view = _view(owner=PlacementOwner.MANAGER)
    (slice_,) = planner.plan((_request("scale-out/replica-0", num_gpus=2, num_gpus_per_engine=2),), view)

    release = planner.release(slice_)

    assert release.slices == (slice_,)
    assert release.remove_placement_group is True
    assert planner.allocations(view) == ()


def test_placement_planner_release_keeps_a_borrowed_group():
    planner = PlacementPlanner()
    for owner in (PlacementOwner.CONTROLLER, PlacementOwner.EXTERNAL):
        view = _view(owner=owner, identity=_FakePlacementGroup(f"group-{owner.value}"))
        (slice_,) = planner.plan((_request("genrm-0", num_gpus=2, num_gpus_per_engine=2),), view)

        release = planner.release(slice_)

        assert release.slices == (slice_,)
        assert release.remove_placement_group is False


def test_placement_planner_release_waits_for_the_last_slice_of_an_owned_group():
    planner = PlacementPlanner()
    view = _view(owner=PlacementOwner.MANAGER)
    first, second = planner.plan(
        (
            _request("teacher-0", num_gpus=2, num_gpus_per_engine=2, bundle_offset=0),
            _request("teacher-1", num_gpus=2, num_gpus_per_engine=2, bundle_offset=2),
        ),
        view,
    )

    assert planner.release(first).remove_placement_group is False
    assert planner.release(second).remove_placement_group is True


def test_placement_planner_release_is_idempotent_and_authorizes_removal_once():
    planner = PlacementPlanner()
    view = _view(owner=PlacementOwner.MANAGER)
    (slice_,) = planner.plan((_request("scale-out/replica-0", num_gpus=2, num_gpus_per_engine=2),), view)

    assert planner.release(slice_).remove_placement_group is True
    repeated = planner.release(slice_)

    assert repeated.slices == ()
    assert repeated.remove_placement_group is False


def test_placement_planner_releases_one_group_id_from_a_view():
    planner = PlacementPlanner()
    view = _view()
    planner.plan(
        (
            _request("rollout/group-0", num_gpus=2, num_gpus_per_engine=2, bundle_offset=0),
            _request("rollout/group-1", num_gpus=2, num_gpus_per_engine=2, bundle_offset=2),
        ),
        view,
    )

    released = planner.release(view, "rollout/group-0")

    assert [item.group_id for item in released.slices] == ["rollout/group-0"]
    assert [item.group_id for item in planner.allocations(view)] == ["rollout/group-1"]


def test_placement_planner_releases_a_whole_group_from_a_view():
    planner = PlacementPlanner()
    view = _view(owner=PlacementOwner.MANAGER)
    planner.plan(
        (
            _request("teacher-0", num_gpus=2, num_gpus_per_engine=2, bundle_offset=0),
            _request("teacher-1", num_gpus=2, num_gpus_per_engine=2, bundle_offset=2),
        ),
        view,
    )

    released = planner.release(view)

    assert len(released.slices) == 2
    assert released.remove_placement_group is True
    assert planner.allocations() == ()


def test_placement_planner_release_of_an_unknown_group_is_not_an_error():
    planner = PlacementPlanner()

    release = planner.release(_view())

    assert release.slices == ()
    assert release.remove_placement_group is False


def test_placement_planner_replans_a_released_group_id():
    planner = PlacementPlanner()
    view = _view()
    (first,) = planner.plan((_request("rollout/group-0", num_gpus=4, bundle_offset=0),), view)
    planner.release(first)

    (second,) = planner.plan((_request("rollout/group-0", num_gpus=2, bundle_offset=0),), view)

    assert second.reserved_size == 2


def test_model_placement_aggregates_slices_and_marks_deferred_models() -> None:
    from relax.engine.inference.placement import ModelPlacement

    planner = PlacementPlanner()
    view = PlacementGroupView((0, 1, 2, 3), (10, 11, 12, 13), PlacementOwner.CONTROLLER, identity="shared")
    requests = [
        PlacementRequest("teacher/t/group-0", "regular", 2, 1, 4, "teacher", 0),
        PlacementRequest("teacher/t/group-1", "regular", 2, 1, 4, "teacher", 2),
    ]
    placement = ModelPlacement.from_slices(planner.plan(requests, view), shared_phase="inference")
    assert placement.pg_owner is PlacementOwner.CONTROLLER
    assert (placement.bundle_indices, placement.gpu_ids) == ((0, 1, 2, 3), (10, 11, 12, 13))
    assert (placement.activation_group, placement.activation_phase) == (view.key, "teacher")

    split = planner.plan([PlacementRequest("genrm/j/group-0", "regular", 2, 1, 4, "inference", 0)], view)
    assert ModelPlacement.from_slices(split, shared_phase="inference").activation_group is None
    with pytest.raises(ValueError, match="one phase"):
        ModelPlacement.from_slices([*split, *planner.allocations(view)[:1]], shared_phase="inference")
