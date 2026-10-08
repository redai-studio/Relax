# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``EnginePool``: the one implementation behind rollout engine groups and the
GenRM / teacher managers. Exercised without a Ray cluster -- fake engine
handles stand in for actors and ``ray.get`` / ``ray.kill`` are patched."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from relax.distributed.ray import engine_pool
from relax.distributed.ray.engine_pool import EnginePool, EnginePoolSpec
from relax.engine.inference.discovery import EngineState


READY, DRAINING, SLEEPING = EngineState.READY, EngineState.DRAINING, EngineState.SLEEPING
ONLOADING, STARTING, DEAD = EngineState.ONLOADING, EngineState.STARTING, EngineState.DEAD


class _Engine:
    """A fake engine actor handle.

    ``method.remote()`` returns a call token the patched ``ray.get`` resolves;
    ``fail`` maps a method to the error its call raises.
    """

    def __init__(self, name: str):
        self.name = name
        self.calls: list[tuple[str, dict]] = []
        self.fail: dict[str, Exception] = {}

    def __getattr__(self, method: str):
        return SimpleNamespace(remote=lambda **kwargs: (self, method, kwargs))

    def called(self) -> list[str]:
        return [method for method, _ in self.calls]


class _Cluster:
    """Stands in for Ray: creates engines, resolves calls, records kills and
    removed placement groups."""

    def __init__(self, monkeypatch):
        self.created: list[_Engine] = []
        self.options: list[dict] = []
        self.ctor: list[tuple[tuple, dict]] = []
        self.killed: list[_Engine] = []
        self.removed_pgs: list[str] = []
        self.get_timeouts: list[float | None] = []
        # Engine names whose creation raises / whose init fails.
        self.fail_creation: set[str] = set()
        self.fail_init: set[str] = set()
        # Called as on_call(engine, method) while a call is being resolved.
        self.on_call = lambda engine, method: None

        monkeypatch.setattr(engine_pool.ray, "get", self.get)
        monkeypatch.setattr(engine_pool.ray, "kill", self.killed.append)
        # ``ray.util.placement_group`` as an attribute is the function of that name; go by module.
        monkeypatch.setattr(
            importlib.import_module("ray.util.placement_group"), "remove_placement_group", self.removed_pgs.append
        )

    def actor_class(self):
        cluster = self

        class _Actor:
            @staticmethod
            def options(**options):
                def remote(*args, **kwargs):
                    name = f"engine-{kwargs['slot']}"
                    if name in cluster.fail_creation:
                        raise RuntimeError(f"cannot schedule {name}")
                    engine = _Engine(name)
                    if name in cluster.fail_init:
                        engine.fail["init"] = RuntimeError(f"{name} failed to load")
                    cluster.created.append(engine)
                    cluster.options.append(options)
                    cluster.ctor.append((args, kwargs))
                    return engine

                return SimpleNamespace(remote=remote)

        return _Actor

    def get(self, ref, timeout=None):
        if isinstance(ref, list):
            self.get_timeouts.append(timeout)
            return [self.get(item) for item in ref]
        engine, method, kwargs = ref
        self.on_call(engine, method)
        engine.calls.append((method, kwargs))
        if method in engine.fail:
            raise engine.fail[method]
        return True


def _make_pool(cluster: _Cluster, num_slots: int, *, owns_pg: bool = False, nodes_per_engine: int = 1, **spec):
    def resolve_placement(slot):
        name = f"pg-{slot}" if owns_pg else "shared-pg"
        return (name, [100 + i for i in range(num_slots)], [10 + i for i in range(num_slots)]), owns_pg, slot

    fields = dict(
        actor_class=cluster.actor_class,
        resolve_placement=resolve_placement,
        actor_options=lambda slot, pg, bundle_index: {"pg": pg, "bundle_index": bundle_index},
        ctor=lambda slot, base_gpu_id: (("args",), {"slot": slot, "base_gpu_id": base_gpu_id}),
        init_kwargs=lambda slot, address: {**address, "slot": slot},
        allocate_addresses=lambda engines: {slot: {"host": "192.0.2.1", "port": 16000 + slot} for slot, _ in engines},
        start_timeout_s=900.0,
    )
    fields.update(spec)
    return EnginePool(EnginePoolSpec(**fields), [None] * num_slots, nodes_per_engine=nodes_per_engine)


@pytest.fixture
def cluster(monkeypatch):
    return _Cluster(monkeypatch)


def _states(pool: EnginePool) -> list[EngineState]:
    return [pool.state(head) for head in pool.head_slots()]


# ----------------------------------------------------------------------
# Bring-up.
# ----------------------------------------------------------------------


def test_engine_pool_start_creates_initializes_and_readies_every_slot(cluster):
    pool = _make_pool(cluster, 2)

    assert pool.start() == [0, 1]

    assert pool.slots == cluster.created
    # Placement: bundle index and base GPU id are looked up at the slot's gpu_index.
    assert cluster.options == [{"pg": "shared-pg", "bundle_index": 100}, {"pg": "shared-pg", "bundle_index": 101}]
    assert cluster.ctor[1] == (("args",), {"slot": 1, "base_gpu_id": 11})
    # init() gets the allocated address through the spec, and the wait is bounded.
    assert cluster.created[1].calls == [("init", {"host": "192.0.2.1", "port": 16001, "slot": 1})]
    assert cluster.get_timeouts == [900.0]
    assert pool.addresses[1] == {"host": "192.0.2.1", "port": 16001}
    assert _states(pool) == [READY, READY]
    assert pool.incarnations == {0: 1, 1: 1}


def test_engine_pool_start_only_fills_empty_slots(cluster):
    pool = _make_pool(cluster, 2)
    pool.start([0])
    survivor = pool.slots[0]

    assert pool.start() == [1]

    assert pool.slots[0] is survivor
    assert survivor.called() == ["init"]
    assert pool.start() == []


def test_engine_pool_start_failure_rolls_back_every_new_engine(cluster):
    pool = _make_pool(cluster, 2, owns_pg=True)
    cluster.fail_init = {"engine-1"}

    with pytest.raises(RuntimeError, match="engine-1 failed to load"):
        pool.start()

    # Both new engines are terminated, not just the one that failed.
    assert cluster.killed == cluster.created and len(cluster.killed) == 2
    assert pool.slots == [None, None]
    assert sorted(cluster.removed_pgs) == ["pg-0", "pg-1"]
    assert _states(pool) == [DEAD, DEAD]


def test_engine_pool_start_failure_leaves_running_engines_alone(cluster):
    pool = _make_pool(cluster, 2, owns_pg=True)
    pool.start([0])
    survivor = pool.slots[0]
    cluster.fail_init = {"engine-1"}

    with pytest.raises(RuntimeError):
        pool.start()

    assert pool.slots == [survivor, None]
    assert pool.state(0) is READY
    assert cluster.killed == [cluster.created[1]]
    assert cluster.removed_pgs == ["pg-1"]


def test_engine_pool_start_rolls_back_when_address_allocation_fails(cluster):
    def no_ports(engines):
        raise OSError("no free port")

    pool = _make_pool(cluster, 2, owns_pg=True, allocate_addresses=no_ports)

    with pytest.raises(OSError, match="no free port"):
        pool.start()

    # The actors existed but never initialized; they must not linger in their slots.
    assert len(cluster.killed) == 2
    assert pool.slots == [None, None]
    assert sorted(cluster.removed_pgs) == ["pg-0", "pg-1"]


def test_engine_pool_start_rolls_back_when_an_actor_cannot_be_created(cluster):
    pool = _make_pool(cluster, 2, owns_pg=True)
    cluster.fail_creation = {"engine-1"}

    with pytest.raises(RuntimeError, match="cannot schedule engine-1"):
        pool.start()

    # engine-0 was created first and is rolled back; engine-1's placement
    # group was already resolved and is returned too.
    assert cluster.killed == cluster.created and len(cluster.killed) == 1
    assert pool.slots == [None, None]
    assert sorted(cluster.removed_pgs) == ["pg-0", "pg-1"]


def test_engine_pool_create_and_fire_init_do_not_wait(cluster):
    pool = _make_pool(cluster, 2)

    new_engines = pool.create()
    handles = pool.fire_init(new_engines, {0: {"port": 1}, 1: {"port": 2}})

    assert [slot for slot, _ in new_engines] == [0, 1]
    assert [(engine.name, method, kwargs) for engine, method, kwargs in handles] == [
        ("engine-0", "init", {"port": 1, "slot": 0}),
        ("engine-1", "init", {"port": 2, "slot": 1}),
    ]
    # Nothing was resolved: the caller owns waiting and failure handling.
    assert all(engine.calls == [] for engine in cluster.created)
    assert _states(pool) == [STARTING, STARTING]


# ----------------------------------------------------------------------
# Lifecycle state.
# ----------------------------------------------------------------------


def _record_states(cluster, pool):
    """Sample the pool's states every time an engine call is resolved."""
    seen: list[tuple[str, tuple]] = []
    cluster.on_call = lambda engine, method: seen.append((method, tuple(state.value for state in _states(pool))))
    return seen


def test_engine_pool_deactivate_goes_through_draining_to_sleeping(cluster):
    pool = _make_pool(cluster, 2)
    pool.start()
    seen = _record_states(cluster, pool)

    pool.deactivate()

    # Both engines are out of routing before the first one is asked to release.
    assert seen == [
        ("release_memory_occupation", ("draining", "draining")),
        ("release_memory_occupation", ("draining", "draining")),
    ]
    assert _states(pool) == [SLEEPING, SLEEPING]
    assert not pool.is_active()


def test_engine_pool_activate_goes_through_onloading_to_ready(cluster):
    pool = _make_pool(cluster, 2)
    pool.start()
    pool.deactivate()
    seen = _record_states(cluster, pool)

    pool.activate()

    assert seen == [
        ("resume_memory_occupation", ("onloading", "onloading")),
        ("resume_memory_occupation", ("onloading", "onloading")),
    ]
    assert _states(pool) == [READY, READY]
    assert pool.is_active()


@pytest.mark.parametrize("prepare", [lambda pool: None, EnginePool.drain, EnginePool.deactivate])
def test_engine_pool_engine_found_dead_becomes_dead_from_any_state(cluster, prepare):
    pool = _make_pool(cluster, 2)
    pool.start()
    prepare(pool)
    pool.slots[1].fail["health_generate"] = ConnectionError("gone")

    assert pool.call_all("health_generate") == [1]

    assert pool.state(1) is DEAD
    assert pool.state(0) is not DEAD


def test_engine_pool_drain_only_marks_ready_engines(cluster):
    pool = _make_pool(cluster, 2)
    pool.start()

    pool.drain()
    pool.drain()

    assert _states(pool) == [DRAINING, DRAINING]
    # Marking is all it does: the engine-side drain belongs to the release call.
    assert all(engine.called() == ["init"] for engine in cluster.created)


def test_engine_pool_states_only_move_along_the_allowed_edges(cluster):
    pool = _make_pool(cluster, 1)
    pool.start()
    visited: list[EngineState] = []
    original_setitem = dict.__setitem__

    class _Recording(dict):
        def __setitem__(self, key, value):
            visited.append(value)
            original_setitem(self, key, value)

    pool._states = _Recording(pool._states)

    pool.deactivate()
    pool.activate()
    pool.settle(SLEEPING)  # an owner-side alignment walks the same edges

    assert visited == [DRAINING, SLEEPING, ONLOADING, READY, SLEEPING]


# ----------------------------------------------------------------------
# Idempotency.
# ----------------------------------------------------------------------


def test_engine_pool_repeated_deactivate_makes_no_engine_call(cluster):
    pool = _make_pool(cluster, 2)
    pool.start()

    pool.deactivate()
    pool.deactivate()

    assert all(engine.called() == ["init", "release_memory_occupation"] for engine in cluster.created)


def test_engine_pool_repeated_activate_makes_no_engine_call(cluster):
    pool = _make_pool(cluster, 2)
    pool.start()

    pool.activate()
    pool.deactivate()
    pool.activate()
    pool.activate()

    assert all(
        engine.called() == ["init", "release_memory_occupation", "resume_memory_occupation"]
        for engine in cluster.created
    )


def test_engine_pool_activate_with_tags_always_reaches_the_engines(cluster):
    pool = _make_pool(cluster, 1)
    pool.start()

    pool.activate(tags=["weights"])

    assert cluster.created[0].calls[-1] == ("resume_memory_occupation", {"tags": ["weights"]})
    assert _states(pool) == [READY]


def test_engine_pool_repeated_shutdown_is_a_noop(cluster):
    pool = _make_pool(cluster, 2, owns_pg=True)
    pool.start()

    pool.shutdown()
    pool.shutdown()

    assert pool.slots == [None, None]
    assert len(cluster.killed) == 2
    assert sorted(cluster.removed_pgs) == ["pg-0", "pg-1"]
    assert all(engine.called() == ["init", "shutdown"] for engine in cluster.created)


# ----------------------------------------------------------------------
# Placement group ownership.
# ----------------------------------------------------------------------


def test_engine_pool_shutdown_never_removes_a_borrowed_placement_group(cluster):
    pool = _make_pool(cluster, 2, owns_pg=False)
    pool.start()

    pool.shutdown()

    assert len(cluster.killed) == 2
    assert cluster.removed_pgs == []


def test_engine_pool_retire_returns_the_dead_engines_own_placement_group(cluster):
    pool = _make_pool(cluster, 2, owns_pg=True)
    pool.start()

    pool.retire([1])

    assert cluster.removed_pgs == ["pg-1"]
    assert pool.slots[0] is not None


# ----------------------------------------------------------------------
# Recovery.
# ----------------------------------------------------------------------


def test_engine_pool_dead_engine_is_retired_at_deactivate_and_rebuilt_at_activate(cluster):
    pool = _make_pool(cluster, 2)
    pool.start()
    survivor, casualty = pool.slots
    casualty.fail["release_memory_occupation"] = ConnectionError("unreachable while draining")

    assert pool.deactivate() == [1]

    # The survivor finished releasing; the dead engine is gone but not yet replaced.
    assert survivor.called() == ["init", "release_memory_occupation"]
    assert casualty in cluster.killed
    assert pool.slots == [survivor, None]
    assert _states(pool) == [SLEEPING, DEAD]

    assert pool.activate() == {1}

    replacement = pool.slots[1]
    assert replacement is not casualty
    # A freshly built engine already holds GPU memory, so only the survivor is resumed.
    assert replacement.called() == ["init"]
    assert survivor.called()[-1] == "resume_memory_occupation"
    assert _states(pool) == [READY, READY]
    assert pool.incarnations == {0: 1, 1: 2}


def test_engine_pool_engine_that_died_while_asleep_is_rebuilt_during_activate(cluster):
    pool = _make_pool(cluster, 2)
    pool.start()
    pool.deactivate()
    casualty = pool.slots[1]
    casualty.fail["resume_memory_occupation"] = ConnectionError("connection refused")

    assert pool.activate() == {1}

    assert pool.slots[1] is not casualty
    assert _states(pool) == [READY, READY]


def test_engine_pool_keeps_serving_when_one_engine_cannot_be_rebuilt(cluster):
    pool = _make_pool(cluster, 2)
    pool.start()
    pool.retire([1])
    cluster.fail_init = {"engine-1"}

    assert pool.recover() == set()

    assert pool.slots[0] is not None and pool.slots[1] is None
    assert _states(pool) == [READY, DEAD]
    # Still usable: the next activation tries the rebuild again.
    cluster.fail_init = set()
    assert pool.activate() == {1}
    assert _states(pool) == [READY, READY]


def test_engine_pool_with_no_engine_left_fails_activation(cluster):
    pool = _make_pool(cluster, 2)
    pool.start()
    pool.deactivate()
    pool.retire([0, 1])
    cluster.fail_init = {"engine-0", "engine-1"}

    with pytest.raises(RuntimeError, match="All engines are dead and could not be rebuilt"):
        pool.activate()


def test_engine_pool_real_errors_are_not_mistaken_for_dead_engines(cluster):
    pool = _make_pool(cluster, 2)
    pool.start()
    pool.slots[1].fail["release_memory_occupation"] = ValueError("a real bug")

    with pytest.raises(ValueError, match="a real bug"):
        pool.deactivate()

    assert cluster.killed == []


# ----------------------------------------------------------------------
# Multi-node engines, teardown and health.
# ----------------------------------------------------------------------


def test_engine_pool_multi_node_engine_is_one_logical_engine(cluster):
    pool = _make_pool(cluster, 4, nodes_per_engine=2)
    pool.start()

    assert list(pool.head_slots()) == [0, 2]
    pool.deactivate()
    # Only head slots serve HTTP, so only they are switched.
    assert [engine.called() for engine in cluster.created] == [
        ["init", "release_memory_occupation"],
        ["init"],
        ["init", "release_memory_occupation"],
        ["init"],
    ]

    pool.retire([2])
    # Retiring a logical engine tears down its follower too.
    assert pool.slots[:2] == cluster.created[:2] and pool.slots[2:] == [None, None]
    assert _states(pool) == [SLEEPING, DEAD]


def test_engine_pool_teardown_runs_the_spec_calls_in_order_then_kills(cluster):
    pool = _make_pool(
        cluster, 2, teardown_calls=("shutdown", "unregister_dcs", "unregister_from_router"), teardown_timeout_s=10
    )
    pool.start()
    doomed, kept = pool.slots
    doomed.fail["unregister_dcs"] = RuntimeError("coordinator unreachable")

    pool.teardown({0})

    # A failing step does not stop the ones after it.
    assert doomed.called() == ["init", "shutdown", "unregister_dcs", "unregister_from_router"]
    assert cluster.killed == [doomed]
    assert pool.slots == [None, kept]


def test_engine_pool_health_probe_reports_failed_slots_and_skips_empty_ones(cluster):
    pool = _make_pool(cluster, 3)
    pool.start()
    pool.retire([1])
    pool.slots[2].fail["health_generate"] = RuntimeError("down")

    assert pool.failed_health_checks(range(3), timeout=3.0) == {2}
    assert pool.slots[0].calls[-1] == ("health_generate", {"timeout": 3.0})


# ----------------------------------------------------------------------
# Owners that switch memory without waiting (rollout).
# ----------------------------------------------------------------------


def test_engine_pool_handles_show_a_switch_in_flight_until_the_owner_settles(cluster):
    pool = _make_pool(cluster, 2)
    pool.start()

    handles = pool.release_handles()
    assert [method for _engine, method, _kwargs in handles] == ["release_memory_occupation"] * 2
    # The owner still records "onloaded": the switch is in flight, not undone.
    pool.settle(READY)
    assert _states(pool) == [DRAINING, DRAINING]
    # The owner recorded the offload as finished.
    pool.settle(SLEEPING)
    assert _states(pool) == [SLEEPING, SLEEPING]

    handles = pool.resume_handles(tags=["weights"])
    assert [kwargs for _engine, _method, kwargs in handles] == [{"tags": ["weights"]}] * 2
    pool.settle(SLEEPING)
    assert _states(pool) == [ONLOADING, ONLOADING]
    pool.settle(READY)
    assert _states(pool) == [READY, READY]


def test_engine_pool_handles_skip_dead_slots(cluster):
    pool = _make_pool(cluster, 2)
    pool.start()
    pool.retire([0])

    assert [engine.name for engine, _method, _kwargs in pool.release_handles()] == ["engine-1"]
    assert pool.state(0) is DEAD
