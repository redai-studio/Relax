# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rollout startup must not keep reserved GPU slices when it fails.

The initial layout is reserved in the task owner's ledger before any engine
exists, so a failed bring-up has to give the slices back while leaving the
Controller-owned placement group alone.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


try:
    from relax.distributed.ray import rollout as module
    from relax.engine.inference.config import EngineGroupSpec, InferenceModelSpec
    from relax.engine.inference.placement import PlacementGroupView, PlacementOwner, PlacementPlanner
    from relax.engine.inference.types import Role, WeightSource

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def _args():
    return SimpleNamespace(
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=4,
        sglang_config=None,
        prefill_num_servers=None,
        rollout_engine_init_timeout=1.0,
        sglang_hf_checkpoint=None,
        hf_checkpoint="/tmp/checkpoint",
        sglang_router_ip=None,
        sglang_router_port=None,
        debug_rollout_only=False,
    )


# The real launcher, before the fixture replaces it.
_REAL_START_ROUTER = module._start_router if HAS_DEPS else None


class _FakeRouterProcess:
    def __init__(self) -> None:
        self.alive = True
        self.pid = 1000

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.alive = False

    def join(self, timeout=None) -> None:
        pass

    def kill(self) -> None:
        self.alive = False


@pytest.fixture
def startup(monkeypatch):
    state = SimpleNamespace(groups=[], fail=False, configure=None, router_args=[], routers=[])
    monkeypatch.setattr(module, "_LAUNCHED_ROUTER_PROCESSES", [])

    def _start_router(router_args, *, force_new=False, launched=None, **kwargs):
        state.router_args.append(router_args)
        if force_new or router_args.sglang_router_ip is None:
            process = _FakeRouterProcess()
            module._LAUNCHED_ROUTER_PROCESSES.append(process)
            if launched is not None:
                launched.append(process)
            state.routers.append(process)
        return "10.0.0.1", 3000 + len(state.router_args)

    monkeypatch.setattr(module, "_start_router", _start_router)
    monkeypatch.setattr(module, "_wait_engine_init_with_progress", lambda *a, **k: None)

    def _engine_group(**kwargs):
        group = MagicMock()
        group.placement = kwargs["placement"]
        group.pg_owner = kwargs["pg_owner"]
        group.kwargs = kwargs
        if state.fail:
            group.start_engines.side_effect = RuntimeError("engine bring-up failed")
        else:
            group.start_engines.return_value = ([], {})
        if state.configure is not None:
            state.configure(group)
        state.groups.append(group)
        return group

    monkeypatch.setattr(module, "EngineGroup", _engine_group)
    return state


def _pg(num_gpus=4):
    return (MagicMock(), list(range(num_gpus)), list(range(num_gpus)))


def _view(pg):
    return PlacementGroupView(tuple(pg[1]), tuple(pg[2]), PlacementOwner.CONTROLLER, identity=pg[0])


def test_rollout_startup_records_its_layout_in_the_supplied_ledger(startup):
    ledger = PlacementPlanner()
    pg = _pg()

    module.start_rollout_servers(
        _args(),
        pg,
        planner=ledger,
    )

    (recorded,) = ledger.allocations(_view(pg))
    assert recorded.group_id == "rollout/default/group-0"
    assert recorded.reserved_offset == 0
    assert recorded.referenced_offsets == (0, 2)
    assert startup.groups[0].placement == recorded


def test_rollout_startup_failure_releases_the_reserved_slices(startup):
    startup.fail = True
    ledger = PlacementPlanner()
    pg = _pg()

    with pytest.raises(RuntimeError, match="engine bring-up failed"):
        module.start_rollout_servers(
            _args(),
            pg,
            planner=ledger,
        )

    assert ledger.allocations(_view(pg)) == ()


def test_rollout_startup_failure_keeps_the_controller_owned_group(startup):
    startup.fail = True
    ledger = PlacementPlanner()
    pg = _pg()

    with pytest.raises(RuntimeError):
        module.start_rollout_servers(
            _args(),
            pg,
            planner=ledger,
        )

    # Releasing a borrowed group never authorizes destroying it.
    assert ledger.release(_view(pg)).remove_placement_group is False


def _fail_after_first_actor(startup, events, *, shutdown_error=None):
    """Make the first group create one actor and then fail to start."""

    def configure(group):
        group.all_engines = group.kwargs["all_engines"]

        def start_engines(*args):
            group.all_engines[0] = "engine-0"
            raise RuntimeError("engine bring-up failed")

        def shutdown_engines(indices, *, strict=False):
            events.append(("shutdown", sorted(indices), strict))
            if shutdown_error is not None:
                raise shutdown_error

        group.start_engines.side_effect = start_engines
        group.shutdown_engines.side_effect = shutdown_engines

    startup.configure = configure


def test_rollout_startup_failure_stops_partially_created_actors_before_releasing(startup):
    events = []
    _fail_after_first_actor(startup, events)
    ledger = PlacementPlanner()
    release = ledger.release
    ledger.release = lambda *a, **k: events.append(("release",)) or release(*a, **k)
    pg = _pg()

    with pytest.raises(RuntimeError, match="engine bring-up failed"):
        module.start_rollout_servers(_args(), pg, planner=ledger)

    # The group was never registered with a server, yet its actor is stopped
    # strictly, and only then is its slice given back.
    assert events == [("shutdown", [0, 1], True), ("release",)]
    assert ledger.allocations(_view(pg)) == ()


def test_rollout_startup_failure_keeps_the_slice_when_actors_may_hold_gpus(startup):
    events = []
    _fail_after_first_actor(startup, events, shutdown_error=RuntimeError("Engines [0] may still hold GPU memory"))
    ledger = PlacementPlanner()
    pg = _pg()

    with pytest.raises(RuntimeError, match="engine bring-up failed"):
        module.start_rollout_servers(_args(), pg, planner=ledger)

    (held,) = ledger.allocations(_view(pg))
    assert held.group_id == "rollout/default/group-0"


def _start_teacher(args, pg, ledger, *, bundle_offset, phase):
    teacher = InferenceModelSpec(
        "default",
        "/tmp/teacher",
        engine_groups=[EngineGroupSpec("regular", 4, 2, {})],
        weight_source=WeightSource.STATIC,
    ).resolved(args)
    return module.start_servers(
        args, [teacher], planner=ledger, role=Role.TEACHER, pg=pg, bundle_offset=bundle_offset, phase=phase
    )


def test_rollout_startup_split_teacher_keeps_rollout_at_the_front(startup):
    args = _args()
    ledger = PlacementPlanner()
    pg = _pg(8)

    # A split teacher sits behind the rollout region and is recorded first.
    _start_teacher(args, pg, ledger, bundle_offset=4, phase=module.PHASE_GENERATE)
    module.start_rollout_servers(args, pg, planner=ledger)

    offsets = {item.group_id: item.reserved_offset for item in ledger.allocations(_view(pg))}
    assert offsets == {"teacher/default/group-0": 4, "rollout/default/group-0": 0}


def test_rollout_startup_shared_teacher_with_same_model_name_does_not_conflict(startup):
    args = _args()
    ledger = PlacementPlanner()
    pg = _pg(4)

    _start_teacher(args, pg, ledger, bundle_offset=0, phase="teacher")
    module.start_rollout_servers(args, pg, planner=ledger)

    assert {item.group_id for item in ledger.allocations(_view(pg))} == {
        "teacher/default/group-0",
        "rollout/default/group-0",
    }


def test_rollout_startup_teacher_router_pins_groups_to_one_replica(startup):
    args = _args()
    args.sglang_router_policy = "cache_aware"
    ledger = PlacementPlanner()
    pg = _pg(8)

    _start_teacher(args, pg, ledger, bundle_offset=4, phase=module.PHASE_GENERATE)
    module.start_rollout_servers(args, pg, planner=ledger)

    teacher_router, rollout_router = startup.router_args
    assert teacher_router.sglang_router_policy == module.TEACHER_ROUTER_POLICY
    assert teacher_router.router_assignment_mode == "min_group"
    # Generation keeps the configured policy.
    assert rollout_router.sglang_router_policy == "cache_aware"


def test_rollout_startup_teacher_router_args_parse_into_the_router_config(monkeypatch):
    import argparse

    from sglang_router.launch_router import RouterArgs

    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
    args = parser.parse_args([])
    args.sglang_router_ip = None
    args.sglang_router_port = None
    args.use_slime_router = False
    args.sglang_router_policy = "cache_aware"
    args.sglang_router_request_timeout_secs = 60
    # What start_servers sets on the teacher's copy of the args.
    args.sglang_router_policy = module.TEACHER_ROUTER_POLICY
    args.router_assignment_mode = "min_group"

    launched = []

    class _Process:
        def __init__(self, target, args):
            launched.append(args[0])

        def start(self):
            pass

        def is_alive(self):
            return True

    monkeypatch.setattr(module, "_LAUNCHED_ROUTER_PROCESSES", [])
    monkeypatch.setattr(module.multiprocessing, "Process", _Process)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(module, "get_host_info", lambda: ("host", "10.0.0.1"))
    monkeypatch.setattr(module, "find_available_port", lambda port: port)

    _REAL_START_ROUTER(args, force_new=True)

    (router_config,) = launched
    assert (router_config.policy, router_config.assignment_mode) == ("manual", "min_group")


def test_rollout_startup_router_that_dies_at_startup_is_recorded_for_cleanup(monkeypatch):
    class _Process:
        exitcode = 1

        def __init__(self, target, args):
            pass

        def start(self):
            pass

        def is_alive(self):
            return False

    args = SimpleNamespace(
        sglang_router_ip=None, sglang_router_port=None, use_slime_router=True, sglang_router_request_timeout_secs=60
    )
    monkeypatch.setattr(module, "_LAUNCHED_ROUTER_PROCESSES", [])
    monkeypatch.setattr(module.multiprocessing, "Process", _Process)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(module, "get_host_info", lambda: ("host", "10.0.0.1"))
    monkeypatch.setattr(module, "find_available_port", lambda port: port)
    launched = []

    with pytest.raises(RuntimeError, match="exited during startup"):
        _REAL_START_ROUTER(args, force_new=True, launched=launched)

    # The caller can still stop exactly the process this call started.
    assert len(launched) == 1 and module._LAUNCHED_ROUTER_PROCESSES == launched


def test_rollout_startup_failure_stops_only_the_routers_it_launched(startup):
    startup.fail = True
    running = _FakeRouterProcess()
    module._LAUNCHED_ROUTER_PROCESSES.append(running)
    args = _args()

    with pytest.raises(RuntimeError, match="engine bring-up failed"):
        _start_teacher(args, _pg(), PlacementPlanner(), bundle_offset=0, phase="teacher")

    (teacher_router,) = startup.routers
    assert not teacher_router.alive
    # Another model's Router keeps serving.
    assert running.alive and module._LAUNCHED_ROUTER_PROCESSES == [running]


def test_rollout_startup_failure_forgets_the_endpoint_of_its_stopped_router(startup):
    startup.fail = True
    args = _args()

    with pytest.raises(RuntimeError):
        module.start_rollout_servers(args, _pg(), planner=PlacementPlanner())

    assert not startup.routers[0].alive
    # A retry must launch a new Router rather than reuse the stopped one.
    assert (args.sglang_router_ip, args.sglang_router_port) == (None, None)


def test_rollout_startup_debug_rollout_only_registers_engines_at_start(startup):
    args = _args()
    args.debug_rollout_only = True

    (server,) = module.start_rollout_servers(args, _pg(), planner=PlacementPlanner()).values()

    # No weight sync ever runs, so the engines must not wait for one.
    assert startup.groups[0].kwargs["skip_router_registration"] is False
    assert server.model_spec.needs_weight_update is False


def test_rollout_startup_planning_failure_removes_the_group_it_created(startup, monkeypatch):
    import importlib

    import relax.core.service as service

    # ``ray.util.placement_group`` the attribute is the function, not the module.
    ray_pg = importlib.import_module("ray.util.placement_group")

    pg = _pg()
    removed = []
    monkeypatch.setattr(service, "create_placement_group", lambda **kwargs: pg)
    monkeypatch.setattr(ray_pg, "remove_placement_group", removed.append)
    ledger = MagicMock()
    ledger.plan.side_effect = ValueError("bad layout")

    with pytest.raises(ValueError, match="bad layout"):
        _start_teacher(_args(), None, ledger, bundle_offset=0, phase="teacher")

    assert removed == [pg[0]] and startup.groups == []


def test_manager_create_role_checks_every_model_before_starting_any(monkeypatch):
    from relax.distributed.ray.inference_manager import InferenceManager

    started = []
    monkeypatch.setattr(module, "start_servers", lambda *a, **k: started.append(a) or {})
    args = _args()
    pg = _pg()

    def teacher(name, offset):
        config = InferenceModelSpec(
            name,
            "/tmp/teacher",
            engine_groups=[EngineGroupSpec("regular", 2, 2, {})],
            weight_source=WeightSource.STATIC,
        ).resolved(args)
        return config, args, {"pg": pg, "bundle_offset": offset, "phase": "teacher", "base_port": 26000}

    manager = InferenceManager()
    # The second teacher overlaps the first one in the same phase.
    with pytest.raises(ValueError, match="overlap"):
        manager.create_role(Role.TEACHER, [teacher("a", 0), teacher("b", 1)])

    assert started == [] and manager.allocations() == ()


def test_manager_create_role_keeps_a_model_that_failed_to_stop_and_retries_it(monkeypatch):
    from relax.distributed.ray.inference_manager import InferenceManager
    from relax.engine.inference.types import LifecycleState

    args = _args()
    pg = _pg(6)
    stopped = []
    stuck = {"a": True}

    def teacher(name, offset):
        config = InferenceModelSpec(
            name,
            "/tmp/teacher",
            engine_groups=[EngineGroupSpec("regular", 2, 2, {})],
            weight_source=WeightSource.STATIC,
        ).resolved(args)
        return config, args, {"pg": pg, "bundle_offset": offset, "phase": "teacher", "base_port": 26000}

    models = [teacher("a", 0), teacher("b", 2), teacher("c", 4)]

    def pool(config):
        def shutdown(planner):
            stopped.append(config.name)
            if stuck.get(config.name):
                raise RuntimeError("Engines [0] may still hold GPU memory")

        return SimpleNamespace(model_spec=config, shutdown=shutdown)

    outcomes = iter([pool(models[0][0]), pool(models[1][0]), RuntimeError("engine bring-up failed")])

    def start_servers(engine_args, configs, **kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return {configs[0].name: outcome}

    monkeypatch.setattr(module, "start_servers", start_servers)
    manager = InferenceManager()

    with pytest.raises(RuntimeError, match="engine bring-up failed"):
        manager.create_role(Role.TEACHER, models)

    # Every started model was asked to stop; the one that failed stays owned,
    # DEAD and resident, so conflicting loads keep waiting for its GPUs.
    assert stopped == ["a", "b"]
    assert manager.model_ids(Role.TEACHER) == ("a",)
    assert manager.snapshot(Role.TEACHER).models[0].state == LifecycleState.DEAD
    assert manager.resident(Role.TEACHER)

    # Creating the role again retries the stop first and starts nothing while
    # the model is still stuck.
    with pytest.raises(RuntimeError, match="Failed to shut down inference role teacher"):
        manager.create_role(Role.TEACHER, models)
    assert stopped == ["a", "b", "a"]

    # Once it stops, the retry starts the role from scratch.
    stuck["a"] = False
    outcomes = iter([pool(config) for config, _, _ in models])
    monkeypatch.setattr(manager, "register", lambda role, pools, **kwargs: tuple(pools))
    assert manager.create_role(Role.TEACHER, models) == ("a", "b", "c")
    assert stopped == ["a", "b", "a", "a"]


def test_manager_shutdown_retries_a_model_left_by_a_failed_start(monkeypatch):
    from relax.distributed.ray.inference_manager import InferenceManager

    args = _args()
    config, other = (
        InferenceModelSpec(
            name,
            "/tmp/teacher",
            engine_groups=[EngineGroupSpec("regular", 2, 2, {})],
            weight_source=WeightSource.STATIC,
        ).resolved(args)
        for name in ("a", "b")
    )
    attempts = []

    def shutdown(planner):
        attempts.append(planner)
        if len(attempts) == 1:
            raise RuntimeError("Engines [0] may still hold GPU memory")

    started = iter([{"a": SimpleNamespace(model_spec=config, shutdown=shutdown)}, RuntimeError("bring-up failed")])

    def start_servers(*a, **k):
        outcome = next(started)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(module, "start_servers", start_servers)
    manager = InferenceManager()
    placement = {"pg": _pg(4), "bundle_offset": 0, "phase": "teacher", "base_port": 26000}
    with pytest.raises(RuntimeError, match="bring-up failed"):
        manager.create_role(Role.TEACHER, [(config, args, placement), (other, args, dict(placement, bundle_offset=2))])

    manager.shutdown(Role.TEACHER)

    assert len(attempts) == 2 and Role.TEACHER not in manager.registered_roles()


def test_manager_create_role_retries_the_stop_of_a_half_started_model(startup):
    """Partial start, a first stop that fails, then a retry that frees it."""
    from relax.distributed.ray.inference_manager import InferenceManager

    stops = []

    def configure(group):
        group.all_engines = group.kwargs["all_engines"]
        group.model_id = group.kwargs["model_id"]
        group.router_ip, group.router_port = group.kwargs["router_ip"], group.kwargs["router_port"]

        def start_engines(*args):
            group.all_engines[0] = "engine-0"
            raise RuntimeError("engine bring-up failed")

        def shutdown_engines(indices, *, strict=False):
            stops.append(sorted(indices))
            if len(stops) == 1:
                raise RuntimeError("Engines [0] may still hold GPU memory")
            group.all_engines[:] = [None] * len(group.all_engines)

        group.start_engines.side_effect = start_engines
        group.shutdown_engines.side_effect = shutdown_engines

    startup.configure = configure
    args = _args()
    config = InferenceModelSpec(
        "default",
        "/tmp/teacher",
        engine_groups=[EngineGroupSpec("regular", 4, 2, {})],
        weight_source=WeightSource.STATIC,
    ).resolved(args)
    manager = InferenceManager()

    with pytest.raises(RuntimeError, match="engine bring-up failed"):
        manager.create_role(Role.TEACHER, [(config, args, {"pg": _pg(), "bundle_offset": 0, "phase": "teacher"})])

    # The half-started engines are still owned, with their slice reserved.
    (held,) = manager.allocations()
    assert held.group_id == "teacher/default/group-0"
    assert manager.model_ids(Role.TEACHER) == ("default",) and manager.resident(Role.TEACHER)

    manager.shutdown(Role.TEACHER)

    assert stops == [[0, 1], [0, 1]]
    assert manager.allocations() == () and Role.TEACHER not in manager.registered_roles()


def test_manager_rollout_start_keeps_engines_it_could_not_stop(monkeypatch):
    from relax.distributed.ray.inference_manager import InferenceManager

    attempts = []
    stuck = SimpleNamespace(
        model_spec=InferenceModelSpec("default", "/tmp/policy"),
        shutdown=lambda planner: attempts.append(planner),
    )

    class _FailingPool:
        def __init__(self, args, pg, *, inference_manager, startup_leftovers):
            startup_leftovers["default"] = stuck
            raise RuntimeError("engine bring-up failed")

    monkeypatch.setattr(module, "RolloutEnginePool", _FailingPool)
    manager = InferenceManager()

    with pytest.raises(RuntimeError, match="engine bring-up failed"):
        manager.create_rollout_role(_args(), _pg())

    assert manager.model_ids(Role.ROLLOUT) == ("default",) and manager.resident(Role.ROLLOUT)
    # Starting generation again first stops what the failed start left behind.
    with pytest.raises(RuntimeError, match="engine bring-up failed"):
        manager.create_rollout_role(_args(), _pg())
    assert len(attempts) == 1


def test_rollout_server_shutdown_stops_every_group_when_one_fails() -> None:
    failing, healthy = MagicMock(), MagicMock()
    for group in (failing, healthy):
        group.all_engines = [object(), object()]
    failing.shutdown_engines.side_effect = RuntimeError("Engines [0] may still hold GPU memory")
    planner = MagicMock()
    planner.release.return_value = SimpleNamespace(remove_placement_group=False)
    server = module.RolloutServer(
        engine_groups=[failing, healthy], router_ip="10.0.0.1", router_port=3000, model_name="default"
    )

    with pytest.raises(RuntimeError, match="may still hold GPU memory"):
        server.shutdown(planner)

    healthy.shutdown_engines.assert_called_once_with({0, 1}, strict=True)
    # Only the group that stopped gives its slice back.
    planner.release.assert_called_once_with(healthy.placement)


def _task_args(*, rollout_gpus, genrm_gpus, actor_gpus=4, **overrides):
    spec = {
        "model_path": "/tmp/judge",
        "num_gpus": genrm_gpus,
        "num_gpus_per_engine": 1,
        "engine_config": None,
        "sampling_config": None,
    }
    args = _args()
    args.__dict__.update(
        rollout_num_gpus=rollout_gpus,
        rollout_num_gpus_per_engine=1,
        colocate=True,
        hybrid=False,
        fully_async=False,
        resource={"actor": [1, actor_gpus], "rollout": [1, rollout_gpus], "genrm": [1, genrm_gpus]},
        _genrm_instances_resolved={"judge": spec},
        **overrides,
    )
    return args


def test_task_layout_accepts_split_rollout_and_genrm():
    from relax.distributed.ray.inference_manager import validate_task_layout

    validate_task_layout(_task_args(rollout_gpus=2, genrm_gpus=2))


def test_task_layout_rejects_a_later_role_before_any_engine_starts(monkeypatch):
    from relax.distributed.ray.inference_manager import validate_task_layout

    monkeypatch.setattr(module, "start_servers", MagicMock(side_effect=AssertionError("started an engine")))
    # GenRM sits behind four rollout bundles of a four-GPU actor group.
    with pytest.raises(ValueError, match="Invalid genrm placement"):
        validate_task_layout(_task_args(rollout_gpus=4, genrm_gpus=2))
