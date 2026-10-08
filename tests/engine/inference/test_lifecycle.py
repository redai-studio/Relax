# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``LifecycleCoordinator``: in what order models that share GPUs are switched
when a run enters training, generation or scoring."""

from __future__ import annotations

from argparse import Namespace

import pytest

from relax.distributed.ray.placement_planner import plan_placement
from relax.engine.inference.lifecycle import (
    GenerateStage,
    LifecycleCoordinator,
    build_train_coordinator,
    local_rollout_participant,
    manager_participant,
)


class _Cluster:
    """Records switches as they are issued and as they are confirmed.

    ``issued`` is every remote call in order; ``confirmed`` marks where the
    coordinator waited. ``effects`` only lists calls that changed something:
    the managers are idempotent, like the real ones.
    """

    def __init__(self):
        self.timeline: list[str] = []
        self.effects: list[str] = []
        self.failing: set[str] = set()

    def wait(self, refs):
        for model, method in refs:
            call = f"{model.name}.{method}"
            if call in self.failing:
                raise RuntimeError(f"{call} failed")
            model.apply(method)
        self.timeline.append("confirmed")

    def issued(self) -> list[str]:
        return [event for event in self.timeline if event != "confirmed"]


class _Manager:
    """Stands in for a manager's Ray handle."""

    def __init__(self, cluster: _Cluster, name: str, *, active: bool = True):
        self.cluster = cluster
        self.name = name
        self.active = active
        for method in ("offload", "onload", "onload_weights", "onload_kv"):
            setattr(self, method, _Remote(self, method))

    def apply(self, method: str) -> None:
        target = method != "offload"
        if self.active != target or method in ("onload_weights", "onload_kv"):
            self.cluster.effects.append(f"{self.name}.{method}")
        self.active = target

    # The in-process rollout switches.
    def _offload_local(self) -> None:
        self.cluster.timeline.append(f"{self.name}.offload")
        if f"{self.name}.offload" in self.cluster.failing:
            raise RuntimeError(f"{self.name}.offload failed")
        self.apply("offload")
        self.cluster.timeline.append("confirmed")

    def _onload_local(self) -> None:
        self.cluster.timeline.append(f"{self.name}.onload")
        self.apply("onload")
        self.cluster.timeline.append("confirmed")

    @property
    def status(self) -> str:
        return "onload" if self.active else "offload"


class _Remote:
    def __init__(self, manager: _Manager, method: str):
        self._manager, self._method = manager, method

    def remote(self):
        self._manager.cluster.timeline.append(f"{self._manager.name}.{self._method}")
        return (self._manager, self._method)


def _genrm_spec(num_gpus):
    return {"model_path": "/judge", "num_gpus": num_gpus, "num_gpus_per_engine": 1, "engine_config": {}}


def _colocate_args(**overrides) -> Namespace:
    values = dict(
        colocate=True,
        hybrid=False,
        rollout_num_gpus=4,
        resource={"actor": [1, 8], "rollout": [1, 4]},
    )
    values.update(overrides)
    return Namespace(**values)


def _shared_defer_args(**overrides) -> Namespace:
    """Rollout and a deferred GenRM on the same eight bundles."""
    values = dict(
        rollout_num_gpus=8,
        resource={"actor": [1, 8], "rollout": [1, 8], "genrm": [1, 8]},
        _genrm_instances_resolved={"__default__": _genrm_spec(8)},
        _genrm_colocate_with_rollout=True,
        defer_reward_to_post_process=True,
    )
    values.update(overrides)
    return _colocate_args(**values)


def _score_coordinator(cluster: _Cluster, args: Namespace, *extra):
    """The coordinator the rollout manager builds for deferred scoring."""
    rollout = _Manager(cluster, "rollout")
    genrm = _Manager(cluster, "genrm", active=False)
    participants = [local_rollout_participant(rollout), manager_participant("genrm", "__default__", genrm), *extra]
    coordinator = LifecycleCoordinator(plan_placement(args), participants, wait=cluster.wait)
    return coordinator, rollout, genrm


# ----------------------------------------------------------------------
# Entering and leaving the score phase.
# ----------------------------------------------------------------------


def test_lifecycle_enter_score_confirms_rollout_offload_before_loading_genrm():
    cluster = _Cluster()
    coordinator, rollout, genrm = _score_coordinator(cluster, _shared_defer_args())

    coordinator.enter_score()

    assert cluster.timeline == ["rollout.offload", "confirmed", "genrm.onload", "confirmed"]
    assert (rollout.active, genrm.active) == (False, True)


def test_lifecycle_enter_score_leaves_models_on_other_bundles_alone():
    cluster = _Cluster()
    # Split layout: rollout on [0, 4), the deferred GenRM on [4, 8).
    args = _colocate_args(
        resource={"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]},
        _genrm_instances_resolved={"__default__": _genrm_spec(4)},
        defer_reward_to_post_process=True,
    )
    coordinator, rollout, genrm = _score_coordinator(cluster, args)

    coordinator.enter_score()

    assert cluster.issued() == ["genrm.onload"]
    assert (rollout.active, genrm.active) == (True, True)


def test_lifecycle_leave_score_puts_only_score_phase_models_to_sleep():
    cluster = _Cluster()
    args = _shared_defer_args(
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="/teacher",
        rollout_num_gpus=4,
        resource={"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4], "teacher": [1, 4]},
        _genrm_instances_resolved={"__default__": _genrm_spec(4)},
    )
    teacher = _Manager(cluster, "teacher")
    coordinator, rollout, genrm = _score_coordinator(
        cluster, args, manager_participant("teacher", "__default__", teacher)
    )
    coordinator.enter_score()
    cluster.timeline.clear()

    coordinator.leave_score()

    # The inline teacher lives in both phases and stays up.
    assert cluster.issued() == ["genrm.offload"]
    assert (genrm.active, teacher.active) == (False, True)


def test_lifecycle_enter_generate_does_not_resume_a_rollout_that_never_slept():
    """In a split layout scoring leaves rollout up. Its onload body resumes the
    engines unconditionally, so it must not run again."""
    cluster = _Cluster()
    args = _colocate_args(
        resource={"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]},
        _genrm_instances_resolved={"__default__": _genrm_spec(4)},
        defer_reward_to_post_process=True,
    )
    coordinator, rollout, genrm = _score_coordinator(cluster, args)
    coordinator.enter_score()
    coordinator.leave_score()
    cluster.timeline.clear()

    coordinator.enter_generate()

    assert cluster.timeline == []
    assert rollout.active is True


def test_lifecycle_enter_generate_in_one_step_wakes_rollout_only():
    cluster = _Cluster()
    coordinator, rollout, genrm = _score_coordinator(cluster, _shared_defer_args())
    coordinator.enter_score()
    coordinator.leave_score()
    cluster.timeline.clear()

    coordinator.enter_generate()

    assert cluster.issued() == ["rollout.onload"]
    assert (rollout.active, genrm.active) == (True, False)


# ----------------------------------------------------------------------
# Failure and retry.
# ----------------------------------------------------------------------


def test_lifecycle_failed_offload_stops_the_switch_before_any_activation():
    cluster = _Cluster()
    coordinator, rollout, genrm = _score_coordinator(cluster, _shared_defer_args())
    cluster.failing = {"rollout.offload"}

    with pytest.raises(RuntimeError, match="rollout.offload failed"):
        coordinator.enter_score()

    assert cluster.issued() == ["rollout.offload"]
    assert genrm.active is False


def test_lifecycle_retry_completes_without_repeating_finished_work():
    cluster = _Cluster()
    coordinator, rollout, genrm = _score_coordinator(cluster, _shared_defer_args())
    cluster.failing = {"genrm.onload"}
    with pytest.raises(RuntimeError, match="genrm.onload failed"):
        coordinator.enter_score()
    assert cluster.effects == ["rollout.offload"]

    cluster.failing = set()
    coordinator.enter_score()

    # Rollout was already released: asking again changed nothing.
    assert cluster.effects == ["rollout.offload", "genrm.onload"]
    assert (rollout.active, genrm.active) == (False, True)


# ----------------------------------------------------------------------
# The training actor's switches.
# ----------------------------------------------------------------------


def _train_coordinator(cluster, args, *, genrm=(), teacher=None, warm_rollout_kv=True):
    coordinator = build_train_coordinator(
        args,
        rollout_manager=_Manager(cluster, "rollout", active=False),
        genrm_managers=list(genrm) or None,
        teacher_manager=teacher,
        warm_rollout_kv=warm_rollout_kv,
    )
    coordinator._wait = cluster.wait
    return coordinator


def _inline_genrm_args(**overrides) -> Namespace:
    values = dict(
        resource={"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]},
        _genrm_instances_resolved={"quality": _genrm_spec(2), "safety": _genrm_spec(2)},
    )
    values.update(overrides)
    return _colocate_args(**values)


def _inline_teacher_args(**overrides) -> Namespace:
    values = dict(
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="/teacher",
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
    )
    values.update(overrides)
    return _colocate_args(**values)


def test_lifecycle_enter_train_releases_every_scorer_and_waits():
    cluster = _Cluster()
    args = _shared_defer_args(
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="/teacher",
        rollout_num_gpus=4,
        resource={"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4], "teacher": [1, 4]},
        _genrm_instances_resolved={"__default__": _genrm_spec(4)},
    )
    genrm, teacher = _Manager(cluster, "genrm"), _Manager(cluster, "teacher")
    coordinator = _train_coordinator(cluster, args, genrm=[genrm], teacher=teacher)

    coordinator.enter_train()

    # Issued together, confirmed before the call returns. Rollout releases
    # itself after each generation and is not asked here.
    assert cluster.timeline == ["genrm.offload", "teacher.offload", "confirmed"]
    assert (genrm.active, teacher.active) == (False, False)


def test_lifecycle_inline_genrm_keeps_the_original_switch_order():
    """Without deferred scoring: only the rollout weights before weight sync;
    the rollout KV cache and GenRM after it; GenRM released before training."""
    cluster = _Cluster()
    quality, safety = _Manager(cluster, "quality"), _Manager(cluster, "safety")
    coordinator = _train_coordinator(cluster, _inline_genrm_args(), genrm=[quality, safety])

    coordinator.enter_train()
    coordinator.enter_generate(GenerateStage.WEIGHTS)
    coordinator.enter_generate(GenerateStage.REST)

    assert cluster.timeline == [
        "quality.offload",
        "safety.offload",
        "confirmed",
        "rollout.onload_weights",
        "confirmed",
        "rollout.onload_kv",
        "quality.onload",
        "safety.onload",
        "confirmed",
    ]


def test_lifecycle_inline_teacher_keeps_the_original_switch_order():
    """The teacher is woken in the same step as the rollout weights and
    released before training."""
    cluster = _Cluster()
    teacher = _Manager(cluster, "teacher")
    coordinator = _train_coordinator(cluster, _inline_teacher_args(), teacher=teacher)

    coordinator.enter_train()
    coordinator.enter_generate(GenerateStage.WEIGHTS)
    coordinator.enter_generate(GenerateStage.REST)

    assert cluster.timeline == [
        "teacher.offload",
        "confirmed",
        "rollout.onload_weights",
        "teacher.onload",
        "confirmed",
        "rollout.onload_kv",
        "confirmed",
    ]


def test_lifecycle_mopd_teachers_follow_the_routes_order():
    cluster = _Cluster()
    args = _inline_teacher_args(
        teacher_hf_checkpoint=None, opd_teacher_routes='{"math": "/ckpt/math", "code": "/ckpt/code"}'
    )
    teachers = [_Manager(cluster, "math"), _Manager(cluster, "code")]
    coordinator = _train_coordinator(cluster, args, teacher=teachers)

    coordinator.enter_generate(GenerateStage.WEIGHTS)

    assert cluster.issued() == ["rollout.onload_weights", "math.onload", "code.onload"]


def test_lifecycle_rest_stage_skips_kv_warmup_when_no_generation_follows():
    cluster = _Cluster()
    coordinator = _train_coordinator(
        cluster,
        _inline_genrm_args(),
        genrm=[_Manager(cluster, "quality"), _Manager(cluster, "safety")],
        warm_rollout_kv=False,
    )

    coordinator.enter_generate(GenerateStage.REST)

    assert cluster.issued() == ["quality.onload", "safety.onload"]


@pytest.mark.parametrize("custom_post_process", [None, "my_module.post_process"])
def test_lifecycle_deferred_genrm_is_not_woken_for_generation(custom_post_process):
    """Whoever runs the deferred scoring -- the framework or a userland hook --
    GenRM stays asleep while rollout generates."""
    cluster = _Cluster()
    genrm = _Manager(cluster, "genrm", active=False)
    args = _shared_defer_args(custom_reward_post_process_path=custom_post_process)
    coordinator = _train_coordinator(cluster, args, genrm=[genrm])

    coordinator.enter_generate(GenerateStage.WEIGHTS)
    coordinator.enter_generate(GenerateStage.REST)

    assert cluster.issued() == ["rollout.onload_weights", "rollout.onload_kv"]
    assert genrm.active is False


def test_lifecycle_nothing_to_switch_makes_no_call():
    cluster = _Cluster()
    coordinator = _train_coordinator(cluster, _colocate_args())

    coordinator.enter_train()

    assert cluster.timeline == []


def test_lifecycle_train_actor_builds_its_coordinator_from_the_handles_it_holds(monkeypatch):
    """The training actor has no coordinator of its own to keep in sync: it
    builds one from whatever managers it was handed."""
    import ray

    TrainRayActor = pytest.importorskip("relax.distributed.ray.train_actor").TrainRayActor

    cluster = _Cluster()
    monkeypatch.setattr(ray, "get", cluster.wait)
    actor = object.__new__(type("_Actor", (TrainRayActor,), {"__abstractmethods__": frozenset()}))
    actor.args = _inline_genrm_args()
    actor.rollout_manager = _Manager(cluster, "rollout", active=False)

    # Before any scorer is set there is nothing to release.
    actor.lifecycle_coordinator().enter_train()
    assert cluster.timeline == []

    actor.set_genrm_manager([_Manager(cluster, "quality"), _Manager(cluster, "safety")])
    actor.lifecycle_coordinator().enter_train()
    actor.lifecycle_coordinator(warm_rollout_kv=False).enter_generate(GenerateStage.REST)

    assert cluster.issued() == ["quality.offload", "safety.offload", "quality.onload", "safety.onload"]
