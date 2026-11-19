# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.distributed.ray import shared_task_startup as module


@pytest.mark.parametrize("fail_release", [False, True])
def test_startup_confirms_release_before_constructing_next_owner(monkeypatch, fail_release):
    trace = []

    def owner(name):
        def offload():
            trace.append(name + " release")
            if fail_release and name == "rollout":
                raise TimeoutError("release unconfirmed")

        return SimpleNamespace(offload=SimpleNamespace(remote=offload))

    rollout = owner("rollout")
    teacher = owner("teacher")
    monkeypatch.setattr(module, "plan_task_placement", lambda *args: trace.append("plan"))
    monkeypatch.setattr(module.ray, "get", lambda ref: ref)
    monkeypatch.setattr(module, "rollback_static_inference_manager", lambda item: trace.append("rollback"))

    def create(*args, **kwargs):
        trace.append("teacher construct")
        return teacher

    monkeypatch.setattr(module, "create_static_inference_manager", create)
    phase_config = []
    coordinator = object()

    def configure(phases, initial):
        phase_config.append((phases, initial))
        return coordinator

    monkeypatch.setattr(module, "create_phase_coordinator", configure)
    item = module.SharedStaticOwner(
        None,
        SimpleNamespace(
            spec=SimpleNamespace(role=SimpleNamespace(value="teacher"), deployment=SimpleNamespace(phase="teacher")),
            binding=None,
        ),
        {},
    )

    def start_rollout():
        trace.append("rollout construct")
        return rollout

    if fail_release:
        with pytest.raises(TimeoutError):
            module.start_shared_task(None, None, [item], start_rollout)
        assert trace == ["plan", "rollout construct", "rollout release", "rollback"]
    else:
        _, _, actual = module.start_shared_task(None, None, [item], start_rollout)
        assert actual is coordinator
        assert actual is not rollout
        phases, initial = phase_config[0]
        assert trace == ["plan", "rollout construct", "rollout release", "teacher construct", "teacher release"]
        assert initial == "startup" and phases[initial] == ()


def test_prepare_shared_task_injects_existing_genrm_owner(monkeypatch):
    from relax.engine.inference.types import Role

    owner = object()
    rollout = object()
    coordinator = object()
    observed = []

    def start(args, binding, static, create, **kwargs):
        observed.extend(static)
        return rollout, (owner,), coordinator

    monkeypatch.setattr(module, "start_shared_task", start)
    args = SimpleNamespace(
        _genrm_instances_resolved={
            "__default__": {
                "model_path": "judge",
                "num_gpus": 2,
                "num_gpus_per_engine": 2,
                "engine_config": {},
                "sampling_config": {},
            }
        },
        _genrm_colocate_with_rollout=True,
        defer_reward_to_post_process=True,
        num_gpus_per_node=4,
        colocate=True,
        use_opd=False,
        rollout_num_gpus=4,
    )
    module.prepare_shared_task(args, object())
    assert args._rollout_engine_manager is rollout
    assert args._genrm_managers == {"__default__": owner}
    assert args._inference_phase_coordinator is coordinator
    assert observed[0].placement.spec.role is Role.GENRM
    assert observed[0].placement.bundle_offset == 0


def test_genrm_factory_consumes_prepared_owners_without_new_allocation():
    from relax.distributed.ray.placement_group import create_genrm_managers

    owners = {"__default__": object()}
    assert create_genrm_managers(SimpleNamespace(_genrm_managers=owners), None) is owners


def test_prepare_shared_mopd_preserves_source_models_and_offsets(monkeypatch):
    observed = []
    owners = [
        SimpleNamespace(
            snapshot=SimpleNamespace(
                remote=lambda role: SimpleNamespace(
                    models=[SimpleNamespace(replicas=[SimpleNamespace(base_url="http://teacher")])]
                )
            )
        )
        for _ in range(2)
    ]

    def start(args, binding, static, create, **kwargs):
        observed.extend(static)
        return object(), tuple(owners), object()

    monkeypatch.setattr(module, "start_shared_task", start)
    monkeypatch.setattr(module.ray, "get", lambda ref: ref)
    args = SimpleNamespace(
        _genrm_instances_resolved={},
        use_opd=True,
        opd_type="sglang",
        colocate=True,
        hybrid=False,
        teacher_hf_checkpoint=None,
        opd_teacher_routes='{"math":"math-model","code":"code-model"}',
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
        rollout_num_gpus=4,
        num_gpus_per_node=8,
        use_critic=False,
    )
    assert module.prepare_shared_task(args, object()) == owners
    assert [item.placement.bundle_offset for item in observed] == [4, 6]
    assert [item.placement.spec.models[0].model_id for item in observed] == ["math", "code"]
    assert args._teacher_inference_managers == dict(zip(("math", "code"), owners))
    assert set(args.opd_teacher_routes_map) == {"math", "code"}


def test_partial_startup_retains_owner_on_config_until_cleanup_confirmed(monkeypatch):
    args = SimpleNamespace()
    rollout = SimpleNamespace(offload=SimpleNamespace(remote=lambda: None))
    monkeypatch.setattr(module, "plan_task_placement", lambda *args: None)
    monkeypatch.setattr(module.ray, "get", lambda ref: ref)
    item = module.SharedStaticOwner(
        None,
        SimpleNamespace(
            spec=SimpleNamespace(role=SimpleNamespace(value="teacher")),
            binding=None,
        ),
        {},
    )

    def fail_construct(*args, **kwargs):
        raise RuntimeError("teacher start failed")

    def fail_cleanup(owner):
        assert args._shared_startup_pending_owners == [rollout]
        raise TimeoutError("cleanup unconfirmed")

    monkeypatch.setattr(module, "create_static_inference_manager", fail_construct)
    monkeypatch.setattr(module, "rollback_static_inference_manager", fail_cleanup)
    with pytest.raises(RuntimeError, match="teacher start failed"):
        module.start_shared_task(args, None, [item], lambda: rollout)
    assert args._shared_startup_pending_owners == [rollout]
    with pytest.raises(TimeoutError, match="cleanup unconfirmed"):
        module.retry_shared_startup_cleanup(args)
    assert args._shared_startup_pending_owners == [rollout]
    with pytest.raises(RuntimeError, match="unconfirmed owners"):
        module.start_shared_task(args, None, [], lambda: pytest.fail("must not restart"))
    cleaned = []
    monkeypatch.setattr(module, "rollback_static_inference_manager", cleaned.append)
    assert module.retry_shared_startup_cleanup(args)
    assert cleaned == [rollout]
    assert not hasattr(args, "_shared_startup_pending_owners")


@pytest.mark.parametrize("fail_static", [False, True])
def test_shared_shutdown_retains_unconfirmed_owners_and_closes_worker(monkeypatch, fail_static):
    events = []
    worker = object()
    coordinator = object()
    rollout = SimpleNamespace(dispose=SimpleNamespace(remote=lambda item: events.append(("worker", item))))
    static = object()
    args = SimpleNamespace(
        _shared_inference_owners=[rollout, static],
        _rollout_engine_manager=rollout,
        _rollout_worker=worker,
        _inference_phase_coordinator=coordinator,
    )
    monkeypatch.setattr(module.ray, "get", lambda ref, **kwargs: ref)
    monkeypatch.setattr(module.ray, "kill", lambda owner: events.append(("kill", owner)))

    def release(owner):
        events.append(("static", owner))
        if fail_static:
            raise TimeoutError("release unconfirmed")

    monkeypatch.setattr(module, "rollback_static_inference_manager", release)
    if fail_static:
        with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
            module.shutdown_shared_task(args)
        assert args._shared_inference_owners == [static]
        assert ("kill", coordinator) not in events
    else:
        module.shutdown_shared_task(args)
        assert not hasattr(args, "_shared_inference_owners")
        assert ("kill", coordinator) in events
    assert ("worker", worker) in events
    assert ("kill", rollout) in events
