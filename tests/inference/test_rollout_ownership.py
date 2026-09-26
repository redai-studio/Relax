# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from relax.distributed.ray.rollout_workload import RolloutWorkload


async def test_workload_preserves_generation_data_and_dynamic_batch_contract():
    from relax.engine.rollout.base_types import RolloutFnTrainOutput

    calls = []
    args = SimpleNamespace(ci_test=False, partial_rollout=True, use_dynamic_global_batch_size=True)
    groups = [[SimpleNamespace(index=1), SimpleNamespace(index=1)], [SimpleNamespace(index=2)]]

    def generate(*values, evaluation):
        calls.append((values, evaluation))
        return RolloutFnTrainOutput(samples=groups)

    facade = SimpleNamespace(
        args=args,
        generate_rollout=generate,
        data_source=object(),
        data_system_client=object(),
        health_monitoring_resume=MagicMock(),
    )

    await RolloutWorkload(facade).generate(3)

    assert facade.rollout_id == 3
    assert facade._dynamic_global_batch_size == 2
    assert calls == [((args, 3, facade.data_source, facade.data_system_client), False)]
    facade.health_monitoring_resume.assert_called_once_with()
    assert not hasattr(facade, "servers")


async def test_workload_evaluation_returns_business_output_without_resource_actions():
    from relax.engine.rollout.base_types import RolloutFnEvalOutput

    expected = RolloutFnEvalOutput(data={"evaluation": {}})
    facade = SimpleNamespace(
        args=SimpleNamespace(),
        eval_generate_rollout=lambda *args, **kwargs: expected,
        data_source=object(),
        data_system_client=object(),
        health_monitoring_resume=MagicMock(),
    )

    assert await RolloutWorkload(facade).evaluate(4) is expected
    assert not hasattr(facade, "inference_manager")


async def test_workload_deferred_entry_keeps_original_facade_for_local_lifecycle(monkeypatch):
    from relax.distributed.ray import rollout_workload

    seen = []
    monkeypatch.setattr(
        rollout_workload, "run_deferred_rollout", lambda manager, rollout_id: seen.append((manager, rollout_id))
    )
    facade = SimpleNamespace(
        args=SimpleNamespace(ci_test=False, inference_defer_roles=["teacher"]),
        health_monitoring_resume=MagicMock(),
    )

    await RolloutWorkload(facade).generate(2)

    assert seen == [(facade, 2)]


@pytest.fixture
def manager_class():
    pytest.importorskip("sglang")

    from relax.distributed.ray.rollout import RolloutManager

    return RolloutManager.__ray_metadata__.modified_class


def test_facade_properties_forward_to_one_real_core_owner(manager_class):
    from relax.distributed.ray.inference_manager import InferenceManager

    facade = object.__new__(manager_class)
    facade.args = SimpleNamespace()
    servers = {"policy": SimpleNamespace(engine_groups=[])}
    facade.servers = servers
    facade.status = "offload"
    monitor = object()
    facade._health_monitors = [monitor]

    assert isinstance(facade.inference_manager, InferenceManager)
    assert facade.servers is facade.inference_manager.servers is servers
    assert facade._engine_lifecycle_lock is facade.inference_manager._lifecycle_lock
    assert facade._inference_observation is facade.inference_manager._inference_observation
    assert facade.inference_manager.status == "offload"
    assert facade.inference_manager.health_monitors == [monitor]
    assert "servers" not in facade.__dict__
    assert "status" not in facade.__dict__
    assert "_health_monitors" not in facade.__dict__


def test_legacy_lifecycle_facade_delegates_without_self_remote_calls(manager_class):
    facade = object.__new__(manager_class)
    facade.inference_manager = MagicMock()
    facade.health_monitoring_pause = MagicMock()

    facade._offload_local()
    facade._onload_local(["weights"])
    facade._shutdown_all_engines(timeout=7.0)

    facade.inference_manager.offload.assert_called_once_with()
    facade.inference_manager.onload.assert_called_once_with(["weights"])
    facade.inference_manager.shutdown_rollout.assert_called_once_with(timeout=7.0)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_rollout_startup_failure_cleans_staged_workers_and_preserves_borrowed_pg(
    manager_class, monkeypatch, cleanup_fails
):
    from relax.distributed.ray import rollout
    from relax.distributed.ray.inference_manager import InferenceCleanupError

    args = SimpleNamespace(sglang_router_ip=None, sglang_router_port=None)
    engine = MagicMock()
    engine.shutdown.remote.return_value = "shutdown-ref"
    group = SimpleNamespace(
        all_engines=[engine], pg=("borrowed", [0], [0]), pg_owned=False, nodes_per_engine=1, rank_offset=0
    )
    server = SimpleNamespace(engine_groups=[group], model_name="policy")
    borrowed_pg = object()

    def fail_start(config, pg, servers):
        assert pg is borrowed_pg
        servers["policy"] = server
        config.sglang_router_ip, config.sglang_router_port = "router.example", 12345
        raise ValueError("startup failed")

    def get(ref, timeout=None):
        if cleanup_fails:
            raise TimeoutError("child process unconfirmed")

    monkeypatch.setattr(rollout, "_start_rollout_servers", fail_start)
    monkeypatch.setattr(rollout.ray, "get", get)
    killed = []
    monkeypatch.setattr(rollout.ray, "kill", killed.append)
    monkeypatch.setattr(rollout, "_LAUNCHED_ROUTER_PROCESSES", [])
    expected = InferenceCleanupError if cleanup_fails else ValueError
    with pytest.raises(expected) as error:
        rollout.start_rollout_servers(args, borrowed_pg)

    assert (args.sglang_router_ip, args.sglang_router_port) == (None, None)
    if cleanup_fails:
        assert error.value.rollout_servers["policy"] is server
        assert group.all_engines == [engine]
        assert killed == []
    else:
        assert group.all_engines == [None]
        assert killed == [engine]
