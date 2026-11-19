# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
import threading
from copy import copy
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import transfer_queue

from tests.cpu.core.controller_test_utils import load_controller_with_stubbed_dependencies


if not hasattr(transfer_queue, "StreamingTokenBudgetSampler"):
    pytest.skip(
        "controller cleanup tests require a TransferQueue build with StreamingTokenBudgetSampler",
        allow_module_level=True,
    )

controller = load_controller_with_stubbed_dependencies("_test_controller_s3_model_cleanup_controller")
ROLES = controller.ROLES


def test_train_start_config_attaches_remote_model_config_without_mutating_args(monkeypatch):
    config = SimpleNamespace(model_source=SimpleNamespace(uri="s3://bucket/model/"))
    model_config = {"model_type": "qwen3", "hidden_size": 4096}
    monkeypatch.setattr(controller, "read_s3_model_config", lambda args: model_config)

    start_config = controller._build_train_start_config(config)

    assert start_config is not config
    assert start_config._model_source_config is model_config
    assert not hasattr(config, "_model_source_config")


def test_train_start_config_fails_open_when_remote_config_is_unavailable(monkeypatch):
    config = SimpleNamespace(model_source=SimpleNamespace(uri="s3://bucket/model/"))
    warnings = []
    monkeypatch.setattr(
        controller,
        "read_s3_model_config",
        lambda _args: (_ for _ in ()).throw(RuntimeError("not found")),
    )
    monkeypatch.setattr(controller.logger, "warning", warnings.append)

    start_config = controller._build_train_start_config(config)

    assert start_config is not config
    assert not hasattr(start_config, "_model_source_config")
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0]
    assert "not found" not in warnings[0]
    assert "s3://" not in warnings[0]


def test_train_start_config_does_not_read_local_model(monkeypatch):
    config = SimpleNamespace(model_source=SimpleNamespace(uri="/models/qwen3"))
    monkeypatch.setattr(
        controller,
        "read_s3_model_config",
        lambda _args: pytest.fail("local model must not use the S3 client"),
    )

    start_config = controller._build_train_start_config(config)

    assert start_config is not config
    assert not hasattr(start_config, "_model_source_config")


class _FakeCleanupTask:
    def __init__(self):
        self.node_ids = []

    def options(self, *, scheduling_strategy):
        self.node_ids.append(scheduling_strategy.node_id)
        return self

    def remote(self, config):
        return f"cleanup-{self.node_ids[-1]}"


def test_controller_s3_cleanup_runs_once_on_each_alive_node(monkeypatch):
    monkeypatch.setenv("RELAX_S3_MODEL_CLEANUP_TASK_TIMEOUT_S", "17.5")
    config = SimpleNamespace(model_source=SimpleNamespace(uri="s3://bucket/model/"), disable_s3_model_cleanup=False)
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    instance.serve_dict = {}
    cleanup_task = _FakeCleanupTask()
    node_a = "a" * 56
    node_b = "b" * 56

    remote_options = {}

    def fake_remote(**kwargs):
        remote_options.update(kwargs)
        return lambda _func: cleanup_task

    monkeypatch.setattr(controller.ray, "remote", fake_remote)
    monkeypatch.setattr(
        controller.ray,
        "nodes",
        lambda: [
            {"Alive": True, "NodeID": node_a},
            {"Alive": False, "NodeID": "c" * 56},
            {"Alive": True, "NodeID": node_b},
        ],
    )
    monkeypatch.setattr(
        controller.ray,
        "wait",
        lambda refs, *, num_returns, timeout: (refs, []) if num_returns == 2 and timeout == 17.5 else None,
    )
    monkeypatch.setattr(controller.ray, "get", lambda refs: [(2, 10), (1, 5)] if len(refs) == 2 else None)

    instance._cleanup_s3_model_weights_after_init()
    assert cleanup_task.node_ids == [node_a, node_b]
    assert remote_options == {"num_cpus": 0, "max_retries": 0}


def test_controller_s3_cleanup_skips_shm_backed_rollout():
    config = SimpleNamespace(
        model_source=SimpleNamespace(),
        sglang_load_format="auto",
        rollout_external=False,
        sglang_config=None,
    )

    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})
    config.sglang_load_format = "dummy"
    assert controller._can_cleanup_s3_model_weights(config, {"rollout": object()})
    config.sglang_load_format = "runai_streamer"
    assert controller._can_cleanup_s3_model_weights(config, {"rollout": object()})
    config.sglang_load_format = "remote"
    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})
    config.sglang_load_format = "full"
    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})
    config.sglang_load_format = "dummy"
    config.sglang_config = "/tmp/sglang.yaml"
    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})
    config.sglang_config = None
    config.rollout_external = True
    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})


def test_controller_s3_cleanup_without_rollout_is_safe():
    config = SimpleNamespace(
        model_source=SimpleNamespace(),
        sglang_load_format="auto",
        rollout_external=False,
        sglang_config=None,
    )

    assert controller._can_cleanup_s3_model_weights(config, {"actor": object()})


def test_controller_s3_cleanup_skips_debug_rollout_only():
    config = SimpleNamespace(
        model_source=SimpleNamespace(),
        debug_rollout_only=True,
        sglang_load_format="dummy",
        rollout_external=False,
        sglang_config=None,
    )

    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})


def test_controller_s3_cleanup_requires_model_source():
    config = SimpleNamespace(
        model_source=None,
        sglang_load_format="runai_streamer",
        rollout_external=False,
        sglang_config=None,
    )

    assert not controller._can_cleanup_s3_model_weights(config, {})
    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})


def test_controller_s3_cleanup_timeout_cancels_pending_tasks(monkeypatch):
    monkeypatch.setenv("RELAX_S3_MODEL_CLEANUP_CANCEL_TIMEOUT_S", "2.5")
    config = SimpleNamespace(model_source=SimpleNamespace(uri="s3://bucket/model/"), disable_s3_model_cleanup=False)
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    instance.serve_dict = {}
    cleanup_task = _FakeCleanupTask()
    pending_ref = object()
    cancelled = []
    wait_calls = []

    monkeypatch.setattr(controller.ray, "remote", lambda **_kwargs: lambda _func: cleanup_task)
    monkeypatch.setattr(controller.ray, "nodes", lambda: [{"Alive": True, "NodeID": "a" * 56}])

    def fake_wait(refs, **kwargs):
        wait_calls.append((refs, kwargs))
        return ([], [pending_ref]) if len(wait_calls) == 1 else ([pending_ref], [])

    monkeypatch.setattr(controller.ray, "wait", fake_wait)
    monkeypatch.setattr(controller.ray, "cancel", lambda ref, *, force: cancelled.append((ref, force)))

    with pytest.raises(TimeoutError, match="1 of 1 node tasks pending"):
        instance._cleanup_s3_model_weights_after_init()
    assert cancelled == [(pending_ref, True)]
    assert wait_calls[1] == (
        [pending_ref],
        {
            "num_returns": 1,
            "timeout": 2.5,
        },
    )


def test_controller_s3_cleanup_timeout_raises_when_cancel_remains_pending(monkeypatch):
    config = SimpleNamespace(model_source=SimpleNamespace(uri="s3://bucket/model/"), disable_s3_model_cleanup=False)
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    instance.serve_dict = {}
    cleanup_task = _FakeCleanupTask()
    pending_ref = object()
    cancelled = []

    monkeypatch.setattr(controller.ray, "remote", lambda **_kwargs: lambda _func: cleanup_task)
    monkeypatch.setattr(controller.ray, "nodes", lambda: [{"Alive": True, "NodeID": "a" * 56}])
    monkeypatch.setattr(controller.ray, "wait", lambda _refs, **_kwargs: ([], [pending_ref]))
    monkeypatch.setattr(controller.ray, "cancel", lambda ref, *, force: cancelled.append((ref, force)))

    with pytest.raises(TimeoutError, match="cancellation did not reach 1 of 1 pending node tasks"):
        instance._cleanup_s3_model_weights_after_init()
    assert cancelled == [(pending_ref, True)]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_controller_s3_cleanup_rejects_non_positive_task_timeout(monkeypatch, value):
    monkeypatch.setenv("RELAX_S3_MODEL_CLEANUP_TASK_TIMEOUT_S", value)
    config = SimpleNamespace(model_source=SimpleNamespace(uri="s3://bucket/model/"), disable_s3_model_cleanup=False)
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    instance.serve_dict = {}
    cleanup_task = _FakeCleanupTask()

    monkeypatch.setattr(controller.ray, "remote", lambda **_kwargs: lambda _func: cleanup_task)
    monkeypatch.setattr(controller.ray, "nodes", lambda: [{"Alive": True, "NodeID": "a" * 56}])

    with pytest.raises(ValueError, match="RELAX_S3_MODEL_CLEANUP_TASK_TIMEOUT_S must be greater than 0"):
        instance._cleanup_s3_model_weights_after_init()


def test_controller_s3_cleanup_disable_skips_node_discovery(monkeypatch):
    config = SimpleNamespace(model_source=SimpleNamespace(), disable_s3_model_cleanup=True)
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    monkeypatch.setattr(controller.ray, "nodes", lambda: (_ for _ in ()).throw(AssertionError("must not run")))

    instance._cleanup_s3_model_weights_after_init()


def test_controller_local_model_source_skips_node_discovery(monkeypatch):
    config = SimpleNamespace(
        model_source=SimpleNamespace(uri="/models/local"),
        disable_s3_model_cleanup=False,
    )
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    monkeypatch.setattr(controller.ray, "nodes", lambda: (_ for _ in ()).throw(AssertionError("must not run")))

    instance._cleanup_s3_model_weights_after_init()


@pytest.mark.parametrize(
    ("role", "load_format", "shares_actor_pg", "expected"),
    [
        ("actor", "runai_streamer", False, controller.ModelCompleteness.FULL),
        ("rollout", "dummy", False, controller.ModelCompleteness.METADATA),
        ("rollout", "runai_streamer", True, controller.ModelCompleteness.NONE),
        ("rollout", "auto", False, controller.ModelCompleteness.NONE),
        ("rollout", "auto", True, controller.ModelCompleteness.FULL),
        ("genrm", "auto", True, controller.ModelCompleteness.NONE),
    ],
)
def test_policy_model_completeness(role, load_format, shares_actor_pg, expected):
    config = SimpleNamespace(
        sglang_load_format=load_format,
        rollout_external=False,
        sglang_config=None,
    )

    assert (
        controller.Controller._policy_model_completeness(
            config,
            role,
            shares_actor_pg=shares_actor_pg,
        )
        == expected
    )


def test_controller_prepares_then_deploys_service(monkeypatch):
    events = []

    class FakeService:
        def __init__(self, _cls, **kwargs):
            events.append(("prepare", kwargs["defer_deploy"]))
            self.pgs = ("pg", [0], [0])

        def deploy(self):
            events.append(("deploy", self.pgs))

    instance = controller.Controller.__new__(controller.Controller)
    instance.config = SimpleNamespace()
    instance.runtime_env = None
    instance._health_manager = SimpleNamespace(status=object())
    monkeypatch.setattr(controller, "Service", FakeService)

    role, service, error = instance._create_service_task(
        "actor",
        object(),
        1,
        None,
        None,
        defer_deploy=True,
    )
    assert (role, error) == ("actor", None)
    assert events == [("prepare", True)]

    role, deployed, error = instance._deploy_service_task(role, service)
    assert (role, deployed, error) == ("actor", service, None)
    assert events == [("prepare", True), ("deploy", service.pgs)]


def test_controller_service_phase_preserves_fully_async_concurrency():
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = SimpleNamespace(fully_async=True)
    barrier = threading.Barrier(2)

    def task(role):
        barrier.wait(timeout=2)
        return (role, f"service-{role}", None)

    assert instance._run_service_phase(
        controller.ServiceStartupPhase.ALLOCATE_RESOURCES,
        task,
        [("actor",), ("rollout",)],
    ) == {
        "actor": "service-actor",
        "rollout": "service-rollout",
    }


@pytest.mark.parametrize(
    ("model_source", "expected"),
    [
        (None, False),
        (SimpleNamespace(uri="/models/qwen"), False),
        (SimpleNamespace(uri="s3://bucket/qwen"), True),
    ],
)
def test_controller_uses_staged_startup_only_for_s3(model_source, expected):
    assert controller._uses_s3_model_prefetch(SimpleNamespace(model_source=model_source)) is expected


def test_controller_local_model_uses_constructor_immediate_deploy(monkeypatch):
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = SimpleNamespace(model_source=SimpleNamespace(uri="/models/qwen"))
    instance.serve_dict = {}
    service = object()
    service_args = [("actor", object(), 1, None, None, ["actor"])]
    phases = []

    def fake_run(phase, task, task_args):
        phases.append((phase, task, task_args))
        return {"actor": service}

    monkeypatch.setattr(instance, "_run_service_phase", fake_run)
    monkeypatch.setattr(
        instance,
        "_start_model_prefetch",
        lambda _pgs: (_ for _ in ()).throw(AssertionError("local models must not prefetch")),
    )

    assert instance._start_services(service_args) == []
    assert instance.serve_dict == {"actor": service}
    assert phases == [(controller.ServiceStartupPhase.DEPLOY, instance._create_service_task, service_args)]


def test_controller_s3_model_allocates_prefetches_then_deploys(monkeypatch):
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = SimpleNamespace(model_source=SimpleNamespace(uri="s3://bucket/qwen"))
    instance.serve_dict = {}
    prepared = SimpleNamespace(pgs=("pg", [0], [0]))
    deployed = object()
    service_args = [("actor", object(), 1, None, None, ["actor"])]
    events = []

    def fake_run(phase, task, task_args):
        if phase == controller.ServiceStartupPhase.ALLOCATE_RESOURCES:
            assert isinstance(task, partial)
            assert task.func == instance._create_service_task
            assert task.keywords == {"defer_deploy": True}
            assert task_args == service_args
            events.append("allocate")
            return {"actor": prepared}
        assert phase == controller.ServiceStartupPhase.DEPLOY
        assert task == instance._deploy_service_task
        assert task_args == [("actor", prepared)]
        events.append("deploy")
        return {"actor": deployed}

    def fake_prefetch(service_pgs):
        assert service_pgs == {"actor": prepared.pgs}
        events.append("prefetch")
        return ["prefetch-ref"]

    monkeypatch.setattr(instance, "_run_service_phase", fake_run)
    monkeypatch.setattr(instance, "_start_model_prefetch", fake_prefetch)

    assert instance._start_services(service_args) == ["prefetch-ref"]
    assert events == ["allocate", "prefetch", "deploy"]
    assert instance.serve_dict == {"actor": deployed}


def test_controller_s3_cleanup_runs_after_initial_sync_before_service_run(monkeypatch):
    events = []

    class Service:
        def __init__(self, role):
            self.role = role

        async def get_rollout_manager(self):
            return object()

        async def set_rollout_manager(self, _manager):
            events.append("set_rollout_manager")

        def update_weights_fully_async(self, recovery_token=None):
            async def update():
                events.append("update_weights")

            return update()

        def recv_weight_fully_async(self, recovery_token=None):
            async def receive():
                events.append(f"receive_{self.role.value}")

            return receive()

        async def get_step(self):
            return 0

        async def set_step(self, _step):
            events.append(f"set_step_{self.role.value}")

        def run(self):
            events.append(f"run_{self.role.value}")
            return None

    instance = controller.Controller.__new__(controller.Controller)
    instance.config = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        fully_async=True,
        hybrid=False,
    )
    instance.serve_dict = {
        ROLES.actor: Service(ROLES.actor),
        ROLES.rollout: Service(ROLES.rollout),
        ROLES.actor_fwd: Service(ROLES.actor_fwd),
        ROLES.reference: Service(ROLES.reference),
    }
    instance._teacher_manager = None
    instance._pending_task_refs = []
    instance._pending_task_refs_lock = threading.Lock()
    instance._restarting = False
    monkeypatch.setattr(controller, "set_managed_opd_teacher_on_actor_service", lambda *_args: _async_noop())
    monkeypatch.setattr(instance, "_cleanup_s3_model_weights_after_init", lambda: events.append("cleanup"))

    instance.training_loop()

    cleanup_index = events.index("cleanup")
    assert events.index("update_weights") < cleanup_index
    assert events.index("set_step_actor") < cleanup_index
    assert cleanup_index < events.index("run_actor")


async def _async_noop():
    return None


@pytest.mark.parametrize("use_agentic_rollout", [False, True])
def test_controller_startup_preserves_tokenizer_caches(monkeypatch, tmp_path, use_agentic_rollout):
    from relax.distributed.ray import shared_task_startup

    # Engine startup is the external scheduling boundary; this regression
    # exercises actual cache publication and the Controller deployment order.
    monkeypatch.setattr(shared_task_startup, "prepare_shared_task", lambda *args, **kwargs: None)

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from relax.utils import s3_model_loader
    from relax.utils.data.processing_utils import load_tokenizer
    from relax.utils.model_source import ModelSource

    # Keep real cache publication, cleanup and tokenizer loading. Only the S3
    # transport and Ray scheduling/service boundaries are replaced.
    source = tmp_path / "source"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(source)
    (source / "config.json").write_text('{"model_type": "bert"}')
    objects = {f"model/{path.name}": path.read_bytes() for path in source.iterdir()}
    client = SimpleNamespace(
        get_paginator=lambda _name: SimpleNamespace(
            paginate=lambda **_kwargs: [{"Contents": [{"Key": k, "Size": len(v)} for k, v in objects.items()]}]
        ),
        head_object=lambda **kwargs: {"ContentLength": len(objects[kwargs["Key"]])},
        get_object=lambda **kwargs: {
            "Body": SimpleNamespace(iter_chunks=lambda **_kwargs: [objects[kwargs["Key"]]], close=lambda: None)
        },
    )
    monkeypatch.setattr(s3_model_loader, "_make_s3_client", lambda **_kwargs: client)
    cache_root = tmp_path / "shm"
    cache_root.mkdir()
    stale = cache_root / "relax_model_0123456789abcdef"
    stale.mkdir()
    (stale / "abandoned.part").write_bytes(b"old job")

    config = SimpleNamespace(
        model_source=ModelSource(uri="s3://bucket/model/"),
        hf_checkpoint="s3://bucket/model/",
        s3_model_shm_root=str(cache_root),
        s3_model_download_workers=1,
        advantage_estimator="grpo",
        loss_type="policy_loss",
        use_agentic_rollout=use_agentic_rollout,
        debug_train_only=False,
        debug_rollout_only=True,
        colocate=False,
        fully_async=False,
        hybrid=False,
        resource={ROLES.rollout: (1, 0)},
        data_source_path="test.data_source",
        num_rollout=2,
    )
    agentic_paths = []

    def create_data_source(args, _cls):
        # Resume a shard reader before a later data-source download could hide
        # an intervening deletion by republishing the same cache path.
        for path in agentic_paths:
            assert load_tokenizer(path, local_files_only=True).encode("hello") == [1]
        return s3_model_loader.prepare_model_maybe_update_args(copy(args), completeness="metadata").path

    def deploy_agentic(*, config, runtime_env):
        agentic_paths.append(create_data_source(config, None))

    class Task:
        def __init__(self, func):
            self.func = func

        def options(self, **_kwargs):
            return self

        def remote(self, *args):
            return partial(self.func, *args)

    class Service:
        def __init__(self, _cls, *, data_source, **_kwargs):
            self.pgs = None
            self.path = data_source

        def deploy(self):
            # Resume the asynchronous data-source constructor after resource
            # allocation/prefetch. A late stale cleanup deletes this path.
            self.tokens = load_tokenizer(self.path, local_files_only=True).encode("hello")

    monkeypatch.setattr(controller.ray, "remote", lambda **_kwargs: Task)
    monkeypatch.setattr(controller.ray, "get", lambda refs: [ref() for ref in refs])
    monkeypatch.setattr(
        controller.ray, "nodes", lambda: [{"Alive": True, "NodeID": "a" * 56, "Resources": {"GPU": 1}}]
    )
    monkeypatch.setattr(controller.device_utils, "get_ray_accelerator_name", lambda: "GPU")
    monkeypatch.setattr(controller, "Service", Service)
    monkeypatch.setattr(
        controller,
        "HealthManager",
        lambda **_kwargs: SimpleNamespace(status=None, mark_healthy=lambda _role: None),
    )
    monkeypatch.setattr(controller.Controller, "_initialize_data_system", lambda self: None)
    monkeypatch.setattr(controller, "create_dcs_deployment", lambda **_kwargs: (None, "test-coordinator"))
    monkeypatch.setattr(controller, "deploy_agentic_chat_api_services", deploy_agentic)
    monkeypatch.setitem(
        sys.modules,
        "relax.core.registry",
        SimpleNamespace(resolve_algorithm=lambda _config: {ROLES.rollout: object}),
    )
    monkeypatch.setattr(controller, "load_function", lambda _path: object)
    monkeypatch.setattr(controller, "create_data_source_actor", create_data_source)
    monkeypatch.setattr(controller.relax_utils, "report_train_start", lambda _config: None)
    monkeypatch.setattr(controller, "_build_train_start_config", lambda args: args)
    monkeypatch.setattr(controller.Controller, "_cleanup_aborted_service", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(controller.Controller, "_validate_gpu_resources", lambda *_args: None)

    instance = controller.Controller(config)

    service = instance.serve_dict[ROLES.rollout]
    assert service.tokens == [1]
    assert Path(service.path, "tokenizer.json").is_file()
    assert not stale.exists()
    assert len(agentic_paths) == int(use_agentic_rollout)
    for path in agentic_paths:
        assert load_tokenizer(path, local_files_only=True).encode("hello") == [1]


@pytest.mark.parametrize("source_uri", ["s3://bucket/model", "/models/student"])
def test_shared_task_constructs_sleeping_owners_after_prefetch_before_deploy(monkeypatch, source_uri):
    from relax.distributed.ray import shared_task_startup

    instance = controller.Controller.__new__(controller.Controller)
    instance.config = SimpleNamespace(model_source=SimpleNamespace(uri=source_uri))
    instance._shared_inference_startup = True
    instance.runtime_env = {}
    instance.serve_dict = {}
    prepared = SimpleNamespace(pgs=("pg", [0], [0]))
    teacher = SimpleNamespace(pgs=("teacher-pg", [0], [0]))
    events = []

    def run(phase, task, task_args):
        if phase is controller.ServiceStartupPhase.ALLOCATE_RESOURCES:
            events.append("allocate")
            return {"rollout": prepared, "teacher": teacher}
        events.append("deploy")
        return {"rollout": prepared, "teacher": teacher}

    def prepare(args, binding, **kwargs):
        assert binding is prepared.pgs
        assert kwargs["bindings"]["teacher"] is teacher.pgs
        events.append("owners sleeping")
        return "teacher"

    monkeypatch.setattr(instance, "_run_service_phase", run)
    monkeypatch.setattr(instance, "_start_model_prefetch", lambda bindings: ["prefetch"])
    monkeypatch.setattr(controller.ray, "get", lambda refs: events.append("prefetch complete"))
    monkeypatch.setattr(shared_task_startup, "prepare_shared_task", prepare)
    instance._start_services([])
    assert events == ["allocate", "prefetch complete", "owners sleeping", "deploy"]
    assert instance._teacher_manager == "teacher"


@pytest.mark.parametrize("release_fails", [False, True])
def test_shared_task_cleanup_confirms_release_before_killing_coordinator(monkeypatch, release_fails):
    from relax.distributed.ray import inference_manager

    events = []

    def switch(phase):
        assert phase == "startup"
        events.append("release")
        if release_fails:
            raise TimeoutError("release unconfirmed")

    coordinator = SimpleNamespace(switch=SimpleNamespace(remote=switch))
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = SimpleNamespace(
        _inference_phase_coordinator=coordinator,
        _rollout_engine_manager="rollout",
        _genrm_managers={"judge": "judge"},
        _teacher_inference_managers={"teacher": "teacher"},
    )
    instance.serve_dict = {}
    monkeypatch.setattr(controller.ray, "get", lambda ref, **kwargs: ref)
    monkeypatch.setattr(controller.ray, "kill", lambda owner: events.append("kill coordinator"))
    monkeypatch.setattr(inference_manager, "rollback_static_inference_manager", events.append)
    if release_fails:
        with pytest.raises(TimeoutError):
            instance._dispose_shared_inference_task()
        assert events == ["release"]
        assert instance.config._inference_phase_coordinator is coordinator
    else:
        assert instance._dispose_shared_inference_task()
        assert events == ["release", "rollout"]
        assert not hasattr(instance.config, "_inference_phase_coordinator")


@pytest.mark.parametrize("remove_fails", [False, True])
def test_aborted_inference_pg_waits_for_task_cleanup_and_retains_retry(monkeypatch, remove_fails):
    from relax.distributed.ray import inference_manager

    events = []
    task = SimpleNamespace(switch=SimpleNamespace(remote=lambda phase: None))
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = SimpleNamespace(
        _rollout_engine_manager=task, _inference_phase_coordinator=task, _inference_startup_token="task-1"
    )
    instance.serve_dict = {}
    service = SimpleNamespace(
        pgs=("teacher-pg", [0], [0]),
        _is_shared_pgs=False,
        _deployed=False,
        _inference_deployment_token="task-1",
        _inference_task_owner=task,
    )
    failed = {"dispose": True, "remove": remove_fails}

    def cleanup(owner):
        assert owner is task
        events.append("dispose")
        if failed["dispose"]:
            raise TimeoutError("engine exit unconfirmed")

    def remove(pg):
        assert pg == "teacher-pg"
        events.append("remove")
        if failed["remove"]:
            raise TimeoutError("PG removal unconfirmed")

    monkeypatch.setattr(controller.ray, "get", lambda ref, **kwargs: ref)
    monkeypatch.setattr(inference_manager, "rollback_static_inference_manager", cleanup)
    monkeypatch.setattr(controller.serve, "delete", lambda role: events.append("delete"))
    monkeypatch.setattr(controller, "remove_placement_group", remove)
    instance._cleanup_aborted_service("teacher", service, deploy=True)
    assert events == [] and service.pgs is not None
    with pytest.raises(TimeoutError, match="engine exit"):
        instance._dispose_shared_inference_task()
    assert events == ["dispose"] and service.pgs is not None
    assert instance.config._rollout_engine_manager is task
    failed["dispose"] = False
    if remove_fails:
        with pytest.raises(TimeoutError, match="PG removal"):
            instance._dispose_shared_inference_task()
        assert service.pgs is not None
        assert instance._pending_inference_pg_services["teacher"] is service
        failed["remove"] = False
    assert instance._dispose_shared_inference_task()
    assert service.pgs is None and not instance._pending_inference_pg_services
    assert events.count("dispose") == 2
    assert events.index("remove") > events.index("dispose")


def test_late_old_service_cleanup_never_deletes_new_role_deployment(monkeypatch):
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = SimpleNamespace(_rollout_engine_manager=object(), _inference_startup_token="new")
    instance.serve_dict = {}
    instance._retired_inference_tokens = {"old"}
    instance._inference_cleanup_confirmed = True
    service = SimpleNamespace(
        pgs=("old-pg", [0], [0]), _is_shared_pgs=False, _deployed=False, _inference_deployment_token="old"
    )
    deleted = []
    removed = []
    monkeypatch.setattr(controller.serve, "delete", deleted.append)
    monkeypatch.setattr(controller, "remove_placement_group", removed.append)
    instance._cleanup_aborted_service("teacher", service, deploy=False)
    instance._release_pending_inference_services()
    assert deleted == []
    assert removed == ["old-pg"]


@pytest.mark.parametrize("failure_stage", ["prefetch", "prepare"])
def test_predeploy_failure_retains_all_pg_wrappers_until_task_cleanup(monkeypatch, failure_stage):
    from relax.distributed.ray import shared_task_startup

    instance = controller.Controller.__new__(controller.Controller)
    instance.config = SimpleNamespace(model_source=SimpleNamespace(uri="s3://bucket/model"))
    instance._shared_inference_startup = True
    instance.runtime_env = {}
    instance.serve_dict = {}
    rollout = SimpleNamespace(pgs=("rollout-pg", [0], [0]), _is_shared_pgs=False)
    teacher = SimpleNamespace(pgs=("teacher-pg", [0], [0]), _is_shared_pgs=False)
    owner = object()
    events = []
    cleanup_fails = {"value": True}

    def run(phase, task, task_args):
        assert phase is controller.ServiceStartupPhase.ALLOCATE_RESOURCES
        return {"rollout": rollout, "teacher": teacher}

    def get(refs, **kwargs):
        if failure_stage == "prefetch":
            raise RuntimeError("prefetch failed")
        return refs

    def prepare(config, *args, **kwargs):
        config._shared_startup_pending_owners = [owner]
        raise RuntimeError("teacher constructor failed")

    def cleanup(item):
        assert item is owner
        events.append("dispose")
        if cleanup_fails["value"]:
            raise TimeoutError("task cleanup unconfirmed")

    monkeypatch.setattr(instance, "_run_service_phase", run)
    monkeypatch.setattr(instance, "_start_model_prefetch", lambda bindings: ["prefetch-ref"])
    monkeypatch.setattr(controller.ray, "get", get)
    monkeypatch.setattr(shared_task_startup, "prepare_shared_task", prepare)
    monkeypatch.setattr(shared_task_startup, "rollback_static_inference_manager", cleanup)
    monkeypatch.setattr(controller.serve, "delete", lambda role: events.append("delete"))
    monkeypatch.setattr(controller, "remove_placement_group", lambda pg: events.append(pg))
    if failure_stage == "prefetch":
        with pytest.raises(RuntimeError, match="prefetch failed"):
            instance._start_services([])
        assert "dispose" not in events
    else:
        with pytest.raises(TimeoutError, match="cleanup unconfirmed"):
            instance._start_services([])
        assert events == ["dispose"]
        assert rollout.pgs is not None and teacher.pgs is not None
        assert instance._pending_inference_pg_services == {"rollout": rollout, "teacher": teacher}
        cleanup_fails["value"] = False
        assert instance._dispose_shared_inference_task()
    assert rollout.pgs is None and teacher.pgs is None
    assert not instance._pending_inference_pg_services
    assert "rollout-pg" in events and "teacher-pg" in events
