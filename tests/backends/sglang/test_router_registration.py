# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from relax.utils.http_utils import router_worker_base_url


@pytest.fixture()
def sglang_engine_module(monkeypatch):
    ray = ModuleType("ray")
    ray.get_runtime_context = lambda: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "ray", ray)

    sglang_router = ModuleType("sglang_router")
    sglang_router.__version__ = "0.3.2"
    monkeypatch.setitem(sys.modules, "sglang_router", sglang_router)

    sglang = ModuleType("sglang")
    sglang_srt = ModuleType("sglang.srt")
    server_args = ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = object
    sglang_utils = ModuleType("sglang.srt.utils")
    sglang_utils.kill_process_tree = lambda _pid: None
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", sglang_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", sglang_utils)

    checkpoint_client = ModuleType("relax.distributed.checkpoint_service.client.engine")
    checkpoint_client.create_client = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "relax.distributed.checkpoint_service.client.engine", checkpoint_client)

    ray_actor = ModuleType("relax.distributed.ray.ray_actor")
    ray_actor.RayActor = object
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.ray_actor", ray_actor)

    device = ModuleType("relax.utils.device")
    device.get_visible_devices_env_var = lambda: "CUDA_VISIBLE_DEVICES"
    monkeypatch.setitem(sys.modules, "relax.utils.device", device)

    async_utils = ModuleType("relax.utils.async_utils")
    async_utils.run = lambda value: value
    monkeypatch.setitem(sys.modules, "relax.utils.async_utils", async_utils)

    env = ModuleType("relax.utils.env")
    env.Envs = SimpleNamespace(
        RELAX_SCALE_OUT_MAX_REASON_ITEMS=3,
        RELAX_SCALE_OUT_MAX_REASON_ITEM_LEN=120,
        RELAX_SCALE_OUT_MAX_REASON_TOTAL_LEN=512,
    )
    monkeypatch.setitem(sys.modules, "relax.utils.env", env)

    http_utils = ModuleType("relax.utils.http_utils")
    http_utils.get_host_info = lambda: ("worker", "127.0.0.1")
    http_utils.router_worker_base_url = router_worker_base_url
    monkeypatch.setitem(sys.modules, "relax.utils.http_utils", http_utils)

    logging_utils = ModuleType("relax.utils.logging_utils")
    logging_utils.get_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "relax.utils.logging_utils", logging_utils)

    megatron_peft_utils = ModuleType("relax.utils.megatron_peft_utils")
    megatron_peft_utils.convert_megatron_to_sglang_target_modules = lambda value: value
    megatron_peft_utils.is_lora_enabled = lambda _args: False
    monkeypatch.setitem(sys.modules, "relax.utils.megatron_peft_utils", megatron_peft_utils)

    # Force a fresh import so the module binds to the stubbed dependencies above,
    # but restore the original module object on teardown. Leaving the key popped
    # corrupts sys.modules for any later test that patches this module: their
    # patch() re-imports a *new* module object distinct from the one already
    # bound in other test files' top-level imports, so the patch silently misses.
    original_module = sys.modules.pop("relax.backends.sglang.sglang_engine", None)
    module = importlib.import_module("relax.backends.sglang.sglang_engine")
    yield module
    if original_module is not None:
        sys.modules["relax.backends.sglang.sglang_engine"] = original_module
    else:
        sys.modules.pop("relax.backends.sglang.sglang_engine", None)


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")

    def json(self):
        return self._payload


class _RouterRequests:
    def __init__(
        self,
        delete_status: int = 202,
        worker_lists: list[list[dict]] | None = None,
        worker_id: str | int | None = "physical-worker-id",
        location: str | None = None,
        body_location: str | None = None,
    ):
        self.delete_status = delete_status
        self.worker_lists = worker_lists
        self.worker_id = worker_id
        self.location = location
        self.body_location = body_location
        self.posts: list[str] = []
        self.deletes: list[str] = []
        self.gets: list[str] = []
        self.registered: list[dict] = []

    def post(self, url, json=None, timeout=None):
        self.posts.append(url)
        self.registered.append({"url": json["url"]})
        headers = {"Location": self.location} if self.location is not None else {}
        payload = {"worker_id": self.worker_id}
        if self.body_location is not None:
            payload["location"] = self.body_location
        return _Response(202, payload, headers)

    def delete(self, url, timeout=None):
        self.deletes.append(url)
        return _Response(self.delete_status)

    def get(self, url, timeout=None):
        self.gets.append(url)
        if self.worker_lists is None:
            # Registration confirms membership; unregistering by worker_id never lists.
            return _Response(200, {"workers": list(self.registered)})
        workers = self.worker_lists[0] if len(self.worker_lists) == 1 else self.worker_lists.pop(0)
        return _Response(200, {"workers": workers})


def _router_get(post, weight_version=None):
    """A ``requests.get`` for one worker: its weight version, and the Router
    listing it once ``post`` registered it."""

    def get(url, timeout=None):
        if url.endswith("/model_info"):
            return _Response(200, {"weight_version": weight_version})
        return _Response(200, {"workers": [{"url": "http://worker:8000"}] if post.called else []})

    return get


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _make_engine(sglang_engine_module):
    engine = sglang_engine_module.SGLangEngine.__new__(sglang_engine_module.SGLangEngine)
    engine.args = SimpleNamespace(use_slime_router=False)
    engine.node_rank = 0
    engine.worker_type = "regular"
    engine.router_ip = "router"
    engine.router_port = 30000
    engine.server_host = "worker"
    engine.server_port = 8000
    engine._router_worker_id = None
    engine._router_unregister_submitted = False
    return engine


def test_missing_load_format_choices_fails_closed(sglang_engine_module):
    with pytest.raises(RuntimeError, match="cannot report whether runai_streamer is supported"):
        sglang_engine_module._preferred_s3_stream_load_format()


def test_unregister_uses_registration_worker_id_once(monkeypatch, sglang_engine_module):
    requests = _RouterRequests()
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert engine.register_to_router()
    assert engine._router_worker_id == "physical-worker-id"
    assert engine.unregister_from_router()
    assert engine.unregister_from_router()

    assert requests.posts == ["http://router:30000/workers"]
    assert requests.deletes == ["http://router:30000/workers/physical-worker-id"]


def test_registration_normalizes_numeric_worker_id(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(worker_id=42)
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert engine.register_to_router()
    assert engine._router_worker_id == "42"
    assert engine.unregister_from_router()

    assert requests.deletes == ["http://router:30000/workers/42"]


@pytest.mark.parametrize(
    ("worker_id", "location", "body_location", "expected_worker_id"),
    [
        (None, "http://router:30000/workers/header-worker-id", None, "header-worker-id"),
        ("", None, "/workers/body-worker-id", "body-worker-id"),
    ],
)
def test_registration_falls_back_to_location_worker_id(
    monkeypatch,
    sglang_engine_module,
    worker_id,
    location,
    body_location,
    expected_worker_id,
):
    requests = _RouterRequests(worker_id=worker_id, location=location, body_location=body_location)
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert engine.register_to_router()
    assert engine._router_worker_id == expected_worker_id
    assert engine.unregister_from_router()

    assert requests.deletes == [f"http://router:30000/workers/{expected_worker_id}"]
    # Only registration lists: once before posting, once to confirm the join.
    assert requests.gets == ["http://router:30000/workers"] * 2


def test_unregister_falls_back_to_exact_non_dp_worker(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(worker_lists=[[{"id": "listed-worker-id", "url": "http://worker:8000"}]])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert engine.unregister_from_router()

    assert requests.deletes == ["http://router:30000/workers/listed-worker-id"]


def test_unregister_does_not_delete_dp_rank_worker_as_fallback(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(worker_lists=[[{"id": "rank-worker-id", "url": "http://worker:8000@0"}]])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert not engine.unregister_from_router()

    assert requests.deletes == []


def test_unregister_rejects_ambiguous_exact_and_dp_rank_workers(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(
        worker_lists=[
            [
                {"id": "exact-worker-id", "url": "http://worker:8000"},
                {"id": "rank-worker-id", "url": "http://worker:8000@0"},
            ]
        ]
    )
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert not engine.unregister_from_router()

    assert requests.deletes == []


def test_unregister_can_retry_after_worker_is_not_listed_yet(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(
        worker_lists=[
            [],
            [{"id": "listed-worker-id", "url": "http://worker:8000"}],
        ]
    )
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert not engine.unregister_from_router()
    assert engine._router_unregister_submitted is False
    assert engine.unregister_from_router()

    assert requests.deletes == ["http://router:30000/workers/listed-worker-id"]


def test_unregister_treats_missing_registration_as_complete(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(delete_status=404)
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)
    engine._router_worker_id = "physical-worker-id"

    assert engine.unregister_from_router()
    assert requests.deletes == ["http://router:30000/workers/physical-worker-id"]


def test_unregister_waits_for_all_dp_rank_workers_to_leave(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(
        worker_lists=[
            [{"url": "http://worker:8000@0"}, {"url": "http://worker:8000@1"}],
            [],
        ]
    )
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    monkeypatch.setattr(sglang_engine_module, "time", _Clock())
    engine = _make_engine(sglang_engine_module)
    engine._router_worker_id = "physical-worker-id"

    assert engine.unregister_from_router(wait_for_removal=True, timeout=5.0)
    assert requests.deletes == ["http://router:30000/workers/physical-worker-id"]
    assert requests.gets == [
        "http://router:30000/workers",
        "http://router:30000/workers",
    ]


def test_unregister_timeout_allows_a_later_retry(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(worker_lists=[[{"url": "http://worker:8000@0"}]])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    monkeypatch.setattr(sglang_engine_module, "time", _Clock())
    engine = _make_engine(sglang_engine_module)
    engine._router_worker_id = "physical-worker-id"

    assert not engine.unregister_from_router(wait_for_removal=True, timeout=1.0)
    assert engine._router_unregister_submitted is False


@pytest.mark.parametrize(
    "node_rank,fully_async,override", [(0, True, 4), (0, True, None), (1, True, 4), (0, False, 4)]
)
def test_engine_dcs_registration_eligibility_and_gpu_metadata(
    monkeypatch, sglang_engine_module, node_rank, fully_async, override
):
    from unittest.mock import MagicMock

    engine = _make_engine(sglang_engine_module)
    engine.node_rank = node_rank
    engine.rank = 2
    engine.num_gpus_per_engine = override
    engine.args = SimpleNamespace(
        fully_async=fully_async, rollout_num_gpus_per_engine=2, coordinator_url="http://dcs.test"
    )
    engine.checkpoint_engine_client = None
    create_client = MagicMock(return_value=object())
    monkeypatch.setattr(sglang_engine_module, "create_client", create_client)

    engine.register_dcs()

    if node_rank == 0 and fully_async:
        create_client.assert_called_once_with(
            args=engine.args,
            coordinator_url="http://dcs.test",
            role="rollout",
            ip="worker",
            port=8000,
            rank=2,
            metadata={"num_gpus_per_engine": override or 2},
        )
        assert engine.checkpoint_engine_client is create_client.return_value
    else:
        create_client.assert_not_called()
        assert engine.checkpoint_engine_client is None


@pytest.mark.parametrize("skip_dcs", [False, True])
def test_engine_startup_precedes_dcs_registration(monkeypatch, sglang_engine_module, skip_dcs):
    engine = _make_engine(sglang_engine_module)
    engine.args = SimpleNamespace(sglang_router_ip="router", sglang_router_port=30000, rollout_external=False)
    engine.rank = 0
    engine.base_gpu_id = 0
    engine.sglang_overrides = {}
    engine.num_gpus_per_engine = 2
    events = []
    monkeypatch.setattr(
        sglang_engine_module,
        "_compute_server_args",
        lambda *a, **kw: ({"node_rank": 0, "host": "worker", "port": 8000}, []),
    )
    engine._init_normal = lambda args: events.append("started")
    engine.register_dcs = lambda: events.append("dcs")

    engine.init("worker:8001", 8000, 8002, skip_router_registration=True, skip_dcs_registration=skip_dcs)

    assert events == (["started"] if skip_dcs else ["started", "dcs"])
    assert engine._skip_router_registration is True


def test_static_engine_rejects_policy_mutations(sglang_engine_module):
    from unittest.mock import MagicMock

    module = sglang_engine_module
    engine = module.SGLangEngine(SimpleNamespace(), rank=0, weight_source="static")
    engine._make_request = MagicMock()
    operations = [
        lambda: engine.register_dcs(),
        lambda: engine.update_weights_from_tensor([]),
        lambda: engine.init_weights_update_group("host", 1, 0, 1, "weights", "nccl"),
        lambda: engine.update_weights_from_distributed([], [], [], "weights"),
        lambda: engine.load_lora_adapter_from_tensors("policy", "", {}),
        lambda: engine.update_lora_from_distributed("policy", [], [], [], {}, "weights"),
        lambda: engine.unload_lora_adapter("policy"),
        lambda: engine.init_weights_send_group_for_remote_instance("host", [], 0, 1),
        lambda: engine.send_weights_to_remote_instance("host", []),
        lambda: engine.post_process_weights(),
    ]
    for operation in operations:
        with pytest.raises(RuntimeError, match="Policy weight updates are forbidden"):
            operation()
    engine._make_request.assert_not_called()


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("genrm", [False, True])
def test_static_engine_uses_common_startup_without_policy_load_plan(
    monkeypatch, sglang_engine_module, external, genrm
):
    from unittest.mock import MagicMock

    module = sglang_engine_module
    engine = module.SGLangEngine(
        SimpleNamespace(rollout_external=external, sglang_router_ip="", sglang_router_port=0),
        rank=0,
        weight_source="static",
        role="genrm" if genrm else "teacher",
    )
    compute = MagicMock(return_value=({"node_rank": 0, "host": "[::1]", "port": 8000}, ["model_path"]))
    monkeypatch.setattr(module, "_compute_genrm_server_args" if genrm else "_compute_server_args", compute)
    engine._init_normal = MagicMock()
    engine._init_external = MagicMock()
    engine.register_dcs = MagicMock()

    engine.init("::1:8001", 8000, 8002, host="::1")

    assert compute.call_args.args[2] == "[::1]:8001"
    assert engine.server_host == "[::1]"
    assert engine.checkpoint_engine_client is None
    engine.register_dcs.assert_not_called()
    if external:
        engine._init_external.assert_called_once_with(
            compute.return_value[0], external_engine_need_check_fields=["model_path"]
        )
        engine._init_normal.assert_not_called()
    else:
        engine._init_normal.assert_called_once_with(compute.return_value[0], apply_policy_load_plan=False)
        engine._init_external.assert_not_called()
    # Every role takes the same startup path; what a static role skips is
    # decided by its adapter's init kwargs, not by the engine.
    assert engine._skip_router_registration is False


def test_engine_defaults_to_rollout_role_with_dcs_weights(sglang_engine_module):
    engine = sglang_engine_module.SGLangEngine(SimpleNamespace(), rank=0)
    assert engine.weight_source == sglang_engine_module.WeightSource.DCS
    assert engine.role == sglang_engine_module.Role.ROLLOUT


def test_genrm_engine_registers_its_model_router_without_dcs(monkeypatch, sglang_engine_module):
    from unittest.mock import MagicMock

    module = sglang_engine_module
    engine = module.SGLangEngine(SimpleNamespace(rollout_external=False), rank=0, weight_source="static", role="genrm")
    monkeypatch.setattr(
        module,
        "_compute_genrm_server_args",
        lambda *args, **kwargs: ({"node_rank": 0, "host": "worker", "port": 8000}, []),
    )
    engine._init_normal = MagicMock()
    engine.register_dcs = MagicMock()
    engine.init("worker:8001", 8000, 8002, router_ip="judge-router", router_port=3100)
    assert engine.router_ip == "judge-router"
    assert engine.router_port == 3100
    assert engine._skip_router_registration is False
    engine.register_dcs.assert_not_called()


@pytest.mark.parametrize("worker_type", ["regular", "prefill", "decode"])
def test_router_registration_preserves_pd_bootstrap_payload(monkeypatch, sglang_engine_module, worker_type):
    from unittest.mock import MagicMock

    engine = _make_engine(sglang_engine_module)
    engine.worker_type = worker_type
    post = MagicMock(return_value=_Response(200, {"worker_id": "worker-id"}))
    monkeypatch.setattr(sglang_engine_module.requests, "post", post)
    monkeypatch.setattr(sglang_engine_module.requests, "get", _router_get(post))

    assert engine.register_to_router(bootstrap_port=9000)

    payload = {"url": "http://worker:8000", "worker_type": worker_type}
    if worker_type == "prefill":
        payload["bootstrap_port"] = 9000
    post.assert_called_once_with("http://router:30000/workers", json=payload, timeout=30)


@pytest.mark.parametrize("version", [None, "", "default"])
def test_inference_observation_policy_unknown_version_does_not_register(monkeypatch, sglang_engine_module, version):
    from unittest.mock import MagicMock

    engine = _make_engine(sglang_engine_module)
    engine.health_generate = MagicMock(return_value=True)
    get = MagicMock(return_value=_Response(200, {"weight_version": version}))
    post = MagicMock()
    monkeypatch.setattr(sglang_engine_module.requests, "get", get)
    monkeypatch.setattr(sglang_engine_module.requests, "post", post)

    observation = engine.get_inference_observation(ensure_router=True)

    assert observation == {
        "base_url": "http://worker:8000",
        "healthy": True,
        "router_registered": False,
        "weight_version": version,
    }
    engine.health_generate.assert_called_once_with(timeout=5.0)
    get.assert_called_once_with("http://worker:8000/model_info", timeout=5.0)
    post.assert_not_called()


@pytest.mark.parametrize("ensure_router", [False, True])
def test_inference_observation_policy_valid_version_registers_when_requested(
    monkeypatch, sglang_engine_module, ensure_router
):
    from unittest.mock import MagicMock

    engine = _make_engine(sglang_engine_module)
    engine.health_generate = MagicMock(return_value=True)
    post = MagicMock(return_value=_Response(200, {"worker_id": "worker-id"}))
    get = MagicMock(side_effect=_router_get(post, "v1"))
    monkeypatch.setattr(sglang_engine_module.requests, "get", get)
    monkeypatch.setattr(sglang_engine_module.requests, "post", post)

    observation = engine.get_inference_observation(ensure_router=ensure_router)

    assert observation["healthy"] is True
    assert observation["weight_version"] == "v1"
    assert observation["router_registered"] is ensure_router
    assert get.call_args_list[0] == (("http://worker:8000/model_info",), {"timeout": 5.0})
    if ensure_router:
        post.assert_called_once_with(
            "http://router:30000/workers",
            json={"url": "http://worker:8000", "worker_type": "regular"},
            timeout=30,
        )
        engine.get_inference_observation(ensure_router=True)
        assert post.call_count == 1
    else:
        post.assert_not_called()


def test_inference_observation_checkpoint_registers_without_weight_version(monkeypatch, sglang_engine_module):
    from unittest.mock import MagicMock

    engine = _make_engine(sglang_engine_module)
    engine.weight_source = sglang_engine_module.WeightSource.STATIC
    engine.health_generate = MagicMock(return_value=True)
    post = MagicMock(return_value=_Response(200, {"worker_id": "worker-id"}))
    get = MagicMock(side_effect=_router_get(post))
    monkeypatch.setattr(sglang_engine_module.requests, "get", get)
    monkeypatch.setattr(sglang_engine_module.requests, "post", post)

    observation = engine.get_inference_observation(ensure_router=True)

    assert observation["healthy"] is True
    assert observation["weight_version"] is None
    assert observation["router_registered"] is True
    assert all(not call.args[0].endswith("/model_info") for call in get.call_args_list)
    post.assert_called_once_with(
        "http://router:30000/workers",
        json={"url": "http://worker:8000", "worker_type": "regular"},
        timeout=30,
    )


@pytest.mark.parametrize("weight_source", ["dcs", "static"])
@pytest.mark.parametrize("failure", ["http_error", "connection_error"])
def test_inference_observation_registration_failure_has_no_ready_evidence(
    monkeypatch, sglang_engine_module, weight_source, failure
):
    from unittest.mock import MagicMock

    engine = _make_engine(sglang_engine_module)
    engine.weight_source = sglang_engine_module.WeightSource(weight_source)
    engine.health_generate = MagicMock(return_value=True)
    post = MagicMock(return_value=_Response(503))
    if failure == "connection_error":
        post.side_effect = sglang_engine_module.requests.exceptions.ConnectionError("router unavailable")
    get = MagicMock(side_effect=_router_get(post, "v1"))
    monkeypatch.setattr(sglang_engine_module.requests, "get", get)
    monkeypatch.setattr(sglang_engine_module.requests, "post", post)

    observation = engine.get_inference_observation(ensure_router=True)

    assert observation["healthy"] is True
    assert observation["weight_version"] == ("v1" if weight_source == "dcs" else None)
    assert observation["router_registered"] is False
    assert engine._router_registered is False
    post.assert_called_once()


@pytest.mark.parametrize("external", [False, True])
def test_engine_shutdown_respects_external_ownership(monkeypatch, sglang_engine_module, external):
    engine = _make_engine(sglang_engine_module)
    engine.args.rollout_external = external
    events = []
    engine.unregister_from_router = lambda: events.append("router")
    engine.unregister_dcs = lambda: events.append("dcs")
    engine.process = SimpleNamespace(pid=42)
    monkeypatch.setattr(sglang_engine_module, "kill_process_tree", lambda pid: events.append(("kill", pid)))
    try:
        engine.shutdown()
        assert events == ([] if external else ["router", ("kill", 42)])
    finally:
        del engine.process  # Do not invoke the destructor's process cleanup in this unit test.


def test_inference_observation_skips_generation_probe_until_memory_is_resumed(monkeypatch, sglang_engine_module):
    from unittest.mock import MagicMock

    constants = ModuleType("sglang.srt.constants")
    constants.GPU_MEMORY_ALL_TYPES = ["kv_cache", "weights", "cuda_graph"]
    monkeypatch.setitem(sys.modules, "sglang.srt.constants", constants)
    engine = _make_engine(sglang_engine_module)
    engine.role = sglang_engine_module.Role.ROLLOUT
    engine.weight_source = sglang_engine_module.WeightSource.DCS
    engine.flush_cache = MagicMock()
    engine._make_request = MagicMock()
    engine.health_generate = MagicMock(return_value=True)
    monkeypatch.setattr(
        sglang_engine_module.requests, "get", MagicMock(return_value=_Response(200, {"weight_version": "v1"}))
    )

    engine.release_memory_occupation()
    engine.resume_memory_occupation(tags=["weights"])
    # Weights are back but the KV cache is not: probing generation crashes SGLang.
    assert engine.get_inference_observation()["healthy"] is False
    engine.health_generate.assert_not_called()

    engine.resume_memory_occupation(tags=["kv_cache", "cuda_graph"])
    assert engine.get_inference_observation()["healthy"] is True
    engine.health_generate.assert_called_once_with(timeout=5.0)


def test_registration_waits_until_router_lists_the_worker(monkeypatch, sglang_engine_module):
    # The addition is queued; the Router lists the worker only on the third poll.
    requests = _RouterRequests(worker_lists=[[], [], [], [{"url": "http://worker:8000"}]])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    monkeypatch.setattr(sglang_engine_module, "time", _Clock())
    engine = _make_engine(sglang_engine_module)

    assert engine.register_to_router()
    assert engine._router_registered is True
    assert requests.posts == ["http://router:30000/workers"]


def test_registration_that_never_joins_is_not_registered(monkeypatch, sglang_engine_module):
    # e.g. the Router cannot reach the engine, so its queued addition fails silently.
    requests = _RouterRequests(worker_lists=[[]])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    monkeypatch.setattr(sglang_engine_module, "time", _Clock())
    engine = _make_engine(sglang_engine_module)

    assert not engine.register_to_router()
    assert engine._router_registered is False


def test_registration_retry_does_not_repost_a_joined_worker(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(worker_lists=[[{"url": "http://worker:8000"}]])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert engine.register_to_router()
    assert engine._router_registered is True
    assert requests.posts == []
