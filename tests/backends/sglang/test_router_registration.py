# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import logging
import sys
import threading
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

    def post(self, url, json=None, timeout=None):
        self.posts.append(url)
        headers = {"Location": self.location} if self.location is not None else {}
        payload = {"worker_id": self.worker_id}
        if self.body_location is not None:
            payload["location"] = self.body_location
        return _Response(202, payload, headers)

    def delete(self, url, timeout=None):
        self.deletes.append(url)
        return _Response(self.delete_status)

    def get(self, url, timeout=None):
        if self.worker_lists is None:
            raise AssertionError("registration-level worker_id should avoid listing DP rank workers")
        self.gets.append(url)
        if not url.endswith("/workers"):
            return _Response(200, {})
        workers = self.worker_lists[0] if len(self.worker_lists) == 1 else self.worker_lists.pop(0)
        return _Response(200, {"workers": workers})


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
    engine._evicted = threading.Event()
    return engine


def test_registration_rejects_eviction_before_rpc_execution(monkeypatch, sglang_engine_module):
    requests = _RouterRequests()
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)
    engine._evicted.set()
    assert engine.register_to_router() is False
    assert requests.posts == []


def test_dead_worker_cleanup_confirms_removal_and_is_idempotent(monkeypatch, sglang_engine_module):
    old = {"id": "old", "url": "http://worker:8000", "is_healthy": False, "metadata": {"relax_actor_id": "A"}}
    other = {"id": "other", "url": "http://other:8000", "is_healthy": False}
    requests = _RouterRequests(worker_lists=[[old, other], [other]])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    cleanup = sglang_engine_module.remove_dead_router_worker
    state = sglang_engine_module.RouterWorkerCleanup("A")
    assert cleanup("http://router:30000", old["url"], state)
    assert cleanup("http://router:30000", old["url"], state)
    assert state.worker_id == "old"
    assert requests.deletes == ["http://router:30000/workers/old"]


@pytest.mark.parametrize(
    "rows",
    [
        [{"id": "live", "url": "http://worker:8000", "is_healthy": True}],
        [{"id": "dp", "url": "http://worker:8000@0", "is_healthy": False}],
        [{"id": "unknown", "url": "http://worker:8000"}],
        [
            {"id": "first", "url": "http://worker:8000", "is_healthy": False},
            {"id": "second", "url": "http://worker:8000", "is_healthy": False},
        ],
    ],
)
def test_dead_worker_cleanup_refuses_ambiguous_or_live_owner(monkeypatch, sglang_engine_module, rows):
    requests = _RouterRequests(worker_lists=[rows])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    state = sglang_engine_module.RouterWorkerCleanup("A")
    assert not sglang_engine_module.remove_dead_router_worker("http://router:30000", "http://worker:8000", state)
    assert requests.deletes == []


def test_dead_worker_cleanup_timeout_retains_id_and_does_not_delete_replacement(monkeypatch, sglang_engine_module):
    old = {"id": "old", "url": "http://worker:8000", "is_healthy": False, "metadata": {"relax_actor_id": "A"}}
    requests = _RouterRequests(delete_status=503, worker_lists=[[old]])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    cleanup = sglang_engine_module.remove_dead_router_worker
    state = sglang_engine_module.RouterWorkerCleanup("A")
    assert not cleanup("http://router:30000", old["url"], state)
    assert state.worker_id == "old" and state.delete_requested
    # The DELETE may have taken effect despite the failed response. Another
    # owner, including an unhealthy new generation, must not be selected again.
    replacement = {**old, "id": "new"}
    requests.worker_lists = [[replacement]]
    assert not cleanup("http://router:30000", old["url"], state)
    assert requests.deletes == ["http://router:30000/workers/old"]


def test_dead_worker_cleanup_wait_is_bounded(monkeypatch, sglang_engine_module):
    old = {"id": "old", "url": "http://worker:8000", "is_healthy": False, "metadata": {"relax_actor_id": "A"}}
    requests = _RouterRequests(worker_lists=[[old]])
    clock = _Clock()
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    monkeypatch.setattr(sglang_engine_module, "time", clock)
    state = sglang_engine_module.RouterWorkerCleanup("A")
    assert not sglang_engine_module.remove_dead_router_worker("http://router:30000", old["url"], state, timeout=1)
    assert clock.now <= 1


@pytest.mark.parametrize("worker_id", ["old", "different"])
def test_dead_worker_cleanup_never_selects_another_actor_generation(monkeypatch, sglang_engine_module, worker_id):
    row = {"id": worker_id, "url": "http://worker:8000", "is_healthy": False, "metadata": {"relax_actor_id": "C"}}
    requests = _RouterRequests(worker_lists=[[row]])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    state = sglang_engine_module.RouterWorkerCleanup("A")
    assert not sglang_engine_module.remove_dead_router_worker("http://router:30000", row["url"], state)
    assert requests.deletes == [] and state.worker_id is None


def test_dead_worker_cleanup_lost_delete_response_is_polled_not_reenqueued(monkeypatch, sglang_engine_module):
    old = {"id": "old", "url": "http://worker:8000", "is_healthy": False, "metadata": {"relax_actor_id": "A"}}
    requests = _RouterRequests(worker_lists=[[old]])
    clock = _Clock()
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    monkeypatch.setattr(sglang_engine_module, "time", clock)

    def lost_response(url, timeout):
        requests.deletes.append(url)
        raise TimeoutError("removal queued by URL; response lost")

    monkeypatch.setattr(requests, "delete", lost_response)
    state = sglang_engine_module.RouterWorkerCleanup("A")
    cleanup = sglang_engine_module.remove_dead_router_worker
    assert not cleanup("http://router:30000", old["url"], state, timeout=1)
    assert not cleanup("http://router:30000", old["url"], state, timeout=1)
    assert len(requests.deletes) == 1
    # Original asynchronous removal finally completes. No second queued
    # removal remains that could later remove a replacement at this URL.
    requests.worker_lists = [[]]
    assert cleanup("http://router:30000", old["url"], state)
    requests.worker_lists = [[{**old, "id": "new", "metadata": {"relax_actor_id": "B"}}]]
    assert not cleanup("http://router:30000", old["url"], state)
    assert len(requests.deletes) == 1


@pytest.mark.parametrize("failure", ["connect_timeout", "router_rejection"])
def test_dead_worker_cleanup_retries_only_definitely_unaccepted_delete(monkeypatch, sglang_engine_module, failure):
    old = {"id": "old", "url": "http://worker:8000", "is_healthy": False, "metadata": {"relax_actor_id": "A"}}
    requests = _RouterRequests(worker_lists=[[old]])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)

    def reject(url, timeout):
        if failure == "connect_timeout":
            raise sglang_engine_module.ConnectTimeout("not connected")
        return _Response(500, {"code": "INTERNAL_SERVER_ERROR", "error": "Job queue not initialized"})

    monkeypatch.setattr(requests, "delete", reject)
    state = sglang_engine_module.RouterWorkerCleanup("A")
    cleanup = sglang_engine_module.remove_dead_router_worker
    assert not cleanup("http://router:30000", old["url"], state)
    assert not state.delete_requested
    monkeypatch.setattr(requests, "delete", lambda url, timeout: _Response(202))
    requests.worker_lists = [[old], []]
    assert cleanup("http://router:30000", old["url"], state)


def test_native_registration_attaches_actual_actor_identity(monkeypatch, sglang_engine_module):
    requests = _RouterRequests()
    payloads = []
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: payloads.append(json) or _Response(202))
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    monkeypatch.setattr(
        sglang_engine_module.ray, "get_runtime_context", lambda: SimpleNamespace(get_actor_id=lambda: "A")
    )
    engine = _make_engine(sglang_engine_module)
    engine._is_scaled_out = True
    assert engine.register_to_router()
    assert payloads[0]["labels"] == {"relax_actor_id": "A"}


@pytest.mark.parametrize("status", ["pending", "processing", "failed"])
def test_dead_worker_cleanup_does_not_duplicate_actor_shutdown(monkeypatch, sglang_engine_module, status):
    old = {"id": "old", "url": "http://worker:8000", "is_healthy": False, "metadata": {"relax_actor_id": "A"}}
    requests = _RouterRequests(worker_lists=[[old], []])
    original_get = requests.get

    def get(url, timeout):
        if url.endswith("/old"):
            return _Response(200, {"job_status": {"job_type": "RemoveWorker", "status": status}})
        return original_get(url, timeout)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    state = sglang_engine_module.RouterWorkerCleanup("A")
    assert sglang_engine_module.remove_dead_router_worker("http://router:30000", old["url"], state)
    assert state.delete_requested
    assert requests.deletes == []


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
    assert requests.gets == []


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
