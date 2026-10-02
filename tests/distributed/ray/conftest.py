# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared fixtures and helpers for autoscale tests."""

import os
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# Make this directory importable so test files can do ``from conftest import X``
sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Conditional import -- tests are skipped when dependencies are missing.
# ---------------------------------------------------------------------------
try:
    import ray

    from relax.distributed.ray.rollout import (
        EngineGroup,
        RolloutEnginePool,
        RolloutServer,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    RolloutEnginePool = None


# ---------------------------------------------------------------------------
# FakeOwnerHandle -- stands in for the task inference owner's actor handle.
# ---------------------------------------------------------------------------
class FakeOwnerHandle:
    """Record every ``owner.<method>.remote(...)`` call and answer the few
    whose results callers read.

    The results are plain values, so tests patch ``ray.get`` to return its
    argument unchanged.
    """

    def __init__(self, urls: dict | None = None) -> None:
        self.calls: list[tuple] = []
        self.urls = urls or {}

    def __getattr__(self, name: str) -> SimpleNamespace:
        if name.startswith("__"):
            raise AttributeError(name)
        return SimpleNamespace(remote=lambda *args, **kwargs: self._answer(name, args, kwargs))

    def _answer(self, name: str, args: tuple, kwargs: dict):
        self.calls.append((name, args, kwargs))
        if name == "create_role":
            return tuple(getattr(config, "name", config) for config, _, _ in args[1])
        if name == "call" and args[2] == "get_urls":
            return self.urls.get(args[1], [])
        return None

    def named(self, name: str) -> list[tuple]:
        return [(args, kwargs) for called, args, kwargs in self.calls if called == name]


# ---------------------------------------------------------------------------
# AwaitableValue -- a value that can be consumed by *both* ``ray.get`` (sync)
# and ``await`` (async), enabling the same mock engine to work in all contexts.
# ---------------------------------------------------------------------------
class AwaitableValue:
    """Wraps a plain value so it behaves like a Ray ObjectRef."""

    def __init__(self, value):
        self.value = value

    def __await__(self):
        yield  # make this a generator-based coroutine
        return self.value

    def __repr__(self):
        return f"AwaitableValue({self.value!r})"


def mock_ray_get(ref_or_list, timeout=None):
    """Drop-in replacement for ``ray.get`` that unwraps
    :class:`AwaitableValue`.

    Uses ``type().__name__`` instead of ``isinstance`` because pytest loads
    conftest.py once as a conftest plugin and test files may load it again
    via ``from conftest import ...``, creating two distinct ``AwaitableValue``
    classes.  ``isinstance`` would fail across the two copies.
    """
    if isinstance(ref_or_list, (list, tuple)):
        return [mock_ray_get(r) for r in ref_or_list]
    if type(ref_or_list).__name__ == "AwaitableValue":
        return ref_or_list.value
    return ref_or_list


# ---------------------------------------------------------------------------
# Mock engine factory
# ---------------------------------------------------------------------------
def make_mock_engine(
    url="http://localhost:30000",
    weight_version="v1",
    healthy=True,
    evicted=False,
):
    """Create a ``MagicMock`` that simulates an ``SGLangEngine`` Ray actor.

    Every ``.remote()`` call returns an :class:`AwaitableValue` so the mock
    works with both ``ray.get`` and ``await asyncio.gather``.
    """
    engine = MagicMock()
    engine.get_url.remote.return_value = AwaitableValue(url)
    engine.get_weight_version.remote.return_value = AwaitableValue(weight_version)
    engine.health_generate.remote.return_value = AwaitableValue(healthy)
    engine.register_dcs.remote.return_value = AwaitableValue(None)
    engine.unregister_dcs.remote.return_value = AwaitableValue(None)
    engine.register_to_router.remote.return_value = AwaitableValue(True)
    engine.unregister_from_router.remote.return_value = AwaitableValue(True)
    engine.shutdown.remote.return_value = AwaitableValue(None)
    engine.pause_generation.remote.return_value = AwaitableValue(None)
    engine.continue_generation.remote.return_value = AwaitableValue(None)
    engine.flush_cache.remote.return_value = AwaitableValue(None)
    engine.is_evicted.remote.return_value = AwaitableValue(evicted)
    engine.set_weight_updating.remote.return_value = AwaitableValue(None)
    engine.get_pid_and_node_id.remote.return_value = AwaitableValue({"pid": 123, "node_id": "node1"})
    engine.init_weights_send_group_for_remote_instance.remote.return_value = AwaitableValue({"success": True})
    engine.send_weights_to_remote_instance.remote.return_value = AwaitableValue({"success": True})
    _transport_fingerprint = {
        "nccl_ib_disable": "1",
        "nccl_socket_ifname": "eth0",
        "nccl_ib_hca": None,
        "nccl_ib_gid_index": None,
    }
    # Stage-1 cheap env gate: seed and new default to the same fingerprint (match).
    engine.get_scale_weight_sync_transport_fingerprint.remote.return_value = AwaitableValue(
        dict(_transport_fingerprint)
    )
    engine.run_scale_weight_sync_precheck.remote.return_value = AwaitableValue(
        {
            "success": True,
            "category": None,
            "error_type": None,
            "error": None,
            "fingerprint": dict(_transport_fingerprint),
            "results": [],
        }
    )
    engine._get_current_node_ip_and_free_port.remote.return_value = AwaitableValue(("127.0.0.1", 15000))
    return engine


# ---------------------------------------------------------------------------
# Mock args factory
# ---------------------------------------------------------------------------
def make_mock_args(**overrides):
    """Create a ``SimpleNamespace`` that mimics the training args structure."""
    defaults = dict(
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=8,
        scale_out_timeout=1800.0,
        scale_out_partial_success_policy="rollback_all",
        scale_in_drain_timeout=30.0,
        scale_in_shutdown_timeout=30.0,
        fully_async=True,
        use_fault_tolerance=False,
        debug_train_only=False,
        offload_rollout=False,
        sglang_pp_size=1,
        scale_out_max_concurrent_weight_syncs=4,
        hf_checkpoint="/tmp/test",
        rollout_health_check_interval=10,
        rollout_health_check_timeout=5,
        slime_router_health_check_failure_threshold=3,
        sglang_dp_size=1,
        rollout_external=False,
        ci_test=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Engine group / server factories
# ---------------------------------------------------------------------------
def make_engine_group(
    args=None,
    engines=None,
    num_gpus_per_engine=2,
    worker_type="regular",
    rank_offset=0,
    is_scaled_out=False,
    router_ip="127.0.0.1",
    router_port=3000,
):
    """Create an ``EngineGroup`` with mock engines."""
    if args is None:
        args = make_mock_args()
    if engines is None:
        engines = [make_mock_engine() for _ in range(max(1, num_gpus_per_engine // args.num_gpus_per_node))]
    return EngineGroup(
        args=args,
        pg=None,
        all_engines=engines,
        num_gpus_per_engine=num_gpus_per_engine,
        num_new_engines=0,
        worker_type=worker_type,
        rank_offset=rank_offset,
        is_scaled_out=is_scaled_out,
        router_ip=router_ip,
        router_port=router_port,
    )


def make_rollout_server(
    engine_groups=None,
    router_ip="127.0.0.1",
    router_port=3000,
    model_name="default",
):
    """Create a ``RolloutServer`` with the given (or default) engine groups."""
    if engine_groups is None:
        engine_groups = [make_engine_group()]
    return RolloutServer(
        engine_groups=engine_groups,
        router_ip=router_ip,
        router_port=router_port,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Testable RolloutManager factory
# ---------------------------------------------------------------------------
def create_test_manager(args=None, servers=None):
    """Create a ``RolloutEnginePool`` instance for testing.

    The pool owns the engines, placement and scaling state that these tests
    exercise; the Ray actor shell (``RolloutManager``) only forwards to it.
    Bypasses ``__init__`` entirely and sets up the minimal state needed by the
    scaling methods.
    """
    if RolloutEnginePool is None:
        pytest.skip("Cannot create test manager: dependencies missing")
    if args is None:
        args = make_mock_args()

    manager = object.__new__(RolloutEnginePool)
    manager.args = args
    manager.servers = servers if servers is not None else {}
    manager._scale_out_requests = {}
    manager._scale_in_requests = {}
    manager._engine_lifecycle_lock = threading.RLock()
    manager._training_weight_updating = False
    manager._scale_out_weight_updating = False
    manager._next_engine_rank = max(
        (
            group.rank_offset + len(group.all_engines)
            for srv in manager.servers.values()
            for group in srv.engine_groups
        ),
        default=0,
    )

    # Mock the distributed lock
    lock = MagicMock()
    lock.acquire.remote.return_value = AwaitableValue(True)
    lock.release.remote.return_value = AwaitableValue(None)
    manager._weight_sync_lock = lock

    manager._health_monitors = []
    manager._stranded_replicas = []
    manager._stranded_lock = threading.Lock()
    manager._stranded_retry_stop = threading.Event()
    manager._stranded_retry_thread = None
    manager._max_terminal_requests = 100
    manager._port_cursors = {}
    manager._eviction_monitor_stop = None
    manager._eviction_monitor_thread = None
    from relax.distributed.ray.inference_manager import InferenceManager
    from relax.engine.inference.config import InferenceModelSpec
    from relax.engine.inference.types import Role

    manager.status = None
    manager._inference_sync_pending = False
    manager.inference_manager = InferenceManager()
    manager._planner = manager.inference_manager.placement
    for name, server in manager.servers.items():
        server.model_name = name
        server.model_spec = server.model_spec or InferenceModelSpec(name, "test-checkpoint", elastic_enabled=True)
        server.ready_gate = manager._serving
    if manager.servers:
        with patch.object(ray, "get", mock_ray_get):
            manager.inference_manager.register(Role.ROLLOUT, manager.servers)
    return manager


# ---------------------------------------------------------------------------
# Shared pytest fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_args():
    return make_mock_args()


@pytest.fixture
def patch_ray_get():
    """Replace ``ray.get`` directly on the module to unwrap
    :class:`AwaitableValue`.

    Using ``patch("ray.get", ...)`` is unreliable when Ray is initialised
    because Ray may wrap the function internally.  Directly replacing the
    attribute on the module object is the safest approach.
    """
    _original = ray.get
    ray.get = mock_ray_get
    try:
        yield
    finally:
        ray.get = _original


@pytest.fixture
def patch_async_helpers():
    """Replace ``ray.get`` and patch ``asyncio.to_thread`` for async tests."""

    async def _mock_to_thread(func, *args):
        return func(*args)

    _original = ray.get
    ray.get = mock_ray_get
    try:
        with (
            patch("asyncio.to_thread", side_effect=_mock_to_thread),
            patch(
                "relax.distributed.ray.rollout.find_available_port",
                return_value=30000,
            ),
        ):
            yield
    finally:
        ray.get = _original
