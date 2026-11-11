# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from relax.distributed.replay_manager.backend.sglang import (
    _install_owner_capture_patch,
    _install_response_patch,
    _install_tq_topology_cache,
)
from relax.distributed.replay_manager.ref_data import ReplayRef


def test_reference_non_owner_keeps_capture_but_skips_routing_snapshot(monkeypatch) -> None:
    collective_inputs = []

    class Capturer:
        def capture(self, routes: np.ndarray) -> None:
            # Stand in for the mandatory capture-side TP all-gather.
            collective_inputs.append(routes.copy())
            self.routes = routes.copy()

        def on_forward_end(self, *, no_copy_to_cpu: bool) -> dict:
            return {"topk": self.routes.copy(), "async": no_copy_to_cpu}

    module = ModuleType("sglang.srt.state_capturer.routed_experts")
    module.RoutedExpertsCapturer = Capturer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    is_owner = False
    _install_owner_capture_patch(lambda: is_owner)
    capturer = Capturer()
    routes = np.array([[0, 255], [17, 8]], dtype=np.int32)
    capturer.capture(routes)
    assert capturer.on_forward_end(no_copy_to_cpu=True) is None
    is_owner = True
    capturer.capture(routes)
    result = capturer.on_forward_end(no_copy_to_cpu=True)
    np.testing.assert_array_equal(result["topk"], routes)
    assert result["async"] is True
    assert len(collective_inputs) == 2
    for captured in collective_inputs:
        np.testing.assert_array_equal(captured, routes)


@pytest.mark.parametrize("component_output", [False, True], ids=["legacy", "component"])
def test_reference_response_waits_for_publication_and_drains_on_empty_input(monkeypatch, component_output):
    # Match SGLang's independent, slotted output component. It intentionally
    # has no scheduler fields, and the scheduler has no stream_output method.
    @dataclass(slots=True)
    class OutputStreamer:
        emitted: list

        def stream_output(self, reqs, return_logprob, skip_req=None):
            self.emitted.extend(req for req in reqs if req is not skip_req)

    class Scheduler:
        max_running_requests = 4

        def get_output_streamer_class(self):
            return OutputStreamer

        def process_input_requests(self, reqs):
            return "processed"

        def maybe_sleep_on_idle(self):
            return "idle"

    if not component_output:
        Scheduler.stream_output = lambda self, reqs, logprob, skip=None: self.emitted.extend(
            req for req in reqs if req is not skip
        )
    module = ModuleType("sglang.srt.managers.scheduler")
    module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert _install_response_patch(lambda scheduler: True)
    scheduler = Scheduler()
    scheduler.emitted = []
    if component_output:
        scheduler.output_streamer = scheduler.get_output_streamer_class()(scheduler.emitted)
        emit = scheduler.output_streamer.stream_output
    else:
        emit = scheduler.stream_output

    publication = Future()
    req = SimpleNamespace(_relax_replay_finish={"moe": publication}, _relax_replay_streams={"moe": object()})
    skipped = SimpleNamespace()
    emit([req, skipped], True, skipped)
    assert not scheduler.emitted
    ref = ReplayRef(2, "a", "a", "0" * 32, "moe", 4, 3, 2, 2, "0" * 64)
    publication.set_result(ref)
    assert scheduler.process_input_requests([]) == "processed"
    assert scheduler.emitted == [req]
    assert req.routed_experts == ref.to_dict()
    assert req._relax_replay_finish is None and req._relax_replay_streams == {}
    assert scheduler.maybe_sleep_on_idle() == "idle"
    assert not scheduler._relax_replay_pending


def test_reference_tq_topology_cache_preserves_thread_restore_and_fallback(monkeypatch) -> None:
    import psutil
    import torch
    import transfer_queue
    import transfer_queue.storage
    import transfer_queue.utils

    queries, threads, fallback = [], [], []

    def physical_cores(*, logical: bool) -> int:
        queries.append(logical)
        return 16

    @contextmanager
    def original(target_num_threads=None, info=""):
        fallback.append((target_num_threads, info))
        yield

    client = ModuleType("transfer_queue.client")
    client.limit_pytorch_auto_parallel_threads = original
    common = ModuleType("transfer_queue.utils.common")
    common.DEFAULT_TORCH_NUM_THREADS = 24
    storage = ModuleType("transfer_queue.storage.simple_storage")
    storage.limit_pytorch_auto_parallel_threads = original
    monkeypatch.setitem(sys.modules, storage.__name__, storage)
    monkeypatch.setattr(transfer_queue.storage, "simple_storage", storage)
    monkeypatch.setitem(sys.modules, client.__name__, client)
    monkeypatch.setitem(sys.modules, common.__name__, common)
    monkeypatch.setattr(transfer_queue, "client", client)
    monkeypatch.setattr(transfer_queue.utils, "common", common)
    monkeypatch.setattr(psutil, "cpu_count", physical_cores)
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    _install_tq_topology_cache()
    _install_tq_topology_cache()
    for module in (client, storage):
        with module.limit_pytorch_auto_parallel_threads(8):
            pass
    with pytest.raises(RuntimeError, match="body failed"):
        with client.limit_pytorch_auto_parallel_threads(8):
            raise RuntimeError("body failed")
    for target in (None, 32):
        with client.limit_pytorch_auto_parallel_threads(target, info="fallback"):
            pass
    assert queries == [False]
    assert threads == [8, 24, 8, 24, 8, 24]
    assert fallback == [(None, "fallback"), (32, "fallback")]
