# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from concurrent.futures import Future
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from relax.distributed.replay_manager.backend.sglang import _install_response_patch
from relax.distributed.replay_manager.ref_data import ReplayRef


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
