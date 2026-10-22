# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep Relax decode deadlines inside SGLang's native failure consensus."""

import ast
import os
from http import HTTPStatus
from importlib import metadata
from types import SimpleNamespace

import pytest
from packaging.version import Version

from tests.backends.sglang.source_utils import sglang_source_path


try:
    _sglang_version = metadata.version("sglang")
except metadata.PackageNotFoundError:
    pytest.skip("SGLang is not installed", allow_module_level=True)
pytestmark = pytest.mark.skipif(Version(_sglang_version) < Version("0.5.21"), reason="SGLang 0.5.21 PD queues")


def _load(relative_path: str, class_name: str, member_name: str | None = None, **namespace):
    path = sglang_source_path(relative_path)
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
    if member_name is not None:
        node = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == member_name)
        node.decorator_list = []
    # The source methods use runtime-only types; defer their annotations.
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[node.name]


def _transport(status, states):
    transport = SimpleNamespace(status=status, abort_count=0)
    transport.poll = lambda: transport.status

    def abort():
        transport.abort_count += 1
        transport.status = states.Failed

    transport.abort = abort
    return transport


@pytest.mark.parametrize("pp_size,consensus", [(1, "failed"), (2, "failed"), (2, "pending")])
def test_decode_deadline_aborts_receiver_before_consensus(pp_size: int, consensus: str, monkeypatch) -> None:
    states = _load("srt/disaggregation/base/conn.py", "KVPoll")
    monkeypatch.setenv("SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT", "100")
    receiver = _transport(states.Bootstrapping, states)
    req = SimpleNamespace(
        rid="request", bootstrap_room=1, time_stats=SimpleNamespace(decode_prealloc_queue_entry_time=1.0)
    )
    failures = []

    def collective(*_):
        assert receiver.abort_count == 1
        assert not failures
        return [None if consensus == "pending" else states.Failed]

    update = _load(
        "srt/disaggregation/decode.py",
        "DecodePreallocQueue",
        "_update_handshake_waiters",
        os=os,
        time=SimpleNamespace(perf_counter=lambda: 200.0),
        KVPoll=states,
        HTTPStatus=HTTPStatus,
        logger=SimpleNamespace(error=lambda *_: None, debug=lambda *_: None),
        poll_and_all_reduce=collective,
        poll_and_all_reduce_pp=collective,
        prepare_abort=lambda *args, **kwargs: failures.append(args[0]),
    )
    queue = SimpleNamespace(
        queue=[SimpleNamespace(req=req, kv_receiver=receiver)],
        pp_size=pp_size,
        tp_rank=0,
        gloo_group=None,
        scheduler=SimpleNamespace(metrics_reporter=SimpleNamespace(enable_metrics=False)),
    )
    update(queue, pp_good_rids=[], pp_bad_rids=[])
    assert failures == ([req] if consensus == "failed" else [])
