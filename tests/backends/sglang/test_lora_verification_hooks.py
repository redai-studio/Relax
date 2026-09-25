# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate test-only injection boundaries; these are not GPU evidence."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def hooks(monkeypatch):
    monkeypatch.delenv("NO7_TEST_CONTROL_FILE", raising=False)
    path = Path(__file__).parent / "lora_test_hooks/sitecustomize.py"
    spec = importlib.util.spec_from_file_location("no7_test_hooks", path)
    value = importlib.util.module_from_spec(spec)
    before = list(sys.meta_path)
    spec.loader.exec_module(value)
    assert sys.meta_path == before
    return value


def test_wrong_kv_injection_does_not_select_the_wrong_weights(hooks, tmp_path, monkeypatch):
    class Request:
        def __init__(self, rid, uid):
            self.rid, self.lora_id = rid, uid
            self.extra_key = "namespace|lora_" + uid

    monkeypatch.setattr(
        hooks,
        "control",
        lambda: {"directory": str(tmp_path), "cache_alias": {"diagnostic": {"actual_uid": "B", "cached_uid": "A"}}},
    )
    module = SimpleNamespace(Req=Request)
    hooks.install_req(module)
    ordinary = Request("ordinary", "B")
    diagnostic = Request("diagnostic", "B")
    assert ordinary.extra_key == "namespace|lora_B"
    assert diagnostic.extra_key == "namespace|lora_A"
    assert diagnostic.lora_id == "B"
    assert json.loads((tmp_path / "diagnostic.alias.json").read_text())["actual_uid"] == "B"
    with pytest.raises(AssertionError, match="actual native adapter"):
        Request("diagnostic", "C")


@pytest.mark.parametrize("cache_class", ["RadixCache", "ChunkCache"])
def test_kv_capture_hook_is_inert_for_unselected_requests(hooks, cache_class):
    calls = []

    class Cache:
        def cache_finished_req(self, req, **kwargs):
            calls.append((req, kwargs))

    module = SimpleNamespace(**{cache_class: Cache})
    hooks.install_cache(module)
    req = SimpleNamespace(rid="ordinary")
    Cache().cache_finished_req(req, kv_len_to_handle=32)
    assert calls == [(req, {"kv_len_to_handle": 32})]


def test_selected_kv_capture_serializes_native_array_before_release(hooks, tmp_path, monkeypatch):
    from array import array

    torch = pytest.importorskip("torch", reason="selected KV capture regression requires CPU PyTorch")
    capture = {"layer": 0, "positions": [1, 3], "elements": [0, 5]}
    monkeypatch.setattr(hooks, "control", lambda: {"directory": str(tmp_path), "capture_kv": {"probe": capture}})
    monkeypatch.setitem(
        sys.modules, "sglang.srt.runtime_context", SimpleNamespace(get_parallel=lambda: SimpleNamespace())
    )
    key = torch.arange(12).reshape(4, 3)
    pool = SimpleNamespace(get_key_buffer=lambda _: key, get_value_buffer=lambda _: key + 100)
    evidence = tmp_path / "probe.kv.json"

    class Cache:
        req_to_token_pool = SimpleNamespace(req_to_token=torch.tensor([[3, 1, 0, 2]]))
        token_to_kv_pool_allocator = SimpleNamespace(get_kvcache=lambda: pool)

        def cache_finished_req(self, req, **kwargs):
            assert evidence.is_file(), "capture must precede native KV release"
            return "released"

    hooks.install_cache(SimpleNamespace(RadixCache=Cache))
    tokens = array("q", [10, 20, 30, 40])
    req = SimpleNamespace(rid="probe", lora_id="A", origin_input_ids=tokens, req_pool_idx=0)
    assert Cache().cache_finished_req(req, kv_len_to_handle=4) == "released"
    result = json.loads(evidence.read_text())
    assert result["input_ids"] == [10, 20, 30, 40]
    assert result["k"] == [3.0, 8.0] and result["v"] == [103.0, 108.0]
    assert req.origin_input_ids is tokens


@pytest.mark.parametrize("enabled", [False, True])
def test_prepare_audit_records_real_calls_only_when_enabled(hooks, tmp_path, monkeypatch, enabled):
    class Control:
        lora_version_control = SimpleNamespace(owner=("cohort", "boot"))

        async def prepare_lora_publication(self, payload):
            return {"actual": payload}

        def _reject_legacy_lora_mutation(self, operation):
            raise ValueError(operation)

        def bind_lora_publication_request(self, obj):
            pass

    monkeypatch.setattr(hooks, "control", lambda: {"directory": str(tmp_path), "capture_prepares": enabled})
    hooks.install_control(SimpleNamespace(Control=Control))
    payload = {"path": str(tmp_path / "B"), "native_lora_id": "instance-B"}
    assert asyncio.run(Control().prepare_lora_publication(payload)) == {"actual": payload}
    records = [
        json.loads(line) for path in tmp_path.glob("prepares-*.jsonl") for line in path.read_text().splitlines()
    ]
    assert records == ([{**payload, "boot": "boot"}] if enabled else [])


def test_mixed_batch_observation_is_test_only_and_uses_actual_pool(hooks, tmp_path, monkeypatch):
    records = [SimpleNamespace(lora_request_kind="business", rid=name, lora_id=name) for name in ("A", "B")]
    module = SimpleNamespace(
        begin_execution_batch=lambda scheduler, batch: records,
        finish_execution_batch=lambda *a: None,
        poll_version_retirements=lambda *a: False,
    )
    hooks.install_execution(module)
    # Empty controls require no pool access at all.
    assert module.begin_execution_batch(None, None) is records
    monkeypatch.setattr(hooks, "control", lambda: {"directory": str(tmp_path), "capture_mixed_batches": True})
    manager = SimpleNamespace(
        configs={"A": SimpleNamespace(r=8, lora_alpha=16), "B": SimpleNamespace(r=4, lora_alpha=4)},
        memory_pool=SimpleNamespace(uid_to_buffer_id={"A": 2, "B": 1}),
    )
    engine = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=SimpleNamespace(lora_manager=manager)))
    batch = SimpleNamespace(reqs=records, forward_mode=SimpleNamespace(is_decode=lambda: True))
    module.begin_execution_batch(engine, batch)
    proof = json.loads((tmp_path / "A.mixed.json").read_text())["decode"]
    assert proof["slot"] == 2 and proof["rank"] == 8 and proof["scaling"] == 2
    assert proof["members"] == ["A", "B"]
    assert not hasattr(records[0], "mixed_batches")  # No evidence retained in native owners.
    engine.ps = SimpleNamespace(tp_rank=1)
    manager.memory_pool.uid_to_buffer_id["A"] = 3
    module.begin_execution_batch(engine, batch)
    assert json.loads((tmp_path / "A.rank-1.mixed.json").read_text())["decode"]["slot"] == 3
    assert json.loads((tmp_path / "A.mixed.json").read_text())["decode"]["slot"] == 2


def test_mixed_batch_hook_leaves_ordinary_reference_requests_untouched(hooks, tmp_path, monkeypatch):
    ordinary = [SimpleNamespace(rid=name, lora_id=name) for name in ("A", "B")]
    module = SimpleNamespace(
        begin_execution_batch=lambda *args: (),
        finish_execution_batch=lambda *args: None,
        poll_version_retirements=lambda *args: False,
    )
    monkeypatch.setattr(hooks, "control", lambda: {"directory": str(tmp_path), "capture_mixed_batches": True})
    hooks.install_execution(module)
    batch = SimpleNamespace(reqs=ordinary)
    assert module.begin_execution_batch(SimpleNamespace(), batch) == ()
    assert not list(tmp_path.iterdir())
    assert all(not hasattr(req, "lora_request_kind") for req in ordinary)


def test_graph_evidence_requires_successful_execution(hooks, tmp_path, monkeypatch):
    class Runner:
        def execute(self, forward_batch, fail=False):
            if fail:
                raise RuntimeError("replay failed")
            return "output"

    hooks.install_graph(SimpleNamespace(DecodeCudaGraphRunner=Runner))
    monkeypatch.setattr(hooks, "control", lambda: {"directory": str(tmp_path), "capture_graph": True})
    batch = SimpleNamespace(lora_ids=[None, "instance-A", "instance-B"])
    with pytest.raises(RuntimeError, match="replay failed"):
        Runner().execute(batch, fail=True)
    assert not list(tmp_path.glob("*.graph.json"))
    assert Runner().execute(batch) == "output"
    assert Runner().execute(batch) == "output"
    assert {json.loads(path.read_text())["native_lora_id"] for path in tmp_path.glob("*.graph.json")} == {
        "instance-A",
        "instance-B",
    }


@pytest.mark.parametrize("overlap,pp,expected", [(True, 1, "forward"), (False, 1, "current"), (False, 2, "forward")])
def test_fault_injection_uses_actual_execution_stream(hooks, overlap, pp, expected):
    engine = SimpleNamespace(enable_overlap=overlap, ps=SimpleNamespace(pp_size=pp), forward_stream="forward")
    assert hooks.execution_stream(engine, SimpleNamespace(current_stream=lambda: "current")) == expected
