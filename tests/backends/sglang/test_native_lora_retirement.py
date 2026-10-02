# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Native pool/manager regression tests for a patched SGLang environment.

These import the production classes, without AST extraction. Controlled events
test reservation and cleanup ordering; they do not prove CUDA stream coverage
or scheduler request drain. Run in the training image after applying
sglang.patch.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest


@pytest.fixture
def native():
    torch = pytest.importorskip("torch", reason="native retirement tests require PyTorch")
    pool = pytest.importorskip("sglang.srt.lora.mem_pool", reason="requires patched SGLang environment")
    manager = pytest.importorskip("sglang.srt.lora.lora_manager", reason="requires patched SGLang environment")
    assert hasattr(pool.LoRAMemoryPool, "begin_remove_lora"), "SGLang does not contain the Relax retirement patch"
    return torch, pool, manager


@pytest.fixture
def retirement(native, monkeypatch):
    torch, pool_module, manager_module = native
    pool = pool_module.LoRAMemoryPool.__new__(pool_module.LoRAMemoryPool)
    pool.max_loras_per_batch = 2
    pool.uid_to_buffer_id = {"A-1": 0, "B-1": 1}
    pool.buffer_id_to_uid = ["A-1", "B-1"]
    pool.clearing_loras = {}
    pool.retiring_loras = set()
    pool.eviction_policy = pool_module.get_eviction_policy("lru")
    for uid in pool.uid_to_buffer_id:
        pool.eviction_policy.mark_used(uid)
    pool.A_buffer = {"q_proj": [torch.ones(2, 2, 4)]}
    pool.B_buffer = {"q_proj": [torch.ones(2, 4, 2)]}
    for name in (
        "embedding_A_buffer",
        "embedding_B_buffer",
        "lm_head_A_buffer",
        "lm_head_B_buffer",
        "new_embeddings_buffer",
    ):
        setattr(pool, name, {})
    log = []

    class Event:
        complete = False

        def record(self, stream):
            log.append("record")

        def query(self):
            return self.complete

    event = Event()
    stream = object()
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    manager = manager_module.LoRAManager.__new__(manager_module.LoRAManager)
    manager.device = torch.device("cpu")
    manager.memory_pool = pool
    manager.max_loras_per_batch = 2
    manager.retiring_loras = {}
    manager.pending_lora_load_events = {}
    manager.lora_refs = {
        uid: pool_module.LoRARef(lora_id=uid, lora_name=uid, lora_path=uid, pinned=True)
        for uid in pool.uid_to_buffer_id
    }
    manager.num_pinned_loras = 2
    manager.configs = {uid: object() for uid in pool.uid_to_buffer_id}
    manager.loras = {uid: object() for uid in pool.uid_to_buffer_id}

    def updated(slots):
        assert slots == {0}
        assert torch.count_nonzero(pool.A_buffer["q_proj"][0][0]) == 0
        log.append("notify")

    manager.lora_modules = [{"derived": SimpleNamespace(on_lora_slots_updated=updated)}]
    return SimpleNamespace(pool=pool, manager=manager, event=event, stream=stream, log=log, module=pool_module)


def test_native_retirement_preserves_slot_pin_and_cpu_weights_until_clear_finishes(retirement):
    r = retirement
    ref = r.manager.lora_refs["A-1"]
    assert r.manager.poll_unload_drained_lora(ref) is None
    assert r.log == ["notify", "record"]
    assert r.pool.uid_to_buffer_id["A-1"] == 0
    assert r.pool.buffer_id_to_uid[0] == "A-1"
    assert "A-1" in r.pool.eviction_policy.access_order
    assert "A-1" in r.manager.configs and "A-1" in r.manager.loras
    assert r.manager.num_pinned_loras == 2
    assert r.manager.poll_unload_drained_lora(ref) is None
    assert r.log == ["notify", "record"]
    r.event.complete = True
    assert r.manager.poll_unload_drained_lora(ref).success
    assert r.pool.buffer_id_to_uid[0] is r.module.EMPTY_SLOT
    assert "A-1" not in r.manager.configs and "A-1" not in r.manager.loras
    assert r.manager.num_pinned_loras == 1
    assert r.manager.poll_unload_drained_lora(ref).success
    assert r.manager.num_pinned_loras == 1


def test_native_retirement_cannot_be_evicted_used_or_bypassed(retirement):
    r = retirement
    r.pool.begin_remove_lora("A-1", r.stream)
    with pytest.raises(RuntimeError, match="pending"):
        r.pool.remove_lora("A-1")
    with pytest.raises(ValueError, match="being cleared"):
        r.pool.prepare_lora_batch({"A-1"}, {}, [], {}, None, None)
    # A is oldest and explicitly unpinned: CLEARING itself must exclude eviction.
    refs = {"A-1": SimpleNamespace(pinned=False), "B-1": SimpleNamespace(pinned=True)}
    with pytest.raises(ValueError, match="No available"):
        r.pool.prepare_lora_batch({"C-1"}, {}, [], refs, None, None)
    r.manager.lora_refs = refs
    r.manager.num_pinned_loras = 1
    assert not r.manager.validate_lora_batch({"C-1"})
    assert r.manager.validate_lora_batch({"B-1"})


@pytest.mark.parametrize("phase", ["clear", "notify", "record", "query"])
def test_native_retirement_error_keeps_reservation(retirement, monkeypatch, phase):
    r = retirement

    def fail(*args):
        raise RuntimeError("injected " + phase)

    if phase == "clear":
        monkeypatch.setattr(r.pool, "_clear_buffer_slot_for_base", fail)
    elif phase == "notify":
        monkeypatch.setattr(r.manager, "_notify_lora_slots_updated", fail)
    else:
        monkeypatch.setattr(r.event, phase, fail)
    ref = r.manager.lora_refs["A-1"]
    result = r.manager.poll_unload_drained_lora(ref)
    assert not result.success
    assert r.pool.buffer_id_to_uid[0] == "A-1"
    assert r.manager.num_pinned_loras == 2
    assert "A-1" in r.manager.configs
    # Retry cannot declare an unrecorded/failed clear complete.
    result = r.manager.poll_unload_drained_lora(ref)
    assert result is None or not result.success


def test_native_retirement_waits_for_pending_load_without_host_synchronization(retirement):
    r = retirement
    load = SimpleNamespace(query=lambda: False)
    r.manager.pending_lora_load_events["A-1"] = load
    r.manager.lora_refs["A-1"] = r.module.LoRARef(lora_id="A-1", lora_name="A-1", lora_path="A-1", pinned=False)
    r.manager.num_pinned_loras = 1
    ref = r.manager.lora_refs["A-1"]
    assert r.manager.poll_unload_drained_lora(ref) is None
    assert r.pool.clearing_loras == {}
    assert r.log == []
    assert r.pool.retiring_loras == {"A-1"}
    # A failed/partial load may not yet be pinned; its in-flight H2D still owns
    # the physical slot until it has completed and cleanup can begin.
    assert not r.manager.validate_lora_batch({"C-1"})
    with pytest.raises(ValueError, match="No available"):
        r.pool.prepare_lora_batch({"C-1"}, {}, [], r.manager.lora_refs, None, None)
    load.query = lambda: True
    assert r.manager.poll_unload_drained_lora(ref) is None
    assert r.log == ["notify", "record"]


def test_native_retirement_cleans_partial_load_and_rejects_wrong_instance(retirement):
    r = retirement
    ref = r.manager.lora_refs["A-1"]
    wrong = r.module.LoRARef(lora_id="A-1", lora_name="different", lora_path="different", pinned=True)
    assert not r.manager.poll_unload_drained_lora(wrong).success
    assert r.log == []
    # A load may fail after config creation, before registering/pinning.
    del r.manager.lora_refs["A-1"]
    del r.manager.loras["A-1"]
    r.manager.num_pinned_loras = 1
    assert r.manager.poll_unload_drained_lora(ref) is None
    r.event.complete = True
    assert r.manager.poll_unload_drained_lora(ref).success
    assert r.manager.num_pinned_loras == 1
    assert "A-1" not in r.manager.configs


@pytest.mark.parametrize("missing", [None, "config", "mapping", "reverse_mapping", "pin", "clearing", "scaling"])
def test_native_ready_residency_checks_exact_instance_and_slot(retirement, monkeypatch, missing):
    r = retirement
    config = SimpleNamespace(r=2, lora_alpha=4)
    r.manager.configs["A-1"] = config
    r.manager.loras["A-1"] = SimpleNamespace(config=config, scaling=2.0)
    r.manager.dtype = r.pool.dtype = "fixture-dtype"
    monkeypatch.setattr(r.pool, "can_support", lambda value: value is config)
    ref = r.manager.lora_refs["A-1"]
    if missing is None:
        r.manager.validate_managed_residency(ref)
        return
    if missing == "config":
        r.manager.configs.pop("A-1")
    elif missing == "mapping":
        r.pool.uid_to_buffer_id.pop("A-1")
    elif missing == "reverse_mapping":
        r.pool.buffer_id_to_uid[0] = "B-1"
    elif missing == "pin":
        r.manager.lora_refs["A-1"] = r.module.LoRARef(lora_id="A-1", pinned=False)
    elif missing == "clearing":
        r.pool.clearing_loras["A-1"] = r.event
    else:
        r.manager.loras["A-1"].scaling = float("nan")
    expected = "configuration mismatch" if missing == "scaling" else "ADAPTER_NOT_READY"
    with pytest.raises(ValueError, match=expected):
        r.manager.validate_managed_residency(ref)
