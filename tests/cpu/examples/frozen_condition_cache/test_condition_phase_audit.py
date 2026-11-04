from types import SimpleNamespace

import pytest
import torch
from experiments.native_runner import ConditionPhaseAudit


class Pipe:
    def __init__(self):
        self.calls = 0

    def encode_prompt(self, value):
        self.calls += 1
        return value

    def unit_runner(self, unit, value):
        self.calls += 1
        if value == "fail":
            raise RuntimeError("original failure")
        return value


def test_qwen_existing_call_identity_rng_bounded_samples_and_restore(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda **_kwargs: pytest.fail("CPU audit must not create CUDA events"),
    )
    pipe = Pipe()
    original = pipe.encode_prompt
    audit = ConditionPhaseAudit(max_samples_per_phase=1)
    audit.install(SimpleNamespace(pipe=pipe), "qwen_image_edit", None)
    value = torch.ones(2)
    rng = torch.get_rng_state().clone()
    for _ in range(3):
        assert pipe.encode_prompt(value) is value
    record = audit.result()
    assert pipe.calls == 3
    assert record["phases"]["training/frozen_encoder"]["calls"] == 3
    assert record["phases"]["training/frozen_encoder"]["sampled_calls"] == 1
    assert record["phases"]["training/frozen_encoder"]["cuda_sample_seconds"] == []
    assert torch.equal(torch.get_rng_state(), rng)
    audit.close()
    audit.close()
    assert pipe.encode_prompt == original and "encode_prompt" not in vars(pipe)


def test_sensenova_only_declared_frozen_unit_and_exception_restore():
    pipe = Pipe()
    audit = ConditionPhaseAudit()
    audit.install(SimpleNamespace(pipe=pipe), "sensenova", None)
    value = object()
    assert pipe.unit_runner(object(), value) is value
    unit = SimpleNamespace(frozen_condition_schema="declared-v1")
    audit.region = "prefill"
    assert pipe.unit_runner(unit, value) is value
    audit.region = "training"
    with pytest.raises(RuntimeError, match="original failure"):
        pipe.unit_runner(unit, "fail")
    record = audit.result()["phases"]
    assert set(record) == {"prefill/frozen_encoder", "training/frozen_encoder"}
    assert record["training/frozen_encoder"]["failed_calls"] == 1
    audit.close()
    assert "unit_runner" not in vars(pipe)


def test_event_collection_is_bounded_and_never_adds_synchronize(monkeypatch):
    created = []

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}
            created.append(self)

        def record(self):
            pass

        def query(self):
            return True

        def elapsed_time(self, end):
            return 2.5

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: pytest.fail("No new synchronize allowed"))
    pipe = Pipe()
    audit = ConditionPhaseAudit(cuda=True, max_samples_per_phase=1)
    audit.install(SimpleNamespace(pipe=pipe), "qwen_image_edit", None)
    assert pipe.encode_prompt(3) == 3
    assert pipe.encode_prompt(4) == 4
    row = audit.result()["phases"]["training/frozen_encoder"]
    assert len(created) == 2 and row["calls"] == 2
    assert row["cuda_sample_seconds"] == [0.0025]
    audit.close()


def test_cache_actual_get_put_wrappers_are_restored():
    cache = SimpleNamespace(get=lambda key: key, put=lambda key, payload: payload)
    original_get, original_put = cache.get, cache.put
    audit = ConditionPhaseAudit()
    audit.install(SimpleNamespace(pipe=Pipe()), "qwen_image_edit", cache)
    payload = object()
    assert cache.put("one", payload) is payload and cache.get("one") == "one"
    assert set(audit.result()["phases"]) == {
        "training/cache_read",
        "training/cache_write",
    }
    audit.close()
    assert cache.get is original_get and cache.put is original_put
