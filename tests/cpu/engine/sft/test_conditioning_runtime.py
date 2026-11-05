# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU coverage of conditioning reuse, collective decisions and ownership."""

from __future__ import annotations

import multiprocessing as mp
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist

from relax.engine.sft import conditioning
from relax.models.generative_sft import ConditioningSpec
from relax.utils.conditioning_cache import build_key


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.conditioning_cache = None


@pytest.fixture
def runtime_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """No model checkpoint or distributed cluster is needed by local tests."""
    monkeypatch.setattr(conditioning, "model_fingerprint", lambda _: "frozen-weights-v1")
    monkeypatch.setattr(conditioning.dist, "is_initialized", lambda: False)
    runtimes = []

    def make(*, enabled: bool = True):
        index = len(runtimes)
        monkeypatch.setenv("RELAX_CONDITION_CACHE", str(int(enabled)))
        monkeypatch.setenv("RELAX_CONDITION_CACHE_MAX_BYTES", str(4 * 1024**2))
        monkeypatch.setenv("RELAX_CONDITION_CACHE_DIR", str(tmp_path / "cache"))
        model = DummyModel()
        runtime = conditioning.ConditioningRuntime(
            model=model,
            model_path="unused-model-path",
            device=torch.device("cpu"),
            group=None,
            rank=0,
            namespace=f"test-run-{index}",
            local_batch_size=1,
        )
        runtimes.append(runtime)
        return runtime, model

    yield make
    for runtime in reversed(runtimes):
        runtime.close()


def _spec(*, cacheable: bool = True) -> ConditioningSpec:
    return ConditioningSpec("dummy.condition.v1", {"input": "same-content", "dtype": "bf16"}, cacheable)


def _restore(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if set(features) != {"value"}:
        raise ValueError("Unexpected conditioning fields")
    return {"condition": features["value"]}


def test_disabled_runtime_never_constructs_or_uses_cache(runtime_factory, tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Disabled runtime must not construct cache storage")

    monkeypatch.setattr(conditioning, "ConditioningCache", forbidden)
    runtime, model = runtime_factory(enabled=False)
    calls = []

    def compute():
        calls.append(1)
        return {"value": torch.arange(6, dtype=torch.bfloat16)}

    first = runtime.resolve(_spec(), compute, _restore)
    second = runtime.resolve(_spec(), compute, _restore)
    assert torch.equal(first["condition"], second["condition"])
    assert first["condition"].dtype == torch.bfloat16
    assert len(calls) == 2
    assert runtime.cache is None
    assert not (tmp_path / "cache").exists()
    assert model.conditioning_cache is runtime
    assert runtime._step["encoding_calls"] == 2
    assert runtime._step["effective_hit"] == 0


def test_cold_then_warm_returns_exact_bf16_without_reencoding(runtime_factory):
    runtime, _ = runtime_factory()
    expected = torch.tensor([0.0, -1.5, 2.25, 100.0], dtype=torch.bfloat16)
    calls = []

    def compute():
        calls.append(1)
        return {"value": expected.clone()}

    cold = runtime.resolve(_spec(), compute, _restore)
    warm = runtime.resolve(_spec(), compute, _restore)
    assert cold["condition"].dtype == warm["condition"].dtype == torch.bfloat16
    assert torch.equal(cold["condition"], expected)
    assert torch.equal(warm["condition"], expected)
    assert len(calls) == 1
    assert runtime._step["local_hit"] == runtime._step["effective_hit"] == 1
    assert runtime._step["encoding_calls"] == 1
    assert runtime._step["cache_read_bytes"] == expected.numel() * expected.element_size()


def test_cpu_hit_owns_storage_after_lease_closes(runtime_factory, monkeypatch):
    runtime, _ = runtime_factory()
    expected = torch.arange(8, dtype=torch.bfloat16)
    runtime.resolve(_spec(), lambda: {"value": expected}, _restore)
    lookup = runtime.cache.lookup
    borrowed_pointers = []

    def tracked_lookup(key):
        lease = lookup(key)
        borrowed_pointers.append(lease.data["value"].data_ptr())
        return lease

    monkeypatch.setattr(runtime.cache, "lookup", tracked_lookup)
    result = runtime.resolve(_spec(), lambda: pytest.fail("Expected a warm hit"), _restore)["condition"]
    assert torch.equal(result.view(torch.uint8), expected.view(torch.uint8))
    assert result.dtype == expected.dtype
    assert result.data_ptr() != borrowed_pointers[0], "Returned CPU data must not borrow a closed mmap lease"
    runtime.close()
    assert torch.equal(result, expected)


def _fake_cuda_events(runtime, monkeypatch, actions):
    runtime.device = torch.device("cuda:0")
    state = {"ready": False}

    class Event:
        def __init__(self, **_kwargs):
            pass

        def record(self, stream=None):
            actions.append("event_record")
            if state.pop("record_error", False):
                raise RuntimeError("event recording failed")

        def query(self):
            actions.append("event_query")
            if state.pop("query_error", False):
                raise RuntimeError("event query failed")
            return state["ready"]

    def synchronize(*_args, **_kwargs):
        actions.append("device_sync")
        if state.pop("sync_error", False):
            raise KeyboardInterrupt("cleanup interrupted")
        state["ready"] = True

    monkeypatch.setattr(conditioning.torch.cuda, "Event", Event)
    monkeypatch.setattr(conditioning.torch.cuda, "current_stream", lambda device: "copy-stream")
    monkeypatch.setattr(conditioning.torch.cuda, "synchronize", synchronize)
    return state


@pytest.mark.parametrize("failure", ["copy", "restore", "record"])
def test_cuda_copy_failure_retains_lease_until_close_and_unanimous_fallback(runtime_factory, monkeypatch, failure):
    runtime, _ = runtime_factory()
    actions = []
    state = _fake_cuda_events(runtime, monkeypatch, actions)
    state["record_error"] = failure == "record"
    expected = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)

    class Lease:
        data = {"value": expected}

        def copy_to(self, device):
            actions.append("copy_enqueued")
            if failure == "copy":
                raise RuntimeError("simulated partial H2D failure")
            return {"value": expected.clone(), "invalid_field": expected.clone()}, 2 * expected.nbytes

        def close(self):
            actions.append("lease_close")

    monkeypatch.setattr(runtime.cache, "lookup", lambda key: Lease())
    monkeypatch.setattr(runtime.cache, "put", lambda *_args: False)
    monkeypatch.setattr(runtime.cache, "invalidate", lambda key: actions.append("invalidate"))
    votes = []

    def vote(ready, group):
        actions.append("vote")
        votes.append(ready)
        return ready

    def compute():
        actions.append("compute")
        return {"value": expected.clone()}

    monkeypatch.setattr(conditioning, "unanimous_ready", vote)
    result = runtime.resolve(_spec(), compute, _restore)
    assert torch.equal(result["condition"], expected)
    assert votes == [False]
    assert "device_sync" not in actions and "lease_close" not in actions
    assert len(runtime._pending_copies) == 1
    assert (runtime._pending_copies[0][0] is None) == (failure == "record")
    assert actions.index("invalidate") < actions.index("vote") < actions.index("compute")
    assert runtime._step["cache_errors"] == 1
    assert (
        runtime._step["condition_h2d_bytes"]
        == (0 if failure == "copy" else 2) * expected.numel() * expected.element_size()
    )
    runtime.close()
    assert actions.index("compute") < actions.index("device_sync") < actions.index("lease_close")
    assert not runtime._pending_copies


def test_packed_copy_counts_actual_transfer_bytes_and_keeps_one_vote(runtime_factory, monkeypatch):
    runtime, _ = runtime_factory()
    actions, votes = [], []
    state = _fake_cuda_events(runtime, monkeypatch, actions)
    expected = torch.arange(4, dtype=torch.bfloat16)

    class Lease:
        data = {"value": expected}

        def copy_to(self, device):
            assert device == torch.device("cuda:0")
            actions.append("packed_copy")
            return {"value": expected.clone()}, 4096

        def close(self):
            actions.append("lease_close")

    def vote(ready, group):
        actions.append("vote")
        votes.append(ready)
        return ready

    monkeypatch.setattr(runtime.cache, "lookup", lambda key: Lease())
    monkeypatch.setattr(conditioning, "unanimous_ready", vote)
    result = runtime.resolve(_spec(), lambda: pytest.fail("Packed hit must skip online encoding"), _restore)
    assert torch.equal(result["condition"], expected)
    assert votes == [True]
    assert actions == ["packed_copy", "event_record", "vote"]
    assert runtime._step["condition_h2d_bytes"] == 4096
    assert runtime._step["cache_read_bytes"] == expected.numel() * expected.element_size()
    assert runtime._step["effective_hit"] == 1
    runtime._release_copies()
    assert "lease_close" not in actions
    state["ready"] = True
    runtime._release_copies()
    assert actions[-1] == "lease_close" and not runtime._pending_copies
    assert "device_sync" not in actions


@pytest.mark.parametrize("limit", ["count", "bytes", "unknown", "close"])
def test_pending_cuda_copies_bound_admission_without_invalidating(runtime_factory, monkeypatch, limit):
    runtime, _ = runtime_factory()
    actions, votes = [], []
    state = _fake_cuda_events(runtime, monkeypatch, actions)
    expected = torch.ones(4)
    if limit == "bytes":
        runtime._pending_copy_limit = 4
        runtime.cache.max_bytes = expected.nbytes
    elif limit in ("unknown", "close"):
        runtime._pending_copy_limit = 4

    class Lease:
        data = {"value": expected}

        def copy_to(self, device):
            actions.append("copy")
            return {"value": expected.clone()}, expected.nbytes

        def close(self):
            if state.pop("lease_error", False):
                raise OSError("lease close interrupted")
            actions.append("lease_close")

    monkeypatch.setattr(runtime.cache, "lookup", lambda key: Lease())
    monkeypatch.setattr(runtime.cache, "put", lambda *_args: pytest.fail("Bypass must not rewrite an existing entry"))
    monkeypatch.setattr(runtime.cache, "invalidate", lambda key: pytest.fail("Capacity is not corruption"))
    monkeypatch.setattr(conditioning, "unanimous_ready", lambda ready, group: votes.append(ready) or ready)
    runtime.resolve(_spec(), lambda: pytest.fail("First hit should fit"), _restore)
    state["query_error"] = limit == "unknown"
    state["ready"] = state["lease_error"] = limit == "close"
    runtime.resolve(_spec(), lambda: {"value": expected.clone()}, _restore)
    assert votes == [True, False] and actions.count("copy") == 1
    assert actions.count("lease_close") == 1  # Only the unsubmitted second lease.
    assert len(runtime._pending_copies) == 1
    assert "device_sync" not in actions
    runtime.end_step()
    assert actions.index("device_sync") < len(actions) - 1
    assert actions[-1] == "lease_close" and not runtime._pending_copies


@pytest.mark.parametrize("failure", ["sync", "lease_close"])
def test_pending_cuda_close_retries_before_closing_cache(runtime_factory, monkeypatch, failure):
    runtime, _ = runtime_factory()
    actions = []
    state = _fake_cuda_events(runtime, monkeypatch, actions)
    state["sync_error"] = failure == "sync"

    class Lease:
        data = {"value": torch.ones(1)}
        fail_close = failure == "lease_close"

        def copy_to(self, device):
            return {"value": self.data["value"].clone()}, 4

        def close(self):
            if self.fail_close:
                self.fail_close = False
                raise KeyboardInterrupt("cleanup interrupted")
            actions.append("lease_close")

    monkeypatch.setattr(runtime.cache, "lookup", lambda key: Lease())
    close = runtime.cache.close

    def close_cache():
        actions.append("cache_close")
        close()

    monkeypatch.setattr(runtime.cache, "close", close_cache)
    runtime.resolve(_spec(), lambda: pytest.fail("Expected hit"), _restore)
    with pytest.raises(KeyboardInterrupt, match="cleanup interrupted"):
        runtime.close()
    assert runtime._pending_copies and not runtime._closed
    assert "lease_close" not in actions and "cache_close" not in actions
    runtime.close()
    assert actions[-2:] == ["lease_close", "cache_close"]
    assert not runtime._pending_copies and runtime._closed


def test_local_hit_with_missing_peer_forces_online_but_preserves_entry(runtime_factory, monkeypatch):
    runtime, _ = runtime_factory()
    expected = torch.arange(4, dtype=torch.bfloat16)
    calls = []

    def compute():
        calls.append(1)
        return {"value": expected.clone()}

    runtime.resolve(_spec(), compute, _restore)
    runtime._step.clear()
    votes = []

    def peer_missing(ready: bool, group: Any) -> bool:
        votes.append((ready, group))
        return False

    monkeypatch.setattr(conditioning, "unanimous_ready", peer_missing)
    actual = runtime.resolve(_spec(), compute, _restore)
    assert votes == [(True, None)]
    assert torch.equal(actual["condition"], expected)
    assert len(calls) == 2
    assert runtime._step["local_hit"] == runtime._step["forced_recompute"] == 1
    assert runtime._step["effective_hit"] == 0
    monkeypatch.setattr(conditioning, "unanimous_ready", lambda ready, group: ready)
    runtime.resolve(_spec(), compute, _restore)
    assert len(calls) == 2


def test_compute_failure_propagates_and_runtime_can_close(runtime_factory):
    runtime, model = runtime_factory()

    def fail():
        raise RuntimeError("encoding failed")

    with pytest.raises(RuntimeError, match="encoding failed"):
        runtime.resolve(_spec(), fail, _restore)
    runtime.close()
    assert model.conditioning_cache is None


def test_invalid_restore_schema_is_invalidated_and_rebuilt_online(runtime_factory, monkeypatch):
    runtime, _ = runtime_factory()
    spec = _spec()
    key = build_key({"weights": runtime.fingerprint, "schema": spec.name, "dependencies": spec.dependencies})
    assert runtime.cache.put(key, {"wrong_field": torch.ones(3, dtype=torch.bfloat16)})
    invalidated = []
    original_invalidate = runtime.cache.invalidate

    def invalidate(key_to_remove):
        invalidated.append(key_to_remove)
        return original_invalidate(key_to_remove)

    monkeypatch.setattr(runtime.cache, "invalidate", invalidate)
    calls = []

    def compute():
        calls.append(1)
        return {"value": torch.tensor([1, 2, 3], dtype=torch.bfloat16)}

    actual = runtime.resolve(spec, compute, _restore)
    assert torch.equal(actual["condition"], torch.tensor([1, 2, 3], dtype=torch.bfloat16))
    assert invalidated == [key]
    assert runtime._step["cache_errors"] == 1
    runtime.resolve(spec, compute, _restore)
    assert len(calls) == 1, "A rebuilt schema must be reusable on the next lookup"


def test_uncacheable_conditions_preserve_fp32_gradient_and_cleanup(runtime_factory, monkeypatch):
    runtime, model = runtime_factory()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Trainable conditions must bypass cache reads and writes")

    monkeypatch.setattr(runtime.cache, "lookup", forbidden)
    monkeypatch.setattr(runtime.cache, "put", forbidden)
    inputs = torch.tensor([1.0, -3.0], dtype=torch.float32)
    result = runtime.resolve(_spec(cacheable=False), lambda: {"value": inputs * model.weight}, _restore)
    result["condition"].square().sum().backward()
    expected = 2 * model.weight.detach() * inputs.square().sum()
    assert result["condition"].requires_grad
    assert model.weight.grad.dtype == torch.float32
    assert torch.equal(model.weight.grad, expected)
    runtime.close()
    assert model.conditioning_cache is None
    assert torch.equal(model.weight.grad, expected), "Cleanup must not clear the optimizer's FP32 gradients"


@pytest.mark.parametrize("mutation", ["copy", "unfreeze_refreeze", "replace"])
def test_conditioning_mutation_permanently_bypasses_stale_entries(runtime_factory, monkeypatch, mutation):
    runtime, model = runtime_factory()
    model.weight.requires_grad_(False)

    def state():
        parameter = model.weight
        return {
            "frozen": not parameter.requires_grad,
            "parameters": {
                "prefix.weight": {
                    "identity": id(parameter),
                    "version": parameter._version,
                    "local_version": parameter._version,
                    "requires_grad": parameter.requires_grad,
                    "dtype": str(parameter.dtype),
                    "shape": tuple(parameter.shape),
                }
            },
        }

    model.conditioning_state = state
    calls = []

    def compute():
        calls.append(1)
        return {"value": torch.ones(3) * model.weight}

    runtime.begin_step()
    runtime.before_forward()
    runtime.resolve(_spec(), compute, _restore)
    runtime.end_step()
    runtime.begin_step()
    if mutation == "copy":
        with torch.no_grad():
            model.weight.copy_(torch.tensor(3.0))
    elif mutation == "replace":
        model.weight = torch.nn.Parameter(torch.tensor(3.0), requires_grad=False)
    else:
        model.weight.requires_grad_(True)
        runtime.before_forward()  # The training transition is observed before forward.
        with torch.no_grad():
            model.weight.add_(1)
        model.weight.requires_grad_(False)
    runtime.before_forward()
    first_reason = runtime._cache_guard_reason

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A changed encoder must neither load nor republish the old cache key")

    monkeypatch.setattr(runtime.cache, "lookup", forbidden)
    monkeypatch.setattr(runtime.cache, "put", forbidden)
    votes = []

    def vote(ready, group):
        votes.append(ready)
        return ready

    monkeypatch.setattr(conditioning, "unanimous_ready", vote)
    result = runtime.resolve(_spec(), compute, _restore)
    assert torch.equal(result["condition"], torch.ones(3) * 3)
    assert runtime._cache_guard_disabled and model.conditioning_reuse_disabled
    assert len(calls) == 2 and votes == [False]
    runtime.before_forward()
    runtime.resolve(_spec(), compute, _restore)
    assert runtime._cache_guard_reason == first_reason
    assert runtime._step["effective_hit"] == 0
    assert runtime._step["cache_errors"] == 0
    runtime.end_step()
    last = runtime.stats()
    assert last["cache_guard_disabled"]
    assert last["cache_guard_reason"] == first_reason
    assert "prefix.weight" in first_reason


def test_initial_trainable_conditioning_state_disables_reuse(runtime_factory):
    runtime, model = runtime_factory()
    model.conditioning_state = lambda: {
        "frozen": False,
        "parameters": {"prefix.weight": {"requires_grad": True}},
    }
    runtime.before_forward()
    assert runtime._cache_guard_disabled
    assert model.conditioning_reuse_disabled
    assert runtime._cache_guard_reason == "prefix.weight.requires_grad is true"


def test_nested_semantic_configuration_is_snapshotted_and_changes_disable_reuse(runtime_factory):
    runtime, model = runtime_factory()
    semantic = {"rope": {"scaling": [1.0]}, "mode": "frozen"}
    model.conditioning_state = lambda: {"frozen": True, "parameters": {}, "semantic": semantic}
    runtime.before_forward()
    expected = runtime.resolve(_spec(), lambda: {"value": torch.tensor([1.0])}, _restore)
    snapshot = runtime._conditioning_semantic_snapshot
    # A fresh declaration with different dictionary insertion order has the
    # same semantics and must continue hitting the existing entry.
    semantic = {"mode": "frozen", "rope": {"scaling": [1.0]}}
    runtime.before_forward()
    actual = runtime.resolve(_spec(), lambda: pytest.fail("Expected a hot condition"), _restore)
    assert torch.equal(actual["condition"], expected["condition"])
    assert not runtime._cache_guard_disabled
    semantic["rope"]["scaling"][0] = 2.0
    runtime.before_forward()
    assert runtime._conditioning_semantic_snapshot == snapshot
    actual = runtime.resolve(_spec(), lambda: {"value": torch.tensor([2.0])}, _restore)
    assert torch.equal(actual["condition"], torch.tensor([2.0]))
    assert runtime._cache_guard_disabled
    assert runtime._cache_guard_reason == "conditioning semantic configuration changed"
    semantic["rope"]["scaling"][0] = 1.0
    runtime.before_forward()
    assert runtime._cache_guard_disabled, "Returning to an earlier config must not revive the poisoned namespace"


@pytest.mark.parametrize("semantic", [{"tensor": torch.ones(1)}, {"invalid": float("nan")}, {1: "invalid-key"}])
def test_semantic_state_rejects_tensors_or_noncanonical_json_without_readback(runtime_factory, semantic):
    runtime, model = runtime_factory()
    model.conditioning_state = lambda: {"frozen": True, "parameters": {}, "semantic": semantic}
    runtime.before_forward()
    assert runtime._cache_guard_disabled
    assert runtime._cache_guard_reason.startswith("conditioning state inspection failed:")


def test_parameter_memory_format_change_without_identity_or_version_change_disables_reuse(
    runtime_factory, monkeypatch
):
    runtime, model = runtime_factory()
    model.encoder = torch.nn.Conv2d(3, 4, 3, bias=False).eval().requires_grad_(False)
    spec = ConditioningSpec("dummy.stride.v1", {"input": "same", "dtype": "float32"}, True)

    def state():
        parameter = model.encoder.weight
        return {
            "frozen": not parameter.requires_grad,
            "parameters": {
                "encoder.weight": {
                    "identity": id(parameter),
                    "version": parameter._version,
                    "local_version": parameter._version,
                    "requires_grad": parameter.requires_grad,
                    "dtype": str(parameter.dtype),
                    "shape": tuple(parameter.shape),
                    "stride": tuple(parameter.stride()),
                }
            },
        }

    model.conditioning_state = state
    calls = []

    def compute():
        calls.append(1)
        return {"value": model.encoder.weight.detach().clone()}

    runtime.before_forward()
    runtime.resolve(spec, compute, _restore)
    runtime.resolve(spec, compute, _restore)
    assert len(calls) == 1 and runtime._step["effective_hit"] == 1
    before = state()["parameters"]["encoder.weight"]
    model.encoder.to(memory_format=torch.channels_last)
    after = state()["parameters"]["encoder.weight"]
    # Module.to swaps .data for this conversion: ordinary version/identity
    # guards alone cannot detect the changed layout and resulting RNG mapping.
    assert before["identity"] == after["identity"] and before["version"] == after["version"]
    assert before["stride"] != after["stride"]
    runtime.before_forward()
    assert runtime._cache_guard_disabled
    assert runtime._cache_guard_reason == "encoder.weight.stride changed"

    def forbidden(*_args, **_kwargs):
        pytest.fail("A layout change must neither read nor republish the old content key")

    monkeypatch.setattr(runtime.cache, "lookup", forbidden)
    monkeypatch.setattr(runtime.cache, "put", forbidden)
    runtime.resolve(spec, compute, _restore)
    model.encoder.to(memory_format=torch.contiguous_format)
    runtime.before_forward()
    runtime.resolve(spec, compute, _restore)
    assert len(calls) == 3 and runtime._cache_guard_disabled
    assert runtime._cache_guard_reason == "encoder.weight.stride changed"


@pytest.mark.parametrize("failure", ["lookup", "put"])
def test_cache_io_errors_keep_online_result_and_downstream_fp32_gradient(runtime_factory, monkeypatch, failure):
    runtime, model = runtime_factory()

    def fail(*_args, **_kwargs):
        raise OSError("simulated cache I/O failure")

    monkeypatch.setattr(runtime.cache, failure, fail)
    expected = torch.tensor([1.25, -0.5], dtype=torch.float32)
    result = runtime.resolve(_spec(), lambda: {"value": expected.clone()}, _restore)
    (result["condition"] * model.weight).sum().backward()
    assert torch.equal(result["condition"], expected)
    assert model.weight.grad.dtype == torch.float32
    assert torch.equal(model.weight.grad, expected.sum())
    assert runtime._step["cache_errors"] == 1


def test_prefetched_lease_avoids_second_lookup_and_keeps_exact_dtype(runtime_factory, monkeypatch):
    monkeypatch.setenv("RELAX_CONDITION_PREFETCH_DEPTH", "2")
    runtime, _ = runtime_factory()
    expected = torch.arange(8, dtype=torch.bfloat16)
    runtime.resolve(_spec(), lambda: {"value": expected}, _restore)
    runtime.begin_step()
    lookup, calls = runtime.cache.lookup, []

    def tracked_lookup(key, **kwargs):
        calls.append(kwargs)
        return lookup(key, **kwargs)

    monkeypatch.setattr(runtime.cache, "lookup", tracked_lookup)
    assert runtime.prefetch_many([_spec()]) == 1
    actual = runtime.resolve(_spec(), lambda: pytest.fail("Warm prefetch must skip encoding"), _restore)["condition"]
    assert len(calls) == 1 and "max_bytes" in calls[0]
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert runtime._step["prefetch_hit"] == runtime._step["effective_hit"] == 1
    assert runtime._step["encoding_calls"] == 0
    assert runtime._prefetch.stats()["bytes"] == 0
    runtime.close()
    assert torch.equal(actual, expected), "The model must own its copy after the mmap/cache are closed"


def test_invalidated_encoder_never_consumes_prefetched_features(runtime_factory, monkeypatch):
    monkeypatch.setenv("RELAX_CONDITION_PREFETCH_DEPTH", "1")
    runtime, _ = runtime_factory()
    runtime.resolve(_spec(), lambda: {"value": torch.ones(4)}, _restore)
    assert runtime.prefetch_many([_spec()]) == 1
    runtime._prefetch._pool.submit(lambda: None).result(timeout=5)
    runtime.begin_step()
    runtime._cache_guard_disabled = True
    actual = runtime.resolve(_spec(), lambda: {"value": torch.full((4,), 2.0)}, _restore)
    assert torch.equal(actual["condition"], torch.full((4,), 2.0))
    assert runtime._step["prefetch_hit"] == runtime._step["effective_hit"] == 0
    assert runtime._step["encoding_calls"] == 1
    assert runtime._prefetch.stats()["bytes"] == 0


def test_prefetch_rekeys_actual_effective_input_before_using_a_future(runtime_factory, monkeypatch):
    monkeypatch.setenv("RELAX_CONDITION_PREFETCH_DEPTH", "1")
    runtime, _ = runtime_factory()
    runtime.resolve(_spec(), lambda: {"value": torch.ones(2)}, _restore)
    assert runtime.prefetch_many([_spec()]) == 1
    changed = ConditioningSpec("dummy.condition.v1", {"input": "changed-after-prefetch", "dtype": "bf16"}, True)
    runtime.begin_step()
    actual = runtime.resolve(changed, lambda: {"value": torch.full((2,), 3.0)}, _restore)
    assert torch.equal(actual["condition"], torch.full((2,), 3.0))
    assert runtime._step["prefetch_hit"] == 0 and runtime._step["encoding_calls"] == 1
    runtime.prefetch_many([])
    runtime._prefetch._pool.submit(lambda: None).result(timeout=5)
    assert runtime._prefetch.stats()["bytes"] == 0


def test_file_corrupted_after_prefetch_is_checked_again_before_use(runtime_factory, monkeypatch):
    monkeypatch.setenv("RELAX_CONDITION_PREFETCH_DEPTH", "1")
    runtime, _ = runtime_factory()
    expected = torch.arange(8, dtype=torch.bfloat16)
    runtime.resolve(_spec(), lambda: {"value": expected}, _restore)
    assert runtime.prefetch_many([_spec()]) == 1
    runtime._prefetch._pool.submit(lambda: None).result(timeout=5)
    path = next(runtime.cache.directory.glob("*.safetensors"))
    with path.open("r+b") as file:
        file.seek(-1, 2)
        original = file.read(1)
        file.seek(-1, 2)
        file.write(bytes([original[0] ^ 255]))
    runtime.begin_step()
    actual = runtime.resolve(_spec(), lambda: {"value": expected.clone()}, _restore)
    assert torch.equal(actual["condition"], expected)
    assert runtime._prefetch.stats()["stale"] == 1
    assert runtime._step["prefetch_hit"] == 0 and runtime._step["encoding_calls"] == 1


def test_prefetched_schema_failure_releases_budget_and_rebuilds_online(runtime_factory, monkeypatch):
    monkeypatch.setenv("RELAX_CONDITION_PREFETCH_DEPTH", "1")
    runtime, _ = runtime_factory()
    key = runtime._conditioning_key(_spec())
    assert runtime.cache.put(key, {"wrong_field": torch.ones(2)})
    assert runtime.prefetch_many([_spec()]) == 1
    expected = torch.arange(2, dtype=torch.bfloat16)
    actual = runtime.resolve(_spec(), lambda: {"value": expected.clone()}, _restore)
    assert torch.equal(actual["condition"], expected)
    assert runtime._step["cache_errors"] == 1
    assert runtime._step["effective_hit"] == 0 and runtime._step["encoding_calls"] == 1
    assert runtime._prefetch.stats()["bytes"] == 0
    with runtime.cache.lookup(key) as lease:
        assert set(lease.data) == {"value"}


def test_prefetch_disabled_with_online_runtime_and_close_drains_reader(runtime_factory, monkeypatch):
    monkeypatch.setenv("RELAX_CONDITION_PREFETCH_DEPTH", "2")
    online, _ = runtime_factory(enabled=False)
    assert not online.prefetch_enabled
    assert online.prefetch_many([_spec()]) == 0
    runtime, _ = runtime_factory()
    runtime.resolve(_spec(), lambda: {"value": torch.ones(2)}, _restore)
    assert runtime.prefetch_many([_spec()]) == 1
    runtime.close()
    stats = runtime._prefetch.stats()
    assert stats["closed"] and stats["bytes"] == stats["outstanding"] == 0


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("phase", ["prefetch", "cache"])
def test_interrupted_runtime_close_is_retryable(runtime_factory, monkeypatch, error_type, phase):
    monkeypatch.setenv("RELAX_CONDITION_PREFETCH_DEPTH", "1")
    runtime, model = runtime_factory()
    runtime.resolve(_spec(), lambda: {"value": torch.ones(2)}, _restore)
    assert runtime.prefetch_many([_spec()]) == 1
    resource = runtime._prefetch if phase == "prefetch" else runtime.cache
    original_close, calls = resource.close, []

    def interrupt_once():
        calls.append(1)
        if len(calls) == 1:
            raise error_type("cleanup interrupted")
        return original_close()

    monkeypatch.setattr(resource, "close", interrupt_once)
    with pytest.raises(error_type, match="cleanup interrupted"):
        runtime.close()
    assert not runtime._closed
    assert model.conditioning_cache is runtime
    runtime.close()
    runtime.close()
    assert runtime._closed
    assert model.conditioning_cache is None
    assert runtime._prefetch.stats()["drained"]
    assert runtime._prefetch.stats()["bytes"] == runtime._prefetch.stats()["outstanding"] == 0


def test_initialization_cleanup_does_not_mask_the_original_error(runtime_factory, monkeypatch, tmp_path):
    from relax.engine.sft import conditioning_prefetch

    monkeypatch.setenv("RELAX_CONDITION_PREFETCH_DEPTH", "1")
    close = conditioning.ConditioningCache.close

    def fail_prefetch(*args, **kwargs):
        raise ValueError("original initialization failure")

    def fail_cleanup(cache):
        close(cache)
        raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(conditioning_prefetch, "ConditioningPrefetch", fail_prefetch)
    monkeypatch.setattr(conditioning.ConditioningCache, "close", fail_cleanup)
    with pytest.raises(ValueError, match="original initialization failure"):
        runtime_factory()
    assert not list((tmp_path / "cache").rglob("client-*")), "Storage must still be cleaned"


def _gloo_cache_worker(rank: int, root: str, queue: Any) -> None:
    import os

    root = Path(root)
    os.environ["RELAX_CONDITION_CACHE"] = "1"
    os.environ["RELAX_CONDITION_CACHE_DIR"] = str(root / "cache")
    os.environ["RELAX_CONDITION_CACHE_MAX_BYTES"] = str(1024**2)
    os.environ["RELAX_CONDITION_PREFETCH_DEPTH"] = "0"
    dist.init_process_group(
        "gloo", init_method=f"file://{root / 'rendezvous'}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=15))
    runtime = None
    try:
        runtime = conditioning.ConditioningRuntime(
            model=DummyModel(),
            model_path=str(root / "model"),
            device=torch.device("cpu"),
            group=group,
            rank=rank,
            namespace=f"mixed-hits-{rank}",
            local_batch_size=1,
        )
        expected = torch.tensor([rank + 1, -2], dtype=torch.bfloat16)
        key = runtime._conditioning_key(_spec())
        if rank == 0:
            assert runtime.cache.put(key, {"value": expected})
        records = []
        for phase in range(3):
            runtime.begin_step()
            if phase == 2 and rank == 1:
                runtime.cache.invalidate(key)
                assert runtime.cache.put(key, {"wrong_field": expected})
            actual = runtime.resolve(_spec(), lambda: {"value": expected.clone()}, _restore)["condition"]
            assert torch.equal(actual, expected) and actual.dtype == torch.bfloat16
            marker = torch.tensor(float(rank + phase))
            dist.all_reduce(marker, group=group)
            assert marker.item() == 1 + 2 * phase
            records.append(
                (runtime._step["effective_hit"], runtime._step["encoding_calls"], runtime._step["cache_errors"])
            )
        queue.put((rank, records))
    finally:
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo is required")
def test_real_gloo_mixed_hits_and_restore_failure_preserve_collective_order(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "weights.safetensors").write_bytes(b"content fingerprint only")
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_gloo_cache_worker, args=(rank, str(tmp_path), queue)) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert not process.is_alive(), "Mixed cache results stranded a training rank"
            assert process.exitcode == 0
        assert sorted(queue.get(timeout=2) for _ in processes) == [
            (0, [(0, 1, 0), (1, 0, 0), (0, 1, 0)]),
            (1, [(0, 1, 0), (1, 0, 0), (0, 1, 1)]),
        ]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        queue.close()
        queue.join_thread()
