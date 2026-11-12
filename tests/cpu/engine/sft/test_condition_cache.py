# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import gc
import weakref

import pytest
import torch

from relax.engine.sft.condition_cache import ConditionCache, ConditionFieldSpec, condition_key


def spec() -> ConditionFieldSpec:
    return ConditionFieldSpec("prefix", ("kv",), ("encoder",), ("prompt",), ("preprocess",), 4)


def key(index: int) -> str:
    return f"{index:064x}"


def test_storage_budget_views_aliases_lru_and_oversized() -> None:
    cache = ConditionCache((spec(),), torch.device("cpu"), 2, 128)
    storage = torch.arange(16, dtype=torch.float32)
    assert cache.put("prefix", key(1), {"kv": (storage[:1], storage[1:2])})
    assert cache.stats()["charged_bytes"] == 64
    assert cache.put("prefix", key(2), {"kv": (storage[2:3],)})
    assert cache.stats()["charged_bytes"] == 128
    restored = cache.get("prefix", key(1))
    assert restored["kv"][0].untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    assert torch.equal(restored["kv"][0], storage[:1])
    assert restored["kv"][0].dtype == storage.dtype
    assert not cache.put("prefix", key(9), {"kv": (torch.zeros(33),)})
    assert cache.stats()["entries"] == 2
    assert cache.put("prefix", key(3), {"kv": (torch.zeros(1),)})
    assert cache.get("prefix", key(2)) is None
    assert cache.get("prefix", key(1)) is not None
    assert cache.stats()["evictions"] == cache.stats()["oversized_skips"] == 1
    assert not cache.put("prefix", key(1), {"kv": (torch.ones(1),)})
    assert cache.stats()["charged_bytes"] == 68
    with pytest.raises(TypeError):
        restored["kv"] = ()


@pytest.mark.parametrize("change", ["value", "shape", "stride", "requires_grad", "storage"])
def test_corruption_removes_entry_and_fails(change: str) -> None:
    cache = ConditionCache((spec(),), torch.device("cpu"), 1, 128)
    cache.put("prefix", key(1), {"kv": (torch.arange(8.0).reshape(2, 4),)})
    value = cache.get("prefix", key(1))["kv"][0]
    if change == "value":
        value[1, 2].add_(1)
    elif change == "shape":
        value.resize_(8)
    elif change == "stride":
        value.transpose_(0, 1)
    elif change == "requires_grad":
        value.requires_grad_(True)
    else:
        value.set_(torch.ones(2, 4))
    with pytest.raises(ValueError):
        cache.get("prefix", key(1))
    assert cache.stats()["charged_bytes"] == cache.stats()["entries"] == 0


def test_empty_storage_structure_limits_and_release() -> None:
    cache = ConditionCache((spec(),), torch.device("cpu"), 1, 32)
    cache.put("prefix", key(1), {"kv": (torch.empty(0),)})
    assert cache.stats()["charged_bytes"] == 0
    with pytest.raises(ValueError, match="count"):
        cache.put("prefix", key(2), {"kv": (torch.empty(0),) * 5})
    with pytest.raises(ValueError, match="structure"):
        cache.put("prefix", key(2), {"kv": (((((torch.empty(0),),),),),)})
    with pytest.raises(ValueError, match="fields"):
        cache.put("prefix", key(2), {"other": torch.zeros(1)})
    with pytest.raises(ValueError, match="frozen"):
        cache.put("prefix", key(2), {"kv": torch.ones(1, requires_grad=True)})
    cache.put("prefix", key(2), {"kv": torch.ones(1)})
    reference = weakref.ref(cache.get("prefix", key(2))["kv"])
    cache.close()
    cache.close()
    gc.collect()
    assert reference() is None
    with pytest.raises(RuntimeError, match="closed"):
        cache.get("prefix", key(2))


@pytest.mark.parametrize("limit", [True, False, 0, -1, 2.5])
def test_cache_limits_reject_invalid_values(limit: object) -> None:
    with pytest.raises(ValueError):
        ConditionCache((spec(),), torch.device("cpu"), limit, 128)
    with pytest.raises(ValueError):
        ConditionCache((spec(),), torch.device("cpu"), 2, limit)


def test_key_canonicalization_dependencies_and_generation() -> None:
    metadata = {
        "weights": "source",
        "text": "tokenizer",
        "precision": "fp32",
        "prompt": "edit",
        "preprocess": [32, 64],
    }
    original = condition_key(spec(), metadata, 0)
    assert condition_key(spec(), dict(reversed(list(metadata.items()))), 0) == original
    assert condition_key(spec(), metadata, 1) != original
    for name in metadata:
        assert condition_key(spec(), {**metadata, name: "changed"}, 0) != original
        with pytest.raises(ValueError, match="Missing"):
            condition_key(spec(), {k: v for k, v in metadata.items() if k != name}, 0)
