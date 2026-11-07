# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import hashlib
import inspect
import sys
from types import SimpleNamespace

import pytest

from relax.backends.megatron import compat


class LegacyOptimizer:
    def __init__(self):
        self.buffers = [
            SimpleNamespace(
                buckets=[
                    SimpleNamespace(offset=0, numel_unpadded=4),
                    SimpleNamespace(offset=8, numel_unpadded=3),
                    SimpleNamespace(offset=16, numel_unpadded=2),
                ],
                param_index_map={"first": (0, 4, 0), "middle": (8, 11, 1), "last": (16, 18, 2)},
            )
        ]

    def sharded_param_state_fully_reshardable(self, fail=False):
        # Coalesced data from the gather excludes both preceding buckets' padding.
        if fail:
            raise RuntimeError("serialization failed")
        packed_states = {
            name: list(range(base, base + 9)) for name, base in [("param", 0), ("exp_avg", 10), ("exp_avg_sq", 20)]
        }
        return {
            parameter: {name: values[start:end] for name, values in packed_states.items()}
            for parameter, (start, end, _) in self.buffers[0].param_index_map.items()
        }


def test_fully_reshardable_checkpoint_offsets_preserve_all_optimizer_values(monkeypatch):
    class Optimizer(LegacyOptimizer):
        pass

    monkeypatch.setitem(
        sys.modules, "megatron.core.optimizer.distrib_optimizer", SimpleNamespace(DistributedOptimizer=Optimizer)
    )
    original = Optimizer.sharded_param_state_fully_reshardable
    optimizer = Optimizer()
    indices = optimizer.buffers[0].param_index_map
    assert not compat.patch_fully_reshardable_checkpoint_padding()
    assert Optimizer.sharded_param_state_fully_reshardable is original
    monkeypatch.setattr(
        compat, "_LEGACY_FULLY_RESHARDABLE", hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    )

    assert compat.patch_fully_reshardable_checkpoint_padding()
    assert not compat.patch_fully_reshardable_checkpoint_padding()
    assert optimizer.sharded_param_state_fully_reshardable() == {
        "first": {"param": [0, 1, 2, 3], "exp_avg": [10, 11, 12, 13], "exp_avg_sq": [20, 21, 22, 23]},
        "middle": {"param": [4, 5, 6], "exp_avg": [14, 15, 16], "exp_avg_sq": [24, 25, 26]},
        "last": {"param": [7, 8], "exp_avg": [17, 18], "exp_avg_sq": [27, 28]},
    }
    assert optimizer.buffers[0].param_index_map is indices
    with pytest.raises(RuntimeError, match="serialization failed"):
        optimizer.sharded_param_state_fully_reshardable(fail=True)
    assert optimizer.buffers[0].param_index_map is indices
