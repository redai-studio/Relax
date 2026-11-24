# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update.sharded_source import BridgeShardSource


def test_native_full_seed_refreshes_the_sparse_directory_when_geometry_changes():
    source = BridgeShardSource(SimpleNamespace(get_hf_weight_chunks=lambda weights: iter([list(weights.items())])))
    original = {"hf.weight": torch.zeros((2, 4), dtype=torch.bfloat16)}
    assert list(source.seed(original)) == [list(original.items())]
    assert source.schema == {"hf.weight": ((2, 4), torch.bfloat16)}
    existing_index = [object()]
    source.index = existing_index

    list(source.seed({"hf.weight": torch.ones((2, 4), dtype=torch.bfloat16)}))
    assert source.index is existing_index
    list(source.seed({"hf.weight": torch.ones((4, 2), dtype=torch.bfloat16)}))
    assert source.index is None
    assert source.schema == {"hf.weight": ((4, 2), torch.bfloat16)}


def test_native_full_seed_rejects_conflicting_slots():
    native = [("hf.weight", torch.zeros((2, 4))), ("hf.weight", torch.zeros((4, 2)))]
    source = BridgeShardSource(SimpleNamespace(get_hf_weight_chunks=lambda _: iter([native])))
    with pytest.raises(ValueError, match="Conflicting native FULL slots"):
        list(source.seed({}))
