# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch


def test_glm52_indexer_ownership_and_pipeline_boundaries(glm_modules):
    module = glm_modules("index_sharing")
    full_layers = (1, 2, *range(3, 79, 4))
    config = SimpleNamespace(
        num_hidden_layers=78,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=["full" if i in full_layers else "shared" for i in range(1, 79)],
    )
    sources = module.index_sources(config)
    assert tuple(i for i, source in enumerate(sources, 1) if i == source) == full_layers
    assert sources[2:6] == (3, 3, 3, 3)
    for offset, size in ((0, 22), (22, 20), (42, 20), (62, 16)):
        module.validate_stage(sources, offset, size)
    with pytest.raises(ValueError, match="shared layer 21"):
        module.validate_stage(sources, 20, 20)
    # A virtual stage must obey the same ownership rule.
    with pytest.raises(ValueError, match="PP/VPP"):
        module.validate_stage(sources, 3, 3)


def test_glm5_legacy_and_explicit_pattern(glm_modules):
    module = glm_modules("index_sharing")
    assert module.index_sources(SimpleNamespace(num_hidden_layers=4)) == (1, 2, 3, 4)
    assert module.index_sources(SimpleNamespace(num_hidden_layers=5, index_topk_pattern="FSSF")) == (1, 1, 1, 4, 5)
    with pytest.raises(ValueError, match="first DSA layer"):
        module.index_sources(SimpleNamespace(num_hidden_layers=2, index_topk_pattern="SF"))
    with pytest.raises(ValueError, match="disagrees"):
        module.index_sources(SimpleNamespace(num_hidden_layers=2, indexer_types=["full", "shared"]))


def test_glm_shared_indices_are_isolated_and_survive_reverse_recompute(glm_modules):
    module = glm_modules("index_sharing")
    first, second = SimpleNamespace(), SimpleNamespace()
    a, b, later = (torch.tensor([[i]], dtype=torch.int32) for i in (3, 9, 15))

    def must_not_compute():
        pytest.fail("A shared layer recomputed its own indexer")

    module.shared_topk(first, 3, 3, lambda: a)
    module.shared_topk(first, 7, 7, lambda: later)
    module.shared_topk(second, 3, 3, lambda: b)
    # Reverse recompute of layer 4 must still see layer 3 of its own microbatch.
    assert module.shared_topk(first, 4, 3, must_not_compute) is a
    assert module.shared_topk(second, 4, 3, must_not_compute) is b
    assert module.shared_topk(first, 8, 7, must_not_compute) is later
    refreshed = a + 1
    module.shared_topk(first, 3, 3, lambda: refreshed)
    assert module.shared_topk(first, 4, 3, must_not_compute) is refreshed
    with pytest.raises(ValueError, match="same PP/VPP stage"):
        module.shared_topk(SimpleNamespace(), 4, 3, must_not_compute)
