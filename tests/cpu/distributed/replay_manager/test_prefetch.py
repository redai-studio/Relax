# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Raw replay storage rejects ambiguous output layouts before body I/O."""

import numpy as np
import pytest

from tests.cpu.distributed.replay_manager.test_reference import published  # noqa: F401


def test_progressive_read_rejects_overlapping_outputs_before_io(published):  # noqa: F811
    _, store, body, _, ref, _ = published
    store.attach([ref], "train_0")
    store.commit([ref])
    with store.reader([ref], partition="train_0") as reader:
        output = np.zeros((4, 2), dtype=np.int32)
        request = [(ref, np.arange(4), np.arange(4), reader._leases[0])]
        reads = body.reads
        with pytest.raises(ValueError, match="non-overlapping"):
            store._read_layers_into([(request, 0, output), (request, 1, output)], on_layer=lambda _: None)
        assert body.reads == reads
    store.retire_partition("train_0")
