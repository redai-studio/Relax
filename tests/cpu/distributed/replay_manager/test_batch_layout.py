# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check consumer coordinates against hand-packed token rows."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from relax.distributed.replay_manager.backend.megatron import consumer_batch_layout
from relax.distributed.replay_manager.ref_data import ReplayRef


@pytest.fixture
def ranks(monkeypatch):
    state = SimpleNamespace(tp=2, cp=1, tp_rank=0, cp_rank=0)
    mpu = SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: state.tp,
        get_context_parallel_world_size=lambda: state.cp,
        get_tensor_model_parallel_rank=lambda: state.tp_rank,
        get_context_parallel_rank=lambda: state.cp_rank,
    )
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))
    name = "relax.backends.megatron.cp_utils"
    path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/cp_utils.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return state


def refs():
    return [
        ReplayRef(2, "a", "b", f"{i:032x}", "moe", length - 1, 1, 2, 1024, "c" * 64, "int32")
        for i, length in [(1, 3), (2, 5)]
    ]


@pytest.mark.parametrize(
    "vl,allgather,expected_by_cp",
    [
        (False, False, ["A0 A1 p0 B0 B1 B2 B3 p0 A0 A1 p0 p0"]),
        # Zigzag CP takes outer chunks on rank 0 and inner chunks on rank 1.
        (False, False, ["A0 p0 B0 B1 p1 p2 A0 p0", "A1 p0 B2 B3 p0 p0 A1 p0"]),
        # All-gather CP splits the globally padded batch contiguously.
        (False, True, ["A0 A1 p0 B0 B1 B2 B3 p0", "A0 A1 p0 p0 p1 p2 p3 p4"]),
        # Unsplit VL forward pads each sample to 8 before zigzag splitting.
        (True, False, ["A0 A1 p4 p5 B0 B1 p2 p3 A0 A1 p4 p5", "p0 p1 p2 p3 B2 B3 p0 p1 p0 p1 p2 p3"]),
    ],
)
def test_consumer_layout_maps_duplicate_records_and_rank_local_padding(ranks, vl, allgather, expected_by_cp):
    a, b = refs()
    ranks.cp = len(expected_by_cp)
    args = SimpleNamespace(qkv_format="thd", is_vl_model=vl, allgather_cp=allgather, data_pad_size_multiplier=2)
    for ranks.cp_rank, expected_text in enumerate(expected_by_cp):
        full = expected_text.split()
        for sp in (False, True):
            for ranks.tp_rank in range(ranks.tp if sp else 1):
                # With sequence parallelism, each TP rank takes its contiguous half.
                half = len(full) // 2
                expected = full[ranks.tp_rank * half : (ranks.tp_rank + 1) * half] if sp else full
                layout = consumer_batch_layout(args, [a, b, a], [3, 5, 3], sequence_parallel=sp)
                assert layout.size == len(expected)
                np.testing.assert_array_equal(layout.valid, [row[0] != "p" for row in expected])
                assert layout.padding_offsets.tolist() == [[int(row[1:])] for row in expected if row[0] == "p"]
                maps = {entry.sample_index: entry for entry in layout.samples}
                assert set(maps) == {i for i, label in enumerate("AB") if any(row[0] == label for row in expected)}
                for i, entry in maps.items():
                    positions = [pos for pos, row in enumerate(expected) if row[0] == "AB"[i]]
                    assert entry.output_rows.tolist() == positions
                    assert entry.source_rows.tolist() == [int(expected[pos][1:]) for pos in positions]


@pytest.mark.parametrize("failure", ["conflict", "uuid", "rows", "length", "count"])
def test_consumer_layout_rejects_invalid_manifests_and_lengths(ranks, failure):
    values = refs()
    lengths = [3, 5]
    if failure == "conflict":
        values.append(replace(values[0], owner="d"))
        lengths.append(3)
    elif failure == "uuid":
        values[0] = replace(values[0], version=1, record="a")
    elif failure == "rows":
        values[0] = replace(values[0], rows=2**31)
        lengths[0] = 2**31 + 1
    elif failure == "length":
        lengths[0] += 1
    else:
        lengths.pop()
    args = SimpleNamespace(qkv_format="thd", is_vl_model=False, allgather_cp=False, data_pad_size_multiplier=1)
    with pytest.raises(ValueError):
        consumer_batch_layout(args, values, lengths, sequence_parallel=False)
