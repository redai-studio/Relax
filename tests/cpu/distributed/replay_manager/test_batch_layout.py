# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cross-check direct coordinates against the real legacy UUID-handle path."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from relax.distributed.replay_manager.backend.megatron import consumer_batch_layout, pack_token_rows
from relax.distributed.replay_manager.consumer import ConsumerBatchLayout
from relax.distributed.replay_manager.ref_data import ReplayRef, TokenHandle


@pytest.fixture
def ranks(monkeypatch):
    state = SimpleNamespace(tp=4, cp=1, tp_rank=0, cp_rank=0)
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
        for i, length in [(1, 5), (2, 9)]
    ]


def legacy(args, values, lengths, sequence_parallel):
    rows = [torch.from_numpy(TokenHandle.expand(value)) for value in values]

    def pad(value, count):
        return torch.cat((value, torch.from_numpy(TokenHandle.padding(count))), dim=0)

    handles = pack_token_rows(args, rows, lengths, pad, sequence_parallel=sequence_parallel).numpy()
    return ConsumerBatchLayout.from_coordinates(TokenHandle.coordinates(handles, values))


@pytest.mark.parametrize(
    "cp,sp,vl,allgather",
    [
        (1, False, False, False),
        (1, True, False, False),
        (2, False, False, False),
        (2, True, False, True),
        (2, True, True, False),
        (2, False, True, True),
    ],
)
def test_direct_layout_matches_legacy_cp_sp_vl_padding_and_duplicate_records(ranks, cp, sp, vl, allgather):
    ranks.cp = cp
    values = refs()
    values.append(values[0])
    lengths = [value.rows + 1 for value in values]
    args = SimpleNamespace(qkv_format="thd", is_vl_model=vl, allgather_cp=allgather, data_pad_size_multiplier=2)
    for cp_rank in range(cp):
        ranks.cp_rank = cp_rank
        for tp_rank in range(ranks.tp if sp else 1):
            ranks.tp_rank = tp_rank
            expected = legacy(args, values, lengths, sp)
            actual = consumer_batch_layout(args, values, lengths, sequence_parallel=sp)
            assert actual.size == expected.size
            np.testing.assert_array_equal(actual.valid, expected.valid)
            np.testing.assert_array_equal(actual.padding_offsets, expected.padding_offsets)
            assert [sample.sample_index for sample in actual.samples] == [
                sample.sample_index for sample in expected.samples
            ]
            assert all(sample.sample_index != 2 for sample in actual.samples)
            for got, want in zip(actual.samples, expected.samples, strict=True):
                np.testing.assert_array_equal(got.source_rows, want.source_rows)
                np.testing.assert_array_equal(got.output_rows, want.output_rows)


@pytest.mark.parametrize("failure", ["conflict", "uuid", "rows", "length", "count"])
def test_direct_layout_preserves_manifest_and_handle_constraints(ranks, failure):
    values = refs()
    lengths = [5, 9]
    if failure == "conflict":
        values.append(replace(values[0], owner="d"))
        lengths.append(5)
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
