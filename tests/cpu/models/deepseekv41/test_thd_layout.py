# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Packed layout contracts, without constructing a model or discovering GPUs."""

import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest


_SOURCE = Path(__file__).resolve().parents[4] / "relax/models/deepseekv41/thd/layout.py"
_SPEC = importlib.util.spec_from_file_location("v41_thd_layout_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
ThdLayout = _MODULE.ThdLayout


def _params(**overrides: object) -> SimpleNamespace:
    values = dict(
        qkv_format="thd",
        cu_seqlens_q_cpu=[0, 3, 4, 9],
        valid_seqlens_q_cpu=[3, 1, 5],
        cu_seqlens_q_padded_cpu=(0, 3, 4, 16),
        cp_partition_mode="contiguous",
        num_valid_tokens=9,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cp_layout_keeps_real_sample_boundaries_separate_from_physical_tail() -> None:
    layouts = [ThdLayout.from_packed_seq_params(_params(), cp_rank=rank, cp_size=4) for rank in range(4)]
    assert [(layout.global_start, layout.local_length) for layout in layouts] == [(0, 4), (4, 4), (8, 4), (12, 4)]
    assert [layout.candidate_width(2) for layout in layouts] == [1, 2, 2, 0]
    assert [
        (span.seq_index, span.local_start, span.local_end, span.position_start) for span in layouts[2].spans()
    ] == [(2, 0, 1, 4)]
    assert layouts[3].spans() == ()
    for layout in layouts:
        assert layout.seq_starts == (0, 3, 4)
        assert layout.seq_lens == (3, 1, 5)
        assert layout.compressed_offsets(2) == (0, 1, 1, 3)
        assert layout.compressed_count(1) == 9
        assert layout.compressed_count(2) == 3
        assert layout.total_tokens == 16


def test_layout_is_a_snapshot_of_host_metadata() -> None:
    params = _params()
    layout = ThdLayout.from_packed_seq_params(params, cp_rank=0, cp_size=1, seqlen_local=16)
    params.cu_seqlens_q_cpu[1] = 2
    params.valid_seqlens_q_cpu[0] = 2
    assert layout.seq_lens == (3, 1, 5)
    with pytest.raises(FrozenInstanceError):
        layout.local_length = 8


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"valid_seqlens_q_cpu": [4, 1, 4]}, "fit within"),
        ({"cu_seqlens_q_padded_cpu": (0, 4, 4, 16)}, "extend only"),
        ({"cu_seqlens_q_padded_cpu": (0, 3, 4, 15)}, "divide evenly"),
        ({"cp_partition_mode": "zigzag"}, "contiguous"),
        ({"num_valid_tokens": 10}, "num_valid_tokens"),
    ],
)
def test_layout_rejects_inconsistent_lengths_or_partition(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ThdLayout.from_packed_seq_params(_params(**overrides), cp_rank=0, cp_size=4)


def test_layout_checks_hidden_length_without_reading_accelerator_metadata() -> None:
    params = _params()
    params.cu_seqlens_q = object()
    params.cu_seqlens_q_padded = object()
    assert ThdLayout.from_packed_seq_params(params, cp_rank=1, cp_size=4).local_length == 4
    with pytest.raises(ValueError, match="hidden-state length"):
        ThdLayout.from_packed_seq_params(params, cp_rank=1, cp_size=4, seqlen_local=8)
    del params.cu_seqlens_q_padded_cpu
    with pytest.raises(ValueError, match="host physical lengths"):
        ThdLayout.from_packed_seq_params(params, cp_rank=1, cp_size=4)


def test_layout_rejects_scalar_objects_in_host_lists_before_converting_them() -> None:
    class DeviceScalar:
        def __index__(self) -> int:
            raise AssertionError("An accelerator scalar must not be read.")

    with pytest.raises(ValueError, match="must contain integers"):
        ThdLayout.from_packed_seq_params(_params(cu_seqlens_q_cpu=[0, DeviceScalar()]), cp_rank=0, cp_size=1)


def test_position_and_validity_tensors_match_host_spans_and_are_reused() -> None:
    torch = pytest.importorskip("torch")
    layout = ThdLayout.from_packed_seq_params(_params(), cp_rank=2, cp_size=4)
    torch.testing.assert_close(layout.positions("cpu"), torch.tensor([4, 0, 0, 0]))
    torch.testing.assert_close(layout.valid_mask("cpu"), torch.tensor([True, False, False, False]))
    assert layout.positions("cpu") is layout.positions("cpu")
    torch.testing.assert_close(
        layout.positions("cpu", global_stream=True), torch.tensor([0, 1, 2, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0])
    )


def test_padding_only_segments_do_not_own_positions_keys_or_candidates() -> None:
    torch = pytest.importorskip("torch")
    params = _params(
        cu_seqlens_q_cpu=[0, 8],
        valid_seqlens_q_cpu=[0],
        cu_seqlens_q_padded_cpu=(0, 8),
        num_valid_tokens=0,
    )
    layout = ThdLayout.from_packed_seq_params(params, cp_rank=1, cp_size=2)
    assert layout.seq_lens == layout.spans() == ()
    assert layout.compressed_offsets(2) == (0,)
    assert layout.compressed_count(2) == layout.candidate_width(2) == 0
    torch.testing.assert_close(layout.positions("cpu"), torch.zeros(4, dtype=torch.int64))
    assert not layout.valid_mask("cpu").any()
