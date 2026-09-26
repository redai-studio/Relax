# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for _broadcast_converted_phase and _broadcast_converted_bucket.

These functions broadcast converted expert tensors across PP and EP process
groups using only NCCL (``dist.all_reduce`` for metadata, ``dist.broadcast``
for data tensors).  We mock ``torch.distributed`` and ``mpu`` to simulate
multi-rank scenarios.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

from relax.utils.types import ParamInfo


# ---------------------------------------------------------------------------
# Module-level mocking: stub out megatron.core so we can import without GPU.
# ---------------------------------------------------------------------------

_MEGATRON_MODULES = [
    "megatron",
    "megatron.core",
    "megatron.core.mpu",
    "megatron.core.transformer",
    "megatron.core.transformer.transformer_layer",
    "megatron.core.tensor_parallel",
    "megatron.bridge",
    "megatron.bridge.models",
    "megatron.bridge.peft",
    "megatron.bridge.peft.lora",
]

_MISSING = object()
_saved = {}
for _mod in _MEGATRON_MODULES:
    _saved[_mod] = sys.modules.get(_mod, _MISSING)
    sys.modules[_mod] = ModuleType(_mod)
    if _mod in {"megatron", "megatron.core", "megatron.core.transformer", "megatron.bridge", "megatron.bridge.peft"}:
        sys.modules[_mod].__path__ = []
sys.modules["megatron.core"].mpu = sys.modules["megatron.core.mpu"]
sys.modules["megatron.core.transformer.transformer_layer"].get_transformer_layer_offset = MagicMock(return_value=0)
sys.modules["megatron.bridge.peft.lora"].LoRAMerge = MagicMock

try:
    pytest.importorskip("triton")

    from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import (  # noqa: E402
        _broadcast_converted_bucket,
        _broadcast_converted_phase,
        _compute_slot_size,
        _decode_metadata,
        _encode_metadata,
    )
finally:
    for _mod, _orig in _saved.items():
        if _orig is _MISSING:
            sys.modules.pop(_mod, None)
        else:
            sys.modules[_mod] = _orig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_param_info(name: str, src_rank: int, shape=(4, 8)) -> ParamInfo:
    return ParamInfo(
        name=name,
        dtype=torch.float32,
        shape=torch.Size(shape),
        attrs={},
        size=4 * 8 * 4,
        src_rank=src_rank,
    )


def _make_converted(name: str, value: float = 1.0):
    """Simulate bridge_converter.convert() output: list of (name, tensor)."""
    return [
        (f"{name}.weight_packed", torch.full((4, 1), value, dtype=torch.int32)),
        (f"{name}.weight_scale", torch.full((1, 8), value, dtype=torch.float16)),
    ]


class FakeHandle:
    def wait(self):
        pass


def _make_phase_mocks(all_converted_per_rank, bucket_infos, group_ranks):
    """Build side_effects for dist.all_reduce and dist.broadcast.

    dist.all_reduce:
      - 1st call (numel=1): slot_size allreduce(MAX) — compute max slot_size
      - 2nd call (numel>1): metadata allreduce(SUM) — sum all ranks' metadata
    dist.broadcast: no-op for data tensors (returns FakeHandle for async).
    """
    group_ranks_set = set(group_ranks)

    # Pre-compute slot_size across all ranks
    all_slot_sizes = []
    for r in group_ranks:
        ac = all_converted_per_rank.get(r)
        if ac is not None:
            all_slot_sizes.append(_compute_slot_size(ac, bucket_infos))
    max_slot_size = max(all_slot_sizes) if all_slot_sizes else 2

    # Pre-compute metadata tensors for each rank at max_slot_size
    meta_cache = {}
    for r in group_ranks:
        ac = all_converted_per_rank.get(r)
        if ac is not None:
            meta_cache[r] = _encode_metadata(ac, bucket_infos, group_ranks_set, r, max_slot_size)

    allreduce_call_count = [0]

    def fake_all_reduce(tensor, op=None, group=None):
        allreduce_call_count[0] += 1
        if tensor.numel() == 1:
            tensor.fill_(max_slot_size)
        else:
            result = torch.zeros_like(tensor)
            for r in group_ranks:
                if r in meta_cache:
                    result += meta_cache[r].to(tensor.device)
            tensor.copy_(result)

    def fake_broadcast(tensor, src, group=None, async_op=False):
        if async_op:
            return FakeHandle()

    return fake_all_reduce, fake_broadcast


# ---------------------------------------------------------------------------
# Metadata encode/decode tests
# ---------------------------------------------------------------------------


class TestMetadataEncodeDecode:
    """Test _encode_metadata / _decode_metadata roundtrip."""

    def test_roundtrip_single(self):
        infos = [_make_param_info("a.experts.0.gate_proj", src_rank=0)]
        converted = _make_converted("a.experts.0.gate_proj", value=1.0)
        slot_size = _compute_slot_size([converted], infos)
        meta_t = _encode_metadata([converted], infos, {0}, rank=0, slot_size=slot_size)
        slots = _decode_metadata(meta_t, slot_size)
        assert len(slots) == 1
        src, tensors_meta = slots[0]
        assert src == 0
        assert len(tensors_meta) == 2
        assert tensors_meta[0][0] == "a.experts.0.gate_proj.weight_packed"
        assert tensors_meta[0][1] == (4, 1)
        assert tensors_meta[0][2] == torch.int32
        assert tensors_meta[1][0] == "a.experts.0.gate_proj.weight_scale"

    def test_roundtrip_with_none(self):
        infos = [
            _make_param_info("a.experts.0.gate_proj", src_rank=0),
            _make_param_info("a.experts.1.gate_proj", src_rank=2),
        ]
        converted = _make_converted("a.experts.0.gate_proj")
        slot_size = _compute_slot_size([converted, None], infos)
        meta_t = _encode_metadata([converted, None], infos, {0, 1}, rank=0, slot_size=slot_size)
        slots = _decode_metadata(meta_t, slot_size)
        assert len(slots) == 2
        assert slots[0] is not None
        assert slots[1] is None

    def test_allreduce_sum_correctness(self):
        """Verify that SUM of two ranks' fixed-width metadata produces correct
        merged result."""
        infos = [
            _make_param_info("a.experts.0.gate_proj", src_rank=0),
            _make_param_info("a.experts.1.gate_proj", src_rank=1),
        ]
        c0 = _make_converted("a.experts.0.gate_proj")
        c1 = _make_converted("a.experts.1.gate_proj")

        slot_size = max(
            _compute_slot_size([c0, None], infos),
            _compute_slot_size([None, c1], infos),
        )
        meta_r0 = _encode_metadata([c0, None], infos, {0, 1}, rank=0, slot_size=slot_size)
        meta_r1 = _encode_metadata([None, c1], infos, {0, 1}, rank=1, slot_size=slot_size)

        merged = meta_r0 + meta_r1

        slots = _decode_metadata(merged, slot_size)
        assert len(slots) == 2
        assert slots[0] is not None
        assert slots[1] is not None
        assert slots[0][0] == 0
        assert slots[1][0] == 1


# ---------------------------------------------------------------------------
# _broadcast_converted_phase tests
# ---------------------------------------------------------------------------


class TestBroadcastQuantizedPhase:
    """Test _broadcast_converted_phase with various group configurations."""

    @staticmethod
    def _run_phase(bucket_infos, all_converted_per_rank, group_ranks, current_rank):
        group = MagicMock()
        fake_ar, fake_bcast = _make_phase_mocks(all_converted_per_rank, bucket_infos, group_ranks)

        with (
            patch("torch.distributed.get_process_group_ranks", return_value=group_ranks),
            patch("torch.distributed.all_reduce", side_effect=fake_ar),
            patch("torch.distributed.broadcast", side_effect=fake_bcast),
        ):
            return _broadcast_converted_phase(
                bucket_infos,
                all_converted_per_rank[current_rank],
                device="cpu",
                rank=current_rank,
                group=group,
            )

    def test_owner_keeps_data(self):
        """Owner rank's converted data passes through unchanged."""
        infos = [_make_param_info("layer0.experts.0.gate_proj", src_rank=0)]
        converted_0 = _make_converted("layer0.experts.0.gate_proj", value=42.0)
        all_per_rank = {0: [converted_0]}

        result = self._run_phase(infos, all_per_rank, group_ranks=[0], current_rank=0)

        assert result[0] is not None
        assert len(result[0]) == 2
        assert result[0][0][0] == "layer0.experts.0.gate_proj.weight_packed"
        assert torch.equal(result[0][0][1], converted_0[0][1])

    def test_non_owner_receives_data(self):
        """Non-owner rank receives tensors with correct shapes and dtypes."""
        infos = [_make_param_info("layer0.experts.0.gate_proj", src_rank=0)]
        converted_0 = _make_converted("layer0.experts.0.gate_proj", value=42.0)
        all_per_rank = {
            0: [converted_0],
            1: [None],
        }

        result = self._run_phase(infos, all_per_rank, group_ranks=[0, 1], current_rank=1)

        assert result[0] is not None
        assert len(result[0]) == 2
        assert result[0][0][0] == "layer0.experts.0.gate_proj.weight_packed"
        assert result[0][0][1].shape == converted_0[0][1].shape
        assert result[0][0][1].dtype == converted_0[0][1].dtype

    def test_multiple_params_mixed_ownership(self):
        """Two params owned by different ranks in the same group."""
        infos = [
            _make_param_info("layer0.experts.0.gate_proj", src_rank=0),
            _make_param_info("layer1.experts.0.gate_proj", src_rank=1),
        ]
        converted_0 = _make_converted("layer0.experts.0.gate_proj", value=10.0)
        converted_1 = _make_converted("layer1.experts.0.gate_proj", value=20.0)
        all_per_rank = {
            0: [converted_0, None],
            1: [None, converted_1],
        }

        r0 = self._run_phase(infos, all_per_rank, group_ranks=[0, 1], current_rank=0)
        assert r0[0] is not None
        assert r0[1] is not None
        assert r0[0][0][0] == "layer0.experts.0.gate_proj.weight_packed"
        assert r0[1][0][0] == "layer1.experts.0.gate_proj.weight_packed"

        r1 = self._run_phase(infos, all_per_rank, group_ranks=[0, 1], current_rank=1)
        assert r1[0] is not None
        assert r1[1] is not None

    def test_foreign_params_skipped(self):
        """Params whose src_rank is not in the group remain None."""
        infos = [
            _make_param_info("layer0.experts.0.gate_proj", src_rank=0),
            _make_param_info("layer0.experts.1.gate_proj", src_rank=2),
        ]
        converted_0 = _make_converted("layer0.experts.0.gate_proj")
        all_per_rank = {
            0: [converted_0, None],
            1: [None, None],
        }

        result = self._run_phase(infos, all_per_rank, group_ranks=[0, 1], current_rank=1)
        assert result[0] is not None
        assert result[1] is None

    def test_src_rank_fallback_to_current_rank(self):
        """When info.src_rank is not in group_ranks, src falls back to current
        rank."""
        infos = [_make_param_info("layer5.experts.0.gate_proj", src_rank=5)]
        converted = _make_converted("layer5.experts.0.gate_proj")
        all_per_rank = {
            0: [converted],
            8: [None],
        }

        result = self._run_phase(infos, all_per_rank, group_ranks=[0, 8], current_rank=0)
        assert result[0] is not None

        result_8 = self._run_phase(infos, all_per_rank, group_ranks=[0, 8], current_rank=8)
        assert result_8[0] is not None


# ---------------------------------------------------------------------------
# _broadcast_converted_bucket tests
# ---------------------------------------------------------------------------


class TestBroadcastQuantizedBucket:
    """Test _broadcast_converted_bucket end-to-end."""

    @staticmethod
    def _run_bucket(bucket_infos, all_converted, pp_size, ep_size):
        with (
            patch("relax.backends.megatron.weight_update.hf_weight_iterator_bridge.dist") as mock_dist,
            patch("relax.backends.megatron.weight_update.hf_weight_iterator_bridge.mpu") as mock_mpu,
        ):
            mock_dist.get_rank.return_value = 0
            mock_mpu.get_pipeline_model_parallel_world_size.return_value = pp_size
            mock_mpu.get_expert_model_parallel_world_size.return_value = ep_size
            mock_mpu.get_pipeline_model_parallel_group.return_value = MagicMock()
            mock_mpu.get_expert_model_parallel_group.return_value = MagicMock()

            return _broadcast_converted_bucket(bucket_infos, all_converted, device="cpu")

    def test_no_broadcast_pp1_ep1(self):
        """PP=1, EP=1: just flatten all_converted."""
        infos = [
            _make_param_info("layer0.experts.0.gate_proj", src_rank=0),
            _make_param_info("layer0.experts.0.up_proj", src_rank=0),
        ]
        c0 = _make_converted("layer0.experts.0.gate_proj")
        c1 = _make_converted("layer0.experts.0.up_proj")

        result = self._run_bucket(infos, [c0, c1], pp_size=1, ep_size=1)

        assert len(result) == 4
        names = [name for name, _ in result]
        assert "layer0.experts.0.gate_proj.weight_packed" in names
        assert "layer0.experts.0.gate_proj.weight_scale" in names
        assert "layer0.experts.0.up_proj.weight_packed" in names
        assert "layer0.experts.0.up_proj.weight_scale" in names

    def test_none_entries_skipped(self):
        """None entries in all_converted produce no output."""
        infos = [
            _make_param_info("layer0.experts.0.gate_proj", src_rank=0),
            _make_param_info("layer0.experts.1.gate_proj", src_rank=8),
        ]
        c0 = _make_converted("layer0.experts.0.gate_proj")

        result = self._run_bucket(infos, [c0, None], pp_size=1, ep_size=1)

        assert len(result) == 2
        names = [name for name, _ in result]
        assert "layer0.experts.0.gate_proj.weight_packed" in names

    def test_nccl_only_no_gloo(self):
        """Verify only dist.all_reduce and dist.broadcast are called (no
        broadcast_object_list or all_gather_object)."""
        infos = [_make_param_info("layer0.experts.0.gate_proj", src_rank=0)]
        converted = _make_converted("layer0.experts.0.gate_proj")
        group = MagicMock()

        allreduce_calls = []
        broadcast_calls = []
        gloo_calls = []

        def capture_allreduce(tensor, op=None, group=None):
            allreduce_calls.append({"numel": tensor.numel()})

        def capture_broadcast(tensor, src, group=None, async_op=False):
            broadcast_calls.append({"dtype": tensor.dtype, "src": src})
            if async_op:
                return FakeHandle()

        def capture_gloo(*args, **kwargs):
            gloo_calls.append(True)

        with (
            patch("torch.distributed.get_process_group_ranks", return_value=[0]),
            patch("torch.distributed.all_reduce", side_effect=capture_allreduce),
            patch("torch.distributed.broadcast", side_effect=capture_broadcast),
            patch("torch.distributed.broadcast_object_list", side_effect=capture_gloo),
            patch("torch.distributed.all_gather_object", side_effect=capture_gloo),
        ):
            _broadcast_converted_phase(infos, [converted], "cpu", rank=0, group=group)

        assert len(allreduce_calls) == 2
        assert len(broadcast_calls) > 0
        assert len(gloo_calls) == 0


# ---------------------------------------------------------------------------
# Integration-style test: simulate PP=2 x EP=2 (4 ranks)
# ---------------------------------------------------------------------------


class TestPPxEPIntegration:
    """Simulate a 4-rank setup: PP=2, EP=2.

    Rank layout:
      rank 0: PP stage 0, EP shard 0 -- owns experts.0 from layer 0
      rank 1: PP stage 1, EP shard 0 -- owns experts.0 from layer 1
      rank 2: PP stage 0, EP shard 1 -- owns experts.1 from layer 0
      rank 3: PP stage 1, EP shard 1 -- owns experts.1 from layer 1

    PP groups: [0, 1], [2, 3]
    EP groups: [0, 2], [1, 3]
    """

    BUCKET_INFOS = [
        _make_param_info("layer0.experts.0.gate_proj", src_rank=0),
        _make_param_info("layer1.experts.0.gate_proj", src_rank=1),
        _make_param_info("layer0.experts.1.gate_proj", src_rank=2),
        _make_param_info("layer1.experts.1.gate_proj", src_rank=3),
    ]

    @staticmethod
    def _build_all_converted(rank):
        result = [None, None, None, None]
        names = [
            "layer0.experts.0.gate_proj",
            "layer1.experts.0.gate_proj",
            "layer0.experts.1.gate_proj",
            "layer1.experts.1.gate_proj",
        ]
        result[rank] = _make_converted(names[rank], value=float(rank + 1))
        return result

    def _simulate_rank(self, rank, pp_group, ep_group):
        all_converted = self._build_all_converted(rank)

        # --- PP phase ---
        pp_group_mock = MagicMock()
        all_converted_pp = {r: self._build_all_converted(r) for r in pp_group}
        fake_ar_pp, fake_bcast_pp = _make_phase_mocks(all_converted_pp, self.BUCKET_INFOS, pp_group)

        with (
            patch("torch.distributed.get_process_group_ranks", return_value=pp_group),
            patch("torch.distributed.all_reduce", side_effect=fake_ar_pp),
            patch("torch.distributed.broadcast", side_effect=fake_bcast_pp),
        ):
            all_converted = _broadcast_converted_phase(
                self.BUCKET_INFOS, all_converted, "cpu", rank=rank, group=pp_group_mock
            )

        # --- EP phase ---
        ep_group_mock = MagicMock()
        ep_results = {}
        for r in ep_group:
            other_pp_group = [0, 1] if r in [0, 1] else [2, 3]
            other_pp_members = {rr: self._build_all_converted(rr) for rr in other_pp_group}
            pp_result = [None] * 4
            for rr in other_pp_group:
                for idx, c in enumerate(other_pp_members[rr]):
                    if c is not None:
                        pp_result[idx] = c
            ep_results[r] = pp_result

        fake_ar_ep, fake_bcast_ep = _make_phase_mocks(ep_results, self.BUCKET_INFOS, ep_group)

        with (
            patch("torch.distributed.get_process_group_ranks", return_value=ep_group),
            patch("torch.distributed.all_reduce", side_effect=fake_ar_ep),
            patch("torch.distributed.broadcast", side_effect=fake_bcast_ep),
        ):
            all_converted = _broadcast_converted_phase(
                self.BUCKET_INFOS, all_converted, "cpu", rank=rank, group=ep_group_mock
            )

        return all_converted

    def test_rank0_receives_all(self):
        """Rank 0 (PP group [0,1], EP group [0,2]) ends up with all 4
        params."""
        result = self._simulate_rank(rank=0, pp_group=[0, 1], ep_group=[0, 2])
        for i in range(4):
            assert result[i] is not None, f"param index {i} is None after PP+EP broadcast"
            assert len(result[i]) == 2

    def test_rank3_receives_all(self):
        """Rank 3 (PP group [2,3], EP group [1,3]) ends up with all 4
        params."""
        result = self._simulate_rank(rank=3, pp_group=[2, 3], ep_group=[1, 3])
        for i in range(4):
            assert result[i] is not None, f"param index {i} is None after PP+EP broadcast"

    def test_all_names_present(self):
        """All 4 param names x 2 tensors each = 8 named tensors in final output."""
        result = self._simulate_rank(rank=0, pp_group=[0, 1], ep_group=[0, 2])
        all_names = []
        for converted in result:
            if converted is not None:
                all_names.extend(n for n, _ in converted)
        expected_names = {
            "layer0.experts.0.gate_proj.weight_packed",
            "layer0.experts.0.gate_proj.weight_scale",
            "layer1.experts.0.gate_proj.weight_packed",
            "layer1.experts.0.gate_proj.weight_scale",
            "layer0.experts.1.gate_proj.weight_packed",
            "layer0.experts.1.gate_proj.weight_scale",
            "layer1.experts.1.gate_proj.weight_packed",
            "layer1.experts.1.gate_proj.weight_scale",
        }
        assert set(all_names) == expected_names
