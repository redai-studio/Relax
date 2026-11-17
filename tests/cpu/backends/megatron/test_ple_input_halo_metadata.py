# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fail-closed physical metadata contract for the experimental input-halo
PLE."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


pytest.importorskip("megatron.core.transformer.ple_input_halo_cp", exc_type=ImportError)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.ple_input_halo_cp import (
    _prepared_lengths,
    _resolve_transport,
    forward_input_halo,
    prepare_input_halo_cp,
)

from relax.backends.megatron.qsa_layout_metadata import bind_physical_layout


@pytest.mark.parametrize(
    "mode,lengths,expected",
    [
        ("adaptive", (524288,), "halo"),
        ("adaptive", (240,), "full"),
        ("adaptive", (256,), "halo"),
        ("adaptive", (524288, 240), "full"),
        ("adaptive", (256, 524288), "halo"),
        ("full", (524288,), "full"),
        ("halo", (240,), "halo"),
    ],
)
def test_transport_policy_preserves_explicit_modes_and_falls_back_for_short_segments(mode, lengths, expected):
    assert _resolve_transport(mode, lengths, size=8, halo=16) == expected


def test_unknown_transport_rejected():
    with pytest.raises(ValueError, match="transport"):
        _resolve_transport("invalid", (524288,), size=8, halo=16)


class InputHaloMetadata(unittest.TestCase):
    def layer(self, rows=416, tp=1):
        return SimpleNamespace(
            halo=16,
            tp_group=SimpleNamespace(size=lambda: tp),
            _ngram_ids=torch.zeros(1, rows, 1, dtype=torch.long),
            parameters=lambda: [],
            _cp_packed_seq_params=SimpleNamespace(),
        )

    def packed(self, physical=(0, 272, 416), bind=True):
        logical = torch.tensor([0, 257, 386], dtype=torch.int32)
        padded = torch.tensor(physical, dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=logical,
            cu_seqlens_kv=logical,
            cu_seqlens_q_padded=padded,
            cu_seqlens_kv_padded=padded,
        )
        if bind:
            bind_physical_layout(packed, physical)
        return packed

    def prepare(self, layer, packed, cp=8):
        with patch("megatron.core.parallel_state.get_context_parallel_world_size", return_value=cp):
            prepare_input_halo_cp(layer, packed)

    def test_physical_boundaries_and_global_ids_preserved(self):
        layer = self.layer()
        packed = self.packed()
        ids = layer._ngram_ids
        self.prepare(layer, packed)
        self.assertEqual(layer._cp_packed_seq_params.ple_input_halo_lengths, (272, 144))
        self.assertIs(layer._cp_packed_seq_params.ple_input_halo_packed, packed)
        self.assertIs(layer._ngram_ids, ids)

    def test_checkpoint_context_restores_earlier_batch(self):
        layer = self.layer()
        first = self.packed()
        self.prepare(layer, first)
        saved = (layer._ngram_ids, layer._cp_packed_seq_params)
        layer._ngram_ids = torch.zeros(1, 512, 1, dtype=torch.long)
        layer._cp_packed_seq_params = SimpleNamespace()
        self.prepare(layer, self.packed((0, 256, 512)))
        self.assertEqual(_prepared_lengths(layer), (256, 256))
        layer._ngram_ids, layer._cp_packed_seq_params = saved
        self.assertEqual(_prepared_lengths(layer), (272, 144))
        self.assertIs(layer._cp_packed_seq_params.ple_input_halo_packed, first)

    def test_missing_binding_rejected(self):
        with self.assertRaises(ValueError):
            self.prepare(self.layer(), self.packed(bind=False))

    def test_mutated_physical_tensor_rejected_before_forward(self):
        layer = self.layer()
        packed = self.packed()
        self.prepare(layer, packed)
        packed.cu_seqlens_q_padded[-1] += 16
        with self.assertRaisesRegex(RuntimeError, "valid prepared physical"):
            forward_input_halo(layer, torch.zeros(52, 1, 8))

    def test_wrong_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.prepare(self.layer(rows=400), self.packed())

    def test_unsupported_topology_rejected(self):
        for tp, cp in [(2, 8), (1, 1)]:
            with self.subTest(tp=tp, cp=cp), self.assertRaises(ValueError):
                self.prepare(self.layer(tp=tp), self.packed(), cp)

    def test_trainable_ple_rejected(self):
        layer = self.layer()
        self.prepare(layer, self.packed())
        layer.parameters = lambda: [torch.nn.Parameter(torch.ones(1))]
        with self.assertRaisesRegex(RuntimeError, "all PLE parameters frozen"):
            forward_input_halo(layer, torch.zeros(52, 1, 8))

    def test_nondivisible_partition_rejected(self):
        with self.assertRaises(ValueError):
            self.prepare(self.layer(rows=415), self.packed((0, 271, 415)))


@pytest.mark.parametrize("vl", [False, True])
@pytest.mark.parametrize("enabled", ["0", "1"])
def test_batch_producer_binds_ple_layout_without_qsa_flag(monkeypatch, vl, enabled):
    from megatron.core import mpu

    from relax.backends.megatron import data
    from relax.backends.megatron.qsa_layout_metadata import valid_physical_layout

    args = SimpleNamespace(
        dynamic_context_parallel=False, enable_mtp_training=False, train_mode="sft", sft=True, allgather_cp=False
    )
    monkeypatch.setattr(data, "get_args", lambda: args)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 8)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setenv("RELAX_QWEN38_QSA_CPU_LAYOUT", "0")
    monkeypatch.setenv("QWEN38_PLE_INPUT_HALO_CP", enabled)
    lengths = [257, 129]
    raw = dict(
        tokens=[torch.arange(n) for n in lengths],
        total_lengths=lengths,
        response_lengths=lengths,
        loss_masks=[torch.ones(n, dtype=torch.int) for n in lengths],
    )
    batch = data.get_batch(
        iter([(raw, None)]),
        list(raw),
        pad_multiplier=16,
        qkv_format="thd",
        is_vl_model=vl,
        pack_device=torch.device("cpu"),
    )
    packed = batch["vlm_packed_seq_params" if vl else "packed_seq_params"]
    record = valid_physical_layout(packed)
    assert (record is not None) == (enabled == "1")
    if record is not None:
        mapped = data._map_packed_seq_params(packed, lambda x: x.clone() if isinstance(x, torch.Tensor) else x)
        assert valid_physical_layout(mapped) == record
        getattr(mapped, record[1])[-1] += 16
        assert valid_physical_layout(mapped) is None


def test_dynamic_cp_rejected_before_batch_collectives(monkeypatch):
    from relax.backends.megatron import data

    monkeypatch.setattr(data, "get_args", lambda: SimpleNamespace(dynamic_context_parallel=True))
    monkeypatch.setenv("QWEN38_PLE_INPUT_HALO_CP", "1")
    with pytest.raises(ValueError, match="dynamic context parallelism"):
        data.get_batch(iter([({"tokens": [torch.arange(16)]}, None)]), ["tokens"], pack_device=torch.device("cpu"))


@pytest.mark.parametrize("attribute,value", [("local_cp_size", 1), ("cp_group", object())])
def test_gpt_rejects_dynamic_cp_before_native_prepare(monkeypatch, attribute, value):
    from megatron.core.models.gpt.gpt_model import GPTModel

    def forbidden_prepare(*args, **kwargs):
        pytest.fail("Dynamic CP reached native PLE prepare")

    module = SimpleNamespace(prepare=forbidden_prepare)
    model = SimpleNamespace(decoder=SimpleNamespace(layers=[SimpleNamespace(per_layer_embedding=module)]))
    packed = SimpleNamespace(qkv_format="thd", cu_seqlens_q=None, cu_seqlens_q_padded=None)
    setattr(packed, attribute, value)
    monkeypatch.setenv("QWEN38_PLE_INPUT_HALO_CP", "1")
    with pytest.raises(ValueError, match="dynamic context parallelism"):
        GPTModel._prepare_per_layer_embeddings(model, torch.zeros(1, 16, dtype=torch.long), packed, None)


@pytest.mark.parametrize("vl", [False, True])
def test_allgather_cp_rejected_before_layout_is_used(monkeypatch, vl):
    from megatron.core import mpu

    from relax.backends.megatron import data

    monkeypatch.setattr(data, "get_args", lambda: SimpleNamespace(dynamic_context_parallel=False))
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setenv("QWEN38_PLE_INPUT_HALO_CP", "1")
    with pytest.raises(ValueError, match="per-document zigzag"):
        data.get_batch(
            iter([({"tokens": [torch.arange(16)]}, None)]),
            ["tokens"],
            allgather_cp=True,
            is_vl_model=vl,
            pack_device=torch.device("cpu"),
        )


@pytest.mark.parametrize("change", ["replacement", "rebind", "ids", "hidden_rows"])
def test_changed_prepared_inputs_rejected_before_communication(change):
    fixture = InputHaloMetadata()
    layer, packed = fixture.layer(), fixture.packed()
    fixture.prepare(layer, packed)
    hidden = torch.zeros(52, 1, 8)
    if change == "replacement":
        packed.cu_seqlens_q_padded = packed.cu_seqlens_q_padded.clone()
    elif change == "rebind":
        packed.cu_seqlens_q_padded = torch.tensor([0, 256, 416], dtype=torch.int32)
        bind_physical_layout(packed, (0, 256, 416))
    elif change == "ids":
        layer._ngram_ids = torch.zeros(1, 400, 1, dtype=torch.long)
    else:
        hidden = torch.zeros(51, 1, 8)
    with patch(
        "megatron.core.parallel_state.get_context_parallel_group",
        return_value=SimpleNamespace(rank=lambda: 0, size=lambda: 8),
    ):
        with pytest.raises(RuntimeError, match="metadata|physical layout"):
            forward_input_halo(layer, hidden)


@pytest.mark.parametrize("case", ["sbhd", "empty_document", "native_prepare_missing"])
def test_invalid_prepare_contract_rejected(case):
    fixture = InputHaloMetadata()
    layer, packed = fixture.layer(), fixture.packed()
    if case == "sbhd":
        packed.qkv_format = "sbhd"
    elif case == "empty_document":
        packed = fixture.packed((0, 0, 416))
    else:
        layer._cp_packed_seq_params = None
    with pytest.raises(ValueError):
        fixture.prepare(layer, packed)
