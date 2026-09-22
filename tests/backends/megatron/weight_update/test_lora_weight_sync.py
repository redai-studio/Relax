# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the LoRA weight-sync path (adapter mode, colocate sync, fully-
async).

Consolidates three previously separate suites:

- **Adapter mode** — delta extraction/sparse filtering (``extract_lora_delta``),
  the mode predicates that gate adapter vs merge paths, and the checkpoint save
  that writes an HF-PEFT dir + adapter-mode metadata.
- **Colocate sync** — the transport-independent ``LoraAdapterSync`` helper:
  ``config_dict`` (HF-PEFT shape + Megatron->SGLang target-module conversion).
- **Fully-async** — the pure/logic pieces gating the DeviceDirectBackend path:
  the adapter-param skip predicate, the delta-skip early-return contract, the
  Megatron->HF target-module expansion, the tp_size=1 merge formula, and the
  base/adapter prefix-join used by the merge lookup.

GPU/megatron-dependent pieces are guarded with ``importorskip`` so they run in CI
but skip in a CPU-only checkout.
"""

import json
import tempfile
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from relax.utils.megatron_peft_utils import (
    convert_megatron_to_hf_target_modules,
    extract_lora_delta,
    is_lora_adapter_mode,
    is_lora_adapter_param,
    is_lora_enabled,
    is_lora_merge_mode,
)


# ---------------------------------------------------------------------------
# Adapter mode: delta extraction, mode predicates, checkpoint metadata
# ---------------------------------------------------------------------------


class TestLoRADeltaExtraction:
    """Test LoRA delta extraction and sparse filtering."""

    def test_extract_lora_delta_first_sync(self):
        """First sync (old_state=None) returns the complete LoRA state."""
        new_state = {
            "lora_a": torch.randn(32, 128),
            "lora_b": torch.randn(128, 32),
        }

        delta, new_saved = extract_lora_delta(new_state, old_lora_state=None)

        assert len(delta) == 2
        assert torch.allclose(delta["lora_a"], new_state["lora_a"])
        assert torch.allclose(delta["lora_b"], new_state["lora_b"])
        # Saved state is decoupled from the returned delta for the next comparison.
        assert torch.allclose(new_saved["lora_a"], new_state["lora_a"])

    def test_extract_lora_delta_sparse_filtering(self):
        """Changes below the threshold are filtered out (delta-skip)."""
        state1 = {
            "lora_a": torch.ones(10, 10),
            "lora_b": torch.ones(10, 10) * 2.0,
        }
        state2 = {
            "lora_a": torch.ones(10, 10) + 1e-8,  # tiny change
            "lora_b": torch.ones(10, 10) * 2.0,
        }

        delta, _ = extract_lora_delta(state2, old_lora_state=state1, threshold=1e-6)

        assert len(delta) == 0

    def test_extract_lora_delta_significant_changes(self):
        """Significant changes are captured as (new - old)."""
        state1 = {
            "lora_a": torch.ones(10, 10),
            "lora_b": torch.ones(10, 10) * 2.0,
        }
        state2 = {
            "lora_a": torch.ones(10, 10) + 0.1,
            "lora_b": torch.ones(10, 10) * 2.0,
        }

        delta, _ = extract_lora_delta(state2, old_lora_state=state1, threshold=1e-6)

        assert "lora_a" in delta
        assert "lora_b" not in delta
        assert torch.allclose(delta["lora_a"], torch.ones(10, 10) * 0.1, atol=1e-7)

    def test_extract_lora_delta_new_parameters(self):
        """A parameter absent from old_state is included in full."""
        state1 = {"lora_a": torch.ones(10, 10)}
        state2 = {
            "lora_a": torch.ones(10, 10),
            "lora_b": torch.randn(10, 10),
        }

        delta, _ = extract_lora_delta(state2, old_lora_state=state1)

        assert "lora_b" in delta
        assert torch.allclose(delta["lora_b"], state2["lora_b"])


class TestModePredicates:
    """Test the LoRA mode predicates used to gate adapter vs merge paths."""

    def test_is_lora_adapter_mode_flag(self):
        args_adapter = MagicMock()
        args_adapter.lora_adapter_mode = True
        args_merge = MagicMock()
        args_merge.lora_adapter_mode = False

        assert is_lora_adapter_mode(args_adapter)
        assert not is_lora_adapter_mode(args_merge)

    def test_is_lora_merge_mode_flag(self):
        args_merge = MagicMock()
        args_merge.lora_merge_mode = True
        args_adapter = MagicMock()
        args_adapter.lora_merge_mode = False

        assert is_lora_merge_mode(args_merge)
        assert not is_lora_merge_mode(args_adapter)

    def test_is_lora_enabled_by_rank(self):
        args_on = MagicMock()
        args_on.lora_rank = 32
        args_off = MagicMock()
        args_off.lora_rank = 0

        assert is_lora_enabled(args_on)
        assert not is_lora_enabled(args_off)

    def test_missing_attrs_default_false(self):
        # Plain object without the LoRA attributes -> predicates must not raise.
        class Empty:
            pass

        empty = Empty()
        assert not is_lora_adapter_mode(empty)
        assert not is_lora_merge_mode(empty)
        assert not is_lora_enabled(empty)


class TestCheckpointModeDetection:
    """Test checkpoint save gathers the adapter and records adapter-mode
    metadata."""

    def test_checkpoint_save_with_metadata(self):
        # checkpoint.py imports megatron.training at module load; skip when unavailable (CPU-only CI).
        pytest.importorskip("megatron.training.checkpointing")

        from relax.backends.megatron.checkpoint import _save_lora_to_checkpoint

        Item = namedtuple("Item", ["param_name", "weight"])

        with tempfile.TemporaryDirectory() as tmpdir:
            model = MagicMock()

            args = MagicMock()
            args.lora_rank = 32
            args.lora_alpha = 32
            args.lora_target_modules = ["linear_qkv", "linear_proj"]
            args.lora_dropout = 0.1
            args.lora_merge_mode = False
            args.lora_adapter_mode = True

            bridge = MagicMock()
            bridge.export_adapter_weights.return_value = [
                Item("base_model.model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(32, 16)),
                Item("base_model.model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(16, 32)),
            ]

            def fake_gather(obj, object_gather_list=None, dst=0, group=None):
                if object_gather_list is not None:
                    object_gather_list[0] = obj

            def fake_all_gather(output, obj, group=None):
                output[0] = obj

            with (
                patch("torch.distributed.get_rank", return_value=0),
                patch("torch.distributed.get_world_size", return_value=1),
                patch("torch.distributed.all_gather_object", side_effect=fake_all_gather),
                patch("torch.distributed.gather_object", side_effect=fake_gather),
                patch("torch.distributed.broadcast_object_list"),
                patch("relax.backends.megatron.checkpoint.get_gloo_group", return_value=None),
                patch("relax.backends.megatron.checkpoint.megatron_bridge_utils.patch_megatron_model"),
            ):
                _save_lora_to_checkpoint(model, tmpdir, args, bridge=bridge)

            adapter_dir = Path(tmpdir) / "lora_adapter"
            assert (adapter_dir / "adapter_config.json").exists()
            assert (adapter_dir / "adapter_model.safetensors").exists()

            meta = json.loads((adapter_dir / "relax_lora_meta.json").read_text())
            assert meta["lora_adapter_mode"] is True
            assert meta["lora_merge_mode"] is False

    def test_empty_adapter_export_is_a_hard_error(self):
        pytest.importorskip("megatron")

        from relax.backends.megatron.checkpoint import _save_lora_to_checkpoint

        args = MagicMock(
            lora_rank=32,
            lora_alpha=32,
            lora_target_modules=["linear_qkv"],
            lora_dropout=0.0,
            lora_merge_mode=False,
            lora_adapter_mode=False,
        )
        bridge = MagicMock()
        bridge.export_adapter_weights.return_value = []

        def fake_all_gather(output, obj, group=None):
            output[0] = obj

        def fake_gather(obj, object_gather_list=None, dst=0, group=None):
            object_gather_list[0] = obj

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.all_gather_object", side_effect=fake_all_gather),
            patch("torch.distributed.gather_object", side_effect=fake_gather),
            patch("torch.distributed.broadcast_object_list"),
            patch("relax.backends.megatron.checkpoint.get_gloo_group", return_value=None),
            patch("relax.backends.megatron.checkpoint.megatron_bridge_utils.patch_megatron_model"),
            pytest.raises(RuntimeError, match="no adapter parameters"),
        ):
            _save_lora_to_checkpoint(MagicMock(), tmpdir, args, bridge=bridge)


# ---------------------------------------------------------------------------
# Colocate sync: the transport-independent LoraAdapterSync helper
# ---------------------------------------------------------------------------


def _make_sync(save="/data/run"):
    # The helper imports relax.backends.megatron.misc_utils; skip if unavailable.
    lora_adapter_sync = pytest.importorskip("relax.backends.megatron.weight_update.lora_adapter_sync")
    args = SimpleNamespace(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["linear_qkv", "linear_proj"],
        save=save,
        hf_checkpoint="/models/base",
    )
    return lora_adapter_sync.LoraAdapterSync(args, model=None)


class TestConfigDict:
    def test_config_dict_shape(self):
        cd = _make_sync().config_dict()
        assert cd["r"] == 8
        assert cd["lora_alpha"] == 16
        assert cd["lora_dropout"] == 0.05
        assert cd["peft_type"] == "LORA"
        assert cd["task_type"] == "CAUSAL_LM"
        assert cd["bias"] == "none"

    def test_config_dict_converts_target_modules_to_engine_flavor(self):
        # Canonical Megatron names must be expanded to the leaf names SGLang matches by.
        cd = _make_sync().config_dict()
        assert "linear_qkv" not in cd["target_modules"]
        assert {"q_proj", "k_proj", "v_proj", "o_proj"}.issubset(set(cd["target_modules"]))


class TestInitialState:
    def test_fresh_sync_state(self):
        h = _make_sync()
        assert h.base_sync_done is False
        assert h.adapter_loaded is False
        assert h.prev_state is None
        assert h.bridge is None


# ---------------------------------------------------------------------------
# Fully-async: DeviceDirectBackend logic gates
# ---------------------------------------------------------------------------


class TestAdapterSkipPredicate:
    """`is_lora_adapter_param` decides what the rollout convert loop skips."""

    def test_megatron_bridge_adapter_names_matched(self):
        assert is_lora_adapter_param("decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight")
        assert is_lora_adapter_param("decoder.layers.0.self_attention.linear_proj.adapter.linear_out.weight")

    def test_standard_peft_adapter_names_matched(self):
        assert is_lora_adapter_param("base_model.model.layers.0.self_attn.q_proj.lora_A.weight")
        assert is_lora_adapter_param("base_model.model.layers.0.self_attn.q_proj.lora_B.weight")

    def test_base_and_to_wrap_names_not_matched(self):
        # The wrapped base weight MUST NOT be classified as an adapter, or it would be
        # dropped from the rollout broadcast.
        assert not is_lora_adapter_param("decoder.layers.0.self_attention.linear_qkv.to_wrap.weight")
        assert not is_lora_adapter_param("decoder.layers.0.self_attention.linear_proj.to_wrap.weight")
        assert not is_lora_adapter_param("decoder.final_layernorm.weight")
        assert not is_lora_adapter_param("embedding.word_embeddings.weight")


class TestDeltaSkip:
    """Adapter mode skips the push when no adapter changed beyond threshold."""

    def test_unchanged_state_yields_empty_delta(self):
        state = {"a": torch.zeros(4, 8), "b": torch.zeros(8, 4)}
        delta, new_state = extract_lora_delta(state, {k: v.clone() for k, v in state.items()})
        assert delta == {}  # empty -> _push_lora_adapter_distributed early-returns on non-first sync
        assert set(new_state) == set(state)

    def test_changed_state_yields_nonempty_delta(self):
        old = {"a": torch.zeros(4, 8)}
        new = {"a": torch.ones(4, 8)}
        delta, _ = extract_lora_delta(new, old)
        assert "a" in delta and delta  # non-empty -> push proceeds

    def test_first_sync_returns_full_state(self):
        new = {"a": torch.randn(4, 8), "b": torch.randn(8, 4)}
        delta, _ = extract_lora_delta(new, None)
        assert set(delta) == set(new)


class TestTargetModuleExpansion:
    """adapter_config.json target_modules must be HF names for SGLang's PEFT
    loader."""

    def test_qkv_and_proj_expand(self):
        out = convert_megatron_to_hf_target_modules(["linear_qkv", "linear_proj"])
        assert out == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_dedup_preserves_order(self):
        out = convert_megatron_to_hf_target_modules(["linear_qkv", "linear_q"])
        assert out == ["q_proj", "k_proj", "v_proj"]  # linear_q -> q_proj already present


class TestMergeContract:
    """The merge-mode fold relies on LoRAMerge(tp_size=1) == base + alpha/dim*(B @ A)."""

    def test_tp1_formula_reference(self):
        # Documents the exact math _merge_full_base depends on (no megatron needed).
        torch.manual_seed(0)
        out_features, in_features, dim = 6, 5, 2
        base = torch.randn(out_features, in_features)
        b = torch.randn(out_features, dim)  # linear_out
        a = torch.randn(dim, in_features)  # linear_in
        alpha, r = 8, dim
        expected = base + (alpha / r) * (b @ a)
        assert expected.shape == base.shape

    def test_tp1_matches_real_loramerge(self):
        from inspect import signature

        lora = pytest.importorskip("megatron.bridge.peft.lora")
        if not hasattr(lora, "LoRAMerge"):
            pytest.skip("this Megatron-Bridge build does not expose LoRAMerge")
        LoRAMerge = lora.LoRAMerge

        if "tp_size" not in signature(LoRAMerge().merge).parameters:
            pytest.skip("installed megatron bridge LoRAMerge.merge lacks tp_size support")

        torch.manual_seed(0)
        base = torch.randn(6, 5)
        b = torch.randn(6, 2)
        a = torch.randn(2, 5)
        alpha, r = 8, 2
        merged = LoRAMerge().merge(base.float(), b.float(), a.float(), alpha, r, tp_size=1, tp_group=None)
        expected = base + (alpha / r) * (b @ a)
        assert torch.allclose(merged, expected, atol=1e-5)


class TestPrefixJoin:
    """Base weight and its adapter must resolve to the SAME prefix key for the
    merge lookup."""

    def test_base_and_adapter_prefixes_match(self):
        pytest.importorskip("megatron.bridge.peft.lora")
        from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import (
            _adapter_base_prefix,
            _base_param_prefix,
        )

        for module in ("linear_qkv", "linear_proj"):
            base = f"decoder.layers.0.self_attention.{module}.to_wrap.weight"
            a_in = f"decoder.layers.0.self_attention.{module}.adapter.linear_in.weight"
            a_out = f"decoder.layers.0.self_attention.{module}.adapter.linear_out.weight"
            key = _base_param_prefix(base)
            assert _adapter_base_prefix(a_in) == key
            assert _adapter_base_prefix(a_out) == key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
