# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the LoRA merge-mode helpers introduced in "support lora rl merge
mode".

Coverage is scoped to the pure, GPU-free logic that is easy to get wrong and hard to
notice when broken:
- Megatron->HF target-module expansion (``convert_megatron_to_hf_target_modules``)
- Adapter-param classification, incl. the ``lora_gate`` false-positive guard
  (``is_lora_adapter_param``)
- HF-PEFT adapter export round-trip (``write_hf_peft_adapter``)
- LoRA-enabled / merge-mode predicates
- Trainable-adapter parameter counting (``count_adapter_parameters``)
- Base<->adapter param-name prefix round-trip in the fast bridge path
"""

import json
import sys
import types
from argparse import Namespace

import pytest
import torch

from relax.utils.megatron_peft_utils import (
    MEGATRON_TO_HF_MODULES,
    build_lora_peft,
    convert_megatron_to_hf_target_modules,
    count_adapter_parameters,
    exclude_frozen_lora_target_modules,
    is_lora_adapter_param,
    is_lora_enabled,
    is_lora_merge_mode,
    write_hf_peft_adapter,
)


def test_exclude_frozen_lora_target_modules_keeps_vision_merger() -> None:
    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.linear_fc1 = torch.nn.Linear(2, 2)
            self.vision_model = torch.nn.Module()
            self.vision_model.decoder = torch.nn.Module()
            self.vision_model.decoder.linear_fc1 = torch.nn.Linear(2, 2)
            self.vision_model.merger = torch.nn.Module()
            self.vision_model.merger.linear_fc1 = torch.nn.Linear(2, 2)

    targets = exclude_frozen_lora_target_modules(ToyModel(), ["linear_fc1"], [r"^(vision_model|visual)\.(?!merger\.)"])

    assert targets == ["decoder.linear_fc1", "vision_model.merger.linear_fc1"]


class TestBuildLoraPeft:
    @staticmethod
    def _install_create_peft_stub(monkeypatch):
        module = types.ModuleType("megatron.bridge.peft.utils")
        module.create_peft = lambda config: config
        monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
        monkeypatch.setitem(sys.modules, "megatron.bridge", types.ModuleType("megatron.bridge"))
        monkeypatch.setitem(sys.modules, "megatron.bridge.peft", types.ModuleType("megatron.bridge.peft"))
        monkeypatch.setitem(sys.modules, "megatron.bridge.peft.utils", module)

    def test_external_baseline_overrides_are_opt_in(self, monkeypatch):
        self._install_create_peft_stub(monkeypatch)
        monkeypatch.setenv("RELAX_LORA_A_INIT_METHOD", "kaiming")
        monkeypatch.setenv("RELAX_LORA_SHARE_EXPERT_ADAPTERS", "false")
        args = Namespace(
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules=["linear_qkv", "linear_fc1"],
            lora_dropout=0.05,
        )

        config = build_lora_peft(args)

        assert config["lora_A_init_method"] == "kaiming"
        assert config["share_expert_adapters"] is False

    def test_invalid_expert_adapter_override_fails_fast(self, monkeypatch):
        self._install_create_peft_stub(monkeypatch)
        monkeypatch.setenv("RELAX_LORA_SHARE_EXPERT_ADAPTERS", "sometimes")
        args = Namespace(lora_rank=8, lora_alpha=16, lora_target_modules=["linear_qkv"], lora_dropout=0.0)

        with pytest.raises(ValueError, match="RELAX_LORA_SHARE_EXPERT_ADAPTERS"):
            build_lora_peft(args)


class TestConvertMegatronToHfTargetModules:
    def test_convert_megatron_to_hf_expands_fused_modules(self):
        """A fused Megatron linear expands one-to-many to its HF
        projections."""
        assert convert_megatron_to_hf_target_modules(["linear_qkv"]) == ["q_proj", "k_proj", "v_proj"]
        assert convert_megatron_to_hf_target_modules(["linear_fc1"]) == ["gate_proj", "up_proj"]
        assert convert_megatron_to_hf_target_modules(["linear_proj"]) == ["o_proj"]

    def test_convert_megatron_to_hf_dedups_preserving_order(self):
        """Overlapping expansions collapse to a stable, de-duplicated list."""
        # linear_qkv -> q/k/v_proj and linear_q -> q_proj both yield q_proj.
        result = convert_megatron_to_hf_target_modules(["linear_qkv", "linear_q"])
        assert result == ["q_proj", "k_proj", "v_proj"]

    def test_convert_megatron_to_hf_passes_unknown_through(self):
        """Unknown / already-HF names are passed through unchanged."""
        assert convert_megatron_to_hf_target_modules(["q_proj", "custom_mod"]) == ["q_proj", "custom_mod"]

    def test_convert_megatron_to_hf_empty(self):
        assert convert_megatron_to_hf_target_modules([]) == []

    def test_mapping_values_are_lists(self):
        """Every mapping value must be a list so ``extend`` stays correct."""
        assert all(isinstance(v, list) and v for v in MEGATRON_TO_HF_MODULES.values())


class TestIsLoraAdapterParam:
    @pytest.mark.parametrize(
        "name",
        [
            "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight",
            "decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight",
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
        ],
    )
    def test_is_lora_adapter_param_matches_both_conventions(self, name):
        assert is_lora_adapter_param(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "decoder.layers.0.self_attention.linear_qkv.to_wrap.weight",
            "decoder.layers.0.mlp.linear_fc1.weight",
            "embedding.word_embeddings.weight",
        ],
    )
    def test_is_lora_adapter_param_rejects_base_weights(self, name):
        assert is_lora_adapter_param(name) is False

    def test_is_lora_adapter_param_guards_against_substring_false_positive(self):
        """A base module merely *containing* the substring must not be
        misclassified.

        Dropping such a weight from the sync buckets would silently corrupt the
        rollout model, so the classifier is anchored on the dotted ``.lora_A.``
        segment.
        """
        assert is_lora_adapter_param("decoder.layers.0.mlp.lora_gate.weight") is False
        assert is_lora_adapter_param("decoder.layers.0.mlp.lora_Attention.weight") is False


class TestWriteHfPeftAdapter:
    def test_write_hf_peft_adapter_round_trip(self, tmp_path):
        """Writes a loadable HF-PEFT dir: config JSON + safetensors with same
        tensors."""
        from safetensors.torch import load_file

        merged = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 16),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(16, 8),
        }
        out = write_hf_peft_adapter(
            merged,
            tmp_path / "lora_adapter",
            lora_rank=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        )

        adapter_dir = tmp_path / "lora_adapter"
        assert out == str(adapter_dir)
        assert (adapter_dir / "adapter_config.json").is_file()
        assert (adapter_dir / "adapter_model.safetensors").is_file()

        config = json.loads((adapter_dir / "adapter_config.json").read_text())
        assert config["r"] == 8
        assert config["lora_alpha"] == 16
        assert config["target_modules"] == ["q_proj", "v_proj"]
        assert config["lora_dropout"] == 0.05
        assert config["peft_type"] == "LORA"
        assert config["bias"] == "none"

        loaded = load_file(str(adapter_dir / "adapter_model.safetensors"))
        assert set(loaded) == set(merged)
        for k, v in merged.items():
            assert torch.allclose(loaded[k], v)

    def test_write_hf_peft_adapter_makes_missing_dirs(self, tmp_path):
        """Nested, non-existent target dirs are created."""
        target = tmp_path / "a" / "b" / "lora_adapter"
        write_hf_peft_adapter(
            {"x.lora_A.weight": torch.zeros(2, 2)},
            target,
            lora_rank=1,
            lora_alpha=1,
            target_modules=["q_proj"],
            lora_dropout=0.0,
        )
        assert (target / "adapter_config.json").is_file()


class TestLoraPredicates:
    def test_is_lora_enabled_true_when_rank_positive(self):
        assert is_lora_enabled(Namespace(lora_rank=8)) is True

    def test_is_lora_enabled_false_when_rank_zero_or_missing(self):
        assert is_lora_enabled(Namespace(lora_rank=0)) is False
        assert is_lora_enabled(Namespace()) is False

    def test_is_lora_merge_mode(self):
        assert is_lora_merge_mode(Namespace(lora_merge_mode=True)) is True
        assert is_lora_merge_mode(Namespace(lora_merge_mode=False)) is False
        assert is_lora_merge_mode(Namespace()) is False


class TestCountAdapterParameters:
    def test_count_adapter_parameters(self, monkeypatch):
        """Counts only trainable adapter params against the full param total.

        ``count_adapter_parameters`` imports
        ``megatron.core.utils.unwrap_model`` lazily; stub it so the pure
        counting logic can be exercised without Megatron installed.
        """
        fake_utils = types.ModuleType("megatron.core.utils")
        fake_utils.unwrap_model = lambda m: m
        monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
        monkeypatch.setitem(sys.modules, "megatron.core", types.ModuleType("megatron.core"))
        monkeypatch.setitem(sys.modules, "megatron.core.utils", fake_utils)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # base weight: frozen, not an adapter -> not counted as adapter
                self.linear_qkv_to_wrap_weight = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
                # adapter in/out: trainable -> counted (4*2 + 2*4 = 16)
                self.linear_qkv_adapter_linear_in_weight = torch.nn.Parameter(torch.zeros(2, 4))
                self.linear_qkv_adapter_linear_out_weight = torch.nn.Parameter(torch.zeros(4, 2))

            def named_parameters(self, *a, **k):
                # Names must carry the dotted adapter segment the classifier anchors on.
                yield "linear_qkv.to_wrap.weight", self.linear_qkv_to_wrap_weight
                yield "linear_qkv.adapter.linear_in.weight", self.linear_qkv_adapter_linear_in_weight
                yield "linear_qkv.adapter.linear_out.weight", self.linear_qkv_adapter_linear_out_weight

        adapter_params, total_params, pct = count_adapter_parameters(Tiny())
        assert adapter_params == 16
        assert total_params == 16 + 16  # base 4*4 + adapter 16
        assert pct == pytest.approx(50.0)

    def test_count_adapter_parameters_ignores_frozen_adapter(self, monkeypatch):
        """An adapter-named param that is frozen is not counted as
        trainable."""
        fake_utils = types.ModuleType("megatron.core.utils")
        fake_utils.unwrap_model = lambda m: m
        monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
        monkeypatch.setitem(sys.modules, "megatron.core", types.ModuleType("megatron.core"))
        monkeypatch.setitem(sys.modules, "megatron.core.utils", fake_utils)

        class Frozen(torch.nn.Module):
            def named_parameters(self, *a, **k):
                yield "linear_qkv.adapter.linear_in.weight", torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)

        adapter_params, total_params, pct = count_adapter_parameters(Frozen())
        assert adapter_params == 0
        assert total_params == 4
        assert pct == 0


class TestBridgeParamPrefixes:
    """Base<->adapter name prefixes must round-trip so merge-mode can pair
    them.

    ``hf_weight_iterator_bridge`` imports Megatron at module scope; skip when
    absent.
    """

    def test_base_and_adapter_prefixes_match(self):
        pytest.importorskip("megatron")
        from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import (
            _adapter_base_prefix,
            _base_param_prefix,
        )

        base = "decoder.layers.0.self_attention.linear_qkv"
        assert _base_param_prefix(f"{base}.to_wrap.weight") == base
        assert _adapter_base_prefix(f"{base}.adapter.linear_in.weight") == base
        assert _adapter_base_prefix(f"{base}.adapter.linear_out.weight") == base
        # The whole point: both sides collapse to the same key.
        assert _base_param_prefix(f"{base}.to_wrap.weight") == _adapter_base_prefix(f"{base}.adapter.linear_in.weight")

    def test_base_prefix_strips_grouped_expert_suffix(self):
        pytest.importorskip("megatron")
        from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import _base_param_prefix

        base = "decoder.layers.0.mlp.experts.linear_fc1"
        # Grouped experts carry a trailing weight index (weight3) that must be stripped.
        assert _base_param_prefix(f"{base}.to_wrap.weight3") == base
