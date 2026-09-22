# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the LoRA MoE / GDN helpers added in "LoRA RL MoE".

Scoped to the GPU-free logic whose breakage is silent rather than loud -- a
mis-named target module or a dropped GDN slice makes the rollout engine ignore
part of the adapter instead of raising:

- SGLang-flavored target-module expansion (``in_proj`` -> ``in_proj_qkvz``)
- GDN adapter repacking, incl. the nonzero-gate hard error
- LoRA region scoping (vision tower exclusion for VL models)
- GDN gate masking hooks
- Module-role bucketing used by the log summary
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest


torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from relax.utils.megatron_peft_utils import (  # noqa: E402
    GDN_SGLANG_FUSED_MODULE,
    convert_megatron_to_hf_target_modules,
    convert_megatron_to_sglang_target_modules,
    install_gdn_gate_mask_hooks,
    lora_module_category,
    repack_gdn_adapter_for_sglang,
    scope_target_modules_to_region,
    summarize_lora_modules,
)


_PREFIX = "base_model.model.model.layers.0.linear_attn."


def _gdn_adapter(*, rank: int = 2, in_dim: int = 4, gate_rows: int = 3, gate_nonzero: bool = False) -> dict:
    """A Bridge-style GDN ``in_proj`` export: one shared A, four B slices."""
    shared_a = torch.arange(rank * in_dim, dtype=torch.float32).reshape(rank, in_dim)
    gate = torch.ones(gate_rows, rank) if gate_nonzero else torch.zeros(gate_rows, rank)
    adapter = {}
    for part, out_rows in (("in_proj_qkv", 5), ("in_proj_z", 2)):
        adapter[f"{_PREFIX}{part}.lora_A.weight"] = shared_a.clone()
        adapter[f"{_PREFIX}{part}.lora_B.weight"] = torch.full((out_rows, rank), float(out_rows))
    for part in ("in_proj_b", "in_proj_a"):
        adapter[f"{_PREFIX}{part}.lora_A.weight"] = shared_a.clone()
        adapter[f"{_PREFIX}{part}.lora_B.weight"] = gate.clone()
    return adapter


class TestConvertMegatronToSglangTargetModules:
    def test_gdn_in_proj_uses_the_sglang_fused_name(self):
        """SGLang keeps GDN's input projection fused, HF splits it four
        ways."""
        assert convert_megatron_to_sglang_target_modules(["in_proj"]) == ["in_proj_qkvz"]
        assert convert_megatron_to_hf_target_modules(["in_proj"]) == [
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
        ]

    def test_non_overridden_modules_keep_the_hf_expansion(self):
        """Only ``in_proj`` differs; everything else follows the HF mapping."""
        megatron = ["linear_qkv", "linear_fc1", "out_proj", "router"]
        assert convert_megatron_to_sglang_target_modules(megatron) == convert_megatron_to_hf_target_modules(megatron)

    def test_dedups_preserving_order(self):
        assert convert_megatron_to_sglang_target_modules(["in_proj", "linear_qkv", "in_proj"]) == [
            "in_proj_qkvz",
            "q_proj",
            "k_proj",
            "v_proj",
        ]


class TestRepackGdnAdapterForSglang:
    def test_fuses_qkv_and_z_into_one_module(self):
        """``lora_B`` concatenates along the output dim, ``lora_A`` is emitted
        once."""
        fused = repack_gdn_adapter_for_sglang(_gdn_adapter())

        b = fused[f"{_PREFIX}{GDN_SGLANG_FUSED_MODULE}.lora_B.weight"]
        a = fused[f"{_PREFIX}{GDN_SGLANG_FUSED_MODULE}.lora_A.weight"]
        assert b.shape == (7, 2)  # 5 (qkv) + 2 (z)
        assert torch.equal(b[:5], torch.full((5, 2), 5.0))
        assert torch.equal(b[5:], torch.full((2, 2), 2.0))
        assert a.shape == (2, 4)

    def test_drops_the_gate_slices_and_leaves_nothing_else_behind(self):
        """The b/a slices SGLang cannot host disappear; unrelated keys
        survive."""
        adapter = _gdn_adapter()
        adapter["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"] = torch.zeros(2, 4)
        fused = repack_gdn_adapter_for_sglang(adapter)

        assert not [k for k in fused if "in_proj_qkv." in k or "in_proj_z." in k]
        assert not [k for k in fused if "in_proj_b." in k or "in_proj_a." in k]
        assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in fused

    def test_adapter_without_gdn_passes_through_unchanged(self):
        adapter = {"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(4, 2)}
        assert repack_gdn_adapter_for_sglang(adapter) is adapter

    def test_nonzero_gate_delta_is_a_hard_error(self):
        """A trained gate delta would be dropped -> rollout diverges from
        training."""
        with pytest.raises(ValueError, match="cannot host LoRA on the GDN b/a slices"):
            repack_gdn_adapter_for_sglang(_gdn_adapter(gate_nonzero=True))

    def test_incomplete_group_is_a_hard_error(self):
        adapter = _gdn_adapter()
        del adapter[f"{_PREFIX}in_proj_a.lora_B.weight"]
        with pytest.raises(ValueError, match="Incomplete GDN in_proj adapter group"):
            repack_gdn_adapter_for_sglang(adapter)


# --------------------------------------------------------------------------
# scope_target_modules_to_region
# --------------------------------------------------------------------------


class _Leaf(nn.Module):
    pass


class _VLModel(nn.Module):
    """Minimal VL tree: a vision tower and a language backbone sharing leaf
    names."""

    def __init__(self):
        super().__init__()
        self.vision_model = nn.ModuleDict({"linear_qkv": _Leaf(), "linear_fc1": _Leaf()})
        self.decoder = nn.ModuleDict({"linear_qkv": _Leaf(), "mlp": _Leaf()})


@pytest.fixture()
def _stub_wildcard_match(monkeypatch):
    """Stand in for ``megatron.bridge.peft.utils.wildcard_match`` (megatron is
    not installed).

    Mirrors the upstream implementation: the pattern is anchored and ``*`` is
    the only metacharacter, so a plain name is a pattern that must match the
    full path exactly.
    """
    import re

    for name in ("megatron", "megatron.bridge", "megatron.bridge.peft", "megatron.bridge.peft.utils"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or ModuleType(name))
    monkeypatch.setattr(
        sys.modules["megatron.bridge.peft.utils"],
        "wildcard_match",
        lambda pattern, name: re.match("^" + pattern.replace("*", "(.*)") + "$", name) is not None,
        raising=False,
    )


class TestScopeTargetModulesToRegion:
    def test_all_is_a_passthrough(self):
        """``all`` must not walk the model -- the leaf-name list is used as-
        is."""
        targets = ["linear_qkv"]
        assert scope_target_modules_to_region(object(), targets, "all") is targets

    def test_language_scope_excludes_the_vision_tower(self, _stub_wildcard_match):
        scoped = scope_target_modules_to_region(_VLModel(), ["linear_qkv", "linear_fc1"], "language")
        assert scoped == ["decoder.linear_qkv"]

    def test_vision_scope_selects_only_the_vision_tower(self, _stub_wildcard_match):
        scoped = scope_target_modules_to_region(_VLModel(), ["linear_qkv", "linear_fc1"], "vision")
        assert sorted(scoped) == ["vision_model.linear_fc1", "vision_model.linear_qkv"]

    def test_empty_region_yields_a_non_matching_sentinel(self, _stub_wildcard_match):
        """A text-only chunk under ``vision`` must wrap nothing, not
        everything."""
        text_only = nn.ModuleDict({"decoder": nn.ModuleDict({"linear_qkv": _Leaf()})})
        scoped = scope_target_modules_to_region(text_only, ["linear_qkv"], "vision")
        assert scoped == ["__relax_lora_no_match__"]
        assert not any(name in scoped for name, _ in text_only.named_modules())


# --------------------------------------------------------------------------
# install_gdn_gate_mask_hooks
# --------------------------------------------------------------------------


class _Adapter(nn.Module):
    def forward(self, x):
        return x.clone()


@pytest.fixture()
def _stub_gated_delta_net(monkeypatch):
    """Install a stub ``GatedDeltaNet`` base class the helper can
    ``isinstance`` against."""

    class GatedDeltaNet(nn.Module):
        pass

    for name in ("megatron", "megatron.core", "megatron.core.ssm", "megatron.core.ssm.gated_delta_net"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or ModuleType(name))
    monkeypatch.setattr(
        sys.modules["megatron.core.ssm.gated_delta_net"], "GatedDeltaNet", GatedDeltaNet, raising=False
    )
    return GatedDeltaNet


def _gdn_model(base_cls, *, wrapped: bool, num_v_heads_local_tp: int = 2):
    layer = base_cls()
    layer.num_v_heads_local_tp = num_v_heads_local_tp
    layer.in_proj = nn.Linear(4, 8)
    if wrapped:
        layer.in_proj.adapter = _Adapter()
    return nn.Sequential(layer)


class TestInstallGdnGateMaskHooks:
    def test_masks_the_trailing_gate_rows_of_the_adapter_output(self, _stub_gated_delta_net):
        model = _gdn_model(_stub_gated_delta_net, wrapped=True, num_v_heads_local_tp=2)
        assert install_gdn_gate_mask_hooks(model) == 1

        adapter = model[0].in_proj.adapter
        out = adapter(torch.ones(3, 8))
        assert torch.equal(out[:, -4:], torch.zeros(3, 4))  # 2 * num_v_heads_local_tp
        assert torch.equal(out[:, :-4], torch.ones(3, 4))

    def test_is_idempotent(self, _stub_gated_delta_net):
        """Re-running (e.g. per VP chunk) must not stack a second hook."""
        model = _gdn_model(_stub_gated_delta_net, wrapped=True)
        assert install_gdn_gate_mask_hooks(model) == 1
        assert install_gdn_gate_mask_hooks(model) == 0
        assert len(model[0].in_proj.adapter._forward_hooks) == 1

    def test_unwrapped_in_proj_is_skipped(self, _stub_gated_delta_net):
        """``in_proj`` outside ``--lora-target-modules`` has no adapter to
        mask."""
        assert install_gdn_gate_mask_hooks(_gdn_model(_stub_gated_delta_net, wrapped=False)) == 0

    def test_returns_zero_when_megatron_has_no_gdn(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "megatron.core.ssm.gated_delta_net", None)
        assert install_gdn_gate_mask_hooks(nn.Sequential(nn.Linear(2, 2))) == 0


# --------------------------------------------------------------------------
# lora_module_category / summarize_lora_modules
# --------------------------------------------------------------------------


class TestLoraModuleCategory:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            # shared_experts must win over the generic routed-expert rule.
            ("decoder.layers.0.mlp.shared_experts.linear_fc1.weight", "shared_expert"),
            ("decoder.layers.0.mlp.experts.linear_fc1.weight0", "routed_expert"),
            ("decoder.layers.0.mlp.router.weight", "router"),
            ("decoder.layers.0.mlp.linear_fc2.weight", "mlp"),
            ("decoder.layers.0.linear_attn.in_proj.weight", "gdn"),
            ("decoder.layers.0.self_attention.linear_qkv.weight", "attention"),
            # Same module, two naming sites: injection-time (leading) and sync-time (prefixed).
            ("vision_model.blocks.0.linear_qkv.weight", "vision"),
            ("module.module.vision_model.blocks.0.linear_qkv.weight", "vision"),
        ],
    )
    def test_buckets_by_architectural_role(self, name, expected):
        assert lora_module_category(name) == expected


class TestSummarizeLoraModules:
    def test_counts_modules_not_params(self):
        """``linear_in``/``linear_out`` and packed ``weight{N}`` collapse to
        one module."""
        names = [
            "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight",
            "decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight",
            "decoder.layers.0.mlp.experts.linear_fc1.lora_A.weight0",
            "decoder.layers.0.mlp.experts.linear_fc1.lora_B.weight0",
            "decoder.layers.0.mlp.experts.linear_fc1.lora_A.weight1",
            "decoder.layers.0.mlp.shared_experts.linear_fc1.lora_A.weight",
        ]
        assert summarize_lora_modules(names) == {"attention": 1, "routed_expert": 1, "shared_expert": 1}

    def test_empty_input(self):
        assert summarize_lora_modules([]) == {}
