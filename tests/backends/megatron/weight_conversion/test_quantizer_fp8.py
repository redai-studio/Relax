# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the FP8 ``modules_to_not_convert`` (ignore-list) quantization
path.

The commit that added Qwen3.6 multimodal QAT support introduced an ignore-list
based FP8 quantizer: when ``quantization_config`` carries ``modules_to_not_convert``
the converter quantizes *every* 2-D float weight except the modules named in that
list. HF checkpoints store routed experts fused (``...experts.<i>.gate_proj``), so
``_checkpoint_module_name`` maps those back to the fused HF module name
(``...experts.gate_up_proj`` / ``.down_proj``) used inside the ignore list.

We stub the triton FP8 kernel and the sglang deps so the module imports without a
GPU, and patch ``_quantize_param`` so the filtering logic is tested in isolation.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Import quantizer_fp8 with its heavy ancestors / leaf deps stubbed so it loads
# without triton, sglang, or the real weight_conversion package __init__.
# ---------------------------------------------------------------------------

_MODULE_NAME = "relax.backends.megatron.weight_conversion.processors.quantizer_fp8"
_SOURCE = (
    pathlib.Path(__file__).resolve().parents[4]
    / "relax/backends/megatron/weight_conversion/processors/quantizer_fp8.py"
)

_STUB_PACKAGES = [
    "relax",
    "relax.backends",
    "relax.backends.megatron",
    "relax.backends.megatron.kernels",
    "relax.backends.megatron.kernels.fp8_kernel",
    "relax.backends.megatron.sglang",
    "relax.backends.megatron.weight_conversion",
    "relax.backends.megatron.weight_conversion.processors",
]


def _load_module():
    saved = {name: sys.modules.get(name) for name in _STUB_PACKAGES}
    for name in _STUB_PACKAGES:
        sys.modules[name] = MagicMock()
    try:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


qf = _load_module()


# ---------------------------------------------------------------------------
# Online weight-scale policy: stock TE blockwise recipe vs checkpoint metadata
# ---------------------------------------------------------------------------


class TestResolveOnlineWeightScaleFormat:
    _BLOCK = [128, 128]

    def test_bf16_defaults_to_pre_5392_ordinary_scales(self, monkeypatch):
        monkeypatch.delenv("RELAX_FP8_BF16_SENDER_PRESERVE_UE8M0_GRID", raising=False)

        resolved = qf._resolve_online_weight_scale_fmt(
            SimpleNamespace(fp8=None, fp8_recipe="delayed"), "ue8m0", self._BLOCK
        )

        assert resolved is None

    def test_bf16_can_opt_in_to_checkpoint_ue8m0_grid(self, monkeypatch):
        monkeypatch.setenv("RELAX_FP8_BF16_SENDER_PRESERVE_UE8M0_GRID", "1")

        resolved = qf._resolve_online_weight_scale_fmt(
            SimpleNamespace(fp8=None, fp8_recipe="delayed"), "ue8m0", self._BLOCK
        )

        assert resolved == "ue8m0"

    def test_bf16_rejects_invalid_ue8m0_grid_switch(self, monkeypatch):
        monkeypatch.setenv("RELAX_FP8_BF16_SENDER_PRESERVE_UE8M0_GRID", "yes")

        with pytest.raises(ValueError, match="must be 0 or 1"):
            qf._resolve_online_weight_scale_fmt(SimpleNamespace(fp8=None, fp8_recipe="delayed"), "ue8m0", self._BLOCK)

    def test_blockwise_fp32_scales_override_checkpoint_ue8m0(self, monkeypatch):
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
        monkeypatch.setenv("RELAX_FP8_BF16_SENDER_PRESERVE_UE8M0_GRID", "not-read-by-fp8")
        monkeypatch.setattr(qf, "should_deepgemm_weight_requant_ue8m0", lambda **_: False)

        resolved = qf._resolve_online_weight_scale_fmt(
            SimpleNamespace(fp8="e4m3", fp8_recipe="blockwise"), "ue8m0", self._BLOCK
        )

        assert resolved is None

    def test_blockwise_pow2_scales_keep_checkpoint_ue8m0(self, monkeypatch):
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0")

        resolved = qf._resolve_online_weight_scale_fmt(
            SimpleNamespace(fp8="e4m3", fp8_recipe="blockwise"), "ue8m0", self._BLOCK
        )

        assert resolved == "ue8m0"

    @pytest.mark.parametrize("bf16_switch", ["0", "1", "not-read-by-fp8"])
    def test_custom_recipe_does_not_follow_bf16_or_stock_blockwise_policy(self, monkeypatch, bf16_switch):
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
        monkeypatch.setenv("RELAX_FP8_BF16_SENDER_PRESERVE_UE8M0_GRID", bf16_switch)

        resolved = qf._resolve_online_weight_scale_fmt(
            SimpleNamespace(fp8="e4m3", fp8_recipe="custom"), "ue8m0", self._BLOCK
        )

        assert resolved == "ue8m0"

    def test_existing_ordinary_scale_format_is_unchanged(self, monkeypatch):
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
        monkeypatch.setattr(
            qf,
            "should_deepgemm_weight_requant_ue8m0",
            lambda **_: (_ for _ in ()).throw(AssertionError("runtime check must not run")),
        )

        resolved = qf._resolve_online_weight_scale_fmt(
            SimpleNamespace(fp8="e4m3", fp8_recipe="blockwise"), None, self._BLOCK
        )

        assert resolved is None

    def test_blockwise_fp32_scales_require_matching_block_shape(self, monkeypatch):
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")

        with pytest.raises(RuntimeError, match="requires a 128x128 online weight block"):
            qf._resolve_online_weight_scale_fmt(
                SimpleNamespace(fp8="e4m3", fp8_recipe="blockwise"), "ue8m0", [64, 128]
            )

    def test_blockwise_fp32_scales_reject_packed_ue8m0_runtime(self, monkeypatch):
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
        monkeypatch.setattr(qf, "should_deepgemm_weight_requant_ue8m0", lambda **_: True)

        with pytest.raises(RuntimeError, match="requires packed UE8M0 scales"):
            qf._resolve_online_weight_scale_fmt(
                SimpleNamespace(fp8="e4m3", fp8_recipe="blockwise"), "ue8m0", self._BLOCK
            )

    def test_blockwise_fp32_scales_require_runtime_capability_check(self, monkeypatch):
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
        monkeypatch.setattr(qf, "should_deepgemm_weight_requant_ue8m0", None)

        with pytest.raises(RuntimeError, match="runtime capability check is unavailable"):
            qf._resolve_online_weight_scale_fmt(
                SimpleNamespace(fp8="e4m3", fp8_recipe="blockwise"), "ue8m0", self._BLOCK
            )


# ---------------------------------------------------------------------------
# _checkpoint_module_name: HF fused-module name resolution
# ---------------------------------------------------------------------------


class TestCheckpointModuleName:
    @pytest.mark.parametrize(
        "hf_name, expected",
        [
            # split expert projections collapse to the fused HF module name
            ("model.layers.0.mlp.experts.3.gate_proj.weight", "model.layers.0.mlp.experts.gate_up_proj"),
            ("model.layers.0.mlp.experts.3.up_proj.weight", "model.layers.0.mlp.experts.gate_up_proj"),
            ("model.layers.0.mlp.experts.3.down_proj.weight", "model.layers.0.mlp.experts.down_proj"),
            # multi-digit expert index
            ("m.mlp.experts.127.down_proj.weight", "m.mlp.experts.down_proj"),
            # non-expert weights just get the ``.weight`` suffix stripped
            ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj"),
            ("model.embed_tokens.weight", "model.embed_tokens"),
            # names without a ``.weight`` suffix are returned as the module itself
            ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj"),
            # ``.weight`` stripping happens before the expert regex match
            ("m.mlp.experts.1.up_proj", "m.mlp.experts.gate_up_proj"),
        ],
    )
    def test_names(self, hf_name, expected):
        assert qf._checkpoint_module_name(hf_name) == expected

    def test_shared_experts_are_not_treated_as_routed(self):
        # ``shared_experts`` (plural, no numeric index) must not match the routed
        # expert regex; it stays a plain module name.
        name = "model.layers.0.mlp.shared_experts.gate_proj.weight"
        assert qf._checkpoint_module_name(name) == "model.layers.0.mlp.shared_experts.gate_proj"


# ---------------------------------------------------------------------------
# _quantize_params_fp8_by_ignore_list: filtering logic
# ---------------------------------------------------------------------------


class TestQuantizeByIgnoreList:
    @staticmethod
    def _stub_quantize_param(name, param, weight_block_size, scale_fmt=None):
        """Deterministic marker so we can assert *which* params were
        quantized."""
        return [(name, "Q"), (name.replace(".weight", ".weight_scale"), "S")]

    def _run(self, monkeypatch, params, ignore):
        monkeypatch.setattr(qf, "_quantize_param", self._stub_quantize_param)
        return qf._quantize_params_fp8_by_ignore_list(params, set(ignore), weight_block_size=None, scale_fmt=None)

    def test_quantizes_eligible_and_passes_through_rest(self, monkeypatch):
        w2d = torch.zeros(4, 4, dtype=torch.float32)
        bf16 = torch.zeros(4, 4, dtype=torch.bfloat16)
        params = [
            ("m.experts.0.gate_proj.weight", w2d),  # eligible -> quantized
            ("m.self_attn.k_proj.weight", bf16),  # eligible (bf16) -> quantized
            ("m.experts.0.up_proj.weight", w2d),  # ignored via fused name -> passthrough
            ("m.norm.weight", torch.zeros(4)),  # 1-D -> passthrough
            ("m.self_attn.q_proj.bias", w2d),  # not ``.weight`` -> passthrough
            ("m.int.weight", torch.zeros(4, 4, dtype=torch.int8)),  # non-float -> passthrough
        ]
        # ignore the fused gate_up module -> up_proj (and gate_proj) map into it.
        out = self._run(monkeypatch, params, ignore={"m.experts.gate_up_proj"})
        out_map = dict(out)

        # gate_proj is quantized only if NOT ignored; here gate_up_proj IS ignored,
        # so BOTH gate_proj and up_proj pass through untouched.
        assert ("m.experts.0.gate_proj.weight", w2d) in out
        assert ("m.experts.0.up_proj.weight", w2d) in out
        # k_proj is eligible and not ignored -> quantized (marker + scale emitted)
        assert out_map["m.self_attn.k_proj.weight"] == "Q"
        assert out_map["m.self_attn.k_proj.weight_scale"] == "S"
        # passthroughs keep their original tensor objects
        assert ("m.norm.weight", params[3][1]) in out
        assert ("m.self_attn.q_proj.bias", w2d) in out
        assert ("m.int.weight", params[5][1]) in out

    def test_ignore_list_targets_fused_expert_name(self, monkeypatch):
        w2d = torch.zeros(2, 2, dtype=torch.float32)
        params = [
            ("m.experts.0.gate_proj.weight", w2d),
            ("m.experts.0.down_proj.weight", w2d),
        ]
        # Only down_proj is protected; gate_proj should still be quantized.
        out = dict(self._run(monkeypatch, params, ignore={"m.experts.down_proj"}))
        assert out["m.experts.0.gate_proj.weight"] == "Q"
        assert out["m.experts.0.down_proj.weight"] is w2d  # passthrough

    def test_empty_ignore_quantizes_all_eligible(self, monkeypatch):
        w2d = torch.zeros(2, 2, dtype=torch.float32)
        params = [("m.experts.0.gate_proj.weight", w2d), ("m.experts.0.down_proj.weight", w2d)]
        out = dict(self._run(monkeypatch, params, ignore=set()))
        assert out["m.experts.0.gate_proj.weight"] == "Q"
        assert out["m.experts.0.down_proj.weight"] == "Q"


# ---------------------------------------------------------------------------
# quantize_params_fp8: dispatch into the ignore-list path
# ---------------------------------------------------------------------------


class TestDispatch:
    _BASE_CONFIG = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}

    def test_routes_to_ignore_list_when_present(self, monkeypatch):
        recorded = {}

        def _recorder(converted, ignore_set, weight_block_size, scale_fmt):
            recorded["args"] = (converted, ignore_set, weight_block_size, scale_fmt)
            return "SENTINEL"

        monkeypatch.setattr(qf, "_quantize_params_fp8_by_ignore_list", _recorder)

        converted = [("m.experts.0.gate_proj.weight", torch.zeros(2, 2))]
        config = {**self._BASE_CONFIG, "modules_to_not_convert": ["a.b", "c.d"], "weight_block_size": [128, 128]}

        result = qf.quantize_params_fp8(
            args=None, megatron_name="whatever", converted_named_params=converted, quantization_config=config
        )

        assert result == "SENTINEL"
        got_converted, got_ignore, got_block, got_scale_fmt = recorded["args"]
        assert got_converted is converted
        assert got_ignore == {"a.b", "c.d"}  # list -> set
        assert got_block == [128, 128]
        assert got_scale_fmt is None

    def test_blockwise_fp32_scale_policy_reaches_quantizer(self, monkeypatch):
        recorded = {}

        def _recorder(converted, ignore_set, weight_block_size, scale_fmt):
            recorded["scale_fmt"] = scale_fmt
            return "SENTINEL"

        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
        monkeypatch.setattr(qf, "should_deepgemm_weight_requant_ue8m0", lambda **_: False)
        monkeypatch.setattr(qf, "_quantize_params_fp8_by_ignore_list", _recorder)
        config = {
            **self._BASE_CONFIG,
            "modules_to_not_convert": ["model.embed_tokens"],
            "weight_block_size": [128, 128],
            "scale_fmt": "ue8m0",
        }

        result = qf.quantize_params_fp8(
            args=SimpleNamespace(fp8="e4m3", fp8_recipe="blockwise"),
            megatron_name="whatever",
            converted_named_params=[],
            quantization_config=config,
        )

        assert result == "SENTINEL"
        assert recorded["scale_fmt"] is None

    def test_blockwise_fp32_scale_policy_reaches_legacy_quantizer(self, monkeypatch):
        recorded = {}

        def _recorder(name, param, weight_block_size, scale_fmt):
            recorded["args"] = (name, param, weight_block_size, scale_fmt)
            return [(name, "Q")]

        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
        monkeypatch.setattr(qf, "should_deepgemm_weight_requant_ue8m0", lambda **_: False)
        monkeypatch.setattr(qf, "_quantize_param", _recorder)
        weight = torch.zeros(128, 128, dtype=torch.bfloat16)
        name = "model.layers.0.self_attn.o_proj.weight"
        config = {
            **self._BASE_CONFIG,
            "weight_block_size": [128, 128],
            "scale_fmt": "ue8m0",
        }

        result = qf.quantize_params_fp8(
            args=SimpleNamespace(fp8="e4m3", fp8_recipe="blockwise"),
            megatron_name="module.module.decoder.layers.0.self_attention.linear_proj.weight",
            converted_named_params=[(name, weight)],
            quantization_config=config,
        )

        assert result == [(name, "Q")]
        got_name, got_weight, got_block, got_scale_fmt = recorded["args"]
        assert got_name == name
        assert got_weight is weight
        assert got_block == [128, 128]
        assert got_scale_fmt is None

    def test_empty_ignore_list_uses_legacy_path(self, monkeypatch):
        # An empty ``modules_to_not_convert`` is falsy -> legacy per-name matching.
        marker = object()

        def _should_not_run(*_a, **_k):
            raise AssertionError("ignore-list path must not be taken for empty list")

        monkeypatch.setattr(qf, "_quantize_params_fp8_by_ignore_list", _should_not_run)
        config = {**self._BASE_CONFIG, "modules_to_not_convert": []}

        # A non-matching megatron name returns the params unchanged (legacy behaviour).
        converted = [("unmatched", marker)]
        out = qf.quantize_params_fp8(
            args=None,
            megatron_name="not.a.decoder.layer",
            converted_named_params=converted,
            quantization_config=config,
        )
        assert out is converted

    def test_no_key_uses_legacy_path(self, monkeypatch):
        monkeypatch.setattr(
            qf,
            "_quantize_params_fp8_by_ignore_list",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        converted = [("x", object())]
        out = qf.quantize_params_fp8(
            args=None,
            megatron_name="not.a.decoder.layer",
            converted_named_params=converted,
            quantization_config=dict(self._BASE_CONFIG),
        )
        assert out is converted


# ---------------------------------------------------------------------------
# _quantize_param: quantization grid vs runtime scale storage
# ---------------------------------------------------------------------------


class TestQuantizeParamScaleFormat:
    _NAME = "model.layers.0.self_attn.q_proj.weight"
    _BLOCK = [128, 128]

    @staticmethod
    def _weight():
        return torch.ones(128, 128, dtype=torch.bfloat16)

    def test_ue8m0_grid_is_used_on_hopper_without_packing(self, monkeypatch):
        qweight = torch.ones(128, 128, dtype=torch.float8_e4m3fn)
        scale = torch.ones(1, 1, dtype=torch.float32)
        monkeypatch.setattr(qf, "should_deepgemm_weight_requant_ue8m0", lambda **_: False)
        monkeypatch.setattr(qf, "quant_weight_ue8m0", lambda *_a, **_k: (qweight, scale))
        monkeypatch.setattr(
            qf,
            "blockwise_cast_to_fp8_triton",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must preserve the UE8M0 grid")),
        )
        monkeypatch.setattr(
            qf,
            "transform_scale_ue8m0",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Hopper stores FP32 scales")),
        )

        out = dict(qf._quantize_param(self._NAME, self._weight(), self._BLOCK, scale_fmt="ue8m0"))

        assert out[self._NAME] is qweight
        assert out[self._NAME.replace(".weight", ".weight_scale_inv")] is scale

    def test_blackwell_runtime_packs_ue8m0_scale(self, monkeypatch):
        qweight = torch.ones(128, 128, dtype=torch.float8_e4m3fn)
        scale = torch.ones(1, 1, dtype=torch.float32)
        packed = torch.ones(1, 1, dtype=torch.int32)
        monkeypatch.setattr(qf, "should_deepgemm_weight_requant_ue8m0", lambda **_: True)
        monkeypatch.setattr(qf, "quant_weight_ue8m0", lambda *_a, **_k: (qweight, scale))
        monkeypatch.setattr(qf, "transform_scale_ue8m0", lambda value, mn: packed)

        out = dict(qf._quantize_param(self._NAME, self._weight(), self._BLOCK))

        assert out[self._NAME] is qweight
        assert out[self._NAME.replace(".weight", ".weight_scale_inv")] is packed

    def test_plain_scale_config_keeps_legacy_quantizer(self, monkeypatch):
        qweight = torch.ones(128, 128, dtype=torch.float8_e4m3fn)
        scale = torch.ones(1, 1, dtype=torch.float32)
        monkeypatch.setattr(qf, "should_deepgemm_weight_requant_ue8m0", lambda **_: False)
        monkeypatch.setattr(qf, "blockwise_cast_to_fp8_triton", lambda *_a, **_k: (qweight, scale))
        monkeypatch.setattr(
            qf,
            "quant_weight_ue8m0",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("plain scales use the legacy path")),
        )

        out = dict(qf._quantize_param(self._NAME, self._weight(), self._BLOCK))

        assert out[self._NAME] is qweight
        assert out[self._NAME.replace(".weight", ".weight_scale_inv")] is scale
