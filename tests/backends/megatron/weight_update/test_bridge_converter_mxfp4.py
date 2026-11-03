# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the Kimi K3 mxfp4-pack-quantized weight-push path.

Covers the BridgeConverter routed-expert branch (BF16 -> packed MXFP4 pairs via
Bridge's ``megatron_to_hf_quant``), the ``quantize_params`` passthrough that
keeps the INT4 packer away from mxfp4 releases, and the nested
``text_config.quantization_config`` resolution used to find K3's push config.
"""

from __future__ import annotations

import contextlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


torch = pytest.importorskip("torch")

pytest.importorskip("megatron.core")

from relax.backends.megatron.weight_conversion.processors import quantize_params  # noqa: E402
from relax.backends.megatron.weight_update.bridge_converter import (  # noqa: E402
    BridgeConverter,
    _is_mxfp4_routed_expert,
    _Mxfp4OnlineQuantizer,
)
from relax.utils.quant_cast import augment_compressed_tensors_ignore, read_quantization_config  # noqa: E402


_MegatronParamMapping = pytest.importorskip("megatron.bridge.models.conversion.param_mapping").MegatronParamMapping


@contextlib.contextmanager
def _fake_quant_kernel(fake):
    """Swap the bridge quant kernel module, restoring sys.modules exactly.

    Restoring a missing entry must pop, not assign None — a None entry poisons
    every later ``from ... import`` of that module in the same pytest session.
    """
    key = "megatron.bridge.models.conversion.quantization_utils"
    saved = sys.modules.get(key)
    sys.modules[key] = MagicMock(quantize_mxfp4_e2m1_like_scale=fake)
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = saved


# ---------------------------------------------------------------------------
# quantize_params passthrough
# ---------------------------------------------------------------------------


def test_quantize_params_mxfp4_format_passthrough():
    """The INT4 packer must not touch mxfp4 releases: BridgeConverter emits
    packed/scale pairs itself and the BF16 router gate must stay BF16."""
    config = {"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized", "ignore": []}
    named = [("moe.gate.weight", torch.zeros(4, 8, dtype=torch.bfloat16))]
    out = quantize_params(None, "unused", named, config)
    assert out == named
    assert not any(name.endswith("_packed") for name, _ in out)


def test_quantize_params_int4_still_packs():
    """Guard the guard: without the mxfp4 format the compressed-tensors path
    still packs (K2.6 behavior unchanged)."""
    pytest.importorskip("fake_int4_quant_cuda")
    if not torch.cuda.is_available():
        pytest.skip("fake_int4_quant_cuda is a CUDA kernel")
    config = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"group_size": 32, "symmetric": True}}},
    }
    named = [("mlp.experts.0.w1.weight", torch.ones(32, 64, dtype=torch.bfloat16, device="cuda"))]
    out = quantize_params(None, "unused", named, config)
    assert any(name.endswith(".weight_packed") for name, _ in out)


# ---------------------------------------------------------------------------
# Nested quantization_config resolution (K3 nests it under text_config)
# ---------------------------------------------------------------------------


def _write_config(tmp_path, config: dict):
    (tmp_path / "config.json").write_text(json.dumps(config))


def test_read_quantization_config_falls_back_to_text_config(tmp_path):
    nested = {"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized"}
    _write_config(tmp_path, {"quantization_config": None, "text_config": {"quantization_config": nested}})
    assert read_quantization_config(tmp_path) == nested


def test_read_quantization_config_top_level_wins(tmp_path):
    top = {"quant_method": "compressed-tensors"}
    _write_config(tmp_path, {"quantization_config": top, "text_config": {"quantization_config": {"other": 1}}})
    assert read_quantization_config(tmp_path) == top


def test_read_quantization_config_missing(tmp_path):
    assert read_quantization_config(tmp_path) is None


def test_read_quantization_config_null_text_config(tmp_path):
    """An explicit ``"text_config": null`` must not crash the fallback
    lookup."""
    _write_config(tmp_path, {"quantization_config": None, "text_config": None})
    assert read_quantization_config(tmp_path) is None


def test_augment_resolves_none_from_nested_config(tmp_path):
    """actor.py reads hf_config.quantization_config, which is None for K3;
    augment must resolve the nested config instead of passing None through."""
    nested = {"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized", "ignore": []}
    _write_config(tmp_path, {"quantization_config": None, "text_config": {"quantization_config": nested}})
    resolved = augment_compressed_tensors_ignore(None, tmp_path)
    assert resolved is not None
    assert resolved["quant_method"] == "compressed-tensors"
    assert resolved["format"] == "mxfp4-pack-quantized"


# ---------------------------------------------------------------------------
# Routed-expert detection and the BridgeConverter branch
# ---------------------------------------------------------------------------


def _expert_mapping(hf_param="language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight", **attrs):
    return SimpleNamespace(is_expert=True, is_adapter=False, hf_param=hf_param, **attrs)


def test_is_mxfp4_routed_expert_positive():
    assert _is_mxfp4_routed_expert(_expert_mapping())
    # The VL bridge flavor leaves the HF name unprefixed.
    assert _is_mxfp4_routed_expert(_expert_mapping("model.layers.3.block_sparse_moe.experts.127.w2.weight"))


def test_is_mxfp4_routed_expert_rejects_non_expert_and_adapter():
    assert not _is_mxfp4_routed_expert(SimpleNamespace(is_expert=False, hf_param="a.weight"))
    assert not _is_mxfp4_routed_expert(
        SimpleNamespace(is_expert=True, is_adapter=True, hf_param="...experts.0.w1.weight")
    )
    assert not _is_mxfp4_routed_expert(_expert_mapping("language_model.model.layers.1.block_sparse_moe.gate.weight"))


def test_convert_mxfp4_expert_emits_native_pair():
    """The branch renames Bridge's ``w*.weight``/``w*.weight_scale_inv``
    outputs to the native ``w*.weight_packed``/``w*.weight_scale`` release
    keys."""
    converter = BridgeConverter.__new__(BridgeConverter)
    converter._mxfp4_quantizer = _Mxfp4OnlineQuantizer()

    packed = torch.zeros(3072, 1792, dtype=torch.int8)
    scale = torch.zeros(3072, 112, dtype=torch.uint8)
    mapping = _expert_mapping(
        megatron_to_hf_quant=MagicMock(
            return_value={
                "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight": packed,
                "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale_inv": scale,
            }
        )
    )
    task = SimpleNamespace(megatron_module=None)
    param = torch.zeros(3072, 3584, dtype=torch.bfloat16)

    out = converter._convert_mxfp4_expert(
        "decoder.layers.1.mlp.experts.w1", "module.module.decoder.layers.1.mlp.experts.w1", param, task, mapping
    )

    assert [(name, tensor.dtype) for name, tensor in out] == [
        ("language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed", torch.uint8),
        ("language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale", torch.uint8),
    ]
    assert out[0][1].data_ptr() == packed.data_ptr()  # view, not a copy
    mapping.megatron_to_hf_quant.assert_called_once()


class _GatherSentinel:
    """Provides ``gather_from_ep_ranks`` via inheritance, mirroring the real
    bridge hierarchy (MegatronParamMapping defines it; leaf classes don't), so
    the converter's class-level patch can be deleted again to reveal it."""

    def gather_from_ep_ranks(self, megatron_weights, megatron_module, hf_param_name):
        raise AssertionError("gather_from_ep_ranks must stay noop-patched during conversion")


class _CollectiveProbeMapping(_GatherSentinel, _MegatronParamMapping):
    HF_WEIGHT = "model.layers.0.block_sparse_moe.experts.0.w1.weight"

    def __init__(self):
        # The real MegatronParamMapping derives is_expert/is_adapter from
        # megatron_param via read-only properties; this spelling makes them
        # True/False there and is inert under the stub class.
        self.megatron_param = "decoder.layers.0.mlp.experts.linear_fc1"
        self.hf_param = self.HF_WEIGHT
        self.pp_group = object()
        self._tp_group = object()
        self._etp_group = object()
        self.ep_group = object()
        self.groups_seen = None

    # The real MegatronParamMapping is an ABC; these two are never called on
    # this path but must exist for instantiation.
    def hf_to_megatron(self, *args, **kwargs):
        raise NotImplementedError

    def megatron_to_hf(self, *args, **kwargs):
        raise NotImplementedError

    def megatron_to_hf_quant(self, weight, module, predicate, quantizer, block_size):
        self.groups_seen = (self.pp_group, self._tp_group, self._etp_group, self.ep_group)
        # The noop patch turns the EP gather into a local passthrough dict.
        gathered = self.gather_from_ep_ranks(weight, module, self.hf_param)
        assert gathered[self.HF_WEIGHT] is weight
        return {
            self.HF_WEIGHT: torch.zeros(4, 2, dtype=torch.int8),
            self.HF_WEIGHT + "_scale_inv": torch.zeros(4, 1, dtype=torch.uint8),
        }


def test_convert_mxfp4_expert_disables_mapping_collectives():
    """Expert conversion runs on the src rank only, so a live mapping
    collective would run on the WORLD group and deadlock.

    Every process group must be None during the call and restored afterwards,
    and the noop ``gather_from_ep_ranks`` patch must be cleaned off the class.
    """
    converter = BridgeConverter.__new__(BridgeConverter)
    converter._mxfp4_quantizer = _Mxfp4OnlineQuantizer()
    mapping = _CollectiveProbeMapping()
    saved = (mapping.pp_group, mapping._tp_group, mapping._etp_group, mapping.ep_group)

    out = converter._convert_mxfp4_expert(
        "decoder.layers.0.mlp.experts.w1",
        "module.module.decoder.layers.0.mlp.experts.w1",
        torch.zeros(4, 4, dtype=torch.bfloat16),
        SimpleNamespace(megatron_module=None),
        mapping,
    )

    assert mapping.groups_seen == (None, None, None, None)
    assert (mapping.pp_group, mapping._tp_group, mapping._etp_group, mapping.ep_group) == saved
    assert "gather_from_ep_ranks" not in _CollectiveProbeMapping.__dict__
    assert [name for name, _ in out] == [
        f"{_CollectiveProbeMapping.HF_WEIGHT}_packed",
        f"{_CollectiveProbeMapping.HF_WEIGHT}_scale",
    ]


def test_convert_mxfp4_expert_rejects_unpaired_output():
    """A quant output whose packed/scale keys don't pair up 1:1 must fail loud,
    not push a half-converted tensor set to the engine."""
    converter = BridgeConverter.__new__(BridgeConverter)
    converter._mxfp4_quantizer = _Mxfp4OnlineQuantizer()
    mapping = _expert_mapping(
        megatron_to_hf_quant=MagicMock(
            return_value={
                "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight": torch.zeros(
                    2, 2, dtype=torch.int8
                )
            }
        )
    )
    with pytest.raises(ValueError, match="unpaired"):
        converter._convert_mxfp4_expert(
            "decoder.layers.1.mlp.experts.w1",
            "module.module.decoder.layers.1.mlp.experts.w1",
            torch.zeros(2, 4, dtype=torch.bfloat16),
            SimpleNamespace(megatron_module=None),
            mapping,
        )


def test_convert_mxfp4_expert_rejects_non_uint8_scale():
    """A drifted kernel contract must raise instead of letting ``.to(uint8)``
    numerically truncate a float exponent grid into zeros."""
    converter = BridgeConverter.__new__(BridgeConverter)
    converter._mxfp4_quantizer = _Mxfp4OnlineQuantizer()
    base = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"
    mapping = _expert_mapping(
        megatron_to_hf_quant=MagicMock(
            return_value={
                base: torch.zeros(2, 2, dtype=torch.int8),
                base + "_scale_inv": torch.zeros(2, 1, dtype=torch.float32),
            }
        )
    )
    with pytest.raises(ValueError, match="uint8 E8M0 scale"):
        converter._convert_mxfp4_expert(
            "decoder.layers.1.mlp.experts.w1",
            "module.module.decoder.layers.1.mlp.experts.w1",
            torch.zeros(2, 4, dtype=torch.bfloat16),
            SimpleNamespace(megatron_module=None),
            mapping,
        )


# ---------------------------------------------------------------------------
# _Mxfp4OnlineQuantizer
# ---------------------------------------------------------------------------


def test_mxfp4_quantizer_rejects_bad_geometry():
    quantizer = _Mxfp4OnlineQuantizer()
    kernel = MagicMock(side_effect=AssertionError("invalid geometry must not reach the kernel"))
    with _fake_quant_kernel(kernel):
        with pytest.raises(ValueError):
            quantizer(torch.zeros(4, 3, dtype=torch.bfloat16), (1, 32))  # cols not divisible
        with pytest.raises(ValueError):
            quantizer(torch.zeros(4, dtype=torch.bfloat16), (1, 32))  # not 2-D
        with pytest.raises(ValueError):
            quantizer(torch.zeros(4, 32, dtype=torch.bfloat16), (1, 64))  # wrong block size
    kernel.assert_not_called()


def test_mxfp4_quantizer_calls_bridge_kernel():
    """With valid geometry the quantizer delegates to the bridge kernel with an
    E8M0 scale template of the expected shape."""
    weight = torch.zeros(8, 64, dtype=torch.bfloat16)
    fake = MagicMock(return_value=(torch.zeros(8, 32, dtype=torch.int8), torch.zeros(8, 2, dtype=torch.uint8)))

    with _fake_quant_kernel(fake):
        packed, scale = _Mxfp4OnlineQuantizer()(weight, (1, 32))

    assert packed.dtype == torch.int8 and scale.dtype == torch.uint8
    args, kwargs = fake.call_args
    assert args[0] is weight
    assert tuple(args[1].shape) == (8, 2)
    assert args[1].dtype == torch.uint8
    assert kwargs == {"block_size": 32}


def test_mxfp4_quantizer_rejects_wrong_output_dtypes():
    """Mirror the offline export's LocalExpertQuantizer contract: packed int8,
    scale uint8.

    A drifted kernel must fail before the rename loop can cast.
    """
    weight = torch.zeros(8, 64, dtype=torch.bfloat16)

    bad_packed = MagicMock(return_value=(torch.zeros(8, 32, dtype=torch.uint8), torch.zeros(8, 2, dtype=torch.uint8)))
    with _fake_quant_kernel(bad_packed), pytest.raises(ValueError, match="packed"):
        _Mxfp4OnlineQuantizer()(weight, (1, 32))

    bad_scale = MagicMock(return_value=(torch.zeros(8, 32, dtype=torch.int8), torch.zeros(8, 2, dtype=torch.float32)))
    with _fake_quant_kernel(bad_scale), pytest.raises(ValueError, match="scale"):
        _Mxfp4OnlineQuantizer()(weight, (1, 32))
