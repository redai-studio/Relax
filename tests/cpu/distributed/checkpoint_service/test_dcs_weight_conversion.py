# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for DCS weight conversion through BridgeConverter.

All tests call **real** project functions or construct **real** Bridge mapping
objects — no hand-written logic duplication.

Covers:
- Real Bridge mapping ``megatron_to_hf`` output + post-processing correctness
- ``strip_param_name_prefix``, ``remove_padding``, ``quantize_params``
"""

from argparse import Namespace
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

# Bridge and MCore are required by this suite. Missing dependencies or API drift
# must fail collection rather than silently removing real conversion coverage.
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import Qwen3VLMoEBridge
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge

from relax.backends.megatron.misc_utils import strip_param_name_prefix
from relax.backends.megatron.weight_conversion.processors import quantize_params, remove_padding
from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_args(**overrides) -> Namespace:
    """Create a minimal args namespace for weight conversion tests."""
    defaults = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=2,
        kv_channels=128,
        ffn_hidden_size=6144,
        moe_ffn_hidden_size=768,
        vocab_size=151936,
        q_lora_rank=None,
        update_weight_buffer_size=1 << 30,
        hf_checkpoint="/fake/checkpoint",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


@contextmanager
def _patch_gather_from_ep_ranks():
    """Monkey-patch ``gather_from_ep_ranks`` on all relevant mapping classes.

    This is the same no-op that ``_convert_to_hf_bridge`` applies in production
    to bypass EP gather when expert weights have already been gathered
    externally.
    """

    def _noop_gather(self_m, megatron_weights, megatron_module, hf_param_name):
        return {str(hf_param_name): megatron_weights}

    saved_originals: dict = {}
    patched_classes = [MegatronParamMapping, GatedMLPMapping, FusedGatedExpertMapping, FusedExpertMapping]
    for cls in patched_classes:
        if "gather_from_ep_ranks" in cls.__dict__:
            saved_originals[cls] = cls.__dict__["gather_from_ep_ranks"]
        cls.gather_from_ep_ranks = _noop_gather
    try:
        yield
    finally:
        for cls in patched_classes:
            if cls in saved_originals:
                cls.gather_from_ep_ranks = saved_originals[cls]
            elif "gather_from_ep_ranks" in cls.__dict__:
                del cls.gather_from_ep_ranks


@lru_cache
def _expert_registry(qwen35: bool):
    """Build the real model registry with a tiny packed-expert checkpoint
    schema."""
    keys = [
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    ]
    source = SimpleNamespace(get_all_keys=lambda: keys)
    bridge = SimpleNamespace(hf_pretrained=SimpleNamespace(state=SimpleNamespace(source=source)))
    bridge_cls = Qwen35VLMoEBridge if qwen35 else Qwen3VLMoEBridge
    return bridge_cls.mapping_registry(bridge)


def _expert_mapping(layer_idx: int, expert_id: int, fc: int, qwen35: bool = False):
    name = f"language_model.decoder.layers.{layer_idx}.mlp.experts.linear_fc{fc}.weight{expert_id}"
    mapping = _expert_registry(qwen35).megatron_to_hf_lookup(name)
    expected_type = FusedGatedExpertMapping if fc == 1 else FusedExpertMapping
    assert isinstance(mapping, expected_type)
    assert mapping.transpose_on_export is (not qwen35)
    # The same eager AutoMapping initialization as BridgeConverter.init_tasks.
    mapping._detected_type = "replicated"
    mapping._mapping = mapping._get_or_create_mapping("replicated")
    return mapping


def _make_expert_gate_up_mapping(layer_idx: int, expert_id: int) -> FusedGatedExpertMapping:
    return _expert_mapping(layer_idx, expert_id, fc=1)


def _make_expert_down_mapping(layer_idx: int, expert_id: int) -> FusedExpertMapping:
    return _expert_mapping(layer_idx, expert_id, fc=2)


def _make_qwen35_expert_gate_up_mapping(layer_idx: int, expert_id: int) -> FusedGatedExpertMapping:
    return _expert_mapping(layer_idx, expert_id, fc=1, qwen35=True)


def _make_qwen35_expert_down_mapping(layer_idx: int, expert_id: int) -> FusedExpertMapping:
    return _expert_mapping(layer_idx, expert_id, fc=2, qwen35=True)


def _expert_task(mapping, param: torch.Tensor) -> WeightConversionTask:
    return WeightConversionTask(
        param_name=mapping.megatron_param,
        global_param_name=mapping.megatron_param,
        mapping=mapping,
        megatron_module=None,
        param_weight=param,
    )


def _convert_real_mapping(mapping, param: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Exercise real initialization, mapping and Relax post-processing.

    Only checkpoint/model discovery is supplied locally: no model download or
    distributed model construction. Mapping outputs and transpose detection are
    not mocked or manually overridden.
    """
    task = _expert_task(mapping, param)
    bridge = SimpleNamespace(
        get_conversion_tasks=lambda model: [task],
        _model_bridge=SimpleNamespace(mapping_registry=lambda: _expert_registry(False)),
    )
    converter = BridgeConverter(_make_args(), model=[], quantization_config=None)
    with (
        patch("megatron.bridge.AutoBridge.from_hf_pretrained", return_value=bridge),
        patch("relax.utils.megatron_bridge_utils.patch_megatron_model", return_value=nullcontext()),
        patch("relax.backends.megatron.weight_update.bridge_converter.named_params_and_buffers", return_value=[]),
    ):
        converter.init_tasks()
    # transpose_on_export is a grouped-export flag, not a per-expert transpose.
    assert converter._bridge_expert_transposes_down is False
    return converter.convert(mapping.megatron_param, param)


def _grouped_export(mapping, param: torch.Tensor, buffers: dict, num_experts: int, hf_state: dict):
    """Call Bridge's real collector; inputs are already EP-gathered (EP=1)."""
    with _patch_gather_from_ep_ranks():
        output = mapping.megatron_to_hf(param, None)
    return MegatronModelBridge._accumulate_grouped_export(
        None, _expert_task(mapping, param), output, SimpleNamespace(num_moe_experts=num_experts), buffers, hf_state
    )


def _convert_mapping_output(
    converted_dict: dict[str, torch.Tensor],
    megatron_param_name: str,
    bridge_expert_transposes_down: bool = False,
) -> list[tuple[str, torch.Tensor]]:
    """Run actual conversion with a supplied Bridge mapping result.

    Only the mapping call is replaced; expert splitting, transposes, name
    resolution and unquantized output all execute production code.
    """
    converter = BridgeConverter(_make_args(), model=[], quantization_config=None)
    mapping = SimpleNamespace(megatron_to_hf=Mock(return_value=converted_dict))
    converter._bridge_task_map = {
        megatron_param_name: SimpleNamespace(mapping=mapping, megatron_module=None),
    }
    converter._bridge_expert_transposes_down = bridge_expert_transposes_down
    result = converter.convert(megatron_param_name, torch.empty(0))
    mapping.megatron_to_hf.assert_called_once()
    return result


# ─── Tests for real Bridge mapping megatron_to_hf output ──────────────────────


class TestBridgeMappingOutput:
    """Test real Bridge mapping ``megatron_to_hf`` output format.

    These tests call the actual Bridge mapping objects and verify their output
    shape/format, confirming the assumptions that the post-processing relies
    on.
    """

    def test_replicated_mapping_passthrough(self):
        """ReplicatedMapping passes tensor through unchanged."""
        m = ReplicatedMapping(
            "decoder.layers.0.self_attention.linear_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
        )
        w = torch.randn(2048, 2048)
        result = m.megatron_to_hf(w, None)
        assert list(result.keys()) == ["model.layers.0.self_attn.o_proj.weight"]
        assert torch.equal(result["model.layers.0.self_attn.o_proj.weight"], w)

    def test_gated_mlp_mapping_splits_gate_up(self):
        """GatedMLPMapping splits fused [gate; up] into separate tensors."""
        m = GatedMLPMapping(
            "decoder.layers.0.mlp.linear_fc1.weight",
            gate="model.layers.0.mlp.gate_proj.weight",
            up="model.layers.0.mlp.up_proj.weight",
        )
        H, D = 768, 2048
        fused = torch.randn(H * 2, D)
        result = m.megatron_to_hf(fused, None)

        assert set(result.keys()) == {
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        }
        gate_expected, up_expected = fused.chunk(2, dim=0)
        assert torch.equal(result["model.layers.0.mlp.gate_proj.weight"], gate_expected)
        assert torch.equal(result["model.layers.0.mlp.up_proj.weight"], up_expected)

    @pytest.mark.parametrize("qwen35", [False, True], ids=["qwen3-vl", "qwen3.5"])
    def test_expert_gate_up_mapping_outputs_untransposed_until_grouped_export(self, qwen35):
        """Per-expert output is [2H,D]; only grouped export applies the
        flag."""
        H, D = 3, 5
        fused = torch.arange(2 * H * D, dtype=torch.float32).reshape(2 * H, D)
        m = _expert_mapping(0, 0, fc=1, qwen35=qwen35)
        key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        with _patch_gather_from_ep_ranks():
            result = m.megatron_to_hf(fused, None)
        assert list(result) == [key]
        assert result[key].shape == (2 * H, D)
        assert torch.equal(result[key], fused)  # gate precedes up, without transpose

        grouped = _grouped_export(m, fused, {}, 1, {})
        expected = fused.unsqueeze(0) if qwen35 else fused.T.unsqueeze(0)
        assert list(grouped) == [key]
        assert grouped[key].shape == expected.shape
        assert torch.equal(grouped[key], expected)
        assert grouped[key].is_contiguous()

    @pytest.mark.parametrize("qwen35", [False, True], ids=["qwen3-vl", "qwen3.5"])
    @pytest.mark.parametrize("hf_layout", ["unknown", "native", "transposed"])
    def test_expert_down_mapping_outputs_untransposed_until_grouped_export(self, qwen35, hf_layout):
        """Collector transposes adaptively to the original checkpoint shape."""
        D, H = 5, 3
        param = torch.arange(D * H, dtype=torch.float32).reshape(D, H)
        m = _expert_mapping(0, 0, fc=2, qwen35=qwen35)
        key = "model.language_model.layers.0.mlp.experts.down_proj"
        with _patch_gather_from_ep_ranks():
            result = m.megatron_to_hf(param, None)
        assert list(result) == [key]
        assert result[key].shape == (D, H)
        assert torch.equal(result[key], param)

        hf_state = (
            {} if hf_layout == "unknown" else {key: torch.empty((1, D, H) if hf_layout == "native" else (1, H, D))}
        )
        grouped = _grouped_export(m, param, {}, 1, hf_state)
        transpose = not qwen35 and hf_layout != "native"
        expected = (param.T if transpose else param).unsqueeze(0)
        assert list(grouped) == [key]
        assert grouped[key].shape == expected.shape
        assert torch.equal(grouped[key], expected)
        assert grouped[key].is_contiguous()


# ─── Tests for Bridge + post-processing output correctness ───────────────────


class TestBridgePostProcessingCorrectness:
    """Verify that real Bridge output + post-processing produces correct HF
    weights.

    Expected behavior (ground truth):
    - gate_up (linear_fc1): Megatron [2H, D] → split into gate [H, D] and up [H, D]
      (same as raw converter: simple chunk, no transpose)
    - down (linear_fc2): Megatron [D, H] → HF [D, H] passthrough
      (same as raw converter: identity)
    """

    def test_expert_gate_up_postprocessed_matches_expected(self):
        """Bridge gate_up + post-processing produces correct gate/up split."""
        H, D = 768, 2048
        expert_id = 3
        megatron_param = torch.randn(H * 2, D)

        # Expected: simple chunk of the Megatron fused weight
        expected_gate, expected_up = megatron_param.chunk(2, dim=0)

        # Bridge + post-processing
        with _patch_gather_from_ep_ranks():
            mapping = _make_expert_gate_up_mapping(layer_idx=0, expert_id=expert_id)
            bridge_output = mapping.megatron_to_hf(megatron_param, None)

        postprocessed = _convert_mapping_output(
            bridge_output, f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert_id}"
        )

        assert len(postprocessed) == 2
        assert postprocessed[0][0] == "model.language_model.layers.0.mlp.experts.3.gate_proj.weight"
        assert postprocessed[1][0] == "model.language_model.layers.0.mlp.experts.3.up_proj.weight"
        assert postprocessed[0][1].shape == (H, D)
        assert postprocessed[1][1].shape == (H, D)
        assert torch.allclose(postprocessed[0][1], expected_gate)
        assert torch.allclose(postprocessed[1][1], expected_up)

    @pytest.mark.parametrize("qwen35", [False, True], ids=["qwen3-vl", "qwen3.5"])
    def test_expert_down_proj_postprocessed_matches_expected(self, qwen35):
        """Real per-expert mapping + Relax is identity, regardless of export
        flag."""
        D, H = 5, 3
        expert_id = 5
        param = torch.arange(D * H, dtype=torch.float32).reshape(D, H)
        mapping = _expert_mapping(0, expert_id, fc=2, qwen35=qwen35)
        postprocessed = _convert_real_mapping(mapping, param)
        assert len(postprocessed) == 1
        assert postprocessed[0][0] == "model.language_model.layers.0.mlp.experts.5.down_proj.weight"
        assert postprocessed[0][1].shape == (D, H)
        assert torch.equal(postprocessed[0][1], param)
        assert postprocessed[0][1].is_contiguous()

    @pytest.mark.parametrize("qwen35", [False, True], ids=["qwen3-vl", "qwen3.5"])
    def test_correctness_across_layers_and_experts(self, qwen35):
        """Preserve all original layers/IDs and check grouped expert
        ordering."""
        H, D = 3, 5
        for layer_idx in [0, 5, 27]:
            buffers = {1: {}, 2: {}}
            originals = {1: {}, 2: {}}
            # Out-of-order arrival must not change final expert axis order.
            for expert_id in [42, 7, 0, *[i for i in range(43) if i not in (0, 7, 42)]]:
                fc1 = torch.arange(2 * H * D, dtype=torch.float32).reshape(2 * H, D)
                fc1 = fc1 + layer_idx * 10000 + expert_id * 100
                fc2 = torch.arange(D * H, dtype=torch.float32).reshape(D, H) + layer_idx * 10000 + expert_id * 100
                expected_gate, expected_up = fc1[:H], fc1[H:]
                base = f"model.language_model.layers.{layer_idx}.mlp.experts"
                mapping_fc1 = _expert_mapping(layer_idx, expert_id, fc=1, qwen35=qwen35)
                post_fc1 = _convert_real_mapping(mapping_fc1, fc1)
                assert [name for name, _ in post_fc1] == [
                    f"{base}.{expert_id}.gate_proj.weight",
                    f"{base}.{expert_id}.up_proj.weight",
                ]
                for (_, tensor), expected in zip(post_fc1, (expected_gate, expected_up)):
                    assert tensor.shape == (H, D)
                    assert torch.equal(tensor, expected)
                    assert tensor.is_contiguous()
                mapping_fc2 = _expert_mapping(layer_idx, expert_id, fc=2, qwen35=qwen35)
                post_fc2 = _convert_real_mapping(mapping_fc2, fc2)
                assert len(post_fc2) == 1
                assert post_fc2[0][0] == f"{base}.{expert_id}.down_proj.weight"
                assert post_fc2[0][1].shape == (D, H)
                assert torch.equal(post_fc2[0][1], fc2)

                for fc, mapping, param in [(1, mapping_fc1, fc1), (2, mapping_fc2, fc2)]:
                    originals[fc][expert_id] = param
                    grouped = _grouped_export(mapping, param, buffers[fc], 43, {})
                    if len(originals[fc]) < 43:
                        assert grouped is None
                    else:
                        key = f"{base}.{'gate_up_proj' if fc == 1 else 'down_proj'}"
                        expected = torch.stack([originals[fc][i] for i in range(43)])
                        if not qwen35:
                            expected = expected.transpose(-1, -2).contiguous()
                        assert list(grouped) == [key]
                        assert grouped[key].shape == expected.shape
                        assert torch.equal(grouped[key], expected)
                        assert grouped[key].is_contiguous()
                        assert buffers[fc] == {}

    def test_non_expert_replicated_no_postprocessing(self):
        """Non-expert params (e.g. layernorm) pass through without post-
        processing."""
        m = ReplicatedMapping(
            "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.0.input_layernorm.weight",
        )
        w = torch.randn(2048)
        bridge_output = m.megatron_to_hf(w, None)

        # No expert_id in name → post-processing is a no-op
        postprocessed = _convert_mapping_output(
            bridge_output, "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight"
        )
        assert len(postprocessed) == 1
        assert postprocessed[0][0] == "model.layers.0.input_layernorm.weight"
        assert torch.equal(postprocessed[0][1], w)

    def test_non_expert_gated_mlp_no_postprocessing(self):
        """Non-expert GatedMLPMapping (dense MLP) passes through without post-
        processing."""
        m = GatedMLPMapping(
            "decoder.layers.0.mlp.linear_fc1.weight",
            gate="model.layers.0.mlp.gate_proj.weight",
            up="model.layers.0.mlp.up_proj.weight",
        )
        H, D = 6144, 2048
        fused = torch.randn(H * 2, D)
        bridge_output = m.megatron_to_hf(fused, None)

        postprocessed = _convert_mapping_output(bridge_output, "decoder.layers.0.mlp.linear_fc1.weight")
        assert len(postprocessed) == 2
        gate_expected, up_expected = fused.chunk(2, dim=0)
        assert postprocessed[0][0] == "model.layers.0.mlp.gate_proj.weight"
        assert postprocessed[1][0] == "model.layers.0.mlp.up_proj.weight"
        assert torch.equal(postprocessed[0][1], gate_expected)
        assert torch.equal(postprocessed[1][1], up_expected)


# ─── Tests for strip_param_name_prefix (real function) ────────────────────────


class TestStripParamNamePrefix:
    """Test the real ``strip_param_name_prefix`` utility."""

    def test_strip_double_module(self):
        assert strip_param_name_prefix("module.module.decoder.layers.0.weight") == "decoder.layers.0.weight"

    def test_strip_single_module(self):
        assert strip_param_name_prefix("module.decoder.layers.0.weight") == "decoder.layers.0.weight"

    def test_no_prefix(self):
        assert strip_param_name_prefix("decoder.layers.0.weight") == "decoder.layers.0.weight"

    def test_triple_module(self):
        assert strip_param_name_prefix("module.module.module.decoder.layers.0.weight") == "decoder.layers.0.weight"


# ─── Tests for remove_padding (real function) ─────────────────────────────────


class TestRemovePadding:
    """Test the real ``remove_padding`` function."""

    def test_embedding_padding_removed(self):
        vocab_size = 100
        padded = torch.randn(128, 64)
        result = remove_padding("module.module.embedding.word_embeddings.weight", padded, vocab_size)
        assert result.shape == (100, 64)
        assert torch.equal(result, padded[:100])

    def test_output_layer_padding_removed(self):
        vocab_size = 100
        padded = torch.randn(128, 64)
        result = remove_padding("module.module.output_layer.weight", padded, vocab_size)
        assert result.shape == (100, 64)
        assert torch.equal(result, padded[:100])

    def test_non_embedding_unchanged(self):
        vocab_size = 100
        param = torch.randn(128, 64)
        result = remove_padding("module.module.decoder.layers.0.weight", param, vocab_size)
        assert result.shape == (128, 64)
        assert torch.equal(result, param)


# ─── Tests for quantize_params (real function) ───────────────────────────────


class TestQuantizeParamsPassthrough:
    """Test the real ``quantize_params`` function."""

    def test_no_quantization_returns_same_object(self):
        """With quantization_config=None, returns the same list object."""
        args = _make_args()
        tensors = [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.randn(768, 2048)),
            ("model.layers.0.mlp.experts.0.up_proj.weight", torch.randn(768, 2048)),
        ]
        result = quantize_params(args, "module.module.decoder.layers.0.weight", tensors, None)
        assert result is tensors


# ─── Tests for expert weight edge cases ───────────────────────────────────────


class TestExpertWeightEdgeCases:
    """Test edge cases using real Bridge mappings and post-processing."""

    def test_expert_id_zero(self):
        """Expert ID 0 works correctly through the full pipeline."""
        H, D = 768, 2048
        param = torch.randn(H * 2, D)

        with _patch_gather_from_ep_ranks():
            mapping = _make_expert_gate_up_mapping(layer_idx=0, expert_id=0)
            bridge_output = mapping.megatron_to_hf(param, None)

        postprocessed = _convert_mapping_output(bridge_output, "decoder.layers.0.mlp.experts.linear_fc1.weight0")
        assert postprocessed[0][0].endswith(".experts.0.gate_proj.weight")
        assert postprocessed[1][0].endswith(".experts.0.up_proj.weight")

        expected_gate, expected_up = param.chunk(2, dim=0)
        assert torch.allclose(postprocessed[0][1], expected_gate)
        assert torch.allclose(postprocessed[1][1], expected_up)

    def test_expert_id_large(self):
        """Large expert IDs (e.g. 127) work correctly."""
        H, D = 768, 2048
        param = torch.randn(H * 2, D)

        with _patch_gather_from_ep_ranks():
            mapping = _make_expert_gate_up_mapping(layer_idx=0, expert_id=127)
            bridge_output = mapping.megatron_to_hf(param, None)

        postprocessed = _convert_mapping_output(bridge_output, "decoder.layers.0.mlp.experts.linear_fc1.weight127")
        assert postprocessed[0][0].endswith(".experts.127.gate_proj.weight")

    def test_contiguous_after_postprocessing(self):
        """Post-processed tensors are contiguous (required for NCCL
        broadcast)."""
        H, D = 768, 2048

        with _patch_gather_from_ep_ranks():
            mapping = _make_expert_gate_up_mapping(layer_idx=0, expert_id=0)
            bridge_output = mapping.megatron_to_hf(torch.randn(H * 2, D), None)

        postprocessed = _convert_mapping_output(bridge_output, "decoder.layers.0.mlp.experts.linear_fc1.weight0")
        for _, tensor in postprocessed:
            assert tensor.is_contiguous()

    def test_dtype_preserved_through_pipeline(self):
        """Post-processing preserves tensor dtype through real Bridge
        mapping."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            H, D = 768, 2048
            param = torch.randn(H * 2, D, dtype=dtype)

            with _patch_gather_from_ep_ranks():
                mapping = _make_expert_gate_up_mapping(layer_idx=0, expert_id=0)
                bridge_output = mapping.megatron_to_hf(param, None)

            postprocessed = _convert_mapping_output(bridge_output, "decoder.layers.0.mlp.experts.linear_fc1.weight0")
            for _, tensor in postprocessed:
                assert tensor.dtype == dtype

    def test_element_count_preserved(self):
        """Total number of elements is preserved through Bridge + post-
        processing."""
        H, D = 768, 2048
        param = torch.randn(H * 2, D)
        original_numel = param.numel()

        with _patch_gather_from_ep_ranks():
            mapping = _make_expert_gate_up_mapping(layer_idx=0, expert_id=0)
            bridge_output = mapping.megatron_to_hf(param, None)

        postprocessed = _convert_mapping_output(bridge_output, "decoder.layers.0.mlp.experts.linear_fc1.weight0")
        total_numel = sum(t.numel() for _, t in postprocessed)
        assert total_numel == original_numel


# ─── Tests for Qwen3.5 Bridge (2D cat, no transpose) ─────────────────────────


class TestQwen35BridgeMappingOutput:
    """Test Qwen3.5 Bridge mapping output format (2D cat, no transpose)."""

    def test_qwen35_gate_up_outputs_2d_cat(self):
        """Qwen3.5 FusedGatedExpertMapping outputs 2D [2*H, D] via cat."""
        with _patch_gather_from_ep_ranks():
            m = _make_qwen35_expert_gate_up_mapping(layer_idx=0, expert_id=3)
            H, D = 768, 2048
            fused = torch.randn(H * 2, D)
            result = m.megatron_to_hf(fused, None)

        key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        assert list(result.keys()) == [key]
        tensor = result[key]
        assert tensor.ndim == 2
        assert tensor.shape == (H * 2, D)

    def test_qwen35_down_proj_no_transpose(self):
        """Qwen3.5 FusedExpertMapping does not transpose."""
        with _patch_gather_from_ep_ranks():
            m = _make_qwen35_expert_down_mapping(layer_idx=0, expert_id=3)
            D, H = 2048, 768
            param = torch.randn(D, H)
            result = m.megatron_to_hf(param, None)

        key = "model.language_model.layers.0.mlp.experts.down_proj"
        assert list(result.keys()) == [key]
        tensor = result[key]
        assert tensor.shape == (D, H)
        assert torch.allclose(tensor, param)

    def test_qwen35_expert_transposes_down_detection(self):
        """Real registries share mapping classes, but carry different export
        flags."""
        for fc in [1, 2]:
            vl = _expert_mapping(0, 3, fc=fc)
            qwen35 = _expert_mapping(0, 3, fc=fc, qwen35=True)
            assert type(vl) is type(qwen35)
            assert vl.transpose_on_export is True
            assert qwen35.transpose_on_export is False
            param = torch.arange(30 if fc == 1 else 15, dtype=torch.float32).reshape(-1, 5)
            # Neither instance transposes in per-parameter conversion.
            vl_output = _convert_real_mapping(vl, param)
            qwen35_output = _convert_real_mapping(qwen35, param)
            assert [name for name, _ in vl_output] == [name for name, _ in qwen35_output]
            for (_, actual), (_, expected) in zip(vl_output, qwen35_output):
                assert actual.shape == expected.shape
                assert torch.equal(actual, expected)


class TestQwen35PostProcessingCorrectness:
    """Verify Qwen3.5 Bridge output + post-processing produces correct HF
    weights."""

    def test_qwen35_gate_up_postprocessed(self):
        """Qwen3.5 gate_up 2D + post-processing produces correct gate/up."""
        H, D = 768, 2048
        expert_id = 3
        megatron_param = torch.randn(H * 2, D)
        expected_gate, expected_up = megatron_param.chunk(2, dim=0)

        with _patch_gather_from_ep_ranks():
            mapping = _make_qwen35_expert_gate_up_mapping(layer_idx=0, expert_id=expert_id)
            bridge_output = mapping.megatron_to_hf(megatron_param, None)

        megatron_name = f"language_model.decoder.layers.0.mlp.experts.linear_fc1.weight{expert_id}"
        postprocessed = _convert_mapping_output(bridge_output, megatron_name, bridge_expert_transposes_down=False)

        assert len(postprocessed) == 2
        assert postprocessed[0][0].endswith(f".experts.{expert_id}.gate_proj.weight")
        assert postprocessed[1][0].endswith(f".experts.{expert_id}.up_proj.weight")
        assert postprocessed[0][1].shape == (H, D)
        assert postprocessed[1][1].shape == (H, D)
        assert torch.allclose(postprocessed[0][1], expected_gate)
        assert torch.allclose(postprocessed[1][1], expected_up)

    def test_qwen35_down_proj_postprocessed(self):
        """Qwen3.5 down_proj passthrough (no transpose undo)."""
        D, H = 2048, 768
        expert_id = 5
        megatron_param = torch.randn(D, H)

        with _patch_gather_from_ep_ranks():
            mapping = _make_qwen35_expert_down_mapping(layer_idx=0, expert_id=expert_id)
            bridge_output = mapping.megatron_to_hf(megatron_param, None)

        megatron_name = f"language_model.decoder.layers.0.mlp.experts.linear_fc2.weight{expert_id}"
        postprocessed = _convert_mapping_output(bridge_output, megatron_name, bridge_expert_transposes_down=False)

        assert len(postprocessed) == 1
        assert postprocessed[0][0].endswith(f".experts.{expert_id}.down_proj.weight")
        assert postprocessed[0][1].shape == (D, H)
        assert torch.allclose(postprocessed[0][1], megatron_param)
