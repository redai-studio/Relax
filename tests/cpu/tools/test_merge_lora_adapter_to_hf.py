# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for ``scripts/tools/merge_lora_adapter_to_hf.py``.

The tool folds ``base + (alpha/r) * (B @ A)`` into a base HF checkpoint offline.
Its correctness rests on three pure steps that this file pins down:

- pairing ``lora_A``/``lora_B`` and refusing anything half-understood
- resolving a pair to its base tensor, incl. grouped 3-D MoE expert parameters
- the merge arithmetic itself (2-D linear and batched per-expert)

Everything else (safetensors streaming, aux-file copying) is plumbing.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from argparse import Namespace

import pytest


torch = pytest.importorskip("torch")

_SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[3] / "scripts/tools/merge_lora_adapter_to_hf.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("merge_lora_adapter_to_hf_under_test", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


merge_lora = _load_module()


class _FakeAdapterFile:
    """Stands in for a ``safe_open`` handle."""

    def __init__(self, tensors):
        self._tensors = tensors

    def get_tensor(self, key):
        return self._tensors[key]


class TestPairAdapterKeys:
    def test_pairs_and_strips_the_peft_prefix(self):
        """Keys are grouped by the base tensor they adapt, sans
        ``base_model.model.``."""
        pairs = merge_lora._pair_adapter_keys(
            [
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                "model.layers.0.mlp.experts.gate_up_proj.lora_A.default.weight",
                "model.layers.0.mlp.experts.gate_up_proj.lora_B.default.weight",
            ]
        )
        assert set(pairs) == {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.experts.gate_up_proj",
        }
        assert pairs["model.layers.0.self_attn.q_proj"]["B"].endswith("lora_B.weight")

    def test_unrecognized_tensor_is_a_hard_error(self):
        """Silently dropping an unknown tensor would produce a wrongly-merged
        model."""
        with pytest.raises(ValueError, match="neither lora_A nor lora_B"):
            merge_lora._pair_adapter_keys(["model.layers.0.self_attn.q_proj.lora_embedding_A"])

    def test_half_pair_is_a_hard_error(self):
        with pytest.raises(ValueError, match="missing their A or B half"):
            merge_lora._pair_adapter_keys(["model.layers.0.self_attn.q_proj.lora_A.weight"])


class TestResolveBaseKeys:
    def test_resolves_linear_weights_and_grouped_expert_params(self):
        """``nn.Linear`` adds ``.weight``; a 3-D grouped expert parameter does
        not."""
        pairs = {"model.layers.0.self_attn.q_proj": {}, "model.layers.0.mlp.experts.gate_up_proj": {}}
        base_keys = {"model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.experts.gate_up_proj"}
        assert merge_lora._resolve_base_keys(pairs, base_keys) == {
            "model.layers.0.self_attn.q_proj": "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.experts.gate_up_proj": "model.layers.0.mlp.experts.gate_up_proj",
        }

    def test_unmatched_pair_is_a_hard_error(self):
        """Usually means ``--base-hf-dir`` is not the base the adapter was
        trained on."""
        with pytest.raises(ValueError, match="no matching tensor in the base model"):
            merge_lora._resolve_base_keys({"model.layers.0.self_attn.q_proj": {}}, {"model.embed_tokens.weight"})


class TestMergeShard:
    def test_merges_a_2d_linear_weight_with_scaling(self):
        lora_a = torch.randn(2, 4)
        lora_b = torch.randn(3, 2)
        base = torch.randn(3, 4, dtype=torch.bfloat16)
        shard = {"w.weight": base.clone()}

        merged = merge_lora._merge_shard(
            shard,
            {"w": "w.weight"},
            _FakeAdapterFile({"a": lora_a, "b": lora_b}),
            {"w": {"A": "a", "B": "b"}},
            config={"r": 2, "lora_alpha": 4},
            device="cpu",
        )

        expected = (base.float() + 2.0 * (lora_b @ lora_a)).to(torch.bfloat16)
        assert merged == ["w.weight"]
        assert shard["w.weight"].dtype == torch.bfloat16
        assert torch.equal(shard["w.weight"], expected)

    def test_merges_grouped_experts_as_a_batched_matmul(self):
        """A 3-D expert parameter folds per-expert, not as one flat matmul."""
        num_experts = 3
        lora_a = torch.randn(num_experts, 2, 4)
        lora_b = torch.randn(num_experts, 5, 2)
        base = torch.zeros(num_experts, 5, 4)
        shard = {"experts.gate_up_proj": base.clone()}

        merge_lora._merge_shard(
            shard,
            {"experts.gate_up_proj": "experts.gate_up_proj"},
            _FakeAdapterFile({"a": lora_a, "b": lora_b}),
            {"experts.gate_up_proj": {"A": "a", "B": "b"}},
            config={"r": 2, "lora_alpha": 1},
            device="cpu",
        )

        for e in range(num_experts):
            assert torch.allclose(shard["experts.gate_up_proj"][e], 0.5 * (lora_b[e] @ lora_a[e]), atol=1e-6)

    def test_shape_mismatch_is_a_hard_error(self):
        """Catches an adapter exported against a differently-shaped base."""
        shard = {"w.weight": torch.zeros(7, 4)}
        with pytest.raises(ValueError, match="does not match base tensor"):
            merge_lora._merge_shard(
                shard,
                {"w": "w.weight"},
                _FakeAdapterFile({"a": torch.zeros(2, 4), "b": torch.zeros(3, 2)}),
                {"w": {"A": "a", "B": "b"}},
                config={"r": 2, "lora_alpha": 2},
                device="cpu",
            )


class TestReadAdapterScaling:
    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            ({"r": 16, "lora_alpha": 32}, 2.0),
            ({"r": 16, "lora_alpha": 32, "use_rslora": True}, 8.0),  # alpha / sqrt(r)
        ],
    )
    def test_scaling_rule(self, tmp_path, config, expected):
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))
        assert merge_lora._adapter_scaling(merge_lora._read_adapter_config(tmp_path)) == pytest.approx(expected)

    def test_missing_rank_or_alpha_is_a_hard_error(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16}))
        with pytest.raises(ValueError, match="missing 'r' / 'lora_alpha'"):
            merge_lora._read_adapter_config(tmp_path)


@pytest.mark.parametrize("packed", [False, True])
def test_offline_merge_expert_adapter_export(tmp_path, monkeypatch, packed):
    from safetensors.torch import load_file, save_file

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    adapter_dir, output_dir = tmp_path / "adapter", tmp_path / "merged"
    base, weights, expected = {}, {}, {}
    for name, inputs, outputs, rank in [
        ("mlp.experts.gate_up_proj", 4, 6, 2),
        ("mlp.experts.down_proj", 3, 4, 3 if packed else 2),
    ]:
        base[name] = torch.randn(3, outputs, inputs)
        a = weights[f"{name}.lora_A.weight"] = torch.randn(3, rank, inputs)
        b = weights[f"{name}.lora_B.weight"] = torch.randn(3, outputs, rank)
        expected[name] = base[name] + (4 / rank) * (b @ a)
    save_file(base, base_dir / "model.safetensors")
    args = Namespace(lora_rank=2, lora_alpha=4, lora_dropout=0.0, lora_target_modules=["linear_fc1", "linear_fc2"])
    if packed:
        bridge = pytest.importorskip("megatron.bridge.models.conversion.peft_bridge")
        if not hasattr(bridge, "convert_adapter_weights_to_peft_state"):
            pytest.skip("Bridge version predates packed parameter export")
        from relax.utils.megatron_peft_utils import write_megatron_peft_adapter

        write_megatron_peft_adapter(weights, adapter_dir, args)
    else:
        adapter_dir.mkdir()
        save_file(weights, adapter_dir / "adapter_model.safetensors")
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["merge", "--base-hf-dir", str(base_dir), "--adapter-dir", str(adapter_dir), "--output-dir", str(output_dir)],
    )
    merge_lora.main()
    result = load_file(output_dir / "model.safetensors")
    for name in base:
        torch.testing.assert_close(result[name], expected[name])
