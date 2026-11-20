# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for ``scripts/tools/merge_lora_adapter_to_hf.py``.

The tool folds ``base + (alpha/r) * (B @ A)`` into a base HF checkpoint offline.
Its correctness rests on three pure steps that this file pins down:

- pairing ``lora_A``/``lora_B`` and refusing anything half-understood
- resolving a pair to its base tensor, incl. grouped 3-D MoE expert parameters
- the merge arithmetic itself (2-D linear and batched per-expert)

Real-file CLI tests additionally cover shard I/O, FP8 bases and auxiliary files.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

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
            scaling=2.0,
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
            scaling=0.5,
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
                scaling=1.0,
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
        assert merge_lora._read_adapter_scaling(tmp_path) == pytest.approx(expected)

    def test_missing_rank_or_alpha_is_a_hard_error(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16}))
        with pytest.raises(ValueError, match="missing 'r' / 'lora_alpha'"):
            merge_lora._read_adapter_scaling(tmp_path)


@pytest.mark.parametrize("layout", ["single", "sharded", "fp8"])
def test_peft_cli_merges_real_shards_and_preserves_auxiliary_files(tmp_path, monkeypatch, layout):
    """Exercise the shared writer, absent from the existing arithmetic tests.

    Tiny CPU safetensors cover the old PEFT CLI and cross-shard FP8 scales; no
    model, GPU, subprocess or distributed process is required.
    """
    from safetensors.torch import load_file, save_file

    base, adapter, output = (tmp_path / name for name in ("base", "adapter", "output"))
    base.mkdir()
    adapter.mkdir()
    weight = torch.tensor([[0.5, -0.25], [0.75, 1.0]], dtype=torch.bfloat16)
    untouched = torch.tensor([0.125, 0.375], dtype=torch.float32)
    a, b = torch.tensor([[0.2, -0.3]]), torch.tensor([[0.4], [-0.5]])
    config = {"model_type": "test_vlm", "vision_config": {"hidden_size": 2}}
    tensors = {"w.weight": weight, "vision.weight": untouched}
    if layout == "fp8":
        config["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        tensors["w.weight"] = weight.to(torch.float8_e4m3fn)
        tensors["w.weight_scale_inv"] = torch.tensor([[0.173]])
        weight = (tensors["w.weight"].float() * 0.173).bfloat16()
    if layout == "single":
        groups = {"model.safetensors": tensors}
    else:
        groups = {
            "model-1.safetensors": {"w.weight": tensors["w.weight"]},
            "model-2.safetensors": {k: v for k, v in tensors.items() if k != "w.weight"},
        }
        index = {"weight_map": {key: name for name, values in groups.items() for key in values}}
        (base / "model.safetensors.index.json").write_text(json.dumps(index))
    for name, values in groups.items():
        save_file(values, base / name)
    (base / "config.json").write_text(json.dumps(config))
    (base / "tokenizer_config.json").write_text('{"chat_template": "keep exactly"}')
    (adapter / "adapter_config.json").write_text('{"r": 1, "lora_alpha": 2}')
    save_file({"w.lora_A.weight": a, "w.lora_B.weight": b}, adapter / "adapter_model.safetensors")
    before = {p.name: p.read_bytes() for p in base.iterdir()}
    monkeypatch.setattr(
        "sys.argv",
        [
            "merge",
            "--base-hf-dir",
            str(base),
            "--adapter-dir",
            str(adapter),
            "--output-dir",
            str(output),
            "--cpu-threads",
            "1",
        ],
    )
    merge_lora.main()
    if (output / "model.safetensors.index.json").exists():
        output_map = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    else:
        output_map = dict.fromkeys(tensors, "model.safetensors")
    actual = load_file(output / output_map["w.weight"])["w.weight"]
    torch.testing.assert_close(actual, (weight.float() + 2 * (b @ a)).bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(load_file(output / output_map["vision.weight"])["vision.weight"], untouched)
    assert json.loads((output / "config.json").read_text())["vision_config"] == config["vision_config"]
    assert (output / "tokenizer_config.json").read_bytes() == before["tokenizer_config.json"]
    assert {p.name: p.read_bytes() for p in base.iterdir()} == before
    if layout == "fp8":
        assert "quantization_config" not in json.loads((output / "config.json").read_text())
        assert "w.weight_scale_inv" not in output_map
    else:
        assert (output / "config.json").read_bytes() == before["config.json"]
        if layout == "sharded":
            assert (output / "model-2.safetensors").read_bytes() == before["model-2.safetensors"]
    dry_output = tmp_path / "dry-output"
    assert merge_lora.merge(base, dry_output, adapter_dir=adapter, dry_run=True)["dry_run"]
    assert not dry_output.exists()
