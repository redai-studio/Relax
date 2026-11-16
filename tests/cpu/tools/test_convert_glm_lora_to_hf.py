# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real CPU DCP/HF round-trip coverage for lightweight GLM export.

Existing PEFT-merge tests do not cover DCP's expert/gate layout or an FP8 base.
These small single-process checkpoints require no model or GPU runtime.
"""

import io
import json
from argparse import Namespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.tools.convert_glm_lora_to_hf import adapter_targets, base_identity, convert


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(7)
    base = tmp_path / "base"
    base.mkdir()
    dcp_dir = tmp_path / "checkpoint"
    weights, adapters, expected = {}, {}, {}

    def pair(module, targets, *, experts=None, fp8=False):
        a = torch.randn((2, 3) if experts is None else (experts, 2, 3)).bfloat16()
        b = torch.randn((4 * len(targets), 2) if experts is None else (experts, 4 * len(targets), 2)).bfloat16()
        prefix = "decoder.layers." + module
        adapters[prefix + ".adapter.linear_in.weight"] = a
        adapters[prefix + ".adapter.linear_out.weight"] = b
        for expert in range(experts or 1):
            for half, target in enumerate(targets):
                key = target.format(expert=expert)
                weight = torch.randn(4, 3).bfloat16()
                if fp8:
                    weight = weight.to(torch.float8_e4m3fn)
                    scale = torch.tensor([[0.173]], dtype=torch.float32)
                    weights[key + "_scale_inv"] = scale
                    initialized = (weight.float() * scale).bfloat16()
                else:
                    initialized = weight
                weights[key] = weight
                aa, bb = (a, b) if experts is None else (a[expert], b[expert])
                bb = bb[half * 4 : (half + 1) * 4]
                expected[key] = (initialized.float() + 2 * bb.float().matmul(aa.float())).bfloat16()

    pair("0.self_attention.linear_q_down_proj", ["model.layers.0.self_attn.q_a_proj.weight"], fp8=True)
    pair("0.mlp.linear_fc1", [f"model.layers.0.mlp.{name}_proj.weight" for name in ("gate", "up")])
    pair(
        "1.mlp.shared_experts.linear_fc1",
        [f"model.layers.1.mlp.shared_experts.{name}_proj.weight" for name in ("gate", "up")],
    )
    pair(
        "1.mlp.experts.experts.linear_fc1",
        [f"model.layers.1.mlp.experts.{{expert}}.{name}_proj.weight" for name in ("gate", "up")],
        experts=2,
    )
    pair("1.mlp.experts.experts.linear_fc2", ["model.layers.1.mlp.experts.{expert}.down_proj.weight"], experts=2)
    # Unadapted matrices must also be dequantized, including partial edge blocks.
    key = "model.layers.1.self_attn.indexer.wq_b.weight"
    weight = torch.randn(130, 129).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.137, 0.43], [0.291, 1.07]])
    weights[key], weights[key + "_scale_inv"] = weight, scale
    expected[key] = (weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:130, :129]).bfloat16()
    weights["model.layers.1.mlp.gate.e_score_correction_bias"] = torch.tensor([0.25, -0.125])
    expected["model.layers.1.mlp.gate.e_score_correction_bias"] = weights[
        "model.layers.1.mlp.gate.e_score_correction_bias"
    ]
    # Put scales in a separate shard to exercise cross-shard companions.
    groups = [
        {k: v for k, v in weights.items() if not k.endswith("_scale_inv")},
        {k: v for k, v in weights.items() if k.endswith("_scale_inv")},
    ]
    weight_map = {}
    for i, values in enumerate(groups):
        name = f"model-{i}.safetensors"
        save_file(values, base / name)
        weight_map.update(dict.fromkeys(values, name))
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 2,
        "dtype": "bfloat16",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    }
    (base / "config.json").write_text(json.dumps(config))
    (base / "tokenizer_config.json").write_text('{"chat_template": "keep exactly"}')
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    spec = {
        "format_version": 1,
        "lora_rank": 2,
        "lora_alpha": 4,
        "lora_dropout": 0,
        "base_hf_checkpoint": base_identity(base, weight_map),
    }
    common = io.BytesIO()
    torch.save([{"args": Namespace(relax_lora_checkpoint_metadata=spec)}], common)
    common.seek(0)
    # Match MCore's embedded common object and TE adapter extra-state objects.
    state = dict(adapters)
    state["common_state/shard_0_1"] = common
    for key in adapters:
        state[key.removesuffix("weight") + "_extra_state/shard_0_1"] = io.BytesIO()
    dcp.save(state, checkpoint_id=dcp_dir)
    return dcp_dir, base, tmp_path / "output", adapters, expected


def test_real_dcp_export_merges_every_adapter_and_removes_quantization(checkpoint):
    dcp_dir, base, output, adapters, expected = checkpoint
    report = convert(dcp_dir, base, output)
    assert report["adapter_tensors"] == len(adapters)
    assert report["all_finite"]
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(expected)
    for key, value in expected.items():
        with safe_open(output / index["weight_map"][key], framework="pt") as handle:
            actual = handle.get_tensor(key)
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    assert "quantization_config" not in json.loads((output / "config.json").read_text())
    assert (output / "tokenizer_config.json").read_bytes() == (base / "tokenizer_config.json").read_bytes()
    assert not output.with_name(output.name + ".partial").exists()


def test_base_identity_mismatch_refuses_to_publish(checkpoint):
    dcp_dir, base, output, _, _ = checkpoint
    (base / "config.json").write_text((base / "config.json").read_text() + "\n")
    with pytest.raises(ValueError, match="identity"):
        convert(dcp_dir, base, output)
    assert not output.exists()


@pytest.mark.parametrize("suffix", ["self_attention.unsupported", "mlp.linear_fc1"])
def test_unknown_or_unmatched_adapter_is_rejected(suffix):
    prefix = "decoder.layers.0." + suffix
    shapes = {prefix + ".adapter.linear_in.weight": (2, 3), prefix + ".adapter.linear_out.weight": (8, 2)}
    with pytest.raises(ValueError, match="Unsupported|mismatch"):
        adapter_targets(shapes, {}, 2)


def test_missing_adapter_half_is_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        adapter_targets({"decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight": (2, 3)}, {}, 2)
