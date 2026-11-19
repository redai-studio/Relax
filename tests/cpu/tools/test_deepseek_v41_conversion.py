# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the V4.1 converter with real, tiny CPU safetensors checkpoints."""

from __future__ import annotations

import importlib.util
import json
import shutil
import stat
from pathlib import Path
from types import ModuleType

import pytest
import torch
from safetensors.torch import load_file, save_file


@pytest.fixture(scope="module")
def converter() -> ModuleType:
    source = Path(__file__).resolve().parents[3] / "examples/models/deepseek-v4.1/tools/convert_to_bf16.py"
    spec = importlib.util.spec_from_file_location("deepseek_v41_converter_under_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(
    directory: Path, *, omit_dense_scale: bool = False, multimodal: bool = False
) -> dict[str, torch.Tensor]:
    directory.mkdir()
    config = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "eos_token_id": 1,
        "text_config": {
            "model_type": "deepseek_v41_text",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "n_routed_experts": 1,
            "engram_layer_ids": [0],
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
        },
        "vision_config": {"hidden_size": 4},
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
    }
    if multimodal:
        config["text_config"].update(
            num_hidden_layers=2,
            vocab_size=8,
            kv_source_layer_ids=[1],
            index_source_layer_ids=[1],
            candidate_source_layer_id=1,
            engram_layer_ids=[0, 1],
            engram_num_embeddings=[2, 3],
            compress_ratios=[0, 1, 0],
        )
        config["vision_config"].update(num_hidden_layers=1, intermediate_size=3, patch_size=2, downsample_ratio=2)
        config["image_token_id"] = 7
        for asset in ("encoding/encoding.py", "inference/image_processor.py", "inference/vision.py", "tokenizer.json"):
            (directory / asset).parent.mkdir(parents=True, exist_ok=True)
            (directory / asset).write_text(f"# {asset}")
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "tokenizer_config.json").write_text('{"eos_token": "</s>"}\n')

    raw = {
        "embed.weight": torch.tensor([[1.0, -2.0], [0.5, -0.0]], dtype=torch.bfloat16),
        "head.weight": torch.tensor([[3.0, -4.0]], dtype=torch.bfloat16),
        "norm.weight": torch.tensor([-0.0, 0.1, -3.5, 1e-10], dtype=torch.float32),
        "layers.0.ffn.gate.weight": torch.tensor([[0.125, -0.75]], dtype=torch.float32),
        "layers.0.engram.wkv.weight": torch.tensor([[0.5, 0.25]], dtype=torch.bfloat16),
        # Include negative zero and NaN encodings: frozen tables must be copied,
        # not converted through floating-point arithmetic.
        "layers.0.engram.embed.weight": torch.tensor([0, 128, 56, 184, 127, 255, 1, 129], dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .reshape(2, 4),
    }
    for name in ("wkv", "wo_a", "wo_b"):
        raw[f"layers.0.attn.{name}.weight"] = torch.tensor([[1.0, -1.5]], dtype=torch.bfloat16)
    first = dict(raw)
    second = {
        "layers.0.engram.embed.scale": torch.tensor([127, 0, 255, 128], dtype=torch.uint8)
        .view(torch.float8_e8m0fnu)
        .reshape(2, 2),
    }
    expected = {**raw, **second}

    dense_name = "layers.0.attn.wq_a.weight"
    dense = torch.arange(33 * 35, dtype=torch.float32).remainder(9).sub(4).reshape(33, 35)
    first[dense_name] = dense.to(torch.float8_e4m3fn)
    if not omit_dense_scale:
        second["layers.0.attn.wq_a.scale"] = torch.tensor([[127, 128], [126, 129]], dtype=torch.uint8)
    # Explicit quadrants test both 32x32 boundaries, including partial blocks.
    dense_expected = dense.clone()
    dense_expected[:32, 32:] *= 2
    dense_expected[32:, :32] *= 0.5
    dense_expected[32:, 32:] *= 4
    expected[dense_name] = dense_expected.to(torch.bfloat16)

    # Low nibble comes first. Cover all 16 E2M1 encodings, including -0,
    # and two independently scaled groups of 32 decoded elements.
    packed = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 4, dtype=torch.uint8)
    values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    expert_expected = torch.tensor([value * 2 for value in values * 2] + [value * 0.5 for value in values * 2])
    for projection in (1, 2, 3):
        prefix = f"layers.0.ffn.experts.0.w{projection}"
        first[prefix + ".weight"] = packed.clone().view(torch.int8).reshape(1, 32)
        second[prefix + ".scale"] = torch.tensor([[128, 126]], dtype=torch.uint8)
        expected[prefix + ".weight"] = expert_expected.clone().reshape(1, 64).to(torch.bfloat16)

    for name in ("vision.weight", "aligner.weight", "image_newline", "mtp.layers.0.weight"):
        first[name] = torch.ones(1, dtype=torch.bfloat16)
    first["layers.0.ffn.gate.bias_vl"] = torch.ones(1, dtype=torch.float32)

    if multimodal:
        from relax.utils.deepseek_v41_assets import vision_tensor_shapes

        del first["vision.weight"], first["aligner.weight"]
        for offset, (name, shape) in enumerate(vision_tensor_shapes(config).items(), start=1):
            dtype = torch.float32 if "norm" in name or "bias_vl" in name else torch.bfloat16
            first[name] = torch.full(shape, offset, dtype=dtype)
            if not name.startswith("layers.1."):
                expected[name] = first[name]
        for site in ("attn", "ffn"):
            name = f"layers.0.hc_{site}_scale"
            first[name] = expected[name] = torch.zeros(3, dtype=torch.float32)
    weight_map = {}
    total_size = 0
    for number, tensors in enumerate((first, second), start=1):
        filename = f"model-{number:05d}-of-00002.safetensors"
        save_file(tensors, directory / filename)
        weight_map.update({name: filename for name in tensors})
        total_size += sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map})
    )
    return expected


def test_conversion_preserves_text_weights_and_frozen_engram(converter: ModuleType, tmp_path: Path) -> None:
    src, dst = tmp_path / "source", tmp_path / "converted"
    expected = _checkpoint(src)
    converter.convert(src, dst, check_only=True)
    assert not dst.exists()

    converter.convert(src, dst)
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert index["weight_map"].keys() == expected.keys()
    assert index["metadata"]["total_size"] == sum(
        tensor.numel() * tensor.element_size() for tensor in expected.values()
    )
    actual = {}
    for filename in set(index["weight_map"].values()):
        shard = load_file(dst / filename, device="cpu")
        assert all(index["weight_map"][name] == filename for name in shard)
        actual.update(shard)
    for name, tensor in expected.items():
        assert actual[name].dtype == tensor.dtype, name
        assert actual[name].shape == tensor.shape, name
        # Byte equality checks F32/BF16 preservation, FP4 signed zero, and
        # frozen FP8/E8M0 NaN payloads without invoking unsupported FP8 ops.
        assert torch.equal(actual[name].view(torch.uint8), tensor.contiguous().view(torch.uint8)), name

    assert json.loads((dst / "config.json").read_text()) == {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "eos_token_id": 1,
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "n_routed_experts": 1,
        "engram_layer_ids": [0],
        "num_nextn_predict_layers": 0,
    }
    assert (dst / "tokenizer_config.json").read_bytes() == (src / "tokenizer_config.json").read_bytes()
    assert json.loads((dst / "conversion.json").read_text())["complete"] is True


def test_missing_scale_is_rejected_before_writing(converter: ModuleType, tmp_path: Path) -> None:
    src, dst = tmp_path / "source", tmp_path / "converted"
    _checkpoint(src, omit_dense_scale=True)
    with pytest.raises(ValueError, match="Missing or incompatible scale.*layers.0.attn.wq_a.weight"):
        converter.convert(src, dst)
    assert not dst.exists()


def test_resume_retains_completed_shards_and_rejects_another_source(converter: ModuleType, tmp_path: Path) -> None:
    src, dst = tmp_path / "source", tmp_path / "converted"
    _checkpoint(src)
    converter.convert(src, dst)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in dst.glob("*.safetensors")}
    converter.convert(src, dst, resume=True)
    after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in dst.glob("*.safetensors")}
    assert after == before

    other = tmp_path / "another-source"
    shutil.copytree(src, other)
    with pytest.raises(ValueError, match="matching conversion.json"):
        converter.convert(other, dst, resume=True)
    assert json.loads((dst / "conversion.json").read_text())["complete"] is True


def test_resume_rejects_a_truncated_output_shard(converter: ModuleType, tmp_path: Path) -> None:
    src, dst = tmp_path / "source", tmp_path / "converted"
    _checkpoint(src)
    converter.convert(src, dst)
    shard = dst / "model-00001-of-00002.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Existing shard is incomplete or incompatible"):
        converter.convert(src, dst, resume=True)
    assert json.loads((dst / "conversion.json").read_text())["complete"] is False


def test_frozen_vision_sft_checkpoint_roundtrip(converter: ModuleType, tmp_path: Path) -> None:
    import torch.distributed.checkpoint as dcp

    from relax.utils.deepseek_v41_assets import check_frozen_base
    from scripts.tools.convert_checkpoint_to_hf_cpu import export_checkpoint

    src, base, dcp_dir, output = (tmp_path / name for name in ("src", "base", "iter_0000001", "export"))
    source = _checkpoint(src, multimodal=True)
    converter.convert(src, base, keep_vision=True, decoder_layers=1, check_only=True)
    assert not base.exists()
    converter.convert(src, base, keep_vision=True, decoder_layers=1)
    converter.convert(src, base, keep_vision=True, decoder_layers=1, resume=True)
    with pytest.raises(ValueError, match="matching conversion.json"):
        other = tmp_path / "other"
        shutil.copytree(src, other)
        converter.convert(other, base, resume=True, decoder_layers=1, keep_vision=True)
    config = json.loads((base / "config.json").read_text())
    assert config["image_token_id"] == 7
    assert "quantization_config" not in config
    text = config["text_config"]
    assert (text["engram_layer_ids"], text["engram_num_embeddings"], text["compress_ratios"]) == ([0], [2], [0])
    assert text["num_hidden_layers"] == 1 and text["num_nextn_predict_layers"] == 0
    assert (
        text["kv_source_layer_ids"] == text["index_source_layer_ids"] == [] and text["candidate_source_layer_id"] == -1
    )
    assert config["vision_config"] == json.loads((src / "config.json").read_text())["vision_config"]

    # Logical global DCP layout: include a nonzero training change in every language tensor.
    state = {}
    expected = dict(source)
    names = {
        "embedding.word_embeddings.weight": "embed.weight",
        "output_layer.weight": "head.weight",
        "decoder.final_layernorm.weight": "norm.weight",
        "decoder.layers.0.mlp.router.weight": "layers.0.ffn.gate.weight",
        "decoder.layers.0.v41_engram.linear_wkv.weight": "layers.0.engram.wkv.weight",
        "decoder.layers.0.self_attention.linear_kv_proj.weight": "layers.0.attn.wkv.weight",
        "decoder.layers.0.self_attention.linear_q_down_proj.weight": "layers.0.attn.wq_a.weight",
        "decoder.layers.0.self_attention.linear_o_group_proj": "layers.0.attn.wo_a.weight",
        "decoder.layers.0.self_attention.linear_proj.weight": "layers.0.attn.wo_b.weight",
    }
    for source_name, hf_name in names.items():
        state[source_name] = expected[hf_name] = source[hf_name] + 1
    for site, mg in (("ffn", "mlp"), ("attn", "self_attention")):
        expected[f"layers.0.hc_{site}_scale"] = torch.tensor([1.0, 2.0, 3.0])
        for index, part in enumerate(("pre", "post", "res"), start=1):
            state[f"decoder.layers.0.{mg}_hyper_connection.alpha_{part}"] = torch.tensor([float(index)])
    expert = "layers.0.ffn.experts.0."
    for i in (1, 2, 3):
        expected[expert + f"w{i}.weight"] = source[expert + f"w{i}.weight"] + i
    fc1 = torch.cat([expected[expert + f"w{i}.weight"] for i in (1, 3)])
    state["decoder.layers.0.mlp.experts.experts.linear_fc1.weight"] = fc1.unsqueeze(0)
    state["decoder.layers.0.mlp.experts.experts.linear_fc2.weight"] = expected[expert + "w2.weight"].unsqueeze(0)
    dcp.save(state, checkpoint_id=dcp_dir, no_dist=True)
    with pytest.raises(ValueError, match="Missing training base identity"):
        export_checkpoint(dcp_dir, base, output, dry_run=True)
    check_frozen_base(dcp_dir, base, record=True)
    report = export_checkpoint(dcp_dir, base, output, allow_base_fallback=("vision", "engram"))
    assert report["base_fallback_keys"] == sorted(
        source.keys()
        - names.values()
        - {*(expert + f"w{i}.weight" for i in (1, 2, 3)), "layers.0.hc_ffn_scale", "layers.0.hc_attn_scale"}
    )
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    assert index.keys() == expected.keys()
    assert stat.S_IMODE(output.stat().st_mode) == stat.S_IMODE((output / "inference").stat().st_mode)
    file_mode = stat.S_IMODE((output / "model.safetensors.index.json").stat().st_mode)
    for shard in set(index.values()):
        assert stat.S_IMODE((output / shard).stat().st_mode) == file_mode
        for name, actual in load_file(output / shard).items():
            assert torch.equal(actual.view(torch.uint8), expected[name].view(torch.uint8)), name
    for asset in (
        "config.json",
        "tokenizer.json",
        "encoding/encoding.py",
        "inference/image_processor.py",
        "inference/vision.py",
    ):
        assert (output / asset).read_bytes() == (base / asset).read_bytes()

    wrong_base = tmp_path / "wrong-base"
    shutil.copytree(base, wrong_base)
    (wrong_base / "encoding/encoding.py").write_text("# different source")
    with pytest.raises(ValueError, match="base differs"):
        export_checkpoint(
            dcp_dir, wrong_base, tmp_path / "wrong", dry_run=True, allow_base_fallback=("vision", "engram")
        )

    del state["decoder.layers.0.v41_engram.linear_wkv.weight"]
    incomplete = tmp_path / "incomplete"
    dcp.save(state, checkpoint_id=incomplete, no_dist=True)
    check_frozen_base(incomplete, base, record=True)
    with pytest.raises(ValueError, match="layers.0.engram.wkv.weight"):
        export_checkpoint(incomplete, base, tmp_path / "bad", dry_run=True, allow_base_fallback=("vision", "engram"))
