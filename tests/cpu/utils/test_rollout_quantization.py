# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import struct
from argparse import Namespace
from types import SimpleNamespace

import pytest

from relax.utils.quant_cast import resolve_receiver_quantization_config


@pytest.mark.parametrize(
    "extra,compatible",
    [
        ({"fmt": "e4m3", "weight_block_size": None, "scale_fmt": None, "modules_to_not_convert": []}, True),
        ({"weight_block_size": [128, 128]}, False),
        ({"scale_fmt": "ue8m0"}, False),
        ({"fmt": "e5m2"}, False),
    ],
)
def test_receiver_fp8_compares_effective_defaults(quantization_args, extra, compatible):
    first = {"quant_method": "fp8", "activation_scheme": "dynamic"}
    recipients = [
        ("first", quantization_args.hf_checkpoint, json.dumps({"quantization_config": first}), None),
        ("second", quantization_args.hf_checkpoint, json.dumps({"quantization_config": {**first, **extra}}), None),
    ]
    if compatible:
        assert resolve_receiver_quantization_config(quantization_args, SimpleNamespace(), recipients) == first
    else:
        with pytest.raises(ValueError, match="different quantization configs"):
            resolve_receiver_quantization_config(quantization_args, SimpleNamespace(), recipients)


@pytest.fixture
def compressed_checkpoint(tmp_path):
    pytest.importorskip("safetensors")
    checkpoint = tmp_path / "packed"
    checkpoint.mkdir()
    config = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "group",
                    "group_size": 128,
                    "symmetric": True,
                },
            }
        },
        "ignore": [],
    }
    (checkpoint / "config.json").write_text(json.dumps({"compression_config": config}))
    # A real, tiny safetensors file exercises the production header scan.
    header = json.dumps(
        {
            "vision_tower.proj.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            "model.layers.0.mlp.experts.0.up_proj.weight_packed": {
                "dtype": "U8",
                "shape": [1],
                "data_offsets": [2, 3],
            },
        }
    ).encode()
    header += b" " * (-len(header) % 8)
    (checkpoint / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0")
    return checkpoint, config


@pytest.mark.parametrize("alias", ["source", "original"])
def test_receiver_s3_alias_uses_local_metadata_and_shards(quantization_args, compressed_checkpoint, alias):
    from relax.utils.model_source import ModelSource

    path, _ = compressed_checkpoint
    args = quantization_args
    args.hf_checkpoint = str(path)
    args.model_source = ModelSource(uri="s3://test-bucket/model")
    args._model_source_original_hf_checkpoint = "s3://test-bucket/original"
    receiver = args.model_source.uri if alias == "source" else args._model_source_original_hf_checkpoint
    actual = resolve_receiver_quantization_config(args, SimpleNamespace(), [("policy", receiver, None, None)])
    assert actual["quant_method"] == "compressed-tensors"
    assert "re:.*vision_tower.*" in actual["ignore"]


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"quantization_config": None}, "compressed-tensors"),
        ({"quantization_config": None, "compression_config": None}, None),
        ({"quantization_config": {}}, None),
        ({"compression_config": {"quant_method": "other"}}, "other"),
    ],
)
def test_receiver_overrides_apply_before_compression_fallback(
    quantization_args, tmp_path, monkeypatch, overrides, expected
):
    from relax.utils import quant_cast

    path = tmp_path / "receiver_config"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"compression_config": {"quant_method": "compressed-tensors"}}))
    # This case tests metadata precedence; the separate packed-file case
    # exercises augmentation and shard discovery.
    monkeypatch.setattr(quant_cast, "augment_compressed_tensors_ignore", lambda config, _, **kwargs: config)
    actual = resolve_receiver_quantization_config(
        quantization_args, SimpleNamespace(), [("policy", str(path), json.dumps(overrides), None)]
    )
    assert actual == (None if expected is None else {"quant_method": expected})


@pytest.fixture
def quantization_args(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    return Namespace(
        hf_checkpoint=str(checkpoint),
        sglang_hf_checkpoint=None,
        sglang_json_model_override_args=None,
        sglang_config=None,
        sglang_quantization=None,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
        fully_async=False,
        hybrid=False,
    )


@pytest.mark.parametrize(
    "override,expected",
    [
        ({}, {"quant_method": "fp8"}),
        ({"quantization_config": None}, None),
        ({"quantization_config": {"quant_method": "other"}}, {"quant_method": "other"}),
    ],
)
def test_rollout_quantization_preserves_explicit_null(quantization_args, override, expected):
    args = quantization_args
    actual = resolve_receiver_quantization_config(
        args, SimpleNamespace(), [("receiver", args.hf_checkpoint, json.dumps(override), None)]
    )
    assert actual == expected


@pytest.mark.parametrize(
    "overrides,root_null,requested_quantization,keeps_fp8",
    [
        ({"text_config": {"use_cache": False}}, False, None, True),
        ({"text_config": {"quantization_config": None}}, False, None, True),
        ({"quantization_config": None, "text_config": {"quantization_config": None}}, False, None, False),
        ({"text_config": {"quantization_config": None}}, True, None, False),
        ({"quantization_config": None}, False, None, False),
        ({"text_config": {"use_cache": False}}, True, None, False),
        ({"text_config": {"use_cache": False}}, True, "fp8", True),
        ({"quantization_config": None}, False, "fp8", True),
    ],
)
def test_rollout_quantization_matches_receiver_text_config(
    quantization_args, overrides, root_null, requested_quantization, keeps_fp8
):
    from pathlib import Path

    from transformers import Qwen3VLConfig

    ModelConfig = pytest.importorskip("sglang.srt.configs.model_config").ModelConfig
    LoadConfig = pytest.importorskip("sglang.srt.configs.load_config").LoadConfig
    get_quant_config = pytest.importorskip("sglang.srt.model_loader.weight_utils").get_quant_config
    expected = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}
    path = Path(quantization_args.hf_checkpoint)
    config = Qwen3VLConfig(text_config={"quantization_config": expected})
    config.architectures = ["Qwen3VLForConditionalGeneration"]
    if root_null:
        config.quantization_config = None
    config.save_pretrained(path)
    receiver = ModelConfig(str(path), model_override_args=json.dumps(overrides), quantization=requested_quantization)
    if receiver.quantization is not None:
        loaded = get_quant_config(receiver, LoadConfig(load_format="dummy"), {})
        assert loaded.get_name() == "fp8"
        assert loaded.is_checkpoint_fp8_serialized
        assert loaded.weight_block_size == [128, 128]
    actual = resolve_receiver_quantization_config(
        quantization_args,
        SimpleNamespace(),
        [("receiver", str(path), json.dumps(overrides), requested_quantization)],
    )
    assert (actual["quant_method"] if actual else None) == receiver.quantization == ("fp8" if keeps_fp8 else None)
    assert actual == (expected if keeps_fp8 else None)


def test_receiver_reuses_shard_headers_without_reusing_group_overrides(
    quantization_args, compressed_checkpoint, monkeypatch
):
    from unittest.mock import Mock

    import safetensors

    path, config = compressed_checkpoint
    open_header = Mock(wraps=safetensors.safe_open)
    monkeypatch.setattr(safetensors, "safe_open", open_header)
    recipients = [
        ("prefill", str(path), None, None),
        ("decode", str(path), json.dumps({"compression_config": {**config, "ignore": ["decode_only"]}}), None),
    ]
    with pytest.raises(ValueError, match="different quantization configs"):
        resolve_receiver_quantization_config(quantization_args, SimpleNamespace(), recipients)
    assert open_header.call_count == 1
    # A later resolution must read the checkpoint again, rather than retain a
    # process-global snapshot after new shards appear.
    import torch
    from safetensors.torch import save_file

    save_file({"new_namespace.weight": torch.ones(1)}, path / "new.safetensors")
    actual = resolve_receiver_quantization_config(
        quantization_args, SimpleNamespace(), [("prefill", str(path), None, None)]
    )
    assert "re:.*new_namespace.*" in actual["ignore"]
    assert open_header.call_count == 3


@pytest.mark.parametrize("method", ["FP8", "modelopt_fp4"])
def test_receiver_mode_names_preserve_dense_block_fp8_layout(quantization_args, method):
    from pathlib import Path

    from transformers import Qwen3Config

    ModelConfig = pytest.importorskip("sglang.srt.configs.model_config").ModelConfig
    if method == "modelopt_fp4":
        from importlib.metadata import version

        from packaging.version import Version
        from sglang.srt.utils import is_hip

        if is_hip():
            pytest.skip("modelopt_fp4 is not supported by SGLang on ROCm")
        if Version(version("sglang")) < Version("0.5.21"):
            pytest.skip("Loading serialized FP8 with modelopt_fp4 requires SGLang >= 0.5.21")
    LoadConfig = pytest.importorskip("sglang.srt.configs.load_config").LoadConfig
    get_quant_config = pytest.importorskip("sglang.srt.model_loader.weight_utils").get_quant_config
    LinearBase = pytest.importorskip("sglang.srt.layers.linear").LinearBase
    Fp8LinearMethod = pytest.importorskip("sglang.srt.layers.quantization.fp8").Fp8LinearMethod
    expected = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}
    path = Path(quantization_args.hf_checkpoint)
    config = Qwen3Config(quantization_config=expected)
    config.architectures = ["Qwen3ForCausalLM"]
    config.save_pretrained(path)
    receiver = ModelConfig(str(path), quantization=method)
    loaded = get_quant_config(receiver, LoadConfig(load_format="dummy"), {})
    layer = LinearBase(128, 128, quant_config=loaded, prefix="model.layers.0.self_attn.q_proj")
    assert receiver.quantization == method.lower()
    assert isinstance(layer.quant_method, Fp8LinearMethod)
    assert layer.quant_method.is_checkpoint_fp8_serialized
    assert loaded.weight_block_size == [128, 128]
    actual = resolve_receiver_quantization_config(
        quantization_args, SimpleNamespace(), [("receiver", str(path), None, method)]
    )
    assert actual == expected
