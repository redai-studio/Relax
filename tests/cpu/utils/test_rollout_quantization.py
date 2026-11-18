# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import struct
from argparse import Namespace
from types import SimpleNamespace

import pytest

from relax.distributed.ray.rollout_weight_config import resolve_rollout_quantization_config
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
        ("first", quantization_args.hf_checkpoint, json.dumps({"quantization_config": first})),
        ("second", quantization_args.hf_checkpoint, json.dumps({"quantization_config": {**first, **extra}})),
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


def test_receiver_compression_metadata_drives_packing(quantization_args, compressed_checkpoint):
    path, config = compressed_checkpoint
    quantization_args.sglang_hf_checkpoint = str(path)
    actual = resolve_rollout_quantization_config(
        quantization_args, SimpleNamespace(quantization_config={"quant_method": "fp8"})
    )
    assert actual["quant_method"] == "compressed-tensors"
    assert actual["format"] == "pack-quantized"
    assert "re:.*vision_tower.*" in actual["ignore"]
    assert config["ignore"] == []


@pytest.mark.parametrize("alias", ["source", "original"])
def test_receiver_s3_alias_uses_local_metadata_and_shards(quantization_args, compressed_checkpoint, alias):
    from relax.utils.model_source import ModelSource

    path, _ = compressed_checkpoint
    args = quantization_args
    args.hf_checkpoint = str(path)
    args.model_source = ModelSource(uri="s3://test-bucket/model")
    args._model_source_original_hf_checkpoint = "s3://test-bucket/original"
    receiver = args.model_source.uri if alias == "source" else args._model_source_original_hf_checkpoint
    actual = resolve_receiver_quantization_config(args, SimpleNamespace(), [("policy", receiver, None)])
    assert actual["quant_method"] == "compressed-tensors"
    assert "re:.*vision_tower.*" in actual["ignore"]


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"quantization_config": None}, "compressed-tensors"),
        ({"quantization_config": None, "compression_config": None}, None),
        ({"quantization_config": {}}, "empty"),
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
    monkeypatch.setattr(quant_cast, "augment_compressed_tensors_ignore", lambda config, _: config)
    actual = resolve_receiver_quantization_config(
        quantization_args, SimpleNamespace(), [("policy", str(path), json.dumps(overrides))]
    )
    assert actual == (None if expected is None else {} if expected == "empty" else {"quant_method": expected})


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
    args.sglang_json_model_override_args = json.dumps(override)
    assert resolve_rollout_quantization_config(args, SimpleNamespace()) == expected


def test_rollout_quantization_uses_receiver_checkpoint(quantization_args, tmp_path):
    pytest.importorskip("transformers")
    receiver = tmp_path / "receiver"
    receiver.mkdir()
    expected = {"quant_method": "other"}
    (receiver / "config.json").write_text(json.dumps({"text_config": {"quantization_config": expected}}))
    quantization_args.sglang_hf_checkpoint = str(receiver)
    assert (
        resolve_rollout_quantization_config(
            quantization_args, SimpleNamespace(quantization_config={"quant_method": "fp8"})
        )
        == expected
    )


def _write_groups(args, tmp_path, models):
    pytest.importorskip("relax.distributed.ray.rollout")
    path = tmp_path / "rollout.yaml"
    path.write_text(json.dumps({"sglang": models}))
    args.sglang_config = str(path)
    args.rollout_num_gpus = sum(group["num_gpus"] for model in models for group in model["engine_groups"])


@pytest.mark.parametrize(
    "override,expected",
    [
        ("{}", {"quant_method": "fp8"}),
        ('{"quantization_config": null}', None),
    ],
)
def test_rollout_quantization_group_json_replaces_global(quantization_args, tmp_path, override, expected):
    args = quantization_args
    args.sglang_json_model_override_args = '{"quantization_config": {"quant_method": "other"}}'
    _write_groups(
        args,
        tmp_path,
        [
            {
                "name": "policy",
                "engine_groups": [
                    {"worker_type": "regular", "num_gpus": 2, "overrides": {"json_model_override_args": override}},
                ],
            }
        ],
    )
    assert resolve_rollout_quantization_config(args, SimpleNamespace()) == expected


@pytest.mark.parametrize("fully_async,hybrid", [(False, False), (True, False), (False, True)])
def test_rollout_quantization_checks_publication_recipients(quantization_args, tmp_path, fully_async, hybrid):
    args = quantization_args
    args.fully_async, args.hybrid = fully_async, hybrid
    _write_groups(
        args,
        tmp_path,
        [
            {
                "name": "policy",
                "engine_groups": [
                    {"worker_type": "regular", "num_gpus": 1},
                    {
                        "worker_type": "placeholder",
                        "num_gpus": 1,
                        "overrides": {"json_model_override_args": '{"quantization_config": null}'},
                    },
                ],
            },
            {
                "name": "second",
                "engine_groups": [
                    {
                        "worker_type": "regular",
                        "num_gpus": 1,
                        "overrides": {"json_model_override_args": '{"quantization_config": null}'},
                    },
                ],
            },
        ],
    )
    if fully_async or hybrid:
        with pytest.raises(ValueError, match="policy/0:regular and second/0:regular"):
            resolve_rollout_quantization_config(args, SimpleNamespace())
    else:
        assert resolve_rollout_quantization_config(args, SimpleNamespace()) == {"quant_method": "fp8"}


@pytest.mark.parametrize("worker_types", [("regular", "regular"), ("prefill", "decode")])
def test_rollout_quantization_rejects_mixed_group_formats(quantization_args, tmp_path, worker_types):
    args = quantization_args
    _write_groups(
        args,
        tmp_path,
        [
            {
                "name": "policy",
                "engine_groups": [
                    {"worker_type": worker_types[0], "num_gpus": 1},
                    {
                        "worker_type": worker_types[1],
                        "num_gpus": 1,
                        "overrides": {"json_model_override_args": '{"quantization_config": null}'},
                    },
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="different quantization configs"):
        resolve_rollout_quantization_config(args, SimpleNamespace())
