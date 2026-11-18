# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from relax.utils.quant_cast import resolve_rollout_quantization_config


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
