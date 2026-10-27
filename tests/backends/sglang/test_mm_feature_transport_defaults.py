# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import dataclasses
from types import SimpleNamespace

import pytest

from relax.backends.sglang.arguments import add_sglang_arguments
from tests.backends.sglang.test_router_registration import sglang_engine_module as sglang_engine_module


def _transport_args(cli):
    parser = add_sglang_arguments(argparse.ArgumentParser(add_help=False))
    parsed = parser.parse_args(cli)
    return SimpleNamespace(
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=8,
        hf_checkpoint="policy-model",
        genrm_model_path="reward-model",
        genrm_num_gpus_per_engine=2,
        genrm_engine_config={},
        seed=42,
        offload_rollout=True,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
        sglang_mm_feature_transport=parsed.sglang_mm_feature_transport,
    )


def _engine_module(request, monkeypatch):
    module = request.getfixturevalue("sglang_engine_module")
    fields = [("mm_feature_transport", str, dataclasses.field(default="cpu"))]
    monkeypatch.setattr(module, "ServerArgs", dataclasses.make_dataclass("ServerArgs", fields))
    monkeypatch.setattr(module, "_to_local_gpu_id", lambda gpu: gpu)
    return module


@pytest.mark.parametrize("worker_type", ["regular", "prefill", "decode"])
@pytest.mark.parametrize(
    "cli,override,expected",
    [
        ([], None, "cuda_ipc"),
        (["--sglang-mm-feature-transport", "cpu"], None, "cpu"),
        (["--sglang-mm-feature-transport", "cuda_ipc"], None, "cuda_ipc"),
        ([], "cpu", "cpu"),
        (["--sglang-mm-feature-transport", "cpu"], "cuda_ipc", "cuda_ipc"),
    ],
)
def test_policy_transport_default_and_explicit_overrides(request, monkeypatch, worker_type, cli, override, expected):
    # Parse with the installed upstream CLI before the engine fixture stubs it.
    args = _transport_args(cli)
    module = _engine_module(request, monkeypatch)
    overrides = {} if override is None else {"mm_feature_transport": override}
    actual, _ = module._compute_server_args(
        args,
        0,
        "localhost:1234",
        1235,
        "localhost",
        1236,
        worker_type=worker_type,
        disaggregation_bootstrap_port=1237,
        base_gpu_id=0,
        sglang_overrides=overrides,
    )
    assert actual["mm_feature_transport"] == expected


@pytest.mark.parametrize(
    "cli,override,expected",
    [
        ([], None, "cuda_ipc"),
        (["--sglang-mm-feature-transport", "cpu"], None, "cpu"),
        (["--sglang-mm-feature-transport", "cpu"], "cuda_ipc", "cuda_ipc"),
    ],
)
def test_genrm_transport_default_and_engine_config_override(request, monkeypatch, cli, override, expected):
    args = _transport_args(cli)
    if override is not None:
        args.genrm_engine_config["mm_feature_transport"] = override
    module = _engine_module(request, monkeypatch)
    actual, _ = module._compute_genrm_server_args(args, 0, "localhost:1234", 1235, "localhost", 1236, base_gpu_id=0)
    assert actual["mm_feature_transport"] == expected
