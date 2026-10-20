# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import dataclasses
from types import SimpleNamespace

import pytest

from tests.backends.sglang.test_router_registration import sglang_engine_module as sglang_engine_module


@pytest.mark.parametrize("worker_type", ["regular", "prefill", "decode"])
@pytest.mark.parametrize("override", [None, True, False])
def test_cpu_cache_is_independent_of_pd_topology(request, monkeypatch, worker_type, override):
    module = request.getfixturevalue("sglang_engine_module")
    fields = [
        ("enable_hierarchical_cache", bool, dataclasses.field(default=False)),
        ("hicache_storage_backend", str, dataclasses.field(default=None)),
        ("disaggregation_mode", str, dataclasses.field(default="null")),
    ]
    monkeypatch.setattr(module, "ServerArgs", dataclasses.make_dataclass("ServerArgs", fields))
    monkeypatch.setattr(module, "_to_local_gpu_id", lambda gpu: gpu)
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=8,
        hf_checkpoint="model",
        seed=42,
        offload_rollout=True,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
        sglang_enable_hierarchical_cache=True,
        sglang_hicache_storage_backend=None,
    )
    overrides = {} if override is None else {"enable_hierarchical_cache": override}
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
    expected = worker_type != "decode" if override is None else override
    assert actual.get("enable_hierarchical_cache", False) is expected
    assert actual["hicache_storage_backend"] is None
    assert actual.get("disaggregation_mode", "null") == ("null" if worker_type == "regular" else worker_type)
