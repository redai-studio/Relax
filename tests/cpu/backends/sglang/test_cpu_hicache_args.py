# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import dataclasses
from types import SimpleNamespace

import pytest

from relax.engine.inference.deferred import needs_rollout_cpu_backup
from relax.utils.opd.opd_utils import is_managed_opd_teacher_colocate
from tests.cpu.backends.sglang.test_router_registration import sglang_engine_module as sglang_engine_module


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


@pytest.mark.parametrize("shared_scoring", [False, True])
@pytest.mark.parametrize("override", [None, True, False])
@pytest.mark.parametrize("scorer", ["genrm", "teacher"])
@pytest.mark.parametrize("mtp", [False, True])
def test_shared_scoring_preserves_synced_policy_in_cpu_backup(
    request, monkeypatch, shared_scoring, override, scorer, mtp
):
    module = request.getfixturevalue("sglang_engine_module")
    monkeypatch.setattr(
        module,
        "ServerArgs",
        dataclasses.make_dataclass(
            "ServerArgs",
            [
                (name, bool, dataclasses.field(default=False))
                for name in ("enable_weights_cpu_backup", "enable_draft_weights_cpu_backup", "enable_memory_saver")
            ],
        ),
    )
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
        sglang_enable_weights_cpu_backup=False,  # parser default, not an explicit override
        colocate=True,
        hybrid=False,
        rollout_num_gpus=8,
        resource={"actor": [1, 8], "rollout": [1, 8], "genrm": [1, 8]},
        _genrm_instances_resolved={"__default__": {"num_gpus": 8}},
        _genrm_colocate_with_rollout=shared_scoring,
        defer_reward_to_post_process=shared_scoring,
        enable_mtp_training=mtp,
        sglang_speculative_algorithm="EAGLE" if mtp else None,
    )
    if scorer == "teacher":
        args.resource = {"actor": [1, 8], "rollout": [1, 8], "teacher": [1, 8]}
        args._genrm_instances_resolved = {}
        args.use_opd = True
        args.opd_type = "sglang"
        args.teacher_hf_checkpoint = "teacher"
        args.opd_teacher_defer = shared_scoring
        assert is_managed_opd_teacher_colocate(args)
    overrides = {} if override is None else {"enable_weights_cpu_backup": override}
    assert needs_rollout_cpu_backup(args) is shared_scoring

    def compute():
        return module._compute_server_args(
            args, 0, "localhost:1234", 1235, "localhost", 1236, base_gpu_id=0, sglang_overrides=overrides
        )[0]

    if shared_scoring and override is False:
        with pytest.raises(ValueError, match="cannot disable enable_weights_cpu_backup"):
            compute()
    else:
        actual = compute()
        assert actual["enable_weights_cpu_backup"] is (shared_scoring or override is True)
        if mtp:
            assert actual["enable_draft_weights_cpu_backup"] is shared_scoring
        assert actual["enable_memory_saver"] is True


def test_shared_mtp_cannot_disable_draft_backup(request, monkeypatch):
    module = request.getfixturevalue("sglang_engine_module")
    monkeypatch.setattr(module, "_to_local_gpu_id", lambda gpu: gpu)
    monkeypatch.setattr(module, "ServerArgs", dataclasses.make_dataclass("ServerArgs", []))
    args = SimpleNamespace(
        offload_rollout=True,
        enable_mtp_training=True,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        hf_checkpoint="model",
        seed=42,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
        colocate=True,
        hybrid=False,
        rollout_num_gpus=8,
        resource={"actor": [1, 8], "rollout": [1, 8], "teacher": [1, 8]},
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="teacher",
        opd_teacher_defer=True,
    )
    with pytest.raises(ValueError, match="cannot disable enable_draft_weights_cpu_backup"):
        module._compute_server_args(
            args,
            0,
            "localhost:1234",
            1235,
            "localhost",
            1236,
            base_gpu_id=0,
            sglang_overrides={"enable_draft_weights_cpu_backup": False},
        )
