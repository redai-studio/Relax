# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from relax.backends.megatron.weight_update.colocate import (
    agree_fused_colocate,
    fused_colocate_reason,
    receiver_fallback_reason,
    run_fused_phase,
)
from relax.backends.sglang.weight_loader import NATIVE_SYNC_FENCE_LOADER, fence_native_weights


def args():
    return SimpleNamespace(
        colocate=True,
        offload_train=True,
        offload_rollout=True,
        enable_weights_backuper=True,
        bf16=True,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        rollout_num_gpus_per_engine=4,
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=1.0,
        use_precision_aware_optimizer=True,
        megatron_to_hf_mode="bridge",
        update_weights_interval=1,
    )


def reason(config, **kwargs):
    options = dict(role="actor", with_ref=False, with_teacher=False, uses_tms=False)
    options.update(kwargs)
    return fused_colocate_reason(config, **options)


def test_fusion_defaults_on_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("RELAX_COLOCATE_FUSED_WEIGHT_SYNC", raising=False)
    assert reason(args()) is None
    monkeypatch.setenv("RELAX_COLOCATE_FUSED_WEIGHT_SYNC", "0")
    assert reason(args()) == "disabled by configuration"


@pytest.mark.parametrize(
    "field,value",
    [
        ("bf16", False),
        ("colocate", False),
        ("fully_async", True),
        ("hybrid", True),
        ("offload_train", False),
        ("offload_rollout", False),
        ("enable_weights_backuper", False),
        ("keep_old_actor", True),
        ("use_critic", True),
        ("use_opd", True),
        ("lora_rank", 8),
        ("rollout_external", True),
        ("pipeline_model_parallel_size", 2),
        ("context_parallel_size", 2),
        ("expert_tensor_parallel_size", 2),
        ("optimizer_offload_fraction", 0.5),
        ("use_precision_aware_optimizer", False),
        ("megatron_to_hf_mode", "raw"),
        ("update_weights_interval", 2),
        ("mtp_num_layers", 1),
        ("chunked_optimizer_state_offload", True),
        ("cuda_graph_impl", "local"),
        ("nccl_ub", True),
        ("tensor_model_parallel_size", 0),
    ],
)
def test_unsupported_configuration_falls_back_before_state_changes(monkeypatch, field, value):
    monkeypatch.delenv("RELAX_COLOCATE_FUSED_WEIGHT_SYNC", raising=False)
    config = args()
    setattr(config, field, value)
    before = vars(config).copy()
    assert reason(config) is not None
    assert vars(config) == before


@pytest.mark.parametrize(
    "option", [dict(uses_tms=True), dict(with_ref=True), dict(with_teacher=True), dict(role="critic")]
)
def test_unsupported_role_and_memory_policy_fall_back(option):
    assert reason(args(), **option) is not None


def test_gpu_adam_uses_conservative_state_offload_contract():
    config = args()
    config.optimizer_cpu_offload = False
    config.use_precision_aware_optimizer = False
    config.use_distributed_optimizer = True
    config.optimizer = "adam"
    config.optimizer_offload_fraction = 0.5  # Ignored when CPU optimizer is disabled.
    assert reason(config) is None
    config.use_precision_aware_optimizer = True
    assert reason(config) is not None
    config.use_precision_aware_optimizer = False
    config.use_distributed_optimizer = False
    assert reason(config) is not None


@pytest.mark.parametrize("loss_type", ["sft", "dpo", "rm"])
def test_offline_predict_keeps_original_backup_and_sleep_lifecycle(loss_type):
    config = args()
    config.loss_type = loss_type
    config.sft_predict_interval = 1
    assert "offline" in reason(config)


def test_all_ranks_fall_back_if_one_rank_lacks_capability(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)

    def gather(output, local, group):
        output[:] = [local, (local[0], "TMS has no fused lifecycle")]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    assert not agree_fused_colocate(None, "actor-group")

    def mismatched(output, local, group):
        output[:] = [local, ((not local[0][0], *local[0][1:]), None)]

    monkeypatch.setattr(dist, "all_gather_object", mismatched)
    with pytest.raises(RuntimeError, match="disagree"):
        agree_fused_colocate(None, "actor-group")


def test_receiver_layout_and_fence_are_qualified_before_fusion():
    layout = dict(device="cuda", dtype="bfloat16", tp_size=4, custom_weight_loader=[NATIVE_SYNC_FENCE_LOADER])
    common = (["node"] * 8, ["node"] * 2, [4, 4], [0, 4])
    assert receiver_fallback_reason(*common, [layout, layout]) is None
    assert receiver_fallback_reason(*common, [layout, {**layout, "custom_weight_loader": []}])
    assert receiver_fallback_reason(*common, [layout, {**layout, "quantization": "fp8"}])
    assert receiver_fallback_reason(["node"] * 8, ["node", "remote"], [4, 4], [0, 4], [layout, layout])
    assert receiver_fallback_reason(["node"] * 8, ["node"], [4], [0], [layout])


def test_native_receiver_fence_rejects_payload_and_waits_all_streams(monkeypatch):
    sync = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    with pytest.raises(ValueError):
        fence_native_weights(None, [("weight", torch.ones(1))])
    sync.assert_not_called()
    fence_native_weights(None, [])
    sync.assert_called_once_with()


def test_actor_snapshot_is_unreadable_until_complete_backing_is_adopted():
    from relax.utils.training.tensor_backper import _TensorBackuperNormal

    tensor = torch.arange(4).float()
    backup = _TensorBackuperNormal(lambda: [("weight", tensor)])
    backup._backups["actor"] = {"weight": tensor.clone()}
    backup.mark_actor_dirty()
    with pytest.raises(RuntimeError, match="awaiting completion"):
        backup.get("actor")
    with pytest.raises(ValueError, match="shape/dtype"):
        backup.adopt_actor_backing({"weight": torch.zeros(5)})
    with pytest.raises(RuntimeError, match="awaiting completion"):
        backup.get("actor")
    backing = {"weight": tensor.clone() + 1}
    backup.adopt_actor_backing(backing)
    assert backup.get("actor") is backing
    assert torch.equal(backup.get("actor")["weight"], tensor + 1)


def test_runtime_failure_is_collective_and_never_becomes_fallback(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        dist, "all_gather_object", lambda out, local, group: out.__setitem__(slice(None), [local, "peer D2H failed"])
    )
    with pytest.raises(RuntimeError, match="peer D2H failed"):
        run_fused_phase(lambda: "local backing remains owned", "actor-group")


def test_optimizer_aliases_and_unknown_state_are_rejected_before_release():
    pytest.importorskip("megatron.core")
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    from relax.backends.megatron.weight_update.train_offload import _SelectiveOffloadStrategy

    master = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.Adam([master])
    wrapped = object.__new__(DistributedOptimizer)
    wrapped.config = SimpleNamespace(use_precision_aware_optimizer=False)
    wrapped.optimizer = optimizer
    wrapped.shard_fp32_from_float16_groups = [[master]]
    grad = torch.zeros(4)
    buffer = SimpleNamespace(param_data=torch.ones(4, dtype=torch.bfloat16), grad_data=grad)
    model = SimpleNamespace(buffers=[buffer], expert_parallel_buffers=[])
    offloader = _SelectiveOffloadStrategy([model], wrapped, SimpleNamespace(optimizer_cpu_offload=False))
    assert offloader.fused_optimizer_reason() is None  # Lazy state is valid.
    master.grad = torch.ones_like(master)
    optimizer.step()
    assert offloader.fused_optimizer_reason() is None
    optimizer.state[master]["exp_avg"] = grad
    assert "aliases" in offloader.fused_optimizer_reason()
    assert grad.untyped_storage().nbytes() == 16
    optimizer.state[master]["exp_avg"] = torch.zeros_like(master)
    optimizer.state[master]["unknown_master"] = master.clone()
    assert "unknown" in offloader.fused_optimizer_reason()
