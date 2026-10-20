# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the real actor lifecycle with CPU tensors, without importing the
GPU stack."""

import ast
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def actor_harness():
    path = Path(__file__).resolve().parents[3] / "relax/backends/megatron/actor.py"
    tree = ast.parse(path.read_text())
    actor_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
    )
    names = {
        "train_actor",
        "_switch_model",
        "_can_reuse_actor_ref_log_probs",
        "_update_actor_weight_backups",
        "load_other_checkpoint",
    }
    methods = [node for node in actor_class.body if isinstance(node, ast.FunctionDef) and node.name in names]
    timer = SimpleNamespace(add=Mock())
    namespace = {
        "torch": torch,
        "timer": lambda *args: nullcontext(),
        "inverse_timer": lambda *args: nullcontext(),
        "Timer": lambda: timer,
        "is_offline_mode": lambda args: args.offline,
        "is_preference_mode": lambda args: False,
        "is_lora_enabled": lambda args: args.lora,
        "algorithm_needs_critic": lambda args: False,
        "is_megatron_main_rank": lambda: True,
        "get_gloo_group": lambda: None,
        "device_utils": SimpleNamespace(maybe_backend_process_on_model_switch=lambda: None),
        "logger": Mock(),
        "get_data_iterator": lambda args, model, data, **kwargs: ([SimpleNamespace(data=data)], [1, 1]),
        "compute_advantages_and_returns": Mock(),
        "capture_hooks": Mock(),
        "log_rollout_data": Mock(),
        "train_dump_utils": Mock(),
        "tracking_utils": Mock(),
        "log_perf_data": Mock(),
        "compute_rollout_step": lambda args, step: step,
        "evaluation_step_for_rollout": lambda args, step: step,
        "mpu": SimpleNamespace(
            get_data_parallel_world_size=lambda **kwargs: 1, get_data_parallel_group=lambda **kwargs: None
        ),
        "dist": SimpleNamespace(
            all_gather_object=lambda output, value, **kwargs: output.__setitem__(0, value),
            barrier=lambda **kwargs: None,
        ),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    actor_type = type("ActorHarness", (), {name: namespace[name] for name in names})
    instance = actor_type()
    instance.args = Namespace(
        colocate=True,
        fully_async=False,
        hybrid=False,
        offline=False,
        keep_old_actor=False,
        use_opd=False,
        num_experts=None,
        fp8=None,
        lora=False,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        custom_megatron_before_log_prob_hook_path=None,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        use_rollout_logprobs=False,
        get_mismatch_metrics=False,
        use_dynamic_batch_size=True,
        log_probs_max_tokens_per_gpu=24576,
        max_tokens_per_gpu=24576,
        compute_advantages_and_returns=True,
        use_critic=False,
        ref_update_interval=1,
        num_rollout=200,
        save=None,
        offload_train=False,
        load=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        ref_ckpt_step=None,
    )
    instance.model = torch.nn.Linear(3, 5, bias=False, dtype=torch.float64)
    with torch.no_grad():
        instance.model.weight.copy_(torch.arange(15, dtype=torch.float64).view(5, 3) / 20)
    instance.optimizer = torch.optim.SGD(instance.model.parameters(), lr=0.03)
    instance.opt_param_scheduler = None
    instance._active_model_tag = "actor"
    instance._ref_actor_weights_synced = False
    instance._is_standard_dpo = lambda: False
    instance._per_step_rollout = False
    instance._run_step_evaluation = Mock()
    instance.rollout_data_postprocess = None
    instance.prof = Mock()
    instance.tokenizer = None
    instance.flops_counter = None
    snapshots = {}

    def backup(tag):
        snapshots[tag] = instance.model.weight.detach().clone()

    def restore(tag):
        with torch.no_grad():
            instance.model.weight.copy_(snapshots[tag])

    instance.weights_backuper = SimpleNamespace(
        backup=Mock(side_effect=backup), restore=restore, backup_tags=["actor", "ref"]
    )
    backup("actor")
    backup("ref")
    instance.forward_calls = []

    def compute_log_prob(iterators, counts, store_prefix="", **kwargs):
        instance.forward_calls.append(store_prefix)
        with torch.no_grad():
            values = instance.model(iterators[0].data["inputs"]).log_softmax(-1)
        return {f"{store_prefix}log_probs": list(values.unbind())}

    instance.compute_log_prob = compute_log_prob
    instance.reference_at_updates = []

    def train(rollout_id, model, optimizer, scheduler, iterators, counts):
        assert not instance._ref_actor_weights_synced
        data = iterators[0].data
        # Two real optimizer updates; both must use the same pre-training reference.
        for i in range(2):
            instance.reference_at_updates.append(torch.stack(data["ref_log_probs"]).clone())
            log_probs = model(data["inputs"][i : i + 1]).log_softmax(-1).squeeze(0)
            reference = data["ref_log_probs"][i]
            loss = -log_probs[i] + 0.001 * torch.nn.functional.kl_div(log_probs, reference.exp(), reduction="sum")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    namespace["train"] = train

    def data():
        return {
            "inputs": torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]], dtype=torch.float64),
            "total_lengths": [5, 5],
            "loss_masks": [torch.ones(5), torch.ones(5)],
        }

    return instance, namespace, data, snapshots


def test_reuse_matches_two_forwards_and_two_optimizer_updates(actor_harness):
    actor, namespace, make_data, snapshots = actor_harness
    actor.train_actor(0, make_data())
    assert actor.forward_calls == ["ref_", ""]
    assert actor._ref_actor_weights_synced
    initial = actor.model.weight.detach().clone()
    data = make_data()
    actor.train_actor(1, data)
    assert actor.forward_calls == ["ref_", "", ""]
    optimized_weights = actor.model.weight.detach().clone()
    assert torch.equal(actor.reference_at_updates[-1], actor.reference_at_updates[-2])
    assert data["ref_log_probs"] is not data["log_probs"]
    for reference, old in zip(data["ref_log_probs"], data["log_probs"], strict=True):
        assert torch.equal(reference, old)
        assert reference.data_ptr() != old.data_ptr()
        assert not reference.requires_grad
    data["log_probs"][0].add_(1)
    assert not torch.equal(data["log_probs"][0], data["ref_log_probs"][0])

    # Replay the same update with the original two-forward path.
    with torch.no_grad():
        actor.model.weight.copy_(initial)
    snapshots["actor"] = initial.clone()
    snapshots["ref"] = initial.clone()
    actor._ref_actor_weights_synced = False
    actor.train_actor(1, make_data())
    assert actor.forward_calls[-2:] == ["ref_", ""]
    torch.testing.assert_close(actor.model.weight, optimized_weights, rtol=0, atol=0)
    namespace["Timer"]().add.assert_called_with("ref_log_probs", 0.0)


def test_reuse_tracks_actual_reference_update_interval(actor_harness):
    actor, _, make_data, _ = actor_harness
    actor.args.ref_update_interval = 2
    calls = []
    for step in range(4):
        actor.forward_calls.clear()
        actor.train_actor(step, make_data())
        calls.append(actor.forward_calls.copy())
    assert calls == [["ref_", ""], ["ref_", ""], [""], ["ref_", ""]]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("colocate", False),
        ("fully_async", True),
        ("hybrid", True),
        ("offline", True),
        ("keep_old_actor", True),
        ("use_opd", True),
        ("num_experts", 8),
        ("fp8", "hybrid"),
        ("fp4", "e2m1"),
        ("lora", True),
        ("use_routing_replay", True),
        ("use_rollout_routing_replay", True),
        ("use_rollout_indexer_replay", True),
        ("custom_megatron_before_log_prob_hook_path", "custom.hook"),
        ("pipeline_model_parallel_size", 2),
        ("context_parallel_size", 2),
        ("dynamic_context_parallel", True),
        ("use_rollout_logprobs", True),
        ("log_probs_max_tokens_per_gpu", 49152),
    ],
)
def test_unsupported_execution_does_not_reuse(actor_harness, name, value):
    actor, _, _, _ = actor_harness
    actor._ref_actor_weights_synced = True
    assert actor._can_reuse_actor_ref_log_probs()
    setattr(actor.args, name, value)
    assert not actor._can_reuse_actor_ref_log_probs()


@pytest.mark.parametrize("interval", [None, 2])
def test_no_reference_update_never_establishes_provenance(actor_harness, interval):
    actor, _, make_data, _ = actor_harness
    actor.args.ref_update_interval = interval
    actor.train_actor(0, make_data())
    assert not actor._ref_actor_weights_synced


def test_failed_ref_backup_invalidates_provenance(actor_harness):
    actor, _, _, _ = actor_harness
    actor._ref_actor_weights_synced = True
    actor.weights_backuper.backup.side_effect = [None, RuntimeError("backup failed")]
    with pytest.raises(RuntimeError, match="backup failed"):
        actor._update_actor_weight_backups(0)
    assert not actor._ref_actor_weights_synced


def test_failed_training_invalidates_provenance(actor_harness):
    actor, namespace, make_data, _ = actor_harness
    actor._ref_actor_weights_synced = True
    namespace["train"] = Mock(side_effect=RuntimeError("optimizer failed"))
    with pytest.raises(RuntimeError, match="optimizer failed"):
        actor.train_actor(1, make_data())
    assert not actor._ref_actor_weights_synced


def test_checkpoint_reload_invalidates_provenance_even_on_failure(actor_harness):
    actor, namespace, _, _ = actor_harness
    actor._ref_actor_weights_synced = True
    namespace["load_checkpoint"] = Mock(side_effect=RuntimeError("load failed"))
    with pytest.raises(RuntimeError, match="load failed"):
        actor.load_other_checkpoint("ref", "checkpoint")
    assert not actor._ref_actor_weights_synced


def test_different_forward_budgets_keep_both_forwards(actor_harness):
    actor, _, make_data, _ = actor_harness
    actor._ref_actor_weights_synced = True
    actor.args.log_probs_max_tokens_per_gpu = 49152
    actor.train_actor(1, make_data())
    assert actor.forward_calls == ["ref_", ""]


def test_reference_without_actor_forward_is_not_reused(actor_harness):
    actor, _, make_data, _ = actor_harness
    actor._ref_actor_weights_synced = True
    actor.args.use_rollout_logprobs = True
    actor.train_actor(1, make_data())
    assert actor.forward_calls == ["ref_"]


def test_mismatch_metrics_actor_forward_can_supply_reference(actor_harness):
    actor, _, make_data, _ = actor_harness
    actor._ref_actor_weights_synced = True
    actor.args.use_rollout_logprobs = True
    actor.args.get_mismatch_metrics = True
    data = make_data()
    actor.train_actor(1, data)
    assert actor.forward_calls == [""]
    assert all(torch.equal(a, b) for a, b in zip(data["log_probs"], data["ref_log_probs"], strict=True))


@pytest.mark.parametrize("change", ["missing_ref", "teacher", "active_ref"])
def test_non_actor_snapshot_does_not_reuse(actor_harness, change):
    actor, _, _, _ = actor_harness
    actor._ref_actor_weights_synced = True
    if change == "missing_ref":
        actor.weights_backuper.backup_tags.remove("ref")
    elif change == "teacher":
        actor.weights_backuper.backup_tags.append("teacher")
    else:
        actor._active_model_tag = "ref"
    assert not actor._can_reuse_actor_ref_log_probs()
