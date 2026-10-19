# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Safety gates and zero-KL inputs for single-update colocate GRPO."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def actor():
    args = SimpleNamespace(
        colocate=True,
        true_on_policy_mode=True,
        fully_async=False,
        hybrid=False,
        advantage_estimator="grpo",
        loss_type="policy_loss",
        compute_advantages_and_returns=True,
        rollout_batch_size=32,
        n_samples_per_prompt=8,
        global_batch_size=256,
        num_iters_per_train_update=1,
        max_staleness=0,
        update_weights_interval=1,
        partial_rollout=False,
        rl_partial_rollouts=False,
        use_dynamic_global_batch_size=False,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        dynamic_context_parallel=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        moe_input_jitter_eps=None,
        bf16=True,
        fp16=False,
        fp8=None,
        fp4=None,
        fp8_param_gather=False,
        moe_router_enable_expert_bias=False,
        moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type=["none"],
        moe_router_force_load_balancing=False,
        moe_router_force_biased=None,
        moe_expert_capacity_factor=None,
        kl_coef=0.0,
        use_kl_loss=False,
        use_opd=False,
        use_critic=False,
        keep_old_actor=False,
        use_rollout_logprobs=False,
        use_routing_replay=True,
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=False,
        enable_mtp_training=False,
        use_opsm=False,
        use_tis=True,
        calculate_per_token_loss=True,
        entropy_coef=0.0,
        recompute_loss_function=False,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        custom_loss_function_path=None,
        custom_model_provider_path=None,
        custom_megatron_before_log_prob_hook_path=None,
        custom_megatron_before_train_step_hook_path=None,
    )
    return SimpleNamespace(
        args=args, weights_backuper=SimpleNamespace(backup_tags=["actor"]), rollout_data_postprocess=None
    )


@pytest.fixture
def inline_guard(monkeypatch):
    monkeypatch.setenv("RELAX_COLOCATE_INLINE_LOG_PROBS", "1")
    return load_function(
        ROOT / "relax/backends/megatron/actor.py",
        "_inline_colocate_old_log_probs",
        {"os": os},
        "MegatronTrainRayActor",
    )


def test_single_fresh_grpo_update_is_eligible(actor, inline_guard):
    assert inline_guard(actor, [22])


@pytest.mark.parametrize(
    "name,value",
    [
        ("colocate", False),
        ("true_on_policy_mode", False),
        ("num_iters_per_train_update", 2),
        ("max_staleness", 1),
        ("update_weights_interval", 2),
        ("global_batch_size", 128),
        ("partial_rollout", True),
        ("use_dynamic_global_batch_size", True),
        ("context_parallel_size", 2),
        ("dynamic_context_parallel", True),
        ("hidden_dropout", 0.1),
        ("attention_dropout", 0.1),
        ("moe_input_jitter_eps", 0.01),
        ("fp8", "hybrid"),
        ("fp4", "nvfp4"),
        ("fp8_param_gather", True),
        ("moe_router_force_load_balancing", True),
        ("moe_router_force_biased", 0.1),
        ("moe_expert_capacity_factor", 1.0),
        ("custom_model_provider_path", "custom.stateful_model"),
        ("moe_router_enable_expert_bias", True),
        ("moe_router_load_balancing_type", ["aux_loss"]),
        ("kl_coef", 0.01),
        ("use_kl_loss", True),
        ("use_opd", True),
        ("use_critic", True),
        ("keep_old_actor", True),
        ("use_rollout_logprobs", True),
        ("use_rollout_routing_replay", False),
        ("enable_mtp_training", True),
        ("use_opsm", True),
        ("custom_tis_function_path", "custom.function"),
        ("custom_megatron_before_train_step_hook_path", "custom.mutate_weights"),
    ],
)
def test_unsafe_algorithm_or_state_is_rejected(actor, inline_guard, name, value):
    setattr(actor.args, name, value)
    with pytest.raises(ValueError, match="Unsupported colocate true-on-policy"):
        inline_guard(actor, [22])


def test_actual_multiple_updates_are_rejected(actor, inline_guard):
    with pytest.raises(ValueError, match="exactly one optimizer step"):
        inline_guard(actor, [11, 11])


def test_extra_model_and_data_mutation_are_rejected(actor, inline_guard):
    actor.weights_backuper.backup_tags.append("ref")
    with pytest.raises(ValueError, match="reference"):
        inline_guard(actor, [22])
    actor.weights_backuper.backup_tags = ["actor"]
    actor.rollout_data_postprocess = lambda args: None
    with pytest.raises(ValueError, match="rollout_data_postprocess"):
        inline_guard(actor, [22])


def test_existing_modes_do_not_enter_colocate_optimization(actor, inline_guard, monkeypatch):
    monkeypatch.delenv("RELAX_COLOCATE_INLINE_LOG_PROBS")
    assert not inline_guard(actor, [11, 11])
    actor.args.true_on_policy_mode = False
    assert not inline_guard(actor, [11, 11])
    actor.args.true_on_policy_mode = True
    actor.args.colocate = False
    actor.args.fully_async = True
    assert not inline_guard(actor, [11, 11])


def test_zero_kl_uses_sampling_shape_without_sampling_values():
    torch = pytest.importorskip("torch")
    calls = []

    def compute_impl(args, **kwargs):
        calls.append(kwargs["kl"])
        return kwargs["kl"], kwargs["kl"]

    namespace = {
        "torch": torch,
        "mpu": SimpleNamespace(is_pipeline_last_stage=lambda: True),
        "maybe_padded_total_lengths": lambda *args: None,
        "compute_advantages_and_returns_impl": compute_impl,
    }
    compute = load_function(ROOT / "relax/backends/megatron/loss.py", "compute_advantages_and_returns", namespace)
    args = SimpleNamespace(
        _colocate_inline_log_probs=True,
        advantage_estimator="grpo",
        kl_coef=0,
        use_rollout_logprobs=False,
        qkv_format="thd",
        use_opd=False,
        normalize_advantages=False,
    )
    for value in (-0.1, -100):
        data = {
            "rollout_log_probs": [torch.full((3,), value, dtype=torch.float16), torch.full((2,), value)],
            "rewards": [1.0, -1.0],
            "response_lengths": [3, 2],
            "total_lengths": [5, 7],
            "loss_masks": [torch.ones(3), torch.ones(2)],
        }
        compute(args, data)
        assert "log_probs" not in data
        assert [tuple(x.shape) for x in data["advantages"]] == [(3,), (2,)]
        assert all(x.dtype == torch.float32 and torch.count_nonzero(x) == 0 for x in data["advantages"])
    assert all(torch.equal(a, b) for a, b in zip(*calls, strict=True))
    with pytest.raises(ValueError, match="requires rollout log-probs"):
        compute(args, {"response_lengths": [3], "total_lengths": [5]})
