# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Safety gates and zero-KL inputs for single-update colocate GRPO."""

from types import SimpleNamespace

import pytest

from relax.utils.training.inline_log_probs import can_reuse_colocate_log_probs


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
def inline_guard():
    return lambda actor, batches: can_reuse_colocate_log_probs(
        actor.args,
        batches,
        backup_tags=actor.weights_backuper.backup_tags,
        rollout_data_postprocess=actor.rollout_data_postprocess,
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
    assert not inline_guard(actor, [22])


def test_actual_multiple_updates_are_rejected(actor, inline_guard):
    assert not inline_guard(actor, [11, 11])


def test_extra_model_and_data_mutation_are_rejected(actor, inline_guard):
    actor.weights_backuper.backup_tags.append("ref")
    assert not inline_guard(actor, [22])
    actor.weights_backuper.backup_tags = ["actor"]
    actor.rollout_data_postprocess = lambda args: None
    assert not inline_guard(actor, [22])


def test_existing_modes_do_not_enter_colocate_optimization(actor, inline_guard):
    assert not inline_guard(actor, [11, 11])
    actor.args.true_on_policy_mode = False
    assert not inline_guard(actor, [11, 11])
    actor.args.true_on_policy_mode = True
    actor.args.colocate = False
    actor.args.fully_async = True
    assert not inline_guard(actor, [11, 11])


def test_zero_kl_uses_sampling_shape_without_sampling_values(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("megatron.core", reason="Loss tests require Megatron Core", exc_type=ImportError)
    calls = []

    def compute_impl(args, **kwargs):
        calls.append(kwargs["kl"])
        return kwargs["kl"], kwargs["kl"]

    from relax.backends.megatron import loss

    monkeypatch.setattr(loss.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(loss, "maybe_padded_total_lengths", lambda *args: None)
    monkeypatch.setattr(loss, "compute_advantages_and_returns_impl", compute_impl)
    compute = loss.compute_advantages_and_returns
    args = SimpleNamespace(
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
        compute(args, data, inline_log_probs=True)
        assert "log_probs" not in data
        assert [tuple(x.shape) for x in data["advantages"]] == [(3,), (2,)]
        assert all(x.dtype == torch.float32 and torch.count_nonzero(x) == 0 for x in data["advantages"])
    assert all(torch.equal(a, b) for a, b in zip(*calls, strict=True))
    with pytest.raises(ValueError, match="requires rollout log-probs"):
        compute(args, {"response_lengths": [3], "total_lengths": [5]}, inline_log_probs=True)


@pytest.mark.parametrize("batch", [256, 512])
def test_auto_selection_uses_actual_batch_and_does_not_mutate_arguments(actor, inline_guard, batch):
    actor.args.global_batch_size = batch
    actor.args.rollout_batch_size = batch // 8
    before = vars(actor.args).copy()
    assert inline_guard(actor, [33])
    assert vars(actor.args) == before
