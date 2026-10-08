# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Zero-token no-signal step handling for the shared Megatron trainer."""

from datetime import timedelta
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")


def _zero_token_process_group_worker(rank: int, world_size: int, init_method: str, tp: int, pp: int) -> None:
    import torch.distributed as dist
    from megatron.core import parallel_state

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        from relax.backends.megatron.model import _is_global_zero_token_step

        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
        # Gloo collectives need CPU tensors for the decision signal.
        torch.cuda.current_device = lambda: "cpu"
        is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        is_first_dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0

        def losses(num_tokens: int) -> list[dict[str, object]]:
            return [{"num_tokens": torch.tensor(num_tokens, dtype=torch.int)}] if is_last_stage else []

        assert _is_global_zero_token_step(losses(0)) is True
        # Only one DP replica sees tokens: every rank of every pipeline must still train.
        assert _is_global_zero_token_step(losses(3 if is_first_dp_rank else 0)) is False
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize(("tp", "pp"), [(2, 2), (1, 2)], ids=["tp2-pp2", "dp2-pp2"])
def test_is_global_zero_token_step_real_process_groups_agree(tmp_path, tp, pp):
    """Real Gloo groups: with PP combined with TP/DP, pipeline-local stage
    indices differ from global ranks, so the broadcast source must be the
    pipeline group's global last rank."""
    import torch.multiprocessing as mp

    world_size = 4
    init_method = f"file://{tmp_path / 'gloo-init'}"
    mp.spawn(
        _zero_token_process_group_worker,
        args=(world_size, init_method, tp, pp),
        nprocs=world_size,
        join=True,
    )


@pytest.fixture()
def model_module(monkeypatch):
    from relax.backends.megatron import model as model_module

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    return model_module


def _two_microbatch_losses(zero: bool) -> list[dict[str, object]]:
    """Two microbatches with explicit CP-local effective token counts."""
    return (
        [
            {"values": torch.tensor([0.0, 1.0, 2.0]), "num_tokens": torch.tensor(0.0)},
            {"values": torch.tensor([0.0, 3.0]), "num_tokens": torch.tensor(0.0)},
        ]
        if zero
        else [
            {"values": torch.tensor([4.0, 1.0, 2.0]), "num_tokens": torch.tensor(4.0)},
            {"values": torch.tensor([6.0, 3.0]), "num_tokens": torch.tensor(6.0)},
        ]
    )


def _patch_mpu(
    monkeypatch,
    model_module,
    *,
    pp_size: int,
    is_last_stage: bool,
    all_reduce_impl,
    broadcast_impl,
    pipeline_last_rank: int = 0,
):
    mpu = model_module.mpu
    dist = model_module.torch.distributed
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda ignore_virtual=False: is_last_stage)
    monkeypatch.setattr(mpu, "get_data_parallel_group", lambda with_context_parallel=False: object())
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_world_size", lambda: pp_size)
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_group", lambda: object())
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_last_rank", lambda: pipeline_last_rank)
    monkeypatch.setattr(dist, "all_reduce", all_reduce_impl)
    monkeypatch.setattr(dist, "broadcast", broadcast_impl)


def test_is_global_zero_token_step_true_pp1_no_broadcast(model_module, monkeypatch):
    """PP=1: every rank is the last stage, so the all-reduced count is already
    consistent locally and no pipeline broadcast is required."""
    calls = {"all_reduce": 0, "broadcast": 0}

    def all_reduce(tensor, group=None):
        calls["all_reduce"] += 1
        # Zero token count: leave the tensor as-is.

    def broadcast(tensor, src=0, group=None):
        calls["broadcast"] += 1

    _patch_mpu(
        monkeypatch,
        model_module,
        pp_size=1,
        is_last_stage=True,
        all_reduce_impl=all_reduce,
        broadcast_impl=broadcast,
    )
    assert model_module._is_global_zero_token_step(_two_microbatch_losses(zero=True)) is True
    assert calls == {"all_reduce": 1, "broadcast": 0}


def test_is_global_zero_token_step_false_pp1_no_broadcast(model_module, monkeypatch):
    """PP=1 with a nonzero token count: the local all-reduced count is
    nonzero."""
    calls = {"all_reduce": 0, "broadcast": 0}

    def all_reduce(tensor, group=None):
        calls["all_reduce"] += 1
        tensor.fill_(10)  # nonzero global token count

    def broadcast(tensor, src=0, group=None):
        calls["broadcast"] += 1

    _patch_mpu(
        monkeypatch,
        model_module,
        pp_size=1,
        is_last_stage=True,
        all_reduce_impl=all_reduce,
        broadcast_impl=broadcast,
    )
    assert model_module._is_global_zero_token_step(_two_microbatch_losses(zero=False)) is False
    assert calls == {"all_reduce": 1, "broadcast": 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to observe host synchronizations")
@pytest.mark.parametrize("pp_size", [1, 2])
def test_is_global_zero_token_step_single_host_sync(monkeypatch, pp_size):
    """The zero-token decision runs every step, so it may only materialize the
    final decision on the host once."""
    import warnings

    from relax.backends.megatron import model as model_module

    _patch_mpu(
        monkeypatch,
        model_module,
        pp_size=pp_size,
        is_last_stage=True,
        all_reduce_impl=lambda tensor, group=None: None,
        broadcast_impl=lambda tensor, src=0, group=None: None,
    )
    losses = [{"num_tokens": torch.tensor(n, dtype=torch.int, device="cuda")} for n in (0, 0)]
    torch.cuda.synchronize()
    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert model_module._is_global_zero_token_step(losses) is True
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    sync_warnings = [w for w in caught if "synchronizing CUDA operation" in str(w.message)]
    assert len(sync_warnings) == 1


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        custom_megatron_before_train_step_hook_path=None,
        ci_test=False,
        enable_mtp_training=False,
        check_for_nan_in_loss_and_grad=True,
        global_batch_size=32,
        dynamic_context_parallel=False,
        use_dynamic_batch_size=False,
        fully_async=False,
        seq_length=512,
        micro_batch_size=1,
        decoder_seq_length=512,
        attention_backend="flash",
    )


@pytest.mark.parametrize(
    ("zero_tokens", "calculate_per_token_loss", "reward_model", "is_last_stage"),
    [
        pytest.param(False, False, False, True, id="sample-mean"),
        pytest.param(True, False, False, True, id="sample-mean-zero-tokens"),
        pytest.param(False, True, False, True, id="token-mean"),
        pytest.param(True, True, False, True, id="token-mean-zero-tokens"),
        pytest.param(False, False, True, True, id="reward-model-pair-mean"),
        pytest.param(True, False, True, True, id="reward-model-pair-mean-zero-tokens"),
        pytest.param(True, False, False, False, id="non-last-stage-zero-tokens"),
    ],
)
def test_train_one_step_preserves_logical_batch_and_capture(
    model_module, monkeypatch, zero_tokens, calculate_per_token_loss, reward_model, is_last_stage
):
    args = _make_args()
    args.calculate_per_token_loss = calculate_per_token_loss
    args.task_type = "causal_lm"
    args.loss_type = "rm" if reward_model else "policy_loss"
    calls = {"optimizer": 0, "scheduler": [], "capture_begin": 0, "capture_end": 0, "critic_update_successful": []}
    token_count = 0.0 if zero_tokens else 4.0
    numerator = 0.0 if zero_tokens else 12.0
    keys = ["loss"]
    metric_values = [numerator]
    if reward_model:
        keys += [
            "rm/score_chosen_mean",
            "rm/score_rejected_mean",
            "rm/_score_chosen_second_moment",
            "rm/_score_rejected_second_moment",
        ]
        # RM metrics are pair sums; logical GBS=2, distinct from the token count=4.
        pair_count = 0.0 if zero_tokens else 2.0
        metric_values += [pair_count * value for value in (3.0, 1.5, 10.0, 2.5)]
    losses = [
        {
            "keys": keys,
            "values": torch.tensor([token_count if calculate_per_token_loss else 0.0, *metric_values]),
            "num_tokens": torch.tensor(token_count),
        }
    ]
    optimizer = SimpleNamespace(
        step=lambda: calls.__setitem__("optimizer", calls["optimizer"] + 1) or (True, 1.0, 0),
        zero_grad=lambda: None,
        param_groups=[],
    )
    scheduler = SimpleNamespace(step=lambda increment: calls["scheduler"].append(increment))
    monkeypatch.setattr(model_module, "get_args", lambda: args)
    monkeypatch.setattr(
        model_module, "get_forward_backward_func", lambda: lambda **_kwargs: losses if is_last_stage else []
    )
    monkeypatch.setattr(
        model_module,
        "maybe_verify_critic_value_head_movement",
        lambda model, optimizer, update_successful: calls["critic_update_successful"].append(update_successful),
    )
    monkeypatch.setattr(model_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)

    def broadcast(tensor, src, group=None):
        if is_last_stage:
            pytest.fail("PP=1 must not broadcast")
        # Supply the last pipeline stage's decision to the real skip helper.
        tensor.fill_(int(zero_tokens))

    _patch_mpu(
        monkeypatch,
        model_module,
        pp_size=1 if is_last_stage else 2,
        is_last_stage=is_last_stage,
        all_reduce_impl=lambda tensor, group=None, op=None: None,
        broadcast_impl=broadcast,
    )
    monkeypatch.setattr(model_module.torch.distributed, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(
        model_module.capture_hooks,
        "begin_step_for",
        lambda *a: calls.__setitem__("capture_begin", calls["capture_begin"] + 1),
    )
    monkeypatch.setattr(
        model_module.capture_hooks,
        "end_step_for",
        lambda: calls.__setitem__("capture_end", calls["capture_end"] + 1),
    )

    metrics, grad_norm = model_module.train_one_step(
        args=args,
        rollout_id=0,
        step_id=3,
        data_iterator=[[]],
        model=[SimpleNamespace(zero_grad_buffer=lambda: None)],
        optimizer=optimizer,
        opt_param_scheduler=scheduler,
        num_microbatches=1,
        step_global_batch_size=2,
    )

    assert calls["optimizer"] == (0 if zero_tokens else 1)
    assert calls["scheduler"] == ([] if zero_tokens else [2])
    assert calls["capture_begin"] == calls["capture_end"] == 1
    # A skipped zero-token step is not a successful update for critic movement checks.
    assert calls["critic_update_successful"] == [not zero_tokens]
    expected_metrics = {
        "loss": 0.0 if zero_tokens else (3.0 if calculate_per_token_loss else 6.0),
        "num_microbatches_mean": 0.0,
        "pack_tokens_mean": 0.0,
        "pack_tokens_max": 0.0,
    }
    if reward_model:
        expected_metrics.update(
            {
                "rm/score_chosen_mean": 0.0 if zero_tokens else 3.0,
                "rm/score_rejected_mean": 0.0 if zero_tokens else 1.5,
                "rm/score_chosen_std": 0.0 if zero_tokens else 1.0,
                "rm/score_rejected_std": 0.0 if zero_tokens else 0.5,
            }
        )
    assert metrics == (expected_metrics if is_last_stage else {})
    assert grad_norm == (0.0 if zero_tokens else 1.0)


def _make_opd_policy_args(clip: str, per_token: bool, use_tis: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        loss_type="policy_loss",
        advantage_estimator="grpo",
        calculate_per_token_loss=per_token,
        qkv_format="thd",
        recompute_loss_function=False,
        allgather_cp=False,
        global_batch_size=2,
        true_on_policy_mode=False,
        use_rollout_logprobs=False,
        use_opsm=False,
        get_mismatch_metrics=False,
        use_tis=use_tis,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=False,
        eps_clip=0.2,
        eps_clip_high=0.28,
        opd_loss_coef=1.0,
        opd_kl_type="reverse_kl",
        opd_jsd_alpha=0.5,
        opd_token_selection="student_sampled",
        opd_log_prob_top_k=0,
        opd_per_token_clip=0.5 if clip == "per_token" else None,
        opd_is_clip=1.0 if clip == "is" else None,
    )


@pytest.mark.parametrize("clip", ["per_token", "is"])
@pytest.mark.parametrize("per_token", [True, False], ids=["token-mean", "sample-mean"])
@pytest.mark.parametrize(
    ("response_lengths", "mask_values", "token_fraction", "sample_fraction", "use_tis"),
    [
        pytest.param([2, 4], [[1, 1], [1, 1, 1, 1]], 0.5, 0.625, False, id="valid"),
        pytest.param([2, 4], [[1, 0], [1, 1, 0, 0]], 2 / 3, 0.75, False, id="partial-mask"),
        pytest.param([2, 4], [[0, 0], [0, 0, 0, 0]], 0.0, 0.0, False, id="fully-masked"),
        pytest.param([2, 0], [[1, 1], []], 1.0, 0.5, False, id="mixed-empty"),
        pytest.param([0, 0], [[], []], 0.0, 0.0, False, id="all-empty"),
        # TIS adds distinct coverage only when the original masks contain valid tokens.
        pytest.param([2, 4], [[1, 1], [1, 1, 1, 1]], 0.5, 0.625, True, id="tis-valid"),
        pytest.param([2, 4], [[1, 0], [1, 1, 0, 0]], 2 / 3, 0.75, True, id="tis-partial-mask"),
        pytest.param([2, 0], [[1, 1], []], 1.0, 0.5, True, id="tis-mixed-empty"),
    ],
)
def test_opd_clip_metrics_use_valid_tokens_and_original_masks(
    monkeypatch: pytest.MonkeyPatch,
    clip: str,
    per_token: bool,
    use_tis: bool,
    response_lengths: list[int],
    mask_values: list[list[int]],
    token_fraction: float,
    sample_fraction: float,
) -> None:
    from relax.backends.megatron import loss as loss_module

    args = _make_opd_policy_args(clip, per_token, use_tis)
    masks = [torch.tensor(mask, dtype=torch.float32) for mask in mask_values]
    n = sum(response_lengths)
    log_probs = torch.full((n,), -0.5, requires_grad=True)
    gaps = torch.tensor([2.0, 2.0, -2.0, 2.0, -2.0, -2.0][:n])
    old_log_probs = (log_probs.detach() - gaps).split(response_lengths)
    batch = {
        "total_lengths": [length + 3 for length in response_lengths],
        "response_lengths": response_lengths,
        "unconcat_tokens": [torch.arange(length + 3) for length in response_lengths],
        "loss_masks": masks,
        "log_probs": old_log_probs,
        "rollout_log_probs": old_log_probs,
        "teacher_log_probs": old_log_probs,
        "advantages": torch.ones(n),
        "dynamic_cp_size": 1,
        "dynamic_cp_rank": 0,
    }
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_world_size", lambda **_kwargs: 1)
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (
            None,
            {
                "log_probs": list(log_probs.split(response_lengths)),
                "entropy": list(torch.zeros_like(log_probs).split(response_lengths)),
            },
        ),
    )
    if use_tis:
        monkeypatch.setattr(
            loss_module,
            "vanilla_tis_function",
            lambda **kwargs: (kwargs["pg_loss"], [torch.zeros_like(mask) for mask in masks], {}),
        )

    loss, _, log = loss_module.loss_function(args, batch, 1, torch.zeros(1, 1, 1, requires_grad=True))
    values = log["values"].tolist()
    denominator = values[0] if per_token else args.global_batch_size
    metrics = loss_module.normalize_reduced_loss_metrics(log["keys"], [denominator, *values[1:]])
    expected = token_fraction if per_token else sample_fraction
    assert metrics[f"opd_{clip}_clip_frac"] == pytest.approx(expected)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(log_probs.grad).all()
    if not any(any(mask) for mask in mask_values):
        assert all(value == 0 for value in metrics.values())
        assert torch.count_nonzero(log_probs.grad) == 0


@pytest.mark.parametrize("clip", ["per_token", "is"])
@pytest.mark.parametrize("per_token", [True, False])
def test_opd_clip_metrics_cp_rank_without_valid_tokens_contributes_zero(
    monkeypatch: pytest.MonkeyPatch,
    clip: str,
    per_token: bool,
) -> None:
    from relax.backends.megatron.cp_utils import get_sum_of_sample_mean
    from relax.utils.opd import opd_utils

    # Isolate diagnostics from OPD loss's existing full-response reduction.
    monkeypatch.setattr(opd_utils, "reduce_opd_loss", lambda batch, values: values.sum() * 0)
    args = _make_opd_policy_args(clip, per_token)
    masks = [torch.tensor([1.0, 1.0, 1.0, 0.0])]
    numerators = []
    # Zigzag CP=2: rank 0 owns response token 3; rank 1 owns tokens 0, 1, 2.
    for rank, flags in enumerate(([1.0], [1.0, 0.0, 1.0])):
        log_probs = torch.full((len(flags),), -0.5)
        teacher_log_probs = log_probs - torch.tensor(flags) * 2
        reducer = get_sum_of_sample_mean(
            [8],
            [4],
            masks,
            per_token,
            dynamic_cp_size=2,
            dynamic_cp_rank=rank,
        )
        _, metrics = opd_utils.compute_policy_opd_loss(
            args=args,
            batch={"response_lengths": [4], "loss_masks": masks, "teacher_log_probs": [teacher_log_probs]},
            metric_reducer=reducer,
            log_probs=log_probs,
            old_log_probs=teacher_log_probs,
            log_probs_and_entropy={},
        )
        numerators.append(metrics[f"opd_{clip}_clip_frac"])
    assert numerators[0] == 0
    denominator = 3 if per_token else 1
    assert sum(numerators) / denominator == pytest.approx(2 / 3)
