# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for fused log-probability and entropy wiring."""

from argparse import Namespace
from unittest.mock import Mock

import pytest
import torch


try:
    from relax.backends.megatron import loss as loss_module
except Exception as exc:
    pytest.skip(f"relax.backends.megatron unavailable: {exc}", allow_module_level=True)


@pytest.mark.parametrize(
    "entropy_coef,expected_entropy_grad",
    [
        pytest.param(0.0, False, id="metric_only"),
        pytest.param(0.01, True, id="entropy_loss"),
    ],
)
def test_get_log_probs_and_entropy_wires_entropy_grad(
    monkeypatch: pytest.MonkeyPatch,
    entropy_coef: float,
    expected_entropy_grad: bool,
) -> None:
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(loss_module.mpu, "get_tensor_model_parallel_group", lambda: None)

    args = Namespace(
        allgather_cp=False,
        entropy_coef=entropy_coef,
        log_probs_chunk_size=-1,
        loss_type="grpo",
        advantage_estimator=None,
        opd_log_prob_top_k=0,
        qkv_format="thd",
        rollout_temperature=1.0,
    )
    logits = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 0.5, 2.0], [-1.0, 3.0, 2.0, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    tokens = torch.tensor([2, 3, 1], dtype=torch.long)

    _, result = loss_module.get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=[tokens],
        total_lengths=[3],
        response_lengths=[2],
        with_entropy=True,
    )

    log_probs = result["log_probs"][0]
    entropy = result["entropy"][0]
    assert entropy.requires_grad is expected_entropy_grad

    objective = log_probs.sum()
    if expected_entropy_grad:
        objective = objective + entropy_coef * entropy.sum()
    objective.backward()
    assert logits.grad is not None


@pytest.mark.parametrize(
    ("coef", "reentrant", "replay_entropy"), [(0.0, True, False), (0.1, True, True), (0.0, False, True)]
)
def test_policy_checkpoint_entropy_matches_eager(monkeypatch, coef, reentrant, replay_entropy):
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(loss_module.mpu, "get_tensor_model_parallel_group", lambda: None)
    kernel = Mock(wraps=loss_module.calculate_log_probs_and_entropy)
    monkeypatch.setattr(loss_module, "calculate_log_probs_and_entropy", kernel)
    args = Namespace(
        loss_type="policy_loss",
        task_type="causal_lm",
        advantage_estimator="grpo",
        true_on_policy_mode=True,
        use_opsm=False,
        use_tis=False,
        get_mismatch_metrics=False,
        use_kl_loss=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        entropy_coef=coef,
        calculate_per_token_loss=True,
        global_batch_size=1,
        qkv_format="thd",
        allgather_cp=False,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        opd_token_selection="student_sampled",
        recompute_loss_function_use_reentrant=reentrant,
    )
    batch = {
        "advantages": torch.tensor([1.0, -0.5]),
        "unconcat_tokens": [torch.tensor([2, 1, 0])],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1.0, 0.0])],
    }
    inputs = torch.tensor([[[1.0, -2.0, 0.5], [2.0, 0.0, -1.0], [0.0, 1.0, 2.0]]])

    def run(recompute):
        args.recompute_loss_function = recompute
        logits = inputs.clone().requires_grad_()
        loss, _, metrics = loss_module.loss_function(args, batch, 1, logits)
        loss.backward()
        entropy = metrics["values"][1 + metrics["keys"].index("entropy_loss")]
        return loss.detach(), entropy, logits.grad

    ref_logits = inputs.clone().requires_grad_()
    # Only the first response token (target 1) contributes, with unit advantage and on-policy ratio 1.
    log_probs = ref_logits[0, 0].log_softmax(dim=-1)
    ref_entropy = -(log_probs.exp() * log_probs).sum()
    ratio = (log_probs[1] - log_probs[1].detach()).exp()
    ref_loss = -ratio - coef * ref_entropy
    (ref_grad,) = torch.autograd.grad(ref_loss, ref_logits)

    for loss, entropy, grad in (run(False), run(True)):
        torch.testing.assert_close(grad, ref_grad)
        torch.testing.assert_close(loss, ref_loss.detach())
        torch.testing.assert_close(entropy, ref_entropy.detach())
    assert [call.kwargs["with_entropy"] for call in kernel.call_args_list] == [True, True, replay_entropy]


def test_sft_chunked_lm_head_matches_log_softmax_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(loss_module.mpu, "get_tensor_model_parallel_group", lambda: None)

    torch.manual_seed(0)
    hidden = torch.randn(1, 4, 3, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(5, 3, dtype=torch.float32, requires_grad=True)
    tokens = torch.tensor([2, 4, 1, 3], dtype=torch.long)
    loss_weights = torch.tensor([0.25, -0.5, 1.5, 0.75], dtype=torch.float32)
    temperature = 0.7
    args = Namespace(
        allgather_cp=False,
        entropy_coef=0.0,
        loss_type="sft",
        advantage_estimator=None,
        opd_log_prob_top_k=0,
        qkv_format="thd",
        rollout_temperature=temperature,
        sft_chunked_logits=True,
        sft_logits_chunk_size=2,
    )

    def lm_head_forward(hidden_sub: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(hidden_sub, weight), None

    _, result = loss_module.get_log_probs_and_entropy(
        hidden,
        args=args,
        unconcat_tokens=[tokens],
        total_lengths=[4],
        response_lengths=[4],
        lm_head_forward=lm_head_forward,
    )
    actual_log_probs = result["log_probs"][0]
    (actual_log_probs * loss_weights).sum().backward()

    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    reference_logits = torch.nn.functional.linear(reference_hidden.squeeze(0), reference_weight) / temperature
    shifted_tokens = torch.cat([tokens[1:], tokens.new_zeros(1)])
    expected_log_probs = (
        torch.log_softmax(reference_logits, dim=-1).gather(-1, shifted_tokens.unsqueeze(-1)).squeeze(-1)
    )
    (expected_log_probs * loss_weights).sum().backward()

    torch.testing.assert_close(actual_log_probs, expected_log_probs, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(weight.grad, reference_weight.grad, atol=1e-6, rtol=1e-6)


def test_sft_chunked_allgather_cp_padding_rank_keeps_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 4)
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_rank", lambda: 3)
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_group", lambda: object())
    monkeypatch.setattr(loss_module.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(loss_module.dist.nn, "all_reduce", lambda value, group: value)

    # Five real tokens padded to eight leave contiguous CP rank 3 with only padding.
    hidden = torch.ones(1, 2, 3, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(5, 3, dtype=torch.bfloat16, requires_grad=True)
    tokens = torch.tensor([2, 4, 1, 3, 2], dtype=torch.long)
    args = Namespace(
        allgather_cp=True,
        entropy_coef=0.0,
        loss_type="sft",
        advantage_estimator=None,
        opd_log_prob_top_k=0,
        qkv_format="thd",
        rollout_temperature=1.0,
        sft_chunked_logits=True,
        sft_logits_chunk_size=2,
        recompute_loss_function=True,
    )

    def lm_head_forward(hidden_sub: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(hidden_sub, weight), None

    def get_log_probs() -> torch.Tensor:
        _, result = loss_module.get_log_probs_and_entropy(
            hidden,
            args=args,
            unconcat_tokens=[tokens],
            total_lengths=[5],
            response_lengths=[5],
            lm_head_forward=lm_head_forward,
        )
        return result["log_probs"][0]

    log_probs = get_log_probs()
    assert log_probs.shape == (1,)
    assert log_probs.dtype == torch.float32
    assert log_probs.requires_grad
    torch.testing.assert_close(log_probs, torch.zeros_like(log_probs))
    log_probs.sum().backward()
    assert hidden.grad is not None
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))

    with torch.no_grad():
        assert not get_log_probs().requires_grad


@pytest.mark.parametrize("entropy_grad", [False, True])
def test_empty_response_keeps_logit_graph_and_entropy_grad_policy(entropy_grad):
    from relax.utils.training.ppo_utils import calculate_log_probs_and_entropy

    logits = torch.empty(0, 5, requires_grad=True)
    tokens = torch.empty(0, dtype=torch.long)
    log_probs, entropy = calculate_log_probs_and_entropy(
        logits, tokens, None, with_entropy=True, with_entropy_grad=entropy_grad
    )
    assert log_probs.requires_grad
    assert entropy.requires_grad is entropy_grad
    (log_probs.sum() + entropy.sum()).backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    with torch.no_grad():
        log_probs, entropy = calculate_log_probs_and_entropy(
            logits, tokens, None, with_entropy=True, with_entropy_grad=entropy_grad
        )
    assert not log_probs.requires_grad
    assert not entropy.requires_grad


@pytest.mark.parametrize("trailing_shape", [(), (3,)])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_allgather_cp_empty_response_keeps_graph_and_topk_shape(monkeypatch, trailing_shape, requires_grad):
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_group", lambda: object())
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(loss_module.dist.nn, "all_reduce", lambda value, group: value)
    monkeypatch.setattr(loss_module, "slice_log_prob_with_cp", lambda value, *args, **kwargs: value)

    # Rank 0 holds tokens [0, 2), while response prediction starts at token 5.
    # Empty values must remain connected to the originating logits even when
    # the route carries trailing score-centering top-k dimensions.
    origin = torch.empty((0, *trailing_shape, 5), requires_grad=requires_grad)
    values = {"log_probs": [origin.sum(dim=-1)]}
    loss_module._allgather_cp_redistribute(
        values,
        logits=torch.empty(1, 2, 5),
        args=Namespace(qkv_format="thd"),
        total_lengths=[10],
        response_lengths=[4],
    )
    result = values["log_probs"][0]
    assert result.shape == (4, *trailing_shape)
    assert result.requires_grad is requires_grad
    assert torch.count_nonzero(result) == 0
    if requires_grad:
        result.sum().backward()
        assert origin.grad is not None
        assert origin.grad.shape == origin.shape
