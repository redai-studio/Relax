# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU equivalence checks; CP shards are simulated without distributed
collectives."""

import importlib
import sys
from argparse import ArgumentParser, Namespace
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from relax.backends.megatron import cp_utils
from tests.cpu.utils.test_arguments_opd_teacher_colocate import _opd_args
from tests.cpu.utils.test_arguments_opd_teacher_colocate import arguments_module as arguments_module
from tests.cpu.utils.test_arguments_score_centering import _sc_args


@pytest.fixture()
def cpu_loss(monkeypatch):
    # Match the existing policy-loss tests: Megatron itself is optional in CPU CI.
    try:
        importlib.import_module("megatron.core")
    except ModuleNotFoundError as exc:
        if exc.name not in {"megatron", "megatron.core"}:
            raise
        megatron, core = ModuleType("megatron"), ModuleType("megatron.core")
        core.mpu = ModuleType("megatron.core.mpu")
        megatron.core = core
        for name, module in (("megatron", megatron), ("megatron.core", core), ("megatron.core.mpu", core.mpu)):
            monkeypatch.setitem(sys.modules, name, module)
    loss_module = importlib.import_module("relax.backends.megatron.loss")
    mpu = SimpleNamespace(
        get_tensor_model_parallel_group=lambda: None,
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
    )
    monkeypatch.setattr(loss_module, "mpu", mpu)
    monkeypatch.setattr(cp_utils, "mpu", mpu)

    # Keep the production slicer, policy loss, reducers and checkpoint dispatch.
    # Only replace the CUDA/TP softmax kernel with its single-process reference.
    def calculate(logits, tokens, _group, *, with_entropy=False, with_entropy_grad=False, **_kwargs):
        log_probs = logits.log_softmax(dim=-1)
        selected = log_probs.gather(-1, tokens.unsqueeze(-1))
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1) if with_entropy else None
        if entropy is not None and not with_entropy_grad:
            entropy = entropy.detach()
        return selected, entropy

    monkeypatch.setattr(loss_module, "calculate_log_probs_and_entropy", calculate)
    return loss_module


def _args(*, response_only, entropy_coef=0.0, checkpoint_mode=None):
    return Namespace(
        response_only_lm_head=response_only,
        loss_type="policy_loss",
        advantage_estimator="grpo",
        qkv_format="thd",
        allgather_cp=False,
        rollout_temperature=0.7,
        entropy_coef=entropy_coef,
        log_probs_chunk_size=-1,
        calculate_per_token_loss=True,
        global_batch_size=4,
        recompute_loss_function=checkpoint_mode is not None,
        recompute_loss_function_use_reentrant=checkpoint_mode == "reentrant",
        true_on_policy_mode=False,
        use_rollout_logprobs=False,
        use_opsm=False,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        use_kl_loss=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        opd_token_selection="student_sampled",
        opd_loss_coef=0.0,
        opd_log_prob_top_k=0,
    )


def _local_hidden(hidden, lengths, padded_lengths, cp_size, cp_rank):
    if cp_size == 1:
        return hidden.unsqueeze(0)
    # Independent construction of Bridge's two equal zigzag shards per sample.
    shards = []
    for sample, padded_length in zip(hidden.split(lengths), padded_lengths, strict=True):
        padded = F.pad(sample, (0, 0, 0, padded_length - sample.size(0)))
        chunks = padded.reshape(2 * cp_size, -1, sample.size(-1))
        shards.extend((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]))
    return torch.cat(shards).unsqueeze(0)


@pytest.mark.parametrize(
    ("cp_size", "dtype", "entropy_coef", "checkpoint_mode"),
    [
        pytest.param(1, torch.float32, 0.0, None, id="cp1-fp32-metric"),
        pytest.param(1, torch.bfloat16, 0.03, "nonreentrant", id="cp1-bf16-entropy-checkpoint"),
        pytest.param(8, torch.float32, 0.0, None, id="cp8-fp32-metric"),
        pytest.param(8, torch.float32, 0.03, "reentrant", id="cp8-fp32-entropy-reentrant"),
        pytest.param(8, torch.bfloat16, 0.0, "reentrant", id="cp8-bf16-metric-reentrant"),
        pytest.param(8, torch.bfloat16, 0.03, "nonreentrant", id="cp8-bf16-entropy-nonreentrant"),
    ],
)
def test_policy_loss_and_accumulated_gradients_match_dense(cpu_loss, cp_size, dtype, entropy_coef, checkpoint_mode):
    """Two microbatches, padding, masked response tokens and empty CP ranks."""
    generator = torch.Generator().manual_seed(314)
    lengths, response_lengths, padded_lengths = [29, 47], [5, 17], [64, 64]
    tokens = [torch.randint(0, 19, (n,), generator=generator) for n in lengths]
    hidden_values = [torch.randn(sum(lengths), 7, generator=generator).to(dtype) for _ in range(2)]
    weight_value = (torch.randn(19, 7, generator=generator) * 0.15).to(dtype)
    masks = [torch.ones(n) for n in response_lengths]
    masks[1][::4] = 0
    advantages = [torch.linspace(-0.8, 1.2, n) for n in response_lengths]
    results = []

    for response_only in (False, True):
        args = _args(response_only=response_only, entropy_coef=entropy_coef, checkpoint_mode=checkpoint_mode)
        reference_args = _args(response_only=False, entropy_coef=entropy_coef)
        # Current Bridge emits per-document zigzag outputs even with allgather
        # enabled; response selection must honor the batch's explicit layout.
        args.allgather_cp = reference_args.allgather_cp = cp_size > 1
        weight = weight_value.clone().requires_grad_()
        hidden_states = [value.clone().requires_grad_() for value in hidden_values]
        losses, metrics, normalizers = [], [], []

        for hidden in hidden_states:
            for cp_rank in range(cp_size):
                local = _local_hidden(hidden, lengths, padded_lengths, cp_size, cp_rank)
                local.retain_grad()
                cp_kwargs = {"dynamic_cp_size": cp_size, "dynamic_cp_rank": cp_rank}
                inputs = {
                    "unconcat_tokens": tokens,
                    "total_lengths": lengths,
                    "response_lengths": response_lengths,
                    "padded_total_lengths": padded_lengths if cp_size > 1 else None,
                    "cp_output_is_zigzag": cp_size > 1,
                    **cp_kwargs,
                }
                with torch.no_grad():
                    _, reference = cpu_loss.get_log_probs_and_entropy(
                        F.linear(local, weight).float(), args=reference_args, **inputs
                    )
                batch = {
                    **inputs,
                    "loss_masks": masks,
                    "log_probs": [lp + 0.05 for lp in reference["log_probs"]],
                    "advantages": [
                        cp_utils.slice_log_prob_with_cp(
                            advantage,
                            length,
                            response_length,
                            padded_total_length=padded_length,
                            **cp_kwargs,
                        )
                        for advantage, length, response_length, padded_length in zip(
                            advantages, lengths, response_lengths, padded_lengths, strict=True
                        )
                    ],
                }
                projected_rows = []

                def head(values):
                    assert values.dtype == dtype
                    assert values.ndim == 3 and values.size(1) == 1
                    projected_rows.append(values.size(0))
                    return F.linear(values, weight), None

                loss, normalizer, log = cpu_loss.loss_function(
                    args,
                    batch,
                    num_microbatches=2,
                    logits=local if response_only else F.linear(local, weight).float(),
                    lm_head_forward=head if response_only else None,
                )
                loss.backward()
                assert local.grad is not None
                if normalizer.item() == 0:
                    assert loss.item() == 0
                    assert torch.count_nonzero(local.grad).item() == 0
                    assert not projected_rows
                elif response_only:
                    assert projected_rows
                    assert all(0 < n <= max(response_lengths) for n in projected_rows)
                losses.append(loss.detach())
                metrics.append(log["values"])
                normalizers.append(normalizer)

        assert sum(n.item() for n in normalizers) == 2 * sum(mask.sum().item() for mask in masks)
        results.append(
            (
                torch.stack(losses),
                torch.stack(metrics),
                weight.grad,
                [hidden.grad for hidden in hidden_states],
            )
        )

    dense, response = results
    tolerances = {"atol": 4e-3, "rtol": 2e-2} if dtype == torch.bfloat16 else {"atol": 2e-6, "rtol": 2e-5}
    for actual, expected in zip(response[:3], dense[:3], strict=True):
        torch.testing.assert_close(actual, expected, **tolerances)
    for actual, expected in zip(response[3], dense[3], strict=True):
        torch.testing.assert_close(actual, expected, **tolerances)


def test_frozen_backbone_checkpoint_keeps_head_gradients(cpu_loss):
    """A deferred head must still train when checkpoint inputs are frozen."""
    torch.manual_seed(42)
    hidden = torch.randn(1, 7, 3)
    weight = torch.randn(11, 3)
    batch = {
        "unconcat_tokens": [torch.arange(7)],
        "total_lengths": [7],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "log_probs": [torch.tensor([-2.0, -3.0])],
        "advantages": [torch.tensor([0.5, -0.4])],
    }
    outputs = []
    for response_only in (False, True):
        train_weight = weight.clone().requires_grad_()
        loss, _, _ = cpu_loss.loss_function(
            _args(response_only=response_only, entropy_coef=0.03, checkpoint_mode="reentrant"),
            batch,
            num_microbatches=1,
            logits=hidden if response_only else F.linear(hidden, train_weight),
            lm_head_forward=(lambda values: (F.linear(values, train_weight), None)) if response_only else None,
        )
        loss.backward()
        assert train_weight.grad is not None
        outputs.append((loss.detach(), train_weight.grad))
    for actual, expected in zip(outputs[1], outputs[0], strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("entropy_coef", [0.0, 0.03])
def test_response_projection_skips_empty_samples_and_preserves_entropy(cpu_loss, entropy_coef):
    args = _args(response_only=True, entropy_coef=entropy_coef)
    hidden = torch.arange(72, dtype=torch.float32).reshape(1, 24, 3).div(100).requires_grad_()
    weight = torch.linspace(-0.3, 0.5, 33).reshape(11, 3).requires_grad_()
    tokens = [torch.arange(n) % 11 for n in (7, 9, 8)]
    inputs = {
        "unconcat_tokens": tokens,
        "total_lengths": [7, 9, 8],
        "response_lengths": [3, 0, 2],
        "with_entropy": True,
    }
    calls = []

    def head(values):
        calls.append(values.detach().clone())
        return F.linear(values, weight), None

    _, actual = cpu_loss.get_log_probs_and_entropy(hidden, args=args, lm_head_forward=head, **inputs)
    _, expected = cpu_loss.get_log_probs_and_entropy(
        F.linear(hidden, weight).float(),
        args=_args(response_only=False, entropy_coef=entropy_coef),
        **inputs,
    )
    assert [x.size(0) for x in calls] == [3, 2]
    torch.testing.assert_close(calls[0].squeeze(1), hidden[0, 3:6])
    torch.testing.assert_close(calls[1].squeeze(1), hidden[0, 21:23])
    for key in ("log_probs", "entropy"):
        for actual_sample, expected_sample in zip(actual[key], expected[key], strict=True):
            assert actual_sample.dtype == torch.float32
            torch.testing.assert_close(actual_sample, expected_sample)
    assert actual["entropy"][0].requires_grad is (entropy_coef != 0)


@pytest.mark.parametrize("topk_argument", ["with_topk", "gather_topk_token_ids"])
def test_response_projection_rejects_topk_before_calling_head(cpu_loss, topk_argument):
    kwargs = {topk_argument: True if topk_argument == "with_topk" else [torch.zeros((2, 1), dtype=torch.long)]}

    def unexpected_head(_values):
        pytest.fail("Unsupported top-k must be rejected before projecting hidden states")

    with pytest.raises((AssertionError, ValueError), match="(?i)top.?k"):
        cpu_loss.get_log_probs_and_entropy(
            torch.zeros(1, 4, 3),
            args=_args(response_only=True),
            unconcat_tokens=[torch.tensor([0, 1, 2, 3])],
            total_lengths=[4],
            response_lengths=[2],
            lm_head_forward=unexpected_head,
            **kwargs,
        )


@pytest.mark.parametrize(("argv", "expected"), [([], False), (["--response-only-lm-head"], True)])
def test_response_only_option_is_opt_in(arguments_module, argv, expected):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)
    assert parser.parse_args(argv).response_only_lm_head is expected


def test_response_only_accepts_sampled_opd_policy(arguments_module):
    args = _opd_args()
    args.loss_type = "policy_loss"
    args.response_only_lm_head = True
    arguments_module.slime_validate_args(args)


def test_response_only_rejects_non_policy_loss(arguments_module):
    args = _opd_args()
    args.loss_type = "custom_loss"
    args.response_only_lm_head = True
    with pytest.raises(AssertionError, match="requires Megatron policy_loss"):
        arguments_module.slime_validate_args(args)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("train_backend", "fsdp", "requires Megatron policy_loss"),
        ("overlap_grad_reduce", True, "--overlap-grad-reduce"),
        ("defer_embedding_wgrad_compute", True, "--defer-embedding-wgrad-compute"),
        ("overlap_moe_expert_parallel_comm", True, "--overlap-moe-expert-parallel-comm"),
        ("enable_mtp_training", True, "--enable-mtp-training"),
        ("opd_log_prob_top_k", 8, "--opd-log-prob-top-k"),
        ("fp8", "hybrid", "--fp8"),
        ("fp4", "nvfp4", "--fp4"),
        ("advantage_estimator", "score_centering", "score_centering candidate scoring"),
    ],
)
def test_response_only_guards_apply_after_yaml_overrides(arguments_module, tmp_path, option, value, message):
    # Use an otherwise valid candidate-scoring configuration so OPD's separate
    # incompatibility does not mask the response-only guard under test.
    args = _sc_args(advantage_estimator="grpo") if value == "score_centering" else _opd_args()
    args.loss_type = "policy_loss"
    args.response_only_lm_head = True
    config = tmp_path / "override.yaml"
    config.write_text(f"{option}: {value}\n")
    args.custom_config_path = str(config)

    with pytest.raises(AssertionError, match=message):
        arguments_module.slime_validate_args(args)
