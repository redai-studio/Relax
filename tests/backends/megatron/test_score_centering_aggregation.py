# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SC accounting through the actual Megatron outer loss and token reducer."""

import pytest
import torch

from tests.backends.megatron.test_score_centering_loss import _args, _backend


def _samples_and_logits(temperature):
    lengths = [2, 5, 1, 0]
    masks = [torch.ones(2), torch.tensor([1, 0, 1, 1, 1]), torch.zeros(1), torch.empty(0)]
    logits = torch.randn(1, sum(length + 2 for length in lengths), 5, generator=torch.Generator().manual_seed(18))
    logits[..., 3:] = 30  # Keep padding dominant to exercise the real scoring path.
    q = torch.tensor([0.1, 0.2, 0.7])
    samples = []
    offset = 0
    for i, (length, mask) in enumerate(zip(lengths, masks, strict=True)):
        response = (torch.arange(length) + i) % 3
        ids = torch.tensor([2, 1]).expand(length, 2).clone()
        sample_q = q[response].log().masked_fill(mask == 0, -torch.inf)
        head_q = q[ids].log().masked_fill(mask[:, None] == 0, -torch.inf)
        samples.append(
            {
                "tokens": torch.cat((torch.tensor([1, 1]), response)),
                "response": response,
                "mask": mask,
                "advantage": torch.full((length,), (1.25, -0.75, 2.0, 0.0)[i]),
                "sample_q": sample_q,
                "head_q": head_q,
                "head_ids": ids.masked_fill(mask[:, None] == 0, -1),
                "positions": torch.arange(offset, offset + length + 2),
            }
        )
        offset += length + 2
    # The first trained token has p(a)/q(a)=6: MIS rejects its sampled term,
    # but its nonzero head correction and its place in the denominator remain.
    logits[0, 1, :3] = torch.tensor([0.6, 0.3, 0.1]).log() * temperature
    return samples, logits


def _batch(samples, indices):
    selected = [samples[i] for i in indices]
    return {
        "advantages": [sample["advantage"] for sample in selected],
        "log_probs": [sample["sample_q"] for sample in selected],
        "rollout_log_probs": [sample["sample_q"] for sample in selected],
        "rollout_topk_token_ids": [sample["head_ids"] for sample in selected],
        "rollout_topk_log_probs": [sample["head_q"] for sample in selected],
        "unconcat_tokens": [sample["tokens"] for sample in selected],
        "total_lengths": [sample["tokens"].numel() for sample in selected],
        "response_lengths": [sample["response"].numel() for sample in selected],
        "loss_masks": [sample["mask"] for sample in selected],
        "dynamic_cp_size": 1,
        "dynamic_cp_rank": 0,
    }


def _run_layout(loss_module, args, samples, base, layout):
    """Sum the outer loss's DP/microbatch outputs like the token finalizer.

    DP partitions are replayed locally; this checks the arithmetic contract,
    not DDP transport. None marks a padding microbatch containing real data.
    """
    logits = base.clone().requires_grad_()
    losses, counts, logs = [], [], []
    for microbatches in layout:
        for indices in microbatches:
            is_dummy = indices is None
            indices = [0] if is_dummy else indices
            batch = _batch(samples, indices)
            batch["__is_dummy__"] = is_dummy
            positions = torch.cat([samples[i]["positions"] for i in indices])
            loss, count, log = loss_module.loss_function(
                args, batch, len(microbatches), logits.index_select(1, positions)
            )
            if is_dummy:
                assert loss == 0 and count == 0
                assert torch.count_nonzero(log["values"]) == 0
            losses.append(loss)
            counts.append(count)
            logs.append(log)
    total_count = sum(counts)
    final_loss = sum(losses) / total_count.clamp_min(1)
    final_loss.backward()
    assert all(log["keys"] == logs[0]["keys"] for log in logs)
    log_values = torch.stack([log["values"] for log in logs]).sum(0)
    metrics = loss_module.normalize_reduced_loss_metrics(logs[0]["keys"], log_values.tolist())
    return final_loss.detach(), logits.grad, total_count, metrics


def _dense_gradient_and_weight_metrics(args, samples, base):
    """Full-vocabulary tail reconstruction, independent of the SC head
    kernel."""
    logits = base.clone().requires_grad_()
    q = torch.tensor([0.1, 0.2, 0.7])
    total, weight_sum, zero_count, count = 0, 0, 0, 0
    for sample in samples:
        positions = sample["positions"][1:-1]
        logp = (logits[0, positions, :3] / args.rollout_temperature).log_softmax(-1)
        p = logp.detach().exp()
        q_hat = p * ((q[0] / p[:, 0])[:, None])
        q_hat[:, 1:] = q[1:]
        ratios = p / q_hat
        sample_ratios = p.gather(-1, sample["response"][:, None]).squeeze(-1) / q[sample["response"]]
        if args.score_centering_is == "mis":
            weights = torch.where((ratios >= 0.5) & (ratios <= 5), ratios, 0)
            sample_weights = torch.where((sample_ratios >= 0.5) & (sample_ratios <= 5), sample_ratios, 0)
        else:
            weights, sample_weights = torch.ones_like(ratios), torch.ones_like(sample_ratios)
        sample_logp = logp.gather(-1, sample["response"][:, None]).squeeze(-1)
        token_loss = -sample["advantage"] * (sample_weights * sample_logp - (q_hat * weights * logp).sum(-1))
        # Entropy itself needs its own full derivative, unlike the detached SC
        # coefficients used above.
        entropy = -(logp.exp() * logp).sum(-1)
        total = total + ((token_loss - args.entropy_coef * entropy) * sample["mask"]).sum()
        weight_sum = weight_sum + (sample_weights * sample["mask"]).sum()
        zero_count = zero_count + ((sample_weights == 0) * sample["mask"]).sum()
        count = count + sample["mask"].sum()
    (total / count).backward()
    return logits.grad, weight_sum / count, zero_count / count


@pytest.mark.parametrize("mode", ["none", "mis"])
def test_score_centering_outer_loss_is_invariant_to_microbatch_and_dp_partition(monkeypatch, mode):
    loss_module, _ = _backend(monkeypatch)
    monkeypatch.setattr(loss_module, "compute_policy_opd_loss", lambda **kwargs: (None, {}))
    args = _args(
        score_centering_vocab_size=3,
        score_centering_is=mode,
        calculate_per_token_loss=True,
        recompute_loss_function=False,
        global_batch_size=4,
    )
    samples, base = _samples_and_logits(args.rollout_temperature)
    original_masks = [sample["mask"].clone() for sample in samples]
    reference = _run_layout(loss_module, args, samples, base, [[[0, 1, 2, 3]]])
    oracle_gradient, expected_weight, expected_zero_fraction = _dense_gradient_and_weight_metrics(args, samples, base)
    torch.testing.assert_close(reference[1], oracle_gradient, atol=1e-6, rtol=1e-5)
    assert reference[2] == 6
    assert reference[3]["sc_importance_weight"] == pytest.approx(expected_weight.item())
    assert reference[3]["sc_zero_weight_fraction"] == pytest.approx(expected_zero_fraction.item())
    if mode == "mis":
        assert expected_zero_fraction > 0
        assert torch.count_nonzero(reference[1][0, 1, :3]) > 0

    for layout in ([[[0], [1, 2], [3]]], [[[0, 2], [3]], [[1], None]]):
        actual = _run_layout(loss_module, args, samples, base, layout)
        torch.testing.assert_close(actual[0], reference[0])
        torch.testing.assert_close(actual[1], reference[1], atol=1e-6, rtol=1e-5)
        assert actual[2] == reference[2]
        assert actual[3] == pytest.approx(reference[3], abs=1e-6, rel=1e-6)
    for sample, original in zip(samples, original_masks, strict=True):
        torch.testing.assert_close(sample["mask"], original)


def test_score_centering_outer_empty_masked_and_dummy_microbatches_have_no_signal(monkeypatch):
    loss_module, _ = _backend(monkeypatch)
    monkeypatch.setattr(loss_module, "compute_policy_opd_loss", lambda **kwargs: (None, {}))
    args = _args(
        score_centering_vocab_size=3,
        score_centering_is="mis",
        calculate_per_token_loss=True,
        recompute_loss_function=False,
        global_batch_size=2,
    )
    samples, base = _samples_and_logits(args.rollout_temperature)
    loss, gradient, count, metrics = _run_layout(loss_module, args, samples, base, [[[2], [3]], [None]])
    assert loss == 0 and count == 0
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) == 0
    assert all(value == 0 for value in metrics.values())
