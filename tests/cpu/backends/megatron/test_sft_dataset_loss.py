# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Source metrics must survive masking, CP and accumulation without changing
gradients."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from relax.utils.training.dataset_metrics import dataset_loss_sums, finalize_dataset_loss_metrics
from tests.cpu.backends.megatron.test_score_centering_loss import _backend


def _args(**overrides):
    from tests.helpers.relax_args import make_args

    values = dict(
        loss_type="sft",
        advantage_estimator=None,
        mtp_only_training=False,
        task_type="causal_lm",
        qkv_format="thd",
        allgather_cp=False,
        calculate_per_token_loss=True,
        recompute_loss_function=False,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        sft_logits_chunk_size=3,
        sft_chunked_logits=False,
        global_batch_size=3,
        _sft_dataset_names=("general", "code/a", "unused"),
    )
    values.update(overrides)
    return make_args(**values)


def test_source_loss_uses_token_weights_across_microbatches_and_missing_sources():
    names = ("a/b", "a%2Fb", "empty")
    losses = torch.tensor([1.0, 90.0, 3.0, 5.0, 7.0, 11.0], requires_grad=True)
    masks = [torch.tensor([1, 0]), torch.ones(3), torch.ones(1)]
    full = dataset_loss_sums(names, [0, 0, 1], losses, masks, count_samples=True)
    first = dataset_loss_sums(names, [0], losses[:2], masks[:1], count_samples=True)
    second = dataset_loss_sums(names, [0, 1], losses[2:], masks[1:], count_samples=True)
    assert list(full) == list(first) == list(second)
    for key in full:
        assert not full[key].requires_grad
        torch.testing.assert_close(full[key], first[key] + second[key])
    metrics = finalize_dataset_loss_metrics({key: value.item() for key, value in full.items()})
    assert metrics["dataset/a%2Fb/loss"] == 4.0  # (1 + 3 + 5 + 7) / 4
    assert metrics["dataset/a%2Fb/tokens"] == 4
    assert metrics["dataset/a%2Fb/samples"] == 2
    assert metrics["dataset/a%2Fb/token_fraction"] == 0.8
    assert metrics["dataset/a%2Fb/sample_fraction"] == pytest.approx(2 / 3)
    assert metrics["dataset/a%252Fb/loss"] == 11
    assert metrics["dataset/empty/samples"] == 0
    assert "dataset/empty/loss" not in metrics


@pytest.mark.parametrize(
    ("path", "per_token"), [("ordinary", False), ("ordinary", True), ("recomputed", True), ("chunked", True)]
)
def test_source_metrics_preserve_training_loss_and_gradients(monkeypatch, path, per_token):
    loss_module, _ = _backend(monkeypatch)
    loss_module.mpu.get_data_parallel_world_size = lambda **_: 1
    monkeypatch.setenv("RELAX_SFT_FUSED_LINEAR_CE", "0")
    ordinary = path in {"ordinary", "recomputed"}
    args = _args(
        calculate_per_token_loss=per_token,
        sft_chunked_logits=not ordinary,
        recompute_loss_function=path == "recomputed",
    )
    tokens = [torch.tensor([1, 2, 3, 1]), torch.tensor([2, 1, 2]), torch.tensor([1, 3])]
    masks = [torch.tensor([0, 1, 1, 0]), torch.tensor([1, 1, 0]), torch.zeros(2)]
    batch = dict(
        unconcat_tokens=tokens,
        total_lengths=[4, 3, 2],
        response_lengths=[4, 3, 2],
        loss_masks=masks,
        dataset_ids=[0, 1, 0],
    )
    generator = torch.Generator().manual_seed(23)
    base = torch.randn(1, 9, 5, generator=generator)
    weight = torch.randn(7, 5, generator=generator)
    targets = torch.cat([F.pad(row[1:], (0, 1)) for row in tokens])
    reference_nll = F.cross_entropy(F.linear(base.squeeze(0), weight), targets, reduction="none")

    def run(enabled):
        args._sft_dataset_names = ("general", "code/a", "unused") if enabled else ()
        hidden = base.clone().requires_grad_()
        head_weight = weight.clone().requires_grad_()

        def head(values):
            return F.linear(values, head_weight), None

        values = F.linear(hidden, head_weight) if ordinary else hidden
        loss, count, log = loss_module.loss_function(
            args, batch, 1, values, lm_head_forward=None if ordinary else head
        )
        loss.backward()
        denominator = count.item() if per_token else len(tokens)
        metrics = loss_module.normalize_reduced_loss_metrics(log["keys"], [denominator, *log["values"].tolist()[1:]])
        return loss.detach(), hidden.grad, head_weight.grad, metrics

    before_loss, before_grad, before_head_grad, before_metrics = run(False)
    after_loss, after_grad, after_head_grad, metrics = run(True)
    torch.testing.assert_close(after_loss, before_loss, rtol=0, atol=0)
    torch.testing.assert_close(after_grad, before_grad, rtol=0, atol=0)
    torch.testing.assert_close(after_head_grad, before_head_grad, rtol=0, atol=0)
    assert metrics["loss"] == before_metrics["loss"]
    assert metrics["dataset/general/loss"] == pytest.approx(reference_nll[1:3].mean().item())
    assert metrics["dataset/code%2Fa/loss"] == pytest.approx(reference_nll[4:6].mean().item())
    assert metrics["dataset/general/samples"] == 2
    assert metrics["dataset/general/tokens"] == 2
    assert metrics["dataset/unused/tokens"] == 0
    assert "dataset/unused/loss" not in metrics


def test_dummy_and_fully_masked_source_statistics(monkeypatch):
    loss_module, _ = _backend(monkeypatch)
    args = _args()
    batch = dict(
        unconcat_tokens=[torch.tensor([1, 2])],
        total_lengths=[2],
        response_lengths=[2],
        loss_masks=[torch.zeros(2)],
        dataset_ids=[0],
    )
    for dummy in (False, True):
        batch["__is_dummy__"] = dummy
        logits = torch.randn(1, 2, 4, requires_grad=True)
        loss, count, log = loss_module.loss_function(args, batch, 1, logits)
        assert count == 0
        loss.backward()
        assert torch.count_nonzero(logits.grad) == 0
        metrics = loss_module.normalize_reduced_loss_metrics(log["keys"], log["values"].tolist())
        assert metrics["dataset/general/samples"] == (0 if dummy else 1)
        assert metrics["dataset/general/tokens"] == 0
        assert metrics["dataset/general/token_fraction"] == 0
        assert metrics["dataset/general/sample_fraction"] == (0 if dummy else 1)
        assert "dataset/general/loss" not in metrics


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_masked_nonfinite_source_does_not_abort_step_metrics(monkeypatch, bad):
    from tests.cpu.backends.megatron.test_nonfinite_step_skip_cpu import Chunk, Optimizer, guard_module, update_block

    loss_module, _ = _backend(monkeypatch)
    args = _args()
    batch = dict(
        unconcat_tokens=[torch.tensor([1, 2]), torch.tensor([2, 1])],
        total_lengths=[2, 2],
        response_lengths=[2, 2],
        loss_masks=[torch.zeros(2), torch.tensor([1, 0])],
        dataset_ids=[0, 1],
    )
    base = torch.zeros(1, 4, 4)
    base[:, :2] = bad

    def run(enabled):
        args._sft_dataset_names = ("general", "code/a", "unused") if enabled else ()
        logits = base.clone().requires_grad_()
        loss, count, log = loss_module.loss_function(args, batch, 1, logits)
        loss.backward()
        grad = logits.grad.clone()
        guard = guard_module()
        optimizer = Optimizer(guard)
        optimizer.param = logits
        optimizer.adam = torch.optim.AdamW([logits], lr=0.01)
        scheduler = SimpleNamespace(step=lambda **_: pytest.fail("skipped step must not advance LR"))
        result = update_block(guard)(optimizer, scheduler, [Chunk()])
        assert result["skipped_nonfinite_grad"] and not result["update_successful"]
        assert optimizer.param.grad is None and not optimizer.adam.state
        torch.testing.assert_close(logits, base, rtol=0, atol=0, equal_nan=True)
        metrics = loss_module.normalize_reduced_loss_metrics(log["keys"], log["values"].tolist())
        return loss.detach(), grad, count, metrics

    before_loss, before_grad, before_count, before_metrics = run(False)
    loss, grad, count, metrics = run(True)
    torch.testing.assert_close(loss, before_loss, equal_nan=True)
    torch.testing.assert_close(grad, before_grad, equal_nan=True)
    torch.testing.assert_close(count, before_count)
    assert not torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.isfinite(torch.tensor(before_metrics["loss"]))
    assert metrics["dataset/general/tokens"] == 0
    assert "dataset/general/loss" not in metrics
    assert metrics["dataset/code%2Fa/loss"] == pytest.approx(torch.log(torch.tensor(4.0)).item())
    # A nonfinite supervised token must remain visible; do not sanitize it.
    sums = dataset_loss_sums(("active",), [0], torch.tensor([bad]), [torch.ones(1)], count_samples=True)
    active = finalize_dataset_loss_metrics({key: value.item() for key, value in sums.items()})
    torch.testing.assert_close(torch.tensor(active["dataset/active/loss"]), torch.tensor(bad), equal_nan=True)


@pytest.mark.parametrize("cp_size", [None, 1, 2], ids=["static", "dynamic-1", "dynamic-2"])
def test_cp_statistics_count_each_source_once(monkeypatch, cp_size):
    loss_module, cp = _backend(monkeypatch, cp_size=4)
    cp.mpu.get_data_parallel_world_size = lambda **_: 1
    args = _args()
    lengths = [8, 7, 6, 5]
    ids = [0, 0, 0, 1]
    masks = [F.pad(torch.ones(n - 1), (0, 1)) for n in lengths]
    logs = []
    for rank in range(4):
        monkeypatch.setattr(cp.mpu, "get_context_parallel_rank", lambda: rank)
        batch = dict(
            total_lengths=lengths.copy(),
            response_lengths=lengths.copy(),
            dataset_ids=ids.copy(),
            loss_masks=masks.copy(),
            row_ids=list(range(4)),
        )
        cp_kwargs = {}
        if cp_size is not None:
            assert cp.dynamic_cp_split_data(batch, 8 // cp_size) == cp_size
            cp_kwargs = dict(dynamic_cp_size=cp_size, dynamic_cp_rank=rank % cp_size)
            batch.update(cp_kwargs)
        local = [
            cp.slice_log_prob_with_cp(
                torch.full((n,), float(row + 1)),
                n,
                n,
                **cp_kwargs,
            )
            for row, n in zip(batch["row_ids"], batch["total_lengths"], strict=True)
        ]
        logs.append(loss_module._sft_loss_metrics(args, batch, torch.tensor(0.0), torch.cat(local)))
    summed = {key: sum(log[key] for log in logs).item() for key in logs[0]}
    metrics = loss_module.normalize_reduced_loss_metrics(list(summed), [22, *summed.values()])
    assert metrics["dataset/general/samples"] == 3
    assert metrics["dataset/code%2Fa/samples"] == 1
    assert metrics["dataset/general/tokens"] == 18
    assert metrics["dataset/code%2Fa/tokens"] == 4
    assert metrics["dataset/general/loss"] == pytest.approx((7 + 2 * 6 + 3 * 5) / 18)
    assert metrics["dataset/code%2Fa/loss"] == 4
