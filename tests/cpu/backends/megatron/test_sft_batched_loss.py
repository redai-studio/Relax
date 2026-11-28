# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from datetime import timedelta
from itertools import accumulate
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from tests.cpu.backends.megatron.test_score_centering_loss import _backend


def _check_batched_loss(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=45))
    try:
        cp_rank, tp_rank = divmod(rank, 2)
        tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        cp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        tp_group = tp_groups[cp_rank]
        cp_group = cp_groups[tp_rank]
        vocab = slice(4 * tp_rank, 4 * (tp_rank + 1))

        generator = torch.Generator().manual_seed(314)
        hidden = torch.randn(32, 3, generator=generator)
        weight = torch.randn(8, 3, generator=generator)
        dense_logits = F.linear(hidden, weight).requires_grad_()
        tokens = [torch.tensor([3, 6, 2, 0, 4, 2, 7, 0]), torch.tensor([2, 5, 1, 4, 6, 3, 2, 1])]
        masks = [torch.tensor([0, 1, 1, 0, 1, 1, 0, 0]), torch.tensor([1, 0, 0, 1, 0, 1, 1, 0])]
        targets = torch.cat([F.pad(row[1:], (0, 1)) for row in tokens])
        dense_loss = (F.cross_entropy(dense_logits[:16] / 0.8, targets, reduction="none") * torch.cat(masks)).sum()
        dense_loss.backward()

        with pytest.MonkeyPatch.context() as patch:
            loss, _ = _backend(patch, rank=rank, cp_size=2, tp_size=2)
            patch.setattr(loss.mpu, "get_tensor_model_parallel_group", lambda: tp_group)
            patch.setattr(loss.mpu, "get_context_parallel_group", lambda: cp_group)
            patch.setattr(loss.mpu, "get_dynamic_data_context_parallel_groups", lambda group_size: cp_group)
            patch.setattr(loss.mpu, "get_context_parallel_rank", lambda: cp_rank)
            patch.setenv("RELAX_LOGPROB_CE_IMPL", "native")

            # The zigzag layout assigns the first and last two rows of each
            # sample to CP rank 0 and the middle four to CP rank 1.
            zigzag_rows = ([0, 1, 6, 7, 8, 9, 14, 15], [2, 3, 4, 5, 10, 11, 12, 13])
            for layout in ("zigzag", "allgather", "allgather_with_padding_rank"):
                allgather_cp = layout != "zigzag"
                span = 16 if layout == "allgather_with_padding_rank" else 8
                rows = list(range(span * cp_rank, span * (cp_rank + 1))) if allgather_cp else zigzag_rows[cp_rank]
                batch = dict(
                    unconcat_tokens=tokens,
                    total_lengths=[8, 8],
                    response_lengths=[8, 8],
                    loss_masks=masks,
                )
                if allgather_cp:
                    batch.update(dynamic_cp_size=2, dynamic_cp_rank=cp_rank)
                for chunked in (False, True):
                    args = SimpleNamespace(
                        loss_type="sft",
                        task_type="causal_lm",
                        advantage_estimator=None,
                        qkv_format="thd",
                        allgather_cp=allgather_cp,
                        calculate_per_token_loss=True,
                        sft_chunked_logits=chunked,
                        sft_logits_chunk_size=5,
                        log_probs_chunk_size=-1,
                        recompute_loss_function=chunked,
                        rollout_temperature=0.8,
                        global_batch_size=2,
                    )
                    for batched in (False, True):
                        args.sft_batched_loss = batched
                        head_rows = []
                        local_weight = weight[vocab].clone().requires_grad_()
                        local_hidden = hidden[rows].unsqueeze(0).clone().requires_grad_()
                        local_logits = dense_logits.detach()[rows, vocab].unsqueeze(0).clone().requires_grad_()

                        def project(values: torch.Tensor) -> tuple[torch.Tensor, None]:
                            head_rows.append(values.size(0))
                            return F.linear(values, local_weight), None

                        value, count, _ = loss.loss_function(
                            args,
                            batch,
                            1,
                            local_hidden if chunked else local_logits,
                            lm_head_forward=project if chunked else None,
                        )
                        if chunked and not allgather_cp:
                            per_sample = 3 + cp_rank
                            if batched:
                                assert sum(head_rows) == 2 * per_sample
                                assert all(0 < size <= args.sft_logits_chunk_size for size in head_rows)
                                assert per_sample not in accumulate(head_rows)
                            else:
                                assert head_rows == [per_sample, per_sample]
                        actual = value.detach().clone()
                        dist.all_reduce(actual, group=cp_group)
                        dist.all_reduce(count, group=cp_group)
                        torch.testing.assert_close(actual, dense_loss.detach())
                        assert count == torch.cat(masks).sum()
                        value.backward()

                        expected_grad = dense_logits.grad[rows, vocab]
                        if chunked:
                            # This local Linear has no TP dgrad reduction. Compare
                            # its contribution from the local vocabulary shard.
                            torch.testing.assert_close(local_hidden.grad[0], expected_grad @ weight[vocab])
                            if layout == "allgather_with_padding_rank" and cp_rank:
                                assert local_weight.grad is None
                            else:
                                torch.testing.assert_close(local_weight.grad, expected_grad.T @ hidden[rows])
                        else:
                            torch.testing.assert_close(local_logits.grad[0], expected_grad)
    finally:
        dist.destroy_process_group()


def test_sft_batched_loss_preserves_tp_cp_loss_and_gradients(tmp_path: Path) -> None:
    mp.spawn(_check_batched_loss, args=(f"file://{tmp_path / 'dist'}",), nprocs=4)


@pytest.mark.parametrize(("chunk_size", "max_rows"), [(3, 3), (-1, 2)])
def test_sft_batched_loss_bounds_gathered_logits(
    monkeypatch: pytest.MonkeyPatch, chunk_size: int, max_rows: int
) -> None:
    loss, _ = _backend(monkeypatch)
    args = SimpleNamespace(
        loss_type="sft",
        task_type="causal_lm",
        advantage_estimator=None,
        qkv_format="thd",
        calculate_per_token_loss=True,
        allgather_cp=False,
        sft_chunked_logits=False,
        log_probs_chunk_size=chunk_size,
        recompute_loss_function=False,
        rollout_temperature=1.0,
        global_batch_size=3,
    )
    batch = dict(
        unconcat_tokens=[torch.tensor([5, 4, 3, 2, 1]), torch.tensor([0, 2, 3, 4]), torch.tensor([3, 5, 1])],
        total_lengths=[5, 4, 3],
        response_lengths=[2, 1, 2],
        loss_masks=[torch.ones(2), torch.ones(1), torch.ones(2)],
    )
    logits = torch.randn(1, 12, 6, generator=torch.Generator().manual_seed(27)).requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    expected = F.cross_entropy(reference[0, [2, 3, 7, 9, 10]], torch.tensor([2, 1, 4, 5, 1]), reduction="sum")

    # Record allocations made by gathering from the model's vocab tensor.
    gathered_rows = []
    model_storage = logits.untyped_storage().data_ptr()
    original_index_select = torch.Tensor.index_select

    def record_gather(tensor, dim, index):
        result = original_index_select(tensor, dim, index)
        if tensor.untyped_storage().data_ptr() == model_storage:
            gathered_rows.append(result.size(dim))
        return result

    monkeypatch.setattr(torch.Tensor, "index_select", record_gather)
    actual, count, _ = loss.loss_function(args, batch, 1, logits)
    assert max(gathered_rows, default=0) <= max_rows
    assert count == 5
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)


def test_sft_batched_loss_bounds_temperature_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    loss, _ = _backend(monkeypatch)
    args = SimpleNamespace(
        loss_type="sft",
        task_type="causal_lm",
        advantage_estimator=None,
        qkv_format="thd",
        calculate_per_token_loss=True,
        allgather_cp=False,
        sft_chunked_logits=False,
        log_probs_chunk_size=-1,
        recompute_loss_function=False,
        rollout_temperature=0.8,
        global_batch_size=2,
    )
    labels = [torch.tensor([1, 2]), torch.tensor([0, 1, 2])]
    batch = dict(
        unconcat_tokens=[torch.tensor([0, 1]), torch.tensor([2, 0, 1])],
        unconcat_pre_shifted_labels=labels,
        total_lengths=[2, 3],
        response_lengths=[2, 3],
        loss_masks=[torch.ones(2), torch.ones(3)],
    )
    logits = torch.randn(1, 5, 3, generator=torch.Generator().manual_seed(32)).requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    expected = F.cross_entropy(reference[0] / 0.8, torch.cat(labels), reduction="sum")

    scaled_rows = []
    model_storage = logits.untyped_storage().data_ptr()

    class TrackTemperatureScale(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func.overloadpacket == torch.ops.aten.div and args[0].untyped_storage().data_ptr() == model_storage:
                scaled_rows.append(result.size(0))
            return result

    with TrackTemperatureScale():
        actual, count, _ = loss.loss_function(args, batch, 1, logits)
    assert scaled_rows, "Expected temperature scaling of the model logits"
    assert max(scaled_rows) <= 3
    assert sum(scaled_rows) == 5
    assert count == 5
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)


def test_sft_batched_loss_restores_selective_rows_before_reduction(monkeypatch: pytest.MonkeyPatch) -> None:
    loss, _ = _backend(monkeypatch)
    monkeypatch.setenv("RELAX_SFT_SELECTIVE_LOGITS", "1")
    monkeypatch.setenv("RELAX_LOGPROB_CE_IMPL", "native")
    args = SimpleNamespace(
        loss_type="sft",
        task_type="causal_lm",
        advantage_estimator=None,
        qkv_format="thd",
        calculate_per_token_loss=True,
        allgather_cp=False,
        sft_chunked_logits=True,
        sft_logits_chunk_size=2,
        rollout_temperature=1.0,
        global_batch_size=2,
    )
    batch = dict(
        unconcat_tokens=[torch.tensor([0, 1, 2]), torch.tensor([2, 0, 1])],
        unconcat_pre_shifted_labels=[torch.tensor([1, -100, -100]), torch.tensor([-100, 1, -100])],
        total_lengths=[3, 3],
        response_lengths=[3, 3],
        loss_masks=[torch.tensor([1, 0, 0]), torch.tensor([0, 1, 0])],
        selective_logit_rows=[(3, torch.tensor([0])), (3, torch.tensor([1]))],
    )
    logits = torch.rand(1, 6, 3, generator=torch.Generator().manual_seed(4)).requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    expected = F.cross_entropy(reference[0, [0, 4]], torch.tensor([1, 1]), reduction="sum")
    projected_rows = []

    def project(hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        projected_rows.append(hidden.size(0))
        return hidden, None

    actual, count, _ = loss.loss_function(args, batch, 1, logits, lm_head_forward=project)
    assert projected_rows == [2]
    assert count == 2
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)
