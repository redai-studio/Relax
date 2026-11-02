# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dense and distributed oracles for SC scoring and Megatron loss
integration."""

import importlib
import sys
from datetime import timedelta
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class _TPReduce(torch.autograd.Function):
    """Megatron's forward SUM / backward identity mapping for CPU tests."""

    @staticmethod
    def forward(ctx, value, process_group):
        result = value.clone()
        dist.all_reduce(result, group=process_group)
        return result

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _backend(monkeypatch, *, rank=0, cp_size=1, tp_size=1):
    # Restore both import caches: first imports under the stub must not leave
    # fake process groups in later tests, including real Megatron GPU tests.
    parent = importlib.import_module("relax.backends.megatron")
    for leaf in ("loss", "cp_utils", "score_centering"):
        name = f"relax.backends.megatron.{leaf}"
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
        sys.modules.pop(name)
        monkeypatch.setattr(parent, leaf, None, raising=False)

    tp_group = None
    if dist.is_initialized():
        if tp_size > 1:
            tp_group = dist.group.WORLD
        else:
            groups = [dist.new_group([i]) for i in range(dist.get_world_size())]
            tp_group = groups[rank]
    mpu = SimpleNamespace(
        get_context_parallel_world_size=lambda: cp_size,
        get_context_parallel_rank=lambda: rank if cp_size > 1 else 0,
        get_context_parallel_group=lambda: dist.group.WORLD if cp_size > 1 else None,
        get_dynamic_data_context_parallel_groups=lambda group_size: dist.group.WORLD,
        get_tensor_model_parallel_world_size=lambda: tp_size,
        get_tensor_model_parallel_rank=lambda: rank if tp_size > 1 else 0,
        get_tensor_model_parallel_group=lambda: tp_group,
    )
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    core.mpu = mpu
    mappings = ModuleType("megatron.core.tensor_parallel.mappings")
    mappings.reduce_from_tensor_model_parallel_region = lambda value: _TPReduce.apply(value, tp_group)
    for name, module in (
        ("megatron", megatron),
        ("megatron.core", core),
        ("megatron.core.tensor_parallel.mappings", mappings),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    loss = importlib.import_module("relax.backends.megatron.loss")
    monkeypatch.setattr(loss, "mpu", mpu)
    cp = importlib.import_module("relax.backends.megatron.cp_utils")
    monkeypatch.setattr(cp, "mpu", mpu)
    return loss, cp


def _args(**overrides):
    values = dict(
        advantage_estimator="score_centering",
        loss_type="policy_loss",
        score_centering_vocab_size=5,
        rollout_temperature=0.7,
        log_probs_chunk_size=-1,
        qkv_format="thd",
        allgather_cp=False,
        entropy_coef=0.1,
        use_kl_loss=False,
        true_on_policy_mode=False,
        use_rollout_logprobs=False,
        use_opsm=False,
        use_tis=False,
        get_mismatch_metrics=False,
        custom_pg_loss_reducer_function_path=None,
        score_centering_is="none",
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("already_imported", [False, True])
def test_score_centering_backend_restores_import_caches(monkeypatch, already_imported):
    parent = importlib.import_module("relax.backends.megatron")
    originals = {}
    for leaf in ("loss", "cp_utils", "score_centering"):
        name = f"relax.backends.megatron.{leaf}"
        if already_imported:
            originals[leaf] = ModuleType(name)
            originals[leaf].mpu = object()
            monkeypatch.setitem(sys.modules, name, originals[leaf])
            monkeypatch.setattr(parent, leaf, originals[leaf], raising=False)
        else:
            monkeypatch.delitem(sys.modules, name, raising=False)
            monkeypatch.delattr(parent, leaf, raising=False)

    with pytest.MonkeyPatch.context() as isolated:
        loss, cp = _backend(isolated)
        assert loss.mpu is cp.mpu is sys.modules["megatron.core"].mpu
        assert parent.loss is loss and parent.cp_utils is cp
        assert parent.score_centering.mpu is loss.mpu
        assert callable(loss.get_log_probs_and_entropy)

    for leaf in ("loss", "cp_utils", "score_centering"):
        name = f"relax.backends.megatron.{leaf}"
        if already_imported:
            assert sys.modules[name] is originals[leaf]
            assert getattr(parent, leaf) is originals[leaf]
        else:
            assert name not in sys.modules
            assert not hasattr(parent, leaf)


def _score_inputs():
    torch.manual_seed(17)
    logits = torch.randn(1, 8, 8)
    # Padded vocabulary dominates if the scoring path forgets the real size.
    logits[..., 5:] = 30
    tokens = torch.tensor([1, 2, 0, 4, 1, 3, 2, 0])
    ids = torch.tensor([[0, 2], [4, 1], [3, 2], [0, 4], [1, 2], [3, 0]])
    return logits, tokens, ids


def _dense_scoring(logits, tokens, ids, temperature):
    logp = (logits[0, 1:7, :5] / temperature).log_softmax(-1)
    sample = logp.gather(-1, tokens[2:, None]).squeeze(-1)
    head = logp.gather(-1, ids)
    entropy = -(logp.exp() * logp).sum(-1)
    return sample, head, entropy


def test_scoring_uses_real_vocab_once_for_sample_head_and_entropy(monkeypatch):
    loss, _ = _backend(monkeypatch)
    args = _args()
    base, tokens, ids = _score_inputs()
    logits = base.clone().requires_grad_()
    oracle_logits = base.clone().requires_grad_()
    _, result = loss.get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=[tokens],
        total_lengths=[8],
        response_lengths=[6],
        gather_topk_token_ids=[ids],
        with_entropy=True,
    )
    expected = _dense_scoring(oracle_logits, tokens, ids, args.rollout_temperature)
    actual = tuple(result[key][0] for key in ("log_probs", "topk_log_probs", "entropy"))
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference)
    sum(value.sum() for value in actual).backward()
    sum(value.sum() for value in expected).backward()
    torch.testing.assert_close(logits.grad, oracle_logits.grad, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(logits.grad[..., 5:]) == 0

    # ActorFwd/reference use the same helper without requesting head candidates.
    _, reference = loss.get_log_probs_and_entropy(
        base,
        args=args,
        unconcat_tokens=[tokens],
        total_lengths=[8],
        response_lengths=[6],
    )
    torch.testing.assert_close(reference["log_probs"][0], expected[0])


@pytest.mark.parametrize("dynamic", [False, True])
def test_scoring_ring_and_dynamic_cp_align_head_rows(monkeypatch, dynamic):
    args = _args()
    base, tokens, ids = _score_inputs()
    expected = _dense_scoring(base, tokens, ids, args.rollout_temperature)
    for rank in range(2):
        loss, cp = _backend(monkeypatch, rank=rank, cp_size=2)
        # CP=2 owns chunks (rank, 3-rank), each containing two logits.
        owned = torch.cat((torch.arange(2 * rank, 2 * rank + 2), torch.arange(6 - 2 * rank, 8 - 2 * rank)))
        cp_kwargs = {"dynamic_cp_size": 2, "dynamic_cp_rank": rank} if dynamic else {}
        local_ids = cp.slice_log_prob_with_cp(ids, 8, 6, **cp_kwargs)
        _, result = loss.get_log_probs_and_entropy(
            base[:, owned],
            args=args,
            unconcat_tokens=[tokens],
            total_lengths=[8],
            response_lengths=[6],
            gather_topk_token_ids=[local_ids],
            with_entropy=True,
            **cp_kwargs,
        )
        for key, reference in zip(("log_probs", "topk_log_probs", "entropy"), expected, strict=True):
            torch.testing.assert_close(result[key][0], cp.slice_log_prob_with_cp(reference, 8, 6, **cp_kwargs))


def test_dynamic_cp_one_skips_static_allgather_redistribution(monkeypatch):
    loss, _ = _backend(monkeypatch, rank=1, cp_size=2)
    base, tokens, ids = _score_inputs()
    args = _args(allgather_cp=True)
    _, result = loss.get_log_probs_and_entropy(
        base,
        args=args,
        unconcat_tokens=[tokens],
        total_lengths=[8],
        response_lengths=[6],
        gather_topk_token_ids=[ids],
        with_entropy=True,
        dynamic_cp_size=1,
        dynamic_cp_rank=0,
    )
    expected = _dense_scoring(base, tokens, ids, args.rollout_temperature)
    for key, reference in zip(("log_probs", "topk_log_probs", "entropy"), expected, strict=True):
        torch.testing.assert_close(result[key][0], reference)


def test_policy_sc_masked_placeholders_and_mismatch_only_preserve_gradient(monkeypatch):
    loss, _ = _backend(monkeypatch)
    base, tokens, ids = _score_inputs()
    mask = torch.tensor([1, 0, 1, 1, 0, 1])
    q_logp = torch.full((6,), -2.0).masked_fill(mask == 0, float("-inf"))
    q_head = torch.full((6, 2), -2.5).masked_fill(mask[:, None] == 0, float("-inf"))
    ids = ids.masked_fill(mask[:, None] == 0, -1)
    monkeypatch.setattr(loss, "compute_policy_opd_loss", lambda **kwargs: (None, {}))
    monkeypatch.setattr(loss, "vanilla_tis_function", lambda **kwargs: pytest.fail("SC entered external TIS"))
    results = []
    for diagnostics in (False, True):
        logits = base.clone().requires_grad_()
        batch = {
            "advantages": [torch.tensor([1.0, float("nan"), -1.0, 2.0, float("nan"), 0.5])],
            "log_probs": [q_logp],
            "rollout_log_probs": [q_logp],
            "rollout_topk_token_ids": [ids],
            "rollout_topk_log_probs": [q_head],
            "unconcat_tokens": [tokens],
            "total_lengths": [8],
            "response_lengths": [6],
            "loss_masks": [mask],
        }
        value, metrics = loss.policy_loss_function(
            _args(get_mismatch_metrics=diagnostics), batch, logits, lambda x: (x * mask).sum()
        )
        assert torch.isfinite(value)
        assert all(torch.isfinite(metric) for metric in metrics.values())
        value.backward()
        assert torch.isfinite(logits.grad).all()
        results.append((value, logits.grad))
    torch.testing.assert_close(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1])


def _distributed_scoring_worker(rank, init_file, kind):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            is_cp = kind.startswith("cp")
            loss, cp = _backend(monkeypatch, rank=rank, tp_size=2 if kind == "tp" else 1, cp_size=2 if is_cp else 1)
            args = _args(allgather_cp=is_cp)
            cp_kwargs = {}
            if kind == "cp_dynamic":
                cp_kwargs = {"dynamic_cp_size": 2, "dynamic_cp_rank": rank}
                # The effective group/rank must win over static max-CP state.
                monkeypatch.setattr(loss.mpu, "get_context_parallel_rank", lambda: 1 - rank)
                monkeypatch.setattr(loss.mpu, "get_context_parallel_world_size", lambda: 4)
            base, tokens, ids = _score_inputs()
            expected_base = base.clone().requires_grad_()
            expected = _dense_scoring(expected_base, tokens, ids, args.rollout_temperature)
            response_length = 1 if kind == "cp_empty_rank" else 6
            if response_length == 1:
                expected = tuple(value[-1:] for value in expected)
                ids = ids[-1:]
            if kind == "tp":
                local = base[..., rank * 4 : (rank + 1) * 4].clone().requires_grad_()
                local_ids = ids
            else:
                local = base[:, rank * 4 : (rank + 1) * 4].clone().requires_grad_()
                local_ids = cp.slice_log_prob_with_cp(ids, 8, response_length, **cp_kwargs)
            _, result = loss.get_log_probs_and_entropy(
                local,
                args=args,
                unconcat_tokens=[tokens],
                total_lengths=[8],
                response_lengths=[response_length],
                gather_topk_token_ids=[local_ids],
                with_entropy=True,
                **cp_kwargs,
            )
            actual = tuple(result[key][0] for key in ("log_probs", "topk_log_probs", "entropy"))
            for value, reference in zip(actual, expected, strict=True):
                if is_cp:
                    reference = cp.slice_log_prob_with_cp(reference, 8, response_length, **cp_kwargs)
                torch.testing.assert_close(value, reference)
            # Mirrors loss_function's graph connection for empty CP ranks.
            (sum(value.sum() for value in actual) + 0 * local.sum()).backward()
            sum(value.sum() for value in expected).backward()
            expected_grad = (
                expected_base.grad[..., rank * 4 : (rank + 1) * 4]
                if kind == "tp"
                else expected_base.grad[:, rank * 4 : (rank + 1) * 4]
            )
            torch.testing.assert_close(local.grad, expected_grad, atol=1e-6, rtol=1e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["tp", "cp", "cp_empty_rank", "cp_dynamic"])
def test_scoring_distributed_selected_probabilities_and_gradients(tmp_path, kind):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU distributed Gloo is unavailable")
    mp.spawn(_distributed_scoring_worker, args=(str(tmp_path / "rendezvous"), kind), nprocs=2, join=True)
