# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Audit real CUDA losses and NCCL CP counts without replacing Megatron."""

import json
import math
from argparse import Namespace
from contextlib import contextmanager
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.utils import weight_version_audit as audit


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and the patched Megatron image")


@contextmanager
def _parallel(rank, world_size, rendezvous, cp_size, *, dynamic=False):
    from megatron.core import mpu

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timedelta(seconds=90)
    )
    mpu.initialize_model_parallel(context_parallel_size=cp_size, dynamic_context_parallel=dynamic)
    try:
        yield
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def cuda_backend(tmp_path_factory):
    with _parallel(0, 1, tmp_path_factory.mktemp("audit-cuda") / "rendezvous", 1):
        from relax.backends.megatron import cp_utils, loss, model

        yield loss, model, cp_utils


@pytest.mark.parametrize("auxiliary", ["none", "entropy", "kl", "opd", "mtp", "opsm"])
def test_cuda_icepop_audit_and_gradient_parity(cuda_backend, tmp_path, monkeypatch, auxiliary):
    from tests.cpu.backends.megatron.test_score_centering_loss import _args

    loss_module, model, cp = cuda_backend
    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    publisher = audit.PublicationLedger(str(tmp_path), "gpu-policy")
    spans = []
    for i in range(2):
        tag = publisher.publish()
        spans += audit.generation_spans(
            {"weight_versions": [{"start": 0, "end": 1, "version": tag}]},
            1,
            offset=2 + i,
            request_id="request",
            attempt_id="attempt",
            piece_index=i,
        )
    publisher.publish()
    row = audit.prepare_row({"session_id": "gpu-session", "spans": spans}, 4, 2, [1, 1])
    args = _args(
        advantage_estimator="grpo",
        use_tis=True,
        use_opsm=auxiliary == "opsm",
        opsm_delta=0.01,
        custom_tis_function_path="relax.backends.megatron.loss.icepop_function",
        tis_clip_low=0.5,
        tis_clip=2.0,
        calculate_per_token_loss=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        entropy_coef=0.1 if auxiliary == "entropy" else 0.0,
        use_kl_loss=auxiliary == "kl",
        kl_loss_coef=0.2,
        use_unbiased_kl=False,
        kl_loss_type="k2",
        opd_token_selection="student_sampled",
        opd_loss_coef=0.5 if auxiliary == "opd" else 0.0,
        opd_kl_type="reverse_kl",
        opd_jsd_alpha=0.5,
        enable_mtp_training=auxiliary == "mtp",
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
    )
    batch = {
        "unconcat_tokens": [torch.tensor([0, 0, 1, 2], device="cuda")],
        "advantages": [torch.ones(2, device="cuda")],
        "log_probs": [torch.full((2,), -math.log(4), device="cuda")],
        "rollout_log_probs": [torch.tensor([-math.log(4) - math.log(10), -math.log(4)], device="cuda")],
        "ref_log_probs": [torch.full((2,), -2.0, device="cuda")],
        "teacher_log_probs": [torch.full((2,), -2.0, device="cuda")],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2, device="cuda")],
    }
    if auxiliary == "opsm":
        batch["advantages"] = [torch.tensor([1.0, -1.0], device="cuda")]
        batch["log_probs"] = [batch["log_probs"][0] + 0.05]
    results = []
    for collect in (False, True):
        if collect:
            batch[audit.PREPARED_FIELD] = [row]
        logits = torch.zeros((1, 4, 4), device="cuda", requires_grad=True)
        reducer = cp.get_sum_of_sample_mean([4], [2], batch["loss_masks"])
        value, metrics = loss_module.policy_loss_function(args, batch, logits, reducer)
        value.backward()
        results.append((value.detach(), logits.grad.clone()))
        assert "_weight_audit_weights" not in metrics
    for baseline, observed in zip(*results, strict=True):
        torch.testing.assert_close(baseline, observed, rtol=0, atol=0)
    counts = model._weight_audit_counts(args, batch)
    assert counts.is_cuda and counts.dtype == torch.int64
    expected = [0, 0, 0, 0] if auxiliary == "opsm" else [1, 0, 1, 0] if auxiliary == "none" else [2, 1, 1, 0]
    assert counts.tolist() == expected
    step = audit.AuditStep(auxiliary)
    step.observe(0, [row], counts=counts)
    summary = step.commit(0)
    assert summary["eligible_tokens"] == expected[0]
    assert summary["token_weighted_age"] == (None if expected[0] == 0 else 1.0 if expected[0] == 1 else 1.5)
    assert summary["aggregate_complete"]
    torch.testing.assert_close(batch["loss_masks"][0], torch.ones(2, device="cuda"))


def reject_policy_tokens(*, pg_loss, loss_masks, **kwargs):
    return pg_loss, [torch.zeros_like(mask) for mask in loss_masks], {}


def test_cuda_opd_entropy_gate(cuda_backend, tmp_path, monkeypatch):
    from tests.cpu.backends.megatron.test_score_centering_loss import _args

    loss_module, model, cp = cuda_backend
    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    args = _args(
        advantage_estimator="grpo",
        use_tis=True,
        custom_tis_function_path=f"{__name__}.reject_policy_tokens",
        calculate_per_token_loss=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        entropy_coef=0.0,
        opd_token_selection="teacher_topk",
        opd_loss_coef=1.0,
        opd_kl_type="forward_kl",
        opd_jsd_alpha=0.5,
        opd_fkl_entropy_gate=True,
        opd_fkl_entropy_threshold=0.5,
    )
    row = audit.prepare_row(
        {"spans": [{"token_start": 2, "token_end": 3}, {"token_start": 3, "token_end": 4}]}, 4, 2, [1, 1]
    )
    batch = {
        "unconcat_tokens": [torch.tensor([0, 0, 1, 2], device="cuda")],
        "advantages": [torch.ones(2, device="cuda")],
        "log_probs": [torch.full((2,), -math.log(4), device="cuda")],
        "rollout_log_probs": [torch.full((2,), -math.log(4), device="cuda")],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2, device="cuda")],
        "opd_topk_token_ids": [torch.tensor([[1, 2], [1, 2]], device="cuda")],
        "opd_topk_teacher_log_probs": [torch.tensor([[0.0, -30.0], [-math.log(2), -math.log(2)]], device="cuda")],
    }
    results = []
    for collect in (False, True):
        if collect:
            batch[audit.PREPARED_FIELD] = [row]
        logits = torch.zeros((1, 4, 4), device="cuda", requires_grad=True)
        reducer = cp.get_sum_of_sample_mean([4], [2], batch["loss_masks"])
        value, metrics = loss_module.policy_loss_function(args, batch, logits, reducer)
        value.backward()
        results.append((value.detach(), logits.grad.clone()))
    for baseline, observed in zip(*results, strict=True):
        torch.testing.assert_close(baseline, observed, rtol=0, atol=0)
    assert metrics["opd_fkl_gate_frac"].item() == 0.5
    assert results[1][1][0, 1].count_nonzero().item() == 0
    assert results[1][1][0, 2].abs().sum().item() > 0
    counts = model._weight_audit_counts(args, batch)
    assert counts.is_cuda and counts.tolist() == [1, 0, 1, 0]
    step = audit.AuditStep("opd-gate")
    step.observe(0, [row], counts=counts)
    summary = step.commit(0)
    assert summary["eligible_tokens"] == 1
    assert summary["known_version_tokens"] == 0
    assert summary["missing_token_rate"] == 1
    assert summary["token_weighted_age"] is None
    report = json.loads(next(tmp_path.glob("*/consumed-rank-0.jsonl")).read_text().splitlines()[0])
    assert [span["eligible_tokens"] for span in report["spans"]] == [0, 1]


def _cp_counts_worker(rank, rendezvous):
    for dynamic in (False, True):
        with _parallel(rank, 2, f"{rendezvous}-{dynamic}", 1 if dynamic else 2, dynamic=dynamic):
            _check_cp_counts(rank, dynamic)


def _check_cp_counts(rank, dynamic):
    from relax.backends.megatron import loss, model

    for layout in ("thd", "bshd"):
        for padded in (False, True):
            for prompt in (0, 2):
                mask = torch.tensor([1, 0, 1, 1, 0, 1, 0][max(prompt, 1) - 1 :], device="cuda")
                batch = {
                    audit.PREPARED_FIELD: [{"spans": [{"token_start": 2, "token_end": 7}]}],
                    "loss_masks": [mask],
                    "total_lengths": [8],
                    "response_lengths": [8 - prompt],
                }
                if dynamic:
                    batch.update(dynamic_cp_size=2, dynamic_cp_rank=rank)
                if layout == "bshd":
                    batch["max_seq_lens"] = [12]
                if padded:
                    batch["padded_total_lengths"] = [12]
                args = Namespace(qkv_format=layout)
                for mode in ("original", "policy", "reject", "text_mtp", "mtp_zero", "vl_mtp", "vl_covered"):
                    args.enable_mtp_training = mode in ("text_mtp", "mtp_zero", "vl_mtp", "vl_covered")
                    args.mtp_num_layers = 1
                    args.mtp_loss_scaling_factor = 0 if mode == "mtp_zero" else 0.1
                    args.is_vl_model = mode.startswith("vl_")
                    reduced = (
                        batch["loss_masks"]
                        if mode in ("original", "policy", "vl_covered")
                        else [torch.zeros_like(mask)]
                    )
                    policy_mask = None
                    if mode == "policy":
                        # Enumerate zigzag target positions independently of the production slicer.
                        chunk = 3 if padded or layout == "bshd" else 2
                        positions = [
                            token
                            for block in (rank, 3 - rank)
                            for token in range(max(block * chunk + 1, prompt), min((block + 1) * chunk + 1, 8))
                        ]
                        policy_mask = torch.tensor([token != 4 for token in positions], device="cuda")
                    batch["weight_audit_loss_masks"], batch["weight_audit_unconfirmed"] = loss._weight_audit_masks(
                        args, batch, reduced, policy_mask, None
                    )
                    counts = model._weight_audit_counts(args, batch)
                    expected = [4 if prompt == 0 else 3, 3, 0]
                    if mode in ("reject", "mtp_zero"):
                        expected = [0, 0, 0]
                    elif mode == "policy":
                        expected = [3 if prompt == 0 else 2, 2, 0]
                    elif mode == "text_mtp":
                        expected = [3, 3, 0]
                    elif mode == "vl_mtp":
                        expected = [0, 0, 3]
                    assert counts.is_cuda
                    assert counts.tolist() == expected, (rank, dynamic, layout, padded, prompt, mode, counts)


def test_cuda_nccl_context_parallel_counts(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices for real NCCL context parallelism")
    mp.spawn(_cp_counts_worker, args=(str(tmp_path / "cp-rendezvous"),), nprocs=2, join=True)
