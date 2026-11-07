# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execute the production train_one_step function with a CPU Megatron schedule.

Only the optional Megatron runtime is replaced. The real function body, Torch
tensors, provenance preparation, loss observer, and report writer all execute.
"""

import ast
import itertools
import json
import logging
import math
import uuid
from argparse import Namespace
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relax.utils import weight_version_audit as audit
from relax.utils.env import Envs
from relax.utils.memory_utils import collect_manual_gc
from relax.utils.training.packing_metrics import PackingMetrics


@pytest.mark.parametrize("chunks", [1, 2])
@pytest.mark.parametrize("dummy", [False, True])
def test_real_train_step_commits_each_logical_loss_once(tmp_path, monkeypatch, chunks, dummy):
    summary = _run(tmp_path, monkeypatch, chunks=chunks, dummy=dummy, repeat_callback=True)
    if dummy:
        assert summary is None
    else:
        assert summary["eligible_tokens"] == 6
        assert summary["token_weighted_age"] == 1


@pytest.mark.parametrize("tp_rank,cp_rank,dynamic_cp_rank", [(1, 0, None), (0, 1, None), (0, 0, 1)])
def test_model_parallel_copies_do_not_write_exposures(tmp_path, monkeypatch, tp_rank, cp_rank, dynamic_cp_rank):
    assert _run(tmp_path, monkeypatch, tp_rank=tp_rank, cp_rank=cp_rank, dynamic_cp_rank=dynamic_cp_rank) is None


def test_dynamic_cp_subgroup_representative_is_counted(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, cp_rank=2, dynamic_cp_rank=0)["eligible_tokens"] == 6


def test_critic_loss_is_not_an_actor_exposure(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, role="critic") is None


def test_failed_schedule_does_not_commit_staged_exposures(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="backward failed"):
        _run(tmp_path, monkeypatch, fail_schedule=True)
    assert not list(tmp_path.glob("*/consumed*.jsonl"))


@pytest.mark.parametrize("converter", [None, "custom.tokens"])
def test_audit_fetch_fields_match_converter_contract(tmp_path, monkeypatch, converter):
    from relax.utils.training.data_fields import build_data_fields

    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    args = Namespace(
        loss_type="policy_loss",
        advantage_estimator="grpo",
        custom_convert_samples_to_train_data_path=converter,
        use_routing_replay=False,
        true_on_policy_mode=True,
        kl_coef=0.0,
        use_kl_loss=False,
        multimodal_keys=None,
        use_opd=False,
    )
    expected = converter is None
    requested = build_data_fields(args)
    assert (audit.AUDIT_FIELD in requested) is expected
    if converter is not None:
        from transfer_queue.controller import DataPartitionStatus

        from relax.utils import utils

        payload = {
            "tokens": [[1, 2, 3]],
            "total_lengths": [3],
            "response_lengths": [2],
            "loss_masks": [[1, 1]],
            "rollout_log_probs": [[-0.5, -0.5]],
            "rewards": [1.0],
            "sample_indices": [0],
            "sample_index_mask_sums": [2],
            "group_index": [0],
        }
        monkeypatch.setattr(utils, "load_function", lambda _path: lambda _args, _samples: payload)
        converted = utils.convert_samples_to_train_data(args, [])
        partition = DataPartitionStatus(
            partition_id="train_0",
            production_status=torch.ones((1, len(converted)), dtype=torch.int8),
            global_indexes={0},
            field_name_mapping={key: index for index, key in enumerate(converted)},
            production_completed=True,
            actual_sample_count=1,
        )
        assert partition.scan_data_status(requested, "actor") == [0]

    class FetchReached(Exception):
        pass

    def fetch(*_args, **kwargs):
        assert (audit.AUDIT_FIELD in kwargs["data_fields"]) is expected
        raise FetchReached

    path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/actor.py"
    tree = ast.parse(path.read_text())
    definition = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "train_async"
    )
    definition.decorator_list = []
    namespace = {
        "logger": logging.getLogger(__name__),
        "create_stream_dataloader": fetch,
        "mpu": SimpleNamespace(get_data_parallel_rank=lambda: 0),
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    with pytest.raises(FetchReached):
        namespace["train_async"](SimpleNamespace(args=args, data_iterator=None), 0)


def _load_definition(path, name, namespace):
    tree = ast.parse(path.read_text())
    definition = next(
        node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), definition],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _run(
    directory,
    monkeypatch,
    *,
    chunks=1,
    dummy=False,
    repeat_callback=False,
    tp_rank=0,
    cp_rank=0,
    dynamic_cp_rank=None,
    role="actor",
    fail_schedule=False,
    final_mask=None,
):
    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(directory))
    publisher = audit.PublicationLedger(str(directory), "policy")
    version = publisher.publish()
    row = audit.prepare_row(
        {
            "sample_uid": "row",
            "session_id": "session",
            "spans": audit.generation_spans(
                {"weight_versions": [{"start": 0, "end": 2, "version": version}]},
                2,
                offset=4,
                request_id="request",
                attempt_id="request:0",
                piece_index=0,
            ),
        },
        6,
        2,
        [1, 1],
    )
    publisher.publish()
    args = Namespace(
        use_rollout_indexer_replay=False,
        ci_test=False,
        enable_mtp_training=False,
        custom_megatron_before_train_step_hook_path=None,
        use_opd=False,
        data_pad_size_multiplier=1,
        qkv_format="thd",
        allgather_cp=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
        calculate_per_token_loss=False,
        check_for_nan_in_loss_and_grad=True,
        dynamic_context_parallel=False,
        task_type="causal_lm",
    )

    class CPUModel:
        def __init__(self):
            self.role = role

        def __call__(self, **_kwargs):
            return torch.ones(1)

        def zero_grad_buffer(self):
            pass

    model = [CPUModel() for _ in range(chunks)]
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *_args, **_kwargs: 7)

    def loss(_args, _batch, _count, logits, **_kwargs):
        if final_mask is not None:
            _batch["weight_audit_loss_masks"] = [torch.tensor(final_mask)]
            _batch.update(total_lengths=[6], response_lengths=[2])
        return logits.sum() * 0, 1, {"keys": ["loss"], "values": torch.tensor([0.0, 1.0])}

    def schedule(**kwargs):
        losses = []
        for _ in range(3):
            for iterator, chunk in zip(kwargs["data_iterator"], model, strict=True):
                output, callback = kwargs["forward_step_func"](iterator, chunk)
            _, _, reduced = callback(output)
            if repeat_callback:
                callback(output)
            losses.append(reduced)
            if fail_schedule:
                raise RuntimeError("backward failed")
        return losses

    path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/model.py"
    iterator_class = _load_definition(path.parent / "data.py", "DataIterator", {})
    iterators = [
        iterator_class(
            {"tokens": [torch.ones(8)], audit.PREPARED_FIELD: [row]},
            micro_batch_indices=[[0], [0], [0]],
            dummy_micro_batch_offsets={0, 1, 2} if dummy else set(),
        )
        for _ in model
    ]

    def cpu_packing(iterator, keys, *_args):
        selected = iterator.get_next(keys)
        selected.update(tokens=torch.stack(selected["tokens"]), packed_seq_params=None, full_loss_masks=None)
        if dynamic_cp_rank is not None:
            selected["dynamic_cp_rank"] = dynamic_cp_rank
        return selected

    namespace = {
        "get_args": lambda: args,
        "torch": torch,
        "weight_version_audit": audit,
        "itertools": itertools,
        "uuid": uuid,
        "partial": partial,
        "math": math,
        "Envs": Envs,
        "PackingMetrics": PackingMetrics,
        "logger": logging.getLogger(__name__),
        "timer": lambda *_args, **_kwargs: nullcontext(),
        "get_batch": cpu_packing,
        "get_model_config": lambda _: SimpleNamespace(cross_entropy_loss_fusion=False),
        "collect_manual_gc": collect_manual_gc,
        "capture_hooks": SimpleNamespace(begin_step_for=lambda *_args: None, end_step_for=lambda: None),
        "_attach_mtp_forward_kwargs": lambda *_args: None,
        "_unwrap_single_sequence_cp_output": lambda output, _batch: output,
        "should_bypass_main_output_layer": lambda _args: False,
        "get_forward_backward_func": lambda: schedule,
        "loss_function": loss,
        "_is_global_zero_token_step": lambda _losses: False,
        "maybe_verify_critic_value_head_movement": lambda *_args: None,
        "normalize_reduced_loss_metrics": lambda keys, values: dict(zip(keys, values[1:], strict=True)),
        "mpu": SimpleNamespace(
            get_context_parallel_world_size=lambda: 1,
            get_context_parallel_rank=lambda: cp_rank,
            get_tensor_model_parallel_rank=lambda: tp_rank,
            get_pipeline_model_parallel_world_size=lambda: 1,
            is_pipeline_last_stage=lambda **_kwargs: True,
            get_data_parallel_group=lambda **_kwargs: object(),
        ),
    }
    namespace["_weight_audit_counts"] = _load_definition(path, "_weight_audit_counts", namespace)
    train_one_step = _load_definition(path, "train_one_step", namespace)
    train_one_step(
        args,
        0,
        0,
        iterators,
        model,
        SimpleNamespace(zero_grad=lambda: None, step=lambda: (True, torch.tensor(1.0), 0)),
        SimpleNamespace(step=lambda **_kwargs: None),
        3,
        3,
    )
    summaries = list(directory.glob("*/consumed*.summary.json"))
    return json.loads(summaries[0].read_text()) if summaries else None


def test_train_step_consumes_final_mask_counts(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, final_mask=[1, 0])["eligible_tokens"] == 3


@pytest.mark.parametrize("failures,dummy", [((1,), False), ((0, 1, 2), False), ((0, 1, 2), True)])
def test_count_failure_marks_summary_incomplete(tmp_path, monkeypatch, failures, dummy):
    original_load = _load_definition
    calls = itertools.count()

    def load(path, name, namespace):
        definition = original_load(path, name, namespace)
        if name != "_weight_audit_counts":
            return definition

        def count(*args, **kwargs):
            if next(calls) in failures:
                raise RuntimeError("injected audit count failure")
            return definition(*args, **kwargs)

        return count

    monkeypatch.setitem(globals(), "_load_definition", load)
    summary = _run(tmp_path, monkeypatch, final_mask=[1, 0], dummy=dummy)
    if dummy:
        assert summary is None
    else:
        assert summary["eligible_tokens"] == 3 - len(failures)
        assert summary["aggregate_complete"] is False
        assert summary["audit_errors"] == len(failures)


def _counts_helper(loss_module, cp):
    path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/model.py"
    return _load_definition(
        path,
        "_weight_audit_counts",
        {
            "torch": torch,
            "mpu": loss_module.mpu,
            "weight_version_audit": audit,
            "get_logits_and_tokens_offset_with_cp": cp.get_logits_and_tokens_offset_with_cp,
        },
    )


@pytest.mark.parametrize("reject_all,opd", [(False, False), (True, False), (False, True)])
def test_real_policy_rs_mask_matches_audited_exposure(tmp_path, monkeypatch, reject_all, opd):
    from tests.cpu.backends.megatron.test_score_centering_loss import _args, _backend

    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    loss_module, cp = _backend(monkeypatch)
    args = _args(
        advantage_estimator="grpo",
        use_tis=True,
        custom_tis_function_path="fixture.reject",
        calculate_per_token_loss=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        entropy_coef=0.0,
        opd_token_selection="student_sampled",
        opd_loss_coef=0.5 if opd else 0,
        opd_kl_type="reverse_kl",
        opd_jsd_alpha=0.5,
    )

    def reject(**kwargs):
        masks = [torch.tensor([0 if reject_all else 1, 0])]
        return kwargs["pg_loss"], masks, {}

    monkeypatch.setattr(loss_module, "load_function", lambda _path: reject)
    publisher = audit.PublicationLedger(str(tmp_path), "policy")
    spans = []
    for i in range(2):
        version = publisher.publish()
        spans += audit.generation_spans(
            {"weight_versions": [{"start": 0, "end": 1, "version": version}]},
            1,
            offset=2 + i,
            request_id="r",
            attempt_id="a",
            piece_index=i,
        )
    row = audit.prepare_row({"spans": spans}, 4, 2, [1, 1])
    publisher.publish()
    logits = torch.zeros((1, 4, 4), requires_grad=True)
    batch = {
        "unconcat_tokens": [torch.tensor([0, 0, 1, 2])],
        "advantages": [torch.ones(2)],
        "log_probs": [torch.full((2,), -math.log(4))],
        "rollout_log_probs": [torch.full((2,), -math.log(4))],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        audit.PREPARED_FIELD: [row],
    }
    if opd:
        batch["teacher_log_probs"] = [torch.full((2,), -2.0)]
    reducer = cp.get_sum_of_sample_mean([4], [2], batch["loss_masks"])
    loss, _ = loss_module.policy_loss_function(args, batch, logits, reducer)
    loss.backward()
    assert batch["loss_masks"][0].tolist() == [1, 1]
    assert (logits.grad[0, 2].abs().sum() > 0).item() == opd
    step = audit.AuditStep("rs")
    counts = _counts_helper(loss_module, cp)(args, batch)
    step.observe(0, [row], counts=counts)
    summary = step.commit(0)
    assert summary["eligible_tokens"] == (2 if opd else 0 if reject_all else 1)
    assert summary["token_weighted_age"] == (1.5 if opd else None if reject_all else 2)


def _cp_audit_worker(rank, rendezvous):
    from datetime import timedelta

    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        mpu = SimpleNamespace(
            get_context_parallel_world_size=lambda: 2,
            get_context_parallel_rank=lambda: rank,
            get_context_parallel_group=lambda: torch.distributed.group.WORLD,
            get_dynamic_data_context_parallel_groups=lambda group_size: torch.distributed.group.WORLD,
        )
        path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/cp_utils.py"
        offsets = _load_definition(path, "get_logits_and_tokens_offset_with_cp", {"mpu": mpu})
        helper = _counts_helper(
            SimpleNamespace(mpu=mpu), SimpleNamespace(get_logits_and_tokens_offset_with_cp=offsets)
        )
        for dynamic in (False, True):
            for prompt in (0, 2):
                # A span crossing CP owners; masked final tokens on different ranks.
                row = {"spans": [{"token_start": 2, "token_end": 7}]}
                mask = torch.tensor([1, 0, 1, 1, 0, 1, 0][max(prompt, 1) - 1 :])
                batch = {
                    audit.PREPARED_FIELD: [row],
                    "weight_audit_loss_masks": [mask],
                    "total_lengths": [8],
                    "response_lengths": [8 - prompt],
                }
                if dynamic:
                    batch.update(dynamic_cp_size=2, dynamic_cp_rank=rank)
                counts = helper(Namespace(qkv_format="thd"), batch)
                assert counts.tolist() == ([4, 3] if prompt == 0 else [3, 3])
                owners = offsets(8, 8 - prompt, dynamic_cp_size=2, dynamic_cp_rank=rank)[3]
                batch["weight_audit_policy_mask"] = torch.cat(
                    [torch.arange(lo, hi) != 4 for lo, hi in owners if lo < hi]
                )
                counts = helper(Namespace(qkv_format="thd"), batch)
                assert counts.tolist() == ([3, 2] if prompt == 0 else [2, 2])
                batch["loss_masks"] = [mask]
                batch["weight_audit_loss_masks"] = [torch.zeros_like(mask)]
                args = Namespace(
                    qkv_format="thd", enable_mtp_training=True, mtp_num_layers=2, mtp_loss_scaling_factor=0.1
                )
                assert helper(args, batch).tolist() == [3, 3]
                batch["weight_audit_opd"] = True
                assert helper(args, batch).tolist() == ([4, 3] if prompt == 0 else [3, 3])
    finally:
        torch.distributed.destroy_process_group()


def test_final_mask_counts_across_real_cp_group(tmp_path):
    torch.multiprocessing.spawn(_cp_audit_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


@pytest.mark.parametrize("auxiliary", ["none", "entropy", "kl", "opd"])
def test_opsm_audit_counts_union_of_active_losses(tmp_path, monkeypatch, auxiliary):
    from tests.cpu.backends.megatron.test_score_centering_loss import _args, _backend

    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    loss_module, cp = _backend(monkeypatch)
    args = _args(
        advantage_estimator="grpo",
        use_opsm=True,
        opsm_delta=0.01,
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
    )
    publisher = audit.PublicationLedger(str(tmp_path), "policy")
    spans = []
    for i in range(2):
        tag = publisher.publish()
        spans += audit.generation_spans(
            {"weight_versions": [{"start": 0, "end": 1, "version": tag}]},
            1,
            offset=2 + i,
            request_id="r",
            attempt_id="a",
            piece_index=i,
        )
    publisher.publish()
    row = audit.prepare_row({"spans": spans}, 4, 2, [1, 1])
    batch = {
        "unconcat_tokens": [torch.tensor([0, 0, 1, 2])],
        "advantages": [torch.tensor([-1.0, 1.0])],
        "log_probs": [torch.full((2,), -math.log(4) + 0.05)],
        "rollout_log_probs": [torch.full((2,), -math.log(4) + 0.05)],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "teacher_log_probs": [torch.full((2,), -2.0)],
        "ref_log_probs": [torch.full((2,), -2.0)],
    }
    gradients = []
    for collect in (False, True):
        if collect:
            batch[audit.PREPARED_FIELD] = [row]
        logits = torch.zeros((1, 4, 4), requires_grad=True)
        reducer = cp.get_sum_of_sample_mean([4], [2], batch["loss_masks"])
        loss, _ = loss_module.policy_loss_function(args, batch, logits, reducer)
        loss.backward()
        gradients.append(logits.grad.clone())
    torch.testing.assert_close(gradients[0], gradients[1])
    if auxiliary == "none":
        assert gradients[1][0, 1].abs().sum() == 0
        assert gradients[1][0, 2].abs().sum() > 0
    step = audit.AuditStep("opsm")
    step.observe(0, [row], counts=_counts_helper(loss_module, cp)(args, batch))
    summary = step.commit(0)
    assert summary["eligible_tokens"] == (1 if auxiliary == "none" else 2)
    assert summary["token_weighted_age"] == (1 if auxiliary == "none" else 1.5)
    assert batch["loss_masks"][0].tolist() == [1, 1]


@pytest.mark.parametrize("auxiliary", ["none", "entropy", "kl", "opd", "mtp", "opsm"])
def test_icepop_audit_counts_union_of_active_losses(tmp_path, monkeypatch, auxiliary):
    from tests.cpu.backends.megatron.test_score_centering_loss import _args, _backend

    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    loss_module, cp = _backend(monkeypatch)
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
    row = audit.prepare_row({"spans": [{"token_start": 2, "token_end": 4}]}, 4, 2, [1, 1])
    batch = {
        "unconcat_tokens": [torch.tensor([0, 0, 1, 2])],
        "advantages": [torch.ones(2)],
        "log_probs": [torch.full((2,), -math.log(4))],
        "rollout_log_probs": [torch.tensor([-math.log(4) - math.log(10), -math.log(4)])],
        "ref_log_probs": [torch.full((2,), -2.0)],
        "teacher_log_probs": [torch.full((2,), -2.0)],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
    }
    if auxiliary == "opsm":
        batch["advantages"] = [torch.tensor([1.0, -1.0])]
        batch["log_probs"] = [batch["log_probs"][0] + 0.05]
    gradients = []
    for collect in (False, True):
        if collect:
            batch[audit.PREPARED_FIELD] = [row]
        logits = torch.zeros((1, 4, 4), requires_grad=True)
        reducer = cp.get_sum_of_sample_mean([4], [2], batch["loss_masks"])
        loss, _ = loss_module.policy_loss_function(args, batch, logits, reducer)
        loss.backward()
        gradients.append(logits.grad.clone())
    torch.testing.assert_close(*gradients)
    if auxiliary == "none":
        assert gradients[1][0, 1].abs().sum() == 0
        assert gradients[1][0, 2].abs().sum() > 0
    counts = _counts_helper(loss_module, cp)(args, batch)
    step = audit.AuditStep("icepop")
    step.observe(0, [row], counts=counts)
    assert step.commit(0)["eligible_tokens"] == (0 if auxiliary == "opsm" else 1 if auxiliary == "none" else 2)
    assert batch["loss_masks"][0].tolist() == [1, 1]


@pytest.mark.parametrize("prompt", [1, 2])
def test_plain_mtp_initializes_audit_target_counts(tmp_path, monkeypatch, prompt):
    from tests.cpu.backends.megatron.test_score_centering_loss import _args, _backend

    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    loss_module, cp = _backend(monkeypatch)
    args = _args(
        advantage_estimator="grpo",
        enable_mtp_training=True,
        mtp_num_layers=2,
        mtp_loss_scaling_factor=0.1,
        opd_token_selection="student_sampled",
        opd_loss_coef=0.0,
        eps_clip=0.2,
        eps_clip_high=0.2,
        entropy_coef=0.0,
    )
    mask = torch.tensor([1, 0, 1])
    total = prompt + len(mask)
    row = audit.prepare_row(
        {"spans": [{"token_start": position, "token_end": position + 1} for position in range(prompt, total)]},
        total,
        len(mask),
        mask.tolist(),
    )
    batch = {
        "unconcat_tokens": [torch.arange(total) % 4],
        "advantages": [torch.ones(len(mask))],
        "log_probs": [torch.full((len(mask),), -math.log(4))],
        "total_lengths": [total],
        "response_lengths": [len(mask)],
        "loss_masks": [mask],
        audit.PREPARED_FIELD: [row],
    }
    reducer = cp.get_sum_of_sample_mean([total], [len(mask)], batch["loss_masks"])
    loss_module.policy_loss_function(args, batch, torch.zeros((1, total, 4)), reducer)
    counts = _counts_helper(loss_module, cp)(args, batch)
    policy_targets = {position for position in range(prompt, total) if mask[position - prompt]}
    mtp_targets = {
        position + depth + 1
        for depth in (1, 2)
        for position in range(total)
        if prompt <= position + depth + 1 < total and mask[position + depth + 1 - prompt]
    }
    targets = policy_targets | mtp_targets
    assert mtp_targets
    assert counts is not None
    assert counts.tolist() == [len(targets), *(int(position in targets) for position in range(prompt, total))]
    step = audit.AuditStep("plain-mtp")
    step.observe(0, batch[audit.PREPARED_FIELD], counts=counts)
    assert step.commit(0)["eligible_tokens"] == len(targets)
    assert batch["loss_masks"][0].tolist() == [1, 0, 1]


@pytest.mark.parametrize("prompt,response,scale", [(2, 3, 0.1), (1, 3, 0.1), (1, 1, 0.1), (2, 3, 0.0)])
def test_rs_rejection_retains_mtp_target_coverage(monkeypatch, prompt, response, scale):
    from tests.cpu.backends.megatron.test_score_centering_loss import _backend

    loss, cp = _backend(monkeypatch)
    args = Namespace(qkv_format="thd", enable_mtp_training=True, mtp_num_layers=2, mtp_loss_scaling_factor=scale)
    row = audit.prepare_row(
        {"spans": [{"token_start": prompt, "token_end": prompt + response}]},
        prompt + response,
        response,
        [1] * response,
    )
    batch = {
        audit.PREPARED_FIELD: [row],
        "weight_audit_loss_masks": [torch.zeros(response)],
        "loss_masks": [torch.ones(response)],
        "total_lengths": [prompt + response],
        "response_lengths": [response],
    }
    counts = _counts_helper(loss, cp)(args, batch)
    # Enumerate target identities reachable by future-token heads inside this document.
    targets = (
        {
            position + depth + 1
            for depth in (1, 2)
            for position in range(prompt + response)
            if prompt <= position + depth + 1 < prompt + response
        }
        if scale
        else set()
    )
    assert counts.tolist() == [len(targets), len(targets)]


def test_unavailable_multimodal_mtp_coverage_marks_report_incomplete(tmp_path, monkeypatch):
    from tests.cpu.backends.megatron.test_score_centering_loss import _backend

    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    loss, cp = _backend(monkeypatch)
    args = Namespace(
        qkv_format="thd", enable_mtp_training=True, mtp_num_layers=1, mtp_loss_scaling_factor=0.1, is_vl_model=True
    )
    batch = {
        audit.PREPARED_FIELD: [audit.prepare_row({}, 4, 2, [1, 1])],
        "weight_audit_loss_masks": [torch.zeros(2)],
        "loss_masks": [torch.ones(2)],
        "total_lengths": [4],
        "response_lengths": [2],
    }
    counts = _counts_helper(loss, cp)(args, batch)
    step = audit.AuditStep("multimodal-mtp")
    step.observe(0, batch[audit.PREPARED_FIELD], counts=counts)
    summary = step.commit(0)
    assert summary["aggregate_complete"] is False
    assert summary["audit_errors"] == 1
    assert step.observed == {}
