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
