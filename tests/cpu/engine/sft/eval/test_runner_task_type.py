# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.cpu.relax_parser import parser_for_flags


@pytest.mark.parametrize(
    ("argv", "callback_name", "metric_key"),
    [
        ([], "compute_sft_eval_step", "eval/ppl"),
        (["--task-type", "seq_cls"], "compute_classification_eval_step", "eval/accuracy"),
    ],
)
def test_runner_dispatches_by_parsed_task_type(monkeypatch, argv, callback_name, metric_key):
    args = parser_for_flags("--task-type", "--problem-type").parse_args(argv)
    args.loss_type = "sft"
    args.global_batch_size = 1

    forward = Mock(
        return_value={
            key: [torch.tensor(value)]
            for key, value in dict(loss_sum=2.0, num_examples=1, correct=1, sum_neg_log_prob=2.0, num_tokens=2).items()
        }
    )
    for name, members in {
        "relax.backends.megatron.data": {"get_data_iterator": lambda *_: (None, 1)},
        "relax.backends.megatron.initialize": {"is_megatron_main_rank": lambda: True},
        "relax.backends.megatron.model": {"forward_only": forward},
        "relax.utils.data.image_refs": {"sft_multimodal_data_fields": lambda _: []},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)

    source = Path(__file__).resolve().parents[5] / "relax/engine/sft/eval/runner.py"
    runner = next(
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "run_sft_eval"
    )
    tracker = Mock()
    namespace = {
        "torch": torch,
        "time": time,
        "mpu": SimpleNamespace(
            get_data_parallel_world_size=lambda **_: 1,
            get_pipeline_model_parallel_world_size=lambda: 1,
            get_data_parallel_group=lambda **_: None,
            is_pipeline_last_stage=lambda: True,
        ),
        "dist": SimpleNamespace(barrier=lambda **_: None, get_rank=lambda: 0, get_world_size=lambda _: 1),
        "device_utils": SimpleNamespace(make_current_torch_device=lambda: torch.device("cpu")),
        "_wait_for_eval_plan": lambda *_: (1, 1),
        "_wait_for_eval_partition_present": lambda *_: None,
        "get_gloo_group": lambda: None,
        "timer": lambda *_: nullcontext(),
        "run": lambda result: result,
        "compute_rollout_step": lambda *_: 1,
        "tracking_utils": tracker,
        "logger": Mock(),
    }
    exec(compile(ast.Module(body=[runner], type_ignores=[]), str(source), "exec"), namespace)
    fetch = Mock(return_value=({}, None))
    actor = SimpleNamespace(
        args=args,
        model=[],
        all_consumed=Mock(side_effect=[False, True]),
        _get_data_from_transfer_queue=fetch,
        data_system_client=Mock(),
    )
    namespace["run_sft_eval"](actor, 3)

    expected_fields = ["tokens", "loss_masks", "total_lengths", "response_lengths"]
    if args.task_type == "seq_cls":
        expected_fields.extend(["classification_labels", "sample_weights"])
    assert fetch.call_args.args[2] == expected_fields
    assert forward.call_args.args[0].__name__ == callback_name
    metrics = tracker.log.call_args.args[1]
    assert metric_key in metrics
    actor.data_system_client.async_clear_partition.assert_called_once_with(partition_id="sft_eval_3_n1_0")
