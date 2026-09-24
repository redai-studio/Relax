# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise streaming schedule finalization without importing training
dependencies."""

from __future__ import annotations

import ast
import contextlib
import sys
import types
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "schedule_name",
    [
        "streaming_forward_backward_no_pipelining",
        "streaming_forward_backward_pipelining_without_interleaving",
    ],
)
def test_streaming_schedules_finalize_with_tensor_data_context_group(monkeypatch, schedule_name):
    """New Megatron finalization needs the combined TP/DP/CP reduction
    group."""
    source = Path(__file__).resolve().parents[3] / "relax/backends/megatron/streaming_schedules.py"
    tree = ast.parse(source.read_text())
    schedule = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == schedule_name)
    # Execute the real schedule body with CPU doubles for CUDA/distributed operations.
    # This avoids importing the backend package and initializing its optional runtime.
    module = types.ModuleType("megatron.core.pipeline_parallel.schedules")
    module.ProcessGroupCollection = types.SimpleNamespace
    monkeypatch.setitem(sys.modules, module.__name__, module)

    tp, dp_cp, tp_dp_cp = object(), object(), object()
    cp = types.SimpleNamespace(size=lambda: 2)
    calls = {"forward": 0, "backward": 0, "finalize": [], "tp_dp_cp": []}

    def combined_group(**kwargs):
        calls["tp_dp_cp"].append(kwargs)
        return tp_dp_cp

    parallel_state = types.SimpleNamespace(
        get_tensor_model_parallel_group=lambda: tp,
        get_context_parallel_group=lambda: cp,
        get_embedding_group=lambda **kwargs: object(),
        get_position_embedding_group=lambda **kwargs: object(),
        get_pipeline_model_parallel_group=lambda: object(),
        get_data_parallel_group=lambda **kwargs: dp_cp,
        get_tensor_and_data_parallel_group=combined_group,
        get_data_parallel_rank=lambda: 0,
    )

    def finalize(models, num_tokens, *, pg_collection, force_all_reduce):
        calls["finalize"].append((models, num_tokens, pg_collection, force_all_reduce))

    config = types.SimpleNamespace(
        no_sync_func=None,
        finalize_model_grads_func=finalize,
        calculate_per_token_loss=True,
        overlap_p2p_comm=False,
        grad_sync_func=None,
        deallocate_pipeline_outputs=False,
    )
    communicator = types.SimpleNamespace(
        total_stages=2,
        current_stage=1,
        is_pp_first_stage=False,
        is_pp_last_stage=True,
        pp_group=object(),
        recv_forward=lambda *args: None,
        send_forward_recv_backward=lambda *args: None,
        send_backward_recv_forward=lambda *args: None,
        send_backward=lambda *args: None,
    )

    def forward_step(forward_fn, iterator, model, count, inputs, store, config, **kwargs):
        assert count == 1
        assert kwargs["cp_group_size"] == 2
        store.append(next(iterator))
        calls["forward"] += 1
        return types.SimpleNamespace(item=lambda: 1.0, numel=lambda: 1), 7

    def backward_step(*args):
        calls["backward"] += 1

    namespace = {
        "contextlib": contextlib,
        "torch": types.SimpleNamespace(zeros=lambda *args, **kwargs: 0, int=int),
        "parallel_state": parallel_state,
        "get_model_config": lambda model: config,
        "P2PCommunicator": lambda **kwargs: communicator,
        "clear_embedding_activation_buffer": lambda *args: None,
        "finish_embedding_wgrad_compute": lambda *args: None,
        "get_tensor_shapes": lambda **kwargs: [],
        "deallocate_output_tensor": lambda *args: None,
        "check_first_val_step": lambda *args: False,
        "forward_step": forward_step,
        "backward_step": backward_step,
        "logger": types.SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    }
    exec(compile(ast.Module(body=[schedule], type_ignores=[]), str(source), "exec"), namespace)
    model = object()
    result = namespace[schedule_name](
        forward_step_func=None,
        data_iterator=iter(["first", "second"]),
        model=model,
        num_microbatches=99,
        seq_length=8,
        micro_batch_size=1,
        force_all_reduce=True,
    )

    assert result == ["first", "second"]
    assert calls["forward"] == calls["backward"] == 2
    assert calls["tp_dp_cp"] == [{"with_context_parallel": True}]
    assert len(calls["finalize"]) == 1
    models, num_tokens, groups, force_all_reduce = calls["finalize"][0]
    assert models == [model]
    assert num_tokens == 14
    assert force_all_reduce is True
    assert groups.tp is tp
    assert groups.cp is cp
    assert groups.dp_cp is dp_cp
    assert groups.tp_dp_cp is tp_dp_cp
