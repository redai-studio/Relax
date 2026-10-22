# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Acceptance reporting must not reduce or rescale current MCore losses
twice."""

import ast
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("total,expected", [(8.0, 0.75), (0.0, None)])
@pytest.mark.parametrize("legacy_raw_ce", [False, True])
def test_current_tracker_is_reduced_once_and_loss_keeps_microbatch_average(
    monkeypatch, total, expected, legacy_raw_ce
):
    calls = []

    class Helper:
        tracker = {
            "loss_values": torch.tensor([1.6]),
            "correct_values": torch.tensor([total * 0.75]),
            "total_values": torch.tensor([total]),
        }

        @staticmethod
        def reduce_metrics_in_tracker():
            calls.append("reduce")

        @staticmethod
        def clean_loss_in_tracker():
            calls.append("clean")
            for value in Helper.tracker.values():
                value.zero_()

    module = types.ModuleType("megatron.core.transformer.multi_token_prediction")
    module.MTPLossLoggingHelper = Helper
    monkeypatch.setitem(sys.modules, module.__name__, module)
    if legacy_raw_ce:
        Helper.tracker["relax_mtp_microbatch_raw_ce"] = torch.tensor([3.0])

    def reduce_legacy(value, group, op):
        assert group == "dp" and op == torch.distributed.ReduceOp.AVG
        calls.append("legacy_reduce")

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_legacy)
    path = Path(__file__).resolve().parents[3] / "relax/backends/megatron/model.py"
    source = path.read_text()
    start = source.index("        mtp_losses = None\n")
    end = source.index("        # per train step log.", start)
    scope = {
        "torch": torch,
        "num_microbatches": [2],
        "step_id": 0,
        "mpu": types.SimpleNamespace(get_data_parallel_group=lambda **kwargs: "dp"),
        "args": types.SimpleNamespace(
            enable_mtp_training=True, ci_test=False, calculate_per_token_loss=True, mtp_loss_scaling_factor=0.1
        ),
    }
    helper = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_mtp_metric_scales"
    )
    scope["Namespace"] = types.SimpleNamespace
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), scope)
    exec(compile(textwrap.dedent(source[start:end]), str(path), "exec"), scope)
    assert calls == (["legacy_reduce"] if legacy_raw_ce else []) + ["reduce", "clean"]
    assert scope["mtp_losses"] == pytest.approx(0.8)
    assert scope["mtp_acceptance"] == expected
    assert scope["mtp_legacy_per_depth"] == ([1.5] if legacy_raw_ce else None)
    assert "relax_mtp_microbatch_raw_ce" not in Helper.tracker


def test_qwen38_tracker_preserves_legacy_microbatch_mean_without_rescaling_objective():
    path = Path(__file__).resolve().parents[3] / "relax/models/common/chunked_mtp_loss.py"
    function = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "_save_mtp_loss_to_tracker"
    )
    objective_calls = []
    helper = types.SimpleNamespace(
        tracker={}, save_loss_to_tracker=lambda *args, **kwargs: objective_calls.append((args, kwargs))
    )
    scope = {
        "torch": torch,
        "MTPLossLoggingHelper": helper,
        "parallel_state": types.SimpleNamespace(
            get_data_parallel_group=lambda **kwargs: "dp", get_context_parallel_world_size=lambda: 1
        ),
        "_MTP_TRACKER_PARAMETERS": {"num_tokens"},
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    config = types.SimpleNamespace(
        mhc_variant="gated_residual", mtp_num_layers=1, mtp_loss_scaling_factor=0.1, calculate_per_token_loss=True
    )
    save = scope["_save_mtp_loss_to_tracker"]
    save(torch.tensor([1.0, 3.0]), torch.tensor(2), 1, config)
    save(torch.ones(6), torch.tensor(6), 1, config)
    # Baseline reporting: (2 + 1) / 2 = 1.5, not token-weighted CE 10 / 8 = 1.25.
    assert helper.tracker["relax_mtp_microbatch_raw_ce"].item() / 2 == pytest.approx(1.5)
    # The runtime's scaled objective tracker remains unchanged.
    assert objective_calls[0][0][0].item() == pytest.approx(0.4)
    assert objective_calls[1][0][0].item() == pytest.approx(0.6)
