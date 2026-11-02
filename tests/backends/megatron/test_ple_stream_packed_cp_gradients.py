# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Apply the packed CP/SP ownership contract to the opt-in streamed PLE
forward."""

import ast
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from tests.backends.megatron import test_ple_packed_cp_gradients as contract


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_streamed_packed_cp_preserves_hidden_gradients(monkeypatch, tp_size):
    load_native = contract._load_ple

    def load_streamed(parallel_state):
        cls = load_native(parallel_state)
        native_forward = cls.forward
        namespace = {
            "torch": torch,
            "Tensor": torch.Tensor,
            "F": F,
            "math": math,
            "parallel_state": parallel_state,
            "checkpoint": checkpoint,
            "PerLayerEmbedding": cls,
        }
        hyper = ast.parse(contract._added_source("megatron/core/transformer/hyper_connection.py"))
        norm = next(
            node
            for node in hyper.body
            if isinstance(node, ast.FunctionDef) and node.name == "gated_residual_group_rmsnorm"
        )
        source = Path(__file__).resolve().parents[3] / "relax/models/qwen3_8/ops/ple_stream.py"
        functions = [
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_stream_conv", "_stream_forward", "_streamed_forward"}
        ]
        exec(compile(ast.Module(body=[norm, *functions], type_ignores=[]), str(source), "exec"), namespace)

        def forward(self, hidden):
            if self._cp_packed_seq_params is None:
                return native_forward(self, hidden)
            return namespace["_streamed_forward"](self, hidden)

        cls.forward = forward
        return cls

    monkeypatch.setattr(contract, "_load_ple", load_streamed)
    contract.test_packed_cp_sp_preserves_input_and_parameter_gradients(monkeypatch, tp_size, frozen=True)
