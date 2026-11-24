# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Apply the packed CP/SP ownership contract to the opt-in streamed PLE
forward."""

import importlib.util
from pathlib import Path

import pytest

from tests.cpu.backends.megatron import test_ple_packed_cp_gradients as contract


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_streamed_packed_cp_preserves_hidden_gradients(monkeypatch, tp_size):
    load_native = contract._load_ple

    def load_streamed(parallel_state, monkeypatch):
        cls = load_native(parallel_state, monkeypatch)
        native_forward = cls.forward
        source = Path(__file__).resolve().parents[4] / "relax/models/qwen3_8/ops/ple_stream.py"
        spec = importlib.util.spec_from_file_location("ple_stream_contract", source)
        streamed = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(streamed)
        monkeypatch.setattr(streamed, "parallel_state", parallel_state)
        monkeypatch.setattr(streamed, "_STREAM_CHUNK_ROWS", 3)

        def forward(self, hidden):
            if self._cp_packed_seq_params is None:
                return native_forward(self, hidden)
            return streamed._streamed_forward(self, hidden)

        cls.forward = forward
        return cls

    monkeypatch.setattr(contract, "_load_ple", load_streamed)
    contract.test_packed_cp_sp_preserves_input_and_parameter_gradients(monkeypatch, tp_size, frozen=True)
