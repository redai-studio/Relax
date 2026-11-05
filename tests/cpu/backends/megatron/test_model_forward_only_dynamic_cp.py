# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for dynamic-CP output handling in forward-only passes."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch


class _FakeIterator:
    micro_batch_indices = [[0]]

    def reset(self) -> None:
        pass


class _FakeModel:
    def __init__(self) -> None:
        self.training = True

    def __call__(self, **_) -> torch.Tensor:
        return torch.tensor([0.0])

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


@pytest.mark.parametrize("memory_level", [0, 1, 2])
def test_forward_only_does_not_merge_dynamic_cp_aggregate_outputs(monkeypatch, memory_level):
    """Per-microbatch aggregates must not enter per-sample CP collectives."""
    model_module = pytest.importorskip("relax.backends.megatron.model")
    cp_utils = pytest.importorskip("relax.backends.megatron.cp_utils")
    events = []
    monkeypatch.setattr(
        model_module,
        "device_module",
        SimpleNamespace(empty_cache=lambda: events.append("release")),
        raising=False,
    )

    args = Namespace(
        allgather_cp=False,
        custom_megatron_before_log_prob_hook_path=None,
        data_pad_size_multiplier=1,
        dynamic_context_parallel=True,
        empty_unused_memory_level=memory_level,
        is_vl_model=False,
        micro_batch_size=1,
        qkv_format="thd",
        seq_length=8,
        use_dynamic_batch_size=True,
        use_rollout_entropy=False,
        uses_unsplit_forward=False,
    )
    batch = {
        "dynamic_cp_rank": 0,
        "dynamic_cp_size": 2,
        "full_loss_masks": None,
        "loss_masks": [torch.ones(2)],
        "max_seq_lens": None,
        "packed_seq_params": None,
        "padded_total_lengths": None,
        "response_lengths": [2],
        "tokens": torch.tensor([[1, 2, 3]]),
        "total_lengths": [3],
        "unconcat_tokens": [torch.tensor([1, 2, 3])],
    }
    iterator = _FakeIterator()
    fake_model = _FakeModel()

    monkeypatch.setattr(model_module, "get_batch", lambda *_, **__: batch)
    monkeypatch.setattr(model_module, "get_model_config", lambda _: SimpleNamespace(timers="unused"))
    monkeypatch.setattr(model_module.mpu, "is_pipeline_last_stage", lambda: True)

    def fake_forward_backward_func(**kwargs):
        output, callback = kwargs["forward_step_func"](iterator, fake_model)
        _, result = callback(output)
        events.append("forward")
        return [result]

    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: fake_forward_backward_func)

    def fail_if_merged(*_, **__):
        raise AssertionError("aggregate output entered dynamic_cp_merge_output")

    monkeypatch.setattr(cp_utils, "dynamic_cp_merge_output", fail_if_merged)

    def aggregate_callback(logits, **_):
        return torch.empty((0,), device=logits.device), {
            "sum_neg_log_prob": [torch.tensor([5.0])],
            "num_tokens": [torch.tensor([2])],
        }

    result = model_module.forward_only(
        aggregate_callback,
        args,
        [fake_model],
        [iterator],
        [1],
        per_sample_output=False,
    )

    assert result["sum_neg_log_prob"][0].item() == 5.0
    assert result["num_tokens"][0].item() == 2
    assert fake_model.training is True
    assert events == ["forward"] + (["release"] if memory_level else [])
