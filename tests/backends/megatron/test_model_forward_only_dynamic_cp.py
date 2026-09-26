# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for dynamic-CP output handling in forward-only passes."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("megatron.training.global_vars")


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


def test_forward_only_does_not_merge_dynamic_cp_aggregate_outputs(monkeypatch):
    """Per-microbatch aggregates must not enter per-sample CP collectives."""
    from relax.backends.megatron import cp_utils
    from relax.backends.megatron import model as model_module

    args = Namespace(
        allgather_cp=False,
        custom_megatron_before_log_prob_hook_path=None,
        data_pad_size_multiplier=1,
        dynamic_context_parallel=True,
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
