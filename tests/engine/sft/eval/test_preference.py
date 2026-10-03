# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from relax.engine.sft.eval.preference import (
    compute_reward_model_eval_step,
    finalize_pair_metrics,
    pair_metric_sums,
    preference_eval_chunk_sizes,
    preference_eval_local_batch_sizes,
)


@pytest.mark.parametrize(
    ("pair_count", "global_batch_size", "dp_size", "expected"),
    [(1, 4, 1, [1]), (6, 4, 2, [2, 1]), (10, 4, 2, [2, 2, 1]), (512, 30, 2, [15] * 17 + [1])],
)
def test_preference_eval_chunks_preserve_actual_pair_count(pair_count, global_batch_size, dp_size, expected):
    assert preference_eval_local_batch_sizes(pair_count, global_batch_size, dp_size) == expected
    chunks = preference_eval_chunk_sizes(pair_count, global_batch_size)
    assert chunks == [size * dp_size for size in expected]
    assert sum(chunks) == pair_count


def test_preference_eval_rejects_chunks_that_cannot_be_split_across_dp():
    with pytest.raises(ValueError, match="divisible by data-parallel size"):
        preference_eval_local_batch_sizes(5, 4, dp_size=2)


def test_reward_model_eval_emits_one_score_per_branch_for_order_restoration():
    _, outputs = compute_reward_model_eval_step(
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
        total_lengths=[2, 2],
        score_positions=[1, 1],
    )

    assert len(outputs["scores"]) == 2
    assert all(score.ndim == 0 for score in outputs["scores"])
    assert torch.stack(outputs["scores"]).tolist() == [1.0, 3.0]


def test_pair_metric_sums_and_finalize_keep_ties_explicit():
    chosen = torch.tensor([2.0, 1.0, 1.0])
    rejected = torch.tensor([1.0, 2.0, 1.0])
    losses = torch.tensor([0.1, 0.2, 0.3])

    metrics = finalize_pair_metrics(pair_metric_sums(chosen, rejected, losses), prefix="rm")

    assert metrics["eval/rm_loss"] == pytest.approx(0.2)
    assert metrics["eval/rm_strict_accuracy"] == pytest.approx(1 / 3)
    assert metrics["eval/rm_tie_rate"] == pytest.approx(1 / 3)
    assert metrics["eval/rm_tie_aware_accuracy"] == pytest.approx(0.5)
    assert metrics["eval/rm_pairs"] == 3


@pytest.mark.parametrize("reference_free", [False, True])
@pytest.mark.parametrize("chosen_completion_length", [1, 2])
@pytest.mark.parametrize("dummy_log_prob", [-0.01, -100.0])
def test_dpo_eval_scores_only_completion_tokens(monkeypatch, reference_free, chosen_completion_length, dummy_log_prob):
    # Execute the actual runner without importing the GPU backend. Only the
    # forward pass, data transport, and distributed/logging boundaries are replaced.
    source = Path(__file__).resolve().parents[4] / "relax/engine/sft/eval/runner.py"
    functions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in {"_run_preference_eval", "_masked_sequence_sums"}
    ]
    masks = [[0, 0] + [1] * chosen_completion_length, torch.tensor([0, 0, 1, 1, 1])]
    original_masks = [torch.as_tensor(mask).clone() for mask in masks]
    rollout_data = {"loss_masks": masks}
    policy = [
        torch.tensor([-99.0, *[-1.0, -2.0][:chosen_completion_length], dummy_log_prob]),
        torch.tensor([-99.0, -3.0, -4.0, -5.0, -60.0]),
    ]
    reference = [
        torch.tensor([-99.0, *[-2.0, -3.0][:chosen_completion_length], -70.0]),
        torch.tensor([-99.0, -4.0, -5.0, -6.0, -80.0]),
    ]
    for name, members in {
        "data": {
            "expand_preference_rollout_data": lambda rows: rows,
            "get_data_iterator": lambda *_: (None, None),
        },
        "initialize": {"is_megatron_main_rank": lambda: True},
        "model": {"forward_only": None},
    }.items():
        module_name = f"relax.backends.megatron.{name}"
        module = ModuleType(module_name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, module_name, module)

    logged = {}
    namespace = {
        "torch": torch,
        "time": time,
        "mpu": SimpleNamespace(
            get_data_parallel_world_size=lambda **_: 1,
            get_pipeline_model_parallel_group=lambda: None,
            get_data_parallel_group=lambda **_: None,
        ),
        "dist": SimpleNamespace(
            barrier=lambda **_: None,
            get_rank=lambda: 0,
            all_reduce=lambda *_, **__: None,
            ReduceOp=SimpleNamespace(SUM=None),
        ),
        "device_utils": SimpleNamespace(make_current_torch_device=lambda: torch.device("cpu")),
        "_wait_for_eval_plan": lambda *_: (1, 1),
        "_wait_for_eval_partition_present": lambda *_: None,
        "get_gloo_group": lambda: None,
        "run": lambda result: result,
        "timer": lambda *_: nullcontext(),
        "compute_rollout_step": lambda *_: 1,
        "tracking_utils": SimpleNamespace(
            log=lambda _args, metrics, **_: logged.update(metrics), flush_metrics=lambda *_: None
        ),
        "logger": SimpleNamespace(info=lambda *_: None),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    actor = SimpleNamespace(
        args=SimpleNamespace(
            sft_objective="dpo", dpo_reference_free=reference_free, dpo_beta=0.1, global_batch_size=2
        ),
        model=None,
        _get_data_from_transfer_queue=lambda *_, **__: (rollout_data, None),
        _switch_model=lambda _: None,
        compute_log_prob=lambda *_, store_prefix: {f"{store_prefix}log_probs": reference if store_prefix else policy},
        data_system_client=SimpleNamespace(async_clear_partition=lambda **_: None),
    )
    namespace["_run_preference_eval"](actor, rollout_id=0)

    # Hand-computed sums include the first completion token and exclude the
    # final dummy label, for both one-token and longer completions.
    policy_chosen = -1.0 if chosen_completion_length == 1 else -3.0
    reference_chosen = -2.0 if chosen_completion_length == 1 else -5.0
    chosen_reward = 0.1 * (policy_chosen if reference_free else policy_chosen - reference_chosen)
    rejected_reward = -1.2 if reference_free else 0.3
    margin = chosen_reward - rejected_reward
    assert logged["eval/dpo_chosen"] == pytest.approx(chosen_reward)
    assert logged["eval/dpo_rejected"] == pytest.approx(rejected_reward)
    assert logged["eval/dpo_margin"] == pytest.approx(margin)
    assert logged["eval/dpo_loss"] == pytest.approx(math.log1p(math.exp(-margin)))
    assert logged["eval/dpo_strict_accuracy"] == float(margin > 0)
    assert logged["eval/dpo_tie_rate"] == 0.0
    assert logged["eval/dpo_pairs"] == 1
    for actual, original in zip(rollout_data["loss_masks"], original_masks, strict=True):
        torch.testing.assert_close(torch.as_tensor(actual), original)
