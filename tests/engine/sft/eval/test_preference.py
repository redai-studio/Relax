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
from relax.engine.sft.runtime import is_dpo_mode
from relax.utils.training.preference_utils import masked_sequence_sums, require_tensor_condition


@pytest.mark.parametrize(
    ("pair_count", "global_batch_size", "dp_size", "expected"),
    [(1, 4, 1, [1]), (6, 4, 2, [2, 1]), (10, 4, 2, [2, 2, 1]), (512, 30, 2, [15] * 17 + [1])],
)
def test_preference_eval_chunks_preserve_actual_pair_count(pair_count, global_batch_size, dp_size, expected):
    chunks = preference_eval_chunk_sizes(pair_count, global_batch_size)
    assert preference_eval_local_batch_sizes(chunks, dp_size) == expected
    assert chunks == [size * dp_size for size in expected]
    assert sum(chunks) == pair_count


def test_preference_eval_rejects_chunks_that_cannot_be_split_across_dp():
    with pytest.raises(ValueError, match="divisible by data-parallel size"):
        preference_eval_local_batch_sizes(preference_eval_chunk_sizes(5, 4), dp_size=2)


def test_reward_model_eval_emits_one_score_per_branch_for_order_restoration():
    _, outputs = compute_reward_model_eval_step(
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
        total_lengths=[2, 2],
    )

    assert len(outputs["scores"]) == 2
    assert all(score.ndim == 0 for score in outputs["scores"])
    assert torch.stack(outputs["scores"]).tolist() == [1.0, 3.0]


@pytest.mark.parametrize("score_scale", [1.0, 5e-7])
def test_pair_metric_sums_and_finalize_keep_ties_explicit(score_scale):
    chosen = torch.tensor([2.0, 1.0, 1.0]) * score_scale
    rejected = torch.tensor([1.0, 2.0, 1.0]) * score_scale
    losses = torch.tensor([0.1, 0.2, 0.3])

    metrics = finalize_pair_metrics(pair_metric_sums(chosen, rejected, losses, epsilon=0.0), prefix="rm")

    assert metrics["eval/rm_loss"] == pytest.approx(0.2)
    assert metrics["eval/rm_strict_accuracy"] == pytest.approx(1 / 3)
    assert metrics["eval/rm_tie_rate"] == pytest.approx(1 / 3)
    assert metrics["eval/rm_tie_aware_accuracy"] == pytest.approx(0.5)
    assert metrics["eval/rm_pairs"] == 3


@pytest.mark.parametrize(
    ("margin", "strict", "ties", "tie_aware"),
    [
        (-2e-6, 0.0, 0.0, 0.0),
        (-1e-6, 0.0, 1.0, 0.5),
        (-5e-7, 0.0, 1.0, 0.5),
        (0.0, 0.0, 1.0, 0.5),
        (5e-7, 1.0, 1.0, 0.5),
        (1e-6, 1.0, 1.0, 0.5),
        (2e-6, 1.0, 0.0, 1.0),
    ],
)
def test_dpo_eval_accuracy_at_tie_boundaries(margin, strict, ties, tie_aware):
    chosen = torch.tensor([margin], dtype=torch.float64)
    metrics = finalize_pair_metrics(
        pair_metric_sums(chosen, torch.zeros_like(chosen), torch.ones_like(chosen)), prefix="dpo"
    )

    assert metrics["eval/dpo_strict_accuracy"] == strict
    assert metrics["eval/dpo_tie_rate"] == ties
    assert metrics["eval/dpo_tie_aware_accuracy"] == tie_aware


@pytest.mark.parametrize(("pair_count", "dp_size", "chunks"), [(1, 1, [1]), (6, 2, [4, 2])])
@pytest.mark.parametrize("reference_free", [False, True])
@pytest.mark.parametrize("chosen_completion_length", [1, 2])
@pytest.mark.parametrize("prompt_length", [2, 4980])
@pytest.mark.parametrize(
    ("cp_size", "cp_rank", "last_stage", "dynamic_cp"),
    [(1, 0, True, False), (16, 0, True, False), (16, 1, True, False), (16, 0, False, False), (16, 0, True, True)],
)
def test_dpo_eval_scores_only_completion_tokens(
    monkeypatch,
    reference_free,
    chosen_completion_length,
    pair_count,
    dp_size,
    chunks,
    prompt_length,
    cp_size,
    cp_rank,
    last_stage,
    dynamic_cp,
):
    # Execute the actual runner without importing the GPU backend. Only the
    # forward pass, data transport, and distributed/logging boundaries are replaced.
    source = Path(__file__).resolve().parents[4] / "relax/engine/sft/eval/runner.py"
    functions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in {"_run_preference_eval", "_preference_eval_branch_sums"}
    ]
    masks = [[0] * prompt_length + [1] * chosen_completion_length, torch.tensor([0] * prompt_length + [1, 1, 1])]
    original_masks = [torch.as_tensor(mask).clone() for mask in masks]
    rollout_data = {"loss_masks": masks}
    policy = [
        torch.tensor([-99.0] * (prompt_length - 1) + [-1.0, -2.0][:chosen_completion_length] + [-100.0]),
        torch.tensor([-99.0] * (prompt_length - 1) + [-3.0, -4.0, -5.0, -60.0]),
    ]
    reference = [
        torch.tensor([-99.0] * (prompt_length - 1) + [-2.0, -3.0][:chosen_completion_length] + [-70.0]),
        torch.tensor([-99.0] * (prompt_length - 1) + [-4.0, -5.0, -6.0, -80.0]),
    ]
    consumed_batch_sizes = []
    iterator_batch_sizes = []

    def get_data_iterator(_args, _model, rows):
        iterator_batch_sizes.append(rows["dynamic_global_batch_size"])
        return None, None

    def all_reduce(values, *, group, **_):
        if group == "dp":
            values.mul_(dp_size)
        elif group == "cp":
            if values.dtype == torch.int32:
                values.fill_(1)
            else:
                values.copy_(reduce_branch_sums(values, group=group))

    for name, members in {
        "data": {
            "expand_preference_rollout_data": lambda rows: rows,
            "get_data_iterator": get_data_iterator,
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
        "is_dpo_mode": is_dpo_mode,
        "Namespace": SimpleNamespace,
        "Any": object,
        "Sequence": list,
        "RolloutBatch": dict,
        "masked_sequence_sums": masked_sequence_sums,
        "require_tensor_condition": require_tensor_condition,
        "torch": torch,
        "time": time,
        "mpu": SimpleNamespace(
            get_data_parallel_world_size=lambda **_: dp_size,
            get_pipeline_model_parallel_group=lambda: "pp",
            get_data_parallel_group=lambda **_: "dp",
            get_context_parallel_world_size=lambda: cp_size,
            get_context_parallel_rank=lambda: cp_rank,
            get_context_parallel_group=lambda: "cp",
            get_tensor_model_parallel_world_size=lambda: 1,
            is_pipeline_last_stage=lambda: last_stage,
        ),
        "dist": SimpleNamespace(
            barrier=lambda **_: None,
            get_rank=lambda: 0,
            all_reduce=all_reduce,
            ReduceOp=SimpleNamespace(SUM=None),
        ),
        "device_utils": SimpleNamespace(make_current_torch_device=lambda: torch.device("cpu")),
        "_wait_for_eval_plan": lambda *_: (len(chunks), pair_count),
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
    # Exercise the production CP mask slicing and branch reduction on CPU;
    # replace only the collective with sums independently computed from dense data.
    for filename, names in {
        "cp_utils.py": {
            "maybe_padded_total_lengths",
            "get_logits_and_tokens_offset_with_cp",
            "_slice_loss_mask_with_cp",
        },
        "loss.py": {"_preference_branch_sums", "_all_reduce_forward_only"},
    }.items():
        path = source.parents[3] / "backends/megatron" / filename
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        module_name = f"relax.backends.megatron.{path.stem}"
        module = ModuleType(module_name)
        module.__dict__.update({name: namespace[name] for name in names})
        monkeypatch.setitem(sys.modules, module_name, module)

    reductions = []
    forwards = []

    def reduce_branch_sums(values, *, group):
        assert group == "cp"
        full_values = reference if not reference_free and len(reductions) % 2 == 0 else policy
        expected = torch.stack(
            [value[prompt_length - 1 : -1].sum() for value in full_values] * consumed_batch_sizes[-1]
        )
        expected_local = []
        for value in full_values:
            chunk = math.ceil(len(value) / (2 * cp_size))
            positions = torch.arange(prompt_length - 1, len(value) - 1)
            owned = (positions // chunk == cp_rank) | (positions // chunk == 2 * cp_size - cp_rank - 1)
            expected_local.append(value[positions[owned]].sum())
        torch.testing.assert_close(values, torch.stack(expected_local * consumed_batch_sizes[-1]))
        reductions.append(values.clone())
        return expected

    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)

    def get_pair_rows(_task, _rollout_id, fields, batch_size, _batch_index, **_):
        assert fields == [
            "pair_ids",
            "chosen_tokens",
            "rejected_tokens",
            "chosen_loss_masks",
            "rejected_loss_masks",
            "chosen_total_lengths",
            "rejected_total_lengths",
        ]
        consumed_batch_sizes.append(batch_size)
        rollout_data["loss_masks"] = masks * batch_size
        rollout_data["total_lengths"] = [len(mask) for mask in masks] * batch_size
        rollout_data["response_lengths"] = rollout_data["total_lengths"]
        return rollout_data, None

    def compute_log_prob(*_, store_prefix):
        forwards.append(store_prefix)
        if not last_stage:
            return {}
        values = reference if store_prefix else policy
        if cp_size > 1 and not dynamic_cp:
            local_values = []
            for value in values:
                chunk = math.ceil(len(value) / (2 * cp_size))
                starts = [cp_rank * chunk, (2 * cp_size - cp_rank - 1) * chunk]
                local_values.append(torch.cat([value[start : min(start + chunk, len(value) - 1)] for start in starts]))
            values = local_values
        return {f"{store_prefix}log_probs": values * consumed_batch_sizes[-1]}

    actor = SimpleNamespace(
        args=SimpleNamespace(
            loss_type="dpo",
            dpo_reference_free=reference_free,
            dpo_beta=0.1,
            global_batch_size=4,
            qkv_format="thd",
            uses_unsplit_forward=True,
            dynamic_context_parallel=dynamic_cp,
        ),
        model=None,
        _get_data_from_transfer_queue=get_pair_rows,
        _switch_model=lambda _: None,
        compute_log_prob=compute_log_prob,
        data_system_client=SimpleNamespace(async_clear_partition=lambda **_: None),
    )
    if not last_stage or cp_rank != 0:
        # Capture local contributions before the mocked PP/DP collectives.
        captured = []
        import relax.engine.sft.eval.preference as preference_module

        monkeypatch.setattr(
            preference_module, "finalize_pair_metrics", lambda values, **_: captured.append(values) or {}
        )
    namespace["_run_preference_eval"](actor, rollout_id=0)
    assert forwards == ([""] if reference_free else ["ref_", ""]) * len(chunks)
    if not last_stage or cp_rank != 0:
        assert torch.count_nonzero(captured[0]) == 0
        if not last_stage:
            assert reductions == []
        return

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
    assert logged["eval/dpo_pairs"] == pair_count
    assert consumed_batch_sizes == [size // dp_size for size in chunks]
    assert iterator_batch_sizes == chunks
    for actual, original in zip(rollout_data["loss_masks"], original_masks * consumed_batch_sizes[-1], strict=True):
        torch.testing.assert_close(torch.as_tensor(actual), original)
