# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU loss/gradient regression for the unit-to-native-loss boundary.

Load the production loss functions unchanged, without their GPU-only import
graph. Only metadata containers, process-group queries and the vocab-parallel
CE kernel are CPU stand-ins; slicing, SFT dispatch and reduction are real code.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from examples.chunkflow.arguments import validate_args
from examples.chunkflow.scheduler import build_chunks
from examples.chunkflow.unit_batch import _rows_for_unit, slice_parent_batch
from examples.chunkflow.units import build_micro_batches


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "options,error",
    [
        ({"calculate_per_token_loss": False}, "--calculate-per-token-loss"),
        ({"fully_async": True, "use_dynamic_batch_size": True}, "streaming iterators"),
        ({"fully_async": False, "use_dynamic_batch_size": True}, None),
        ({"fully_async": True, "use_dynamic_batch_size": False}, None),
        ({"use_chunkflow": False, "calculate_per_token_loss": False}, None),
        ({"use_chunkflow": False, "fully_async": True, "use_dynamic_batch_size": True}, None),
        ({"advantage_estimator": "gspo"}, "sequence-level objectives"),
        ({"use_opsm": True}, "sequence-level objectives"),
        ({"advantage_estimator": "grpo"}, None),
        ({"advantage_estimator": "sapo"}, None),
        ({"use_chunkflow": False, "advantage_estimator": "gspo", "use_opsm": True}, None),
        ({"num_distributed_optimizer_instances": 2}, "--num-distributed-optimizer-instances 1"),
    ],
)
def test_validation_rejects_unsupported_loss_and_streaming_contracts(options, error):
    args = SimpleNamespace(
        **{"use_chunkflow": True, "chunkflow_chunk_size": 128, "calculate_per_token_loss": True, **options}
    )
    if error is None:
        validate_args(args)
    else:
        with pytest.raises(ValueError, match=error):
            validate_args(args)


def _load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert set(names) <= definitions.keys()
    # Include native helpers extracted by upstream refactors as well as the
    # entry points, without importing the module's GPU-only dependencies.
    selected = set(names)
    pending = list(names)
    while pending:
        node = definitions[pending.pop()]
        dependencies = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
        new = (dependencies & definitions.keys()) - selected
        selected.update(new)
        pending.extend(new)
    functions = [node for name, node in definitions.items() if name in selected]
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + functions, type_ignores=[]), path, "exec"), namespace)


def _cpu_log_probs(logits, targets, group, **kwargs):
    assert group is None
    return -F.cross_entropy(logits, targets, reduction="none").unsqueeze(-1), None


@pytest.fixture
def native_loss(monkeypatch):
    # No device probing, CUDA tensors or distributed initialization in this test.
    monkeypatch.setitem(
        sys.modules, "megatron.core.packed_seq_params", SimpleNamespace(PackedSeqParams=SimpleNamespace)
    )
    monkeypatch.setitem(sys.modules, "relax.backends.megatron.data", SimpleNamespace(PrepackedBatch=dict))
    namespace = {
        "torch": torch,
        "mpu": SimpleNamespace(
            get_context_parallel_world_size=lambda: 1, get_tensor_model_parallel_group=lambda: None
        ),
        "is_sft_mode": lambda args: args.loss_type == "sft",
        "calculate_log_probs_and_entropy": _cpu_log_probs,
        "validate_opd_topk_gather": lambda *args: None,
        "ALGORITHM_SPECS": {},
    }
    _load_functions(
        "relax/backends/megatron/cp_utils.py",
        ["get_cp_local_mask_sums", "get_cp_local_num_tokens", "get_sum_of_sample_mean"],
        namespace,
    )
    _load_functions(
        "relax/backends/megatron/loss.py",
        ["get_responses", "get_log_probs_and_entropy", "sft_loss_function", "loss_function"],
        namespace,
    )
    return SimpleNamespace(**namespace)


def _parent(mode):
    samples = [torch.tensor([1, 2, 3, 4, 5, 6, 7]), torch.tensor([8, 9]), torch.tensor([10, 11])]
    lengths = [len(sample) for sample in samples]
    cu = [0, 7, 9, 11]
    if mode == "rl":
        response_lengths = [4, 1, 0]
        masks = [torch.tensor([1, 0, 1, 1]), torch.tensor([1]), torch.empty(0, dtype=torch.long)]
        full_masks = [
            F.pad(mask, (length - response - 1, 1)) for mask, length, response in zip(masks, lengths, response_lengths)
        ]
    else:
        response_lengths = lengths
        # Predictor-coordinate mask, as returned by get_batch (including its final zero).
        masks = [torch.tensor([0, 1, 1, 0, 1, 1, 0]), torch.tensor([1, 0]), torch.tensor([0, 0])]
        full_masks = masks
    parent = {
        "tokens": torch.cat(samples).unsqueeze(0),
        "unconcat_tokens": samples,
        "total_lengths": lengths,
        "response_lengths": response_lengths,
        "loss_masks": masks,
        "full_loss_masks": torch.cat(full_masks).unsqueeze(0),
        "packed_seq_params": SimpleNamespace(cu_seqlens_q_cpu=cu),
    }
    if mode == "pre_shifted":
        # Deliberately differ from input[i+1], including a supervised final row.
        labels = [torch.tensor([-100, 12, 13, -100, 14, 15, 16]), torch.tensor([17, -100]), torch.tensor([-100, -100])]
        parent["unconcat_pre_shifted_labels"] = labels
        parent["loss_masks"] = [label.ne(-100).long() for label in labels]
        parent["full_loss_masks"] = torch.cat(parent["loss_masks"]).unsqueeze(0)
    if mode == "rl":
        for key in ("advantages", "log_probs", "ref_log_probs", "rollout_log_probs", "values", "returns"):
            parent[key] = [torch.tensor([2.0, -3.0, 4.0, -5.0]), torch.tensor([6.0]), torch.empty(0)]
    return parent


@pytest.mark.parametrize("mode", ["sft", "pre_shifted", "rl"])
@pytest.mark.parametrize("chunk_size", [2, 64])
def test_unit_loss_and_gradients_match_unchunked_native_slicing(native_loss, mode, chunk_size):
    parent = _parent(mode)
    args = SimpleNamespace(
        loss_type="policy_loss" if mode == "rl" else "sft",
        qkv_format="thd",
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        calculate_per_token_loss=True,
        recompute_loss_function=False,
        allgather_cp=False,
        global_batch_size=3,
    )
    full_logits = torch.randn(1, 11, 19, generator=torch.Generator().manual_seed(42), requires_grad=True)
    chunk_logits = full_logits.detach().clone().requires_grad_()

    def loss(batch, logits):
        if mode != "rl":
            result, count, _ = native_loss.loss_function(args, batch, 1, logits)
            return result, count
        # Weighted token NLL exercises the native RL response slicer and the
        # advantages alignment without depending on a particular RL algorithm.
        pairs = native_loss.get_responses(
            logits, args=args, **{key: batch[key] for key in ("unconcat_tokens", "total_lengths", "response_lengths")}
        )
        result = logits.sum() * 0
        for (predictions, targets), advantages, mask in zip(
            pairs, batch["advantages"], batch["loss_masks"], strict=True
        ):
            assert predictions.size(0) == targets.numel() == mask.numel()
            result = result + (F.cross_entropy(predictions, targets, reduction="none") * advantages * mask).sum()
        return result, sum(mask.sum() for mask in batch["loss_masks"])

    expected, expected_count = loss(parent, full_logits)
    units = build_micro_batches(build_chunks(parent["total_lengths"], chunk_size=chunk_size), "staggered")
    actual, actual_count = chunk_logits.sum() * 0, 0
    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=0)
        rows = _rows_for_unit(parent["packed_seq_params"].cu_seqlens_q_cpu, unit)
        logits = torch.cat([chunk_logits[:, start:end] for start, end in rows], dim=1)
        logits = F.pad(logits, (0, 0, 0, batch["tokens"].size(1) - logits.size(1)))
        value, count = loss(batch, logits)
        actual, actual_count = actual + value, actual_count + count
        if mode == "rl":
            expected_aux = torch.tensor([0, 0, 2, -3, 4, -5, 0, 6, 0, 0, 0], dtype=torch.float32)
            for key in ("advantages", "log_probs", "ref_log_probs", "rollout_log_probs", "values", "returns"):
                torch.testing.assert_close(batch[key][0], torch.cat([expected_aux[start:end] for start, end in rows]))
    torch.testing.assert_close(actual_count, expected_count)
    torch.testing.assert_close(actual / actual_count, expected / expected_count)
    (actual / actual_count).backward()
    (expected / expected_count).backward()
    torch.testing.assert_close(chunk_logits.grad, full_logits.grad)
