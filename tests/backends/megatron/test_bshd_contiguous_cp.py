# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only BSHD contiguous-CP layout, target alignment, and gradient
regression tests."""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


torch = pytest.importorskip("torch")
checkpoint = pytest.importorskip("torch.utils.checkpoint").checkpoint
F = torch.nn.functional
_ROOT = Path(__file__).resolve().parents[3]


def _load_functions(path: str, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    nodes = [
        node
        for node in ast.parse((_ROOT / path).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *nodes], type_ignores=[])
    exec(compile(module, path, "exec"), namespace)
    return namespace


def _runtime(cp_size: int, cp_rank: int, tp_size: int = 1) -> tuple[dict[str, Any], dict[str, Any]]:
    mpu = SimpleNamespace(
        get_context_parallel_world_size=lambda: cp_size,
        get_context_parallel_rank=lambda: cp_rank,
        get_context_parallel_group=lambda: "fixture-cp-group",
        get_tensor_model_parallel_world_size=lambda: tp_size,
    )
    cp = _load_functions(
        "relax/backends/megatron/cp_utils.py",
        {
            "slice_bshd_contiguous_cp",
            "slice_with_cp",
            "slice_log_prob_with_cp",
            "get_logits_and_tokens_offset_with_cp",
            "maybe_padded_total_lengths",
        },
        {"torch": torch, "F": F, "mpu": mpu, "Callable": Callable},
    )
    args = SimpleNamespace(qkv_format="bshd", allgather_cp=True, rollout_temperature=1.0)
    data = _load_functions(
        "relax/backends/megatron/data.py",
        {"get_batch", "_build_allgather_thd_metadata"},
        {
            **cp,
            "get_args": lambda: args,
            "DataIterator": type("UnusedDataIterator", (), {}),
            "PrepackedBatch": type("PrepackedBatch", (dict,), {}),
            "PackedSeqParams": SimpleNamespace,
            "move_tensors_to_device": lambda batch, device: batch,
        },
    )
    loss = _load_functions(
        "relax/backends/megatron/loss.py",
        {"get_responses", "_allgather_cp_redistribute"},
        {**cp, "is_sft_mode": lambda args: False, "dist": SimpleNamespace(nn=SimpleNamespace())},
    )
    return data, loss


def _raw_batch() -> dict[str, Any]:
    return {
        "tokens": [torch.arange(10, 16), torch.arange(100, 108)],
        "loss_masks": [torch.tensor([1, 0, 1, 1]), torch.tensor([1, 1, 0, 1, 1])],
        "total_lengths": [6, 8],
        "response_lengths": [4, 5],
        "max_seq_lens": [8, 8],
    }


def _post_process(
    data: dict[str, Any], raw: dict[str, Any], args: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    cp_module = ModuleType("relax.backends.megatron.cp_utils")
    cp_module.maybe_padded_total_lengths = data["maybe_padded_total_lengths"]
    cp_module.slice_log_prob_with_cp = data["slice_log_prob_with_cp"]
    monkeypatch.setitem(sys.modules, cp_module.__name__, cp_module)
    stream = _load_functions(
        "relax/utils/data/stream_dataloader.py",
        {"post_process_rollout_data", "_add_sft_token_lengths"},
        {
            "torch": torch,
            "mpu": data["mpu"],
            "device_utils": SimpleNamespace(make_current_torch_device=lambda: torch.device("cpu")),
            "is_sft_mode": lambda args: False,
            "iter_opd_cp_float_fields": lambda: ("teacher_log_probs",),
        },
    )
    stream["post_process_rollout_data"](args, raw)


@pytest.mark.parametrize(("cp_size", "cp_rank"), [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_bshd_allgather_batch_pads_and_slices_each_sample_mask_like_tokens(
    cp_size: int, cp_rank: int, tp_size: int
) -> None:
    data, _ = _runtime(cp_size, cp_rank, tp_size)
    raw = _raw_batch()
    expected_tokens = [F.pad(row, (0, 8 - len(row))) for row in raw["tokens"]]
    expected_masks = [torch.tensor([0, 1, 0, 1, 1, 0, 0, 0]), torch.tensor([0, 0, 1, 1, 0, 1, 1, 0])]
    batch = data["get_batch"](
        iter([(raw, None)]),
        list(raw),
        pad_multiplier=1,
        qkv_format="bshd",
        allgather_cp=True,
        pack_device=torch.device("cpu"),
    )
    start, end = cp_rank * (8 // cp_size), (cp_rank + 1) * (8 // cp_size)
    torch.testing.assert_close(batch["tokens"], torch.stack(expected_tokens)[:, start:end])
    torch.testing.assert_close(batch["full_loss_masks"], torch.stack(expected_masks).int()[:, start:end])
    assert batch["tokens"].shape == batch["full_loss_masks"].shape
    assert batch["packed_seq_params"] is None
    assert batch["max_seq_lens"] == [8, 8]


def test_bshd_allgather_batch_records_actual_padding_and_keeps_cp1_response_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, loss = _runtime(1, 0, tp_size=2)
    raw = _raw_batch()
    _post_process(
        data,
        raw,
        SimpleNamespace(qkv_format="bshd", allgather_cp=True, data_pad_size_multiplier=4, use_opd=False),
        monkeypatch,
    )
    batch = data["get_batch"](
        iter([(raw, None)]),
        list(raw),
        pad_multiplier=4,
        qkv_format="bshd",
        allgather_cp=True,
        pack_device=torch.device("cpu"),
    )
    assert batch["max_seq_lens"] == [16, 16]
    assert batch["tokens"].shape == batch["full_loss_masks"].shape == (2, 16)
    logits = torch.stack((torch.arange(16), 100 + torch.arange(16))).float().unsqueeze(-1)
    responses = list(
        loss["get_responses"](
            logits,
            args=SimpleNamespace(qkv_format="bshd", allgather_cp=True, rollout_temperature=1.0),
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=[6, 8],
            response_lengths=[4, 5],
            max_seq_lens=batch["max_seq_lens"],
        )
    )
    torch.testing.assert_close(responses[0][0].flatten(), torch.arange(1, 5).float())
    torch.testing.assert_close(responses[1][0].flatten(), torch.arange(102, 107).float())


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_bshd_allgather_ingestion_and_new_response_fields_share_final_cp_padding(
    cp_rank: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, loss = _runtime(2, cp_rank, tp_size=2)
    args = SimpleNamespace(
        qkv_format="bshd",
        allgather_cp=True,
        data_pad_size_multiplier=2,
        use_opd=False,
        rollout_temperature=1.0,
    )
    raw = _raw_batch()
    reference = [torch.arange(1, 5).float(), torch.arange(102, 107).float()]
    for key in ("rollout_log_probs", "ref_log_probs", "advantages", "returns", "values", "teacher_log_probs"):
        raw[key] = [row.clone() for row in reference]
    _post_process(data, raw, args, monkeypatch)
    assert raw["max_seq_lens"] == [16, 16]
    batch = data["get_batch"](
        iter([(raw, None)]),
        list(raw),
        pad_multiplier=2,
        qkv_format="bshd",
        allgather_cp=True,
        pack_device=torch.device("cpu"),
    )
    assert batch["max_seq_lens"] == [16, 16]
    assert batch["tokens"].shape == batch["full_loss_masks"].shape == (2, 8)
    logits = (
        torch.stack((torch.arange(cp_rank * 8, (cp_rank + 1) * 8), 100 + torch.arange(cp_rank * 8, (cp_rank + 1) * 8)))
        .float()
        .unsqueeze(-1)
    )
    response_chunks = list(
        loss["get_responses"](
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=[6, 8],
            response_lengths=[4, 5],
            max_seq_lens=batch["max_seq_lens"],
        )
    )
    new_values = {"log_probs": [chunk.flatten() for chunk, _ in response_chunks]}
    for sample, (chunk, targets) in enumerate(response_chunks):
        expected = reference[sample] if cp_rank == 0 else reference[sample][:0]
        torch.testing.assert_close(chunk.flatten(), expected)
        full_tokens = batch["unconcat_tokens"][sample]
        torch.testing.assert_close(targets, full_tokens[-(4, 5)[sample] :] if cp_rank == 0 else full_tokens[:0])

    def all_reduce(value: torch.Tensor, group: Any) -> torch.Tensor:
        assert group == "fixture-cp-group"
        full_reference = torch.cat(reference)
        torch.testing.assert_close(value, full_reference if cp_rank == 0 else torch.zeros_like(full_reference))
        return full_reference

    loss["dist"].nn.all_reduce = all_reduce
    loss["_allgather_cp_redistribute"](
        new_values,
        logits=logits,
        args=args,
        total_lengths=[6, 8],
        response_lengths=[4, 5],
        max_seq_lens=[16, 16],
    )
    assert [len(row) for row in new_values["log_probs"]] == ([3, 2] if cp_rank == 0 else [1, 3])
    for key in ("rollout_log_probs", "ref_log_probs", "advantages", "returns", "values", "teacher_log_probs"):
        for old, new in zip(batch[key], new_values["log_probs"], strict=True):
            torch.testing.assert_close(old, new)


def test_bshd_allgather_batch_does_not_change_length_after_response_fields_were_sliced() -> None:
    data, _ = _runtime(2, 0, tp_size=2)
    raw = _raw_batch()
    with pytest.raises(ValueError, match="cannot change it"):
        data["get_batch"](
            iter([(raw, None)]),
            list(raw),
            pad_multiplier=2,
            qkv_format="bshd",
            allgather_cp=True,
            pack_device=torch.device("cpu"),
        )


@pytest.mark.parametrize("qkv_format", ["bshd", "thd"])
@pytest.mark.parametrize("cp_rank", [0, 1])
def test_ingestion_preserves_default_zigzag_and_thd_lengths(
    qkv_format: str, cp_rank: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _runtime(2, cp_rank, tp_size=2)
    raw = _raw_batch()
    reference = [torch.arange(1, 5).float(), torch.arange(102, 107).float()]
    raw["rollout_log_probs"] = [row.clone() for row in reference]
    _post_process(
        data,
        raw,
        SimpleNamespace(qkv_format=qkv_format, allgather_cp=False, data_pad_size_multiplier=2, use_opd=False),
        monkeypatch,
    )
    assert raw["max_seq_lens"] == [8, 8]
    for sample, row in enumerate(reference):
        expected = data["slice_log_prob_with_cp"](
            row,
            (6, 8)[sample],
            (4, 5)[sample],
            qkv_format,
            8 if qkv_format == "bshd" else None,
        )
        torch.testing.assert_close(raw["rollout_log_probs"][sample], expected)


@pytest.mark.parametrize(("cp_size", "cp_rank"), [(1, 0), (2, 0), (2, 1)])
def test_bshd_allgather_sft_masks_use_predictor_coordinates(cp_size: int, cp_rank: int) -> None:
    data, _ = _runtime(cp_size, cp_rank)
    raw = _raw_batch()
    raw["response_lengths"] = [6, 8]
    raw["loss_masks"] = [torch.tensor([0, 0, 1, 1, 1, 1]), torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])]
    batch = data["get_batch"](
        iter([(raw, None)]),
        list(raw),
        pad_multiplier=1,
        qkv_format="bshd",
        allgather_cp=True,
        pack_device=torch.device("cpu"),
    )
    full_masks = torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0]], dtype=torch.int)
    torch.testing.assert_close(batch["full_loss_masks"], full_masks.chunk(cp_size, dim=1)[cp_rank])


@pytest.mark.parametrize(("cp_size", "cp_rank"), [(1, 0), (2, 0), (2, 1)])
def test_bshd_default_zigzag_layout_is_unchanged(cp_size: int, cp_rank: int) -> None:
    data, _ = _runtime(cp_size, cp_rank)
    raw = _raw_batch()
    original_rows = raw["tokens"].copy()
    batch = data["get_batch"](
        iter([(raw, None)]),
        list(raw),
        pad_multiplier=1,
        qkv_format="bshd",
        allgather_cp=False,
        pack_device=torch.device("cpu"),
    )
    full_rows = torch.stack([F.pad(row, (0, 8 - len(row))) for row in original_rows])
    if cp_size == 1:
        expected = full_rows
    else:
        parts = full_rows.chunk(2 * cp_size, dim=1)
        expected = torch.cat((parts[cp_rank], parts[2 * cp_size - cp_rank - 1]), dim=1)
    torch.testing.assert_close(batch["tokens"], expected)


@pytest.mark.parametrize(("cp_size", "cp_rank"), [(1, 0), (2, 0), (2, 1)])
def test_thd_allgather_concat_layout_is_unchanged(cp_size: int, cp_rank: int) -> None:
    data, _ = _runtime(cp_size, cp_rank)
    raw = _raw_batch()
    original_rows = raw["tokens"].copy()
    batch = data["get_batch"](
        iter([(raw, None)]),
        list(raw),
        pad_multiplier=1,
        qkv_format="thd",
        allgather_cp=True,
        pack_device=torch.device("cpu"),
    )
    torch.testing.assert_close(batch["tokens"], torch.cat(original_rows).chunk(cp_size)[cp_rank].unsqueeze(0))
    assert batch["packed_seq_params"].qkv_format == "thd"
    assert batch["max_seq_lens"] == [8, 8]


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_bshd_contiguous_slice_preserves_gradients_and_padding(cp_rank: int) -> None:
    data, _ = _runtime(2, cp_rank)
    values = torch.arange(6).float().requires_grad_()
    sliced = data["slice_bshd_contiguous_cp"](values, 0, 8, cp_size=2, cp_rank=cp_rank)
    sliced.sum().backward()
    expected_grad = torch.zeros_like(values)
    expected_grad[cp_rank * 4 : min((cp_rank + 1) * 4, 6)] = 1
    torch.testing.assert_close(values.grad, expected_grad)
    if cp_rank == 1:
        torch.testing.assert_close(sliced, torch.tensor([4.0, 5.0, 0.0, 0.0]))


def test_bshd_contiguous_responses_and_redistribution_match_full_targets_and_gradients() -> None:
    args = SimpleNamespace(qkv_format="bshd", allgather_cp=True, rollout_temperature=1.0)
    tokens = _raw_batch()["tokens"]
    rank_logits = [
        torch.stack((torch.arange(4), 100 + torch.arange(4))).float().unsqueeze(-1).requires_grad_(),
        torch.stack((torch.arange(4, 8), 100 + torch.arange(4, 8))).float().unsqueeze(-1).requires_grad_(),
    ]
    rank_responses, namespaces = [], []
    for rank in range(2):
        _, loss = _runtime(2, rank)
        responses = list(
            loss["get_responses"](
                rank_logits[rank],
                args=args,
                unconcat_tokens=tokens,
                total_lengths=[6, 8],
                response_lengths=[4, 5],
                max_seq_lens=[8, 8],
            )
        )
        rank_responses.append(responses)
        namespaces.append(loss)
    expected_values = (([1, 2, 3], [102, 103]), ([4], [104, 105, 106]))
    expected_tokens = (([12, 13, 14], [103, 104]), ([15], [105, 106, 107]))
    contributions = []
    for rank in range(2):
        full_rows = []
        for sample, (chunk, targets) in enumerate(rank_responses[rank]):
            torch.testing.assert_close(chunk.flatten(), torch.tensor(expected_values[rank][sample]).float())
            torch.testing.assert_close(targets, torch.tensor(expected_tokens[rank][sample]))
            start = max(rank * 4, (2, 3)[sample] - 1) - ((2, 3)[sample] - 1)
            full_rows.append(F.pad(chunk.flatten(), (start, (4, 5)[sample] - start - len(chunk))))
        contributions.append(torch.cat(full_rows))
    full_responses = contributions[0] + contributions[1]
    torch.testing.assert_close(full_responses, torch.tensor([1, 2, 3, 4, 102, 103, 104, 105, 106]).float())

    output_sum = 0
    for rank in range(2):
        loss = namespaces[rank]

        def all_reduce(value: torch.Tensor, group: Any, rank: int = rank) -> torch.Tensor:
            assert group == "fixture-cp-group"
            torch.testing.assert_close(value, contributions[rank])
            return full_responses

        loss["dist"].nn.all_reduce = all_reduce
        result = {"values": [chunk.flatten() for chunk, _ in rank_responses[rank]]}
        loss["_allgather_cp_redistribute"](
            result,
            logits=rank_logits[rank],
            args=args,
            total_lengths=[6, 8],
            response_lengths=[4, 5],
            max_seq_lens=[8, 8],
        )
        for sample, reference in enumerate(full_responses.split([4, 5])):
            expected = loss["slice_log_prob_with_cp"](reference, (6, 8)[sample], (4, 5)[sample], "bshd", 8)
            torch.testing.assert_close(result["values"][sample], expected)
            output_sum = output_sum + result["values"][sample].sum()
    output_sum.backward()
    for rank in range(2):
        expected_grad = torch.zeros_like(rank_logits[rank])
        for sample, (prompt, total) in enumerate(((2, 6), (3, 8))):
            start = max(prompt - 1, rank * 4) - rank * 4
            end = min(total - 1, (rank + 1) * 4) - rank * 4
            expected_grad[sample, start:end] = 1
        torch.testing.assert_close(rank_logits[rank].grad, expected_grad)


def test_bshd_contiguous_cp1_redistribution_is_noop() -> None:
    _, loss = _runtime(1, 0)
    result = {"values": [torch.arange(4).float(), torch.arange(5).float()]}
    before = result["values"]
    loss["_allgather_cp_redistribute"](
        result,
        logits=torch.zeros(2, 8, 1),
        args=SimpleNamespace(qkv_format="bshd"),
        total_lengths=[6, 8],
        response_lengths=[4, 5],
        max_seq_lens=[8, 8],
    )
    assert result["values"] is before


@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("with_grad", [False, True])
def test_bshd_empty_contiguous_cp_checkpoint_keeps_loss_grad_and_detaches_metrics(
    cp_rank: int, with_grad: bool
) -> None:
    _, loss = _runtime(2, cp_rank)
    args = SimpleNamespace(qkv_format="bshd", allgather_cp=True, rollout_temperature=1.0, entropy_coef=0.0)
    logits = torch.arange(cp_rank * 8, (cp_rank + 1) * 8).float().view(1, 8, 1).requires_grad_()
    calls: list[tuple[str, bool, bool]] = []
    reference = torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])

    def all_reduce(value: torch.Tensor, group: Any) -> torch.Tensor:
        assert group == "fixture-cp-group"
        key = ("log_probs", "entropy")[len(calls) % 2]
        calls.append((key, torch.is_grad_enabled(), value.requires_grad))
        # Supply the other rank's values while retaining only this rank's graph.
        return value + (reference - value.detach())

    loss["dist"].nn.all_reduce = all_reduce

    def compute_loss(model_logits: torch.Tensor) -> torch.Tensor:
        chunk, _ = next(
            loss["get_responses"](
                model_logits,
                args=args,
                unconcat_tokens=[torch.arange(6)],
                total_lengths=[6],
                response_lengths=[6],
                max_seq_lens=[16],
            )
        )
        # Match calculate_log_probs_and_entropy's detached empty-result branch.
        log_probs = chunk[:, 0] if len(chunk) else chunk.new_zeros((0,))
        values = {"log_probs": [log_probs], "entropy": [log_probs.detach()]}
        loss["_allgather_cp_redistribute"](
            values,
            logits=model_logits,
            args=args,
            total_lengths=[6],
            response_lengths=[6],
            max_seq_lens=[16],
        )
        assert values["log_probs"][0].numel() == (4 if cp_rank == 0 else 1)
        assert not values["entropy"][0].requires_grad
        return values["log_probs"][0].sum()

    if with_grad:
        output = checkpoint(compute_loss, logits, use_reentrant=True)
        output.backward()
        assert logits.grad is not None
        assert calls[2:] == [("log_probs", True, True), ("entropy", True, False)]
    else:
        with torch.no_grad():
            output = compute_loss(logits)
        assert not output.requires_grad
    torch.testing.assert_close(output, torch.tensor(6.0 if cp_rank == 0 else 4.0))
    assert calls[:2] == [("log_probs", False, False), ("entropy", False, False)]
