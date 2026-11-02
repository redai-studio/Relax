# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Run production packing/forward/head functions without loading CUDA
extensions.

Only process-group queries and the vocabulary-parallel head are CPU stand-ins.
These tests check token identities, masks, CE and gradients, not just shapes.
They do not emulate distributed MTP boundary communication.
"""

import __future__

import ast
import sys
from argparse import Namespace
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


ROOT = Path(__file__).resolve().parents[3]


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    selected = [functions[name] for name in names]
    for node in selected:
        node.decorator_list = []
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def production_namespace(cp_size, rank, tp_size=2):
    from relax.utils.env import Envs

    namespace = {
        "Envs": Envs,
        "torch": torch,
        "F": F,
        "Namespace": Namespace,
        "Callable": Callable,
        "Sequence": Sequence,
        "PackedSeqParams": SimpleNamespace,
        "DataIterator": type("DataIterator", (), {}),
        "PrepackedBatch": type("PrepackedBatch", (dict,), {}),
        "get_args": lambda: Namespace(enable_mtp_training=True),
        "mpu": SimpleNamespace(
            get_context_parallel_world_size=lambda: cp_size,
            get_context_parallel_rank=lambda: rank,
            get_tensor_model_parallel_world_size=lambda: tp_size,
        ),
        "pad_sequence": pad_sequence,
        "move_tensors_to_device": lambda batch, device: batch,
        "ce_implementation": lambda _: "native",
        "_COMPUTE_MTP_ACCEPTANCE_COUNTS": None,
    }
    load_functions(ROOT / "relax/backends/megatron/cp_utils.py", ["slice_with_cp"], namespace)
    load_functions(ROOT / "relax/backends/megatron/data.py", ["get_batch"], namespace)
    load_functions(ROOT / "relax/backends/megatron/model.py", ["_attach_mtp_forward_kwargs"], namespace)
    load_functions(ROOT / "relax/models/common/chunked_mtp_loss.py", ["_chunked_head_ce"], namespace)
    return namespace


def make_batch(namespace, lengths):
    tokens = [(torch.arange(length) + i) % 7 for i, length in enumerate(lengths)]
    masks = [(torch.arange(length) % 5 != 0).int() for length in lengths]
    raw = {"tokens": tokens, "loss_masks": masks, "total_lengths": lengths, "response_lengths": lengths}
    return namespace["get_batch"](
        iter([(raw, None)]), tuple(raw), pad_multiplier=1024, is_vl_model=True, pack_device=torch.device("cpu")
    )


def legacy_kwargs(batch):
    return {
        "input_ids": batch.get("unsplit_tokens", batch["tokens"]),
        "attention_mask": batch.get("unsplit_attention_mask"),
        "position_ids": None,
        "packed_seq_params": batch.get("vlm_packed_seq_params", batch["packed_seq_params"]),
        "loss_mask": None,
    }


class CPUHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(21, dtype=torch.float64).reshape(7, 3) / 21)
        self.sequence_parallel = False

    def forward(self, hidden, weight=None, runtime_gather_output=None):
        return F.linear(hidden, self.weight if weight is None else weight), None


def token_ce(labels, logits):
    return F.cross_entropy(logits.transpose(0, 1).reshape(-1, 7), labels.reshape(-1), reduction="none").reshape_as(
        labels
    )


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("lengths", [[4425], [17, 25], [261981]])
def test_legacy_packing_mask_tokens_ce_and_gradients(cp_size, lengths):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for rank in range(cp_size):
            namespace = production_namespace(cp_size, rank)
            batch = make_batch(namespace, lengths)
            kwargs = legacy_kwargs(batch)
            namespace["_attach_mtp_forward_kwargs"](Namespace(enable_mtp_training=True), batch, kwargs)
            labels = kwargs["mtp_kwargs"]["mtp_labels"]
            mask = kwargs["loss_mask"]
            if cp_size > 1:
                assert kwargs["mtp_kwargs"]["labels_are_shifted"] is True
                # Independent global-index oracle for per-sample zigzag packing.
                expected_tokens = []
                expected_masks = []
                for sample, pad_to in enumerate(batch["padded_total_lengths"]):
                    chunk = pad_to // (2 * cp_size)
                    index = torch.cat(
                        [
                            torch.arange(rank * chunk, (rank + 1) * chunk),
                            torch.arange((2 * cp_size - rank - 1) * chunk, (2 * cp_size - rank) * chunk),
                        ]
                    )
                    padded = batch["unsplit_tokens"][sample, :pad_to]
                    expected_tokens.append(F.pad(padded[1:], (0, 1))[index])
                    raw_mask = (torch.arange(lengths[sample]) % 5 != 0).int()
                    global_mask = F.pad(F.pad(raw_mask[1:], (0, 1)), (0, pad_to - lengths[sample]))
                    expected_masks.append(global_mask[index])
                assert torch.equal(labels, torch.cat(expected_tokens).unsqueeze(0))
                assert torch.equal(mask, torch.cat(expected_masks).unsqueeze(0))
            else:
                assert mask is batch["full_loss_masks"]
            assert labels.shape == mask.shape
            if lengths == [4425] and cp_size == 4:
                assert batch["unsplit_full_mtp_loss_mask"].shape == (1, 4432)
                assert labels.shape == mask.shape == (1, 1108)

            head = CPUHead()
            hidden = torch.sin(torch.arange(labels.numel() * 3, dtype=torch.float64)).reshape(-1, 1, 3)
            hidden.requires_grad_()
            reference = token_ce(labels, head(hidden)[0]) * mask
            expected_grad = torch.autograd.grad(reference.sum(), (hidden, head.weight), retain_graph=True)
            chunked, _, _ = namespace["_chunked_head_ce"](
                hidden, labels, head, None, None, token_ce, None, 1024, loss_mask=mask
            )
            actual = chunked * mask
            actual_grad = torch.autograd.grad(actual.sum(), (hidden, head.weight))
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            for got, expected in zip(actual_grad, expected_grad, strict=True):
                torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-9)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("position_kind", ["none", "2d", "mrope"])
@pytest.mark.parametrize("attention_mask_present", [False, True])
def test_single_sequence_mask_matches_bridge_input_mode(rank, position_kind, attention_mask_present):
    namespace = production_namespace(4, rank)
    batch = make_batch(namespace, [4425])
    kwargs = legacy_kwargs(batch)
    if not attention_mask_present:
        kwargs["attention_mask"] = None
    if position_kind == "2d":
        kwargs["position_ids"] = torch.arange(4432).unsqueeze(0)
    elif position_kind == "mrope":
        kwargs["position_ids"] = torch.arange(4432).reshape(1, 1, -1).expand(3, 1, -1)
    namespace["_attach_mtp_forward_kwargs"](Namespace(enable_mtp_training=True), batch, kwargs)
    legacy = attention_mask_present and position_kind != "mrope"
    expected = batch["unsplit_mtp_loss_mask"] if legacy else batch["unsplit_full_mtp_loss_mask"]
    assert kwargs["loss_mask"] is expected


def test_disabled_mtp_preserves_forward_and_existing_mask():
    namespace = production_namespace(4, 0)
    batch = make_batch(namespace, [4425])
    kwargs = legacy_kwargs(batch)
    existing = object()
    kwargs["loss_mask"] = existing
    namespace["_attach_mtp_forward_kwargs"](Namespace(enable_mtp_training=False), batch, kwargs)
    assert "mtp_kwargs" not in kwargs
    namespace["_attach_mtp_forward_kwargs"](Namespace(enable_mtp_training=True), batch, kwargs)
    assert kwargs["loss_mask"] is existing


@pytest.mark.parametrize("bad_tensor", ["labels", "mask"])
def test_chunked_head_rejects_global_cp_supervision_before_head_calls(bad_tensor):
    namespace = production_namespace(4, 0)
    head = CPUHead()
    hidden = torch.zeros(1108, 1, 3, dtype=torch.float64)
    labels = torch.zeros(1, 4432 if bad_tensor == "labels" else 1108, dtype=torch.long)
    mask = torch.ones(1, 4432 if bad_tensor == "mask" else 1108)
    with pytest.raises(ValueError, match="CP-local head layout"):
        namespace["_chunked_head_ce"](hidden, labels, head, None, None, token_ce, None, 1024, loss_mask=mask)


@pytest.mark.parametrize("mismatch", [False, True])
def test_tp2_sequence_parallel_gather_and_flag_restore(monkeypatch, mismatch):
    namespace = production_namespace(4, 0)
    mappings = ModuleType("megatron.core.tensor_parallel.mappings")
    gathered = []

    def gather(hidden, tensor_parallel_output_grad, group):
        gathered.append(group)
        return torch.cat([hidden, hidden])

    mappings.gather_from_sequence_parallel_region = gather
    monkeypatch.setitem(sys.modules, mappings.__name__, mappings)
    head = CPUHead()
    head.sequence_parallel = True
    head.tp_group = object()
    hidden = torch.zeros(554, 1, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.zeros(1, 4432 if mismatch else 1108, dtype=torch.long)
    mask = torch.ones_like(labels)
    if mismatch:
        with pytest.raises(ValueError, match="CP-local head layout"):
            namespace["_chunked_head_ce"](hidden, labels, head, None, None, token_ce, None, 1024, loss_mask=mask)
    else:
        loss, _, _ = namespace["_chunked_head_ce"](
            hidden, labels, head, None, None, token_ce, None, 1024, loss_mask=mask
        )
        assert loss.shape == labels.shape
        loss.sum().backward()
        assert hidden.grad is not None
    assert gathered == [head.tp_group]
    assert head.sequence_parallel is True


def test_mtp_process_rejects_mismatch_before_distributed_roll():
    namespace = production_namespace(4, 0)
    load_functions(
        ROOT / "relax/models/common/chunked_mtp_loss.py",
        ["_main_update_token_normalization_enabled", "process_mtp_loss"],
        namespace,
    )
    namespace["_chunked_mtp_enabled"] = lambda: True

    def forbidden_roll(*args, **kwargs):
        pytest.fail("Invalid mask reached a distributed roll")

    namespace["_roll_mtp_labels_and_mask"] = forbidden_roll
    config = Namespace(cross_entropy_loss_fusion=False, mtp_num_layers=1)
    with pytest.raises(ValueError, match="same CP-local layout before rolling"):
        namespace["process_mtp_loss"](
            torch.zeros(2216, 1, 3),
            torch.zeros(1, 1108, dtype=torch.long),
            torch.ones(1, 4432),
            CPUHead(),
            None,
            None,
            False,
            token_ce,
            config,
        )


@pytest.mark.parametrize("rank", range(4))
def test_zero_supervision_singleton_keeps_local_mask_and_zero_gradient(rank):
    namespace = production_namespace(4, rank)
    raw = {
        "tokens": [torch.arange(4425) % 7],
        "loss_masks": [torch.zeros(4425, dtype=torch.int)],
        "total_lengths": [4425],
        "response_lengths": [4425],
    }
    batch = namespace["get_batch"](
        iter([(raw, None)]), tuple(raw), pad_multiplier=1024, is_vl_model=True, pack_device=torch.device("cpu")
    )
    kwargs = legacy_kwargs(batch)
    namespace["_attach_mtp_forward_kwargs"](Namespace(enable_mtp_training=True), batch, kwargs)
    labels, mask = kwargs["mtp_kwargs"]["mtp_labels"], kwargs["loss_mask"]
    assert labels.shape == mask.shape == (1, 1108)
    head = CPUHead()
    hidden = torch.zeros(1108, 1, 3, dtype=torch.float64, requires_grad=True)
    loss, _, _ = namespace["_chunked_head_ce"](hidden, labels, head, None, None, token_ce, None, 1024, loss_mask=mask)
    (loss * mask).sum().backward()
    assert not hidden.grad.any()
    assert not head.weight.grad.any()


@pytest.mark.parametrize("per_token,microbatches,values", [(True, 4, [0.2, 0.4]), (False, 4, [0.8, 1.6])])
def test_mtp_metrics_preserve_objective_and_recover_raw_ce(per_token, microbatches, values):
    namespace = {}
    load_functions(ROOT / "relax/backends/megatron/model.py", ["_mtp_metric_scales"], namespace)
    args = Namespace(calculate_per_token_loss=per_token, mtp_num_layers=2, mtp_loss_scaling_factor=0.2)
    objective, raw = namespace["_mtp_metric_scales"](args, microbatches, token_tracker=True)
    torch.testing.assert_close(torch.tensor(values) * objective, torch.tensor([0.2, 0.4]))
    torch.testing.assert_close(torch.tensor(values) * raw, torch.tensor([2.0, 4.0]))


def test_mtp_metrics_keep_legacy_normalization_and_handle_zero_scale():
    namespace = {}
    load_functions(ROOT / "relax/backends/megatron/model.py", ["_mtp_metric_scales"], namespace)
    args = Namespace(calculate_per_token_loss=True, mtp_num_layers=2, mtp_loss_scaling_factor=0.2)
    assert namespace["_mtp_metric_scales"](args, 4, token_tracker=False) == (0.25, None)
    args.mtp_loss_scaling_factor = 0
    assert namespace["_mtp_metric_scales"](args, 4, token_tracker=True) == (1.0, None)


def test_single_sequence_output_preserves_hidden_gradient_and_checks_mask():
    namespace = {"torch": torch}
    load_functions(ROOT / "relax/backends/megatron/model.py", ["_unwrap_single_sequence_cp_output"], namespace)
    unwrap = namespace["_unwrap_single_sequence_cp_output"]
    hidden = torch.ones(4, 1, 3, requires_grad=True)
    mask = torch.ones(1, 4)
    batch = {"unsplit_full_mtp_loss_mask": torch.ones(1, 16), "unsplit_mtp_loss_mask": mask}
    assert unwrap(hidden, batch) is hidden
    assert unwrap((hidden, mask), batch) is hidden
    unwrap((hidden, mask), batch).sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
    with pytest.raises(ValueError, match="CP-local"):
        unwrap((hidden, torch.ones(1, 16)), batch)
