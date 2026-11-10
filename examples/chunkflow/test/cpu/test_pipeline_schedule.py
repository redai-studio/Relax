# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Blocking P2P simulation and dependency checks for the PP schedule."""

from __future__ import annotations

import ast
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.chunkflow.arguments import add_arguments, validate_args
from examples.chunkflow.scheduler import build_chunks
from examples.chunkflow.units import build_micro_batches, make_model_plan


@pytest.mark.parametrize("pp_size", [1, 2, 4])
def test_schedule_uses_actual_pp_without_duplicate_option(monkeypatch, pp_size):
    # Load the real selector without importing the GPU-only Megatron dependency graph.
    path = Path(__file__).resolve().parents[2] / "schedules.py"
    tree = ast.parse(path.read_text())
    selector = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_schedule")
    single, pipeline = object(), object()
    namespace = {
        "__package__": "examples.chunkflow",
        "os": os,
        "chunkflow_forward_backward_no_pipelining": single,
    }
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [selector], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    monkeypatch.setitem(
        sys.modules,
        "examples.chunkflow.pipeline_schedules",
        SimpleNamespace(chunkflow_forward_backward_pipelining_without_interleaving=pipeline),
    )
    parser = add_arguments(ArgumentParser())
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    args = parser.parse_args(
        ["--use-chunkflow", "--chunkflow-chunk-size", "4096", "--pipeline-model-parallel-size", str(pp_size)]
    )
    args.calculate_per_token_loss = True
    validate_args(args)
    monkeypatch.delenv("RELAX_CHUNKFLOW_PIPELINE_UNTESTED", raising=False)
    if pp_size == 1:
        assert namespace["get_schedule"](args, pp_size) is single
    else:
        with pytest.raises(NotImplementedError, match="RELAX_CHUNKFLOW_PIPELINE_UNTESTED"):
            namespace["get_schedule"](args, pp_size)
        monkeypatch.setenv("RELAX_CHUNKFLOW_PIPELINE_UNTESTED", "1")
        assert namespace["get_schedule"](args, pp_size) is pipeline


def _channel(stage, action):
    if action == "sf":
        return (stage, stage + 1, "fwd")
    if action == "rf":
        return (stage - 1, stage, "fwd")
    if action == "sb":
        return (stage - 1, stage, "bwd")
    return (stage, stage + 1, "bwd")


def _simulate(calls_by_stage):
    pp = len(calls_by_stage)
    pos = defaultdict(lambda: {"send": 0, "recv": 0})
    per_call = []
    for stage, calls in enumerate(calls_by_stage):
        stage_calls = []
        for recvs, sends in calls:
            phases = []
            for _part in ("shape", "data"):
                msgs = []
                for a in recvs:
                    ch = _channel(stage, a)
                    msgs.append((ch, "recv", pos[ch]["recv"]))
                    pos[ch]["recv"] += 1
                for a in sends:
                    ch = _channel(stage, a)
                    msgs.append((ch, "send", pos[ch]["send"]))
                    pos[ch]["send"] += 1
                phases.append(msgs)
            stage_calls.append(phases)
        per_call.append(stage_calls)

    sends_posted = defaultdict(int)
    recvs_posted = defaultdict(int)
    ptr = [(0, 0) for _ in range(pp)]
    posted = [None] * pp

    def enter(stage):
        c, p = ptr[stage]
        if c >= len(per_call[stage]):
            posted[stage] = None
            return
        posted[stage] = [list(per_call[stage][c][p]), 0]
        for ch, kind, _p in posted[stage][0]:
            (sends_posted if kind == "send" else recvs_posted)[ch] += 1

    for stage in range(pp):
        enter(stage)

    guard = 0
    while True:
        guard += 1
        assert guard < 10_000_000, "iteration limit"
        progress = False
        for stage in range(pp):
            if posted[stage] is None:
                continue
            msgs, done = posted[stage]
            still = []
            for ch, kind, p in msgs:
                if (recvs_posted if kind == "send" else sends_posted)[ch] > p:
                    done += 1
                else:
                    still.append((ch, kind, p))
            if done != posted[stage][1]:
                posted[stage] = [still, done]
                progress = True
            if not still:
                c, p = ptr[stage]
                ptr[stage] = (c, p + 1) if p + 1 < len(per_call[stage][c]) else (c + 1, 0)
                enter(stage)
                progress = True
        if all(ptr[s][0] >= len(per_call[s]) for s in range(pp)):
            return
        assert progress, f"p2p deadlock: stalled at {[(s, ptr[s]) for s in range(pp) if posted[s]]}"


def _calls(W, n, is_first, is_last):
    calls = []
    for _ in range(W):
        calls.append((["rf"] if not is_first else [], []))
        calls.append(([], ["sf"]) if not is_last else ([], []))
    n_steady = n - W
    if n_steady > 0:
        calls.append((["rf"] if not is_first else [], []))
    for i in range(n_steady):
        calls.append((["rb"] if not is_last else [], ["sf"] if not is_last else []))
        calls.append((["rf"] if (not is_first and i < n_steady - 1) else [], ["sb"] if not is_first else []))
    for _ in range(W):
        calls.append((["rb"] if not is_last else [], []))
        calls.append(([], ["sb"]) if not is_first else ([], []))
    return calls


def _schedule(lengths, pp, policy, k, stage):
    chunks = build_chunks(lengths, chunk_size=512)
    units = build_micro_batches(chunks, policy)
    plan = make_model_plan(units, chunks, k or 1 << 30, pp)
    return chunks, units, (plan.forward, plan.backward, plan.recompute, plan.warmups[stage])


@pytest.mark.parametrize(
    "lengths,pp,policy,k",
    [
        ([96, 96, 96, 96], 4, "staggered", 0),
        ([2600], 4, "staggered", 0),
        ([2600], 4, "staggered", 1),
        ([2048, 700, 300, 1500], 4, "staggered", 0),
        ([2048, 700, 300, 1500], 4, "staggered", 1),
        ([2000, 1800, 2100, 1900], 3, "staggered", 2),
        ([2048, 700, 300, 1500], 2, "lockstep", 1),
    ],
)
def test_pipeline_plan_is_complete_dependency_valid_and_deadlock_free(lengths, pp, policy, k):
    calls = []
    for stage in range(pp):
        chunks, units, (fwd, bwd, need_fr, warmup) = _schedule(lengths, pp, policy, k, stage)
        n = len(fwd)
        assert sorted(fwd) == sorted(bwd) == sorted(u.unit_id for u in units)
        assert set(need_fr) <= set(bwd)
        assert bool(need_fr) == (k > 0)
        assert 0 <= warmup <= n
        calls.append(_calls(warmup, n, stage == 0, stage == pp - 1))
        by_id = {u.unit_id: u for u in units}
        for i, uid in enumerate(bwd):
            done = set(fwd[: min(warmup + i + 1, n)])
            assert uid in done, f"backward {uid} before its forward"
            for group in by_id[uid].groups:
                assert all(u.unit_id in done for u in units if group in u.groups), (
                    f"backward {uid} before group {group} full"
                )
        fwd_pos = {uid: i for i, uid in enumerate(fwd)}
        bwd_pos = {uid: i for i, uid in enumerate(bwd)}
        unit_of = {c.chunk_id: u.unit_id for u in units for c in u.chunks}
        groups = defaultdict(list)
        for chunk in chunks:
            if chunk.is_dependent:
                groups[chunk.group].append(chunk)
        for members in groups.values():
            members.sort(key=lambda chunk: chunk.group_idx)
            forward_positions = [fwd_pos[unit_of[c.chunk_id]] for c in members]
            backward_positions = [bwd_pos[unit_of[c.chunk_id]] for c in reversed(members)]
            assert forward_positions == sorted(forward_positions)
            assert backward_positions == sorted(backward_positions)
    _simulate(calls)


@pytest.mark.parametrize("padding", [0, 180])
def test_plan_preserves_real_samples_and_excludes_alignment_padding(padding):
    pytest.importorskip("megatron.core")
    import torch

    from examples.chunkflow.schedules import _plan
    from examples.chunkflow.unit_batch import slice_parent_batch

    cu = [0, 1100, 1612]
    if padding:
        cu.append(cu[-1] + padding)
    tokens = torch.arange(cu[-1]).view(1, -1)
    mask = torch.zeros_like(tokens)
    mask[0, 1090:1099] = 1
    parent = dict(
        tokens=tokens,
        full_loss_masks=mask,
        total_lengths=[1100, 512],
        packed_seq_params=SimpleNamespace(cu_seqlens_q_cpu=cu),
    )
    chunks, units = _plan(parent, 512, "staggered")
    assert len(units) == 4
    assert sum(c.num_tokens for c in chunks) == 1612
    assert sorted((s.seq_id, s.prefix, s.length) for c in chunks for s in c.segments) == [
        (0, 0, 512),
        (0, 512, 512),
        (0, 1024, 76),
        (1, 0, 512),
    ]
    batches = [slice_parent_batch(parent, unit, pad_token_id=0) for unit in units]
    targets = torch.cat([b["unconcat_tokens"][0][1:][b["loss_masks"][0].bool()] for b in batches])
    torch.testing.assert_close(targets, torch.arange(1091, 1100))
    for source_lengths in (None, [1099, 513]):
        parent["total_lengths"] = source_lengths
        unverified, _ = _plan(parent, 512, "staggered")
        assert sum(c.num_tokens for c in unverified) == cu[-1]


@pytest.mark.parametrize("parent_window", [None, 0, 1, 2, 4, 7])
@pytest.mark.parametrize("stage", range(4))
@pytest.mark.parametrize("keep_last_k", [0, 1])
def test_executor_routes_planned_shapes_for_ragged_units(monkeypatch, stage, keep_last_k, parent_window):
    pytest.importorskip("megatron.core")
    import torch
    from megatron.training import global_vars

    from examples.chunkflow import pipeline_schedules as executor
    from examples.chunkflow.runtime import UNIT_ATTR, ChunkflowRuntime
    from examples.chunkflow.schedules import _plan
    from relax.backends.megatron.data import PrepackedBatch

    if parent_window is None:
        monkeypatch.delenv("RELAX_CHUNKFLOW_PARENT_WINDOW", raising=False)
    else:
        monkeypatch.setenv("RELAX_CHUNKFLOW_PARENT_WINDOW", str(parent_window))
    config = SimpleNamespace(
        variable_seq_lengths=True,
        batch_p2p_comm=True,
        sequence_parallel=True,
        hidden_size=4,
        overlap_p2p_comm=False,
        no_sync_func=None,
        deallocate_pipeline_outputs=False,
        grad_sync_func=None,
        finalize_model_grads_func=None,
    )
    args = SimpleNamespace(
        chunkflow_chunk_size=256,
        chunkflow_pack_policy="staggered",
        chunkflow_keep_last_k=keep_last_k,
        data_pad_size_multiplier=128,
        qkv_format="thd",
        allgather_cp=False,
        uses_unsplit_forward=False,
        loss_type="sft",
        calculate_per_token_loss=True,
    )
    parents, forward_shapes, backward_shapes = [], [], []
    first_lengths = [600, 350, 200, 129, 1]
    for lengths in (first_lengths, [513, 128, 33], [128], [129], [700]):
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        real_rows = cu[-1]
        if pad := (-real_rows) % 256:
            cu.append(real_rows + pad)
        tokens = (torch.arange(cu[-1]) + len(parents) * 10000).view(1, -1)
        mask = torch.ones_like(tokens)
        mask[0, real_rows:] = 0
        parent = PrepackedBatch(
            tokens=tokens,
            full_loss_masks=mask,
            total_lengths=lengths,
            packed_seq_params=SimpleNamespace(cu_seqlens_q_cpu=cu),
            replay_micro_batch_index=len(parents),
        )
        parents.append(parent)
        chunks, units = _plan(parent, 256, "staggered")
        plan = make_model_plan(units, chunks, keep_last_k or 1 << 30, 4)
        fwd, bwd = plan.forward, plan.backward
        shapes = {u.unit_id: ((u.num_tokens + 127) // 128 * 64, 1, 4) for u in units}
        forward_shapes.extend(shapes[uid] for uid in fwd)
        backward_shapes.extend(shapes[uid] for uid in bwd)
    assert len(set(forward_shapes)) > 1

    class Communicator:
        total_stages, current_stage = 4, stage
        is_pp_first_stage, is_pp_last_stage = stage == 0, stage == 3
        pp_group = SimpleNamespace(size=lambda: 4, rank=lambda: stage)

        def __init__(self):
            self.config = config
            self.positions = defaultdict(int)

        def receive(self, shapes, order, key):
            assert self.config.variable_seq_lengths is False
            assert self.config.batch_p2p_comm is False
            expected = order[self.positions[key]]
            self.positions[key] += 1
            assert shapes == [expected]
            return [torch.ones(expected)]

        def send(self, tensors, order, key):
            assert self.config.variable_seq_lengths is False
            assert self.config.batch_p2p_comm is False
            tensors = tensors if isinstance(tensors, list) else [tensors]
            assert [tuple(t.shape) for t in tensors] == [order[self.positions[key]]]
            self.positions[key] += 1

        def recv_forward(self, shapes, first):
            return None if first else self.receive(shapes, forward_shapes, "rf")

        def recv_backward(self, shapes, last):
            return None if last else self.receive(shapes, backward_shapes, "rb")

        def send_forward(self, tensors, last):
            if not last:
                self.send(tensors, forward_shapes, "sf")

        def send_backward(self, tensors, first):
            if not first:
                self.send(tensors, backward_shapes, "sb")

        def send_forward_recv_backward(self, tensors, shapes, last):
            self.send_forward(tensors, last)
            return self.recv_backward(shapes, last)

        def send_backward_recv_forward(self, tensors, shapes, first):
            self.send_backward(tensors, first)
            return self.recv_forward(shapes, first)

    groups, released, first_flags, seen_tokens = {}, [], [], defaultdict(list)

    def forward(_callback, iterator, *_args, **_kwargs):
        batch, _ = next(iterator)
        unit = getattr(batch["packed_seq_params"], UNIT_ATTR)
        mb = _kwargs["current_microbatch"]
        first_flags.append(_kwargs["is_first_microbatch"])
        assert (batch["tokens"][0, : unit.num_tokens] // 10000 == mb).all()
        offset = 0
        for chunk in unit.chunks:
            owner_mb = int(batch["tokens"][0, offset] // 10000)
            if chunk.is_dependent:
                owner = groups.setdefault(chunk.group, owner_mb)
                assert owner == owner_mb, "two parents share a live dependency group"
                model.chunkflow_runtime.store(1, chunk.group)
            if not getattr(batch["packed_seq_params"], "chunkflow_recompute", False):
                seen_tokens[owner_mb].append(chunk.num_tokens)
            offset += chunk.num_tokens
        shape = ((unit.num_tokens + 127) // 128 * 64, 1, 4)
        return torch.ones(()) if stage == 3 else torch.ones(shape), torch.tensor(1)

    real_zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: real_zeros(*a, **dict(kw, device="cpu")))
    monkeypatch.setattr(global_vars, "get_args", lambda: args)
    monkeypatch.setattr(executor, "get_model_config", lambda _: config)
    monkeypatch.setattr(executor, "forward_step", forward)
    monkeypatch.setattr(executor, "deallocate_output_tensor", lambda *a: None)
    monkeypatch.setattr(executor, "check_first_val_step", lambda *a: a[-1])
    monkeypatch.setattr(
        executor,
        "backward_step",
        lambda inputs, *_: None if inputs is None else [torch.ones_like(t) for t in inputs],
    )
    model = SimpleNamespace(chunkflow_runtime=ChunkflowRuntime(256))
    real_release = model.chunkflow_runtime.release_group

    def release(group):
        released.append(groups.pop(group))
        assert real_release(group) == 1

    monkeypatch.setattr(model.chunkflow_runtime, "release_group", release)
    comm = Communicator()
    executor.chunkflow_forward_backward_pipelining_without_interleaving(
        forward_step_func=None,
        data_iterator=iter((p, {}) for p in parents),
        model=model,
        num_microbatches=len(parents),
        seq_length=32768,
        decoder_seq_length=32768,
        micro_batch_size=1,
        p2p_communicator=comm,
        pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: 1), tp=SimpleNamespace(size=lambda: 2)),
    )
    assert first_flags.count(True) == 1
    assert not groups
    assert sorted(released) == [0, 0, 1, 4]
    assert {mb: sum(values) for mb, values in seen_tokens.items()} == {
        i: sum(p["total_lengths"]) for i, p in enumerate(parents)
    }
    assert config.variable_seq_lengths is True and comm.config is config
    assert config.batch_p2p_comm is True
    for key, expected in (("sf", stage != 3), ("rf", stage != 0), ("sb", stage != 0), ("rb", stage != 3)):
        assert comm.positions[key] == (len(forward_shapes) if expected else 0)
