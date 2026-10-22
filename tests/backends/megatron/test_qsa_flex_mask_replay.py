# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise CP1 replay on CPU without importing the Megatron runtime."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[3] / "relax/models/qwen3_8"


def _load_functions(filename, names, namespace):
    tree = ast.parse((_ROOT / filename).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    prefix = ast.parse(
        "from __future__ import annotations\n"
        "from contextvars import ContextVar\n"
        "from dataclasses import dataclass, field\n"
        "from functools import wraps\n"
    ).body
    exec(compile(ast.Module(body=prefix + nodes, type_ignores=[]), filename, "exec"), namespace)


class _CudaFacade(torch.Tensor):
    @property
    def is_cuda(self):
        return True


@pytest.mark.parametrize("is_thd", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_cp1_mask_replay_is_scoped_to_each_checkpoint(monkeypatch, is_thd, enabled):
    replay = ModuleType("relax.models.qwen3_8.qsa_cp")
    monkeypatch.setitem(sys.modules, replay.__name__, replay)
    _load_functions("qsa_cp.py", {"_ReplayTape", "_cached", "_with_replay"}, replay.__dict__)
    replay._active_tape = replay.ContextVar("test_tape", default=None)
    masks = []
    used_masks = []

    def build(selection, mask_mod, sequence_length):
        mask = object()
        masks.append(mask)
        return mask

    def attention(q, k, v, **kwargs):
        used_masks.append(kwargs["block_mask"])
        return q

    namespace = {
        "__name__": "qsa_flex_replay_test.block",
        "__package__": "qsa_flex_replay_test",
        "torch": torch,
        "Envs": SimpleNamespace(RELAX_QWEN38_QSA_RECOMPUTE_REPLAY=enabled),
        "qsa": SimpleNamespace(
            HAVE_FLEX_ATTENTION=True,
            _qsa_mask_mod_factory=lambda selection: object(),
            _compiled_flex_attention=lambda: attention,
        ),
        "_build_sparse_block_mask": build,
    }
    _load_functions("qsa_block_mask.py", {"_sparse_flex_forward"}, namespace)
    query = torch.zeros((5, 2, 4) if is_thd else (5, 1, 2, 4)).as_subclass(_CudaFacade)
    owner = SimpleNamespace(softmax_scale=0.5)
    selection = SimpleNamespace(selected_bits=object(), compress_ratio=4, bits_per_row=1)

    def forward():
        return namespace["_sparse_flex_forward"](owner, query, query, query, selection, is_thd)

    first = replay._with_replay(forward)
    torch.testing.assert_close(first(), first())
    second = replay._with_replay(forward)
    second()
    second()
    assert len(masks) == (2 if enabled else 4)
    assert (used_masks[0] is used_masks[1]) == enabled
    assert used_masks[0] is not used_masks[2]
    assert replay._active_tape.get() is None
    forward()
    forward()
    assert used_masks[-2] is not used_masks[-1]
