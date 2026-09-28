# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execute the actor's save method with a controlled persistence boundary."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("rank,fail", [(0, False), (1, False), (0, True)])
def test_checkpoint_sync_rotation_follows_successful_save(rank, fail):
    source = Path(__file__).resolve().parents[3] / "relax/backends/megatron/actor.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "MegatronTrainRayActor"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "save_model")
    method.decorator_list = []
    events = []
    group = object()

    def save(*args, **kwargs):
        events.append("save")
        if fail:
            raise OSError("save failed")

    def get_rank(*, group):
        assert group is namespace["get_gloo_group"]()
        return rank

    namespace = {
        "dist": SimpleNamespace(get_rank=get_rank, barrier=lambda **kw: events.append("barrier")),
        "get_gloo_group": lambda: group,
        "save": save,
        "rotate_ckpt": lambda *a, **kw: events.append("rotate"),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    actor = SimpleNamespace(
        args=SimpleNamespace(debug_rollout_only=False, offload_train=False, async_save=False, save_hf=None),
        model=None,
        optimizer=None,
        opt_param_scheduler=None,
        role="actor",
    )
    if fail:
        with pytest.raises(OSError, match="save failed"):
            namespace["save_model"](actor, 7)
    else:
        namespace["save_model"](actor, 7)
    assert events == (["barrier", "save", "rotate"] if rank == 0 and not fail else ["barrier", "save"])
