# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execute the actor's save method with a controlled persistence boundary."""

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.utils.agentic_obs.run_hooks import annotate_failure_stage
from relax.utils.rotate_ckpt import rotate_ckpt


@pytest.fixture
def save_method():
    source = Path(__file__).resolve().parents[4] / "relax/backends/megatron/actor.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "MegatronTrainRayActor"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "save_model")
    method.decorator_list = []
    namespace = {"os": os}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["save_model"], namespace


@pytest.mark.parametrize("rank,fail", [(0, False), (1, False), (0, True)])
def test_checkpoint_sync_rotation_follows_successful_save(save_method, rank, fail):
    method, namespace = save_method
    events = []
    group = object()

    def save(*args, **kwargs):
        events.append("save")
        if fail:
            raise OSError("save failed")

    def get_rank(*, group):
        assert group is namespace["get_gloo_group"]()
        return rank

    namespace.update(
        {
            "dist": SimpleNamespace(get_rank=get_rank, barrier=lambda **kw: events.append("barrier")),
            "get_gloo_group": lambda: group,
            "save": save,
            "rotate_ckpt": lambda *a, **kw: events.append("rotate"),
        }
    )
    actor = SimpleNamespace(
        args=SimpleNamespace(debug_rollout_only=False, offload_train=False, async_save=False, save_hf=None),
        model=None,
        optimizer=None,
        opt_param_scheduler=None,
        role="actor",
        _is_standard_dpo=lambda: False,
        _flush_resident_sft_parameters=lambda: None,
    )
    if fail:
        with pytest.raises(OSError, match="save failed"):
            method(actor, 7)
    else:
        method(actor, 7)
    assert events == (["barrier", "save", "rotate"] if rank == 0 and not fail else ["barrier", "save"])


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("failure", [None, "save", "finalize"])
def test_checkpoint_async_preserves_recovery_point(save_method, monkeypatch, tmp_path, rotate, failure):
    method, namespace = save_method
    old = tmp_path / "iter_0000019"
    old.mkdir()
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("19", encoding="utf-8")
    # A retry can encounter a partial directory from an earlier crash.
    current = tmp_path / "iter_0000023"
    current.mkdir()
    future = tmp_path / "iter_0000027"
    future.mkdir()
    pending = None

    def save(step, *args, **kwargs):
        nonlocal pending
        # No deletion is allowed before this save has a successful predecessor.
        assert (old if step == 23 else current).is_dir()
        path = tmp_path / f"iter_{step:07d}"
        (path / "shard").write_text("partial", encoding="utf-8")
        if failure == "save":
            raise OSError("save failed")
        pending = step

    def finalize(*, blocking):
        nonlocal pending
        assert blocking
        if pending is not None:
            completed = pending
            pending = None
            if failure == "finalize":
                raise OSError("finalize failed")
            tracker.write_text(str(completed), encoding="utf-8")

    monkeypatch.setitem(
        sys.modules, "megatron.training.async_utils", SimpleNamespace(maybe_finalize_async_save=finalize)
    )
    namespace.update(
        dist=SimpleNamespace(get_rank=lambda **kwargs: 0, barrier=lambda **kwargs: None),
        get_gloo_group=lambda: object(),
        save=save,
        rotate_ckpt=rotate_ckpt,
    )
    actor = SimpleNamespace(
        args=SimpleNamespace(
            debug_rollout_only=False,
            offload_train=False,
            async_save=True,
            save_hf=None,
            save=str(tmp_path),
            max_actor_ckpt_to_keep=1,
            rotate_ckpt=rotate,
            save_interval=4,
        ),
        model=None,
        optimizer=None,
        opt_param_scheduler=None,
        role="actor",
        _is_standard_dpo=lambda: False,
        _flush_resident_sft_parameters=lambda: None,
    )
    if failure == "save":
        with pytest.raises(OSError, match="save failed"):
            method(actor, 23)
        assert old.is_dir()
    else:
        method(actor, 23)
        assert old.is_dir()
        assert tracker.read_text() == "19"
        if failure == "finalize":
            with pytest.raises(OSError, match="finalize failed"):
                method(actor, 27)
            assert old.is_dir()
            # Megatron removes the failed request before its finalizers run.
            # Retrying with an empty queue must still preserve the tracker target.
            method(actor, 27)
            assert old.is_dir()
            assert tracker.read_text() == "19"
        else:
            method(actor, 27, force_sync=True)
            assert not old.exists()
            assert not current.exists()
            assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "27"
    assert future.is_dir()


@pytest.mark.parametrize("tracker_value", [None, "release", "23"])
def test_checkpoint_async_entry_skips_rotation_without_an_older_commit(
    save_method, monkeypatch, tmp_path, tracker_value
):
    method, namespace = save_method
    old = tmp_path / "iter_0000019"
    old.mkdir()
    (tmp_path / "iter_0000023").mkdir()
    if tracker_value is not None:
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(tracker_value, encoding="utf-8")

    def save(*args, **kwargs):
        assert old.is_dir()
        raise OSError("save failed")

    monkeypatch.setitem(
        sys.modules,
        "megatron.training.async_utils",
        SimpleNamespace(maybe_finalize_async_save=lambda **kwargs: None),
    )
    namespace.update(
        dist=SimpleNamespace(get_rank=lambda **kwargs: 0, barrier=lambda **kwargs: None),
        get_gloo_group=lambda: object(),
        save=save,
        rotate_ckpt=rotate_ckpt,
    )
    actor = SimpleNamespace(
        args=SimpleNamespace(
            debug_rollout_only=False,
            offload_train=False,
            async_save=True,
            save=str(tmp_path),
            max_actor_ckpt_to_keep=1,
            rotate_ckpt=False,
        ),
        model=None,
        optimizer=None,
        opt_param_scheduler=None,
        role="actor",
        _is_standard_dpo=lambda: False,
        _flush_resident_sft_parameters=lambda: None,
    )
    with pytest.raises(OSError, match="save failed"):
        method(actor, 23)
    assert old.is_dir()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("failure", [None, "finalize", "sidecar"])
def test_checkpoint_async_dpo_rotation_requires_sidecar(save_method, monkeypatch, rank, failure):
    method, namespace = save_method
    events = []

    def finalize(*, blocking):
        assert blocking
        events.append("finalize")
        if failure == "finalize" and "save" in events:
            raise OSError("finalize failed")

    def sidecar(*args):
        events.append("sidecar")
        if failure == "sidecar":
            raise OSError("sidecar failed")

    monkeypatch.setitem(
        sys.modules, "megatron.training.async_utils", SimpleNamespace(maybe_finalize_async_save=finalize)
    )
    namespace.update(
        dist=SimpleNamespace(get_rank=lambda **kwargs: rank, barrier=lambda **kwargs: events.append("barrier")),
        get_gloo_group=lambda: object(),
        save=lambda *args, **kwargs: events.append("save"),
        rotate_ckpt=lambda *args, **kwargs: events.append("rotate"),
        write_reference_identity=sidecar,
        reference_identity_path=lambda *args: None,
    )
    actor = SimpleNamespace(
        args=SimpleNamespace(
            debug_rollout_only=False, offload_train=False, async_save=True, save_hf=None, save="unused"
        ),
        model=None,
        optimizer=None,
        opt_param_scheduler=None,
        role="actor",
        _is_standard_dpo=lambda: True,
        _dpo_reference_identity=object(),
        _flush_resident_sft_parameters=lambda: None,
    )
    fails = failure == "finalize" or (failure == "sidecar" and rank == 0)
    if fails:
        with pytest.raises(OSError, match=f"{failure} failed"):
            method(actor, 23)
        assert "rotate" not in events
    else:
        method(actor, 23)
        expected = ["finalize", "barrier", "save", "finalize"]
        if rank == 0:
            expected.append("sidecar")
        expected.append("barrier")
        if rank == 0:
            expected.append("rotate")
        assert events == expected


@pytest.mark.parametrize("failure", ["write", "previous_async", "final_async"])
def test_checkpoint_failure_annotation_covers_write_and_async_finalization(monkeypatch, tmp_path, failure):
    import sys
    from types import ModuleType

    source = Path(__file__).resolve().parents[4] / "relax/backends/megatron/actor.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "MegatronTrainRayActor"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "save_model")
    # Keep the actual observability decorator; isolate the unrelated timer.
    method.decorator_list = [node for node in method.decorator_list if isinstance(node, ast.Call)]
    error = OSError("Disk quota exceeded")
    finalizations = []

    def finalize(*, blocking):
        assert blocking
        finalizations.append(None)
        if failure == "previous_async" or len(finalizations) == 2:
            raise error

    async_module = ModuleType("megatron.training.async_utils")
    async_module.maybe_finalize_async_save = finalize
    monkeypatch.setitem(sys.modules, async_module.__name__, async_module)

    def save(*args, **kwargs):
        if failure == "write":
            raise error

    group = object()
    namespace = {
        "annotate_failure_stage": annotate_failure_stage,
        "os": os,
        "dist": SimpleNamespace(get_rank=lambda **kw: 0, barrier=lambda **kw: None),
        "get_gloo_group": lambda: group,
        "save": save,
        "rotate_ckpt": lambda *a, **kw: None,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    actor = SimpleNamespace(
        args=SimpleNamespace(
            debug_rollout_only=False,
            offload_train=False,
            async_save=failure != "write",
            save_hf=None,
            dump_agentic_obs="unused",
            save=str(tmp_path),
        ),
        model=None,
        optimizer=None,
        opt_param_scheduler=None,
        role="actor",
        _is_standard_dpo=lambda: False,
        _flush_resident_sft_parameters=lambda: None,
    )
    with pytest.raises(OSError) as caught:
        namespace["save_model"](actor, 7, force_sync=True)
    assert caught.value is error and error._relax_run_failure_stage == "checkpoint"
    assert len(finalizations) == {"write": 0, "previous_async": 1, "final_async": 2}[failure]
