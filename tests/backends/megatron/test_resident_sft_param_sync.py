# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise actual train-loop hook lifecycle without importing CUDA
Megatron."""

import ast
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.engine.sft.runtime import (
    is_dpo_mode,
    is_offline_mode,
    is_preference_mode,
    resident_sft_param_overlap_enabled,
    should_skip_sft_weight_management,
)
from relax.utils.env import Envs


_ROOT = Path(__file__).resolve().parents[3]


def _functions(path, names, namespace, *, class_method=False):
    tree = ast.parse((_ROOT / path).read_text())
    nodes = list(ast.walk(tree)) if class_method else tree.body
    functions = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + functions,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def _args():
    return Namespace(
        loss_type="sft",
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        align_grad_reduce=False,
        align_param_gather=True,
        reset_optimizer_states=False,
        manual_gc=False,
        enable_mtp_training=False,
        global_batch_size=1536,
        num_rollout=3,
        _relax_resident_sft_actor_verified=True,
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("intermediate_reader", [False, True])
def test_resident_sft_preserves_hooks_across_single_step_calls(monkeypatch, enabled, intermediate_reader):
    monkeypatch.setenv("RELAX_SFT_RESIDENT_PARAM_OVERLAP", "1" if enabled else "0")
    monkeypatch.setenv("RELAX_QWEN38_EMPTY_CACHE_BEFORE_GRAD_SYNC", "0")
    args = _args()
    config = SimpleNamespace(no_sync_func=None, param_sync_func=None)
    trace, forward_hooks = [], []

    class DDP:
        def __init__(self):
            self.hooks = True
            self.current_parameters = True

        def train(self):
            pass

        def no_sync(self):
            return nullcontext()

        def start_grad_sync(self):
            pass

        def start_param_sync(self, force_sync=False):
            self.current_parameters = True

        def disable_forward_pre_hook(self, param_sync=True):
            trace.append(("disable", param_sync))
            self.hooks = False
            if param_sync:
                self.current_parameters = True

        def enable_forward_pre_hook(self):
            trace.append(("enable",))
            self.hooks = True

    class Iterator:
        rollout_data = {}

        def reset(self):
            pass

    chunk = DDP()

    def train_step(*_):
        forward_hooks.append(chunk.hooks)
        if chunk.hooks:
            chunk.start_param_sync()
        assert chunk.current_parameters, "Forward consumed incomplete optimizer shards"
        chunk.current_parameters = False
        return {}, 0.0

    namespace = dict(
        Envs=Envs,
        get_args=lambda: args,
        get_model_config=lambda _: config,
        DDP=DDP,
        DataIterator=Iterator,
        StreamingTQIterator=type("Streaming", (), {}),
        ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY="counts",
        resident_sft_param_overlap_enabled=resident_sft_param_overlap_enabled,
        has_conditional_branch_sync=lambda _: False,
        finalize_model_grads=lambda *_: None,
        timer=lambda *_args, **_kwargs: nullcontext(),
        train_one_step=train_step,
        log_step_peak_memory=lambda *_args: nullcontext(),
        torch=SimpleNamespace(
            distributed=SimpleNamespace(get_rank=lambda **_: 0, group=SimpleNamespace(WORLD=object()))
        ),
        mpu=SimpleNamespace(
            get_data_parallel_rank=lambda **_: 1,
            get_tensor_model_parallel_rank=lambda: 0,
            get_pipeline_model_parallel_rank=lambda: 0,
            get_expert_model_parallel_rank=lambda: 0,
            get_context_parallel_rank=lambda: 0,
        ),
    )
    _functions(
        Path("relax/backends/megatron/model.py"),
        {
            "train",
            "should_disable_forward_pre_hook",
            "enable_forward_pre_hook",
            "disable_forward_pre_hook",
            "flush_resident_sft_param_sync",
        },
        namespace,
    )
    optimizer = SimpleNamespace(scale_loss=lambda value: value)
    for rollout in range(3):
        namespace["train"](rollout, [chunk], optimizer, None, [Iterator()], [20])
        assert chunk.current_parameters is (not enabled or rollout == 2)
        if enabled and intermediate_reader and rollout == 0:
            namespace["flush_resident_sft_param_sync"]([chunk])
            assert chunk.current_parameters
    expected_hooks = [False, not intermediate_reader, True] if enabled else [False, False, False]
    assert forward_hooks == expected_hooks
    assert trace.count(("disable", False)) == (1 + int(intermediate_reader) if enabled else 3)
    assert trace.count(("disable", True)) == (1 + int(intermediate_reader) if enabled else 3)
    namespace["flush_resident_sft_param_sync"]([chunk])
    assert not config._relax_resident_sft_hooks_ready


@pytest.mark.parametrize(
    "field,value",
    [
        ("loss_type", "ppo"),
        ("overlap_grad_reduce", False),
        ("overlap_param_gather", False),
        ("keep_old_actor", True),
        ("eval_prompt_data", "eval.jsonl"),
        ("sft_predict_interval", 1),
        ("save", "checkpoint"),
        ("fully_async", True),
        ("custom_megatron_before_train_step_hook_path", "custom.weight_reader"),
        ("moe_shared_expert_overlap", True),
    ],
)
def test_resident_sft_rejects_incompatible_readers(monkeypatch, field, value):
    monkeypatch.setenv("RELAX_SFT_RESIDENT_PARAM_OVERLAP", "1")
    args = _args()
    setattr(args, field, value)
    with pytest.raises(ValueError):
        resident_sft_param_overlap_enabled(args)


@pytest.mark.parametrize("with_backup", [False, True])
def test_resident_sft_reader_flushes_before_backup_and_clears_pending(with_backup):
    events = []

    def flush(model):
        events.append("sync")

    def backup(tag):
        assert events == ["sync"] and tag == "actor"
        events.append("backup")

    actor = SimpleNamespace(model=[], _resident_sft_params_pending=True)
    if with_backup:
        actor.weights_backuper = SimpleNamespace(backup=backup)
    namespace = {"flush_resident_sft_param_sync": flush}
    _functions(
        Path("relax/backends/megatron/actor.py"), {"_flush_resident_sft_parameters"}, namespace, class_method=True
    )
    namespace["_flush_resident_sft_parameters"](actor)
    namespace["_flush_resident_sft_parameters"](actor)
    assert events == (["sync", "backup"] if with_backup else ["sync"])
    assert not actor._resident_sft_params_pending


@pytest.mark.parametrize("active_tag", [None, "actor", "ref"])
@pytest.mark.parametrize("tags", [[], ["actor"], ["actor", "ref"], ["actor", "teacher"]])
def test_resident_sft_actor_checks_actual_backup_tags(monkeypatch, tags, active_tag):
    monkeypatch.setenv("RELAX_SFT_RESIDENT_PARAM_OVERLAP", "1")
    args = _args()
    args._relax_resident_sft_actor_verified = False
    actor = SimpleNamespace(
        args=args,
        role="actor",
        _per_step_rollout=False,
        _active_model_tag=active_tag,
        weights_backuper=SimpleNamespace(backup_tags=tags),
    )
    namespace = {"resident_sft_param_overlap_enabled": resident_sft_param_overlap_enabled}
    _functions(
        Path("relax/backends/megatron/actor.py"), {"_resident_sft_overlap_enabled"}, namespace, class_method=True
    )
    if tags == ["actor"] and active_tag == "actor":
        assert namespace["_resident_sft_overlap_enabled"](actor)
        assert args._relax_resident_sft_actor_verified
    else:
        with pytest.raises(RuntimeError):
            namespace["_resident_sft_overlap_enabled"](actor)
        assert not args._relax_resident_sft_actor_verified


@pytest.mark.parametrize("prepacked", [False, True])
@pytest.mark.parametrize("final_rollout", [False, True])
def test_resident_sft_live_initialization_reaches_train_without_snapshots(monkeypatch, prepacked, final_rollout):
    monkeypatch.setenv("RELAX_SFT_RESIDENT_PARAM_OVERLAP", "1")
    args = _args()
    args._relax_resident_sft_actor_verified = False
    args.compute_advantages_and_returns = False
    args.use_routing_replay = False
    args.use_rollout_routing_replay = False
    actor = SimpleNamespace(
        args=args,
        role="actor",
        with_ref=False,
        with_opd_teacher=False,
        _per_step_rollout=False,
        model=[],
        optimizer=object(),
        opt_param_scheduler=object(),
        rollout_data_postprocess=None,
    )
    iterator = SimpleNamespace(close=Mock())
    data_iterator, num_microbatches = [iterator], [1]
    make_iterator = Mock(return_value=(data_iterator, num_microbatches))
    train = Mock()

    class ReachedAfterTrain(BaseException):
        pass

    actor.prof = SimpleNamespace(step=Mock(side_effect=ReachedAfterTrain))
    namespace = dict(
        self=actor,
        with_ref=False,
        with_opd_teacher=False,
        logger=Mock(),
        TensorBackuper=Mock(),
        should_skip_sft_weight_management=should_skip_sft_weight_management,
        resident_sft_param_overlap_enabled=resident_sft_param_overlap_enabled,
        is_preference_mode=is_preference_mode,
        is_offline_mode=is_offline_mode,
        is_dpo_mode=is_dpo_mode,
        algorithm_needs_critic=lambda _: False,
        get_data_iterator=make_iterator,
        timer=lambda *_: nullcontext(),
        inverse_timer=lambda *_: nullcontext(),
        log_rollout_data=Mock(),
        train=train,
    )
    source = _ROOT / "relax/backends/megatron/actor.py"
    tree = ast.parse(source.read_text())
    init = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_init")
    # Execute the actual weight-management initialization branch, including the
    # real skip predicate; Ray/CUDA bootstrap is outside this CPU regression.
    setup = [
        node
        for node in init.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in {"skip_weight_management", "use_tensor_backuper"}
                for target in node.targets
            )
        )
        or (isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "use_tensor_backuper")
    ]
    assert len(setup) == 3
    exec(compile(ast.Module(body=setup, type_ignores=[]), str(source), "exec"), namespace)
    namespace["TensorBackuper"].create.assert_not_called()
    for name in ("weights_backuper", "_active_model_tag", "weight_updater", "checkpoint_engine_client"):
        assert not hasattr(actor, name)

    _functions(
        Path("relax/backends/megatron/actor.py"),
        {"_resident_sft_overlap_enabled", "_is_standard_dpo", "train_actor"},
        namespace,
        class_method=True,
    )
    actor._resident_sft_overlap_enabled = MethodType(namespace["_resident_sft_overlap_enabled"], actor)
    actor._is_standard_dpo = MethodType(namespace["_is_standard_dpo"], actor)
    rollout_id = args.num_rollout - 1 if final_rollout else 0
    rollout_data = {}
    kwargs = (
        {"prepared_data_iterator": data_iterator, "prepared_num_microbatches": num_microbatches} if prepacked else {}
    )
    with pytest.raises(ReachedAfterTrain):
        namespace["train_actor"](actor, rollout_id, rollout_data, **kwargs)
    train.assert_called_once_with(
        rollout_id, actor.model, actor.optimizer, actor.opt_param_scheduler, data_iterator, num_microbatches
    )
    assert args._relax_resident_sft_actor_verified
    assert actor._resident_sft_params_pending is (not final_rollout)
    iterator.close.assert_called_once()
    if prepacked:
        make_iterator.assert_not_called()
    else:
        make_iterator.assert_called_once_with(args, actor.model, rollout_data)
    for name in ("weights_backuper", "_active_model_tag", "weight_updater", "checkpoint_engine_client"):
        assert not hasattr(actor, name)


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "reference"),
        ("_per_step_rollout", True),
        ("rollout_manager", object()),
        ("with_ref", True),
        ("with_opd_teacher", True),
        ("_active_model_tag", "ref"),
        ("_active_model_tag", "teacher"),
    ],
)
@pytest.mark.parametrize("with_backup", [False, True])
def test_resident_sft_actor_rejects_consumers_and_model_swapping(monkeypatch, field, value, with_backup):
    monkeypatch.setenv("RELAX_SFT_RESIDENT_PARAM_OVERLAP", "1")
    args = _args()
    args._relax_resident_sft_actor_verified = False
    actor = SimpleNamespace(args=args, role="actor", _per_step_rollout=False)
    if with_backup:
        actor.weights_backuper = SimpleNamespace(backup_tags=["actor"])
        actor._active_model_tag = "actor"
    setattr(actor, field, value)
    namespace = dict(
        resident_sft_param_overlap_enabled=resident_sft_param_overlap_enabled,
        should_skip_sft_weight_management=should_skip_sft_weight_management,
    )
    _functions(
        Path("relax/backends/megatron/actor.py"), {"_resident_sft_overlap_enabled"}, namespace, class_method=True
    )
    with pytest.raises(RuntimeError, match="Resident SFT overlap"):
        namespace["_resident_sft_overlap_enabled"](actor)
    assert not args._relax_resident_sft_actor_verified


def test_resident_sft_live_actor_rejects_offload(monkeypatch):
    monkeypatch.setenv("RELAX_SFT_RESIDENT_PARAM_OVERLAP", "1")
    args = _args()
    args._relax_resident_sft_actor_verified = False
    args.offload_train = True
    actor = SimpleNamespace(args=args, role="actor", _per_step_rollout=False)
    namespace = dict(
        resident_sft_param_overlap_enabled=resident_sft_param_overlap_enabled,
        should_skip_sft_weight_management=should_skip_sft_weight_management,
    )
    _functions(
        Path("relax/backends/megatron/actor.py"), {"_resident_sft_overlap_enabled"}, namespace, class_method=True
    )
    with pytest.raises(RuntimeError, match="Resident SFT overlap"):
        namespace["_resident_sft_overlap_enabled"](actor)
    assert not args._relax_resident_sft_actor_verified
