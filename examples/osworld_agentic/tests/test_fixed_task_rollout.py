# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU coverage checks using the real examples hook and actual transfer
function."""

import ast
import asyncio
import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def hook(monkeypatch):
    name = "fixed_task_rollout_under_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples/osworld_agentic/fixed_task_rollout.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def group(task, group_index, group_size=8, turns=2, index_base=None):
    if index_base is None:
        index_base = group_index * 8
    return [
        SimpleNamespace(metadata={"task_id": task}, group_index=group_index, index=index_base + sample, reward=0.0)
        for sample in range(group_size)
        for _ in range(turns)
    ]


def install_original(monkeypatch, generate=None, reward=None):
    if generate is not None:
        module = ModuleType("relax.agentic.rollout")
        module.generate_rollout = generate

        async def prepare_group(group_input, launch_decision=None):
            return None

        runtime = SimpleNamespace(prepare_group=prepare_group)
        module.get_agentic_resident_pipeline = lambda: SimpleNamespace(runtime_domain=runtime)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    module = ModuleType("examples.osworld_agentic.app.reward_post_process")
    module.post_process_rewards = reward or (lambda args, samples: [sample.reward for sample in samples])
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda value: value))
    logs = ModuleType("relax.utils.logging_utils")
    logs.get_logger = lambda name: SimpleNamespace(info=lambda *args: None)
    monkeypatch.setitem(sys.modules, logs.__name__, logs)


def source_for(task_ids):
    calls = []
    return SimpleNamespace(
        begin_rollout_step=SimpleNamespace(remote=lambda step: (calls.append(("begin", step)), task_ids)[1]),
        finish_rollout_step=SimpleNamespace(remote=lambda step: calls.append(("finish", step))),
        calls=calls,
    )


def actual_transfer(monkeypatch, hook):
    """Extract the real coroutine, avoiding Ray/SGLang imports on a CPU
    host."""
    source = ROOT / "relax/agentic/pipeline/transfer.py"
    tree = ast.parse(source.read_text())
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_transfer_batch_to_data_system"
        )
    )
    namespace = {
        "Any": Any,
        "Namespace": SimpleNamespace,
        "Sample": Any,
        "mark_sample_agentic_event": lambda *args: None,
        "_release_transferred_pixels": lambda samples: None,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    utils = ModuleType("relax.utils.utils")
    utils.convert_samples_to_train_data = lambda args, samples: {"rewards": hook.post_process_rewards(args, samples)}
    utils.build_rollout_custom_meta = lambda data: []
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    return namespace[function.name]


def test_fixed_tasks_two_steps_cover_74_once_with_small_real_transfers(hook, monkeypatch):
    task_ids = tuple((f"task-{index}" for index in range(74)))
    source = source_for(task_ids)
    args = SimpleNamespace(n_samples_per_prompt=8)
    transfer = actual_transfer(monkeypatch, hook)
    puts = []

    class Client:
        async def async_put(self, **kwargs):
            if kwargs["is_last"]:
                hook._ACTIVE_STEP.assert_complete()
            puts.append((kwargs["partition_id"], kwargs["is_last"], len(kwargs["data"]["rewards"])))

    client = Client()

    def generate(args, step, data_source, data_system_client, evaluation=False):
        groups = [group(task, step * 74 + index) for index, task in enumerate(task_ids)]
        random.Random(step + 42).shuffle(groups)

        async def transfer_all():
            for start in range(0, 74, 9):
                await transfer(
                    args=args,
                    batch_samples=groups[start : start + 9],
                    rollout_id=step,
                    data_system_client=data_system_client,
                    is_last=start + 9 >= 74,
                )

        asyncio.run(transfer_all())
        return SimpleNamespace(samples=groups)

    install_original(monkeypatch, generate=generate)
    for step in (17, 18):
        output = hook.generate_rollout(args, step, source, client)
        assert len(output.samples) == 74
        assert hook._ACTIVE_STEP is None
    assert source.calls == [("begin", 17), ("finish", 17), ("begin", 18), ("finish", 18)]
    assert len(puts) == 18
    assert sum((last for _, last, _ in puts)) == 2
    assert max((count for _, _, count in puts)) == 9 * 8 * 2


@pytest.mark.parametrize("count", [7, 9])
def test_fixed_tasks_reject_wrong_unique_trajectory_count(hook, count):
    state = hook._StepValidation(0, ("a",), 8)
    with pytest.raises(RuntimeError, match="unique trajectories"):
        state.accept_batch(group("a", 0, group_size=count, turns=3))
    assert state.groups == {}


@pytest.mark.parametrize("task_ids, size", [((), 8), (("a", "a"), 8), (("",), 8), (("a",), 0)])
def test_fixed_tasks_reject_invalid_plan(hook, task_ids, size):
    with pytest.raises(ValueError):
        hook._StepValidation(0, task_ids, size)


def test_fixed_tasks_reject_duplicate_task_within_transfer(hook):
    state = hook._StepValidation(0, ("a", "b"), 8)
    with pytest.raises(RuntimeError, match="duplicate task"):
        state.accept_batch(group("a", 0) + group("a", 1))
    assert state.groups == {}


def test_fixed_tasks_reject_duplicate_task_across_transfers(hook):
    state = hook._StepValidation(0, ("a", "b"), 8)
    state.accept_batch(group("a", 0))
    with pytest.raises(RuntimeError, match="duplicate task"):
        state.accept_batch(group("a", 1))
    assert len(state.groups) == 1


def test_fixed_tasks_reject_repeated_group(hook):
    state = hook._StepValidation(0, ("a", "b"), 8)
    state.accept_batch(group("a", 0))
    with pytest.raises(RuntimeError, match="more than once"):
        state.accept_batch(group("a", 0))


def test_fixed_tasks_reject_cross_group_sample_identity(hook):
    state = hook._StepValidation(0, ("a", "b"), 8)
    with pytest.raises(RuntimeError, match="reuses Sample.index"):
        state.accept_batch(group("a", 0) + group("b", 1, index_base=0))


def test_fixed_tasks_reject_mixed_task_group(hook):
    state = hook._StepValidation(0, ("a", "b"), 8)
    mixed = group("a", 0)
    mixed[-1].metadata["task_id"] = "b"
    with pytest.raises(RuntimeError, match="mixes multiple"):
        state.accept_batch(mixed)


@pytest.mark.parametrize(
    "field,value", [("index", None), ("index", True), ("index", -1), ("group_index", None), ("group_index", True)]
)
def test_fixed_tasks_reject_missing_or_invalid_identity(hook, field, value):
    state = hook._StepValidation(0, ("a",), 8)
    samples = group("a", 0)
    setattr(samples[0], field, value)
    with pytest.raises(RuntimeError, match="nonnegative integer"):
        state.accept_batch(samples)


def test_fixed_tasks_reject_unexpected_task_and_empty_transfer(hook):
    state = hook._StepValidation(0, ("a",), 8)
    with pytest.raises(RuntimeError, match="unexpected task"):
        state.accept_batch(group("outside", 0))
    with pytest.raises(RuntimeError, match="empty sample"):
        state.accept_batch([])


def test_fixed_tasks_preserve_zero_rewards_and_multiturn_outputs(hook, monkeypatch):
    args = SimpleNamespace(n_samples_per_prompt=8)
    state = hook._StepValidation(0, ("a",), 8)
    hook._ACTIVE_STEP = state
    original_result = ([0.0] * 24, [0.0] * 24)
    install_original(monkeypatch, reward=lambda args, samples: original_result)
    samples = group("a", 0, turns=3)
    assert hook.post_process_rewards(args, samples) is original_result
    state.assert_complete()
    state.validate_output([samples])
    assert len(state.sample_indices) == 8


def test_fixed_tasks_bad_final_batch_never_calls_reward_or_put(hook, monkeypatch):
    args = SimpleNamespace(n_samples_per_prompt=8)
    hook._ACTIVE_STEP = hook._StepValidation(0, ("a", "b"), 8)
    calls = []
    install_original(monkeypatch, reward=lambda args, samples: calls.append("reward") or [])
    transfer = actual_transfer(monkeypatch, hook)

    class Client:
        async def async_put(self, **kwargs):
            calls.append("put")

    asyncio.run(
        transfer(
            args=args,
            batch_samples=[SimpleNamespace(samples=group("a", 0)).samples],
            rollout_id=0,
            data_system_client=Client(),
        )
    )
    assert calls == ["reward", "put"]
    with pytest.raises(RuntimeError, match="duplicate task"):
        asyncio.run(
            transfer(args=args, batch_samples=[group("a", 1)], rollout_id=0, data_system_client=Client(), is_last=True)
        )
    assert calls == ["reward", "put"]


def test_fixed_tasks_incomplete_or_changed_output_fails(hook):
    state = hook._StepValidation(0, ("a", "b"), 8)
    a, b = (group("a", 0), group("b", 1))
    state.accept_batch(a)
    with pytest.raises(RuntimeError, match="incomplete"):
        state.assert_complete()
    state.accept_batch(b)
    with pytest.raises(RuntimeError, match="incorrect number"):
        state.validate_output([a])
    with pytest.raises(RuntimeError, match="differs"):
        state.validate_output([a, group("b", 2)])


def test_fixed_tasks_controlled_drop_propagates_without_finishing_or_leaking_state(hook, monkeypatch):
    args = SimpleNamespace(n_samples_per_prompt=8)
    source = source_for(("a", "b"))

    def dropped(*args, **kwargs):
        raise RuntimeError("fixed task source exhausted after a dropped group")

    install_original(monkeypatch, generate=dropped)
    with pytest.raises(RuntimeError, match="source exhausted"):
        hook.generate_rollout(args, 0, source)
    assert source.calls == [("begin", 0)]
    assert hook._ACTIVE_STEP is None


def test_fixed_tasks_missing_transfer_validation_cannot_finish_step(hook, monkeypatch):
    source = source_for(("a",))
    install_original(monkeypatch, generate=lambda *args, **kwargs: SimpleNamespace(samples=[group("a", 0)]))
    with pytest.raises(RuntimeError, match="incomplete"):
        hook.generate_rollout(SimpleNamespace(n_samples_per_prompt=8), 0, source)
    assert source.calls == [("begin", 0)]
    assert hook._ACTIVE_STEP is None


def test_fixed_tasks_eval_delegates_without_opening_training_step(hook, monkeypatch):
    source = source_for(("a",))
    sentinel = object()
    calls = []
    install_original(monkeypatch, generate=lambda *args, **kwargs: calls.append(kwargs) or sentinel)

    def unexpected_pipeline_lookup():
        pytest.fail("Evaluation must not install training retries")

    sys.modules["relax.agentic.rollout"].get_agentic_resident_pipeline = unexpected_pipeline_lookup
    assert hook.generate_rollout(object(), 3, source, evaluation=True) is sentinel
    assert calls == [{"evaluation": True}]
    assert source.calls == []


def test_fixed_tasks_reward_hook_requires_active_wrapper(hook, monkeypatch):
    install_original(monkeypatch)
    with pytest.raises(RuntimeError, match="requires the fixed-task rollout wrapper"):
        hook.post_process_rewards(object(), group("a", 0))


def test_training_installs_retry_before_generation_and_reuses_it(hook, monkeypatch):
    source = source_for(("a",))
    installed_methods = []

    def generate(args, rollout_id, *unused, **kwargs):
        runtime = sys.modules["relax.agentic.rollout"].get_agentic_resident_pipeline().runtime_domain
        assert runtime._osworld_fixed_task_prepare_retry_attempts == 3
        installed_methods.append(runtime.prepare_group)
        samples = group("a", 0)
        hook._ACTIVE_STEP.accept_batch(samples)
        return SimpleNamespace(samples=[samples])

    install_original(monkeypatch, generate=generate)
    for step in range(2):
        hook.generate_rollout(SimpleNamespace(n_samples_per_prompt=8), step, source)
    assert installed_methods[0] is installed_methods[1]
    assert source.calls == [("begin", 0), ("finish", 0), ("begin", 1), ("finish", 1)]
