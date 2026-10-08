# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Framework-run deferred scoring: a finished batch is scored in the score
phase, after rollout released the GPUs and before anything is published."""

from __future__ import annotations

import ast
import asyncio
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import ray

import relax.engine.rewards as rewards_module
from relax.distributed.ray.placement_planner import plan_placement
from relax.engine.inference import deferred
from relax.engine.inference.lifecycle import LifecycleCoordinator, LifecycleParticipant
from relax.utils import utils as relax_utils
from relax.utils.types import Sample


def _args(**overrides) -> Namespace:
    """Rollout and a deferred GenRM sharing the same eight bundles."""
    values = dict(
        colocate=True,
        hybrid=False,
        rollout_num_gpus=8,
        resource={"actor": [1, 8], "rollout": [1, 8], "genrm": [1, 8]},
        _genrm_instances_resolved={"__default__": {"num_gpus": 8}},
        _genrm_colocate_with_rollout=True,
        defer_reward_to_post_process=True,
        custom_reward_post_process_path=None,
        group_rm=False,
        custom_rm_path=None,
    )
    values.update(overrides)
    return Namespace(**values)


def _coordinator(args: Namespace, events: list[str]) -> LifecycleCoordinator:
    def switch(event: str):
        def run() -> list:
            events.append(event)
            return []

        return run

    participants = [
        LifecycleParticipant("rollout", "__default__", switch("rollout offload"), switch("rollout onload")),
        LifecycleParticipant("genrm", "__default__", switch("genrm offload"), switch("genrm onload")),
    ]
    return LifecycleCoordinator(plan_placement(args), participants, wait=lambda refs: None)


@pytest.fixture
def judge(monkeypatch):
    """Stands in for the configured reward function; records what it scored."""
    state = SimpleNamespace(events=[], scored=[], fail=None, gate=None)

    async def async_rm(args, sample, **kwargs):
        state.events.append(f"score {sample.index}")
        if state.gate is not None:
            await state.gate.wait()
        if state.fail is not None:
            raise state.fail
        state.scored.append(sample.index)
        return float(sample.index)

    async def batched_async_rm(args, samples, **kwargs):
        state.events.append(f"score group {[sample.index for sample in samples]}")
        if state.fail is not None:
            raise state.fail
        state.scored.extend(sample.index for sample in samples)
        return [float(sample.index) for sample in samples]

    monkeypatch.setattr(rewards_module, "async_rm", async_rm)
    monkeypatch.setattr(rewards_module, "batched_async_rm", batched_async_rm)
    return state


def _groups(*sizes: int) -> list[list[Sample]]:
    groups, index = [], 0
    for size in sizes:
        groups.append([Sample(index=index + i, response="answer") for i in range(size)])
        index += size
    return groups


async def test_deferred_scoring_offloads_rollout_before_loading_genrm(judge):
    args, groups = _args(), _groups(2)
    coordinator = _coordinator(args, judge.events)

    await deferred.run_deferred_scoring(args, groups, coordinator=coordinator)

    assert judge.events == ["rollout offload", "genrm onload", "score 0", "score 1", "genrm offload"]
    assert [sample.reward for sample in groups[0]] == [0.0, 1.0]


async def test_deferred_scoring_fills_missing_rewards_only(judge):
    args, groups = _args(), _groups(2, 2)
    groups[0][1].reward = 0.5  # assigned while it generated
    groups[1][0].reward = 0.25

    await deferred.run_deferred_scoring(args, groups, coordinator=_coordinator(args, judge.events))

    assert sorted(judge.scored) == [0, 3]
    assert [sample.reward for group in groups for sample in group] == [0.0, 0.5, 0.25, 3.0]


async def test_deferred_scoring_does_not_switch_when_every_sample_has_a_reward(judge):
    args, groups = _args(), _groups(2)
    for sample in groups[0]:
        sample.reward = 1.0

    await deferred.run_deferred_scoring(args, groups, coordinator=_coordinator(args, judge.events))

    assert judge.events == []


async def test_deferred_scoring_scores_whole_groups_for_a_group_reward_model(judge):
    args, groups = _args(group_rm=True), _groups(2, 2)
    for sample in groups[1]:
        sample.reward = 9.0  # this group is already scored

    await deferred.run_deferred_scoring(args, groups, coordinator=_coordinator(args, judge.events))

    assert "score group [0, 1]" in judge.events
    assert sorted(judge.scored) == [0, 1]


async def test_deferred_scoring_is_inactive_with_custom_post_process(judge, monkeypatch):
    args, groups = _args(custom_reward_post_process_path="my_module.post_process"), _groups(2)
    monkeypatch.setattr(deferred, "build_local_coordinator", lambda args: pytest.fail("the hook owns the swap"))

    await deferred.run_deferred_scoring(args, groups)

    assert judge.events == []
    assert [sample.reward for sample in groups[0]] == [None, None]
    assert not deferred.is_framework_deferred_reward(args)
    assert not deferred.is_framework_deferred_reward(_args(defer_reward_to_post_process=False))


async def test_deferred_scoring_restores_rollout_after_evaluation(judge):
    args, groups = _args(), _groups(1)

    await deferred.run_deferred_scoring(
        args, groups, restore_rollout=True, coordinator=_coordinator(args, judge.events)
    )

    assert judge.events == ["rollout offload", "genrm onload", "score 0", "genrm offload", "rollout onload"]


async def test_deferred_scoring_puts_the_judge_back_to_sleep_when_scoring_fails(judge):
    args, groups = _args(), _groups(1)
    judge.fail = RuntimeError("judge unreachable")

    with pytest.raises(RuntimeError, match="judge unreachable"):
        await deferred.run_deferred_scoring(args, groups, coordinator=_coordinator(args, judge.events))

    assert judge.events[-1] == "genrm offload"


async def test_deferred_scoring_scores_one_batch_at_a_time(judge):
    args = _args()
    first, second = _groups(1), [[Sample(index=10, response="answer")]]
    judge.gate = asyncio.Event()

    tasks = [
        asyncio.create_task(deferred.run_deferred_scoring(args, batch, coordinator=_coordinator(args, judge.events)))
        for batch in (first, second)
    ]
    await asyncio.sleep(0.05)
    # The first batch is mid-score; the second must not have touched the phase.
    assert judge.events == ["rollout offload", "genrm onload", "score 0"]
    judge.gate.set()
    await asyncio.gather(*tasks)

    # Never interleaved: the judge is not put to sleep under a batch still scoring.
    assert judge.events == [
        "rollout offload",
        "genrm onload",
        "score 0",
        "genrm offload",
        "rollout offload",
        "genrm onload",
        "score 10",
        "genrm offload",
    ]


@pytest.mark.parametrize(
    ("instances", "actor_names"),
    [
        ({"__default__": {"num_gpus": 8}}, ["relax_genrm_manager"]),
        (
            {"quality": {"num_gpus": 4}, "safety": {"num_gpus": 4}},
            ["relax_genrm_manager_quality", "relax_genrm_manager_safety"],
        ),
    ],
)
def test_deferred_scoring_coordinator_switches_rollout_in_process_and_judges_by_name(
    monkeypatch, instances, actor_names
):
    events: list[str] = []
    rollout_manager = SimpleNamespace(
        status="onload",
        _offload_local=lambda: events.append("rollout offload"),
        _onload_local=lambda: events.append("rollout onload"),
    )
    # The real module needs sglang; only its accessor is used here.
    rollout_module = ModuleType("relax.distributed.ray.rollout")
    rollout_module.get_local_rollout_manager = lambda: rollout_manager
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.rollout", rollout_module)

    def get_actor(name):
        def remote(method):
            return SimpleNamespace(remote=lambda: events.append(f"{name} {method}"))

        return SimpleNamespace(onload=remote("onload"), offload=remote("offload"))

    monkeypatch.setattr(ray, "get_actor", get_actor)
    monkeypatch.setattr(ray, "get", lambda refs: None)

    coordinator = deferred.build_local_coordinator(_args(_genrm_instances_resolved=instances))
    coordinator.enter_score()
    coordinator.leave_score()

    # Rollout through its in-process body (a remote call to itself would
    # deadlock), the judges through their managers.
    assert events == [
        "rollout offload",
        *(f"{name} onload" for name in actor_names),
        *(f"{name} offload" for name in actor_names),
    ]


# ----------------------------------------------------------------------
# Through the publishing exit.
# ----------------------------------------------------------------------


class _DataSystem:
    def __init__(self):
        self.published: list[list[Sample]] = []

    async def async_put(self, *, data, partition_id, custom_meta, is_last):
        self.published.append(data.samples)


@pytest.fixture
def publishing(monkeypatch, judge):
    """``transfer_batch_to_data_system`` with the conversion stubbed out."""
    args = _args()
    monkeypatch.setattr(deferred, "build_local_coordinator", lambda args: _coordinator(args, judge.events))
    monkeypatch.setattr(
        relax_utils,
        "convert_samples_to_train_data",
        lambda args, samples: SimpleNamespace(samples=list(samples), numel=lambda: len(samples)),
    )
    monkeypatch.setattr(relax_utils, "build_rollout_custom_meta", lambda batch: {})
    monkeypatch.setattr(relax_utils, "CURRENT_ROLLOUT_BATCH", [])
    return args, _DataSystem()


async def test_deferred_scoring_runs_before_the_batch_is_published(publishing, judge):
    args, data_system = publishing
    groups = _groups(2, 2)

    await relax_utils.transfer_batch_to_data_system(args, groups, 2, 0, data_system)

    assert judge.events[:2] == ["rollout offload", "genrm onload"]
    (published,) = data_system.published
    # Every published sample carries its reward.
    assert [sample.reward for sample in published] == [0.0, 1.0, 2.0, 3.0]


async def test_deferred_scoring_publishes_nothing_when_scoring_fails(publishing, judge):
    args, data_system = publishing
    judge.fail = RuntimeError("judge unreachable")

    with pytest.raises(RuntimeError, match="judge unreachable"):
        await relax_utils.transfer_batch_to_data_system(args, _groups(2, 2), 2, 0, data_system)

    assert data_system.published == []
    assert relax_utils.CURRENT_ROLLOUT_BATCH == []


async def test_deferred_scoring_leaves_inline_rewards_alone_when_not_deferred(publishing, judge):
    _args_unused, data_system = publishing
    groups = _groups(2)
    for sample in groups[0]:
        sample.reward = 1.0

    await relax_utils.transfer_batch_to_data_system(
        _args(defer_reward_to_post_process=False), groups, 1, 0, data_system
    )

    assert judge.events == []
    assert len(data_system.published) == 1


# ----------------------------------------------------------------------
# Configuration.
# ----------------------------------------------------------------------


def test_deferred_scoring_rejects_agentic_rollout():
    with pytest.raises(ValueError, match="not supported with --use-agentic-rollout"):
        deferred.validate_deferred_scoring_args(_args(use_agentic_rollout=True))

    # A userland hook owns the swap itself, and a run without the flag is untouched.
    deferred.validate_deferred_scoring_args(
        _args(use_agentic_rollout=True, custom_reward_post_process_path="my_module.post_process")
    )
    deferred.validate_deferred_scoring_args(_args(use_agentic_rollout=True, defer_reward_to_post_process=False))


# ----------------------------------------------------------------------
# The userland defer script keeps working.
# ----------------------------------------------------------------------


def test_legacy_defer_script_entry_points_exist():
    """examples/generate_reward_model/post_process_genrm_swap.py finds the
    GenRM manager by actor name and offloads rollout in-process.

    Both entry points must stay as the script uses them.
    """
    root = Path(__file__).resolve().parents[3]
    script = (root / "examples/generate_reward_model/post_process_genrm_swap.py").read_text()
    assert 'ray.get_actor("relax_genrm_manager")' in script
    assert "get_local_rollout_manager()" in script and "._offload_local()" in script

    from relax.distributed.ray.placement_group import genrm_manager_actor_name

    assert genrm_manager_actor_name(["__default__"], "__default__") == "relax_genrm_manager"
    assert genrm_manager_actor_name(["quality", "safety"], "safety") == "relax_genrm_manager_safety"

    # rollout.py needs sglang to import, so read it instead.
    rollout = ast.parse((root / "relax/distributed/ray/rollout.py").read_text())
    functions = {node.name: node for node in rollout.body if isinstance(node, ast.FunctionDef)}
    assert functions["get_local_rollout_manager"].args.args == []
    (manager,) = [node for node in rollout.body if isinstance(node, ast.ClassDef) and node.name == "RolloutManager"]
    methods = {node.name: node for node in manager.body if isinstance(node, ast.FunctionDef)}
    assert [arg.arg for arg in methods["_offload_local"].args.args] == ["self"]
