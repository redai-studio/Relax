# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``Controller.register_all_serve`` validates inference placement before the
first placement group or engine is created.

The managed OPD teacher is the first thing ``register_all_serve`` starts, so
the stub that replaces it doubles as the "startup went on" marker.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import transfer_queue

from relax.distributed.ray.placement_planner import PlacementError
from tests.core.controller_test_utils import load_controller_with_stubbed_dependencies


if not hasattr(transfer_queue, "StreamingTokenBudgetSampler"):
    pytest.skip(
        "controller tests require a TransferQueue build with StreamingTokenBudgetSampler",
        allow_module_level=True,
    )

controller = load_controller_with_stubbed_dependencies("_test_controller_placement_preflight_controller")


class _TeacherStartReached(Exception):
    pass


def _genrm_spec(num_gpus):
    return {
        "model_path": "/judge",
        "num_gpus": num_gpus,
        "num_gpus_per_engine": 1,
        "engine_config": {},
        "sampling_config": {},
    }


def _colocate_teacher_config(**overrides):
    values = dict(
        colocate=True,
        hybrid=False,
        fully_async=False,
        rollout_num_gpus=4,
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="/teacher",
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
    )
    values.update(overrides)
    return Namespace(**values)


def _stub_startup(monkeypatch):
    teacher_starts = []

    def fake_start_teacher(config, *, runtime_env=None):
        teacher_starts.append(config)
        raise _TeacherStartReached

    monkeypatch.setattr(controller, "validate_ppo_config", lambda config: None)
    monkeypatch.setattr(controller, "maybe_start_managed_opd_teacher", fake_start_teacher)
    return teacher_starts


def test_controller_rejects_conflicting_placement_before_teacher_start(monkeypatch):
    teacher_starts = _stub_startup(monkeypatch)
    config = _colocate_teacher_config(
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4], "genrm": [1, 4]},
        _genrm_instances_resolved={"__default__": _genrm_spec(4)},
    )

    with pytest.raises(PlacementError, match="genrm/__default__.*teacher/__default__"):
        controller.Controller.register_all_serve(SimpleNamespace(config=config, runtime_env=None))

    assert teacher_starts == []


def test_controller_runs_preflight_before_teacher_start(monkeypatch):
    teacher_starts = _stub_startup(monkeypatch)
    config = _colocate_teacher_config()

    with pytest.raises(_TeacherStartReached):
        controller.Controller.register_all_serve(SimpleNamespace(config=config, runtime_env=None))

    assert teacher_starts == [config]
