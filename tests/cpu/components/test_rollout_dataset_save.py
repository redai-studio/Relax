# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import errno
import os
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from ray.exceptions import RayTaskError

from relax.components.recovery import RecoveryGate
from relax.components.rollout import Rollout
from relax.core.controller import Controller


@pytest.fixture
def rollout(tmp_path):
    component = object.__new__(Rollout.func_or_class.__mro__[1])
    component.config = Namespace(
        use_health_check=True,
        inference_recovery=False,
        eval_interval=None,
        num_rollout=1,
        offload_rollout=False,
        save=str(tmp_path),
        save_interval=100,
        rotate_ckpt=False,
        rollout_global_dataset=False,
    )
    component._recovery_gate = RecoveryGate(component.config, "rollout")
    component.healthy = SimpleNamespace(
        update_heartbeat=SimpleNamespace(remote=Mock()), report_error=SimpleNamespace(remote=Mock())
    )
    component.rollout_manager = SimpleNamespace(generate=SimpleNamespace(remote=AsyncMock(return_value=True)))
    component.data_system_client = SimpleNamespace(async_get_partition_list=AsyncMock(return_value=[]))
    component.data_source = SimpleNamespace(save=SimpleNamespace(remote=AsyncMock()))
    component.step = 0
    component.status = "running"
    component._logger_instance = None
    component._run_task = None
    component._stop_event = asyncio.Event()
    component._eval_error = None
    component._eval_handlers = {}
    return component


@pytest.mark.parametrize(
    ("storage_errno", "inference_recovery"),
    [(errno.EDQUOT, False), (errno.ENOSPC, True)],
    ids=["remote-quota-error", "local-recovery-disk-full"],
)
async def test_rollout_dataset_save_exhaustion_ends_training(rollout, storage_errno, inference_recovery):
    cause = OSError(storage_errno, os.strerror(storage_errno))
    error = (
        cause if inference_recovery else RayTaskError("save", "Traceback: save failed", cause).as_instanceof_cause()
    )
    rollout.data_source.save.remote.side_effect = error
    rollout.config.inference_recovery = inference_recovery

    response = rollout.run()
    if inference_recovery:
        controller = object.__new__(Controller)
        controller._inference_recovery = object()
        controller._restart_lock = threading.Lock()
        controller._check_inference_recovery = Mock(side_effect=AssertionError("Unexpected inference recovery"))
        response = controller._supervise_inference("rollout", response, asyncio.Lock())

    with pytest.raises(RuntimeError, match=os.strerror(storage_errno)) as failure:
        await response

    assert failure.value.__cause__ is error
    assert rollout.step == 0
    rollout.healthy.report_error.remote.assert_not_called()


async def test_other_rollout_dataset_save_errors_reach_health_checker(rollout):
    rollout.data_source.save.remote.side_effect = OSError(errno.EIO, "Transient I/O failure")

    await rollout.run()

    rollout.healthy.report_error.remote.assert_called_once()
    role, message = rollout.healthy.report_error.remote.call_args.args
    assert role == "rollout"
    assert "Transient I/O failure" in message
    assert rollout.step == 0
