# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Each GPU test file is independently runnable and owns its deployment."""

import os
import sys

import pytest

from .processes import owned_process


@pytest.fixture
def auto_acceptance(tmp_path):
    model, gpus = os.environ.get("RELAX_LORA_MODEL"), os.environ.get("RELAX_LORA_GPUS")
    if not model or not gpus:
        pytest.skip("opt-in GPU deployment: set RELAX_LORA_MODEL and RELAX_LORA_GPUS=2,3")
    selected = gpus.split(",")
    if len(selected) != 2:
        pytest.fail("RELAX_LORA_GPUS must specify exactly two physical GPU indices")

    def run(task):
        from .__main__ import wait_child

        log = tmp_path / (task + ".log")
        with owned_process(
            [
                sys.executable,
                "-m",
                "tests.engine.lora.acceptance",
                "--model",
                model,
                "--gpus",
                *selected,
                "--tasks",
                task,
                "--output",
                str(tmp_path / task),
                *(["--deterministic"] if os.environ.get("RELAX_LORA_DETERMINISTIC") == "1" else []),
            ],
            dict(os.environ),
            log,
            [],
        ) as process:
            wait_child(process, 7200, task, log)

    return run
