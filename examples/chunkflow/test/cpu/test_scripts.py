# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preview the real recipes without sourcing a cluster entrypoint."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from examples.chunkflow.arguments import add_arguments, validate_args


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "script,model,experts",
    [("run-qwen3-4B-sft-8xgpu.sh", "Qwen3-4B", "1"), ("run-qwen36-35B-A3B-sft-8xgpu.sh", "Qwen3.6-35B-A3B", "8")],
)
def test_recipe_preview_has_recoverable_paths_and_supported_configuration(tmp_path, script, model, experts):
    from argparse import ArgumentParser

    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MODEL_DIR": str(tmp_path / "models with spaces"),
        "DATA_DIR": str(tmp_path / "data"),
        "SAVE_DIR": str(tmp_path / "checkpoints"),
        "EXP_NAME": "resume-test",
        "CHUNK_SIZE": "1024",
        "KEEP_LAST_K": "2",
    }
    command = ["bash", str(ROOT / "examples/chunkflow/scripts" / script)]
    outputs = [subprocess.check_output(command, env=env, text=True) for _ in range(2)]
    for output in outputs:
        argv = shlex.split(output)

        def value(option):
            return argv[argv.index(option) + 1]

        assert argv[:3] == ["python3", "-m", "relax.entrypoints.train"]
        assert value("--hf-checkpoint") == str(tmp_path / "models with spaces" / model)
        assert value("--load") == value("--save") == str(tmp_path / "checkpoints/resume-test")
        assert value("--context-parallel-size") == value("--pipeline-model-parallel-size") == "1"
        assert value("--expert-model-parallel-size") == experts
        assert "--recompute-granularity" not in argv
        args, _ = add_arguments(ArgumentParser()).parse_known_args(argv[3:])
        validate_args(args)
        assert args.use_chunkflow and args.chunkflow_chunk_size == 1024 and args.chunkflow_keep_last_k == 2
        if experts != "1":
            assert (ROOT / value("--custom-config-path")).is_file()
            assert value("--moe-router-load-balancing-type") == "none"
