# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import shlex

from relax.utils.external import command_utils


def test_execute_train_preserves_equals_in_extra_env_values(monkeypatch):
    monkeypatch.setenv("SLIME_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("SLIME_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setattr(command_utils, "check_has_nvlink", lambda: False)
    data_url = "https://example.com/data?split=train&format=jsonl&label=reader's"
    commands = []
    monkeypatch.setattr(command_utils, "exec_command", commands.append)

    command_utils.execute_train(
        train_args="--loss-type sft",
        num_gpus_per_node=1,
        megatron_model_type=None,
        config=command_utils.ExecuteTrainConfig(
            extra_env_vars=f"DATA_URL={data_url} PADDING=YQ== EMPTY="
        ),
    )

    runtime_env_option = next(
        argument for argument in shlex.split(commands[-1]) if argument.startswith("--runtime-env-json=")
    )
    env_vars = json.loads(runtime_env_option.split("=", 1)[1])["env_vars"]
    assert env_vars["DATA_URL"] == data_url
    assert env_vars["PADDING"] == "YQ=="
    assert env_vars["EMPTY"] == ""
