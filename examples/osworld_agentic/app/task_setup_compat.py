# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Compatibility for three tasks whose setup assumes the desktop user's home
cwd."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_HOME_CWD_TASK_IDS = frozenset(
    {
        "4d117223-a354-47fb-8b45-62ab1390a95f",
        "5c1075ca-bb34-46a3-a7a0-029bd7463e79",
        "6f56bf42-85b8-4fbb-8e06-6c44960184ba",
    }
)
_SCRIPT_PATHS = {
    "setup.sh": "/home/user/setup.sh",
    "eval.sh": "/home/user/eval.sh",
    "eval_p1.sh": "/home/user/eval_p1.sh",
}
_SHELL_COMMANDS = {
    "chmod +x setup.sh": "chmod +x /home/user/setup.sh",
    "./setup.sh": "cd /home/user && ./setup.sh",
    "chmod +x eval.sh": "chmod +x /home/user/eval.sh",
    "bash eval.sh": "cd /home/user && bash eval.sh",
    "chmod +x eval_p1.sh": "chmod +x /home/user/eval_p1.sh",
    "bash eval_p1.sh": "cd /home/user && bash eval_p1.sh",
}
_GUI_CD_BY_TASK = {
    "4d117223-a354-47fb-8b45-62ab1390a95f": (
        "pyautogui.write('cd testDir')",
        "pyautogui.write('cd /home/user/testDir')",
    ),
    "5c1075ca-bb34-46a3-a7a0-029bd7463e79": (
        "pyautogui.write('cd test_environment')",
        "pyautogui.write('cd /home/user/test_environment')",
    ),
}


def _normalize_execute(parameters: dict[str, Any], *, gui_cd: tuple[str, str] | None = None) -> None:
    command = parameters.get("command")
    if parameters.get("shell") is True and isinstance(command, str):
        parameters["command"] = _SHELL_COMMANDS.get(command, command)
    elif (
        gui_cd is not None
        and isinstance(command, list)
        and len(command) == 3
        and command[:2] == ["python", "-c"]
        and isinstance(command[2], str)
    ):
        # Match the actual task's initialization call, leaving the rest of its
        # Python program (including terminal launch, waits and clear) unchanged.
        command[2] = command[2].replace(*gui_cd)


def _normalize_stages(stages: Any, *, gui_cd: tuple[str, str] | None = None) -> None:
    if not isinstance(stages, list):
        return
    for stage in stages:
        if not isinstance(stage, dict) or not isinstance(stage.get("parameters"), dict):
            continue
        parameters = stage["parameters"]
        if stage.get("type") == "download":
            for item in parameters.get("files", []):
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    item["path"] = _SCRIPT_PATHS.get(item["path"], item["path"])
        elif stage.get("type") == "execute":
            _normalize_execute(parameters, gui_cd=gui_cd)


def _normalize_evaluator(evaluator: Any) -> None:
    if not isinstance(evaluator, dict):
        return
    _normalize_stages(evaluator.get("postconfig"))
    result = evaluator.get("result")
    if isinstance(result, dict) and result.get("type") == "vm_command_line":
        _normalize_execute(result)


def normalize_task_setup(task_config: dict[str, Any]) -> dict[str, Any]:
    """Return an isolated config copy with the allowlisted task's paths
    aligned.

    The Nex server runs as uid 1000 with cwd '/', which is not writable. These
    tasks use relative script paths and working directories, while the desktop
    user's home is /home/user. Align setup, GUI and evaluator entry points
    without changing downloaded script contents, task instructions, or
    evaluator expectations. Other tasks are copied without modification.
    Applying this twice is safe.
    """
    normalized = deepcopy(task_config)
    task_id = normalized.get("id")
    if task_id not in _HOME_CWD_TASK_IDS:
        return normalized
    _normalize_stages(normalized.get("config"), gui_cd=_GUI_CD_BY_TASK.get(task_id))
    _normalize_evaluator(normalized.get("evaluator"))
    evaluator_list = normalized.get("evaluator_list")
    if isinstance(evaluator_list, dict):
        for evaluator in evaluator_list.values():
            _normalize_evaluator(evaluator)
    return normalized
