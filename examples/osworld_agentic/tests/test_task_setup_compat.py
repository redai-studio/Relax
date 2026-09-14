# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU path contracts using the affected task's observed configuration
shape."""

from __future__ import annotations

import copy
import subprocess
import tempfile
import unittest
from pathlib import Path

from examples.osworld_agentic.app.task_setup_compat import normalize_task_setup


TASK_ID = "4d117223-a354-47fb-8b45-62ab1390a95f"
ENVIRONMENT_TASK_ID = "5c1075ca-bb34-46a3-a7a0-029bd7463e79"
HOME_TASK_ID = "6f56bf42-85b8-4fbb-8e06-6c44960184ba"
URL_PREFIX = "https://huggingface.co/datasets/Gen-Verse/perturb_ubuntu_osworld_file_cache/resolve/main/os/"
OPEN_TERMINAL = (
    "import pyautogui; import time; time.sleep(0.5); "
    "pyautogui.click({SCREEN_WIDTH_HALF}, {SCREEN_HEIGHT_HALF}); time.sleep(0.5); "
    "pyautogui.hotkey('ctrl', 'alt', 't'); time.sleep(0.5)"
)
ENTER_DIRECTORY = (
    "import pyautogui; import time; time.sleep(0.5); pyautogui.write('cd testDir'); "
    "time.sleep(0.5); pyautogui.press('enter'); time.sleep(0.5); "
    "pyautogui.write('clear'); time.sleep(0.5); pyautogui.press('enter')"
)


def _download(name, task_id=TASK_ID):
    return {
        "type": "download",
        "parameters": {"files": [{"url": URL_PREFIX + task_id + "/" + name, "path": name}]},
    }


def _execute(command, shell=True):
    parameters = {"command": command}
    if shell:
        parameters["shell"] = True
    return {"type": "execute", "parameters": parameters}


def _actual_shape():
    evaluator = {
        "postconfig": [_download("eval.sh"), _execute("chmod +x eval.sh")],
        "func": "check_include_exclude",
        "result": {"type": "vm_command_line", "command": "bash eval.sh", "shell": True},
        "expected": {
            "type": "rule",
            "rules": {"include": ["All files have the correct permissions."], "exclude": []},
        },
    }
    return {
        "id": TASK_ID,
        "snapshot": "os",
        "prompt": "Fixture sentinel: setup.sh, eval.sh and cd testDir must not be rewritten here.",
        "task_template": "Fixture sentinel: cd testDir",
        "reward": {"unchanged": True},
        "config": [
            _download("setup.sh"),
            _execute("chmod +x setup.sh"),
            _execute("./setup.sh"),
            _execute(["python", "-c", OPEN_TERMINAL], shell=False),
            {"type": "activate_window", "parameters": {"window_name": "Terminal"}},
            _execute(["python", "-c", ENTER_DIRECTORY], shell=False),
        ],
        "evaluator": evaluator,
        "evaluator_list": {"evaluator1": copy.deepcopy(evaluator)},
    }


def _additional_shape(task_id):
    task = _actual_shape()
    task["id"] = task_id
    task["config"] = [_download("setup.sh", task_id), _execute("chmod +x setup.sh")]
    evaluator = task["evaluator"]
    if task_id == ENVIRONMENT_TASK_ID:
        task["config"].extend([_download("eval.sh", task_id), _execute("chmod +x eval.sh")])
        evaluator.pop("postconfig")
    else:
        evaluator["postconfig"][0] = _download("eval.sh", task_id)
    task["config"].extend(
        [
            _execute("./setup.sh"),
            _execute(["python", "-c", OPEN_TERMINAL], shell=False),
            {"type": "activate_window", "parameters": {"window_name": "Terminal"}},
        ]
    )
    if task_id == ENVIRONMENT_TASK_ID:
        task["config"].append(
            _execute(["python", "-c", ENTER_DIRECTORY.replace("cd testDir", "cd test_environment")], shell=False)
        )
    task["evaluator_list"] = {"evaluator1": copy.deepcopy(evaluator)}
    if task_id == HOME_TASK_ID:
        second = copy.deepcopy(evaluator)
        second["postconfig"] = [_download("eval_p1.sh", task_id), _execute("chmod +x eval_p1.sh")]
        second["result"]["command"] = "bash eval_p1.sh"
        second["expected"] = {"type": "rule", "rules": {"include": ["phase-one sentinel"], "exclude": []}}
        task["evaluator_list"]["evaluator2"] = second
    return task


class TestTaskSetupCompatibility(unittest.TestCase):
    def test_setup_and_gui_share_absolute_home_workspace(self):
        original = _actual_shape()
        actual = normalize_task_setup(original)
        stages = actual["config"]
        self.assertEqual(stages[0]["parameters"]["files"][0]["path"], "/home/user/setup.sh")
        self.assertEqual(stages[1]["parameters"]["command"], "chmod +x /home/user/setup.sh")
        self.assertEqual(stages[2]["parameters"]["command"], "cd /home/user && ./setup.sh")
        self.assertEqual(stages[3], original["config"][3])
        self.assertEqual(stages[4], original["config"][4])
        self.assertEqual(
            stages[5]["parameters"]["command"],
            ["python", "-c", ENTER_DIRECTORY.replace("'cd testDir'", "'cd /home/user/testDir'")],
        )

    def test_both_evaluator_locations_use_same_script_and_cwd(self):
        original = _actual_shape()
        actual = normalize_task_setup(original)
        for evaluator in (actual["evaluator"], actual["evaluator_list"]["evaluator1"]):
            self.assertEqual(evaluator["postconfig"][0]["parameters"]["files"][0]["path"], "/home/user/eval.sh")
            self.assertEqual(evaluator["postconfig"][1]["parameters"]["command"], "chmod +x /home/user/eval.sh")
            self.assertEqual(evaluator["result"]["command"], "cd /home/user && bash eval.sh")
            self.assertEqual(evaluator["func"], original["evaluator"]["func"])
            self.assertEqual(evaluator["expected"], original["evaluator"]["expected"])

    def test_deepcopy_does_not_mutate_original_or_share_nested_state(self):
        original = _actual_shape()
        before = copy.deepcopy(original)
        actual = normalize_task_setup(original)
        self.assertEqual(original, before)
        self.assertIsNot(actual, original)
        self.assertIsNot(actual["config"], original["config"])
        actual["evaluator_list"]["evaluator1"]["expected"]["rules"]["include"].append("sentinel")
        self.assertEqual(original, before)

    def test_idempotent_and_returns_new_copy_each_time(self):
        first = normalize_task_setup(_actual_shape())
        second = normalize_task_setup(first)
        self.assertEqual(second, first)
        self.assertIsNot(second, first)
        self.assertIsNot(second["evaluator"], first["evaluator"])

    def test_non_allowlisted_task_is_unchanged_copy(self):
        original = _actual_shape()
        original["id"] = "not-one-of-the-three-allowlisted-tasks"
        actual = normalize_task_setup(original)
        self.assertEqual(actual, original)
        self.assertIsNot(actual, original)
        self.assertIsNot(actual["config"], original["config"])

    def test_environment_task_downloads_both_scripts_and_opens_absolute_directory(self):
        original = _additional_shape(ENVIRONMENT_TASK_ID)
        actual = normalize_task_setup(original)
        stages = actual["config"]
        self.assertEqual(stages[0]["parameters"]["files"][0]["path"], "/home/user/setup.sh")
        self.assertEqual(stages[2]["parameters"]["files"][0]["path"], "/home/user/eval.sh")
        self.assertEqual(stages[1]["parameters"]["command"], "chmod +x /home/user/setup.sh")
        self.assertEqual(stages[3]["parameters"]["command"], "chmod +x /home/user/eval.sh")
        self.assertEqual(stages[4]["parameters"]["command"], "cd /home/user && ./setup.sh")
        self.assertEqual(stages[5], original["config"][5])
        self.assertEqual(stages[6], original["config"][6])
        self.assertEqual(
            stages[7]["parameters"]["command"][2],
            original["config"][7]["parameters"]["command"][2].replace(
                "'cd test_environment'", "'cd /home/user/test_environment'"
            ),
        )
        for evaluator in (actual["evaluator"], actual["evaluator_list"]["evaluator1"]):
            self.assertEqual(evaluator["result"]["command"], "cd /home/user && bash eval.sh")

    def test_home_task_preserves_terminal_setup_and_normalizes_both_evaluator_scripts(self):
        original = _additional_shape(HOME_TASK_ID)
        actual = normalize_task_setup(original)
        self.assertEqual(actual["config"][3:], original["config"][3:])
        self.assertEqual(actual["config"][2]["parameters"]["command"], "cd /home/user && ./setup.sh")
        for evaluator, script in (
            (actual["evaluator"], "eval.sh"),
            (actual["evaluator_list"]["evaluator1"], "eval.sh"),
            (actual["evaluator_list"]["evaluator2"], "eval_p1.sh"),
        ):
            self.assertEqual(evaluator["postconfig"][0]["parameters"]["files"][0]["path"], f"/home/user/{script}")
            self.assertEqual(evaluator["postconfig"][1]["parameters"]["command"], f"chmod +x /home/user/{script}")
            self.assertEqual(evaluator["result"]["command"], f"cd /home/user && bash {script}")

    def test_all_three_tasks_are_idempotent_deep_copies_preserving_evaluator_contracts(self):
        for original in (_actual_shape(), _additional_shape(ENVIRONMENT_TASK_ID), _additional_shape(HOME_TASK_ID)):
            with self.subTest(task=original["id"]):
                before = copy.deepcopy(original)
                actual = normalize_task_setup(original)
                self.assertEqual(original, before)
                self.assertEqual(normalize_task_setup(actual), actual)
                for key in ("prompt", "task_template", "reward"):
                    self.assertEqual(actual[key], original[key])
                for old, new in (
                    (original["evaluator"], actual["evaluator"]),
                    *[(value, actual["evaluator_list"][key]) for key, value in original["evaluator_list"].items()],
                ):
                    self.assertIsNot(new, old)
                    self.assertEqual(new["func"], old["func"])
                    self.assertEqual(new["expected"], old["expected"])
                    self.assertEqual(new["result"]["shell"], old["result"]["shell"])
                    for old_stage, new_stage in zip(old.get("postconfig", []), new.get("postconfig", [])):
                        if old_stage["type"] == "download":
                            self.assertEqual(
                                new_stage["parameters"]["files"][0]["url"],
                                old_stage["parameters"]["files"][0]["url"],
                            )
                for old_stage, new_stage in zip(original["config"], actual["config"]):
                    if old_stage["type"] == "download":
                        self.assertEqual(
                            new_stage["parameters"]["files"][0]["url"],
                            old_stage["parameters"]["files"][0]["url"],
                        )

    def test_gui_rewrite_is_specific_to_each_allowlisted_task(self):
        for task_id in (TASK_ID, ENVIRONMENT_TASK_ID, HOME_TASK_ID):
            with self.subTest(task=task_id):
                other_call = (
                    "pyautogui.write('cd testDir')" if task_id != TASK_ID else "pyautogui.write('cd test_environment')"
                )
                original = {"id": task_id, "config": [_execute(["python", "-c", other_call], shell=False)]}
                self.assertEqual(normalize_task_setup(original), original)

    def test_unknown_task_preserves_all_new_relative_script_patterns(self):
        original = _additional_shape(HOME_TASK_ID)
        original["id"] = "not-one-of-the-three-allowlisted-tasks"
        self.assertEqual(normalize_task_setup(original), original)

    def test_unknown_script_names_are_not_generalized(self):
        original = {
            "id": HOME_TASK_ID,
            "config": [_download("another.sh"), _execute("chmod +x another.sh"), _execute("bash another.sh")],
        }
        self.assertEqual(normalize_task_setup(original), original)

    def test_urls_prompt_reward_and_evaluator_expectations_are_unchanged(self):
        original = _actual_shape()
        actual = normalize_task_setup(original)
        for key in ("id", "snapshot", "prompt", "task_template", "reward"):
            self.assertEqual(actual[key], original[key])
        for old_stages, new_stages in (
            (original["config"], actual["config"]),
            (original["evaluator"]["postconfig"], actual["evaluator"]["postconfig"]),
        ):
            self.assertEqual(
                new_stages[0]["parameters"]["files"][0]["url"],
                old_stages[0]["parameters"]["files"][0]["url"],
            )
        for key in ("func", "expected"):
            self.assertEqual(actual["evaluator"][key], original["evaluator"][key])

    def test_unrelated_commands_downloads_and_non_shell_strings_are_unchanged(self):
        original = _actual_shape()
        sentinels = [
            {"type": "download", "parameters": {"files": [{"url": "fixture", "path": "elsewhere/setup.sh"}]}},
            _execute("echo ./setup.sh"),
            _execute("chmod +x setup.sh", shell=False),
            _execute(["echo", "-c", ENTER_DIRECTORY], shell=False),
        ]
        original["config"].extend(copy.deepcopy(sentinels))
        actual = normalize_task_setup(original)
        self.assertEqual(actual["config"][-len(sentinels) :], sentinels)

    def test_missing_optional_evaluator_fields_are_accepted(self):
        original = {"id": TASK_ID, "config": [_download("setup.sh")]}
        actual = normalize_task_setup(original)
        self.assertEqual(actual["config"][0]["parameters"]["files"][0]["path"], "/home/user/setup.sh")
        self.assertNotIn("evaluator", actual)

    def test_script_cwd_chain_runs_from_unrelated_server_directory(self):
        # Execute the normalized shell commands against a local VM-directory
        # analogue. Only the fixed /home/user prefix is mapped to the fixture;
        # this avoids touching the real home or requiring an OSWorld sandbox.
        normalized = normalize_task_setup(_actual_shape())
        scripts = {
            "setup.sh": "#!/bin/sh\nmkdir -p testDir\nprintf marker > testDir/probe\n",
            "eval.sh": "#!/bin/sh\ntest -f testDir/probe && cat testDir/probe\n",
        }
        with tempfile.TemporaryDirectory(prefix="osworld-cwd-") as temporary:
            vm_root = Path(temporary)
            desktop_home = vm_root / "desktop-home"
            server_cwd = vm_root / "server-cwd"
            desktop_home.mkdir()
            server_cwd.mkdir()

            def local_path(value):
                return value.replace("/home/user", str(desktop_home))

            for stages in (normalized["config"][:3], normalized["evaluator"]["postconfig"]):
                for stage in stages:
                    parameters = stage["parameters"]
                    if stage["type"] == "download":
                        for item in parameters["files"]:
                            destination = Path(local_path(item["path"]))
                            destination.write_text(scripts[destination.name])
                    else:
                        subprocess.run(local_path(parameters["command"]), shell=True, cwd=server_cwd, check=True)
            result = subprocess.run(
                local_path(normalized["evaluator"]["result"]["command"]),
                shell=True,
                cwd=server_cwd,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout, "marker")
            self.assertTrue((desktop_home / "testDir/probe").is_file())
            self.assertFalse((server_cwd / "testDir").exists())


if __name__ == "__main__":
    unittest.main()
