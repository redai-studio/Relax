# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import os
import re
from typing import Any

import yaml
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv


ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


def extract_action(response_text: str) -> tuple[str, bool]:
    """Extract the admissible action from a model response.

    Returns ``(action, is_valid)``. The action is the lower-cased content
    inside the *first* ``<action>`` block; a response is "valid" when an
    ``<action>`` block is present and it contains no Chinese characters.
    """
    match = _ACTION_RE.search(response_text)
    if match is None:
        return response_text.strip().lower()[-30:], False
    action = match.group(1).strip().lower()
    is_valid = _CHINESE_RE.search(response_text) is None
    return action, is_valid


def _extract_task(text_obs: str) -> str:
    marker = "Your task is to: "
    idx = text_obs.find(marker)
    return text_obs[idx + len(marker) :].strip() if idx != -1 else ""


def _format_admissible(admissible: list[str]) -> str:
    return "\n ".join(f"'{a}'" for a in admissible if a != "help")


class _PinnedTWEnv(AlfredTWEnv):
    """AlfredTWEnv that registers exactly one game file (no corpus walk)."""

    def __init__(self, config: dict[str, Any], game_file: str, train_eval: str) -> None:
        self._pinned_game_file = game_file
        super().__init__(config, train_eval=train_eval)

    def collect_game_files(self, verbose: bool = False) -> None:  # noqa: D401 - override
        self.game_files = [self._pinned_game_file]
        self.num_games = 1


class AlfworldEnv:
    """Thin, single-episode driver around a pinned AlfredTWEnv instance."""

    def __init__(
        self,
        *,
        game_file: str,
        config_path: str,
        max_turns: int,
        history_length: int,
        alfworld_data: str | None = None,
    ) -> None:
        alfworld_data = alfworld_data or os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
        # game_file from prepare_data.py is relative to $ALFWORLD_DATA.
        if not os.path.isabs(game_file):
            game_file = os.path.join(alfworld_data, game_file)

        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        base_env = _PinnedTWEnv(config, game_file, train_eval="eval_out_of_distribution")
        self.env = base_env.init_env(batch_size=1)

        self.max_turns = max_turns
        self.history_length = history_length
        self.turn = 0
        self.task = ""
        self.admissible: list[str] = []
        self.prev_obs = ""
        self.history: list[tuple[str, str]] = []  # (observation, action) pairs

    # ------------------------------------------------------------------ #
    def reset(self) -> str:
        obs, infos = self.env.reset()
        text = obs[0]
        self.task = _extract_task(text)
        self.admissible = list(infos["admissible_commands"][0])
        self.prev_obs = text
        self.turn = 0
        self.history = []
        return self._build_prompt(text, init=True)

    def step(self, response_text: str) -> tuple[str | None, bool, dict[str, Any]]:
        action, is_valid = extract_action(response_text)
        obs, _scores, dones, infos = self.env.step([action])
        text = obs[0]
        won = bool(infos["won"][0])
        env_done = bool(dones[0])

        self.history.append((self.prev_obs, action))
        self.prev_obs = text
        self.admissible = list(infos["admissible_commands"][0])
        self.turn += 1

        reached_max = self.turn >= self.max_turns
        done = env_done or reached_max
        info = {
            "won": won,
            "action": action,
            "is_action_valid": is_valid,
            "env_done": env_done,
        }
        next_prompt = None if done else self._build_prompt(text)
        return next_prompt, done, info

    # ------------------------------------------------------------------ #
    def _build_prompt(self, text_obs: str, *, init: bool = False) -> str:
        admissible = _format_admissible(self.admissible)
        if init or self.history_length <= 0 or not self.history:
            return ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=text_obs,
                admissible_actions=admissible,
            )
        recent = self.history[-self.history_length :]
        start = len(self.history) - len(recent)
        action_history = "".join(
            f"\n[Observation {start + j + 1}: '{obs}', Action {start + j + 1}: '{act}']"
            for j, (obs, act) in enumerate(recent)
        )
        return ALFWORLD_TEMPLATE.format(
            task_description=self.task,
            step_count=len(self.history),
            history_length=len(recent),
            action_history=action_history,
            current_step=len(self.history) + 1,
            current_observation=text_obs,
            admissible_actions=admissible,
        )
