# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""WebShop agentic rollout mainline (one process per session, thin client).

Reads a session input containing ``metadata['goal_idx']`` (a WebShop goal index
produced by ``prepare_data.py``), drives a multi-turn shopping interaction on
that pinned goal, and writes back a session output whose ``metadata`` carries the
environment outcome (``won`` / ``task_score`` / turn stats). Reward is computed
downstream by ``reward_webshop.reward_func`` from this metadata.

Two endpoints:
  * the policy — Relax OpenAI-compatible endpoint at ``$OPENAI_BASE_URL``.
  * the env    — cluster-shared WebShop server (``server.py``) at ``$WEBSHOP_URL``.
The heavy WebShop catalog lives in that server; this process is a thin client.
The session's cart is keyed by ``$RELAX_SESSION_ID`` (globally unique), so the
same ``goal_idx`` rolled out ``group_size`` times never cross-contaminates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from app import prompt
from app.env_client import WebshopClient


# Interaction budget (turns) and how many past (obs, action) pairs to show.
# Default 15 aligns with SDAR env.max_steps=15 for WebShop.
MAX_TURNS = int(os.environ.get("WEBSHOP_MAX_TURNS", "15"))
HISTORY_LENGTH = int(os.environ.get("WEBSHOP_HISTORY_LENGTH", "2"))


def read_session_input(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_session_output(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


async def run_webshop(metadata: dict[str, Any]) -> dict[str, Any]:
    goal_idx = metadata.get("goal_idx")
    if goal_idx is None:
        raise ValueError("session metadata is missing required key 'goal_idx'")

    env = WebshopClient(
        base_url=os.environ["WEBSHOP_URL"],
        instance_id=os.environ["RELAX_SESSION_ID"],
    )

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=9999,
    )

    won = False
    task_score = 0.0
    stop_reason = "max_turns"
    num_turns = 0
    num_valid_actions = 0

    try:
        raw_obs, available = env.reset(int(goal_idx))
        task = prompt.extract_task(raw_obs)
        obs = prompt.format_obs(raw_obs, task)
        history: list[tuple[str, str]] = []
        first_prompt = prompt.build_prompt(
            task=task,
            obs=obs,
            available_actions=available,
            history=history,
            history_length=HISTORY_LENGTH,
            init=True,
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": first_prompt}]

        for _turn in range(MAX_TURNS):
            try:
                # enable_thinking is controlled by the training script's
                # --apply-chat-template-kwargs (serving applies it as the default
                # chat_template_kwargs); no per-request override needed here.
                response = await client.chat.completions.create(model="model", messages=messages)
            except APIStatusError as exc:
                error = exc.response.json().get("error")
                if isinstance(error, dict) and error.get("code") == "context_length_exceeded":
                    stop_reason = "finish_length"
                    break
                raise

            response_text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": response_text})
            if response.choices[0].finish_reason == "length":
                stop_reason = "finish_length"
                break

            action, is_valid = prompt.extract_action(response_text)
            raw_obs, reward, done, info = env.step(action)
            num_turns += 1
            num_valid_actions += int(bool(is_valid))
            task_score = float(info["task_score"])
            if info["won"]:
                won = True

            history.append((obs, action))
            obs = prompt.format_obs(raw_obs, task)
            available = info["available_actions"]

            if done:
                stop_reason = "env_done"
                break
            next_prompt = prompt.build_prompt(
                task=task,
                obs=obs,
                available_actions=available,
                history=history,
                history_length=HISTORY_LENGTH,
            )
            messages.append({"role": "user", "content": next_prompt})
    finally:
        env.close()

    return {
        "metadata": {
            "won": won,
            "success": 1.0 if won else 0.0,
            "task_score": task_score,
            "goal_idx": int(goal_idx),
            "split": metadata.get("split"),
            "num_turns": num_turns,
            "num_valid_actions": num_valid_actions,
            "stop_reason": stop_reason,
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one WebShop agent session.")
    parser.add_argument("--input-json", required=True, help="Path to the session input JSON.")
    parser.add_argument("--output-json", required=True, help="Path to write the session output JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_input = read_session_input(args.input_json)
    metadata = session_input.get("metadata") or {}
    session_output = asyncio.run(run_webshop(metadata=metadata))
    write_session_output(args.output_json, session_output)


if __name__ == "__main__":
    main()
