# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""DeepEyes command agent mainline."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import yaml
from app.env_deepeyes import DeepeyesToolEnv, load_initial_image
from openai import APIStatusError, AsyncOpenAI


def read_session_input(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_session_output(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


async def run_deepeyes(messages: list[dict[str, Any]]) -> dict[str, Any]:
    config = yaml.safe_load(Path(__file__).with_name("deepeyes_config.yaml").read_text(encoding="utf-8"))
    max_turns = int(config["max_turns"])
    normalize_bbox = bool(config.get("normalize_bbox", True))
    env = DeepeyesToolEnv(
        max_turns=max_turns,
        current_image=load_initial_image(messages),
        normalize_bbox=normalize_bbox,
    )

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=9999,
    )
    stop_reason = "max_turns"
    env_infos: list[dict[str, Any]] = []

    for _turn in range(max_turns):
        try:
            response = await client.chat.completions.create(model="model", messages=messages)
        except APIStatusError as exc:
            error = exc.response.json().get("error")
            if isinstance(error, dict) and error.get("code") == "context_length_exceeded":
                stop_reason = "finish_length"
                break
            raise
        response_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": response_text})
        if response.choices[0].finish_reason == "length":
            stop_reason = "finish_length"
            break
        observation_message, done, info = env.step(response_text)
        env_infos.append(info)
        if done:
            stop_reason = "env_done"
            break
        messages.append(observation_message)

    return {
        "metadata": {
            "num_turn": env.turn,
            "stop_reason": stop_reason,
            "env_infos": env_infos,
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DeepEyes agent session.")
    parser.add_argument("--input-json", required=True, help="Path to a JSON file containing a messages field.")
    parser.add_argument("--output-json", required=True, help="Path to write the session output JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_input = read_session_input(args.input_json)
    session_output = asyncio.run(
        run_deepeyes(
            messages=session_input["messages"],
        )
    )
    write_session_output(args.output_json, session_output)


if __name__ == "__main__":
    main()
