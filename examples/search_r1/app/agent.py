# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Search-R1 managed-agent entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .multiagent import run_multiagent_session
from .retriever import RetrieverClient
from .vanilla import run_vanilla_session


def read_session_input(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_session_output(path: str | Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    if isinstance(payload, list):
        text = "\n".join(json.dumps(record, ensure_ascii=False) for record in payload) + "\n"
    else:
        text = json.dumps(payload, ensure_ascii=False)
    Path(path).write_text(text, encoding="utf-8")


async def _run_from_environment(session_input: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    from openai import AsyncOpenAI  # type: ignore[import-not-found]

    config = yaml.safe_load(Path(__file__).with_name("search_r1_config.yaml").read_text(encoding="utf-8"))
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"].rstrip("/"),
        timeout=9999,
    )
    async with RetrieverClient(
        endpoint=os.environ["SEARCH_R1_RETRIEVER_URL"],
        topk=int(config["retriever_topk"]),
        timeout_s=float(config["retriever_timeout_s"]),
    ) as retriever:
        common = {
            "messages": session_input["messages"],
            "aliases": session_input["metadata"]["answers"],
            "client": client,
            "retriever": retriever,
            "max_observation_chars": int(config["max_observation_chars"]),
            "max_search_turns": int(config["max_search_turns"]),
            "structure_weight": float(config["structure_weight"]),
            "final_weight": float(config["final_weight"]),
            "rollout_mode": os.environ["RELAX_ROLLOUT_MODE"],
        }
        mode = os.environ["SEARCH_R1_MODE"]
        if mode == "vanilla":
            return await run_vanilla_session(**common)
        if mode == "multiagent":
            return await run_multiagent_session(
                **common,
                searcher_max_search_turns=int(config["searcher_max_search_turns"]),
            )
        raise ValueError(f"Unsupported SEARCH_R1_MODE: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one vanilla or multi-agent Search-R1 session.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = asyncio.run(_run_from_environment(read_session_input(args.input_json)))
    write_session_output(args.output_json, output)


if __name__ == "__main__":
    main()
