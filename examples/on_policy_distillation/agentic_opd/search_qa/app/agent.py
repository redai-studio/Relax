# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Search-QA agentic rollout mainline (one process per session).

Reads a session input whose ``metadata`` carries ``question`` and
``ground_truth``, drives a multi-turn ``<search>`` / ``<answer>`` interaction
against the Relax OpenAI-compatible endpoint, and writes back the environment
outcome (binary EM ``score`` / ``won`` + turn stats). The reward is computed
downstream by ``reward_search.reward_func`` from that metadata.

Every step is a fresh one-user-message chat rebuilt from the task and the
bounded search history (matches SDAR's multi-turn rollout). All turns in a
trajectory receive the final episode reward.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.env_search import SearchEnv
from examples.on_policy_distillation.agentic_opd.search_qa.reward_search import compute_score


MAX_TURNS = int(os.environ.get("SEARCH_MAX_TURNS", "4"))
HISTORY_LENGTH = int(os.environ.get("SEARCH_HISTORY_LENGTH", "4"))
RETRIEVAL_URL = os.environ.get("SEARCH_RETRIEVAL_URL", "http://127.0.0.1:8000/retrieve")
RETRIEVAL_TOPK = int(os.environ.get("SEARCH_RETRIEVAL_TOPK", "3"))

# Retry budget for the inference call. Under a large prompt batch the Relax
# Serve proxy can be briefly recycled, producing a burst of connection refusals
# / 5xx; we ride it out instead of crashing the session. LLM_MAX_RETRIES caps
# the SDK-style backoff attempts, LLM_TRANSIENT_RETRY_SECONDS the outer wall-clock
# budget. Neither changes training semantics -- the same turn is just generated later.
LLM_MAX_RETRIES = int(os.environ.get("SEARCH_LLM_MAX_RETRIES", "8"))
LLM_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("SEARCH_LLM_REQUEST_TIMEOUT_SECONDS", "60"))
LLM_TRANSIENT_RETRY_SECONDS = float(os.environ.get("SEARCH_LLM_TRANSIENT_RETRY_SECONDS", "180"))

# Treat a reply cut off at the token cap as a wasted turn instead of ending the
# episode. Defaults off; the mopd_agentic calibration script turns it on.
SOFT_LENGTH_PENALTY = os.environ.get("SEARCH_SOFT_LENGTH_PENALTY", "0") == "1"
EXPORT_ALL_TURNS = os.environ.get("SEARCH_EXPORT_ALL_TURNS", "1") == "1"


def _export_all_turns(metadata: dict[str, Any]) -> bool:
    return EXPORT_ALL_TURNS and metadata.get("split") != "eval"


def read_session_input(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_session_output(path: str | Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    if isinstance(payload, list):
        text = "\n".join(json.dumps(record) for record in payload)
    else:
        text = json.dumps(payload)
    Path(path).write_text(text, encoding="utf-8")


def _is_context_length_error(exc: APIStatusError) -> bool:
    """True if a status error is SGLang's ``context_length_exceeded`` signal.

    Guards the body parse: the proxy can return a non-JSON error body when
    briefly unhealthy, and parsing that would raise instead of classifying.
    """
    try:
        error = exc.response.json().get("error")
    except (ValueError, json.JSONDecodeError):
        return False
    return isinstance(error, dict) and error.get("code") == "context_length_exceeded"


def _is_transient_status(exc: APIStatusError) -> bool:
    """True for HTTP statuses worth retrying, not real client errors.

    Retryable: 429/408/409, any 5xx, and 404 -- a 404 here is a transient Ray
    Serve routing-table miss while the endpoint (re)deploys, not a bad URL, so it
    clears on its own. Genuine client errors (400/401/403/422) are not retried.
    """
    status = exc.status_code
    return status in (404, 408, 409, 429) or status >= 500


class TransientLLMExhaustedError(RuntimeError):
    pass


async def _create_chat_completion(client: AsyncOpenAI, messages: list[dict[str, Any]]) -> Any:
    """Issue one chat turn within a bounded transient retry budget."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + LLM_TRANSIENT_RETRY_SECONDS
    delay = 1.0
    attempts = 0
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TransientLLMExhaustedError("transient LLM retry budget exhausted")
        attempts += 1
        try:
            return await asyncio.wait_for(
                client.chat.completions.create(model="model", messages=messages),
                timeout=min(max(1.0, LLM_REQUEST_TIMEOUT_SECONDS), remaining),
            )
        except APIStatusError as exc:
            if _is_context_length_error(exc):
                return None
            if not _is_transient_status(exc):
                raise
            if loop.time() >= deadline or attempts >= max(1, LLM_MAX_RETRIES):
                raise TransientLLMExhaustedError("transient LLM status retry budget exhausted") from exc
        except (APIConnectionError, APITimeoutError, asyncio.TimeoutError) as exc:
            if loop.time() >= deadline or attempts >= max(1, LLM_MAX_RETRIES):
                raise TransientLLMExhaustedError("transient LLM request retry budget exhausted") from exc
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TransientLLMExhaustedError("transient LLM retry budget exhausted")
        await asyncio.sleep(min(delay, 15.0, remaining))
        delay *= 2


async def run_search(metadata: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    question = metadata.get("question")
    if not question:
        raise ValueError("session metadata is missing required key 'question'")

    env = SearchEnv(
        question=question,
        ground_truth=metadata.get("ground_truth"),
        max_turns=MAX_TURNS,
        history_length=HISTORY_LENGTH,
        retrieval_url=RETRIEVAL_URL,
        topk=RETRIEVAL_TOPK,
    )
    first_prompt = env.reset()
    messages: list[dict[str, Any]] = [{"role": "user", "content": first_prompt}]

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=httpx.Timeout(timeout=max(1.0, LLM_REQUEST_TIMEOUT_SECONDS), connect=30.0),
        max_retries=0,
    )

    won = False
    score = 0.0
    stop_reason = "max_turns"
    num_valid_actions = 0
    num_truncated_turns = 0
    num_searches = 0
    num_retrieval_errors = 0
    num_llm_errors = 0
    turn_records: list[dict[str, Any]] = []

    for _turn in range(MAX_TURNS):
        try:
            response = await _create_chat_completion(client, messages)
        except TransientLLMExhaustedError:
            num_llm_errors += 1
            stop_reason = "llm_unavailable"
            break
        if response is None:
            # context_length_exceeded: the windowed prompt no longer fits, so the
            # episode ends here as a length finish (mirrors finish_reason=="length").
            stop_reason = "finish_length"
            break

        response_text = response.choices[0].message.content or ""
        turn_records.append(
            {
                "name": f"turn_{len(turn_records)}",
                "messages": [*messages, {"role": "assistant", "content": response_text}],
            }
        )
        if response.choices[0].finish_reason == "length":
            num_truncated_turns += 1
            if not SOFT_LENGTH_PENALTY:
                stop_reason = "finish_length"
                break

        next_prompt, done, info = env.step(response_text)
        num_valid_actions += int(bool(info["is_action_valid"]))
        num_searches += int(bool(info["is_search"]))
        num_retrieval_errors += int(bool(info["retrieval_error"]))
        if info["won"]:
            won = True
        if done:
            score = float(info["score"])
            stop_reason = "env_done" if info["env_done"] else "max_turns"
            break
        messages = [{"role": "user", "content": next_prompt}]

    episode_metadata = {
        "won": won,
        "success": 1.0 if won else 0.0,
        "score": score,
        "em": score,
        "num_turns": env.turn,
        "num_searches": num_searches,
        "num_valid_actions": num_valid_actions,
        "num_truncated_turns": num_truncated_turns,
        "num_retrieval_errors": num_retrieval_errors,
        "num_llm_errors": num_llm_errors,
        "stop_reason": stop_reason,
    }
    reward = compute_score(episode_metadata)
    if not turn_records:
        return {"metadata": episode_metadata, "reward": reward}

    exported_records = turn_records if _export_all_turns(metadata) else turn_records[-1:]
    for turn_index, record in enumerate(turn_records):
        if record not in exported_records:
            continue
        record["metadata"] = {
            **episode_metadata,
            "turn_index": turn_index,
            "num_exported_turns": len(exported_records),
        }
        record["reward"] = reward
    return exported_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Search-QA agent session.")
    parser.add_argument("--input-json", required=True, help="Path to the session input JSON.")
    parser.add_argument("--output-json", required=True, help="Path to write the session output JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_input = read_session_input(args.input_json)
    metadata = session_input.get("metadata") or {}
    session_output = asyncio.run(run_search(metadata=metadata))
    write_session_output(args.output_json, session_output)


if __name__ == "__main__":
    main()
