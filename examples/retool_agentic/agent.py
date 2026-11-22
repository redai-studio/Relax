# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""One stateless-tool ReTool episode per managed Agentic process."""

import asyncio
import copy
import json
import math
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from examples.retool_agentic.protocol import (
    CHAT_TEMPLATE,
    DIRECT_PREFIX,
    feedback_for_turn,
    observation_text,
    render_prompt,
)
from examples.retool_agentic.sandbox import SandboxClient


class ModelClient:
    def __init__(self, base_url: str, session_id: str) -> None:
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.session_id = session_id

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.session_id}"},
            method="POST",
        )
        # No client retry: retrying a generated turn can fork the Session.
        # The managed process's active-time budget bounds stuck requests;
        # an HTTP read must survive a Runtime gate without a wall timeout.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=None) as response:
            return json.load(response)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._request, payload)


async def run_episode(
    payload: dict[str, Any],
    *,
    count_tokens: Callable[[str], int],
    complete: Any,
    client: Any,
) -> dict[str, Any]:
    messages = copy.deepcopy(payload["messages"])
    metadata = payload.get("metadata", {})
    config = metadata["retool_agentic"]
    budget, context, max_turns = config["response_budget"], config["context_budget"], config["max_turns"]
    if any(type(value) is not int or value < 1 for value in (budget, context, max_turns)):
        raise ValueError("Positive ReTool response, context and turn budgets required")
    timeout_limit = config.get("max_consecutive_timeouts", 0)
    tool_budget = config.get("tool_wall_time_budget_seconds", 0)
    if type(timeout_limit) is not int or timeout_limit < 0:
        raise ValueError("max_consecutive_timeouts must be a nonnegative integer")
    if type(tool_budget) not in (int, float) or not math.isfinite(tool_budget) or tool_budget < 0:
        raise ValueError("tool_wall_time_budget_seconds must be finite and nonnegative")
    tool_enabled = metadata.get("tool_enabled", True)
    if not tool_enabled:
        messages = [{"role": "user", "content": DIRECT_PREFIX + metadata["question"]}]
    initial_tokens = count_tokens(render_prompt(messages))
    used = 0
    timeout_streak = 0
    result: dict[str, Any] = dict(
        retool_final_answer="",
        tool_feedback_counts={},
        author_tool_events=[],
        tool_calls=0,
        retool_assistant_turns=0,
        retool_tool_wall_seconds=0.0,
        retool_max_timeout_streak=0,
    )
    try:
        for turn in range(max_turns):
            room = min(budget - used, context - initial_tokens - used)
            if room <= 0:
                result["retool_stop_reason"] = "token_budget"
                break
            request: dict[str, Any] = dict(
                model="policy",
                messages=copy.deepcopy(messages),
                stream=False,
                max_completion_tokens=room,
                chat_template_kwargs={"chat_template": CHAT_TEMPLATE},
            )
            if tool_enabled:
                request["stop"] = ["</code>"]
            if "seed" in config:
                request["seed"] = config["seed"] + turn * 1000003
            response = await complete(request)
            choice = response["choices"][0]
            message, finish = choice["message"], choice["finish_reason"]
            text = message["content"]
            actual = response["usage"]["completion_tokens"]
            if type(actual) is not int or not 0 < actual <= room:
                raise RuntimeError("Need actual bounded engine completion token usage")
            used += actual
            result.update(retool_assistant_turns=turn + 1, num_turns=turn + 1 + result["tool_calls"])
            # Reuse the exact canonical assistant message returned by Relax.
            messages.append(message)
            if finish == "length":
                if not tool_enabled or "<code>" not in text:
                    result["retool_final_answer"] = text
                result["retool_stop_reason"] = "engine_length"
                break
            if finish != "stop":
                raise RuntimeError(f"Unexpected ReTool model finish reason: {finish}")
            if not tool_enabled or ("<code>" not in text and "</code>" not in text):
                result.update(retool_final_answer=text, retool_stop_reason="final_answer")
                break
            if turn + 1 == max_turns or used >= budget or initial_tokens + used >= context:
                result["retool_stop_reason"] = "max_turns_or_budget"
                break
            started = time.perf_counter()
            feedback, code = await feedback_for_turn(text, client)
            elapsed = time.perf_counter() - started
            status = feedback["status"]
            counts = result["tool_feedback_counts"]
            counts[status] = counts.get(status, 0) + 1
            result["tool_calls"] += 1
            result["num_turns"] = turn + 1 + result["tool_calls"]
            if code is not None:
                result["retool_tool_wall_seconds"] += elapsed
            timeout_streak = timeout_streak + 1 if status == "timeout" else 0
            result["retool_max_timeout_streak"] = max(result["retool_max_timeout_streak"], timeout_streak)
            result["author_tool_events"].append(
                dict(
                    submission=result["tool_calls"],
                    exact_executed_code=code,
                    feedback=feedback,
                    new_online_execution=code is not None,
                    wall_seconds=elapsed,
                )
            )
            if timeout_limit and timeout_streak >= timeout_limit:
                result["retool_stop_reason"] = "consecutive_tool_timeouts"
                break
            # Includes the service queue and HTTP wait, excludes model calls.
            # The current call finishes under its own timeout before checking.
            if tool_budget and result["retool_tool_wall_seconds"] >= tool_budget:
                result["retool_stop_reason"] = "tool_wall_time_budget"
                break
            observation = observation_text(feedback)
            size = count_tokens(observation)
            if size >= min(budget - used, context - initial_tokens - used):
                result["retool_stop_reason"] = "observation_budget"
                break
            messages.append({"role": "tool", "content": observation})
            used += size
        return {"metadata": result}
    finally:
        if client is not None:
            await client.close()


def main() -> None:
    from tokenizers import Tokenizer

    payload = json.loads(Path(os.environ["RELAX_INPUT_JSON"]).read_text())
    tokenizer = Tokenizer.from_file(os.environ["RETOOL_TOKENIZER_PATH"])
    model = ModelClient(os.environ["RELAX_BASE_URL"], os.environ["RELAX_SESSION_ID"])
    sandbox = (
        SandboxClient(os.environ["SANDBOX_FUSION_URL"])
        if payload.get("metadata", {}).get("tool_enabled", True)
        else None
    )
    output = asyncio.run(
        run_episode(
            payload,
            count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
            complete=model.complete,
            client=sandbox,
        )
    )
    path = Path(os.environ["RELAX_OUTPUT_JSON"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False))
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
