# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Group RM backed by retained mini-swe-agent sandboxes and Relax GenRM."""

import asyncio
import time
from typing import Any
from urllib.parse import quote

import httpx

from examples.gagar.core.gar import reward_records
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def outcome(sample: Any) -> float:
    return sample.reward["test_reward"] if isinstance(sample.reward, dict) else sample.reward


async def reward_func(args: Any, samples: list[Any], **kwargs: Any) -> list[dict]:
    original = [outcome(sample) for sample in samples]
    baseline = reward_records(original, original, [1.0] * len(samples))
    # Eval samples have no retained-sandbox metadata and keep binary rewards.
    if not all("gagar" in (sample.metadata or {}) for sample in samples):
        logger.info("GAGAR skipped_missing_metadata: outcomes=%s", original)
        return baseline
    metadata = [sample.metadata["gagar"] for sample in samples]
    group_id = metadata[0]["group_id"]
    server_url = metadata[0]["server_url"].rstrip("/")
    if any(m["group_id"] != group_id or m["server_url"].rstrip("/") != server_url for m in metadata):
        raise ValueError("one grading group must belong to one mini-swe-agent server/task")

    from relax.utils.utils import get_serve_url

    group_url = f"{server_url}/gagar/groups/{quote(group_id, safe='')}"
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        try:
            if not any(original):
                logger.info("GAGAR group %s skipped_all_zero", group_id)
                return baseline
            logger.info("GAGAR group %s started: outcomes=%s", group_id, original)
            response = await client.post(
                f"{server_url}/gagar/groups",
                json={
                    "group_id": group_id,
                    "session_ids": [m["session_id"] for m in metadata],
                    "bridge_url": server_url,
                    "genrm_url": get_serve_url("genrm"),
                },
            )
            response.raise_for_status()
            state = response.json()
            wait_budget = state.get("timeout_seconds", 1800) + 30
            while state["status"] == "running":
                # Server owns the hard deadline and process cancellation.
                if time.monotonic() - started > wait_budget:
                    raise TimeoutError("grading group exceeded its wait budget")
                await asyncio.sleep(1)
                response = await client.get(group_url)
                response.raise_for_status()
                state = response.json()
            if state["status"] != "completed":
                raise ValueError(state.get("error") or f"grader {state['status']}")
            result = state["result"]
            records = reward_records(original, result["rewards"], result["weights"])
            logger.info(
                "GAGAR group %s completed: outcomes=%s corrected=%s weights=%s scores=%s seconds=%.2f",
                group_id,
                original,
                result["rewards"],
                result["weights"],
                [record["score"] for record in records],
                time.monotonic() - started,
            )
            return records
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError) as exc:
            logger.warning("GAGAR group %s uses original outcomes: %s", group_id, exc)
            return reward_records(original, original, [1.0] * len(samples), fallback=True)
        finally:
            for sample in samples:
                sample.metadata["grader_seconds"] = time.monotonic() - started
            try:
                response = await client.post(f"{group_url}/release")
                response.raise_for_status()
            except httpx.HTTPError as exc:
                # The server's independent lease sweeper also covers groups
                # dropped by export/filtering before this function is called.
                logger.warning("GAGAR group %s release deferred to lease expiry: %s", group_id, exc)
