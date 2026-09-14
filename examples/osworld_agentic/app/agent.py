# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""OSWorld agentic-stack session driver.

One process per Relax-managed session. Drives the vendored HybridAgentLocal
(``RelaxHybridAgent``) against a session-owned OSWorld VM (``OSWorldEnv``).

Input JSON is written by Relax's ``AgenticPipeline._write_agent_input`` — it
carries the chat prompt (unused by us, kept for schema compat), plus the
task_config in ``metadata`` so we can hand it to ``DesktopEnv.reset``.

Output JSON follows ``SessionOutput`` — only ``metadata`` and ``reward`` are
read back by Relax; everything else is ours to shape.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml
from app import shotdump
from app.env_osworld import OSWorldEnv, StepObs, parse_mcp_action
from app.hybrid_agent import (
    ContextLengthExceededError,
    RelaxHybridAgent,
    SessionDiscardedError,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("osworld_config.yaml")

logger = logging.getLogger("osworld_agentic.agent")


def _load_config() -> dict[str, Any]:
    path = Path(os.environ.get("OSWORLD_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_session_input(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_session_output(path: str | Path, payload: dict[str, Any]) -> None:
    """Emit the session result Relax reads back.

    The hybrid agent rebuilds its prompt every step, so the session forest ends
    up with one leaf per turn and Relax's implicit export (which insists on a
    single committed leaf) drops the whole session. We therefore use the
    explicit path: one JSONL record per turn, each carrying the exact
    conversation state that turn committed. See PITFALLS.md section 11.
    """
    records = payload.pop("_turn_records", []) or []
    metadata = payload.get("metadata", {})
    reward = payload.get("reward")
    # The launcher passes --reward-key score, and Relax then does
    # sample.reward[reward_key] (relax/agentic/rollout.py:927,
    # relax/engine/rewards/__init__.py:414). Export a dict, not a float.
    _reward_key = os.environ.get("OSWORLD_REWARD_KEY", "score")
    if _reward_key and not isinstance(reward, dict):
        reward = {_reward_key: float(reward or 0.0)}
        payload["reward"] = reward
    try:
        if records:
            lines = [
                json.dumps(
                    {
                        "name": f"turn_{index:03d}",
                        "messages": record["messages"],
                        "tools": record.get("tools") or [],
                        "chat_template_kwargs": record.get("chat_template_kwargs") or {},
                        "metadata": {**metadata, "turn_index": index},
                        "reward": reward,
                    },
                    ensure_ascii=False,
                )
                for index, record in enumerate(records)
            ]
            Path(path).write_text("\n".join(lines), encoding="utf-8")
        else:
            Path(path).write_text(json.dumps(payload), encoding="utf-8")
    except FileNotFoundError:
        # Relax already discarded the session tmpdir; nothing to do.
        pass


def _extract_task_config(session_input: dict[str, Any]) -> dict[str, Any]:
    """Pull the OSWorld task config out of the Relax session-input envelope.

    Two accepted layouts, in priority order:

    1. ``session_input["task_config"]`` — full OSWorld task JSON dumped inline
       by our data-prep pipeline (``scripts/prepare_data.py``).
    2. ``session_input["metadata"]["task_config"]`` — same but nested (Relax
       preserves arbitrary metadata fields).
    """

    def _coerce(value: Any) -> dict[str, Any] | None:
        # prepare_data.py stores the config as a JSON string (Arrow cannot type
        # the heterogeneous struct); smoke_test.sh passes a plain dict.
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    direct = _coerce(session_input.get("task_config"))
    if direct is not None:
        return direct
    meta = session_input.get("metadata") or {}
    nested = _coerce(meta.get("task_config"))
    if nested is not None:
        return nested
    raise ValueError("session input contains no `task_config` — did prepare_data.py run?")


async def _predict_one_step(
    agent: RelaxHybridAgent,
    instruction: str,
    obs: StepObs,
    step_idx: int,
) -> tuple[str, list[str], dict[str, Any]]:
    """Run agent.predict in a worker thread (sync httpx + PIL inside)."""
    obs_dict = {
        "screenshot": obs.screenshot_bytes,
        "cur_app": obs.cur_app,
        "app_info": obs.app_info,
        "tool_list": obs.tool_list,
    }
    return await asyncio.to_thread(
        agent.predict,
        instruction,
        obs_dict,
        step_idx=step_idx,
    )


async def run_session(
    session_input: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    config = _load_config()
    max_turns = int(os.environ.get("OSWORLD_MAX_TURNS") or config["max_turns"])  # env override (2026-09-05)
    task_config = _extract_task_config(session_input)
    instruction = task_config.get("instruction") or metadata.get("instruction") or ""
    shotdump.init((task_config or {}).get("id", ""), instruction)

    env = OSWorldEnv(
        task_config=task_config,
        provider_name=config.get("provider_name", "nex"),
        snapshot_name=config.get("snapshot_name", "init_state"),
        screen_width=int(config.get("screen_width", 1920)),
        screen_height=int(config.get("screen_height", 1080)),
        headless=bool(config.get("headless", True)),
        require_a11y_tree=bool(config.get("require_a11y_tree", False)),
    )

    agent = RelaxHybridAgent(
        # Vendor-side knobs
        platform="ubuntu",
        model=os.environ.get("OPENAI_MODEL", "model"),
        max_steps=max_turns,
        max_image_history_length=int(config.get("max_image_history_length", 2)),
        max_reward_image_history_length=int(config.get("max_reward_image_history_length", 1)),
        max_tokens=int(config.get("max_tokens", 4096)),
        top_p=float(config.get("top_p", 0.9)),
        temperature=float(config.get("temperature", 0.8)),
        action_space="hybrid",
        context_policy=str(config.get("context_policy", "baseline")),
        repetition_penalty=float(config.get("repetition_penalty", 1.0)),
        tool_retriever_backend=str(config.get("tool_retriever_backend", "bm25")),
        tool_retriever_top_k=int(config.get("tool_retriever_top_k", 18)),
        # Relax-side transport knobs (RelaxHybridAgent-only)
        llm_call_timeout_s=float(config.get("llm_call_timeout_s", 600.0)),
        max_llm_call_retries=int(config.get("max_llm_call_retries", 3)),
        backoff_seconds=float(config.get("llm_call_backoff_s", 5.0)),
        stop_token_ids=list(config.get("stop_token_ids") or []),
    )
    agent.reset(logging.getLogger("osworld_agentic.hybrid_agent"))

    stop_reason = "max_turns"
    action_types: list[str] = []
    parse_failed_steps = 0
    reward = 0.0
    last_error: str | None = None

    session_start = time.monotonic()
    try:
        obs = await env.reset()

        for step_idx in range(max_turns):
            shotdump.shot(step_idx, getattr(obs, "screenshot_bytes", None))
            try:
                response, pyautogui_actions, info = await _predict_one_step(
                    agent,
                    instruction,
                    obs,
                    step_idx=step_idx,
                )
            except SessionDiscardedError:
                stop_reason = "discarded_by_pipeline"
                break
            except ContextLengthExceededError as exc:
                stop_reason = "context_length_exceeded"
                last_error = str(exc)
                break
            except Exception as exc:  # noqa: BLE001
                stop_reason = "predict_error"
                last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("predict raised at step %d", step_idx)
                break

            action_type = info.get("action_type", "gui")
            action_types.append(action_type)
            if info.get("parse_failed") or action_type == "parse_failed":
                parse_failed_steps += 1

            # Terminate signals from the vendor agent
            if any(a in ("DONE", "FAIL") for a in pyautogui_actions):
                stop_reason = "agent_done" if "DONE" in pyautogui_actions else "agent_fail"
                # Still call env.step so DesktopEnv marks the final action
                # in `action_history` (evaluate() checks the last entry).
                terminal = "DONE" if "DONE" in pyautogui_actions else "FAIL"
                await env.exec_pyautogui(terminal)
                agent.record_exec_result(action_type, True, {"ok": True, "terminal": terminal})
                break

            # Execute all sub-actions in sequence. Upstream returns a single
            # element list for the hybrid schema, but the API is a list — respect it.
            step_ok = True
            step_result: dict[str, Any] = {}
            step_done = False
            for action in pyautogui_actions:
                mcp = parse_mcp_action(action)
                if mcp is not None:
                    tool_name, params = mcp
                    step_obs = await env.exec_mcp(tool_name, params)
                    step_result = step_obs.exec_result
                    step_ok = step_ok and step_obs.exec_ok
                else:
                    step_obs = await env.exec_pyautogui(action)
                    step_result = step_obs.exec_result
                obs = step_obs
                step_done = step_done or step_obs.done

            agent.record_exec_result(action_type, step_ok, step_result)
            shotdump.turn(
                {
                    "step": step_idx,
                    "action_type": action_type,
                    "actions": pyautogui_actions,
                    "response": response,
                    "ok": step_ok,
                }
            )

            if step_done:
                stop_reason = "env_done"
                break

        # Final scoring lives with the env (VM-side postconfig + metrics).
        reward = await env.evaluate()
        shotdump.finish(
            {
                "reward": reward,
                "stop_reason": stop_reason,
                "turns": len(action_types),
                "parse_failed_steps": parse_failed_steps,
            }
        )
    finally:
        await env.close()

    elapsed = time.monotonic() - session_start
    return {
        "reward": float(reward),
        "_turn_records": list(getattr(agent, "turn_records", [])),
        "metadata": {
            "stop_reason": stop_reason,
            "agent_turns": len(action_types),
            "action_types": action_types,
            "parse_failed_steps": parse_failed_steps,
            "task_id": task_config.get("id"),
            "data_source": metadata.get("data_source"),
            "session_wallclock_s": elapsed,
            "last_error": last_error,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OSWorld agentic session driver.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("AGENT_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = parse_args()
    session_input = read_session_input(args.input_json)
    metadata = session_input.get("metadata") or {}
    try:
        output = asyncio.run(run_session(session_input, metadata))
    except Exception as exc:  # noqa: BLE001
        # Never die silently: Relax waits for this file, and a session that
        # never reports back blocks its whole rollout group forever.
        logger.exception("run_session failed; reporting a zero-reward session")
        task_config = {}
        try:
            task_config = _extract_task_config(session_input)
        except Exception:  # noqa: BLE001
            pass
        output = {
            "reward": 0.0,
            "metadata": {
                "stop_reason": "session_error",
                "agent_turns": 0,
                "action_types": [],
                "parse_failed_steps": 0,
                "task_id": task_config.get("id"),
                "data_source": metadata.get("data_source"),
                "session_wallclock_s": 0.0,
                "last_error": f"{type(exc).__name__}: {exc}",
            },
        }
    write_session_output(args.output_json, output)


if __name__ == "__main__":
    main()
