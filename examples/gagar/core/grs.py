# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared mini-swe-agent loop, GenRM client and frozen-evidence tools."""

import argparse
import importlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import httpx
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, Submitted

from examples.gagar.app.config import load_task_config


def parse_action(response: str, *, thinking: bool = False) -> dict:
    # GenRM /generate returns text, not native OpenAI tool_calls. Keep the raw
    # response in history and parse only the final answer, never a thinking block.
    answer = response.strip()
    # Qwen prefills <think> in the prompt; /generate can omit that opening tag.
    # Require the closing tag in thinking mode, even if reasoning looks like JSON.
    if thinking or answer.startswith("<think>"):
        _, separator, answer = answer.partition("</think>")
        if not separator:
            raise ValueError("unfinished thinking block")
        answer = answer.strip()
    if answer.startswith("```") and answer.endswith("```"):
        answer = answer.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # Qwen can prefix a complete JSON action with this marker, with or without
    # its closing marker. Do not repair JSON or extract it from surrounding text.
    if answer.startswith("<tool_call>"):
        answer = answer.removeprefix("<tool_call>").strip().removesuffix("</tool_call>").strip()
    action = json.loads(answer)
    if (
        not isinstance(action, dict)
        or not isinstance(action.get("name"), str)
        or action["name"] not in {"bash", "read_evidence", "submit_grades"}
    ):
        raise ValueError("return one JSON action named bash, read_evidence or submit_grades")
    return action


class GenRMModel:
    """mini-swe-agent Model protocol over the existing GenRM HTTP API.

    The grader venv intentionally needs only the base recipe dependencies;
    importing GenRMClient there would also import Ray, Torch and TensorDict. No
    separate model server or new Relax model API is introduced.
    """

    def __init__(
        self, service_url: str, *, timeout: float = 120, unix_socket: str | None = None, thinking: bool = False
    ) -> None:
        self.url = service_url.rstrip("/") + "/generate"
        transport = httpx.HTTPTransport(uds=unix_socket) if unix_socket else None
        self.client = httpx.Client(timeout=timeout, trust_env=False, transport=transport)
        self.audit_path: Path | None = None
        self.group_id = ""
        self.thinking = thinking
        self.n_queries = 0
        self.http_attempts = 0

    def record(self, event: str, **fields: Any) -> None:
        if self.audit_path is not None:
            try:
                with self.audit_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "time": time.time(),
                                "group_id": self.group_id,
                                "step": self.n_queries,
                                "event": event,
                                **fields,
                            }
                        )
                        + "\n"
                    )
            except OSError:
                # Diagnostics must not change grading when the audit disk is full.
                return

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        self.n_queries += 1
        self.record("query", message_count=len(messages), history_chars=sum(len(m["content"]) for m in messages))
        history = []
        for message in messages:
            item = {"role": message["role"], "content": message["content"]}
            if self.thinking and item["role"] == "assistant":
                # /generate prefills <think>, so restore its wrapper in history.
                reasoning, closed, answer = item["content"].partition("</think>")
                reasoning = reasoning.strip().removeprefix("<think>").strip()
                item["content"] = f"<think>\n{reasoning}\n</think>\n\n{answer.strip() if closed else ''}"
            history.append(item)
        payload = {"messages": history}
        for attempt in range(3):
            started = time.monotonic()
            self.http_attempts += 1
            try:
                response = self.client.post(self.url, json=payload)
                response.raise_for_status()
                content = response.json()["response"]
                self.record(
                    "model_response",
                    attempt=attempt + 1,
                    http_status=response.status_code,
                    seconds=time.monotonic() - started,
                    response_chars=len(content),
                    thinking_chars=len(content.partition("</think>")[0]) if self.thinking else 0,
                )
                break
            except httpx.HTTPError as exc:
                self.record(
                    "model_error",
                    attempt=attempt + 1,
                    error_type=type(exc).__name__,
                    http_status=exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None,
                    seconds=time.monotonic() - started,
                )
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500 or attempt == 2:
                    raise
                time.sleep(0.5 * 2**attempt)
        message = {"role": "assistant", "content": content, "extra": {"actions": [], "cost": 0.0}}
        try:
            message["extra"]["actions"] = [parse_action(content, thinking=self.thinking)]
            action = message["extra"]["actions"][0]
            fields = {key: action[key] for key in ("rollout_n", "offset") if type(action.get(key)) is int}
            if action.get("section") in ("patch", "trajectory", "test_output", "submission", "code"):
                fields["section"] = action["section"]
            ranking = action.get("ranking")
            if isinstance(ranking, list) and all(
                isinstance(tie, list) and all(type(i) is int for i in tie) for tie in ranking
            ):
                fields["ranking"] = ranking
            self.record("tool_call", name=action["name"], **fields)
            if action["name"] == "submit_grades":
                evaluations = action.get("evaluations")
                self.record(
                    "submission",
                    evaluated_ids=[
                        item["rollout_n"]
                        for item in evaluations
                        if isinstance(item, dict) and type(item.get("rollout_n")) is int
                    ]
                    if isinstance(evaluations, list)
                    else [],
                )
        except (ValueError, IndexError) as exc:
            self.record("format_error", error=str(exc))
            feedback = str(exc)
            if isinstance(exc, json.JSONDecodeError):
                if not exc.doc.strip():
                    feedback = "Your final answer is empty. Return one complete JSON action."
                elif not exc.doc.lstrip().startswith("{"):
                    feedback = (
                        "Your final answer starts with text or XML instead of the JSON object. "
                        "Remove the introductory text and any XML tags. "
                        "Do not rewrite the command merely to remove surrounding prose."
                    )
                elif exc.msg == "Extra data":
                    feedback = "Extra data after the JSON action. Return only one object with no trailing text."
                else:
                    nearby = json.dumps(exc.doc[max(0, exc.pos - 40) : exc.pos + 60])
                    feedback = (
                        f"Invalid JSON at character {exc.pos}: {exc.msg}. "
                        f"Nearby text, shown as an escaped string: {nearby}. "
                        "Use this fragment to locate the error, not as the answer. Fix the JSON encoding. "
                        "Shell single quotes do not escape JSON double quotes. "
                        r"Inside JSON strings, encode double quotes as \", line breaks as \n and backslashes as \\. "
                    )
            elif str(exc) == "unfinished thinking block":
                feedback = "Finish thinking and close it with </think>, then return one complete JSON action."
            raise FormatError(
                message,
                {
                    "role": "user",
                    "content": (
                        f"Invalid tool call: {feedback} "
                        + ("Keep explanations inside your thinking block. " if self.thinking else "")
                        + "Your final answer must contain only the JSON object, "
                        "with no prose or Markdown before or after it. "
                        "Do not wrap actions in XML tool_call/function/parameter tags. "
                    ),
                },
            ) from exc
        return message

    def format_message(self, **kwargs: Any) -> dict:
        return kwargs

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        for action, output in zip(message.get("extra", {}).get("actions", []), outputs):
            text = str(output.get("output", ""))
            self.record(
                "tool_result",
                name=action["name"],
                rollout_n=action.get("rollout_n") if type(action.get("rollout_n")) is int else None,
                returncode=output.get("returncode"),
                output_chars=len(text),
                next_offset=output.get("next_offset"),
                rejection=text[:1000] if text.startswith("Tool rejected:") else None,
            )
        return [
            {"role": "user", "content": "Tool result:\n" + json.dumps(output, ensure_ascii=False)}
            for output in outputs
        ]

    def get_template_vars(self, **kwargs: Any) -> dict:
        return {}

    def serialize(self) -> dict:
        return {"info": {"model_type": "relax_genrm"}}


class GraderAgent(DefaultAgent):
    def query(self) -> dict:
        remaining = self.config.step_limit - self.n_calls
        time_limit = self.config.wall_time_limit_seconds
        seconds_left = time_limit - (time.time() - self._start_time)
        has_budget = (self.config.step_limit <= 0 or remaining > 0) and (time_limit <= 0 or seconds_left > 0)
        if has_budget and (0 < remaining <= 3 or 0 < seconds_left <= time_limit / 5):
            budget = f"{remaining} model calls remain" if self.config.step_limit > 0 else "Model calls are unlimited"
            if time_limit > 0:
                budget += f"; {int(seconds_left)} seconds remain"
            self.add_messages(
                self.model.format_message(
                    role="user",
                    content=(
                        f"{budget}, including submission. Stop further investigation "
                        "and call submit_grades using the evidence already collected. Do not invent evidence. "
                        "Correct any submission validation errors before the budget expires."
                    ),
                )
            )
        return super().query()


def _trajectory_view(messages: list[dict]) -> list[dict]:
    # Keep the frozen audit unchanged; avoid repeating model/tool payloads that
    # are already present in the top-level messages sent to the grader.
    view = []
    for message in messages:
        extra = message.get("extra")
        if not isinstance(extra, dict):
            view.append(message)
            continue
        extra = dict(extra)
        response = extra.get("response")
        choices = response.get("choices") if isinstance(response, dict) else None
        if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
            mirror = choices[0].get("message")
            if (
                isinstance(mirror, dict)
                and mirror
                and all(key in message and message[key] == value for key, value in mirror.items())
            ):
                extra["response"] = {
                    **response,
                    "choices": [{key: value for key, value in choices[0].items() if key != "message"}],
                }
        raw = extra.get("raw_output")
        if isinstance(raw, str) and isinstance(message.get("content"), str) and raw in message["content"]:
            del extra["raw_output"]
        view.append({**message, "extra": extra})
    return view


def read_evidence(manifest: dict, action: dict, sections: set[str]) -> dict:
    index, section = action["rollout_n"], action["section"]
    offset = action.get("offset", 0)
    if type(index) is not int or not 0 <= index < len(manifest["candidates"]):
        raise ValueError("unknown rollout_n")
    if section not in sections or type(offset) is not int or offset < 0:
        raise ValueError("invalid evidence section or offset")
    evidence = manifest["candidates"][index].get(section, "No evidence recorded")
    if section == "trajectory" and isinstance(evidence, list):
        evidence = _trajectory_view(evidence)
    text = evidence if isinstance(evidence, str) else json.dumps(evidence, ensure_ascii=False)
    return {
        "output": text[offset : offset + 12000],
        "next_offset": offset + 12000 if offset + 12000 < len(text) else None,
        "returncode": 0,
    }


def submit_result(result: dict) -> None:
    submission = json.dumps(result, ensure_ascii=False)
    raise Submitted(
        {"role": "exit", "content": submission, "extra": {"exit_status": "Submitted", "submission": submission}}
    )


class GraderBackend(Protocol):
    """Task-owned command execution and release of its grader tool session."""

    def execute(self, action: dict) -> dict: ...

    def close(self) -> None: ...


class GraderEnvironment:
    """Shared evidence and submission tools with task-owned bash and
    scoring."""

    def __init__(
        self,
        manifest: dict,
        *,
        backend: GraderBackend,
        grade_to_weights: Callable[[list[float], dict], dict],
        sections: set[str],
        evidence_access: str,
        environment_type: str,
    ) -> None:
        self.manifest = manifest
        self.backend = backend
        self.grade_to_weights = grade_to_weights
        self.sections = sections
        self.evidence_access = evidence_access
        self.environment_type = environment_type

    def execute(self, action: dict, cwd: str = "") -> dict:
        try:
            name = action["name"]
            if name == "bash":
                return self.backend.execute(action)
            if name == "read_evidence":
                return read_evidence(self.manifest, action, self.sections)
            if name == "submit_grades":
                rewards = [candidate["reward"] for candidate in self.manifest["candidates"]]
                submit_result(self.grade_to_weights(rewards, action))
            raise ValueError("unknown tool")
        except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
            return {"output": f"Tool rejected: {exc}", "returncode": -1}

    def close(self) -> None:
        self.backend.close()

    def get_template_vars(self, **kwargs: Any) -> dict:
        return {"evidence_access": self.evidence_access}

    def serialize(self) -> dict:
        # Keep capabilities and complete evidence out of per-step serialization.
        return {"info": {"environment_type": self.environment_type}}


def run_grader(manifest: dict, output: Path, env: Any, config: dict) -> None:
    model = GenRMModel(manifest["genrm_url"], unix_socket=manifest.get("genrm_socket"), **config.get("model", {}))
    model.audit_path = output.with_name("events.jsonl")
    model.group_id = manifest.get("group_id", output.parent.name)
    started = time.monotonic()
    agent = None
    error = ""
    try:
        candidates = [
            {"rollout_n": c["rollout_n"], "reward": c["reward"], "exit_status": c.get("exit_status", "")}
            for c in manifest["candidates"]
        ]
        agent_config = dict(config["agent"])
        agent = GraderAgent(model, env, output_path=output.with_name("trajectory.json"), **agent_config)
        model.record(
            "start",
            passing_ids=[c["rollout_n"] for c in candidates if c["reward"]],
            outcomes=[c["reward"] for c in candidates],
            step_limit=agent.config.step_limit,
        )
        result = agent.run(manifest["task"], candidates=json.dumps(candidates, ensure_ascii=False))
        if result.get("exit_status") != "Submitted":
            raise RuntimeError(f"grader did not submit: {result.get('exit_status')}")
        output.write_text(result["submission"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        model.client.close()
        last = agent.messages[-1].get("extra", {}) if agent and agent.messages else {}
        summary = {
            "group_id": model.group_id,
            "exit_status": last.get("exit_status", "initialization_failed"),
            "api_calls": model.n_queries,
            "http_attempts": model.http_attempts,
            "seconds": time.monotonic() - started,
            "error": error,
            "passing_ids": [c["rollout_n"] for c in manifest["candidates"] if c["reward"]],
        }
        model.record("finished", **{key: value for key, value in summary.items() if key != "group_id"})
        try:
            output.with_name("summary.json").write_text(json.dumps(summary, indent=2))
        except OSError:
            model.record("diagnostics_error", file="summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one task's agentic grader using Relax GenRM.")
    parser.add_argument("--task", choices=("code_generate", "minisweagent"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.input.read_text())
    if args.config is not None:
        config = load_task_config(args.task, args.config)
    elif "grader_config" in manifest:
        # The worker passes its resolved config into an isolated container.
        config = manifest["grader_config"]
    else:
        config = load_task_config(args.task)
    module, name = config["environment"]["adapter"].rsplit(".", 1)
    factory = getattr(importlib.import_module(module), name)
    env = factory(manifest, config["environment"])
    try:
        run_grader(manifest, args.output, env, config)
    finally:
        env.close()


if __name__ == "__main__":
    main()
