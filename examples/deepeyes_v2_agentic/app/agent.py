# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""DeepEyes V2 agentic-stack agent driver.

One process per Relax-managed session. Talks to ``AgenticChatAPIService`` over
OpenAI-compatible chat completions. Parses inline ``<answer>`` / ``<code>`` /
``<tool_call>`` tags and routes to ``DeepEyesV2Env``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from app.env_deepeyes_v2 import (
    DeepEyesV2Env,
    ToolObs,
    encode_image_data_uri,
    extract_answer,
    extract_tool_call,
)
from app.sandboxes import SandboxExecutor, get_sandbox_backend
from PIL import Image


CONFIG_PATH = Path(__file__).with_name("deepeyes_v2_config.yaml")


def read_session_input(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_session_output(path: str | Path, payload: dict[str, Any]) -> None:
    # The launcher may remove a discarded session's temporary directory while
    # the agent is finishing its output write.
    try:
        Path(path).write_text(json.dumps(payload), encoding="utf-8")
    except FileNotFoundError:
        pass


def load_initial_image(messages: list[dict[str, Any]]) -> Image.Image | None:
    """Pull the first image attached to the last user message, if any.

    Dataset prompts arrive with image data URLs in
    ``content[*].image_url.url``.
    """
    if not messages:
        return None
    last = messages[-1]
    content = last.get("content")
    if not isinstance(content, list):
        return None
    for item in content:
        if item.get("type") != "image_url":
            continue
        url = item["image_url"]["url"]
        _, _, encoded = url.partition(",")
        with BytesIO(base64.b64decode(encoded)) as fh:
            img = Image.open(fh)
            img.load()
            return img
    return None


def build_tool_message(*, body_text: str, images: list[Image.Image]) -> dict[str, Any]:
    """Build a ``role:"tool"`` observation message.

    Image-bearing messages MUST use the OpenAI content-part array with
    ``data:image/png;base64,...`` URLs — that's the only shape the agentic chat
    service accepts on a tool turn (see
    ``relax/agentic/session/service.py:485``).
    """
    if not images:
        return {"role": "tool", "content": body_text}
    parts: list[dict[str, Any]] = [{"type": "text", "text": body_text}]
    parts.extend({"type": "image_url", "image_url": {"url": encode_image_data_uri(img)}} for img in images)
    return {"role": "tool", "content": parts}


# Precedence: terminal <answer> wins over <code> wins over <tool_call> wins over format_error.
# Under the unified schema, a <tool_call> with name=python_exec is also a code branch
# (env.exec_code's extractor knows both <code>...</code> and the tool_call payload form).
def classify_branch(text: str) -> str:
    if extract_answer(text) is not None:
        return "answer"
    if "<code>" in text:
        return "code"
    if "<tool_call>" in text:
        tc = extract_tool_call(text)
        if tc and tc.get("name") == "python_exec":
            return "code"
        return "tool_call"
    return "format_error"


def _build_executor(backend_name: str, ensure_sandbox_timeout_s: int) -> SandboxExecutor:
    """Instantiate the sandbox backend (with optional config from
    ``SANDBOX_CONFIG_PATH``) and hand it to the executor singleton.

    Mirrors xide's ``_get_sandbox_executor`` indirection: backend kwargs live
    in a YAML pointed to by ``SANDBOX_CONFIG_PATH`` so the agent process never
    hardcodes container paths.

    Env-var overrides (so the launcher can swap backends/images without
    editing the YAML):

    * ``SANDBOX_BACKEND``: overrides the backend name from caller.
    * ``APPTAINER_IMAGE_PATH``: overrides ``image`` in the loaded YAML
      (only meaningful for apptainer-style backends).

    Relative paths in the YAML (e.g. ``image: ./foo.sif``) resolve against
    the YAML's parent directory, not the agent process CWD.
    """
    backend_name = os.environ.get("SANDBOX_BACKEND", backend_name)
    config_path = os.environ.get("SANDBOX_CONFIG_PATH")
    backend_kwargs: dict[str, Any] = {}
    config_dir: Path | None = None
    if config_path:
        config_path_p = Path(config_path)
        backend_kwargs = yaml.safe_load(config_path_p.read_text(encoding="utf-8")) or {}
        config_dir = config_path_p.parent
    image_override = os.environ.get("APPTAINER_IMAGE_PATH")
    if image_override:
        backend_kwargs["image"] = image_override
    if config_dir is not None:
        for key in ("image",):
            val = backend_kwargs.get(key)
            if isinstance(val, str) and val and not os.path.isabs(val):
                backend_kwargs[key] = str(config_dir / val)
    backend = get_sandbox_backend(name=backend_name, config=backend_kwargs)
    return SandboxExecutor.get_or_create(backend, ensure_session_timeout_s=ensure_sandbox_timeout_s)


async def run_session(messages: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    from openai import (  # type: ignore[import-not-found]
        APIStatusError,
        AsyncOpenAI,
    )

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    max_turns = int(config["max_turns"])
    ensure_sandbox_timeout_s = int(config["ensure_sandbox_timeout_s"])

    data_index = str(metadata.get("data_index") or metadata.get("index") or "?")
    executor = _build_executor(config["sandbox_backend"], ensure_sandbox_timeout_s)
    env = DeepEyesV2Env(
        data_index=data_index,
        sandbox_executor=executor,
        image=load_initial_image(messages),
        code_timeout_s=int(config["code_timeout_s"]),
        ensure_sandbox_timeout_s=ensure_sandbox_timeout_s,
    )

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"].rstrip("/"),
        timeout=9999,
    )

    stop_reason = "max_turns"
    branch_counts = {"answer": 0, "code": 0, "tool_call": 0, "format_error": 0}
    final_answer: str | None = None
    last_error: str | None = None

    # Token-id stops only: matched on the GPU side, bypass the detokenizer
    # subprocess. String stops (SGLang "stop") add per-token substring-match
    # work in the detokenizer and can starve its heartbeat under agentic
    # high concurrency, so we avoid them entirely. See deepeyes_v2_config.yaml.
    extra_body: dict[str, Any] = {}
    stop_token_ids = config.get("stop_token_ids") or []
    if stop_token_ids:
        extra_body["stop_token_ids"] = list(stop_token_ids)

    try:
        for _turn in range(max_turns):
            try:
                response = await client.chat.completions.create(
                    model=os.environ.get("OPENAI_MODEL", "model"),
                    messages=messages,
                    extra_body=extra_body,
                )
            except APIStatusError as exc:
                err = (exc.response.json() or {}).get("error", {})
                code = err.get("code") if isinstance(err, dict) else None
                if code == "context_length_exceeded":
                    stop_reason = "finish_length"
                    break
                # A session discarded after the rollout target closes returns
                # 404 session_discarded. Its output JSON is no longer consumed.
                if code == "session_discarded":
                    stop_reason = "discarded_by_pipeline"
                    break
                raise

            text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": text})
            if response.choices[0].finish_reason == "length":
                stop_reason = "finish_length"
                break

            branch = classify_branch(text)
            branch_counts[branch] += 1

            if branch == "answer":
                final_answer = extract_answer(text)
                stop_reason = "env_done"
                break
            if branch == "format_error":
                stop_reason = "format_error"
                last_error = "no_recognised_tag"
                break

            obs: ToolObs
            if branch == "code":
                obs = await env.exec_code(text)
            else:
                obs = await env.exec_tool(text)

            if obs.error:
                last_error = obs.error
            messages.append(build_tool_message(body_text=obs.body_text, images=obs.images))
            if obs.done:
                stop_reason = "env_done" if obs.error is None else f"env_error:{obs.error}"
                break
    finally:
        await env.close()

    # This app uses the implicit SessionOutput form from
    # relax.agentic.runner.ipc. Relax already captured the chat trajectory.
    return {
        "metadata": {
            "stop_reason": stop_reason,
            "branch_counts": branch_counts,
            "final_answer": final_answer,
            "last_error": last_error,
            "data_source": metadata.get("data_source"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepEyes V2 agentic session.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_input = read_session_input(args.input_json)
    metadata = session_input.get("metadata") or {}
    output = asyncio.run(run_session(messages=session_input["messages"], metadata=metadata))
    write_session_output(args.output_json, output)


if __name__ == "__main__":
    main()
