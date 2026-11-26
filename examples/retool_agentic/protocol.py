# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Render one assistant stream, including verbatim interpreter observations."""

import re
from typing import Any

from jinja2 import Environment, StrictUndefined


# A user message ends without a trailing newline so Agentic's dummy-prefix
# trimming leaves an observation's leading newline intact. Only the first
# assistant segment gets a ChatML header; later segments continue its stream.
CHAT_TEMPLATE = r"""{%- if messages[0]['role'] != 'system' -%}
{{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' -}}
{%- endif -%}
{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
{{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' -}}
{%- elif message['role'] == 'user' -%}
{{- '<|im_start|>user\n' + message['content'] + '<|im_end|>' -}}
{%- elif message['role'] == 'assistant' -%}
{%- if loop.index0 > 0 and messages[loop.index0 - 1]['role'] == 'user' -%}
{{- '\n<|im_start|>assistant\n' -}}
{%- endif -%}
{{- message['content'] -}}
{%- elif message['role'] == 'tool' -%}
{{- message['content'] -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt and messages[-1]['role'] == 'user' -%}
{{- '\n<|im_start|>assistant\n' -}}
{%- endif -%}"""
_TEMPLATE = Environment(undefined=StrictUndefined).from_string(CHAT_TEMPLATE)


def render_prompt(messages: list[dict[str, Any]]) -> str:
    return _TEMPLATE.render(messages=messages, add_generation_prompt=True)


def observation_text(feedback: dict[str, str]) -> str:
    status = feedback["status"]
    text = feedback["stdout"] if status == "ok" else feedback["stderr"] or status
    # The wrapper is nonempty even for successful code with no stdout.
    return "\n<interpreter>\n" + text + "\n</interpreter>\n"


VERSION = "author_code_stateless_v1"

DIRECT_PREFIX = (
    "Solve the following problem step by step. No external tools are available; "
    "solve using your own reasoning. The last part of your response should be "
    "<answer>\\boxed{ANSWER}</answer>.\n\n*user question:*\n"
)

PREFIX = (
    "Solve the following problem step by step. You now have the ability to selectively write executable Python code "
    "to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output "
    "(wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive "
    "at the final answer. The Python code should be complete scripts, including necessary imports.\n"
    "Each code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\n"
    "Each call is a fresh stateless Python3 program; explicitly print results. No network, arbitrary files, "
    "subprocesses or threads;30seconds/1GiB per call. Do not invent interpreter outputs. "
    "End the assistant turn immediately after </code> and use the actual returned output to continue.\n"
    "The last part of your response should be <answer>\\boxed{ANSWER}</answer>.\n\n*user question:*\n"
)


def parse_code(text: str) -> str:
    if text.count("<code>") != 1 or text.count("</code>") != 1:
        raise ValueError("Exactly one complete code block per assistant turn")
    if "<interpreter>" in text or "</interpreter>" in text or not text.rstrip().endswith("</code>"):
        raise ValueError("Model cannot author an observation or continue past its tool boundary")
    block = text[text.index("<code>") + 6 : text.index("</code>")]
    match = re.fullmatch(r"\s*```python\s*\n(.*?)\n```\s*", block, re.DOTALL)
    if not match or not match.group(1).strip():
        raise ValueError("Require nonempty complete Python fenced source")
    return match.group(1)


async def feedback_for_turn(text: str, client: Any) -> tuple[dict, str | None]:
    try:
        code = parse_code(text)
    except ValueError as exc:
        return {"status": "invalid_call", "stdout": "", "stderr": str(exc)}, None
    if len(code.encode("utf-8")) > 65536:
        return {
            "status": "invalid_call",
            "stdout": "",
            "stderr": "Python source exceeds sandbox 65536-byte limit",
        }, None
    # Only this NEW model-emitted code is executed, exactly once. Infrastructure
    # exceptions propagate rather than becoming a model failure reward.
    feedback = await client.execute(code)
    if feedback.get("status") not in {"ok", "code_error", "timeout"}:
        raise RuntimeError("Unexpected sandbox infrastructure response")
    return feedback, code
