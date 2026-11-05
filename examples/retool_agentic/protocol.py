# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Render one assistant stream, including verbatim interpreter observations."""

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
