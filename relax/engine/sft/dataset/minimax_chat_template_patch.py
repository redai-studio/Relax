# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Enable assistant-response masks for MiniMax-M3 chat templates."""

from collections.abc import Mapping
from typing import Any

from relax.engine.sft.dataset.chat_template_patch import HAS_GENERATION_MARKER, TemplatePatchResult
from relax.engine.sft.dataset.sample import CanonicalSample


def try_patch_minimax_chat_template(
    _sample: CanonicalSample, template: str | None, kwargs: Mapping[str, Any]
) -> TemplatePatchResult | None:
    if not template or "]<]minimax[>[" not in template or HAS_GENERATION_MARKER(template):
        return None
    assistant = "{%- if message.role == 'assistant' -%}"
    tool = "{%- elif message.role == 'tool' -%}"
    eos = "{{- eos_token ~ '\\n' }}"
    before, start_marker, conversation = template.partition(assistant)
    block, end_marker, rest = conversation.partition(tool)
    # Miles' native SFT mask starts after the assistant header. The thinking
    # block, tool calls and visible text are all part of the generated reply.
    start = "{%- if reasoning_content -%}" if "{%- if reasoning_content -%}" in block else "{%- if content -%}"
    if not start_marker or not end_marker or block.count(start) != 1 or block.count(eos) != 1:
        return None
    block = block.replace(start, "{%- generation -%}\n        " + start)
    block = block.replace(eos, eos + "\n        {%- endgeneration -%}")
    return TemplatePatchResult(before + start_marker + block + end_marker + rest, dict(kwargs), "minimax_m3_assistant")
