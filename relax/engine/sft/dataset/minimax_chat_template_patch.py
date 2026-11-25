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
    start = "{%- if content -%}"
    eos = "{{- eos_token ~ '\\n' }}"
    before, start_marker, conversation = template.partition(assistant)
    block, end_marker, rest = conversation.partition(tool)
    if not start_marker or not end_marker or block.count(start) != 1 or block.count(eos) != 1:
        return None
    # MiniMax emits the same closer after a rationale and before a direct reply.
    # Mark the closer in both branches while leaving the rationale as context.
    think_end = "{{- think_end_token }}"
    block = block.replace("~ reasoning_content ~ think_end_token }}", "~ reasoning_content }}" + think_end)
    block = block.replace(think_end, "{%- generation -%}" + think_end + "{%- endgeneration -%}")
    block = block.replace(start, "{%- generation -%}\n        " + start)
    block = block.replace(eos, eos + "\n        {%- endgeneration -%}")
    return TemplatePatchResult(before + start_marker + block + end_marker + rest, dict(kwargs), "minimax_m3_assistant")
