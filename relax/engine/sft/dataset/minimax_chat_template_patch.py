# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Enable assistant-response masks for MiniMax-M3 chat templates."""

import re
from collections.abc import Mapping
from typing import Any

from relax.engine.sft.dataset.chat_template_patch import TemplatePatchResult
from relax.engine.sft.dataset.sample import CanonicalSample


_GENERATION = re.compile(r"{%-?\s*generation\s*-?%}")


def try_patch_minimax_chat_template(
    _sample: CanonicalSample, template: str | None, kwargs: Mapping[str, Any]
) -> TemplatePatchResult | None:
    if not template or "]<]minimax[>[" not in template or _GENERATION.search(template):
        return None
    assistant = "{%- if message.role == 'assistant' -%}"
    tool = "{%- elif message.role == 'tool' -%}"
    start = "{%- if content -%}"
    eos = "{{- eos_token ~ '\\n' }}"
    before, start_marker, conversation = template.partition(assistant)
    block, end_marker, rest = conversation.partition(tool)
    if not start_marker or not end_marker or block.count(start) != 1 or block.count(eos) != 1:
        return None
    block = block.replace(start, "{%- generation -%}\n        " + start)
    block = block.replace(eos, eos + "\n        {%- endgeneration -%}")
    return TemplatePatchResult(before + start_marker + block + end_marker + rest, dict(kwargs), "minimax_m3_assistant")
