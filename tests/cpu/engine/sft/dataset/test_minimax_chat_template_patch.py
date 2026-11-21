# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from transformers.utils.chat_template_utils import render_jinja_template

from relax.engine.sft.dataset.minimax_chat_template_patch import try_patch_minimax_chat_template
from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample


# The assistant and tool branches use the same whitespace controls as the
# MiniMax template. A plain answer has a closing think marker in its prefix.
_MINIMAX_TEMPLATE = r"""
{%- set ns_token = ']<]minimax[>[' -%}
{%- for message in messages -%}
    {%- if message.role == 'assistant' -%}
        {{- 'ai\n</think>' }}
        {%- set content = message.content -%}
        {%- if content -%}
            {{- content }}
        {%- endif -%}
        {{- eos_token ~ '\n' }}
    {%- elif message.role == 'tool' -%}
        {{- 'tool\n' ~ message.content ~ eos_token }}
    {%- else -%}
        {{- 'user\n' ~ message.content ~ eos_token }}
    {%- endif -%}
{%- endfor -%}
"""


def test_minimax_patch_marks_answer_and_eos_without_changing_render() -> None:
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="Question", learn=False),
            CanonicalMessage(role="assistant", content="The answer", learn=True),
        ],
        metadata={"source_dataset": "unit-test", "row_index": 0},
    )
    conversations = [[{"role": message.role, "content": message.content} for message in sample.messages]]
    patch = try_patch_minimax_chat_template(sample, _MINIMAX_TEMPLATE, {"eos_token": "<eos>"})
    assert patch is not None
    original, _ = render_jinja_template(conversations, chat_template=_MINIMAX_TEMPLATE, eos_token="<eos>")
    rendered, spans = render_jinja_template(
        conversations, chat_template=patch.template, return_assistant_tokens_mask=True, **patch.kwargs
    )
    assert rendered == original
    assert [rendered[0][start:end] for start, end in spans[0]] == ["The answer<eos>\n"]
