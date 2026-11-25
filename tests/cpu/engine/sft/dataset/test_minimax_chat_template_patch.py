# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast
from transformers.utils.chat_template_utils import render_jinja_template

from relax.engine.sft.dataset.chat_template import render_with_loss_mask
from relax.engine.sft.dataset.minimax_chat_template_patch import try_patch_minimax_chat_template
from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample


_MINIMAX_TEMPLATE = r"""
{%- set ns_token = ']<]minimax[>[' -%}
{%- set bos_token = ']~b]' -%}
{%- set think_begin_token = '<mm:think>' -%}
{%- set think_end_token = '</mm:think>' -%}
{%- for message in messages -%}
    {%- if message.role == 'assistant' -%}
        {{- bos_token ~ 'ai' ~ '\n' }}
        {%- set content = message.content -%}
        {%- set reasoning_content = message.reasoning_content -%}
        {%- if reasoning_content -%}
            {{- think_begin_token ~ reasoning_content ~ think_end_token }}
        {%- else -%}
            {{- think_end_token }}
        {%- endif -%}
        {%- if content -%}
            {{- content }}
        {%- endif -%}
        {%- if message.tool_calls -%}
            {{- ns_token ~ '<tool_call>' ~ message.tool_calls[0].function.name ~ ns_token ~ '</tool_call>' }}
        {%- endif -%}
        {{- eos_token ~ '\n' }}
    {%- elif message.role == 'tool' -%}
        {{- 'tool\n' ~ message.content ~ eos_token }}
    {%- else -%}
        {{- 'user\n' ~ message.content ~ eos_token }}
    {%- endif -%}
{%- endfor -%}
"""


def test_minimax_patch_supervises_full_assistant_replies() -> None:
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="Question", learn=False),
            CanonicalMessage(
                role="assistant",
                content="",
                reasoning_content="Inspect the picture.",
                tool_calls=[{"type": "function", "function": {"name": "crop", "arguments": {}}}],
                learn=True,
            ),
            CanonicalMessage(role="tool", content="Crop result", learn=False),
            CanonicalMessage(role="assistant", content="The answer", learn=True),
        ],
        metadata={"source_dataset": "unit-test", "row_index": 0},
    )
    conversations = [[vars(message) for message in sample.messages]]
    patch = try_patch_minimax_chat_template(sample, _MINIMAX_TEMPLATE, {"eos_token": "<eos>"})
    assert patch is not None
    original, _ = render_jinja_template(conversations, chat_template=_MINIMAX_TEMPLATE, eos_token="<eos>")
    rendered, spans = render_jinja_template(
        conversations, chat_template=patch.template, return_assistant_tokens_mask=True, **patch.kwargs
    )
    assert rendered == original
    learned = "".join(rendered[0][start:end] for start, end in spans[0])
    assert learned == (
        "<mm:think>Inspect the picture.</mm:think>]<]minimax[>[<tool_call>crop]<]minimax[>[</tool_call><eos>\n"
        "</mm:think>The answer<eos>\n"
    )
    assert try_patch_minimax_chat_template(sample, patch.template, patch.kwargs) is None


def test_minimax_excluded_turn_remains_in_context() -> None:
    backend = Tokenizer(models.BPE(vocab={chr(i): i for i in range(128)}, merges=[]))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", chat_template=_MINIMAX_TEMPLATE)
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="First?", learn=False),
            CanonicalMessage(
                role="assistant", content="First reply", reasoning_content="Skipped reasoning", learn=False
            ),
            CanonicalMessage(role="user", content="Second?", learn=False),
            CanonicalMessage(role="assistant", content="Second reply", reasoning_content="Kept reasoning", learn=True),
        ],
        metadata={"source_dataset": "unit-test", "row_index": 0},
    )
    ids, mask = render_with_loss_mask(sample, tokenizer=tokenizer)
    assert "Skipped reasoning</mm:think>First reply" in tokenizer.decode(ids)
    assert tokenizer.decode(ids[mask.bool()]) == "<mm:think>Kept reasoning</mm:think>Second reply<eos>\n"
