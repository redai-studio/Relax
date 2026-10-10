# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""DeepSeek-V4 chat-template compatibility patches for SFT.

DeepSeek-V4's assistant turn emits a scaffold think token right after
``<｜Assistant｜>`` when the predecessor is a user/developer/tool turn: with
``thinking`` off (the SFT default) it emits a bare ``</think>``. For CoT SFT
data whose ``content`` already carries its own ``<think>…</think>…answer``, that
scaffold produces a malformed DOUBLE-``</think>`` render
(``<｜Assistant｜></think><think>…``). The renderer normalizes those messages to
``reasoning_content`` + ``content`` and this patcher renders their think block
per message. Plain historical assistants keep the native scaffold, matching the
model's official multi-turn format.

The official history gate is also backported with the same
``preserve_thinking`` bool/null/auto contract as Qwen. This only controls
reasoning that the input message already carries; it never reconstructs text
that is absent from the sample.

Mirrors ``qwen_chat_template_patch.py``; registered in
``chat_template._CHAT_TEMPLATE_PATCHERS``.
"""

import hashlib
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from relax.engine.sft.dataset.chat_template_patch import TemplatePatchResult
from relax.engine.sft.dataset.sample import CanonicalSample


# DeepSeek-V4 markers (fullwidth U+FF5C `｜`, U+2581 `▁`), used only to detect a
# DeepSeek template so this patcher never collides with the Qwen one.
_DS_USER = "<｜User｜>"
_DS_ASSISTANT = "<｜Assistant｜>"

# The assistant-turn think scaffold emitted after ``<｜Assistant｜>`` on a
# user/developer/tool predecessor (``chat_template.jinja`` lines 186-194, the
# nested ``{%- if keep_reasoning and thinking -%}…{%- endif -%}`` block inside
# the ``elif ep.is_ud`` branch). Copied byte-for-byte incl. indentation; the
# ``count() == 1`` guard in ``_patch_deepseek_scaffold`` turns any future
# template edit into a loud failure instead of a silent mis-render.
_DS_ASSISTANT_SCAFFOLD = (
    "      {%- if keep_reasoning and thinking -%}\n"
    "        {{- thinking_start_token -}}\n"
    "        {%- if message['reasoning_content'] is defined and message['reasoning_content'] -%}\n"
    "          {{- message['reasoning_content'] -}}\n"
    "        {%- endif -%}\n"
    "        {{- thinking_end_token -}}\n"
    "      {%- else -%}\n"
    "        {{- thinking_end_token -}}\n"
    "      {%- endif -%}"
)

# The corresponding scaffold for an assistant whose effective predecessor is
# not a user/developer/tool turn (for example, consecutive assistant turns).
# It is inactive under the SFT default ``thinking=false``, but must be guarded
# as well so an explicit thinking-mode render cannot prepend ``</think>`` to
# content that already embeds its own think block.
_DS_ASSISTANT_NON_TRANSITION_SCAFFOLD = (
    "      {%- if keep_reasoning and thinking -%}\n"
    "        {%- if message['reasoning_content'] is defined and message['reasoning_content'] -%}\n"
    "          {{- message['reasoning_content'] -}}\n"
    "        {%- endif -%}\n"
    "        {{- thinking_end_token -}}\n"
    "      {%- endif -%}"
)
_DS_HISTORY_GATE = "    {%- set keep_reasoning = tp.has or (loop.index0 > last_user_idx.value) -%}"
_DS_PRESERVE_HISTORY_GATE = (
    "    {%- set keep_reasoning = tp.has or "
    "(preserve_thinking is defined and preserve_thinking is true) or "
    "(loop.index0 > last_user_idx.value) -%}"
)
_EMBEDDED_THINK_INDICES_KEY = "_relax_deepseek_embedded_think_indices"
_PER_MESSAGE_GUARD = f"loop.index0 not in {_EMBEDDED_THINK_INDICES_KEY}"
_PATCH_NAME = "deepseek_v4_scaffold_think"
# Optional explicit override, carried inside --apply-chat-template-kwargs JSON.
# Popped before the kwargs reach the jinja render context. No dedicated CLI flag.
_SUPPRESS_KEY = "deepseek_suppress_scaffold_think"


def split_embedded_thinking(content: str | list[dict]) -> tuple[str | None, str | list[dict]]:
    """Split a leading ``<think>…</think>`` block without changing its
    bytes."""
    if not isinstance(content, str) or not content.startswith("<think>"):
        return None, content
    close_pos = content.find("</think>", len("<think>"))
    if close_pos < 0:
        return None, content
    return content[len("<think>") : close_pos], content[close_pos + len("</think>") :]


def _assistant_indices_with_think(sample: CanonicalSample) -> tuple[int, ...]:
    """Indices of all assistant turns carrying structured or embedded
    thinking."""
    indices = []
    for index, message in enumerate(sample.messages):
        if message.role != "assistant":
            continue
        embedded_reasoning, _ = split_embedded_thinking(message.content)
        if message.reasoning_content is not None or embedded_reasoning is not None:
            indices.append(index)
    return tuple(indices)


def _has_learnable_historical_thinking(sample: CanonicalSample) -> bool:
    """Whether a later user-like turn makes supervised thinking historical."""
    last_user_like_index = max(
        (index for index, message in enumerate(sample.messages) if message.role in {"user", "developer", "tool"}),
        default=-1,
    )
    if last_user_like_index < 0:
        return False
    return any(
        index < last_user_like_index
        and message.role == "assistant"
        and message.learn
        and (message.reasoning_content is not None or split_embedded_thinking(message.content)[0] is not None)
        for index, message in enumerate(sample.messages)
    )


@lru_cache(maxsize=32)
def _patch_deepseek_history_gate(template: str) -> tuple[str, bool] | None:
    """Backport the Qwen-style preserve override to DeepSeek's history gate."""
    old_count = template.count(_DS_HISTORY_GATE)
    native_count = template.count(_DS_PRESERVE_HISTORY_GATE)
    looks_like_deepseek_history = "set keep_reasoning" in template and "last_user_idx.value" in template
    if old_count == 0 and native_count == 0 and not looks_like_deepseek_history:
        return None
    if old_count == 1 and native_count == 0:
        return template.replace(_DS_HISTORY_GATE, _DS_PRESERVE_HISTORY_GATE, 1), True
    if old_count == 0 and native_count == 1:
        return template, False

    template_hash = hashlib.sha256(template.encode()).hexdigest()[:16]
    raise RuntimeError(
        "Cannot safely patch the DeepSeek-V4 history-thinking gate: "
        f"expected one old gate or one native gate, found old={old_count} and native={native_count} "
        f"(template sha256={template_hash})."
    )


def _guard_scaffold(scaffold: str, *, emits_plain_scaffold: bool) -> str:
    plain_scaffold = "          {{- thinking_end_token -}}\n" if emits_plain_scaffold else ""
    return (
        f"      {{%- if not ({_PER_MESSAGE_GUARD}) -%}}\n"
        "        {%- if message['reasoning_content'] is defined and message['reasoning_content'] is not none -%}\n"
        "          {%- if keep_reasoning -%}\n"
        "            {{- thinking_start_token -}}\n"
        "            {{- message['reasoning_content'] -}}\n"
        "            {{- thinking_end_token -}}\n"
        "          {%- else -%}\n"
        f"{plain_scaffold}"
        "          {%- endif -%}\n"
        "        {%- endif -%}\n"
        "      {%- else -%}\n"
        f"{scaffold}\n"
        "      {%- endif -%}"
    )


@lru_cache(maxsize=32)
def _patch_deepseek_scaffold(template: str) -> str:
    """Render structured thinking or the native scaffold per assistant
    message."""
    guard_count = template.count(_PER_MESSAGE_GUARD)
    if guard_count == 2:
        return template

    transition_count = template.count(_DS_ASSISTANT_SCAFFOLD)
    non_transition_count = template.count(_DS_ASSISTANT_NON_TRANSITION_SCAFFOLD)
    if guard_count != 0 or transition_count != 1 or non_transition_count != 1:
        template_hash = hashlib.sha256(template.encode()).hexdigest()[:16]
        raise RuntimeError(
            "Cannot safely patch the DeepSeek-V4 assistant think scaffold: "
            "expected exactly one transition and one non-transition scaffold "
            f"with no existing guards, found transition={transition_count}, "
            f"non_transition={non_transition_count}, guards={guard_count} "
            f"(template sha256={template_hash})."
        )
    return template.replace(
        _DS_ASSISTANT_SCAFFOLD,
        _guard_scaffold(_DS_ASSISTANT_SCAFFOLD, emits_plain_scaffold=True),
        1,
    ).replace(
        _DS_ASSISTANT_NON_TRANSITION_SCAFFOLD,
        _guard_scaffold(_DS_ASSISTANT_NON_TRANSITION_SCAFFOLD, emits_plain_scaffold=False),
        1,
    )


def try_patch_deepseek_chat_template(
    sample: CanonicalSample,
    template: str | None,
    kwargs: Mapping[str, Any],
) -> TemplatePatchResult | None:
    """Render the DeepSeek-V4 assistant think scaffold per SFT message.

    Auto policy (default): preserve supervised thinking made historical by a
    later user-like message, and suppress each assistant scaffold iff that
    message's content already embeds a ``<think>…</think>`` block. Explicit
    ``preserve_thinking`` and ``deepseek_suppress_scaffold_think`` booleans in
    ``apply_chat_template_kwargs`` override those decisions; the latter is
    popped before render because it is Relax-internal.
    """
    if not isinstance(template, str):
        return None
    if _DS_ASSISTANT not in template or _DS_USER not in template:
        return None

    history_patch = _patch_deepseek_history_gate(template)
    if history_patch is None:
        patched_template = template
        history_changed = False
    else:
        patched_template, history_changed = history_patch

    resolved_kwargs = dict(kwargs)
    preserve_thinking = resolved_kwargs.get("preserve_thinking")
    if preserve_thinking is not None and not isinstance(preserve_thinking, bool):
        raise ValueError("apply_chat_template_kwargs.preserve_thinking must be true, false, or null")
    if preserve_thinking is None and _has_learnable_historical_thinking(sample):
        resolved_kwargs["preserve_thinking"] = True

    explicit = resolved_kwargs.pop(_SUPPRESS_KEY, None)
    resolved_kwargs.pop(_EMBEDDED_THINK_INDICES_KEY, None)
    if explicit is not None and not isinstance(explicit, bool):
        raise ValueError(f"apply_chat_template_kwargs.{_SUPPRESS_KEY} must be true, false, or null")

    if explicit is False:
        suppress_indices: tuple[int, ...] = ()
    elif explicit is True:
        suppress_indices = tuple(index for index, message in enumerate(sample.messages) if message.role == "assistant")
    else:
        suppress_indices = _assistant_indices_with_think(sample)

    if not suppress_indices:
        # The scaffold may be unchanged while the history gate was patched.
        # Still return a result so the escape-hatch key stays popped.
        return TemplatePatchResult(
            template=patched_template,
            kwargs=resolved_kwargs,
            patch_name=_PATCH_NAME,
            changed=history_changed,
        )

    resolved_kwargs[_EMBEDDED_THINK_INDICES_KEY] = suppress_indices
    scaffold_template = _patch_deepseek_scaffold(patched_template)
    return TemplatePatchResult(
        template=scaffold_template,
        kwargs=resolved_kwargs,
        patch_name=_PATCH_NAME,
        changed=history_changed or scaffold_template != template,
    )
