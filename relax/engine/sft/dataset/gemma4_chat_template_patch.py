# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""gemma-4 chat-template patches for SFT."""

import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from relax.engine.sft.dataset.chat_template_patch import TemplatePatchResult
from relax.engine.sft.dataset.sample import CanonicalSample


_PATCH_NAME = "gemma4_sft"
GEMMA4_SFT_THINKING_ENV = "GEMMA4_SFT_THINKING"
GEMMA4_SFT_TOOL_TURN_BOUNDARY_ENV = "GEMMA4_SFT_TOOL_TURN_BOUNDARY"
GEMMA4_SFT_MS_SWIFT_COMPAT_ENV = "GEMMA4_SFT_MS_SWIFT_COMPAT"
GEMMA4_SFT_THINKING_SENTINEL = "{# relax_gemma4_sft_thinking_v1 #}"
GEMMA4_SFT_TOOL_TURN_BOUNDARY_SENTINEL = "{# relax_gemma4_sft_tool_turn_boundary_v1 #}"
GEMMA4_SFT_MS_SWIFT_COMPAT_SENTINEL = "{# relax_gemma4_sft_ms_swift_compat_v1 #}"
_TRUE_VALUES = {"1", "true", "True"}

# The empty thought block ms-swift's `gemma4` template counts in the loss.
# The official Jinja emits it only for generation prompts, not training turns.
GEMMA4_EMPTY_THOUGHT = "<|channel>thought\n<channel|>"
GEMMA4_MS_SWIFT_COMPAT_KWARG = "_relax_gemma4_ms_swift_compat"

# Keep the input conversion in the template so an exported tokenizer applies
# the same policy without depending on Relax's Python rendering entry point.
_MS_SWIFT_COMPAT_PREFIX = (
    "{%- set ns_relax_compat = namespace(messages=[]) -%}\n"
    "{%- for message in messages -%}\n"
    "    {%- set ns_relax_compat.messages = ns_relax_compat.messages + "
    "[dict(message.items() | rejectattr('0', 'in', ['name', 'tool_call_id']))] -%}\n"
    "{%- endfor -%}\n"
    "{%- set messages = ns_relax_compat.messages -%}\n"
)

_ANCHOR = (
    "    {%- if thinking_text and thinking_gate -%}\n"
    "        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}\n"
    "    {%- endif -%}\n"
)
_REPLACEMENT = _ANCHOR + (
    "    {%- if role == 'model' and not continue_same_model_turn and not thinking_text -%}\n"
    "        {{- '<|channel>thought\\n<channel|>' -}}\n"
    "    {%- endif -%}\n"
)

# The official Gemma4 template keeps a tool call, its response, and the next
# assistant message inside one model turn. That is convenient for replaying a
# full tool trajectory, but it leaves no supervised stop token after
# ``<tool_call|>``. Serving stops a tool call at the turn boundary, so training
# must expose that boundary too. Start a fresh model turn before the first
# embedded tool response; the response remains context-only in the loss mask.
_TOOL_RESPONSE_STATE_ANCHOR = "            {%- set ns_tr_out = namespace(flag=false) -%}\n"
_TOOL_RESPONSE_STATE_REPLACEMENT = _TOOL_RESPONSE_STATE_ANCHOR + (
    "            {%- set ns_tool_turn = namespace(opened=false) -%}\n"
)
_TOOL_RESPONSE_CALL_ANCHOR = "{{- format_tool_response_block("
_TOOL_RESPONSE_CALL_REPLACEMENT = (
    "{%- if message.get('tool_calls') and not ns_tool_turn.opened -%}\n"
    "                            {{- '<turn|>\\n<|turn>model\\n' -}}\n"
    "                            {%- set ns_tool_turn.opened = true -%}\n"
    "                        {%- endif -%}\n"
    "                        {{- format_tool_response_block("
)
_UNRESOLVED_TOOL_CALL_ANCHOR = (
    "        {%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}\n"
    "            {{- '<|tool_response>' -}}\n"
)
_UNRESOLVED_TOOL_CALL_REPLACEMENT = (
    "        {%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}\n            {{- '<turn|>\\n' -}}\n"
)


def _is_gemma4_template(template: str) -> bool:
    """gemma-4 by its delimiters, not by model name (the caller may not know
    it)."""
    return "<|turn>" in template and "<channel|>" in template and "strip_thinking" in template


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0") in _TRUE_VALUES


def gemma4_sft_patch_options_from_env() -> tuple[bool, bool, bool]:
    """Return thinking, tool-boundary, and ms-swift-compat flags."""
    return (
        _env_enabled(GEMMA4_SFT_THINKING_ENV),
        _env_enabled(GEMMA4_SFT_TOOL_TURN_BOUNDARY_ENV),
        _env_enabled(GEMMA4_SFT_MS_SWIFT_COMPAT_ENV),
    )


@lru_cache(maxsize=64)
def _patch_template(
    template: str,
    *,
    thinking_enabled: bool,
    tool_turn_boundary_enabled: bool,
    ms_swift_compat: bool = False,
) -> tuple[str, bool] | None:
    """Apply independently enabled thinking and tool-turn patches.

    None = not gemma-4, don't touch.
    """
    if not _is_gemma4_template(template):
        return None
    patched = template
    changed = False

    if ms_swift_compat and GEMMA4_SFT_MS_SWIFT_COMPAT_SENTINEL not in patched:
        patched = GEMMA4_SFT_MS_SWIFT_COMPAT_SENTINEL + "\n" + _MS_SWIFT_COMPAT_PREFIX + patched
        changed = True

    if thinking_enabled and GEMMA4_SFT_THINKING_SENTINEL not in patched:
        # A template exported by the first implementation can already contain
        # the replacement but not the dedicated thinking sentinel. Recognize
        # that shape to avoid injecting the empty-thought branch twice.
        if _REPLACEMENT not in patched:
            count = patched.count(_ANCHOR)
            if count == 0:
                raise RuntimeError(
                    f"{_PATCH_NAME}: thinking anchor not found in the gemma-4 chat template. "
                    "Upstream changed it; re-derive the patch instead of letting it silently no-op."
                )
            if count > 1:
                raise RuntimeError(f"{_PATCH_NAME}: thinking anchor appears {count}x; refusing to guess.")
            patched = patched.replace(_ANCHOR, _REPLACEMENT, 1)
        patched = GEMMA4_SFT_THINKING_SENTINEL + "\n" + patched
        changed = True

    if not tool_turn_boundary_enabled or GEMMA4_SFT_TOOL_TURN_BOUNDARY_SENTINEL in patched:
        return patched, changed

    # Older/minimal Gemma4 templates may not implement tools at all. Keep the
    # other patches usable for those templates; tool-capable templates must
    # match every structural anchor so an upstream rewrite fails loudly.
    if "format_tool_response_block" not in patched:
        return patched, changed

    state_count = patched.count(_TOOL_RESPONSE_STATE_ANCHOR)
    if state_count != 1:
        raise RuntimeError(
            f"{_PATCH_NAME}: tool-response state anchor appears {state_count}x; "
            "upstream changed it, refusing to guess."
        )
    call_count = patched.count(_TOOL_RESPONSE_CALL_ANCHOR)
    if call_count == 0:
        raise RuntimeError(f"{_PATCH_NAME}: no tool-response render calls found; upstream changed the template.")
    unresolved_count = patched.count(_UNRESOLVED_TOOL_CALL_ANCHOR)
    if unresolved_count != 1:
        raise RuntimeError(
            f"{_PATCH_NAME}: unresolved tool-call anchor appears {unresolved_count}x; "
            "upstream changed it, refusing to guess."
        )

    patched = patched.replace(_TOOL_RESPONSE_STATE_ANCHOR, _TOOL_RESPONSE_STATE_REPLACEMENT, 1)
    patched = patched.replace(_TOOL_RESPONSE_CALL_ANCHOR, _TOOL_RESPONSE_CALL_REPLACEMENT)
    patched = patched.replace(_UNRESOLVED_TOOL_CALL_ANCHOR, _UNRESOLVED_TOOL_CALL_REPLACEMENT, 1)
    patched = GEMMA4_SFT_TOOL_TURN_BOUNDARY_SENTINEL + "\n" + patched
    return patched, True


def _validate_tool_trajectory(sample: CanonicalSample) -> None:
    """Reject unresolved tool calls in the middle of a training trajectory."""
    for index, message in enumerate(sample.messages[:-1]):
        if message.role == "assistant" and message.tool_calls and sample.messages[index + 1].role != "tool":
            raise ValueError(
                "Gemma4 SFT tool-turn boundaries require every non-terminal assistant tool call "
                "to be followed immediately by a tool response"
            )


def try_patch_gemma4_thinking(
    sample: CanonicalSample,
    template: str | None,
    kwargs: Mapping[str, Any],
) -> TemplatePatchResult | None:
    """Align Gemma4 SFT rendering with serving using independent flags.

    The scaffold adds an empty thought block at the start of each new model
    turn. Its four tokens count toward the loss, matching ms-swift without
    enabling the distinct `<|think|>` system prefix. Tool calls also end with a
    supervised turn token before embedded tool responses, so serving can stop
    after one tool-call batch instead of decoding repeated calls.
    """
    thinking_enabled, tool_turn_boundary_enabled, ms_swift_compat = gemma4_sft_patch_options_from_env()
    if not (thinking_enabled or tool_turn_boundary_enabled or ms_swift_compat):
        return None
    if not template:
        return None
    patched = _patch_template(
        template,
        thinking_enabled=thinking_enabled,
        tool_turn_boundary_enabled=tool_turn_boundary_enabled,
        ms_swift_compat=ms_swift_compat,
    )
    if patched is None:
        return None
    if tool_turn_boundary_enabled:
        _validate_tool_trajectory(sample)
    new_template, changed = patched
    new_kwargs = dict(kwargs)
    # ms-swift drops OpenAI tool-call correlation fields while converting to
    # its internal tool_call/tool roles. The renderer consumes this private
    # option and omits those fields only in explicit compatibility mode.
    if ms_swift_compat:
        new_kwargs[GEMMA4_MS_SWIFT_COMPAT_KWARG] = True
    return TemplatePatchResult(
        template=new_template,
        kwargs=new_kwargs,
        patch_name=_PATCH_NAME,
        changed=changed,
    )
