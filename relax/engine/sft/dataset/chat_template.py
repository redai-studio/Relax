# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Render CanonicalSample → (input_ids, loss_mask) tensors via tokenizer chat
template.

Two paths (spec §7.5):
1. Preferred: `apply_chat_template(..., return_assistant_tokens_mask=True)` — relies
   on the model's official jinja template containing `{% generation %}` tags.
2. Fallback: per-message tokenize and concatenate, using `CanonicalMessage.learn`
   to build the mask. Used when the template lacks `{% generation %}`.
"""

import hashlib
import re
import threading
from collections.abc import Mapping
from typing import Any

import torch

from relax.engine.sft.dataset.chat_template_patch import TemplatePatchResult, apply_chat_template_patchers
from relax.engine.sft.dataset.qwen_chat_template_patch import try_patch_qwen_chat_template
from relax.engine.sft.dataset.sample import CanonicalSample
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
# `-?` accepts the dashed-whitespace-control variants `{%- generation -%}` and
# `{%-generation-%}` that Jinja allows. The dashes only suppress surrounding
# whitespace and are semantically identical to the plain `{% generation %}`
# form for the purpose of marking assistant-token spans, so they should be
# recognised as the same marker.
_GENERATION_MARKER_RE = re.compile(r"{%-?\s*generation\s*-?%}")
_CHAT_TEMPLATE_PATCHERS = (try_patch_qwen_chat_template,)
_FALLBACK_WARNED: set[int] = set()  # tokenizer id → warned once
_TEMPLATE_LOGGED: set[tuple[int, int, str]] = set()  # tokenizer id + template hash + preserve mode


def HAS_GENERATION_MARKER(template_str: str | None) -> bool:  # noqa: N802
    if not template_str:
        return False
    return bool(_GENERATION_MARKER_RE.search(template_str))


def _to_chat_messages(sample: CanonicalSample) -> list[dict[str, Any]]:
    """Convert CanonicalMessage list to dict format expected by
    apply_chat_template."""
    out = []
    for m in sample.messages:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls is not None:
            d["tool_calls"] = m.tool_calls
        out.append(d)
    return out


def _render_with_assistant_mask(
    sample: CanonicalSample,
    *,
    tokenizer,
    apply_chat_template_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Path 1: ask tokenizer for the assistant-only mask directly."""
    apply_chat_template_kwargs = _thread_local_chat_template_kwargs(
        tokenizer,
        apply_chat_template_kwargs,
    )
    result = tokenizer.apply_chat_template(
        _to_chat_messages(sample),
        tools=sample.tools,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        return_assistant_tokens_mask=True,
        **(apply_chat_template_kwargs or {}),
    )
    input_ids = result["input_ids"]
    masks = result["assistant_masks"]
    if isinstance(masks, list):
        masks = torch.tensor(masks)
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    if masks.dim() == 2:
        masks = masks.squeeze(0)
    return input_ids.long(), masks.long()


def _thread_local_chat_template_kwargs(tokenizer, apply_chat_template_kwargs: dict | None) -> dict:
    """Avoid sharing HF AssistantTracker state across prefetch threads.

    Transformers caches compiled Jinja templates by the template string. The
    compiled environment owns the ``AssistantTracker`` used by
    ``return_assistant_tokens_mask=True``, and that tracker is not thread-safe.
    Appending a Jinja comment makes each prefetch thread use a distinct
    compiled environment without changing rendered text.
    """
    kwargs = dict(apply_chat_template_kwargs or {})
    template = kwargs.get("chat_template") or getattr(tokenizer, "chat_template", None)
    if isinstance(template, str):
        kwargs["chat_template"] = f"{template}{{# relax_thread={threading.get_ident()} #}}"
    return kwargs


_THINK_OPEN = "<think>\n"
_IM_END = "<|im_end|>"
# Qwen3.5 wraps tool messages inside a user block as
# `<tool_response>\n{content}\n</tool_response>`, so role=="tool" has no
# `<|im_start|>tool\n` header — scan the wrapper instead.
_TOOL_RESPONSE_OPEN = "<tool_response>\n"
_TOOL_RESPONSE_CLOSE = "\n</tool_response>"


def _render_per_message_fallback(
    sample: CanonicalSample,
    *,
    tokenizer,
    apply_chat_template_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Path 2: single full render + char-level mask projected back through
    `offset_mapping`.

    Approach mirrors slime PR THUDM/slime#1742: rendering messages one at a
    time breaks on templates that validate the message list as a whole
    (e.g. Qwen3.5-VL aborts with "No user query found in messages." on an
    assistant-only list) and also on templates that re-render past turns
    based on the full message sequence (Qwen3.5-VL drops `<think>` blocks
    from prior assistants once a new user turn appears, so any
    chunked-prefix length delta is wrong on multi-turn). Instead:

      1. ``apply_chat_template(messages, tokenize=False)`` → ``rendered_text``
      2. fast-tokenize that text with ``return_offsets_mapping=True``
      3. sanity-check the re-tokenize matches the direct tokenize
      4. scan the text for ChatML ``<|im_start|>{role}\\n…<|im_end|>`` spans
         in declaration order, marking chars 1 for messages where
         ``learn=True`` (skipping the leading ``<think>\\n`` opener inside
         an assistant turn so the tag itself stays out of the loss)
      5. project char-mask → token-mask via a prefix-sum on ``offset_mapping``

    Requires a fast tokenizer; raises ``ValueError`` otherwise. Assumes
    Qwen-style ChatML wrapping — non-ChatML templates should expose
    ``{% generation %}`` markers so Path 1 handles them natively.
    """
    msgs = _to_chat_messages(sample)
    extra_kwargs = apply_chat_template_kwargs or {}
    rendered_text = tokenizer.apply_chat_template(msgs, tools=sample.tools, tokenize=False, **extra_kwargs)

    tokenized = tokenizer(rendered_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = tokenized["input_ids"]
    offset_mapping = tokenized.get("offset_mapping")
    if offset_mapping is None:
        raise ValueError(
            "SFT loss-mask fallback requires a fast tokenizer with "
            "`return_offsets_mapping` support; got a slow tokenizer."
        )

    expected = tokenizer.apply_chat_template(msgs, tools=sample.tools, tokenize=True, **extra_kwargs)
    if isinstance(expected, Mapping):
        expected = expected["input_ids"]
    if isinstance(expected, torch.Tensor):
        if expected.dim() > 1:
            expected = expected[0]
        expected = expected.tolist()
    elif len(expected) > 0 and isinstance(expected[0], list):
        expected = expected[0]
    if list(token_ids) != list(expected):
        raise RuntimeError(
            "Rendered-text re-tokenization does not match direct "
            "`apply_chat_template(..., tokenize=True)` output; mask projection "
            "via offset_mapping would be unreliable."
        )

    char_mask = bytearray(len(rendered_text))  # zeros
    cursor = 0
    for msg in sample.messages:
        if msg.role == "tool":
            open_pos = rendered_text.find(_TOOL_RESPONSE_OPEN, cursor)
            if open_pos < 0:
                raise RuntimeError(
                    f"could not locate <tool_response> for tool message after cursor {cursor} "
                    f"in rendered chat template output"
                )
            content_start = open_pos + len(_TOOL_RESPONSE_OPEN)
            close_pos = rendered_text.find(_TOOL_RESPONSE_CLOSE, content_start)
            if close_pos < 0:
                raise RuntimeError("could not locate </tool_response> for tool message")
            span_end = close_pos
            cursor = close_pos + len(_TOOL_RESPONSE_CLOSE)
        else:
            header = f"<|im_start|>{msg.role}\n"
            header_pos = rendered_text.find(header, cursor)
            if header_pos < 0:
                raise RuntimeError(
                    f"could not locate {msg.role!r} message after cursor {cursor} in rendered chat template output"
                )
            content_start = header_pos + len(header)
            end_pos = rendered_text.find(_IM_END, content_start)
            if end_pos < 0:
                raise RuntimeError(f"could not locate <|im_end|> for {msg.role!r} message")
            span_end = end_pos + len(_IM_END)
            if span_end < len(rendered_text) and rendered_text[span_end] == "\n":
                span_end += 1
            cursor = span_end

        if not msg.learn:
            continue

        mask_start = content_start
        if msg.role == "assistant" and rendered_text[content_start : content_start + len(_THINK_OPEN)] == _THINK_OPEN:
            mask_start += len(_THINK_OPEN)
        for pos in range(mask_start, span_end):
            char_mask[pos] = 1

    psum = [0] * (len(char_mask) + 1)
    for i, c in enumerate(char_mask):
        psum[i + 1] = psum[i] + c

    loss_mask = [0] * len(token_ids)
    for i, (s, e) in enumerate(offset_mapping):
        if e > s and psum[e] - psum[s] > 0:
            loss_mask[i] = 1

    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(loss_mask, dtype=torch.long)


def _merge_per_sample_kwargs(sample: CanonicalSample, apply_chat_template_kwargs: dict | None) -> dict:
    """Merge global apply_chat_template_kwargs with any per-sample override in
    ``sample.metadata['apply_chat_template_kwargs']``. Per-sample wins.

    Mirrors the pattern in ``relax/utils/data/data_utils.py`` so a sample can
    override (e.g.) ``chat_template`` or ``enable_thinking`` on the fly. Most
    callers leave this empty and rely on the launcher-supplied ``--apply-chat-
    template-kwargs`` global.
    """
    per_sample = None
    if sample.metadata is not None and isinstance(sample.metadata, dict):
        per_sample = sample.metadata.get("apply_chat_template_kwargs")
    return {**(apply_chat_template_kwargs or {}), **(per_sample or {})}


def _resolve_sft_template_kwargs(
    sample: CanonicalSample,
    *,
    tokenizer,
    apply_chat_template_kwargs: dict | None,
) -> TemplatePatchResult:
    """Merge template kwargs and apply the unique matching model adapter."""
    merged = _merge_per_sample_kwargs(sample, apply_chat_template_kwargs)
    effective_template = merged.get("chat_template") or getattr(tokenizer, "chat_template", None)
    return apply_chat_template_patchers(
        sample,
        effective_template,
        merged,
        patchers=_CHAT_TEMPLATE_PATCHERS,
    )


def render_with_loss_mask(
    sample: CanonicalSample,
    *,
    tokenizer,
    apply_chat_template_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render a single sample.

    Returns 1D `(input_ids, loss_mask)` int64 tensors.

    ``apply_chat_template_kwargs`` is forwarded to
    ``tokenizer.apply_chat_template`` so callers can pass e.g.
    ``{"chat_template": "<custom jinja>"}`` to override the model's
    native chat template — useful when the native template is designed
    for inference and silently drops content needed for training
    (e.g. DeepSeek-R1 distill templates strip ``<think>...</think>``).
    """
    patch_result = _resolve_sft_template_kwargs(
        sample,
        tokenizer=tokenizer,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )
    merged = patch_result.kwargs
    effective_template = patch_result.template
    # Effective template is the override if provided, else the tokenizer's
    # bound template. We must consult the EFFECTIVE template (not the
    # tokenizer's) to pick path 1 vs path 2 — passing a chat_template kwarg
    # that has {% generation %} markers should still take the fast path
    # even if the tokenizer's own bound template lacks them.
    # Log the effective template once per tokenizer instance so users can
    # confirm an apply_chat_template_kwargs override is actually taking
    # effect (sha256 + short prefix; the full template can be 1000s of chars).
    tok_id = id(tokenizer)
    eff = effective_template or ""
    log_key = (tok_id, hash(eff), repr(merged.get("preserve_thinking")))
    if log_key not in _TEMPLATE_LOGGED:
        template_hash = hashlib.sha256(eff.encode()).hexdigest()[:16]
        bound = getattr(tokenizer, "chat_template", None) or ""
        if patch_result.patch_name:
            status = "patched" if patch_result.changed else "native"
            source = f"{patch_result.patch_name} ({status})"
        elif merged.get("chat_template"):
            source = "override (apply_chat_template_kwargs)"
        else:
            source = "tokenizer.chat_template"
        eff_preview = eff[:120].replace("\n", "\\n")
        logger.info(
            f"SFT chat_template source={source} "
            f"sha256={template_hash} "
            f"len={len(eff)} "
            f"matches_bound={eff == bound} "
            f"has_generation_marker={HAS_GENERATION_MARKER(eff)} "
            f"preserve_thinking={merged.get('preserve_thinking')!r} "
            f"preview={eff_preview!r}"
        )
        _TEMPLATE_LOGGED.add(log_key)

    if HAS_GENERATION_MARKER(effective_template):
        return _render_with_assistant_mask(sample, tokenizer=tokenizer, apply_chat_template_kwargs=merged)

    if tok_id not in _FALLBACK_WARNED:
        logger.warning(
            "Tokenizer chat_template does not contain {%% generation %%} tag — "
            "falling back to per-message tokenization for SFT loss_mask. "
            "Mask boundaries may differ slightly from the template-aware path. "
            "(This warning is shown once per tokenizer instance.)"
        )
        _FALLBACK_WARNED.add(tok_id)
    return _render_per_message_fallback(sample, tokenizer=tokenizer, apply_chat_template_kwargs=merged)


def render_to_text(
    sample: CanonicalSample,
    *,
    tokenizer,
    apply_chat_template_kwargs: dict | None = None,
) -> str:
    """Render a sample to the chat-template text WITHOUT tokenizing.

    Used by the multimodal path: the text (containing un-expanded
    ``<|image_pad|>`` etc. placeholders) is fed into the HF processor, which
    expands those placeholders to per-image-grid token runs and produces the
    ``input_ids`` the model actually consumes.
    """
    patch_result = _resolve_sft_template_kwargs(
        sample,
        tokenizer=tokenizer,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )
    return tokenizer.apply_chat_template(
        _to_chat_messages(sample),
        tools=sample.tools,
        tokenize=False,
        **patch_result.kwargs,
    )
