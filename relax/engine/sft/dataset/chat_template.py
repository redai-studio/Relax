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
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

from relax.engine.sft.dataset.chat_template_patch import TemplatePatchResult, apply_chat_template_patchers
from relax.engine.sft.dataset.gemma4_chat_template_patch import try_patch_gemma4_thinking
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
_CHAT_TEMPLATE_PATCHERS = (try_patch_qwen_chat_template, try_patch_gemma4_thinking)
_FALLBACK_WARNED: set[int] = set()  # tokenizer id → warned once
_EMPTY_THINK_UNSUPPORTED_WARNED: set[int] = set()  # tokenizer id → warned once
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


def _last_round_learn_indices(sample: CanonicalSample) -> set[int]:
    """Learnable messages belonging to the final user round.

    "Last round" is every learnable response after the final user query, not
    just the last assistant message: an assistant tool-call, its tool response,
    and the final answer are all one round.
    """
    last_user_index = max((i for i, m in enumerate(sample.messages) if m.role == "user"), default=-1)
    return {i for i, m in enumerate(sample.messages) if i > last_user_index and m.learn}


def _render_with_assistant_mask(
    sample: CanonicalSample,
    *,
    tokenizer,
    apply_chat_template_kwargs: dict | None = None,
    last_turn_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Path 1: ask tokenizer for the assistant-only mask directly."""
    apply_chat_template_kwargs = _thread_local_chat_template_kwargs(
        tokenizer,
        apply_chat_template_kwargs,
    )
    messages, template_kwargs = _prepare_chat_messages(
        sample, apply_chat_template_kwargs, last_turn_only=last_turn_only
    )
    result = tokenizer.apply_chat_template(
        messages,
        tools=sample.tools,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        return_assistant_tokens_mask=True,
        **template_kwargs,
    )
    input_ids = result["input_ids"]
    masks = result["assistant_masks"]
    if isinstance(masks, list):
        masks = torch.tensor(masks)
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    if masks.dim() == 2:
        masks = masks.squeeze(0)
    masks = masks.long()
    if last_turn_only:
        # HF's assistant_masks mark every assistant span with a run of 1s
        # (0s over tool/user turns between them). The last round can span
        # several such runs (tool-call + answer), so keep one trailing run per
        # learnable assistant message in it — not just the final run, which
        # would drop an earlier tool-call span and diverge from Path 2.
        n_runs = _last_round_assistant_run_count(sample)
        masks = _keep_last_n_mask_runs(masks, n_runs)
    return input_ids.long(), masks


# Roles that apply_chat_template wraps in a {% generation %} block, so HF's
# return_assistant_tokens_mask marks them with a run of 1s. Tool/user/system
# turns are never marked, so they separate assistant runs with 0s.
_ASSISTANT_MASKED_ROLES = {"assistant", "function_call"}


def _last_round_assistant_run_count(sample: CanonicalSample) -> int:
    """Number of learnable assistant-masked messages in the final user round.

    Assumes one message renders as exactly one ``{% generation %}`` block (one
    run of 1s), which holds for every shipped template. A custom template that
    split a message across two blocks would keep one run too few — dropping,
    never over-including, a supervised span (fails safe).
    """
    last_round = _last_round_learn_indices(sample)
    return sum(1 for i in last_round if sample.messages[i].role in _ASSISTANT_MASKED_ROLES)


def _keep_last_n_mask_runs(masks: torch.Tensor, n: int) -> torch.Tensor:
    """Zero out all but the final ``n`` contiguous runs of 1s in a 1D 0/1 mask.

    ``n <= 0`` clears the mask; ``n`` >= the number of runs is a no-op.
    """
    if masks.numel() == 0 or int(masks.sum()) == 0:
        return masks
    if n <= 0:
        return torch.zeros_like(masks)
    m = masks.tolist()
    total = len(m)
    out = [0] * total
    runs_kept = 0
    i = total - 1
    while i >= 0 and runs_kept < n:
        if m[i] == 0:
            i -= 1
            continue
        # walk back over this contiguous run of 1s, copying it into out
        while i >= 0 and m[i] == 1:
            out[i] = 1
            i -= 1
        runs_kept += 1
    return torch.tensor(out, dtype=masks.dtype)


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
_THINK_OPEN_TAG = "<think>"
_THINK_CLOSE_TAG = "</think>"
_NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"
_IM_END = "<|im_end|>"
# Qwen3.5 wraps tool messages inside a user block as
# `<tool_response>\n{content}\n</tool_response>`, so role=="tool" has no
# `<|im_start|>tool\n` header — scan the wrapper instead.
_TOOL_RESPONSE_OPEN = "<tool_response>\n"
_TOOL_RESPONSE_CLOSE = "\n</tool_response>"


@dataclass(frozen=True)
class _Dialect:
    """Turn delimiters for the fallback's text scan.

    ChatML and gemma-4 frame turns differently; same scan, other delimiters.
    """

    name: str
    role_names: dict  # canonical role -> the name the template renders
    header_fmt: str  # "{role}" placeholder
    end: str
    think_open: str | None  # None: nothing to exclude — the whole reply is learned
    think_close: str | None  # None: skip only the opener (ChatML <think>\n)
    supports_tools: bool

    def header(self, role: str) -> str:
        return self.header_fmt.format(role=self.role_names.get(role, role))


_CHATML = _Dialect(
    name="chatml",
    role_names={},
    header_fmt="<|im_start|>{role}\n",
    end=_IM_END,
    think_open=_THINK_OPEN,
    think_close=None,
    supports_tools=True,
)

# gemma-4 renders the assistant role as "model". Reasoning is a delimited block,
# so the mask must resume after <channel|> rather than after a fixed-length
# opener. Matches THUDM/slime's gen_multi_turn_loss_mask_gemma4.
_GEMMA4 = _Dialect(
    name="gemma4",
    role_names={"assistant": "model"},
    header_fmt="<|turn>{role}\n",
    end="<turn|>",
    think_open="<|channel>thought\n",
    think_close="<channel|>",
    supports_tools=False,
)

# Same delimiters, but the reasoning block stays IN the loss. Only reachable via
# GEMMA4_SFT_THINKING=1, which patches the Jinja to emit an empty thought block
# on every assistant turn -- see gemma4_chat_template_patch.py.
_GEMMA4_THINKING = replace(_GEMMA4, name="gemma4_thinking", think_open=None, think_close=None)


def _detect_dialect(rendered_text: str) -> _Dialect:
    """Pick delimiters from what the template actually emitted."""
    if "<|turn>" in rendered_text and "<turn|>" in rendered_text:
        if os.environ.get("GEMMA4_SFT_THINKING", "0") in ("1", "true", "True"):
            return _GEMMA4_THINKING
        return _GEMMA4
    return _CHATML


def _prepare_chat_messages(
    sample: CanonicalSample,
    apply_chat_template_kwargs: dict | None,
    *,
    last_turn_only: bool,
) -> tuple[list[dict[str, Any]], dict]:
    """Prepare template input, including the ms-swift non-thinking prefix.

    ``add_non_thinking_prefix`` is an ms-swift preprocessing option rather than
    a Hugging Face chat-template variable.  Consume it here so launchers can
    request the same training input without forwarding an inert kwarg to Jinja.
    With last-round loss, ms-swift only adds the prefix to assistant messages
    after the final user query; otherwise it considers every assistant message.
    """
    template_kwargs = dict(apply_chat_template_kwargs or {})
    add_non_thinking_prefix = bool(template_kwargs.pop("add_non_thinking_prefix", False))
    messages = _to_chat_messages(sample)
    if not add_non_thinking_prefix:
        return messages, template_kwargs

    start_index = (
        max((i for i, message in enumerate(messages) if message["role"] == "user"), default=-1)
        if last_turn_only
        else -1
    )
    for index, message in enumerate(messages):
        content = message.get("content")
        if (
            index >= start_index
            and message["role"] == "assistant"
            and isinstance(content, str)
            and not content.startswith((_THINK_OPEN_TAG, _NON_THINKING_PREFIX))
        ):
            message["content"] = _NON_THINKING_PREFIX + content
    return messages, template_kwargs


def _empty_think_region_end(rendered_text: str, content_start: int, span_end: int) -> int:
    """If the assistant content at ``content_start`` opens with a `<think>`
    block whose inner text is empty/whitespace, return the char offset just
    past the closing `</think>` and any trailing whitespace (so the whole
    empty-think region can be kept out of the loss). Otherwise return -1.

    Handles both `<think>\\n\\n</think>` (Qwen3.5 non-thinking default) and
    `<think></think>`. Only the region up to ``span_end`` is considered.
    """
    if rendered_text[content_start : content_start + len(_THINK_OPEN_TAG)] != _THINK_OPEN_TAG:
        return -1
    inner_start = content_start + len(_THINK_OPEN_TAG)
    close_pos = rendered_text.find(_THINK_CLOSE_TAG, inner_start, span_end)
    if close_pos < 0:
        return -1
    if rendered_text[inner_start:close_pos].strip() != "":
        return -1  # non-empty think — keep it (only default opener-skip applies)
    end = close_pos + len(_THINK_CLOSE_TAG)
    # swallow trailing whitespace/newlines after </think> so they don't train
    while end < span_end and rendered_text[end] in (" ", "\n", "\t", "\r"):
        end += 1
    return end


def _render_per_message_fallback(
    sample: CanonicalSample,
    *,
    tokenizer,
    apply_chat_template_kwargs: dict | None = None,
    last_turn_only: bool = False,
    ignore_empty_think: bool = False,
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
    msgs, extra_kwargs = _prepare_chat_messages(sample, apply_chat_template_kwargs, last_turn_only=last_turn_only)
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

    last_round_learn_indices = _last_round_learn_indices(sample) if last_turn_only else set()

    char_mask = bytearray(len(rendered_text))  # zeros
    dialect = _detect_dialect(rendered_text)
    cursor = 0
    for msg_idx, msg in enumerate(sample.messages):
        if msg.role == "tool":
            if not dialect.supports_tools:
                raise RuntimeError(
                    f"tool messages are not supported by the {dialect.name!r} loss-mask dialect; "
                    f"add its tool-call delimiters to _Dialect first"
                )
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
            header = dialect.header(msg.role)
            header_pos = rendered_text.find(header, cursor)
            if header_pos < 0:
                raise RuntimeError(
                    f"could not locate {msg.role!r} message after cursor {cursor} in rendered chat "
                    f"template output (dialect={dialect.name!r}, header={header!r})"
                )
            content_start = header_pos + len(header)
            end_pos = rendered_text.find(dialect.end, content_start)
            if end_pos < 0:
                raise RuntimeError(
                    f"could not locate {dialect.end!r} for {msg.role!r} message (dialect={dialect.name!r})"
                )
            span_end = end_pos + len(dialect.end)
            if span_end < len(rendered_text) and rendered_text[span_end] == "\n":
                span_end += 1
            cursor = span_end

        if not msg.learn:
            continue
        if last_turn_only and msg_idx not in last_round_learn_indices:
            continue

        mask_start = content_start
        if msg.role == "assistant":
            think_end = (
                _empty_think_region_end(rendered_text, content_start, span_end)
                if ignore_empty_think and dialect is _CHATML
                else -1
            )
            if think_end >= 0:
                # empty `<think>…</think>` region (plus trailing whitespace)
                # stays out of the loss.
                mask_start = think_end
            elif dialect.think_open is not None and rendered_text.startswith(dialect.think_open, content_start):
                if dialect.think_close is None:
                    # ChatML: only the opener is excluded; the reasoning body is learned.
                    mask_start += len(dialect.think_open)
                else:
                    # Delimited block (gemma-4): the whole reasoning span stays out of
                    # the loss, so training targets only the visible reply.
                    close_pos = rendered_text.find(dialect.think_close, content_start, span_end)
                    if close_pos < 0:
                        raise RuntimeError(f"found {dialect.think_open!r} without a matching {dialect.think_close!r}")
                    mask_start = close_pos + len(dialect.think_close)
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
    last_turn_only: bool = False,
    ignore_empty_think: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render a single sample.

    Returns 1D `(input_ids, loss_mask)` int64 tensors.

    ``apply_chat_template_kwargs`` is forwarded to
    ``tokenizer.apply_chat_template`` so callers can pass e.g.
    ``{"chat_template": "<custom jinja>"}`` to override the model's
    native chat template — useful when the native template is designed
    for inference and silently drops content needed for training
    (e.g. DeepSeek-R1 distill templates strip ``<think>...</think>``).

    ``last_turn_only``: when True, only the final learnable round contributes to
    the loss; earlier assistant/function_call turns are masked. Applied to both
    render paths.

    ``ignore_empty_think``: when True, an empty ``<think></think>`` block inside
    an assistant turn is kept entirely out of the loss (not just its opener tag).
    Only supported on the per-message fallback path (the ``{% generation %}``
    template path has no think info).
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
        if ignore_empty_think and tok_id not in _EMPTY_THINK_UNSUPPORTED_WARNED:
            logger.warning(
                "--sft-ignore-empty-think has no effect on the {%% generation %%} "
                "template path (HF assistant_masks are token-level with no think "
                "info). It only applies to the per-message fallback path."
            )
            _EMPTY_THINK_UNSUPPORTED_WARNED.add(tok_id)
        return _render_with_assistant_mask(
            sample, tokenizer=tokenizer, apply_chat_template_kwargs=merged, last_turn_only=last_turn_only
        )

    if tok_id not in _FALLBACK_WARNED:
        logger.warning(
            "Tokenizer chat_template does not contain {%% generation %%} tag — "
            "falling back to per-message tokenization for SFT loss_mask. "
            "Mask boundaries may differ slightly from the template-aware path. "
            "(This warning is shown once per tokenizer instance.)"
        )
        _FALLBACK_WARNED.add(tok_id)
    return _render_per_message_fallback(
        sample,
        tokenizer=tokenizer,
        apply_chat_template_kwargs=merged,
        last_turn_only=last_turn_only,
        ignore_empty_think=ignore_empty_think,
    )


def render_to_text(
    sample: CanonicalSample,
    *,
    tokenizer,
    apply_chat_template_kwargs: dict | None = None,
    last_turn_only: bool = False,
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
    messages, template_kwargs = _prepare_chat_messages(sample, patch_result.kwargs, last_turn_only=last_turn_only)
    return tokenizer.apply_chat_template(
        messages,
        tools=sample.tools,
        tokenize=False,
        **template_kwargs,
    )
