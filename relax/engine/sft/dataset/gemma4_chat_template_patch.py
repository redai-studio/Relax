# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""gemma-4 thinking-structure chat-template patch for SFT."""

import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from relax.engine.sft.dataset.chat_template_patch import TemplatePatchResult
from relax.engine.sft.dataset.sample import CanonicalSample


_PATCH_NAME = "gemma4_thinking"
_ENV_FLAG = "GEMMA4_SFT_THINKING"

# The empty thought block ms-swift's `gemma4` template counts in the loss.
# Unreachable in the official Jinja, which is why this patch exists.
GEMMA4_EMPTY_THOUGHT = "<|channel>thought\n<channel|>"

_ANCHOR = "{{- strip_thinking(message['content']) -}}"
_REPLACEMENT = (
    "{%- if enable_thinking and not thinking_text -%}"
    "{{- '<|channel>thought\\n<channel|>' -}}"
    "{%- endif -%}"
    "{{- strip_thinking(message['content']) -}}"
)


def _is_gemma4_template(template: str) -> bool:
    """gemma-4 by its delimiters, not by model name (the caller may not know
    it)."""
    return "<|turn>" in template and "<channel|>" in template and "strip_thinking" in template


@lru_cache(maxsize=32)
def _patch_template(template: str) -> tuple[str, bool] | None:
    """Insert the empty-thought emission.

    None = not gemma-4, don't touch.
    """
    if not _is_gemma4_template(template):
        return None
    # Fail loud rather than fuzzy-match.
    count = template.count(_ANCHOR)
    if count == 0:
        raise RuntimeError(
            f"{_PATCH_NAME}: anchor not found in the gemma-4 chat template. "
            "Upstream changed it; re-derive the patch instead of letting it "
            "silently no-op."
        )
    if count > 1:
        raise RuntimeError(f"{_PATCH_NAME}: anchor appears {count}x; refusing to guess which one to patch.")
    return template.replace(_ANCHOR, _REPLACEMENT, 1), True


def try_patch_gemma4_thinking(
    sample: CanonicalSample,
    template: str | None,
    kwargs: Mapping[str, Any],
) -> TemplatePatchResult | None:
    """Emit ms-swift's `gemma4` thinking scaffold. Off unless
    GEMMA4_SFT_THINKING=1.

    The scaffold adds a `<|think|>` system turn and an empty thought block that
    counts toward the loss -- 4 identical tokens on every sample. Only worth
    turning on to match a baseline that has them.
    """
    if os.environ.get(_ENV_FLAG, "0") not in ("1", "true", "True"):
        return None
    if not template:
        return None
    patched = _patch_template(template)
    if patched is None:
        return None
    new_template, _ = patched
    new_kwargs = dict(kwargs)
    # Drives the `<|turn>system\n<|think|>\n<turn|>\n` block (template line ~189).
    new_kwargs["enable_thinking"] = True
    return TemplatePatchResult(
        template=new_template,
        kwargs=new_kwargs,
        patch_name=_PATCH_NAME,
        changed=True,
    )
