# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dispatch model-specific, per-call chat-template adaptations for SFT."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from relax.engine.sft.dataset.sample import CanonicalSample


# `-?` accepts the dashed-whitespace-control variants `{%- generation -%}` and
# `{%-generation-%}` that Jinja allows. The dashes only suppress surrounding
# whitespace and are semantically identical to the plain `{% generation %}`
# form for the purpose of marking assistant-token spans, so they should be
# recognised as the same marker.
_GENERATION_MARKER_RE = re.compile(r"{%-?\s*generation\s*-?%}")


def HAS_GENERATION_MARKER(template_str: str | None) -> bool:  # noqa: N802
    if not template_str:
        return False
    return bool(_GENERATION_MARKER_RE.search(template_str))


@dataclass(frozen=True)
class TemplatePatchResult:
    """A patched template and kwargs without mutating tokenizer state."""

    template: str | None
    kwargs: dict[str, Any]
    patch_name: str | None = None
    changed: bool = False


TemplatePatcher: TypeAlias = Callable[
    [CanonicalSample, str | None, Mapping[str, Any]],
    TemplatePatchResult | None,
]


def apply_chat_template_patchers(
    sample: CanonicalSample,
    template: str | None,
    kwargs: Mapping[str, Any],
    *,
    patchers: Sequence[TemplatePatcher],
) -> TemplatePatchResult:
    """Apply the unique matching patcher, or return a strict no-op result."""
    matches: list[TemplatePatchResult] = []
    for patcher in patchers:
        result = patcher(sample, template, dict(kwargs))
        if result is not None:
            matches.append(result)

    if len(matches) > 1:
        names = [result.patch_name or "<unnamed>" for result in matches]
        raise RuntimeError(f"Multiple chat-template patchers matched the same template: {names}")
    if matches:
        result = matches[0]
        changed = result.template != template
        resolved_kwargs = dict(result.kwargs)
        if changed:
            if result.template is None:
                raise RuntimeError(f"Chat-template patcher {result.patch_name!r} removed the effective template")
            resolved_kwargs["chat_template"] = result.template
        elif "chat_template" in resolved_kwargs and resolved_kwargs["chat_template"] != result.template:
            raise RuntimeError(
                f"Chat-template patcher {result.patch_name!r} returned inconsistent template and kwargs"
            )
        return TemplatePatchResult(
            template=result.template,
            kwargs=resolved_kwargs,
            patch_name=result.patch_name,
            changed=changed,
        )
    return TemplatePatchResult(template=template, kwargs=dict(kwargs))
