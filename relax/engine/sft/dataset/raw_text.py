# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Raw-text rendering for upstream SFT comparison recipes."""

import os
from typing import Any

import torch

from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_RAW_TEXT_CONCAT_ENV = "RELAX_SFT_RAW_TEXT_CONCAT"
_RAW_TEXT_CONCAT_LOGGED = False


def raw_text_concat_enabled() -> bool:
    """True when the upstream-comparison rendering mode is switched on.

    Read from the environment on every call (not cached) so the mode can be
    toggled per run without touching any constructor signature.
    """
    return os.environ.get(_RAW_TEXT_CONCAT_ENV) == "1"


def _render_raw_text_concat(sample: CanonicalSample, *, tokenizer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Render a sample the way upstream nemo_automodel does — no chat template.

    Reproduces ``SFTSingleTurnPreprocessor._tokenize_function``
    (nemo_automodel/components/datasets/utils.py): tokenize the context and the
    target separately with the *bare* tokenizer, drop a trailing context EOS
    and a redundant target BOS, concatenate, and put the loss on the target
    tokens only. Preserve the initial BOS even when the context is empty.

    This exists so our loss is numerically comparable to upstream reference runs
    (e.g. examples/llm_finetune/kimi/k3_hellaswag.yaml). Our normal path renders
    through the model's chat template, which for Kimi K3 wraps the answer in
    structural tokens that land *inside* the loss span — roughly half of the
    masked tokens become near-deterministic scaffolding and mechanically halve
    the reported per-token loss.

    Trade-off: this trains base-model-style continuation, not instruct/chat
    behaviour, and drops K3's system preamble. Use it for upstream comparisons
    only, never to produce a deployable chat model.

    ``loss_mask`` marks the tokens that are *learned* (relax's convention; the
    training loop applies the next-token shift itself), so it is 0 across the
    context and 1 across the target.
    """
    global _RAW_TEXT_CONCAT_LOGGED

    target_index = next(
        (i for i in range(len(sample.messages) - 1, -1, -1) if sample.messages[i].learn),
        None,
    )
    if target_index is None:
        raise ValueError(
            f"{_RAW_TEXT_CONCAT_ENV}=1: sample has no learnable message to use as the target "
            f"(roles={[m.role for m in sample.messages]})."
        )

    def _text(message: CanonicalMessage) -> str:
        if not isinstance(message.content, str):
            raise TypeError(
                f"{_RAW_TEXT_CONCAT_ENV}=1 is text-only, but message role={message.role!r} "
                f"carries {type(message.content).__name__} content (multimodal is unsupported)."
            )
        return message.content

    # Upstream feeds one context string; joining the preceding turns keeps that
    # shape for the single-turn datasets this mode targets.
    context = "".join(_text(message) for message in sample.messages[:target_index])
    target = _text(sample.messages[target_index])

    context_ids = tokenizer(context)["input_ids"]
    target_ids = tokenizer(target)["input_ids"]
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    # An empty context may consist only of BOS, including tokenizers that
    # share a BOS/EOS ID. It must remain as the predictor for the first target.
    if context_ids and context_ids[-1] == eos_id and not (len(context_ids) == 1 and context_ids[0] == bos_id):
        context_ids = context_ids[:-1]
    if target_ids and target_ids[0] == bos_id:
        if not context_ids:
            context_ids = target_ids[:1]
        target_ids = target_ids[1:]

    if not _RAW_TEXT_CONCAT_LOGGED:
        logger.warning(
            f"SFT rendering: {_RAW_TEXT_CONCAT_ENV}=1 -- bypassing the chat template and training on "
            "raw context+target concatenation (upstream-comparison mode). This is NOT the normal "
            "instruct/chat recipe; unset the variable to restore template rendering."
        )
        _RAW_TEXT_CONCAT_LOGGED = True

    input_ids = [*context_ids, *target_ids]
    loss_mask = [0] * len(context_ids) + [1] * len(target_ids)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(loss_mask, dtype=torch.long)
