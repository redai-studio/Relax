# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SFT leaf helpers used inside backends/megatron/{data,loss}.py.

Lives under ``relax.utils`` (rather than ``relax.engine.sft``) so the
Megatron backend modules can call into it without pulling the whole SFT
engine subtree — which itself imports the backend — into the import graph.
Only depends on ``torch`` / ``torch.nn.functional``.

SFT batches use the ``response_length == total_length`` convention (no
prompt/response split). Both helpers below dispatch on that condition.
"""

import re

import torch
import torch.nn.functional as F


_LAYER_PARAMETER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def sft_grad_debug_bucket(parameter_name: str) -> str:
    """Map a parameter name to a compact layer/module gradient bucket."""
    if any(tag in parameter_name for tag in ("embed_tokens", "word_embeddings", "embedding.")):
        return "embedding"
    if "lm_head" in parameter_name or "output_layer" in parameter_name:
        return "lm_head"
    match = _LAYER_PARAMETER_RE.search(parameter_name)
    if match is None:
        return "other"
    layer = int(match.group(1))
    suffix = parameter_name[match.end() :]
    if "indexer" in suffix:
        kind = "qsa_indexer"
    elif any(tag in suffix for tag in ("self_attn", "self_attention", ".attention.")):
        kind = "attention"
    elif "expert" in suffix:
        kind = "experts"
    elif "router" in suffix or suffix.startswith("mlp.gate."):
        kind = "router"
    elif any(tag in suffix for tag in ("gdn", "linear_attn", "linear_attention")):
        kind = "gdn"
    else:
        kind = "other"
    return f"layer_{layer:03d}.{kind}"


def align_loss_mask_for_sft(loss_mask: torch.Tensor) -> torch.Tensor:
    """Shift loss_mask left by 1 for next-token loss alignment."""
    return F.pad(loss_mask, (0, 1), value=0)[1:]


def compute_sft_response_chunk(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-token slice for the SFT branch of ``get_responses``.

    SFT has ``prompt_length == 0`` (``response_length == total_length``). The
    RL slice ``logits[start-1:end-1]`` underflows when ``start==0`` (first
    sample) and is mis-aligned for later samples. Use next-token slice
    instead: ``logit[i]`` predicts ``token[i+1]``. The last position is a
    dummy label (masked out by ``align_loss_mask_for_sft``).
    """
    logits_chunk = logits[start:end]
    tokens_chunk = F.pad(tokens[1:], (0, 1), value=0)
    return logits_chunk, tokens_chunk
