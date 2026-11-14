# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch

from relax.backends.megatron.misc_utils import strip_param_name_prefix


def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Remove vocab padding: param[:vocab_size] for embedding/output layers,
    else unchanged."""
    logical_name = strip_param_name_prefix(name).removeprefix("language_model.")
    if logical_name in {"embedding.word_embeddings.weight", "output_layer.weight"}:
        return param[:vocab_size]
    return param
