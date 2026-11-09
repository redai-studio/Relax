# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""An explicit output-token range shared by policy scoring and sampling."""

from argparse import Namespace
from typing import Any

from relax.engine.sft.runtime import is_offline_mode
from relax.utils.misc import get_hf_config


def resolve_policy_vocab_size(args: Namespace, *, model_vocab_size: int | None = None) -> int | None:
    """Resolve the optional ``policy_vocab_size`` custom YAML setting.

    This is independent of Megatron's embedding/padding size and does not infer
    the valid output range from a tokenizer's length.
    """
    limit = getattr(args, "policy_vocab_size", None)
    if limit is None:
        return None
    if is_offline_mode(args):
        raise ValueError("policy_vocab_size is supported only for online RL policy scoring and sampling")
    if getattr(args, "train_backend", "megatron") != "megatron":
        raise ValueError("policy_vocab_size currently requires the Megatron training backend")
    if model_vocab_size is None:
        config = get_hf_config(args.hf_checkpoint)
        model_vocab_size = getattr(config, "text_config", config).vocab_size
    if type(limit) is not int or not 0 < limit <= model_vocab_size:
        raise ValueError(f"policy_vocab_size must be an integer in [1, {model_vocab_size}], got {limit!r}")
    return limit


def configure_policy_vocab_request(args: Namespace, payload: dict[str, Any]) -> None:
    """Exclude the same output rows on SGLang as on the training backend."""
    if getattr(args, "policy_vocab_size", None) is None:
        return
    config = get_hf_config(args.hf_checkpoint)
    model_vocab_size = getattr(config, "text_config", config).vocab_size
    limit = resolve_policy_vocab_size(args, model_vocab_size=model_vocab_size)
    sampling = dict(payload["sampling_params"])
    bias = dict(sampling.get("logit_bias") or {})
    bias.update({str(token): -1e30 for token in range(limit, model_vocab_size)})
    sampling["logit_bias"] = bias
    payload["sampling_params"] = sampling
