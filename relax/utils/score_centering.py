# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Capture and validate Score Centering behavior probabilities on the CPU."""

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pybase64

from relax.utils.types import Sample


SCORE_CENTERING_FIELDS = ("rollout_topk_token_ids", "rollout_topk_log_probs")
_PROBABILITY_TOLERANCE = 1e-5


def score_centering_enabled(args: Any) -> bool:
    return getattr(args, "advantage_estimator", None) == "score_centering"


def resolve_score_centering_vocab_size(args: Any) -> int:
    """Cache the model's real output vocabulary, excluding Megatron padding."""
    vocab_size = getattr(args, "score_centering_vocab_size", None)
    if vocab_size is None:
        from relax.utils.misc import get_hf_config

        config = get_hf_config(args.hf_checkpoint)
        config = getattr(config, "text_config", config)
        vocab_size = int(config.vocab_size)
        args.score_centering_vocab_size = vocab_size
    if vocab_size <= 0:
        raise ValueError("Score Centering requires a positive model vocabulary size")
    return int(vocab_size)


def configure_score_centering_request(args: Any, payload: dict[str, Any]) -> str | None:
    """Request probabilities of the actual per-call sampler, without changing
    it.

    Filtered sampling uses SGLang's support IDs and sampled behavior
    probability to recover the common renormalization of the returned candidate
    logprobs.
    """
    if not score_centering_enabled(args):
        return None
    k = int(args.rollout_top_logprobs_num)
    if k <= 0:
        raise ValueError("Score Centering requires --rollout-top-logprobs-num > 0")
    temperature = float(args.rollout_temperature)
    sampling = dict(payload["sampling_params"])
    for key, default in (("temperature", temperature), ("top_p", 1.0), ("top_k", -1), ("min_p", 0.0)):
        if sampling.get(key) is None:
            sampling[key] = default
    if not math.isfinite(temperature) or temperature <= 0 or float(sampling["temperature"]) != temperature:
        raise ValueError("Score Centering requires the same positive rollout_temperature on every generation call")
    if temperature < 1e-6:
        raise ValueError(
            "SGLang maps temperature < 1e-6 to greedy sampling; returned softmax logprobs are not behavior q"
        )
    if temperature != 1.0 and os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "0").lower() in {"1", "true"}:
        raise ValueError(
            "Score Centering requires SGLANG_RETURN_ORIGINAL_LOGPROB=0: pre-temperature logprobs are not behavior q"
        )
    if getattr(args, "sglang_rl_on_policy_target", None):
        raise ValueError(
            "SGLang rl_on_policy_target returns rounded BF16 logprobs without the sampling normalization; "
            "Score Centering requires normalized behavior q"
        )
    if getattr(args, "sglang_speculative_algorithm", None):
        raise ValueError(
            "SGLang v0.5.17 does not return per-accepted-token sampling support probabilities for speculative "
            "decoding; Score Centering requires aligned behavior q"
        )
    filtered = float(sampling["top_p"]) < 1 or int(sampling["top_k"]) > 0 or float(sampling["min_p"]) > 0
    if filtered and int(sampling["top_k"]) <= 0:
        raise ValueError(
            "SGLang return_sampling_mask requires finite top_k to return behavior q for top-p/min-p filtering"
        )
    if filtered and getattr(args, "sglang_sampling_backend", None) == "ascend":
        raise ValueError("SGLang ascend sampling does not return the sampling support required to recover behavior q")
    if float(sampling["top_p"]) < 1 and getattr(args, "sglang_sampling_backend", None) != "pytorch":
        # FlashInfer switches to sequential top-k/top-p renormalization when
        # ANY request enables min-p, but SGLang reports joint-filter support.
        # Checking only this request's min_p cannot ensure truthful behavior q.
        raise ValueError(
            "Score Centering top-p filtering requires --sglang-sampling-backend pytorch: "
            "SGLang's batch-level min-p path can return incorrect behavior support with FlashInfer, "
            "even for a request with min_p=0; alternatively set top_p=1"
        )
    payload["sampling_params"] = sampling
    payload["return_logprob"] = True
    payload["top_logprobs_num"] = k
    payload.pop("sampling_logprobs_mode", None)
    if filtered:
        payload["return_sampling_mask"] = True
    return "support" if filtered else "selected"


def _decode_logprob_pairs(meta_info: Mapping[str, Any], prefix: str) -> list[tuple[float, int]]:
    values = np.frombuffer(pybase64.b64decode(meta_info[f"{prefix}_val_b64"], validate=True), dtype=np.float32)
    encoded_ids = meta_info.get(f"{prefix}_idx_b64")
    if encoded_ids is None:
        raise ValueError(f"Score Centering requires {prefix}_idx_b64 alongside encoded probabilities")
    ids = np.frombuffer(pybase64.b64decode(encoded_ids, validate=True), dtype=np.int32)
    if values.size != ids.size:
        raise ValueError(f"Score Centering {prefix} encoded values and token IDs differ in length")
    return list(zip(values.tolist(), ids.tolist(), strict=True))


def _validate_rows(
    token_ids: np.ndarray,
    sampled_logprobs: np.ndarray,
    ids: np.ndarray,
    logprobs: np.ndarray,
    active: np.ndarray,
    *,
    context: str,
    vocab_size: int | None,
) -> None:
    """Validate on the CPU before queue writes or accepting a generation
    chunk."""

    def reject(mask: np.ndarray, message: str) -> None:
        locations = np.argwhere(mask)
        if locations.size:
            raise ValueError(f"{context}, position {int(locations[0, 0])}: {message}")

    reject((ids < -1).any(-1), "rollout_topk_token_ids contains an invalid negative ID")
    if vocab_size is not None:
        reject((ids >= vocab_size).any(-1), "rollout_topk_token_ids exceeds the real vocabulary")
        reject(active & ((token_ids < 0) | (token_ids >= vocab_size)), "sampled token exceeds the real vocabulary")
    valid = ids >= 0
    reject((~valid & ~np.isneginf(logprobs)).any(-1), "candidate padding must have ID=-1 and logprob=-inf")
    reject(active & ~valid.any(-1), "rollout_topk_token_ids is empty at a trained position")
    reject(
        active & (valid & (~np.isfinite(logprobs) | (logprobs > _PROBABILITY_TOLERANCE))).any(-1),
        "rollout_topk_log_probs must be finite and <= 0 at valid candidates",
    )
    reject(
        active & (~np.isfinite(sampled_logprobs) | (sampled_logprobs > _PROBABILITY_TOLERANCE)),
        "rollout_log_probs must be finite and <= 0 at trained positions",
    )
    sorted_ids = np.sort(ids, axis=-1)
    reject(
        active & ((sorted_ids[:, 1:] >= 0) & (sorted_ids[:, 1:] == sorted_ids[:, :-1])).any(-1),
        "rollout_topk_token_ids contains duplicate candidates",
    )
    safe_logprobs = np.where(active[:, None] & valid, logprobs, -np.inf)
    head_mass = np.exp(safe_logprobs.astype(np.float64)).sum(-1)
    reject(active & (head_mass > 1 + _PROBABILITY_TOLERANCE), "rollout_topk_log_probs head probability exceeds 1")
    matches = valid & (ids == token_ids[:, None]) & active[:, None]
    reject(
        (matches & ~np.isclose(logprobs, sampled_logprobs[:, None], rtol=1e-5, atol=1e-5)).any(-1),
        "sampled and candidate log-q disagree",
    )


def parse_score_centering_response(
    meta_info: Mapping[str, Any],
    k: int,
    *,
    sampling_mode: str = "selected",
    expected_tokens: Sequence[int] | None = None,
    vocab_size: int | None = None,
) -> tuple[list[int], list[float], np.ndarray, np.ndarray]:
    """Parse same-call sampled/head q; never rescore or silently fill missing
    rows."""
    if k <= 0 or (vocab_size is not None and k > vocab_size):
        raise ValueError("Score Centering candidate count must lie in [1, real vocabulary size]")
    sampled_rows = meta_info.get("output_token_logprobs")
    if sampled_rows is None and "output_token_logprobs_val_b64" in meta_info:
        sampled_rows = _decode_logprob_pairs(meta_info, "output_token_logprobs")
    if sampled_rows is None:
        if expected_tokens is not None and len(expected_tokens) == 0:
            sampled_rows = []
        else:
            raise ValueError("Score Centering requires SGLang output_token_logprobs from the generation call")
    tokens, sampled = [], []
    for position, row in enumerate(sampled_rows):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError(f"Score Centering position {position}: malformed output_token_logprobs")
        if not isinstance(row[1], (int, np.integer)) or row[1] < 0:
            raise ValueError(f"Score Centering position {position}: invalid sampled token ID")
        tokens.append(int(row[1]))
        sampled.append(float(row[0]))
    if expected_tokens is not None and tokens != list(expected_tokens):
        raise ValueError("Score Centering output_token_logprobs token IDs do not match generated output IDs")
    n = len(tokens)
    ids = np.full((n, k), -1, dtype=np.int32)
    logprobs = np.full((n, k), -np.inf, dtype=np.float32)
    rows = meta_info.get("output_top_logprobs")
    if rows is None and "output_top_logprobs_val_b64" in meta_info:
        entries = _decode_logprob_pairs(meta_info, "output_top_logprobs")
        if len(entries) != n * k or meta_info.get("output_top_logprobs_k") != [k] * n:
            raise ValueError("Score Centering encoded output_top_logprobs must contain exactly K entries per token")
        rows = [entries[start : start + k] for start in range(0, len(entries), k)]
    if n and (rows is None or len(rows) != n):
        raise ValueError("Score Centering requires output_top_logprobs for every generated token")
    if rows is not None and len(rows) != n:
        raise ValueError("Score Centering output_top_logprobs row count differs from generated token count")
    for position, row in enumerate(rows or []):
        if row is None or len(row) != k:
            raise ValueError(
                f"Score Centering position {position}: output_top_logprobs must contain exactly K entries"
            )
        for column, entry in enumerate(row):
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise ValueError(f"Score Centering position {position}: malformed output_top_logprobs entry")
            token_id, logprob = entry[1], float(entry[0])
            if not isinstance(token_id, (int, np.integer)) or not 0 <= token_id <= np.iinfo(np.int32).max:
                raise ValueError(f"Score Centering position {position}: invalid candidate token ID")
            # Grammar masks may produce zero-probability entries in a full
            # top-K response. Preserve their slots as explicit padding.
            if math.isinf(logprob) and logprob < 0:
                continue
            ids[position, column] = token_id
            logprobs[position, column] = logprob
    _validate_rows(
        np.asarray(tokens, dtype=np.int64),
        np.asarray(sampled, dtype=np.float32),
        ids,
        logprobs,
        np.ones(n, dtype=bool),
        context="Score Centering generation before filtering",
        vocab_size=vocab_size,
    )
    if sampling_mode == "support":
        supports = meta_info.get("output_token_sampling_mask")
        probabilities = meta_info.get("output_token_sampling_logprobs")
        if n and (supports is None or probabilities is None):
            raise ValueError(
                "Score Centering requires actual behavior q in output_token_sampling_mask and "
                "output_token_sampling_logprobs; this SGLang response lacks support probabilities"
            )
        if len(supports or []) != n or len(probabilities or []) != n:
            raise ValueError("Score Centering sampling support rows must match generated token count")
        for position, (support_ids, support_logprob) in enumerate(
            zip(supports or [], probabilities or [], strict=True)
        ):
            row_ids = np.asarray(support_ids)
            if row_ids.ndim != 1 or not np.isscalar(support_logprob) or not len(row_ids):
                raise ValueError(f"Score Centering position {position}: malformed sampling support probabilities")
            if not np.issubdtype(row_ids.dtype, np.integer) or (row_ids < 0).any():
                raise ValueError(f"Score Centering position {position}: invalid sampling support IDs")
            if (row_ids > np.iinfo(np.int32).max).any() or (vocab_size is not None and (row_ids >= vocab_size).any()):
                raise ValueError(f"Score Centering position {position}: sampling support ID exceeds vocabulary")
            if len(np.unique(row_ids)) != len(row_ids):
                raise ValueError(f"Score Centering position {position}: duplicate sampling support IDs")
            support_logprob = float(support_logprob)
            if not math.isfinite(support_logprob) or support_logprob > _PROBABILITY_TOLERANCE:
                raise ValueError(f"Score Centering position {position}: invalid sampled behavior log-q")
            match = np.flatnonzero(row_ids == tokens[position])
            if len(match) != 1:
                raise ValueError(f"Score Centering position {position}: sampled token is absent from sampling support")
            if len(row_ids) == 1:
                if not math.isclose(support_logprob, 0.0, abs_tol=_PROBABILITY_TOLERANCE):
                    raise ValueError(f"Score Centering position {position}: singleton support must have probability 1")
                # argmax and topk can choose different IDs at a tied maximum.
                # The emitted singleton support, not topk's tie break, defines q.
                ids[position] = -1
                logprobs[position] = -np.inf
                ids[position, 0] = tokens[position]
                logprobs[position, 0] = support_logprob
                sampled[position] = support_logprob
                continue
            # Pure top-k/top-p/min-p truncation multiplies every retained
            # probability by the same factor, including the sampled token.
            log_normalizer_shift = support_logprob - sampled[position]
            if log_normalizer_shift < -_PROBABILITY_TOLERANCE:
                raise ValueError(f"Score Centering position {position}: filtering increased the normalization mass")
            in_support = np.isin(ids[position], row_ids) & (ids[position] >= 0)
            logprobs[position, in_support] += log_normalizer_shift
            logprobs[position, ~in_support] = -np.inf
            ids[position, ~in_support] = -1
            sampled[position] = support_logprob
            if np.isin(row_ids, ids[position]).all() and not np.isclose(
                np.exp(logprobs[position].astype(np.float64)).sum(), 1.0, rtol=1e-5, atol=1e-5
            ):
                raise ValueError(f"Score Centering position {position}: complete sampling support must sum to 1")
    elif sampling_mode != "selected":
        raise ValueError(f"Unknown Score Centering sampling mode: {sampling_mode}")
    _validate_rows(
        np.asarray(tokens, dtype=np.int64),
        np.asarray(sampled, dtype=np.float32),
        ids,
        logprobs,
        np.ones(n, dtype=bool),
        context="Score Centering generation",
        vocab_size=vocab_size,
    )
    return tokens, sampled, ids, logprobs


def append_score_centering_candidates(sample: Sample, ids: np.ndarray, logprobs: np.ndarray) -> None:
    """Append only the newly generated segment, retaining old behavior q."""
    for field, values in zip(SCORE_CENTERING_FIELDS, (ids, logprobs), strict=True):
        previous = getattr(sample, field)
        setattr(sample, field, values if previous is None else np.concatenate((previous, values), axis=0))


def validate_score_centering_sample(sample: Sample, k: int, *, vocab_size: int | None = None) -> None:
    context = f"Score Centering sample {sample.index}"

    def has_content(value: Any) -> bool:
        if value is None:
            return False
        if hasattr(value, "numel"):
            return value.numel() > 0
        if hasattr(value, "__len__"):
            return len(value) > 0
        return True

    media = sample.multimodal_inputs or {}
    if any(has_content(media.get(key)) for key in ("images", "videos", "audio", "audios")) or any(
        has_content(value) for value in (sample.multimodal_train_inputs or {}).values()
    ):
        raise ValueError(f"{context}: Score Centering currently requires text-only training inputs")
    if k <= 0 or (vocab_size is not None and k > vocab_size):
        raise ValueError(f"{context}: candidate count must lie in [1, real vocabulary size]")
    ids, logprobs = (getattr(sample, field, None) for field in SCORE_CENTERING_FIELDS)
    expected_shape = (sample.response_length, k)
    if not isinstance(ids, np.ndarray) or ids.shape != expected_shape or ids.dtype != np.int32:
        raise ValueError(f"{context}: rollout_topk_token_ids must be int32 {expected_shape}")
    if not isinstance(logprobs, np.ndarray) or logprobs.shape != expected_shape or logprobs.dtype != np.float32:
        raise ValueError(f"{context}: rollout_topk_log_probs must be float32 {expected_shape}")
    sampled = np.asarray(sample.rollout_log_probs, dtype=np.float32)
    if sampled.shape != (sample.response_length,):
        raise ValueError(f"{context}: rollout_log_probs must match response_length")
    if sample.response_length > len(sample.tokens):
        raise ValueError(f"{context}: response_length exceeds token count")
    tokens = np.asarray(sample.tokens[-sample.response_length :] if sample.response_length else [], dtype=np.int64)
    active = np.asarray(
        sample.loss_mask if sample.loss_mask is not None else np.ones(sample.response_length), dtype=bool
    )
    if active.shape != (sample.response_length,):
        raise ValueError(f"{context}: loss_mask must match response_length")
    _validate_rows(tokens, sampled, ids, logprobs, active, context=context, vocab_size=vocab_size)
