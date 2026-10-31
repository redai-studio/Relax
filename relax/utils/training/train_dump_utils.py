# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

from relax.utils.logging_utils import get_logger
from relax.utils.multimodal.stats import get_sample_multimodal_stats


logger = get_logger(__name__)


def _decode_tokens_to_chars(tokens: List[int], tokenizer) -> List[str]:
    """Decode each token to its corresponding string representation."""
    if tokenizer is None or not tokens:
        return []
    try:
        decoded_chars = []
        for token_id in tokens:
            try:
                char = tokenizer.decode([token_id], skip_special_tokens=False)
                decoded_chars.append(char)
            except Exception:
                decoded_chars.append(f"<{token_id}>")
        return decoded_chars
    except Exception as e:
        logger.warning(f"Failed to decode tokens: {e}")
        return []


def _tensor_to_list(tensor_or_list):
    """Convert tensor to list if needed."""
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.tolist()
    return tensor_or_list


def _reorganize_train_data_by_sample(rollout_data: Dict[str, Any], tokenizer=None) -> List[Dict[str, Any]]:
    """Reorganize train data from field-oriented to sample-oriented format.

    Args:
        rollout_data: Dict with keys like 'tokens', 'rewards', etc., each containing lists of values
        tokenizer: Optional tokenizer for decoding tokens

    Returns:
        List of sample dicts, each containing all fields for that sample
    """
    # Determine the number of samples from any available field
    num_samples = 0
    for key, value in rollout_data.items():
        if isinstance(value, (list, tuple)) and len(value) > 0:
            num_samples = len(value)
            break

    if num_samples == 0:
        return []

    # Use rollout_data keys for per-sample fields to keep extensibility
    per_sample_fields = list(rollout_data.keys())
    logger.info(
        "Reorganize train data with fields: %s",
        ", ".join(per_sample_fields) if per_sample_fields else "<empty>",
    )

    samples = []
    for i in range(num_samples):
        sample = {"sample_index": i}

        for field in per_sample_fields:
            if field in rollout_data:
                value = rollout_data[field]
                if isinstance(value, (list, tuple)) and i < len(value):
                    sample[field] = _tensor_to_list(value[i])
                elif isinstance(value, (list, tuple)):
                    sample[field] = None

        # Add decoded_tokens if tokens are available
        if "tokens" in sample and sample["tokens"] is not None:
            tokens = sample["tokens"]
            if tokenizer:
                sample["decoded_tokens"] = _decode_tokens_to_chars(tokens, tokenizer)
            else:
                sample["decoded_tokens"] = []

        samples.append(sample)

    return samples


def save_debug_train_data(args, *, rollout_id, rollout_data, tokenizer=None):
    """Save debug train data reorganized by sample with tokenizer decoding.

    Args:
        args: Arguments containing save_debug_train_data path template
        rollout_id: The rollout ID
        rollout_data: Dict with training data organized by field
        tokenizer: Optional tokenizer for decoding tokens. If provided, will be used
                  to decode tokens to readable strings. Pass from caller context
                  (e.g., Actor.tokenizer) for efficiency.
    """
    if (path_template := args.save_debug_train_data) is not None:
        rank = torch.distributed.get_rank()
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Reorganize data by sample
        samples = _reorganize_train_data_by_sample(rollout_data, tokenizer)

        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                samples=samples,
            ),
            path,
        )


def save_debug_rollout_data(args, data, rollout_id: int, evaluation: bool, tokenizer=None) -> None:
    """Save debug rollout data with tokenizer decoding support.

    Args:
        args: Arguments containing save_debug_rollout_data path template
        data: Either a list of Sample objects, or a dict (for evaluation data)
        rollout_id: The rollout ID
        evaluation: Whether this is evaluation data
        tokenizer: Optional tokenizer for decoding tokens. If provided, will be used
                  to decode tokens to readable strings. Pass from caller context
                  (e.g., GenerateState.tokenizer) for efficiency.
    """
    if (path_template := args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
        logger.info(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        if evaluation:
            # Evaluation data is a dict with dataset_name -> info structure
            samples_list = [sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
        else:
            # Regular rollout data is a list of Sample objects
            samples_list = [sample.to_dict() for sample in data]

        # Add decoded_tokens field for each sample if tokenizer is provided
        for sample_dict in samples_list:
            tokens = sample_dict.get("tokens", [])
            if tokens and tokenizer:
                sample_dict["decoded_tokens"] = _decode_tokens_to_chars(tokens, tokenizer)
            else:
                sample_dict["decoded_tokens"] = []

        dump_data = dict(samples=samples_list)
        torch.save(dict(rollout_id=rollout_id, **dump_data), path)


def _summarize_multimodal_inputs(mm_inputs: dict) -> dict:
    """Summarize raw multimodal inputs (images, videos, audio) for logging."""
    summary = {}
    for key, value in mm_inputs.items():
        if isinstance(value, list):
            items = []
            for item in value:
                try:
                    if isinstance(item, Image.Image):
                        items.append(str(item))
                        continue
                except ImportError:
                    pass
                items.append(str(type(item).__name__))
            summary[key] = items if items else len(value)
        else:
            summary[key] = str(type(value).__name__)
    return summary


def _summarize_multimodal_train_inputs(mm_train_inputs: dict) -> dict:
    """Summarize processed multimodal train inputs (tensors) for logging."""
    import torch

    summary = {}
    for key, value in mm_train_inputs.items():
        if isinstance(value, torch.Tensor):
            summary[key] = f"Tensor(shape={list(value.shape)}, dtype={value.dtype})"
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            summary[key] = [f"Tensor(shape={list(t.shape)}, dtype={t.dtype})" for t in value]
        else:
            summary[key] = str(type(value).__name__)
    return summary


def _sample_to_summary_record(sample, rollout_id: int, idx: int, dataset_name: str | None = None) -> dict:
    total_length = len(sample.tokens) if sample.tokens else 0
    response_length = sample.response_length
    prompt_length = max(total_length - response_length, 0)
    multimodal_stats = get_sample_multimodal_stats(sample)
    metadata = sample.metadata or {}
    record = {
        "rollout_id": rollout_id,
        "sample_index": idx,
        "prompt": sample.prompt,
        "response": sample.response,
        "reward": sample.reward,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "total_length": total_length,
        "prompt_token_count": prompt_length,
        "response_token_count": response_length,
        "total_token_count": total_length,
        "image_count": multimodal_stats["image_count"],
        "image_token_count": multimodal_stats["image_token_count"],
        "multimodal_token_count": multimodal_stats["multimodal_token_count"],
        "agent_turns": metadata.get("rollout_turns", 1),
        "status": sample.status.value if hasattr(sample.status, "value") else str(sample.status),
        "group_index": sample.group_index,
    }
    if sample.label is not None:
        record["label"] = sample.label
    if sample.multimodal_inputs is not None:
        record["multimodal_inputs"] = _summarize_multimodal_inputs(sample.multimodal_inputs)
    if sample.multimodal_train_inputs is not None:
        record["multimodal_train_inputs"] = _summarize_multimodal_train_inputs(sample.multimodal_train_inputs)
    if dataset_name is not None:
        record["dataset"] = dataset_name
    return record


def _write_summary_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_rollout_result_jsonl(args, rollout_id: int, samples: list) -> None:
    """Save a lightweight per-step rollout result as a JSONL file.

    Always-on: writes one JSONL per rollout step with prompt, response,
    reward and sequence length for every sample.

    Args:
        args: Arguments containing ``rollout_result_dir``.
        rollout_id: The rollout ID (used as step number in the filename).
        samples: A list of :class:`Sample` objects from the rollout batch.
    """
    result_dir = args.rollout_result_dir
    if result_dir is None:
        return

    path = Path(result_dir) / "train" / f"{rollout_id}.jsonl"
    records = [_sample_to_summary_record(s, rollout_id, i) for i, s in enumerate(samples)]

    try:
        _write_summary_jsonl(path, records)
        logger.info(f"Saved rollout result ({len(records)} samples) to {path}")
    except Exception as e:
        logger.warning(f"Failed to save rollout result to {path}: {e}")


def save_eval_summary_jsonl(args, rollout_id: int, data: dict) -> None:
    """Save a lightweight per-step eval summary as a JSONL file.

    Similar to :func:`save_rollout_result_jsonl` but for evaluation rollouts.
    Each record includes an extra ``dataset`` field indicating which eval
    dataset the sample belongs to.

    Args:
        args: Arguments containing ``rollout_result_dir``.
        rollout_id: The rollout ID.
        data: Eval data dict — ``{dataset_name: {"samples": list[Sample], ...}}``.
    """
    result_dir = args.rollout_result_dir
    if result_dir is None:
        return

    eval_dir = Path(result_dir) / "eval"
    path = eval_dir / f"{rollout_id}.jsonl"

    records = []
    for dataset_name, info in data.items():
        samples = info.get("samples")
        if not samples:
            continue
        for i, s in enumerate(samples):
            records.append(_sample_to_summary_record(s, rollout_id, i, dataset_name=dataset_name))

    if not records:
        return

    try:
        _write_summary_jsonl(path, records)
        logger.info(f"Saved eval summary ({len(records)} samples) to {path}")
    except Exception as e:
        logger.warning(f"Failed to save eval summary to {path}: {e}")
