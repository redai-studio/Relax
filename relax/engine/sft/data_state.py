# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Persist consumed batch boundaries independently of producer prefetch
progress."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_DATA_OPTIONS = (
    "prompt_data",
    "seed",
    "global_batch_size",
    "eval_size",
    "input_key",
    "multimodal_keys",
    "conversation_key_map",
    "metadata_key",
    "tool_key",
    "system_prompt",
    "apply_chat_template",
    "apply_chat_template_kwargs",
    "max_tokens_per_gpu",
    "seq_length",
    "hf_checkpoint",
    "tokenizer_path",
    "tokenizer_model",
    "image_min_token_num",
    "image_max_token_num",
    "sft_oversize_strategy",
    "sft_invalid_multimodal_strategy",
    "sft_training_mode",
    "sft_cpt_template",
    "sft_loss_last_turn_only",
    "sft_ignore_empty_think",
)


def _signature(config: Any, size: int) -> str:
    value = {key: getattr(config, key, None) for key in _DATA_OPTIONS}
    value["train_size"] = size
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _state_path(root: str, step: int) -> Path:
    return Path(root) / "sft_data_state" / f"{step:07d}.json"


def restore_data_state(config: Any, dataset: Any, step: int, size: int) -> None:
    """Select the exact checkpoint boundary; future prefetched states are
    ignored."""
    root = getattr(config, "load", None)
    path = _state_path(root, step) if root and step else None
    if path is not None and path.exists():
        state = json.loads(path.read_text())
        if (
            state.get("version") != 1
            or state.get("next_step") != step
            or state.get("signature") != _signature(config, size)
        ):
            raise ValueError(f"SFT data checkpoint does not match resumed data configuration: {path}")
        epoch, position = state["epoch"], state["position"]
        if type(epoch) is not int or type(position) is not int or epoch < 0 or not 0 <= position <= size:
            raise ValueError(f"Invalid SFT data cursor: {path}")
        dataset.index_manager.reset(position=position, epoch_id=epoch)
        dataset.shuffle(epoch, position=position)
        logger.info("Restored SFT data boundary step=%s epoch=%s position=%s", step, epoch, position)
        return
    dataset.index_manager.reset(position=0, epoch_id=0)
    dataset.shuffle(0)
    if not step:
        return
    # Old checkpoints contain no raw cursor. Replaying accepted batches is slower
    # but preserves filtering semantics; step * GBS is incorrect after a skip.
    logger.warning("No SFT data cursor at step=%s; replaying %s accepted batches to restore position", step, step)
    for replay_step in range(step):
        samples, _ = dataset.get_batch(config.global_batch_size)
        if len(samples) != config.global_batch_size:
            raise RuntimeError(f"Cannot replay SFT batch {replay_step}: incomplete accepted batch")
    logger.info(
        "Replayed SFT data boundary step=%s epoch=%s position=%s",
        step,
        dataset.index_manager.current_epoch,
        dataset.index_manager.position,
    )


def save_data_state(config: Any, dataset: Any, next_step: int, size: int, samples: list[Any]) -> None:
    """Write before publishing a batch, so a model save always has its
    boundary."""
    root = getattr(config, "save", None)
    if not root:
        return
    ids_hash = hashlib.sha256(json.dumps([sample.source_idx for sample in samples]).encode()).hexdigest()
    state = {
        "version": 1,
        "next_step": next_step,
        "signature": _signature(config, size),
        "epoch": dataset.index_manager.current_epoch,
        "position": dataset.index_manager.position,
        "source_ids_sha256": ids_hash,
    }
    path = _state_path(root, next_step)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(state, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info(
        "Saved SFT data boundary next_step=%s epoch=%s position=%s source_ids_sha256=%s",
        next_step,
        state["epoch"],
        state["position"],
        ids_hash,
    )
