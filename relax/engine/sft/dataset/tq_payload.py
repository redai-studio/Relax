# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Lossless SimpleStorage transport for short, variable-length text rows."""

from typing import Any

import torch
from tensordict import TensorDict


_SEQUENCE_FIELDS = ("tokens", "loss_masks")
_TEXT_FIELDS = {*_SEQUENCE_FIELDS, "total_lengths", "response_lengths", "dataset_ids"}


def restore_sft_text_rows(fields: dict[str, Any]) -> dict[str, Any]:
    """Restore variable-length data in the storage unit after transfer."""
    lengths = fields["total_lengths"].tolist()
    for name in _SEQUENCE_FIELDS:
        fields[name] = [row[:length] for row, length in zip(fields[name].unbind(0), lengths, strict=True)]
    return fields


def compact_sft_text_payload(batch: dict[str, Any]) -> TensorDict | None:
    """Return a dense payload when the padding overhead is bounded."""
    if batch.keys() - _TEXT_FIELDS:
        return None

    lengths = batch["total_lengths"]
    total = sum(lengths)
    padded = max(lengths) * len(lengths)
    if total <= 0 or padded > 8 * total:
        return None
    # Bound the memory cost of keeping the padded storage behind the row views.
    bytes_per_token = sum(batch[name][0].element_size() for name in _SEQUENCE_FIELDS)
    if (padded - total) * bytes_per_token > 32 * 1024**2:
        return None
    for name in _SEQUENCE_FIELDS:
        rows = batch[name]
        if any(
            row.ndim != 1 or len(row) != length or row.dtype != rows[0].dtype or row.device.type != "cpu"
            for row, length in zip(rows, lengths, strict=True)
        ):
            return None

    from relax.utils.utils import dict_to_tensordict

    metadata = {name: value for name, value in batch.items() if name not in _SEQUENCE_FIELDS}
    compact = dict_to_tensordict(metadata, batch_size=len(lengths))
    for name in _SEQUENCE_FIELDS:
        compact[name] = torch.nn.utils.rnn.pad_sequence(batch[name], batch_first=True)
    return compact
