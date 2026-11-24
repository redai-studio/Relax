# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native Relax SFT reader for immutable, author-text-preserving masked
data."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from relax.engine.sft.dataset.streaming import ProcessedSample, SFTStreamingDataset


def read_author_sample(row: dict, capacity: int, source_idx: int) -> ProcessedSample:
    n = row["positions"]
    if not 2 <= n <= capacity or row["token_offset"] % 4:
        raise ValueError("Invalid author SFT token boundary")
    with open(row["tokens_path"], "rb") as stream:
        stream.seek(row["token_offset"])
        token_bytes = stream.read(n * 4)
    with open(row["mask_path"], "rb") as stream:
        stream.seek(row["mask_offset"])
        mask_bytes = stream.read(n)
    if len(token_bytes) != n * 4 or len(mask_bytes) != n:
        raise OSError("Short author SFT immutable data read")
    ids = np.frombuffer(token_bytes, dtype="<u4").astype(np.int64)
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).astype(np.int64)
    if mask[0] or not np.all((mask == 0) | (mask == 1)) or int(mask.sum()) != row["assistant_loss_tokens"]:
        raise ValueError("Author SFT mask changed")
    eos = np.count_nonzero((ids == row["eos_token_id"]) & (mask == 1))
    if eos != row["supervised_assistant_EOS"]:
        raise ValueError("Author SFT supervised EOS changed")
    return ProcessedSample(torch.from_numpy(ids), torch.from_numpy(mask), n, None, source_idx)


class AuthorMaskedSFTDataset(SFTStreamingDataset):
    @classmethod
    def from_args(
        cls, config: Any, *, tokenizer: Any, processor_pool: Any, pad_token_ids: frozenset[int]
    ) -> AuthorMaskedSFTDataset:
        return cls(
            path=config.prompt_data,
            tokenizer=tokenizer,
            processor_pool=None,
            capacity=config.seq_length,
            prompt_key="author_masked",
            seed=config.seed,
            prefetch_max_cached=0,
            pad_token_ids=pad_token_ids,
            training_mode="sft",
            oversize_strategy="keep",
        )

    def _process_one(self, idx: int) -> ProcessedSample:
        return read_author_sample(self.reader[idx], self.capacity, idx)

    async def get_batch_async(self, n: int) -> tuple[list[ProcessedSample], bool]:
        # The generic async renderer parses raw chat rows. These rows are
        # already canonical masked tokens; retain the native index manager
        # and inline batch path instead of accidentally rendering them again.
        return self.get_batch(n)
