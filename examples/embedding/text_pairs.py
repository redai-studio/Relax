# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Small text-pair dataset for the embedding SFT example."""

import json
from pathlib import Path
from typing import Any

import torch

from relax.engine.sft.dataset.streaming import ProcessedSample
from relax.utils.data.streaming_dataset import IndexManager


class TextPairDataset:
    def __init__(self, pairs: list[tuple[torch.Tensor, torch.Tensor]], seed: int) -> None:
        self.pairs = pairs
        self.index_manager = IndexManager(len(pairs), seed=seed)

    @classmethod
    def from_args(cls, config: Any, *, tokenizer: Any, processor_pool: Any, pad_token_ids: Any) -> "TextPairDataset":
        pairs = []
        with Path(config.prompt_data).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                texts = [row[key] for key in ("query", "positive")]
                if not all(isinstance(text, str) and text.strip() for text in texts):
                    raise ValueError(f"Expected nonempty query and positive strings at line {line_number}")
                query, positive = tokenizer(texts, add_special_tokens=False)["input_ids"]
                if not all(0 < len(ids) <= config.seq_length for ids in (query, positive)):
                    raise ValueError(f"Text pair at line {line_number} exceeds the configured sequence length")
                pairs.append((torch.tensor(query, dtype=torch.long), torch.tensor(positive, dtype=torch.long)))
        if not pairs or len(pairs) % config.global_batch_size:
            raise ValueError("The example requires a nonempty dataset with a whole number of global pair batches")
        return cls(pairs, config.seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def shuffle(self, epoch_id: int, position: int = 0) -> None:
        self.index_manager.shuffle(epoch_id)
        self.index_manager.position = position

    def get_batch(self, n: int) -> tuple[list[ProcessedSample], bool]:
        indices, crossed_epoch = self.index_manager.get_next_indices(n)
        samples = []
        for index in indices:
            query, positive = self.pairs[index]
            tokens = torch.cat((query, positive))
            pair = {"query": {"input_ids": query}, "positive": {"input_ids": positive}}
            samples.append(ProcessedSample(tokens, torch.ones_like(tokens), tokens.numel(), pair, index))
        return samples, crossed_epoch

    async def get_batch_async(self, n: int) -> tuple[list[ProcessedSample], bool]:
        return self.get_batch(n)
