# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pair-atomic NLI input for the existing SFT producer and processor pool."""

import asyncio
from typing import Any

import torch

from relax.engine.sft.dataset.streaming import ProcessedSample
from relax.utils.data.streaming_dataset import IndexManager, StreamingReader


def _encode_pair(row: dict[str, Any], emb_token_id: int, max_length: int) -> dict[str, dict[str, torch.Tensor]]:
    if "query" in row and "positive" in row:
        branches = {side: row[side] for side in ("query", "positive")}
    else:
        messages = row.get("messages", [])
        if len(messages) != 4 or [item["role"] for item in messages] != ["user", "assistant"] * 2:
            raise ValueError("NLI requires exactly two user/assistant conversations per pair")
        branches = {side: {"messages": messages[start : start + 2]} for side, start in (("query", 0), ("positive", 2))}
    output = {}
    for side, branch in branches.items():
        if not isinstance(branch, dict) or set(branch) not in ({"input_ids"}, {"messages"}):
            raise ValueError("NLI branches support text messages or input_ids only; media requires a separate dataset")
        if "input_ids" in branch:
            ids = torch.as_tensor(branch["input_ids"], dtype=torch.long)
        else:
            from relax.utils.data.processor_pool import get_worker_processor

            if any(not isinstance(item.get("content"), str) for item in branch["messages"]):
                raise ValueError("NLI messages must contain text only")
            processor = get_worker_processor()
            rendered = processor.apply_chat_template(branch["messages"], tokenize=False, add_generation_prompt=False)
            encoded = processor(text=[rendered], padding=False, truncation=False, return_tensors="pt")
            ids = encoded["input_ids"][0]
        if ids.device.type != "cpu" or ids.ndim != 1 or not 2 <= ids.numel() <= max_length:
            raise ValueError("NLI branch is invalid or overlength; no pairs are filtered or truncated")
        positions = torch.nonzero(ids == emb_token_id, as_tuple=False).flatten()
        if positions.numel() == 0 or int(positions[0]) == 0:
            raise ValueError("Each NLI branch requires an <emb> token with a preceding token")
        output[side] = {"input_ids": ids.contiguous()}
    return output


class NLIPairDataset:
    """One TQ row is one pair; batch sizes and resume positions count pairs.

    Final batches repeat the current epoch's head. The public shuffle method
    translates SFT's consumed-row cursor back to this padded epoch contract.
    Sampling uses Relax's IndexManager, rather than promising HF RNG parity.
    """

    def __init__(
        self, path: str, *, emb_token_id: int, batch_size: int, max_length: int, processor_pool: Any, seed: int = 42
    ) -> None:
        self.path = path
        self.reader = StreamingReader(path)
        self.index_manager = IndexManager(len(self.reader), seed=seed)
        self.emb_token_id = emb_token_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.processor_pool = processor_pool
        if not len(self.reader) or batch_size <= 0:
            raise ValueError("NLI needs a nonempty dataset and positive pair batch size")

    @classmethod
    def from_args(cls, config: Any, *, tokenizer: Any, processor_pool: Any, pad_token_ids: Any) -> "NLIPairDataset":
        from relax.utils.training.retrieval_utils import retrieval_emb_token_id

        del pad_token_ids
        return cls(
            config.prompt_data,
            emb_token_id=retrieval_emb_token_id(tokenizer, len(tokenizer)),
            batch_size=config.global_batch_size,
            max_length=config.seq_length,
            processor_pool=processor_pool,
            seed=config.seed,
        )

    def __len__(self) -> int:
        return len(self.reader)

    def shuffle(self, epoch_id: int, position: int = 0) -> None:
        consumed = epoch_id * len(self) + position
        padded_size = (len(self) + self.batch_size - 1) // self.batch_size * self.batch_size
        epoch, offset = divmod(consumed, padded_size)
        if offset >= len(self) or offset % self.batch_size:
            raise ValueError("NLI resume must point to a complete pair-batch boundary")
        self.index_manager.shuffle(epoch)
        self.index_manager.position = offset

    async def get_batch_async(self, n: int) -> tuple[list[ProcessedSample], bool]:
        if n != self.batch_size:
            raise ValueError("NLI producer must request its configured complete pair batch")
        manager = self.index_manager
        crossed_epoch = manager.indices is not None and manager.position == len(self)
        if manager.indices is None:
            manager.shuffle(0)
        elif crossed_epoch:
            manager.shuffle(manager.current_epoch + 1)
        take = min(n, len(self) - manager.position)
        indices = manager.indices[manager.position : manager.position + take]
        indices += [manager.indices[index % len(self)] for index in range(n - take)]
        rows = [self.reader[index] for index in indices]
        if self.processor_pool is None:
            if any(not all("input_ids" in row.get(side, {}) for side in ("query", "positive")) for row in rows):
                raise ValueError("Raw NLI messages require the existing SFT processor pool")
            pairs = [_encode_pair(row, self.emb_token_id, self.max_length) for row in rows]
        else:
            loop = asyncio.get_running_loop()
            pairs = await asyncio.gather(
                *[
                    loop.run_in_executor(
                        self.processor_pool.executor, _encode_pair, row, self.emb_token_id, self.max_length
                    )
                    for row in rows
                ]
            )
        manager.position += take
        samples = []
        for index, pair in zip(indices, pairs, strict=True):
            tokens = torch.cat([pair[side]["input_ids"] for side in ("query", "positive")])
            samples.append(ProcessedSample(tokens, torch.ones_like(tokens), tokens.numel(), pair, index))
        return samples, crossed_epoch

    def stop(self) -> None:
        """SFT owns the shared processor pool's lifecycle."""
