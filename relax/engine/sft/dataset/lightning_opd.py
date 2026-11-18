# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Read fixed student trajectories with response-aligned teacher log
probabilities."""

import asyncio
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence

import torch

from relax.engine.sft.dataset.streaming import ProcessedSample
from relax.utils.data.data_utils import resolve_path_plan
from relax.utils.data.streaming_dataset import IndexManager


class LightningOPDDataset:
    def __init__(
        self,
        path: str | list[str],
        *,
        tokenizer=None,
        capacity: int | None = None,
        prompt_key: str = "prompt",
        metadata_key: str = "metadata",
        seed: int = 42,
    ) -> None:
        import pyarrow.parquet as pq

        paths, row_slice = resolve_path_plan(path)
        self.tokenizer = tokenizer
        self.capacity = capacity
        self.metadata_key = metadata_key
        self.groups = []
        self.offsets = [0]
        self._cached_group = None
        for filename in paths:
            file = pq.ParquetFile(filename)
            key = "prompt" if prompt_key == "input" and prompt_key not in file.schema_arrow.names else prompt_key
            if key not in file.schema_arrow.names or metadata_key not in file.schema_arrow.names:
                raise ValueError(f"{filename}: expected Lightning-OPD columns {key!r} and {metadata_key!r}")
            file_row = 0
            for group in range(file.num_row_groups):
                n = file.metadata.row_group(group).num_rows
                self.groups.append((filename, file, group, file_row, key))
                self.offsets.append(self.offsets[-1] + n)
                file_row += n
        self.rows = range(self.offsets[-1])[row_slice or slice(None)]
        if not self.rows:
            raise ValueError(f"Empty Lightning-OPD dataset: {path}")
        self.index_manager = IndexManager(len(self), seed=seed)

    def __len__(self) -> int:
        return len(self.rows)

    def shuffle(self, epoch_id: int, position: int = 0) -> None:
        self.index_manager.reset(position=position, epoch_id=epoch_id)

    def get_batch(self, n: int) -> tuple[list[ProcessedSample], bool]:
        indices, crossed = self.index_manager.get_next_indices(n)
        return self.get_batch_by_indices(indices), crossed

    async def get_batch_async(self, n: int) -> tuple[list[ProcessedSample], bool]:
        return await asyncio.to_thread(self.get_batch, n)

    def get_batch_by_indices(self, indices: Sequence[int]) -> list[ProcessedSample]:
        # Keep at most one Arrow row group resident across updates.
        grouped = defaultdict(list)
        for idx in indices:
            grouped[bisect_right(self.offsets, self.rows[idx]) - 1].append(idx)
        samples = {}
        for g, group_indices in grouped.items():
            filename, file, group, file_row, key = self.groups[g]
            cached = self._cached_group
            if cached is None or cached[0] != g:
                cached = (g, file.read_row_group(group, columns=[key, self.metadata_key]))
                self._cached_group = cached
            table = cached[1]
            for idx in group_indices:
                offset = self.rows[idx] - self.offsets[g]
                row = table.slice(offset, 1).to_pylist()[0]
                try:
                    samples[idx] = self._sample(idx, row[key], row[self.metadata_key])
                except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as exc:
                    raise ValueError(f"{filename} row {file_row + offset}: {exc}") from exc
        return [samples[i] for i in indices]

    def _sample(self, idx: int, prompt: str, metadata: dict) -> ProcessedSample:
        if not isinstance(prompt, str) or not isinstance(metadata, dict):
            raise ValueError("expected a rendered prompt string and a metadata struct")
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response = metadata["response_tokens"]
        if not prompt_ids or not response or any(type(t) is not int for t in response):
            raise ValueError("expected a nonempty prompt and integer response token IDs")
        if (saved := metadata.get("prompt_ids")) is not None and saved != prompt_ids:
            raise ValueError("cached prompt_ids differ from the student tokenizer encoding")
        tokens = torch.tensor(prompt_ids + response, dtype=torch.long)
        if tokens.min() < 0 or tokens.max() >= len(self.tokenizer):
            raise ValueError("token ID outside the student vocabulary")
        if self.capacity is not None and tokens.numel() > self.capacity:
            raise ValueError(f"sequence length {tokens.numel()} exceeds training capacity {self.capacity}")
        mask = torch.as_tensor(metadata["loss_mask"])
        teacher = torch.tensor(metadata["teacher_log_probs"], dtype=torch.float32)
        if mask.shape != (len(response),) or teacher.shape != mask.shape:
            raise ValueError("response_tokens, loss_mask and teacher_log_probs must be aligned vectors")
        if not ((mask == 0) | (mask == 1)).all() or not mask.any():
            raise ValueError("loss_mask must be binary with at least one trainable response token")
        if not torch.isfinite(teacher).all() or (teacher > 0).any():
            raise ValueError("teacher_log_probs must contain finite log probabilities")
        return ProcessedSample(
            tokens=tokens,
            loss_mask=mask.int(),
            total_length=tokens.numel(),
            multimodal_train_inputs=None,
            source_idx=idx,
            teacher_log_probs=teacher,
        )
