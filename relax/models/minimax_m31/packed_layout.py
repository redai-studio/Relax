# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from Miles, commit 3a7659a4, under the Apache License, Version 2.0.

from collections.abc import Sequence

import torch


class PackedBlockLayout:
    """One block grid per document, using boundaries supplied by the batch
    producer."""

    def __init__(self, bounds: Sequence[int], block_size: int, device: torch.device) -> None:
        lengths = [end - start for start, end in zip(bounds[:-1], bounds[1:], strict=True)]
        blocks = [(length + block_size - 1) // block_size for length in lengths]
        self.num_blocks = sum(blocks)
        total = bounds[-1]
        document_ids = torch.arange(len(lengths), device=device)
        token_counts = torch.tensor(lengths, device=device)
        block_counts = torch.tensor(blocks, device=device)
        token_document = torch.repeat_interleave(document_ids, token_counts, output_size=total)
        block_document = torch.repeat_interleave(document_ids, block_counts, output_size=self.num_blocks)

        endpoints = torch.tensor(bounds, device=device)
        block_starts = block_counts.cumsum(0) - block_counts
        self.token_start = endpoints[token_document]
        self.token_block_start = block_starts[token_document]
        self.positions = torch.arange(total, device=device) - self.token_start
        self.token_block = self.token_block_start + self.positions // block_size
        local_block = torch.arange(self.num_blocks, device=device) - block_starts[block_document]
        self.block_first_token = endpoints[block_document] + local_block * block_size
        self.block_last_token = torch.minimum(self.block_first_token + block_size, endpoints[block_document + 1]) - 1
