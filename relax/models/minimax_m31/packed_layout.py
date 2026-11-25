# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (Miles).
# Copyright 2025 Zhipu AI (Slime).
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax from Miles (https://github.com/radixark/miles), commit 3a7659a4.
# Miles includes software developed from Slime (https://github.com/THUDM/slime).

from collections.abc import Sequence

import torch


class PackedBlockLayout:
    """One block grid per document, using boundaries supplied by the batch
    producer."""

    def __init__(
        self, bounds: Sequence[int], block_size: int, device: torch.device, query_ids: torch.Tensor | None = None
    ) -> None:
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
        if query_ids is not None:
            token_document = token_document[query_ids]
        self.token_start = endpoints[token_document]
        self.token_block_start = block_starts[token_document]
        self.query_ids = torch.arange(total, device=device) if query_ids is None else query_ids
        self.positions = self.query_ids - self.token_start
        local_block = torch.arange(self.num_blocks, device=device) - block_starts[block_document]
        self.block_first_token = endpoints[block_document] + local_block * block_size
        self.block_last_token = torch.minimum(self.block_first_token + block_size, endpoints[block_document + 1]) - 1
