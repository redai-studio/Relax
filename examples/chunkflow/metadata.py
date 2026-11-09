# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from collections.abc import Sequence

import torch


def int32_metadata(values: Sequence[int], device: torch.device) -> torch.Tensor:
    """Copy immutable host metadata on the consumer's current stream."""
    # Immutable source and current-stream ordering keep the async copy safe.
    return torch.tensor(values, dtype=torch.int32, device="cpu").to(device, non_blocking=True)
