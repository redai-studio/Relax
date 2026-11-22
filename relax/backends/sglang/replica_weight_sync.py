# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch


def replica_parameters(parameters: dict) -> list[str]:
    """Aliases and overlapping native views retain their original transfer."""
    spans = {
        name: (value.data_ptr(), value.data_ptr() + value.numel() * value.element_size())
        for name, value in parameters.items()
    }
    return sorted(
        name
        for name, value in parameters.items()
        if value.is_cuda
        and value.dtype == torch.bfloat16
        and value.is_contiguous()
        and not any(
            other != name and max(begin, spans[name][0]) < min(end, spans[name][1])
            for other, (begin, end) in spans.items()
        )
    )
