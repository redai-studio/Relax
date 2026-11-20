# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optional CPU launch ranges; GPU durations come from profiler
correlations."""

import os
from contextlib import AbstractContextManager, nullcontext


def profile_range(name: str) -> AbstractContextManager:
    if os.environ.get("RELAX_QWEN38_RECORD_FUNCTION_TAGS", "0") != "1":
        return nullcontext()
    import torch

    if not torch.autograd._profiler_enabled():
        return nullcontext()
    return torch.profiler.record_function("QSA/" + name)
