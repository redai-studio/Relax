# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optional CPU launch ranges; GPU durations come from profiler
correlations."""

from contextlib import AbstractContextManager, nullcontext

from relax.utils.env import Envs


def profile_range(name: str) -> AbstractContextManager:
    if not Envs.RELAX_QWEN38_RECORD_FUNCTION_TAGS:
        return nullcontext()
    import torch

    if not torch.autograd._profiler_enabled():
        return nullcontext()
    return torch.profiler.record_function("QSA/" + name)
