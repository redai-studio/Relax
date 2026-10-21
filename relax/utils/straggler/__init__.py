# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Always-on, low-overhead straggler (slow rank) analysis for the Megatron
backend; off unless ``RELAX_STRAGGLER_PROFILER`` is set.

Data flow: ``timers`` (Megatron call sites) -> ``collector`` (per rank, one
Gloo gather per window) -> ``worker`` thread on the primary rank ->
``detector`` -> ``reporter``. ``runtime`` binds it to torch / Megatron.
"""

from relax.utils.straggler.runtime import StragglerProfiler, install_straggler_profiler, straggler_timers


__all__ = [
    "StragglerProfiler",
    "install_straggler_profiler",
    "straggler_timers",
]
