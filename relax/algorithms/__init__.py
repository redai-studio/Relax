# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Algorithm registry: names, capabilities and implementations in one place.

Nothing in this package may import ``megatron``, ``ray``, ``transfer_queue``,
``tensordict``, ``relax.components`` or ``relax.backends`` at module level. The
registry is imported by argument parsing and by both worker processes, so a
heavy import here would pull the whole training stack into ``--help`` and into
the CPU-only test runner.
"""

from relax.algorithms.spec import (
    ALGORITHM_SPECS,
    AlgorithmSpec,
    algorithm_needs_critic,
    get_algorithm,
    list_algorithm_names,
)


__all__ = [
    "ALGORITHM_SPECS",
    "AlgorithmSpec",
    "algorithm_needs_critic",
    "get_algorithm",
    "list_algorithm_names",
]
