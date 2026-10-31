# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron parameter defaults mirrored for the test side.

These fields are declared by the Megatron parsers, are not Relax parameters,
but are read directly by Relax business code. CPU unit tests do not install
Megatron and cannot take the defaults from upstream, so a mirror is kept here
and used **only to build test parameter objects**; whether the mirror matches
the real upstream is verified by
``tests/backends/megatron/test_args_compat.py::test_megatron_test_defaults_match_upstream``
(only runs when Megatron is installed).

When adding a field, its upstream default source must be confirmed at the same
time, otherwise the mirror degenerates into a second "defaults written from
memory" table.
"""

from typing import Any


MEGATRON_TEST_DEFAULTS: dict[str, Any] = {
    "context_parallel_size": 1,
    "dynamic_context_parallel": False,
    "seq_length": None,
    # The profiler fields are declared by the Megatron parser (see the
    # "--record-memory-history ... defined in Megatron" comment in arguments.py).
    # ProfilingConfig.profile_ranks defaults to [] on the pinned GPU dev parser
    # (ArgumentGroupFactory uses the factory result as the argparse default),
    # while the NPU image (docker/Dockerfile.npu, core_v0.16.1) declares [0];
    # the mirror keeps the GPU value and
    # test_megatron_test_defaults_match_upstream accepts both documented
    # variants. Tests that need the old None behaviour must pass
    # profile_ranks=None explicitly.
    "use_pytorch_profiler": False,
    "record_memory_history": False,
    "profile_ranks": [],
    "profile_step_start": 10,
    "profile_step_end": 12,
}
