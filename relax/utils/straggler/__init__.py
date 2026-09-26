# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Always-on, low-overhead straggler (slow rank) analysis for the Megatron
backend.

See ``relax/utils/straggler/collector.py`` for the data flow. Enabled with the
``RELAX_STRAGGLER_PROFILER`` environment variable; off by default. Only the
entry points used by the Megatron backend are re-exported here; the detector
and timer internals live in their own modules.
"""

from relax.utils.straggler.collector import install_straggler_collector, straggler_timers
from relax.utils.straggler.reporter import report_straggler_window


__all__ = [
    "install_straggler_collector",
    "report_straggler_window",
    "straggler_timers",
]
