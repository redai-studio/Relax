#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Entrypoint for offline repetition diagnosis of dumped rollout data.

Thin wrapper around :func:`relax.utils.repetition_diagnose.main` so
the diagnostic can be launched the same way as other Relax entrypoints::

    python -m relax.entrypoints.diagnose_repetition <dump>/rollout_data/0.pt -o report.json
"""

from relax.utils.repetition_diagnose import main


if __name__ == "__main__":
    main()
