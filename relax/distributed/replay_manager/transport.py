# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compatibility import for callers migrating to the storage module."""

from .storage import TQReplayStorage as TQReplayTransport


__all__ = ["TQReplayTransport"]
