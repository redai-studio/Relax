# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline trajectory replay (Task 34) — versioned, self-verifying bundles.

This package contains only additive, offline code. It must never import Ray
Serve, SGLang, Megatron runtime or start a distributed process group so that a
replay bundle can be inspected, validated and replayed on a plain CPU host.
"""

from relax.utils.replay.schema import FORMAT_MAJOR, FORMAT_VERSION


__all__ = ["FORMAT_MAJOR", "FORMAT_VERSION"]
