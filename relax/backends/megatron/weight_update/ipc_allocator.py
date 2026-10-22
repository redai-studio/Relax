# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Allocate only CUDA IPC transport buffers outside expandable segments."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable

import torch


_EXPANDABLE = re.compile(r"(?<![\w])expandable_segments\s*:\s*(True|False|true|false)(?=\s*(?:,|$))")


class IpcBucketAllocator:
    """Capture allocator state once per update, scope each bucket allocation.

    CUDA allocator configuration is process-global. Call only from the
    serialized actor weight-update path, with no concurrent allocator-
    configuration writers or concurrent allocations that require expandable
    segments. A newer Torch getter avoids snapshots when the last config
    explicitly names expandable_segments. Older Torch or an ambiguous partial
    config needs one allocator snapshot per update, never one per parameter
    bucket.
    """

    def __init__(self, device: torch.device) -> None:
        self._enabled = False
        self._depth = 0
        self._getter: Callable[[], str] | None = None
        self._setter: Callable[[str], None] | None = None
        self._settings = ""
        self._disabled_settings = ""
        self._restore_effective_settings: str | None = None
        # ROCm also calls its device type 'cuda'; leave it and other backends alone.
        if device.type != "cuda" or torch.version.hip is not None:
            return
        if torch.cuda.memory.get_allocator_backend() != "native":
            return
        self._getter = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
        self._setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
        if self._setter is None:
            self._setter = getattr(torch.cuda.memory, "_set_allocator_settings", None)
        if self._setter is None:
            raise RuntimeError("CUDA IPC allocation requires an allocator settings setter")

        settings = self._getter() if self._getter is not None else None
        matches = list(_EXPANDABLE.finditer(settings)) if isinstance(settings, str) else []
        if matches:
            enabled = matches[-1].group(1).lower() == "true"
        else:
            # A partial setter can leave expandable_segments active even when
            # omitted from the last settings string. Never infer it from env.
            snapshot = torch.cuda.memory._snapshot().get("allocator_settings", {})
            if settings is None:
                settings = snapshot.get("PYTORCH_CUDA_ALLOC_CONF")
            enabled = snapshot.get("expandable_segments")
            if not isinstance(settings, str) or not isinstance(enabled, bool):
                raise RuntimeError("Cannot determine current CUDA allocator configuration for IPC")
        if not isinstance(settings, str):
            raise RuntimeError("CUDA allocator settings getter did not return a string")
        self._settings = settings
        self._enabled = enabled
        if not enabled:
            return
        if _EXPANDABLE.search(settings):
            self._disabled_settings = _EXPANDABLE.sub("expandable_segments:False", settings)
        else:
            prefix = settings.rstrip()
            separator = "," if prefix and not prefix.endswith(",") else ""
            self._disabled_settings = prefix + separator + "expandable_segments:False"
            # Reapply the effective flag, then restore the exact last string.
            # Parsing a string without this flag preserves its effective value.
            self._restore_effective_settings = prefix + separator + "expandable_segments:True"

    @contextmanager
    def allocation(self) -> Iterator[None]:
        """Restore allocator settings before IPC serialization."""
        expected = self._disabled_settings if self._depth and self._enabled else self._settings
        if self._getter is not None and self._getter() != expected:
            raise RuntimeError("CUDA allocator configuration changed during a weight update")
        if self._depth or not self._enabled:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        assert self._setter is not None
        try:
            # Preserve every option; parsing a short override resets some knobs.
            self._setter(self._disabled_settings)
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
        finally:
            if self._restore_effective_settings is not None:
                self._setter(self._restore_effective_settings)
            self._setter(self._settings)
