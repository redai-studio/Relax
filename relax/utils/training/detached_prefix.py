# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Worker-local detached prefix state ownership.

Callers supply authoritative token/position keys and all attention, multimodal,
adapter and model identity in semantics. State may contain KV, convolution and
recurrent tensors. A suffix always receives an independent detached copy.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class PrefixKey:
    trajectory: str
    semantics: tuple
    tokens: tuple[int, ...]
    positions: tuple


class DetachedPrefixCache:
    """Bounded cache of flat tensor tuples, owned by exactly one model/worker.

    The actor runtime owns this cache for one gradient accumulation/update only
    and clears it before optimizer.step, so later weight loads and model swaps
    cannot access stale entries. Other callers must explicitly bump_version.
    """

    def __init__(self, mode: str = "baseline", max_bytes: int = 0) -> None:
        if mode not in ("baseline", "detached_prefix", "detached_prefix_reuse"):
            raise ValueError(f"Unknown prefix mode: {mode}")
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.mode = mode
        self.max_bytes = max_bytes
        self.version = 0
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.builds = self.extensions = self.computed_tokens = 0

    def bump_version(self) -> None:
        self.version += 1
        self.entries.clear()
        self.bytes = 0

    @staticmethod
    def copy_state(state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(t.detach().clone() for t in state)

    def prepare(
        self,
        key: PrefixKey,
        build: Callable[[int, tuple[torch.Tensor, ...] | None], tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, ...]:
        if self.mode == "baseline":
            raise RuntimeError("Baseline must execute the original full forward")
        if len(key.tokens) != len(key.positions):
            raise ValueError("One authoritative position key is required per token")
        matched = None
        if self.mode == "detached_prefix_reuse":
            for candidate in self.entries:
                n = len(candidate.tokens)
                if (
                    candidate.trajectory == key.trajectory
                    and candidate.semantics == key.semantics
                    and n <= len(key.tokens)
                    and candidate.tokens == key.tokens[:n]
                    and candidate.positions == key.positions[:n]
                    and (matched is None or n > len(matched.tokens))
                ):
                    matched = candidate
        offset = 0
        state = None
        if matched is not None:
            offset = len(matched.tokens)
            state = self.copy_state(self.entries[matched])
            self.entries.move_to_end(matched)
            self.hits += 1
        if matched is None or offset < len(key.tokens):
            with torch.no_grad():
                state = self.copy_state(build(offset, state))
            self.computed_tokens += len(key.tokens) - offset
            self.builds += int(matched is None)
            self.extensions += int(matched is not None)
        size = sum(t.numel() * t.element_size() for t in state)
        if self.mode == "detached_prefix_reuse" and size <= self.max_bytes:
            if key in self.entries:
                old = self.entries.pop(key)
                self.bytes -= sum(t.numel() * t.element_size() for t in old)
            while self.entries and self.bytes + size > self.max_bytes:
                _, old = self.entries.popitem(last=False)
                self.bytes -= sum(t.numel() * t.element_size() for t in old)
            self.entries[key] = self.copy_state(state)
            self.bytes += size
        return self.copy_state(state)
