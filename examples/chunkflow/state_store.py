# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ChunkState:
    tensors: dict[str, torch.Tensor]

    def _leaf(self, name: str, live: torch.Tensor, accumulate_fp32: bool = False) -> torch.Tensor:
        if name not in self.tensors:
            leaf = live.detach().requires_grad_(True)
            if accumulate_fp32:
                leaf._chunkflow_grad_sum = None
                leaf.register_post_accumulate_grad_hook(_accumulate_gradient)
            self.tensors[name] = leaf
        return self.tensors[name]

    def link_tensor(self, name: str, live: torch.Tensor, *, accumulate_fp32: bool = False) -> None:
        """Join future chunks' gradients to a state already used by this
        output."""
        leaf = self._leaf(name, live, accumulate_fp32)

        def hook(grad: torch.Tensor) -> torch.Tensor:
            cached = _cached_gradient(leaf)
            return grad if cached is None else (grad.to(cached.dtype) + cached).to(grad.dtype)

        # Fp has no graph; Fr attaches the hook to the recomputed state.
        if live.requires_grad:
            live.register_hook(hook)

    def link_output(self, name: str, live: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Connect a separate native state output that the caller would
        discard."""
        return _StateGradient.apply(output, live, self._leaf(name, live))


def _accumulate_gradient(leaf: torch.Tensor) -> None:
    grad = leaf.grad
    dtype = torch.float32 if grad.dtype in (torch.float16, torch.bfloat16) else grad.dtype
    if leaf._chunkflow_grad_sum is None:
        leaf._chunkflow_grad_sum = grad.to(dtype=dtype, copy=True)
    else:
        leaf._chunkflow_grad_sum.add_(grad)
    leaf.grad = None


def _cached_gradient(leaf: torch.Tensor) -> torch.Tensor | None:
    return getattr(leaf, "_chunkflow_grad_sum", leaf.grad)


class _StateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, output: torch.Tensor, state: torch.Tensor, leaf: torch.Tensor) -> torch.Tensor:
        ctx.leaf = leaf
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        return grad_output, _cached_gradient(ctx.leaf), None


class StateStoreError(RuntimeError):
    pass


class StateStore:
    def __init__(self) -> None:
        self._chunks: dict[int, ChunkState] = {}

    def put(self, chunk_id: int, state: ChunkState) -> None:
        if chunk_id != len(self._chunks):
            raise StateStoreError(
                f"put() must be strictly ascending starting at 0: "
                f"expected chunk_id={len(self._chunks)}, got {chunk_id}"
            )
        self._chunks[chunk_id] = state

    def lookup(self, upto_chunk_id: int) -> list[ChunkState]:
        if upto_chunk_id >= len(self._chunks):
            missing = list(range(len(self._chunks), upto_chunk_id + 1))
            raise StateStoreError(
                f"lookup(upto_chunk_id={upto_chunk_id}) requested missing/freed chunk id(s) {missing}"
            )
        return [self._chunks[i] for i in range(upto_chunk_id + 1)]

    def __len__(self) -> int:
        return len(self._chunks)
