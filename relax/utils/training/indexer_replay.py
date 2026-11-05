# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


_VALID_STAGES = {"fallthrough", "replay_forward", "replay_backward"}


@dataclass
class _LayerIndexerReplay:
    entries: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    forward_index: int = 0
    backward_index: int = 0

    def record(self, topk: torch.Tensor, row_valid: torch.Tensor) -> None:
        if topk.ndim != 2 or topk.dtype != torch.int32:
            raise ValueError(f"indexer top-k must be 2-D int32, got shape={tuple(topk.shape)}, dtype={topk.dtype}")
        if row_valid.ndim != 1 or row_valid.dtype != torch.bool or row_valid.shape[0] != topk.shape[0]:
            raise ValueError(
                "indexer replay row-valid mask must be 1-D bool with the same row count as top-k, "
                f"got topk={tuple(topk.shape)}, row_valid={tuple(row_valid.shape)}/{row_valid.dtype}"
            )
        self.entries.append((topk, row_valid))

    def pop(self, stage: str) -> tuple[torch.Tensor, torch.Tensor]:
        if stage == "replay_forward":
            index = self.forward_index
            self.forward_index += 1
        elif stage == "replay_backward":
            index = self.backward_index
            self.backward_index += 1
        else:
            raise ValueError(f"Cannot pop indexer replay in stage {stage!r}")
        if index >= len(self.entries):
            raise RuntimeError(
                f"Indexer replay {stage} consumed more microbatches than recorded: index={index}, "
                f"recorded={len(self.entries)}"
            )
        return self.entries[index]


class IndexerReplay:
    """Step-local DSV4 C4 indexer choices, keyed by global one-based layer
    number."""

    _enabled = False
    _stage = "fallthrough"
    _layers: dict[int, _LayerIndexerReplay] = {}

    @classmethod
    def begin_step(cls, layer_numbers: list[int]) -> None:
        cls._enabled = True
        cls._stage = "fallthrough"
        cls._layers = {layer_number: _LayerIndexerReplay() for layer_number in layer_numbers}

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

    @classmethod
    def get_stage(cls) -> str:
        return cls._stage

    @classmethod
    def set_stage(cls, stage: str) -> None:
        if stage not in _VALID_STAGES:
            raise ValueError(f"Invalid indexer replay stage {stage!r}; expected one of {sorted(_VALID_STAGES)}")
        cls._stage = stage

    @classmethod
    @contextmanager
    def forward_stage(cls) -> Iterator[None]:
        """Use the forward cursor during a train forward, then restore backward
        replay."""
        old_stage = cls._stage
        switch_stage = cls._enabled and old_stage == "replay_backward"
        if switch_stage:
            cls._stage = "replay_forward"
        try:
            yield
        finally:
            if switch_stage:
                cls._stage = old_stage

    @classmethod
    def record(
        cls,
        layer_number: int,
        topk: torch.Tensor,
        row_valid: torch.Tensor,
    ) -> None:
        try:
            replay = cls._layers[layer_number]
        except KeyError as exc:
            raise RuntimeError(f"Unexpected DSV4 indexer replay layer {layer_number}") from exc
        replay.record(topk, row_valid)

    @classmethod
    def replay(cls, layer_number: int, computed_topk: torch.Tensor | None) -> torch.Tensor | None:
        if not cls._enabled or cls._stage == "fallthrough":
            return computed_topk
        try:
            replay = cls._layers[layer_number]
        except KeyError as exc:
            raise RuntimeError(f"Missing rollout indexer replay for DSV4 layer {layer_number}") from exc

        rollout_topk, row_valid = replay.pop(cls._stage)
        if computed_topk is None:
            return None
        if rollout_topk.shape != computed_topk.shape:
            raise RuntimeError(
                f"DSV4 indexer replay shape mismatch at layer {layer_number}: "
                f"rollout={tuple(rollout_topk.shape)}, trainer={tuple(computed_topk.shape)}"
            )
        if rollout_topk.device != computed_topk.device or row_valid.device != computed_topk.device:
            raise RuntimeError(
                f"DSV4 indexer replay device mismatch at layer {layer_number}: "
                f"rollout={rollout_topk.device}, mask={row_valid.device}, trainer={computed_topk.device}"
            )
        return torch.where(row_valid.unsqueeze(1), rollout_topk, computed_topk)

    @classmethod
    def reset_forward(cls, require_consumed: bool = True) -> None:
        for layer_number, replay in cls._layers.items():
            if require_consumed and replay.forward_index != len(replay.entries):
                raise RuntimeError(
                    f"DSV4 layer {layer_number} consumed {replay.forward_index}/{len(replay.entries)} "
                    "forward indexer replay microbatches"
                )
            replay.forward_index = 0

    @classmethod
    def finish_step(cls, expect_backward: bool) -> None:
        for layer_number, replay in cls._layers.items():
            if replay.forward_index != len(replay.entries):
                raise RuntimeError(
                    f"DSV4 layer {layer_number} consumed {replay.forward_index}/{len(replay.entries)} "
                    "training-forward indexer replay microbatches"
                )
            if expect_backward and replay.backward_index != len(replay.entries):
                raise RuntimeError(
                    f"DSV4 layer {layer_number} consumed {replay.backward_index}/{len(replay.entries)} "
                    "recompute-backward indexer replay microbatches"
                )
        cls.clear()

    @classmethod
    def clear(cls) -> None:
        cls._enabled = False
        cls._stage = "fallthrough"
        cls._layers = {}


def maybe_replay_indexer_topk(
    layer_number: int,
    computed_topk: torch.Tensor | None,
) -> torch.Tensor | None:
    return IndexerReplay.replay(layer_number, computed_topk)
