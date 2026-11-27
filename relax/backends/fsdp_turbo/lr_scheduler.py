# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Learning-rate schedule state for FSDPTurbo training."""

from typing import Callable

import torch


class MegatronParityWarmupCosineScheduler:
    """Megatron-compatible linear-warmup/cosine scheduler.

    Write ``param_group["lr"]`` directly to match Megatron rounding; ``step()``
    runs after ``optimizer.step()``.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, lr_fn: Callable[[int], float]) -> None:
        self.optimizer = optimizer
        self._lr_fn = lr_fn
        self.last_epoch = 0
        self._apply_lr()

    def _apply_lr(self) -> None:
        lr = self._lr_fn(self.last_epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self) -> dict[str, int]:
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.last_epoch = state["last_epoch"]
        self._apply_lr()

    def step(self) -> None:
        self.last_epoch += 1
        self._apply_lr()
