# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright (c) 2023 Zhongjie Duan.
# Adapted from the Qwen-Image schedule in DiffSynth-Studio f7b7db96.
# Source: https://github.com/modelscope/DiffSynth-Studio
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import torch


class QwenImageTrainingSchedule:
    """Flow-matching training schedule and BSMNTW loss weighting for Qwen-
    Image."""

    def __init__(self) -> None:
        self.num_train_timesteps = 1000
        sigmas = torch.linspace(1.0, 0.0, 1001, dtype=torch.float32)[:-1]
        sigmas = math.exp(0.8) / (math.exp(0.8) + (1 / sigmas - 1))
        one_minus_z = 1 - sigmas
        scale_factor = one_minus_z[-1] / (1 - 0.02)
        self.sigmas = 1 - one_minus_z / scale_factor
        self.timesteps = self.sigmas * self.num_train_timesteps
        y = torch.exp(-2 * ((self.timesteps - 500) / 1000) ** 2)
        y_shifted = y - y.min()
        self.linear_timesteps_weights = y_shifted * (1000 / y_shifted.sum())

    def lookup_index(self, timestep: torch.Tensor) -> torch.Tensor:
        """Return the schedule index whose timestep is closest to
        ``timestep``."""
        if timestep.device.type != "cpu" or timestep.numel() != 1:
            raise ValueError("Qwen training schedule requires one CPU timestep")
        return torch.argmin((self.timesteps - timestep).abs())

    def add_noise(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Interpolate ``sample`` toward ``noise`` at the schedule entry for
        ``timestep``."""
        sigma = self.sigmas[self.lookup_index(timestep)]
        return (1 - sigma) * sample + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Return the flow-matching velocity target ``noise - sample``."""
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        """Return the scalar training loss weight at ``timestep``."""
        return self.linear_timesteps_weights[self.lookup_index(timestep)]
