# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright (c) 2023 Zhongjie Duan.
# Adapted and modified by Relax from DiffSynth-Studio, revision f7b7db96.
# Source: https://github.com/modelscope/DiffSynth-Studio
# SPDX-License-Identifier: Apache-2.0
# See the repository root LICENSE for the full license text.
from __future__ import annotations

from typing import Any

import torch


class FlowMatchScheduler:
    def __init__(self, template: Any = "SenseNova-U1") -> None:
        if template != "SenseNova-U1":
            raise ValueError(template)
        self.num_train_timesteps = 1000
        self.set_timesteps_fn = self.set_timesteps_sensenova_u1

    @staticmethod
    def set_timesteps_sensenova_u1(
        num_inference_steps: Any = 50, denoising_strength: Any = 1.0, shift: Any = 3.0
    ) -> Any:
        num_train_timesteps = 1000
        sigmas = torch.linspace(denoising_strength, 0.0, num_inference_steps + 1)[:-1]
        if shift is not None and shift != 1.0:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def set_training_weight(self) -> Any:
        steps = 1000
        x = self.sigmas * self.num_train_timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            # This is an empirical formula.
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing

    def set_timesteps(
        self, num_inference_steps: Any = 100, denoising_strength: Any = 1.0, training: Any = False, **kwargs: Any
    ) -> Any:
        self.sigmas, self.timesteps = self.set_timesteps_fn(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs,
        )
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def step(self, model_output: Any, timestep: Any, sample: Any, to_final: Any = False, **kwargs: Any) -> Any:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(self, original_samples: Any, noise: Any, timestep: Any) -> Any:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample: Any, noise: Any, timestep: Any) -> Any:
        target = noise - sample
        return target

    def training_weight(self, timestep: Any) -> Any:
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
