# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Sequence
from math import prod

import torch
from megatron.bridge.models.minimax_m3.modeling_minimax_m3_vl import MiniMaxM3VLModel


class MiniMaxM31VLModel(MiniMaxM3VLModel):
    """Run the pretrained vision tower separately for each packed sample."""

    def _project_vision(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor | Sequence[Sequence[Sequence[int]]]
    ) -> torch.Tensor:
        if isinstance(grid_thw, torch.Tensor):
            if grid_thw.device.type != "cpu":
                raise ValueError("MiniMax media grids must stay on CPU.")
            grids = [grid_thw]
        else:
            grids = [torch.tensor(sample_grids, dtype=torch.long) for sample_grids in grid_thw]
        patch_counts = [sum(prod(grid) for grid in sample_grids.tolist()) for sample_grids in grids]
        if not patch_counts or any(count <= 0 for count in patch_counts) or sum(patch_counts) != pixel_values.shape[0]:
            raise ValueError("MiniMax media grids do not match the processor's flattened patches.")
        # Bridge's vision attention spans every patch in a call. Preserve that
        # behavior within each sample and prevent attention across packed samples.
        project = super()._project_vision
        features = [
            project(patches, grid) for patches, grid in zip(pixel_values.split(patch_counts), grids, strict=True)
        ]
        return torch.cat(features, dim=0)

    @staticmethod
    def _scatter_features(
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor | None,
        *,
        token_id: int,
        features: torch.Tensor | None,
        modality: str,
    ) -> torch.Tensor:
        if features is None:
            return inputs_embeds
        if input_ids is None:
            raise ValueError(f"input_ids are required for MiniMax {modality} features.")
        mask = input_ids == token_id
        torch._assert_async(
            mask.sum() == features.shape[0], f"MiniMax {modality} feature count does not match placeholder tokens."
        )
        return inputs_embeds.masked_scatter(mask.unsqueeze(-1), features.to(inputs_embeds))
