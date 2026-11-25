# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import TYPE_CHECKING, Any

import torch
from megatron.bridge.models.minimax_m3.modeling_minimax_m3_vl import MiniMaxM3VLModel


if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext
    from megatron.core.packed_seq_params import PackedSeqParams


class MiniMaxM31VLModel(MiniMaxM3VLModel):
    """Run the pretrained vision tower separately for each packed sample."""

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | Sequence[Sequence[Sequence[int]]] | None = None,
        video_grid_thw: torch.Tensor | Sequence[Sequence[Sequence[int]]] | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        runtime_gather_output: bool | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        extra_block_kwargs: dict[str, Any] | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        loss_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise ValueError("MiniMax-M3.1-Flash SFT supports text and image inputs.")
        if self.config._pg_collection.cp.size() > 1:
            # Bridge injects image features into a single packed stream before
            # slicing for CP and SP. Relax stages the unsplit data as one
            # padded row per document.
            assert packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
            bounds = packed_seq_params.cu_seqlens_q_cpu
            lengths = [end - start for start, end in zip(bounds[:-1], bounds[1:], strict=True)]
            assert input_ids is not None and input_ids.size(0) == len(lengths)
            input_ids = torch.cat([row[:length] for row, length in zip(input_ids, lengths, strict=True)]).unsqueeze(0)
            # Bridge needs a full-length reference tensor to slice even on
            # pipeline stages that have no embeddings. THD RoPE itself uses cu_seqlens.
            if not self.pre_process:
                positions = torch.arange(max(lengths), device=input_ids.device)
                position_ids = torch.cat([positions[:length] for length in lengths]).unsqueeze(0)
            attention_mask = None
        if labels is None:
            # Relax SFT consumes the batch mask when computing its external loss.
            loss_mask = None
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            labels=labels,
            inference_context=inference_context,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            inference_params=inference_params,
            loss_mask=loss_mask,
            **kwargs,
        )

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
