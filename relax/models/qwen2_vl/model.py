# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Original HF Qwen2-VL vision tower with a Megatron language transformer."""

import inspect
from typing import Any

import torch
from megatron.core.transformer.module import MegatronModule


class Qwen2VLModel(MegatronModule):
    def __init__(self, config: Any, pre_process: bool = True, post_process: bool = True) -> None:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VisionTransformerPretrainedModel,
        )
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLModel as HFQwen2VLModel,
        )

        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.language_model = config.provide_language_model(pre_process=pre_process, post_process=post_process)
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config) if pre_process else None
        self._hf_helpers = HFQwen2VLModel
        self._rope_accepts_token_types = (
            "mm_token_type_ids" in inspect.signature(HFQwen2VLModel.get_rope_index).parameters
        )

    @property
    def decoder(self) -> Any:
        return self.language_model.decoder

    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor: Any) -> None:
        self.language_model.set_input_tensor(input_tensor)

    def get_vision_position_ids(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._hf_helpers.get_vision_position_ids(self, *args, **kwargs)

    def _position_ids(
        self, input_ids: torch.Tensor, packed_seq_params: Any, image_grid_thw: Any, video_grid_thw: Any
    ) -> torch.Tensor:
        if image_grid_thw is None and video_grid_thw is None:
            # Text PT needs ordinary positions reset at each packed boundary.
            # Build them on device, avoiding HF's multimodal CPU inspection in
            # the common NLI hot path.
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            if packed_seq_params is not None:
                if packed_seq_params.qkv_format != "thd":
                    raise ValueError("Qwen2-VL packed input requires qkv_format=thd")
                starts = packed_seq_params.cu_seqlens_q_padded
                if starts is None:
                    starts = packed_seq_params.cu_seqlens_q
                starts = starts.to(device=input_ids.device, dtype=positions.dtype)
                index = torch.searchsorted(starts, positions, right=True) - 1
                positions = positions - starts[index]
            return positions.view(1, 1, -1).expand(3, 1, -1)

        from megatron.bridge.training.utils.packed_seq_utils import (
            repack_mcore_thd_position_ids,
            unpack_mcore_thd_tensor_for_position_ids,
        )

        mask, starts, lengths = None, None, None
        rope_tokens = input_ids
        if packed_seq_params is not None:
            if packed_seq_params.qkv_format != "thd":
                raise ValueError("Qwen2-VL packed input requires qkv_format=thd")
            rope_tokens, mask, starts, lengths = unpack_mcore_thd_tensor_for_position_ids(input_ids, packed_seq_params)
        kwargs = {"image_grid_thw": image_grid_thw, "video_grid_thw": video_grid_thw, "attention_mask": mask}
        if self._rope_accepts_token_types:
            token_types = torch.zeros_like(rope_tokens, dtype=torch.int)
            token_types.masked_fill_(rope_tokens == self.config.image_token_id, 1)
            token_types.masked_fill_(rope_tokens == self.config.video_token_id, 2)
            kwargs["mm_token_type_ids"] = token_types
        positions, _ = self._hf_helpers.get_rope_index(self, rope_tokens, **kwargs)
        if starts is not None:
            positions = repack_mcore_thd_position_ids(
                positions, padded_starts=starts, lengths=lengths, total_length=input_ids.shape[1]
            )
        return positions

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        packed_seq_params: Any = None,
        loss_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if kwargs.get("inference_context") is not None or kwargs.get("inference_params") is not None:
            raise ValueError("Qwen2-VL incremental Megatron inference is not implemented")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Qwen2-VL expects Relax's packed [1, total_tokens] input")
        if attention_mask is not None:
            raise ValueError("Qwen2-VL uses packed attention boundaries, not a padded attention mask")
        decoder_input = None
        if self.pre_process:
            decoder_input = self.language_model.embedding(input_ids=input_ids, position_ids=None).transpose(0, 1)
            for pixels, grid, token_id in (
                (pixel_values, image_grid_thw, self.config.image_token_id),
                (pixel_values_videos, video_grid_thw, self.config.video_token_id),
            ):
                if pixels is None:
                    continue
                if grid is None:
                    raise ValueError("Qwen2-VL vision pixels require the matching grid_thw")
                visual_output = self.visual(pixels.to(dtype=self.visual.dtype), grid_thw=grid)
                features = getattr(visual_output, "pooler_output", visual_output)
                if not isinstance(features, torch.Tensor):
                    raise TypeError("Qwen2-VL vision must return a tensor or pooler_output tensor")
                mask = (input_ids == token_id).unsqueeze(-1).expand_as(decoder_input)
                torch._assert_async(mask.sum() == features.numel(), "Qwen2-VL vision token and feature counts differ")
                decoder_input = decoder_input.masked_scatter(mask, features.to(decoder_input))
            decoder_input = decoder_input.transpose(0, 1).contiguous()
        if position_ids is None:
            position_ids = self._position_ids(input_ids, packed_seq_params, image_grid_thw, video_grid_thw)
        return self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=None,
            decoder_input=decoder_input,
            labels=labels,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
        )

    def freeze(self, language: bool, vision: bool, projection: bool) -> None:
        if language:
            self.language_model.requires_grad_(False)
        if self.visual is not None:
            if vision:
                self.visual.patch_embed.requires_grad_(False)
                self.visual.blocks.requires_grad_(False)
            if projection:
                self.visual.merger.requires_grad_(False)
