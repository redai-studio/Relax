# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import Any

import torch
from megatron.core.transformer.module import MegatronModule

from relax.backends.megatron.cp_utils import gdn_cp_slice

from .positions import image_positions
from .vision import build_vision, image_features


class KeyeVLModel(MegatronModule):
    uses_unsplit_forward = True

    def __init__(self, config: Any, language_model: torch.nn.Module) -> None:
        super().__init__(config=config)
        if config.tensor_model_parallel_size != 1 or config.pipeline_model_parallel_size != 1:
            raise ValueError("Keye multimodal training currently requires TP1/PP1; CP and EP are supported")
        self.language_model = language_model
        for name in ("pre_process", "post_process", "pg_collection", "share_embeddings_and_output_weights"):
            setattr(self, name, getattr(language_model, name))
        self.visual, self.mlp_AR = build_vision(config.hf_config, config.hf_model_path)
        for module in (self.visual, self.mlp_AR):
            module.to(dtype=config.params_dtype)
            for parameter in module.parameters():
                parameter.allreduce = True
                parameter.sequence_parallel = False

    def freeze(self, language: bool, vision: bool, projection: bool) -> None:
        for module, frozen in ((self.language_model, language), (self.visual, vision), (self.mlp_AR, projection)):
            if frozen:
                module.requires_grad_(False)

    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, value: Any) -> None:
        self.language_model.set_input_tensor(value)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Any = None,
        attention_mask: Any = None,
        packed_seq_params: Any = None,
        pixel_values: Any = None,
        grid_thws: Any = None,
        image_grid_thw: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        boundaries = packed_seq_params.cu_seqlens_q_cpu
        cp = self.pg_collection.cp
        size, rank = cp.size(), cp.rank()
        widths = [end - start for start, end in zip(boundaries, boundaries[1:])]
        if input_ids.shape[0] == len(widths):
            ids = torch.cat([row[:length] for row, length in zip(input_ids, widths)]).unsqueeze(0)
            valid = (
                torch.cat([row[:length] for row, length in zip(attention_mask, widths)]).bool()
                if attention_mask is not None
                else torch.arange(ids.numel(), device=ids.device)
                < getattr(packed_seq_params, "num_valid_tokens", ids.numel())
            )
            # Keep document resets while computing coordinates in the padded BSHD layout.
            positions = image_positions(input_ids, grid_thws, self.config.hf_config.image_token_id)
            positions = torch.cat([positions[:, row, :length] for row, length in enumerate(widths)], -1)
        else:
            ids = input_ids
            valid = torch.arange(ids.numel(), device=ids.device) < getattr(
                packed_seq_params, "num_valid_tokens", ids.numel()
            )
            positions = image_positions(ids, grid_thws, self.config.hf_config.image_token_id)[:, 0]
            positions = torch.cat(
                [
                    positions[:, start:end] - positions[:, start : start + 1]
                    for start, end in zip(boundaries, boundaries[1:])
                ],
                -1,
            )
        hidden = self.language_model.embedding(input_ids=ids, position_ids=None).transpose(0, 1)
        if pixel_values is not None:
            features = image_features(self.visual, self.mlp_AR, pixel_values, grid_thws)
            mask = ids == self.config.hf_config.image_token_id
            torch._assert_async(mask.sum() == features.shape[0], "Keye image token/feature counts differ")
            hidden = hidden.masked_scatter(mask.unsqueeze(-1), features.to(hidden))
        elif self.training and torch.is_grad_enabled():
            p = next(self.visual.parameters())
            patch = self.config.hf_config.vision_config.patch_size
            dummy = image_features(
                self.visual, self.mlp_AR, p.new_zeros(4, 3, patch, patch), torch.tensor([[1, 2, 2]])
            )
            hidden = hidden + dummy.sum() * 0
        packed_seq_params.keye_positions = positions.transpose(0, 1)
        packed_seq_params.keye_valid_mask = valid
        if size > 1:
            hidden = gdn_cp_slice(hidden.transpose(0, 1), boundaries, size, rank).transpose(0, 1)
            ids = gdn_cp_slice(ids.transpose(0, 1), boundaries, size, rank).transpose(0, 1)
        return self.language_model(
            input_ids=ids,
            position_ids=positions.unsqueeze(1),
            attention_mask=None,
            decoder_input=hidden.transpose(0, 1).contiguous(),
            packed_seq_params=packed_seq_params,
            **kwargs,
        )
