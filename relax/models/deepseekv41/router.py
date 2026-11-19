# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Use the checkpoint's modality-specific selection bias with Core's MoE router."""

from __future__ import annotations

from copy import copy
from typing import Any

import torch
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.moe.router import TopKRouter

from .vision import load_frozen_weights


class V41VisionRouter(TopKRouter):
    def __init__(self, config: Any, *, layer_number: int, **kwargs: Any) -> None:
        super().__init__(config=config, layer_number=layer_number, **kwargs)
        self.image_token_id = config.v41_image_token_id
        name = f"layers.{layer_number - 1}.ffn.gate.bias_vl"
        bias = load_frozen_weights(config.v41_hf_checkpoint, [name], self.expert_bias.device)[name]
        if bias.dtype != torch.float32 or bias.shape != self.expert_bias.shape:
            raise ValueError(f"Image router bias {name} must be an FP32 vector matching the text expert bias.")
        self.register_buffer("vision_bias", mark_keep_in_fp32(bias), persistent=False)
        self.frozen_expert_bias = True
        # TE's fused router accepts a single expert-bias vector for the whole batch.
        self._per_token_config = copy(config)
        self._per_token_config.moe_router_fusion = False

    def routing(
        self,
        logits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        packed_seq_params: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids is None:
            return super().routing(logits, padding_mask, input_ids, packed_seq_params)
        if input_ids.shape != logits.shape[:2]:
            raise ValueError(
                "V4.1 visual routing requires sequence-parallel token IDs aligned with the router logits."
            )
        image = input_ids.reshape(-1, 1) == self.image_token_id
        # Reuse all of Core's routing, including replay, dispatch format and token
        # dropping. Only its score-selection bias differs. This short-lived view
        # never replaces the original module's persistent buffer during forward.
        view = copy(self)
        view.config = self._per_token_config
        view._buffers = {**self._buffers, "expert_bias": torch.where(image, self.vision_bias, self.expert_bias)}
        return TopKRouter.routing(view, logits, padding_mask, input_ids, packed_seq_params)
