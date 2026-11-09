# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Packed CP1/PP1 Gemma4 image-input adapter for Relax's external SFT loss."""

import inspect
from copy import copy

import torch
import torch.nn.functional as F
from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4VLModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region


class PackedGemma4VLModel(Gemma4VLModel):
    def _scatter_modality_features(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        features: torch.Tensor,
        token_id: int,
        modality_name: str,
    ) -> torch.Tensor:
        mask = (input_ids == token_id).unsqueeze(-1).expand_as(inputs_embeds)
        n_features = features.numel() // inputs_embeds.shape[-1]
        torch._assert_async(
            (input_ids == token_id).sum() == n_features,
            f"Gemma4 {modality_name} token count does not match encoder features",
        )
        return inputs_embeds.masked_scatter(mask, features.to(inputs_embeds.device, inputs_embeds.dtype))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        *,
        loss_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        fp32_output: bool = True,
    ) -> torch.Tensor:
        if self.config.context_parallel_size != 1 or self.config.pipeline_model_parallel_size != 1:
            raise NotImplementedError("Packed Gemma4 image SFT requires CP1 and PP1")
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError("Packed Gemma4 image SFT requires THD sequence boundaries")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Packed Gemma4 image SFT requires input_ids shaped [1, T]")
        if attention_mask is not None:
            raise ValueError("Packed Gemma4 image SFT derives its mask from sequence boundaries and image tokens")
        if input_features is not None:
            raise NotImplementedError("Packed Gemma4 audio inputs have not been integrated")
        if labels is not None:
            raise ValueError("Packed Gemma4 image SFT uses Relax's external loss")
        del loss_mask, padding_mask

        image_mask = input_ids == self.config.image_token_id
        lm_input_ids = input_ids.masked_fill(image_mask, self.config.text_config.pad_token_id)
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embedding(input_ids=lm_input_ids, position_ids=None)
            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()
            if getattr(self.language_model.config, "scale_embeddings_by_hidden_size", False):
                inputs_embeds = inputs_embeds * self.language_model.config.hidden_size**0.5
        if pixel_values is not None:
            features = self.get_image_features(pixel_values, image_position_ids=image_position_ids)
            inputs_embeds = self._scatter_modality_features(
                inputs_embeds,
                input_ids,
                features,
                self.config.image_token_id,
                "image",
            )
            if getattr(self.config.text_config, "use_bidirectional_attention", None) == "vision":
                # Sliding layers consume image groups; global layers stay causal.
                # Keep per-batch metadata off the caller's/recomputed params.
                packed_seq_params = copy(packed_seq_params)
                starts = image_mask & ~F.pad(image_mask[:, :-1], (1, 0), value=False)
                groups = starts.long().cumsum(-1)
                packed_seq_params.gemma4_vision_group_ids = torch.where(image_mask, groups, -1).reshape(-1)

        inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()
        if self.config.sequence_parallel:
            inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds, group=self.config._pg_collection.tp)
        language_inputs = dict(
            input_ids=lm_input_ids,
            position_ids=position_ids,
            attention_mask=None,
            decoder_input=inputs_embeds,
            labels=None,
            loss_mask=None,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )
        # MCore versions with logit-dtype configuration keep the native head
        # dtype and no longer accept the per-forward fp32_output keyword.
        if "fp32_output" in inspect.signature(self.language_model.forward).parameters:
            language_inputs["fp32_output"] = fp32_output
        return self.language_model(**language_inputs)
