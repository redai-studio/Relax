# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import inspect
from functools import partial
from types import SimpleNamespace

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from torch import Tensor


class MiMoV2ImageModel(GPTModel):
    """MiMo image encoder and patch merger on the first language pipeline
    stage."""

    def build_visual(self) -> None:
        from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource
        from torch.utils.checkpoint import checkpoint
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        visual_cls = get_class_from_dynamic_module("modeling_mimo_v2.MiMoVisionTransformer", self.config.hf_model_path)
        self.visual = visual_cls(SimpleNamespace(**self.config.vision_config)).to(dtype=self.config.params_dtype)
        # The released HF class constructs merger biases even in checkpoints that omit them.
        keys = set(SafeTensorsStateSource(self.config.hf_model_path).get_all_keys())
        for name, module in self.visual.merger.named_modules():
            if getattr(module, "bias", None) is not None and f"visual.merger.{name}.bias" not in keys:
                module.register_parameter("bias", None)
        self.visual.requires_grad_(not self.config.freeze_vision_model)
        self.visual.merger.requires_grad_(not self.config.freeze_vision_projection)
        for parameter in self.visual.parameters():
            # The encoder is replicated across TP. SP scatter gathers its output gradient on each rank.
            parameter.average_gradients_across_tp_domain = True
        if self.config.recompute_granularity == "full" and not self.config.freeze_vision_model:
            for block in self.visual.blocks:
                block.forward = partial(checkpoint, block.forward, use_reentrant=False)

    def image_embeddings(
        self, input_ids: Tensor, position_ids: Tensor | None, pixels: Tensor | None, grid: Tensor | None
    ) -> Tensor:
        embeddings = self.embedding(input_ids=input_ids, position_ids=position_ids).transpose(0, 1)
        mask = input_ids == self.config.image_token_id
        if (pixels is None) != (grid is None):
            raise ValueError("MiMo image inputs require both pixel_values and image_grid_thw.")
        if pixels is not None:
            features = self.visual(pixel_values=pixels, grid_thw=grid)
            torch._assert_async(mask.sum() == features.shape[0], "MiMo image token and feature counts must match")
            embeddings = embeddings.masked_scatter(mask.unsqueeze(-1), features.to(embeddings))
        else:
            torch._assert_async(~mask.any(), "MiMo image tokens require pixel_values")
            if self.training and not (self.config.freeze_vision_model and self.config.freeze_vision_projection):
                # A text-only DP rank must still participate in visual gradient reduction.
                merge = self.visual.spatial_merge_size
                weight = self.visual.patch_embed.proj.weight
                dummy = weight.new_zeros((merge * merge, weight[0].numel()))
                features = self.visual(pixel_values=dummy, grid_thw=torch.tensor([[1, merge, merge]]))
                embeddings = embeddings + features.reshape(-1)[:0].sum().to(embeddings)
        embeddings = embeddings.transpose(0, 1).contiguous()
        if self.config.sequence_parallel:
            embeddings = scatter_to_sequence_parallel_region(embeddings, group=self.pg_collection.tp).contiguous()
        return embeddings

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        decoder_input: Tensor | None = None,
        *args,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        padding_mask: Tensor | None = None,
        fp32_output: bool | None = None,
        **kwargs,
    ) -> Tensor:
        if self.pre_process and decoder_input is None:
            decoder_input = self.image_embeddings(input_ids, position_ids, pixel_values, image_grid_thw)
            if padding_mask is not None and self.config.sequence_parallel:
                padding_mask = scatter_to_sequence_parallel_region(
                    padding_mask.T.contiguous(), group=self.pg_collection.tp
                ).T.contiguous()
        # Newer MCore versions configure logit dtype on the head; older ones accept it per forward.
        if fp32_output is not None and "fp32_output" in inspect.signature(GPTModel.forward).parameters:
            kwargs["fp32_output"] = fp32_output
        return super().forward(
            input_ids, position_ids, attention_mask, decoder_input, *args, padding_mask=padding_mask, **kwargs
        )
