# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Hash-locked Bridge checkpoint body with retained QSA one-layer chunks."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import te_checkpoint
from megatron.core import tensor_parallel
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from torch import Tensor


def qsa_bridge_checkpointed_forward(
    self,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Tensor,
    context_mask: Tensor,
    rotary_pos_emb: Tensor,
    attention_bias: Tensor,
    packed_seq_params: PackedSeqParams,
    padding_mask: Optional[Tensor],
    use_inner_fp8_context: bool,
    # args for deepstack
    visual_pos_masks: Optional[torch.Tensor] = None,
    deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
):
    """Forward method with activation checkpointing."""

    def custom(start: int, end: int):
        # Pipeline schedules may prepare a later microbatch before this checkpoint
        # replays. Keep each layer's PLE inputs with its own checkpoint closure.
        ple_states = {}
        for index in range(start, end):
            ple = getattr(self._get_layer(index), "per_layer_embedding", None)
            if ple is not None:
                ple_states[index] = (ple._ngram_ids, ple._position_in_sequence, ple._cp_packed_seq_params)

        def custom_forward(
            hidden_states,
            attention_mask,
            context,
            context_mask,
            rotary_pos_emb,
            padding_mask,
            visual_pos_masks,
            *deepstack_visual_embeds_args,
        ):
            deepstack_visual_embeds = list(deepstack_visual_embeds_args) if deepstack_visual_embeds_args else None
            for index in range(start, end):
                layer = self._get_layer(index)
                ple = getattr(layer, "per_layer_embedding", None)
                if ple is not None:
                    previous_ple_state = (ple._ngram_ids, ple._position_in_sequence, ple._cp_packed_seq_params)
                    ple._ngram_ids, ple._position_in_sequence, ple._cp_packed_seq_params = ple_states[index]
                inner_fp8_context = (
                    get_fp8_context(self.config, layer.layer_number - 1) if use_inner_fp8_context else nullcontext()
                )
                try:
                    with inner_fp8_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                        )

                        if self.pre_process and deepstack_visual_embeds is not None:
                            l_no = layer.layer_number - 1
                            if l_no in range(len(deepstack_visual_embeds)):
                                hidden_states = self._deepstack_process(
                                    hidden_states,
                                    visual_pos_masks,
                                    deepstack_visual_embeds[l_no],
                                )
                finally:
                    if ple is not None:
                        ple._ngram_ids, ple._position_in_sequence, ple._cp_packed_seq_params = previous_ple_state
            return hidden_states, context

        return custom_forward

    deepstack_visual_embeds_tuple = tuple(deepstack_visual_embeds) if deepstack_visual_embeds else ()

    def checkpoint_handler(forward_func):
        """Determines whether to use the `te_checkpoint` or
        `tensor_parallel.checkpoint`"""
        if self.config.fp8:
            return te_checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                tensor_parallel.random.get_cuda_rng_tracker,
                self.tp_group,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                padding_mask,
                visual_pos_masks,
                *deepstack_visual_embeds_tuple,
            )
        else:
            return tensor_parallel.checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                padding_mask,
                visual_pos_masks,
                *deepstack_visual_embeds_tuple,
            )

    if self.config.recompute_method == "uniform":
        # Uniformly divide the total number of Transformer layers and checkpoint
        # the input activation of each divided chunk.
        # A method to further reduce memory usage reducing checkpoints.
        layer_idx = 0
        while layer_idx < self.num_layers_per_pipeline_rank:
            chunk_end = min(layer_idx + self.config.recompute_num_layers, self.num_layers_per_pipeline_rank)
            forward_func = custom(layer_idx, chunk_end)
            if chunk_end == layer_idx + 1 and getattr(
                self._get_layer(layer_idx), "_relax_qsa_attention_retained", False
            ):
                hidden_states, context = forward_func(
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    padding_mask,
                    visual_pos_masks,
                    *deepstack_visual_embeds_tuple,
                )
            else:
                hidden_states, context = checkpoint_handler(forward_func)

            layer_idx += self.config.recompute_num_layers

    elif self.config.recompute_method == "block":
        # Checkpoint the input activation of only a set number of individual
        # Transformer layers and skip the rest.
        # A method fully use the device memory removing redundant re-computation.
        recompute_skip_num_layers = 0
        for layer_idx in range(self.num_layers_per_pipeline_rank):
            # Skip recomputation when input grad computation is not needed.
            # Need to have at least one input tensor with gradient computation
            # for re-enterant autograd engine.
            if self.config.fp8 and not hidden_states.requires_grad:
                recompute_skip_num_layers += 1
            if (
                layer_idx >= recompute_skip_num_layers
                and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
            ):
                hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
            else:
                hidden_states, context = custom(layer_idx, layer_idx + 1)(
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    padding_mask,
                    visual_pos_masks,
                    *deepstack_visual_embeds_tuple,
                )
    else:
        raise ValueError("Invalid activation recompute method.")

    return hidden_states
