# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge grouped-expert LoRA with a single split backward per token partition.

Adapted from megatron.bridge.peft.utils.GroupedExpertLinearAdapter. Keep its
GEMM selection, expert-TP collectives, padding, dropout, scale, empty-token
anchor and weight-module calls (DDP prehooks) intact. Avoid token repartitioning
and weight indexing for ETP1 grouped GEMM; other paths replace independent
narrow nodes with one split node. Parameter layout and checkpoint keys do not
change. Imported lazily by training_ops before K3 model construction.
"""

import math
from itertools import accumulate

import torch
from megatron.bridge.peft.utils import (
    GroupedExpertLinearAdapter,
    _process_group_size,
    pad_seq_to_mult,
    unpad_seq_to_mult,
)
from torch import nn


def _forward_per_expert(
    self,
    x: torch.Tensor,
    *,
    expert_splits: list[int],
    expert_tp_size: int,
) -> torch.Tensor:
    """Apply the adapter using the per-expert fallback path."""

    linear_in_weight = self.linear_in()
    linear_out_weight = self.linear_out()
    outputs = []
    # One multi-output node assembles dx once, instead of E full-size zeros/adds.
    for expert_idx, expert_input in enumerate(torch.split(x, expert_splits, dim=0)):
        pad_len = 0
        if expert_input.numel() > 0:
            expert_input, pad_len = pad_seq_to_mult(expert_input, expert_tp_size)
            if self.config.cpu_offloading and self.config.cpu_offloading_activations:
                expert_input.activation_offloading = True

        hidden = nn.functional.linear(expert_input, linear_in_weight[expert_idx])
        if not self.input_is_parallel:
            hidden = self._gather_along_last_dim(hidden)
        hidden = self.activation(hidden)

        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            hidden.activation_offloading = True
        expert_output = nn.functional.linear(hidden, linear_out_weight[expert_idx])
        if self.input_is_parallel:
            expert_output = self._gather_along_last_dim(expert_output)

        if self.dropout_position == "post":
            expert_output = self.dropout(expert_output)
        if pad_len > 0:
            expert_output = unpad_seq_to_mult(expert_output, pad_len)
        outputs.append(expert_output)

    return torch.cat(outputs, dim=0)


def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Apply the local expert-specific LoRA update to grouped expert inputs."""

    expert_splits = self._extract_expert_splits(args, kwargs)
    total_tokens = sum(expert_splits)
    if total_tokens != x.shape[0]:
        raise ValueError(
            f"Expert splits for {self.base_linear_name} sum to {total_tokens}, but received {x.shape[0]} tokens"
        )
    if self.dropout_position == "pre":
        x = self.dropout(x)

    expert_tp_size = _process_group_size(self.expert_tp_group, self.config.expert_tensor_parallel_size or 1)
    output_features = self.linear_out.weight.shape[1]
    if self.input_is_parallel:
        output_features *= expert_tp_size
    if x.shape[0] == 0:
        linear_in_weight = self.linear_in()
        linear_out_weight = self.linear_out()
        grad_anchor = linear_in_weight.reshape(-1)[0] + linear_out_weight.reshape(-1)[0]
        return (x.new_empty((0, output_features)) + grad_anchor * 0.0) * (self.alpha / self.dim)

    fp8_enabled = self._is_te_fp8_enabled()
    use_te_fp8 = fp8_enabled and self._can_use_te_grouped_linear_fp8(x)
    use_grouped_mm = not fp8_enabled and self._can_use_grouped_mm(x)
    if not use_te_fp8 and not use_grouped_mm:
        return self._forward_per_expert(x, expert_splits=expert_splits, expert_tp_size=expert_tp_size) * (
            self.alpha / self.dim
        )

    if use_grouped_mm and expert_tp_size == 1:
        # Inputs are already contiguous in expert order. Keep zero-token experts
        # as repeated offsets instead of copying x and gathering/scattering weights.
        # Call the weight modules to preserve DDP parameter-gather prehooks.
        offs = torch.tensor(list(accumulate(expert_splits)), device=x.device, dtype=torch.int32)
        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            x.activation_offloading = True
        hidden = nn.functional.grouped_mm(x, self.linear_in().transpose(1, 2), offs=offs)
        hidden = self.activation(hidden)
        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            hidden.activation_offloading = True
        expert_output = nn.functional.grouped_mm(hidden, self.linear_out().transpose(1, 2), offs=offs)
        if self.dropout_position == "post":
            expert_output = self.dropout(expert_output)
        return expert_output * (self.alpha / self.dim)

    active_expert_indices = []
    grouped_inputs = []
    padded_splits = []
    pad_lengths = []
    grouped_padding_multiple = math.lcm(expert_tp_size, 16) if use_te_fp8 else expert_tp_size
    for expert_idx, expert_input in enumerate(torch.split(x, expert_splits, dim=0)):
        if expert_splits[expert_idx] == 0:
            continue
        expert_input, pad_len = pad_seq_to_mult(expert_input, grouped_padding_multiple)
        active_expert_indices.append(expert_idx)
        grouped_inputs.append(expert_input)
        padded_splits.append(expert_input.shape[0])
        pad_lengths.append(pad_len)

    grouped_input = grouped_inputs[0] if len(grouped_inputs) == 1 else torch.cat(grouped_inputs, dim=0)
    if self.config.cpu_offloading and self.config.cpu_offloading_activations:
        grouped_input.activation_offloading = True

    offs = None if use_te_fp8 else self._build_grouped_mm_offsets(padded_splits, device=x.device)

    active_linear_in = self.linear_in(active_expert_indices)
    active_expert_tuple = tuple(active_expert_indices)
    hidden = self._forward_grouped_projection(
        grouped_input,
        weight=active_linear_in,
        m_splits=padded_splits,
        use_te_fp8=use_te_fp8,
        projection="linear_in",
        active_expert_indices=active_expert_tuple,
        offs=offs,
    )
    if not self.input_is_parallel:
        hidden = self._gather_along_last_dim(hidden)
    hidden = self.activation(hidden)

    if self.config.cpu_offloading and self.config.cpu_offloading_activations:
        hidden.activation_offloading = True
    active_linear_out = self.linear_out(active_expert_indices)
    expert_output = self._forward_grouped_projection(
        hidden,
        weight=active_linear_out,
        m_splits=padded_splits,
        use_te_fp8=use_te_fp8,
        projection="linear_out",
        active_expert_indices=active_expert_tuple,
        offs=offs,
    )
    if self.input_is_parallel:
        expert_output = self._gather_along_last_dim(expert_output)

    if self.dropout_position == "post":
        expert_output = self.dropout(expert_output)

    if all(pad_len == 0 for pad_len in pad_lengths):
        return expert_output * (self.alpha / self.dim)

    outputs = []
    for output_chunk, pad_len in zip(torch.split(expert_output, padded_splits, dim=0), pad_lengths):
        outputs.append(unpad_seq_to_mult(output_chunk, pad_len) if pad_len > 0 else output_chunk)

    return torch.cat(outputs, dim=0) * (self.alpha / self.dim)


def install_expert_split() -> None:
    """Install once on the existing class, including aliases imported by
    LoRA."""
    if getattr(GroupedExpertLinearAdapter.forward, "_relax_expert_split", False):
        return
    required = ("_forward_per_expert", "_forward_grouped_projection", "_is_te_fp8_enabled")
    if any(not hasattr(GroupedExpertLinearAdapter, name) for name in required):
        raise RuntimeError("K3 expert split requires Bridge's grouped-projection adapter implementation")
    forward._relax_expert_split = True
    GroupedExpertLinearAdapter._forward_per_expert = _forward_per_expert
    GroupedExpertLinearAdapter.forward = forward
