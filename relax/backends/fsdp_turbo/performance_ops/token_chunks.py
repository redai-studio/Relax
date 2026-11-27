# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep narrow GEMMs on globally aligned token blocks without changing
parameters."""

from types import MethodType

import torch
import torch.nn.functional as F


def _blocks(hidden_states, module):
    chunk_size = module._relax_token_chunk_size
    rank, size = module._relax_token_cp_rank, module._relax_token_cp_size
    length = hidden_states.shape[1]
    left = (rank * length) % chunk_size if size > 1 else 0
    right = (-(left + length)) % chunk_size
    padded = F.pad(hidden_states, (0, 0, left, right))
    return padded.split(chunk_size, dim=1), left, length


def _linear(self, hidden_states):
    blocks, left, length = _blocks(hidden_states, self)
    outputs = [F.linear(block, self.weight, self.bias) for block in blocks]
    return torch.cat(outputs, dim=1)[:, left : left + length]


def _qhc(self, hidden_states):
    blocks, left, length = _blocks(hidden_states, self)
    outputs = [self._relax_unchunked_forward(block) for block in blocks]
    if self.block_inject_weight is None:
        return torch.cat(outputs, dim=1)[:, left : left + length]
    mixed = torch.cat([out[0] for out in outputs], dim=1)[:, left : left + length]
    injection = torch.cat([out[2] for out in outputs], dim=1)[:, left : left + length]
    return mixed, hidden_states, injection


def _moe(self, hidden_states):
    batch, length, hidden = hidden_states.shape
    blocks, left, _ = _blocks(hidden_states, self)
    shared_rows, weight_rows, index_rows = [], [], []
    for row in range(batch):
        shared_chunks, weights, indices = [], [], []
        for block in blocks:
            chunk = block[row]
            shared_chunks.append(torch.sigmoid(self.shared_expert_gate(chunk)) * self.shared_expert(chunk))
            _, weight, index = self.gate(chunk)
            weights.append(weight)
            indices.append(index)
        shared_rows.append(torch.cat(shared_chunks, dim=0)[left : left + length])
        weight_rows.append(torch.cat(weights, dim=0)[left : left + length])
        index_rows.append(torch.cat(indices, dim=0)[left : left + length])
    weights = torch.stack(weight_rows).reshape(batch * length, -1)
    indices = torch.stack(index_rows).reshape(batch * length, -1)
    expert = self.experts(hidden_states.reshape(-1, hidden), indices, weights)
    return expert.reshape(batch, length, hidden) + torch.stack(shared_rows)


def install_token_chunks(model, chunk_size):
    counts = {"linear": 0, "qhc": 0, "moe": 0}
    for module in model.modules():
        name = type(module).__name__
        projections = ()
        if name == "Qwen4ExpTextAttention":
            projections = (module.k_proj, module.v_proj)
        elif name == "Qwen4ExpTextGatedDeltaNet":
            projections = (module.in_proj_a, module.in_proj_b)
        for projection in projections:
            projection._relax_token_chunk_size = chunk_size
            projection._relax_token_cp_rank = 0
            projection._relax_token_cp_size = 1
            projection.forward = MethodType(_linear, projection)
            counts["linear"] += 1
        if name == "Qwen4ExpTextGatedResidual":
            module._relax_unchunked_forward = module.forward
            module._relax_token_chunk_size = chunk_size
            module._relax_token_cp_rank = 0
            module._relax_token_cp_size = 1
            module.forward = MethodType(_qhc, module)
            counts["qhc"] += 1
        elif name == "Qwen4ExpTextSparseMoeBlock":
            module._relax_token_chunk_size = chunk_size
            module._relax_token_cp_rank = 0
            module._relax_token_cp_size = 1
            module.forward = MethodType(_moe, module)
            counts["moe"] += 1
    return counts


def initialize_token_distribution(turbo):
    from fsdp_turbo.distributed.context_parallel.ulysses.utils import get_ulysses_context

    _, rank, size = get_ulysses_context()
    for module in turbo.model.modules():
        if hasattr(module, "_relax_token_chunk_size"):
            module._relax_token_cp_rank = rank
            module._relax_token_cp_size = size
