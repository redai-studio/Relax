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
    # Preserve one gate invocation and one complete router-logit record per layer.
    _, all_weights, all_indices = self.gate(hidden_states.reshape(batch * length, hidden))
    shared_rows = []
    for row in range(batch):
        shared_chunks = []
        for block in blocks:
            chunk = block[row]
            shared_chunks.append(torch.sigmoid(self.shared_expert_gate(chunk)) * self.shared_expert(chunk))
        shared_rows.append(torch.cat(shared_chunks, dim=0)[left : left + length])
    expert = self.experts(hidden_states.reshape(-1, hidden), all_indices, all_weights)
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


def validate_token_chunk_sharding(model, distributed) -> None:
    """Reject independent sharding of children called a rank-local number of
    times."""
    from fsdp_turbo.utils.str_match import module_name_match

    fsdp_plan = distributed.fsdp_plan
    efsdp_patterns = (
        distributed.ep_plan.apply_efsdp_modules
        if distributed.expert_parallel_size > 1 and distributed.expert_fully_shard_parallel_size > 1
        else ()
    )
    repeated = set()
    for module in model.modules():
        name = type(module).__name__
        if name == "Qwen4ExpTextGatedResidual":
            repeated.update(child for child in module.modules() if child is not module)
        elif name == "Qwen4ExpTextSparseMoeBlock":
            repeated.update(module.shared_expert.modules())
            repeated.update(module.shared_expert_gate.modules())
    for name, module in model.named_modules():
        if module not in repeated:
            continue
        # E-FSDP has its own selection; ordinary FSDP ignored_modules do not exclude it.
        if any(module_name_match(p, name) for p in efsdp_patterns):
            raise ValueError(f"Token chunks cannot independently E-FSDP shard repeated submodule {name!r}")
        if any(module_name_match(p, name) for p in fsdp_plan.ignored_modules):
            continue
        if any(module_name_match(p, name) for p in fsdp_plan.apply_modules):
            raise ValueError(f"Token chunks cannot independently shard repeated submodule {name!r}")
