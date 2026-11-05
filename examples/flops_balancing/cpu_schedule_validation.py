# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Autograd validation on tiny Dense/MoE reference models (CPU or CUDA).

These models are not Qwen3.5 and do not validate Megatron/TP/PP/CP/EP kernels.
Run with torchrun --standalone --nproc-per-node=2 -m <this module> for real DP.
"""

import argparse
import json
import os
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from transfer_queue import SeqlenBalancedSampler
from transfer_queue.controller import DataPartitionStatus

from relax.utils.data.data import get_minimum_num_micro_batch_size
from relax.utils.data.flops_sampler import FlopsBalancedSampler
from relax.utils.data.flops_schedule import build_workload_microbatch_plan, expand_workload_microbatch_plan
from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions
from relax.utils.training.flops_counter import SequenceFlopsEstimator
from relax.utils.training.numerical_comparison import compare_named_tensors, compare_tensors


class TinyReferenceLM(nn.Module):
    """A causal attention block and Dense or top-2 MoE FFN, without dropout."""

    def __init__(self, moe: bool) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.attention = nn.MultiheadAttention(16, 2, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(16)
        self.moe = moe
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(16, 24), nn.GELU(), nn.Linear(24, 16)) for _ in range(4 if moe else 1)]
        )
        self.router = nn.Linear(16, 4) if moe else None
        self.head = nn.Linear(16, 32)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens)
        causal = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device).triu(1)
        attended, _ = self.attention(
            hidden, hidden, hidden, attn_mask=causal, key_padding_mask=~valid, need_weights=False
        )
        hidden = self.norm(hidden + attended)
        if self.router is None:
            update = self.experts[0](hidden)
        else:
            scores, indexes = self.router(hidden).topk(2, dim=-1)
            weights = torch.zeros(*hidden.shape[:-1], 4, device=hidden.device).scatter(
                -1, indexes, scores.softmax(dim=-1)
            )
            # Keep unselected experts in the graph; this does not test sparse EP kernels.
            outputs = torch.stack([expert(hidden) for expert in self.experts], dim=-2)
            update = (outputs * weights.unsqueeze(-1)).sum(dim=-2)
        return self.head(hidden + update)


def _update(
    initial: TinyReferenceLM,
    samples: list[torch.Tensor],
    groups: list[list[int]],
    dummy: set[int],
    *,
    dp_size: int,
    per_token: bool,
) -> dict[str, Any]:
    model = deepcopy(initial)
    trained = DistributedDataParallel(model) if dp_size > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    device = next(model.parameters()).device
    total_tokens = sum(len(tokens) - max(1, len(tokens) // 2) for tokens in samples)
    denominator = total_tokens if per_token else len(samples)
    reported_sum = 0.0
    for offset, indexes in enumerate(groups):
        lengths = [len(samples[index]) for index in indexes]
        tokens = torch.zeros(len(indexes), max(lengths), dtype=torch.long)
        valid = torch.zeros_like(tokens, dtype=torch.bool)
        supervised = torch.zeros(len(indexes), max(lengths) - 1, dtype=torch.bool)
        for row, index in enumerate(indexes):
            length = lengths[row]
            tokens[row, :length] = samples[index]
            valid[row, :length] = True
            supervised[row, max(1, length // 2) - 1 : length - 1] = True
        tokens, valid, supervised = tokens.to(device), valid.to(device), supervised.to(device)
        logits = trained(tokens, valid)
        losses = F.cross_entropy(logits[:, :-1].reshape(-1, 32), tokens[:, 1:].reshape(-1), reduction="none")
        losses = losses.reshape_as(supervised)
        sums = (losses * supervised).sum(dim=-1)
        objective = sums.sum() if per_token else (sums / supervised.sum(dim=-1)).sum()
        if offset in dummy:
            objective = objective * 0
        reported_sum += float(objective.detach())
        # Compensate for DDP's gradient averaging before global normalization.
        (objective * dp_size / denominator).backward()
    gradients = {key: value.grad.detach().clone() for key, value in model.named_parameters()}
    optimizer.step()
    updates = {key: value.detach() - before[key] for key, value in model.named_parameters()}
    loss_sum = torch.tensor(reported_sum, dtype=torch.float64, device=device)
    if dp_size > 1:
        dist.all_reduce(loss_sum)
    return {"loss": loss_sum / denominator, "gradients": gradients, "updates": updates}


def run_validation(group_size: int = 8, device: torch.device = torch.device("cpu")) -> dict[str, Any]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    lengths = [3, 4, 5, 6, 7, 8, 12, 15]
    if len(lengths) % world:
        raise ValueError("Reference fixture sample count must be divisible by DP size")
    generator = torch.Generator().manual_seed(1605)
    samples = [torch.randint(1, 32, (length,), generator=generator) for length in lengths]
    estimator = SequenceFlopsEstimator(10, 4)
    partition = DataPartitionStatus(partition_id="sft_0")
    partition.set_custom_meta({index: {"total_lengths": length} for index, length in enumerate(lengths)})
    report: dict[str, Any] = {
        "scope": "tiny reference models; not native Qwen3.5/Megatron validation",
        "device": str(device),
        "dp_size": world,
        "torch": torch.__version__,
        "cases": [],
    }
    for moe in (False, True):
        torch.manual_seed(16)
        initial = TinyReferenceLM(moe).to(device)
        for per_token in (False, True):
            oracle = _update(
                initial, samples, [[index] for index in range(len(samples))], set(), dp_size=1, per_token=per_token
            )
            for mode in ("A", "B", "C", "D"):
                sampler = (
                    FlopsBalancedSampler(estimator, dp_size=world)
                    if mode in ("C", "D")
                    else SeqlenBalancedSampler(dp_size=world)
                )
                indexes, _ = sampler.sample(
                    list(range(len(samples))), len(samples) // world, "sft", "sft_0", dp_rank=rank, partition=partition
                )
                local_lengths = [lengths[index] for index in indexes]
                if mode in ("B", "D"):
                    local_plan = build_workload_microbatch_plan(local_lengths, estimator, 16, pad_multiplier=1)
                    count = len(local_plan.partitions)
                else:
                    count = get_minimum_num_micro_batch_size(local_lengths, 16)
                count_tensor = torch.tensor(count)
                if world > 1:
                    dist.all_reduce(count_tensor, op=dist.ReduceOp.MAX)
                slots = (int(count_tensor) + group_size - 1) // group_size * group_size
                if mode in ("B", "D"):
                    plan = expand_workload_microbatch_plan(local_plan, slots)
                    groups, dummy = plan.partitions, set(plan.dummy_offsets)
                else:
                    groups = get_seqlen_balanced_partitions(local_lengths, min(slots, len(indexes)), False)
                    dummy = set(range(len(groups), slots))
                    shortest = min(groups, key=lambda group: sum(local_lengths[i] for i in group))
                    groups.extend([shortest[:] for _ in dummy])
                global_groups = [[indexes[i] for i in group] for group in groups]
                local_real = [i for offset, group in enumerate(global_groups) if offset not in dummy for i in group]
                all_real = [None] * world
                if world > 1:
                    dist.all_gather_object(all_real, local_real)
                else:
                    all_real = [local_real]
                assert sorted(index for group in all_real for index in group) == list(range(len(samples)))
                result = _update(initial, samples, global_groups, dummy, dp_size=world, per_token=per_token)
                comparisons = {
                    "loss": compare_tensors(oracle["loss"], result["loss"], atol=1e-6, rtol=1e-5),
                    "gradients": compare_named_tensors(oracle["gradients"], result["gradients"], atol=1e-6, rtol=1e-5),
                    "fp32_updates": compare_named_tensors(oracle["updates"], result["updates"], atol=1e-7, rtol=1e-4),
                }
                case = {
                    "model": "tiny_moe" if moe else "tiny_dense",
                    "mode": mode,
                    "normalization": "token" if per_token else "sample",
                    "microbatches": slots,
                    "local_dummy_count": len(dummy),
                    "comparison": comparisons,
                    "passed": all(item["passed"] for item in comparisons.values()),
                }
                report["cases"].append(case)
                if not case["passed"]:
                    raise AssertionError(json.dumps(case))
    report["passed"] = all(case["passed"] for case in report["cases"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--microbatch-group-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.microbatch_group_size <= 0:
        parser.error("microbatch-group-size must be positive")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < int(os.environ.get("LOCAL_WORLD_SIZE", "1")):
            parser.error("CUDA validation requires a visible GPU for every local process")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.backends.cuda.matmul.allow_tf32 = False
    if distributed:
        dist.init_process_group("nccl" if args.device == "cuda" else "gloo", timeout=timedelta(seconds=120))
    try:
        report = run_validation(args.microbatch_group_size, device)
        if not distributed or dist.get_rank() == 0:
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "passed": report["passed"],
                        "cases": len(report["cases"]),
                        "dp_size": report["dp_size"],
                        "scope": report["scope"],
                    }
                )
            )
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
