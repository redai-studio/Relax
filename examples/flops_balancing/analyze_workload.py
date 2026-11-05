# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Inspect production CPU schedules on frozen real-data lengths, without GPUs.

FLOPs imbalance and scheduling time are diagnostics, not training speedups.
TP/CP padding is modeled; PP bubbles and expert routing/communication are not.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from transfer_queue import SeqlenBalancedSampler
from transfer_queue.controller import DataPartitionStatus

from relax.utils.data.data import get_minimum_num_micro_batch_size
from relax.utils.data.flops_sampler import FlopsBalancedSampler
from relax.utils.data.flops_schedule import build_workload_microbatch_plan, expand_workload_microbatch_plan
from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions
from relax.utils.data.streaming_dataset import IndexManager
from relax.utils.training.flops_counter import SequenceFlopsEstimator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    if min(args.dp, args.tp, args.cp, args.global_batch_size, args.max_tokens) <= 0:
        parser.error("Sizes and budget must be positive")
    if args.global_batch_size % args.dp:
        parser.error("global-batch-size must be divisible by dp")
    manifest = json.loads((args.workload / "manifest.json").read_text())
    config = json.loads((args.model / "config.json").read_text(), object_hook=lambda data: SimpleNamespace(**data))
    estimator = SequenceFlopsEstimator.from_config(config)
    lengths = manifest["total_lengths"]
    order = IndexManager(len(lengths), seed=1605)
    order.shuffle(0)
    report = {
        "scope": __doc__,
        "config": vars(args) | {"model": str(args.model), "workload": str(args.workload), "output": str(args.output)},
        "coefficients": vars(estimator),
        "modes": {},
    }
    for mode in "ABCD":
        rows = []
        for start in range(0, len(lengths) - args.global_batch_size + 1, args.global_batch_size):
            begin = time.perf_counter()
            batch_lengths = [lengths[index] for index in order.indices[start : start + args.global_batch_size]]
            partition = DataPartitionStatus(partition_id=f"sft_{start}")
            partition.set_custom_meta({index: {"total_lengths": length} for index, length in enumerate(batch_lengths)})
            sampler = (
                FlopsBalancedSampler(estimator, dp_size=args.dp)
                if mode in "CD"
                else SeqlenBalancedSampler(dp_size=args.dp)
            )
            placements = [
                sampler.sample(
                    list(range(len(batch_lengths))),
                    args.global_batch_size // args.dp,
                    "sft",
                    f"sft_{start}",
                    dp_rank=rank,
                    partition=partition,
                )[0]
                for rank in range(args.dp)
            ]
            assert sorted(index for group in placements for index in group) == list(range(args.global_batch_size))
            local_lengths = [[batch_lengths[index] for index in placement] for placement in placements]
            plans = []
            if mode in "BD":
                plans = [
                    build_workload_microbatch_plan(group, estimator, args.max_tokens, tp_size=args.tp, cp_size=args.cp)
                    for group in local_lengths
                ]
                count = max(len(plan.partitions) for plan in plans)
                plans = [expand_workload_microbatch_plan(plan, count) for plan in plans]
                groups = [plan.partitions for plan in plans]
            else:
                count = max(
                    get_minimum_num_micro_batch_size(group, args.max_tokens * args.cp) for group in local_lengths
                )
                groups = [
                    get_seqlen_balanced_partitions(group, min(count, len(group)), False) for group in local_lengths
                ]
                for group in groups:
                    group.extend([group[0][:] for _ in range(count - len(group))])
            work = [estimator.estimate(group) for group in local_lengths]
            violations = 0
            for group_lengths, microbatches in zip(local_lengths, groups, strict=True):
                for group in microbatches:
                    alignment = 2 * args.cp if args.cp > 1 else 1
                    total = sum((group_lengths[i] + alignment - 1) // alignment * alignment for i in group)
                    per_gpu = total // args.cp
                    padded = (per_gpu + args.tp * 128 - 1) // (args.tp * 128) * (args.tp * 128)
                    violations += padded > args.max_tokens
            rows.append(
                dict(
                    microbatches=count,
                    max_rank_flops_over_mean=max(work) / statistics.mean(work),
                    final_capacity_violations=violations,
                    planner_ms=(time.perf_counter() - begin) * 1000,
                )
            )
        report["modes"][mode] = dict(
            steps=rows,
            microbatches=sum(row["microbatches"] for row in rows),
            mean_rank_imbalance=statistics.mean(row["max_rank_flops_over_mean"] for row in rows),
            capacity_violations=sum(row["final_capacity_violations"] for row in rows),
            median_planner_ms=statistics.median(row["planner_ms"] for row in rows),
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                mode: {key: value for key, value in result.items() if key != "steps"}
                for mode, result in report["modes"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
