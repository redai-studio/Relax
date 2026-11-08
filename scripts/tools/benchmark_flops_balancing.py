# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare CPU scheduling plans; this does not measure training throughput."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from relax.utils.data.flops_balancing import guarded_dp_assignment, partition_by_flops, summarize_partitions
from relax.utils.data.packing import get_capacity_safe_balanced_partitions, get_microbatch_count
from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions
from relax.utils.training.flops_counter import SampleFlopsCost


def compare_schedules(config: Any, lengths: list[int], dp_size: int, token_capacity: int) -> dict[str, Any]:
    """Compare the actual token-only DP baseline with FLOPs-based
    assignment."""
    cost_model = SampleFlopsCost.from_config(config)
    costs = cost_model.estimate(lengths)
    baseline = get_seqlen_balanced_partitions(lengths, dp_size, equal_size=True)
    started = perf_counter()
    identities = [i for rank in baseline for i in rank]
    positions = {identity: position for position, identity in enumerate(identities)}
    before = [[positions[i] for i in rank] for rank in baseline]
    transformed, decision = guarded_dp_assignment(
        [lengths[i] for i in identities],
        [costs[i] for i in identities],
        before,
        token_capacity,
    )
    proposed = [[identities[i] for i in rank] for rank in transformed]
    target = max(get_microbatch_count([lengths[i] for i in rank], token_capacity) for rank in proposed)
    plans = [
        partition_by_flops(
            [lengths[i] for i in rank],
            [costs[i] for i in rank],
            token_capacity,
            get_capacity_safe_balanced_partitions([lengths[i] for i in rank], target, token_capacity),
        )
        for rank in proposed
    ]
    elapsed = perf_counter() - started
    return {
        "measurement": "CPU scheduling mechanism; not GPU training throughput",
        "sample_count": len(lengths),
        "dp_size": dp_size,
        "token_capacity": token_capacity,
        "token_only": summarize_partitions(lengths, costs, baseline),
        "flops_aware": summarize_partitions(lengths, costs, proposed),
        "local_microbatch_counts": [len(plan) for plan in plans],
        "local_plans": [
            summarize_partitions([lengths[i] for i in rank], [costs[i] for i in rank], plan)
            for rank, plan in zip(proposed, plans)
        ],
        "scheduler_seconds": elapsed,
        "dp_guardrail": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True, help="Hugging Face config.json; no weights loaded")
    parser.add_argument("--lengths", type=Path, required=True, help="JSON list of token lengths from a fixed batch")
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--token-capacity", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.model_config.read_text(), object_hook=lambda values: SimpleNamespace(**values))
    report = compare_schedules(config, json.loads(args.lengths.read_text()), args.dp_size, args.token_capacity)
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
