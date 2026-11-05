# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare length/FLOPs packing on CPU with a local config, without weights."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from relax.utils.data.workload_balancing import pack_by_workload
from relax.utils.training.flops_counter import SequenceFlopsEstimator


def compare_capacity_partitions(
    config: Any, lengths: list[int], max_tokens: int, alignment: int = 1
) -> dict[str, Any]:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    estimator = SequenceFlopsEstimator.from_config(config)
    workloads = estimator.estimate_per_sample(lengths)
    capacity_lengths = [((length + alignment - 1) // alignment) * alignment for length in lengths]
    results: dict[str, Any] = {
        "scope": "CPU formula comparison only; no weights, GPU training or throughput measurement",
        "coefficients": {"linear": estimator.linear_flops, "quadratic": estimator.quadratic_flops},
        "max_tokens": max_tokens,
        "alignment": alignment,
    }
    for mode, weights in (("length_weighted", lengths), ("flops_weighted", workloads)):
        partitions = pack_by_workload(capacity_lengths, weights, max_tokens)
        results[mode] = {
            "partitions": [
                {
                    "sample_indices": partition,
                    "tokens": sum(lengths[index] for index in partition),
                    "capacity_tokens": sum(capacity_lengths[index] for index in partition),
                    "estimated_flops": sum(workloads[index] for index in partition),
                }
                for partition in partitions
            ],
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("qwen35_text_fixture.json"),
        help="Local HF config.json, or the bundled test fixture by default; weights are never loaded",
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024] * 5 + [4096])
    parser.add_argument("--max-tokens", type=int, default=5120)
    parser.add_argument("--alignment", type=int, default=1)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream, object_hook=lambda fields: SimpleNamespace(**fields))
    result = compare_capacity_partitions(config, args.lengths, args.max_tokens, args.alignment)
    result["config_source"] = str(args.config)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
