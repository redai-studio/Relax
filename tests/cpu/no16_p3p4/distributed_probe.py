# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Two real CPU Gloo ranks, production iterator bodies, no GPU or model
forward."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from .support import iterator


def worker(rank: int, directory: str) -> None:
    import torch
    import torch.distributed as dist

    from relax.utils.data.flops_scheduling import runtime_context

    output = Path(directory)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
        init_method="file://" + str(output / "rendezvous"),
    )
    group = dist.group.WORLD
    assert dist.get_rank(group) == rank
    config = json.loads((Path(__file__).resolve().parents[1] / "no16_p2/configs/Qwen3.5-27B.json.txt").read_text())
    results = []
    try:
        for mode in ("off", "microbatch", "dp", "both"):
            args = SimpleNamespace(flops_scheduling=mode, enable_mtp_training=False, use_dynamic_batch_size=True)
            active = runtime_context(args, config)
            batches, k, reductions, _ = iterator(
                [6, 6, 6] if rank == 0 else [9, 2, 1],
                active,
                dp=2,
                vpp=2,
                group_size=2,
                real_torch=torch,
                process_group=group,
            )
            assert k == [4]
            assert sorted(i for b in batches[:4] if not b.get("__is_dummy__", False) for i in b["ids"]) == [0, 1, 2]
            results.append(dict(mode=mode, K=k, collective_inputs=reductions, coverage=True))
        for mode in ("dp", "both"):
            args.flops_scheduling = mode
            active = runtime_context(args, config)
            batches, k, reductions, _ = iterator(
                [6, 6, 6] if rank == 0 else [9, 2, 1],
                active,
                dp=2,
                vpp=2,
                group_size=2,
                real_torch=torch,
                process_group=group,
                balance_data=True,
            )
            assert k == [4] and [len(values) for values in reductions] == [5, 1, 1]
            assert sorted(i for b in batches[:4] if not b.get("__is_dummy__", False) for i in b["ids"]) == [0, 1, 2]
            results.append(dict(path="balanced_backend", mode=mode, K=k, collective_inputs=reductions, coverage=True))
        args.flops_scheduling = "both"
        active = runtime_context(args, config)
        for damage in ("one_rank_bad_length", "different_mini_counts"):
            try:
                if damage == "one_rank_bad_length":
                    iterator(
                        [0, 1] if rank == 0 else [1, 1],
                        active,
                        dp=2,
                        real_torch=torch,
                        process_group=group,
                        global_batch_size=4,
                    )
                else:
                    iterator(
                        [1, 1],
                        active,
                        dp=2,
                        counts=[2] if rank == 0 else [1, 1],
                        real_torch=torch,
                        process_group=group,
                        global_batch_size=4,
                    )
            except ValueError as error:
                results.append(dict(damage=damage, rejected=True, message=str(error)))
            else:
                raise AssertionError("Every rank must reject inconsistent metadata before K")
        (output / f"rank{rank}.json").write_text(
            json.dumps(dict(rank=rank, backend="CPU Gloo", results=results), indent=2) + "\n"
        )
    finally:
        dist.destroy_process_group(group)


def main() -> None:
    import torch.multiprocessing as mp

    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    mp.spawn(worker, args=(str(output),), nprocs=2, join=True)
    (output / "summary.json").write_text(
        json.dumps(
            dict(
                ranks=2,
                modes=4,
                balanced_backend_modes=2,
                negative_cases=2,
                all_passed=True,
                scope="Real CPU Gloo and production planning; modeled VPP only, no GPU/PP/TP/EP execution or gradients",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
