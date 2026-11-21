# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real CPU Gloo, explicit Actor control groups and bounded process cleanup.

Run with the existing Torch environment. TQ selection uses pinned source, not a
running queue service; no CUDA or Megatron model is initialized.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from relax.utils.data.global_flops_pack_protocol import (
    STEP_PLANS,
    agree_cpu,
    agree_invocation,
    checked_drained,
    checked_fetch,
    checked_schedule,
    seal_header,
)

from .support import backend, concat, partition, pull, restore, setup


def worker(rank: int, world: int, directory: str) -> None:
    import torch
    import torch.distributed as dist

    out = Path(directory)
    records = []
    try:
        dist.init_process_group(
            "gloo",
            init_method=(out / "rendezvous").as_uri(),
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=12),
        )
        control = dist.new_group(list(range(world)), backend="gloo", timeout=timedelta(seconds=12))
        tp = 1 if world == 2 else 2
        dp_rank = rank // tp
        peers = [
            dist.new_group([d * tp + i for i in range(tp)], backend="gloo", timeout=timedelta(seconds=12))
            for d in range(2)
        ]

        def run(name, action, expected_error=False):
            trace = []

            def agree(values):
                trace.append(list(values))
                return agree_cpu(values, control)

            error = None
            try:
                action(agree)
            except (ValueError, RuntimeError, TimeoutError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            assert bool(error) == expected_error, (name, error, expected_error)
            records.append(
                dict(
                    case=name,
                    expected_error=expected_error,
                    error=error,
                    collective_shapes=[len(values) for values in trace],
                    inputs=trace,
                )
            )

        for kind in ("sft", "rl"):
            args, ctx, contract, sampler, _ = setup(out / "config", kind, capacity=10, tp=tp, global_batch_size=4)
            ctx = replace(ctx, control_group=control)
            task = "sft_train" if kind == "sft" else "train"
            part = partition([6, 6, 6, 1])
            rows, meta = pull(sampler, part, contract, dp_rank, task=task)
            payload = restore(part, rows, meta, ctx, contract, dp_rank, task=task)
            plan = sampler.plans[("p", task, 0)]
            assert sorted(map(len, plan.rank_mbs)) == [1, 2] and plan.K == 2

            def normal(agree):
                indices, K, dummy = checked_schedule(payload, contract, ctx.estimator, dp_rank, agree)
                assert K == [2] and len(dummy) == 2 - len(plan.rank_mbs[dp_rank])
                actual, counts, _ = backend(args, ctx, contract, dp_rank, payload, group=control, torch=torch)
                assert counts == K
                assert sorted(row for b in actual if not b.get("__is_dummy__") for row in b["ids"]) == sorted(rows)
                assert len(indices) == 2

            run(kind + "_unequal_real_mbs", normal)
            for bad_rank in (0, 1):
                for fault in ("missing", "mapping", "length", "digest", "K", "stale", "metadata", "capacity"):
                    bad = copy.deepcopy(payload)
                    if dp_rank == bad_rank:
                        h = bad[STEP_PLANS][0]["header"]
                        if fault == "missing":
                            bad.pop(STEP_PLANS)
                        elif fault == "mapping":
                            bad[STEP_PLANS][0]["payload_rows"][0] = 999
                        elif fault == "length":
                            bad["total_lengths"][0] += 1
                        elif fault == "digest":
                            h["global_plan_digest"] = "f" * 64
                        elif fault == "K":
                            h["K"] += 1
                            h["slots"].append({**copy.deepcopy(h["slots"][0]), "dummy": True})
                        elif fault == "stale":
                            h["identity"][3] = -1
                        elif fault == "metadata":
                            h["model_config_signature"] = "wrong"
                        else:
                            h["slots"][0]["physical_tokens"] = 999
                        if STEP_PLANS in bad:
                            bad[STEP_PLANS][0]["header"] = seal_header(h)
                    run(
                        f"{kind}_{fault}_rank{bad_rank}",
                        lambda agree, bad=bad: checked_schedule(bad, contract, ctx.estimator, dp_rank, agree),
                        True,
                    )
            # Every model-parallel partner participates even though only TP0 fetches.
            for bad_rank in (0, 1):

                def fail_fetch(agree):
                    def fetch():
                        if dp_rank == bad_rank:
                            raise RuntimeError("injected metadata publication / TQ fetch error")
                        return [payload, meta]

                    checked_fetch(fetch, rank % tp == 0, agree, timeout_seconds=2, poll_seconds=0)

                run(f"{kind}_fetch_error_dp{bad_rank}", fail_fetch, True)
            polls = [0]

            def ready_fetch(agree):
                def fetch():
                    polls[0] += 1
                    return [None, None] if dp_rank == 1 and polls[0] < 3 else [payload, meta]

                data = checked_fetch(fetch, rank % tp == 0, agree, timeout_seconds=2, poll_seconds=0)
                objects = [data[0]]
                dist.broadcast_object_list(objects, src=dp_rank * tp, group=peers[dp_rank])
                assert objects[0][STEP_PLANS] == payload[STEP_PLANS]

            run(kind + "_ready_then_tp_broadcast", ready_fetch)

            def timeout(agree):
                checked_fetch(lambda: [None, None], rank % tp == 0, agree, timeout_seconds=0, poll_seconds=0)

            run(kind + "_bounded_readiness_timeout", timeout, True)
        # Multi-row identities, two separate minis and per-mini VPP padding.
        args, ctx, contract, sampler, _ = setup(out / "config", "rl", capacity=6, tp=tp, vpp=2, mb_group=2)
        ctx = replace(ctx, control_group=control)
        part = partition([4, 4, 4, 1, 1, 1, 1, 1, 3, 2, 2, 1, 1], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7])
        minis = []
        for mini in range(2):
            rows, meta = pull(sampler, part, contract, dp_rank, mini=mini, task="train")
            minis.append(restore(part, rows, meta, ctx, contract, dp_rank, mini=mini, task="train"))
        merged = concat(minis)

        def multi(agree):
            _, counts, _ = checked_schedule(merged, contract, ctx.estimator, dp_rank, agree)
            assert len(counts) == 2 and all(k % 2 == 0 for k in counts)
            assert merged["rollout_mini_global_sample_counts"] == [4, 4]

        run("rl_two_mini_vpp", multi)
        for bad_rank in (0, 1):
            bad = copy.deepcopy(merged)
            if dp_rank == bad_rank:
                bad = copy.deepcopy(minis[0])
            run(
                f"rl_mini_count_rank{bad_rank}",
                lambda agree, bad=bad: checked_schedule(bad, contract, ctx.estimator, dp_rank, agree),
                True,
            )
            run(
                f"invocation_count_rank{bad_rank}",
                lambda agree: agree_invocation(1 if dp_rank == bad_rank else 2, 0, agree),
                True,
            )

            def bad_status(agree):
                def check():
                    if dp_rank == bad_rank:
                        raise RuntimeError("injected all_consumed RPC error")
                    return False

                checked_drained(check, agree)

            run(f"queue_status_rank{bad_rank}", bad_status, True)
        (out / f"rank{rank}.json").write_text(
            json.dumps(dict(rank=rank, world=world, records=records), indent=2), encoding="utf-8"
        )
        dist.destroy_process_group()
    except BaseException:
        (out / f"rank{rank}_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "config").mkdir()
    world = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if world not in (2, 4):
        raise ValueError("Use DP2 or orthogonal TP2/DP2")
    context = mp.get_context("spawn")
    processes = [context.Process(target=worker, args=(rank, world, str(out))) for rank in range(world)]
    started = time.monotonic()
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(max(0, 90 - (time.monotonic() - started)))
        alive = [process for process in processes if process.is_alive()]
        assert not alive, "Bounded Gloo probe timed out"
        assert all(process.exitcode == 0 for process in processes), [p.exitcode for p in processes]
        records = [json.loads((out / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(world)]
        assert all(
            [x["case"] for x in record["records"]] == [x["case"] for x in records[0]["records"]] for record in records
        )
        summary = dict(
            ranks=world,
            DP=2,
            TP=world // 2,
            cases_per_rank=len(records[0]["records"]),
            all_passed=True,
            seconds=time.monotonic() - started,
            scope="Real CPU Gloo; source-contract TQ; no NCCL/model training",
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(5)


if __name__ == "__main__":
    main()
