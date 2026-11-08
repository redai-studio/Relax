# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real CPU Gloo fetch-lifecycle regressions; no queue service or GPU
training."""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from no16.source import ROOT, load

from relax.utils.data.global_flops_pack_protocol import PackFirstExhausted, agree_cpu, checked_fetch


def worker(rank, world, directory):
    import torch.distributed as dist

    out = Path(directory)
    records = []
    try:
        dist.init_process_group(
            "gloo",
            init_method=(out / "rendezvous").as_uri(),
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=15),
        )
        control = dist.new_group(list(range(world)), backend="gloo", timeout=timedelta(seconds=15))
        tp = world // 2
        dp = rank // tp
        fetching = rank % tp == 0
        peers = [
            dist.new_group([d * tp + i for i in range(tp)], backend="gloo", timeout=timedelta(seconds=15))
            for d in range(2)
        ]
        cases = [
            "empty_done",
            "empty_live_recovers",
            "dp0_ready",
            "dp1_ready",
            "mixed_done",
            "fetch_error_dp0",
            "fetch_error_dp1",
            "status_error_dp0",
            "status_error_dp1",
            "timeout",
        ]
        for name in cases:
            trace = []
            fetch_calls = []
            status_calls = []
            rounds = [0]
            payload = {"dp": dp, "frozen_rows": [100 + dp]}

            def agree(values):
                trace.append(list(values))
                if len(values) == 5:
                    rounds[0] += 1
                return agree_cpu(values, control)

            def fetch():
                fetch_calls.append(1)
                if name == "fetch_error_dp" + str(dp):
                    raise RuntimeError("injected queue RPC error")
                if name.startswith("status_error") or name in ("empty_done", "mixed_done", "timeout"):
                    return [None, None]
                ready = (
                    name in ("fetch_error_dp0", "fetch_error_dp1")
                    or rounds[0] >= 2
                    or name == "dp" + str(dp) + "_ready"
                )
                return [payload, "meta"] if ready else [None, None]

            def drained():
                status_calls.append(1)
                if name == "status_error_dp" + str(dp):
                    raise RuntimeError("injected consumed RPC error")
                if name == "mixed_done":
                    return dp == 0 or rounds[0] >= 2
                return name == "empty_done"

            topology = SimpleNamespace(
                get_tensor_model_parallel_rank=lambda: rank % tp,
                get_pipeline_model_parallel_rank=lambda: 0,
                get_context_parallel_rank=lambda: 0,
            )
            raw = load(
                ROOT / "relax/backends/megatron/actor.py",
                class_name="MegatronTrainRayActor",
                names={"_pack_first_local_consumed"},
                injected=dict(
                    mpu=topology,
                    run=lambda value: value,
                    _agree_drained=lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Nested MP collective")),
                ),
            )
            actor = SimpleNamespace(
                data_system_client=SimpleNamespace(async_check_consumption_status=lambda task, pid: drained())
            )

            expected = (
                "end"
                if name in ("empty_done", "mixed_done")
                else (
                    "fetch"
                    if name.startswith("fetch_error")
                    else "status"
                    if name.startswith("status_error")
                    else "timeout"
                    if name == "timeout"
                    else "ready"
                )
            )
            observed = None
            message = None
            try:
                batch = checked_fetch(
                    fetch,
                    fetching,
                    agree,
                    is_drained=lambda: raw._pack_first_local_consumed(actor, "sft_train", 0, partition_id="p"),
                    timeout_seconds=0.05 if name == "timeout" else 3,
                    poll_seconds=0.01,
                )
                observed = "ready"
                if fetching:
                    assert batch[0] is payload
                objects = [batch[0]]
                dist.broadcast_object_list(objects, src=dp * tp, group=peers[dp])
                assert objects[0] == payload
            except PackFirstExhausted as exc:
                observed = "end"
                message = str(exc)
            except TimeoutError as exc:
                observed = "timeout"
                message = str(exc)
            except RuntimeError as exc:
                message = str(exc)
                if "Pack-first fetch failed" in message:
                    observed = "fetch"
                elif "Pack-first queue status failed" in message:
                    observed = "status"
                else:
                    raise
            assert observed == expected, (name, observed, message)
            if name in ("dp0_ready", "dp1_ready"):
                assert not status_calls  # Any ready batch suppresses drained checks on all ranks.
                if fetching:
                    assert len(fetch_calls) == (1 if name == f"dp{dp}_ready" else 3)
            if name == "empty_done":
                assert len(status_calls) == (1 if fetching else 0) and (
                    len(fetch_calls) == 1 if fetching else not fetch_calls
                )
            if name == "empty_live_recovers":
                assert len(status_calls) == (2 if fetching else 0)
            records.append(
                dict(
                    case=name,
                    outcome=observed,
                    message=message,
                    fetches=len(fetch_calls),
                    status_checks=len(status_calls),
                    collective_shapes=[len(v) for v in trace],
                    inputs=trace,
                )
            )
        (out / f"rank{rank}.json").write_text(json.dumps(dict(rank=rank, records=records), indent=2), encoding="utf-8")
        dist.destroy_process_group()
    except BaseException:
        (out / f"rank{rank}_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    out = Path(sys.argv[1]).resolve()
    world = int(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    if world not in (2, 4):
        raise ValueError("Use DP2 or TP2/DP2")
    context = mp.get_context("spawn")
    processes = [context.Process(target=worker, args=(r, world, str(out))) for r in range(world)]
    started = time.monotonic()
    for p in processes:
        p.start()
    try:
        for p in processes:
            p.join(max(0, 90 - (time.monotonic() - started)))
        assert all(not p.is_alive() and p.exitcode == 0 for p in processes), [p.exitcode for p in processes]
        rows = [json.loads((out / f"rank{r}.json").read_text()) for r in range(world)]
        expected = [(c["case"], c["outcome"], c["collective_shapes"]) for c in rows[0]["records"]]
        assert all(
            [(c["case"], c["outcome"], c["collective_shapes"]) for c in row["records"]] == expected for row in rows
        )
        summary = dict(
            ranks=world,
            cases_per_rank=len(expected),
            all_passed=True,
            collective_orders_equal=True,
            seconds=time.monotonic() - started,
            scope="Real CPU Gloo only; queue/status RPCs are test seams",
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary))
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(5)


if __name__ == "__main__":
    main()
