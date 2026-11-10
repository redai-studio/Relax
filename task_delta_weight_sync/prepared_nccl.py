# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepared-data NCCL transport experiment, separate from weight
installation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from time import monotonic
from typing import Any


def wire_bucket(named: list) -> list:
    """Run-length encode the actual tensor order, without retaining weights."""
    runs = []
    for _, tensor in named:
        key = [str(tensor.dtype).removeprefix("torch."), tensor.numel()]
        if runs and runs[-1][:2] == key:
            runs[-1][2] += 1
        else:
            runs.append([*key, 1])
    return runs


def plan_capacity(plan: list) -> tuple[dict, dict]:
    capacities, totals = {}, {}
    if not isinstance(plan, list) or not plan:
        raise ValueError("A real native wire plan is required")
    for bucket in plan:
        local = {}
        for dtype, elements, repeat in bucket:
            if (
                dtype not in {"bfloat16", "float32"}
                or type(elements) is not int
                or elements <= 0
                or type(repeat) is not int
                or repeat <= 0
            ):
                raise ValueError("Invalid native wire operation")
            local[dtype] = local.get(dtype, 0) + elements * repeat
        for dtype, elements in local.items():
            capacities[dtype] = max(capacities.get(dtype, 0), elements)
            totals[dtype] = totals.get(dtype, 0) + elements * (2 if dtype == "bfloat16" else 4)
    if sum(elements * (2 if dtype == "bfloat16" else 4) for dtype, elements in capacities.items()) > 2 << 30:
        raise ValueError("Prepared scratch exceeds the bounded two-GiB budget")
    return capacities, totals


def prepare(plan: list, device: Any, sample: Any = None) -> dict:
    import torch

    capacities, totals = plan_capacity(plan)
    buffers = {
        dtype: torch.empty(elements, dtype=getattr(torch, dtype), device=device)
        for dtype, elements in capacities.items()
    }
    for buffer in buffers.values():
        if sample is None:
            buffer.zero_()
        else:
            buffer.copy_(sample.reshape(-1)[:1].to(device=buffer.device, dtype=buffer.dtype).expand_as(buffer))
    views = []
    for bucket in plan:
        local, offset = [], {dtype: 0 for dtype in buffers}
        for dtype, elements, repeat in bucket:
            for _ in range(repeat):
                local.append(buffers[dtype][offset[dtype] : offset[dtype] + elements])
                offset[dtype] += elements
        views.append(local)
    torch.cuda.synchronize(device)
    return dict(
        buffers=buffers,
        views=views,
        dtype_bytes=totals,
        scratch_bytes=sum(t.numel() * t.element_size() for t in buffers.values()),
    )


def replay(state: dict, group: Any) -> dict:
    import torch
    import torch.distributed as dist

    device = next(iter(state["buffers"].values())).device
    ready = torch.ones(1, dtype=torch.int32, device=device)
    dist.all_reduce(ready, group=group)
    torch.cuda.synchronize(device)  # Ready rendezvous and all preparation excluded.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = monotonic()
    start.record()
    for bucket in state["views"]:
        handles = [dist.broadcast(tensor, src=0, group=group, async_op=True) for tensor in bucket]
        for handle in handles:
            handle.wait()  # Reusing the scratch follows all preceding writes.
    end.record()
    end.synchronize()
    return dict(
        cuda_event_s=start.elapsed_time(end) / 1000,
        local_wall_s=monotonic() - started,
        group_rank=dist.get_rank(group),
        group_world_size=dist.get_world_size(group),
        buckets=len(state["views"]),
        collectives=sum(map(len, state["views"])),
        dtype_bytes=state["dtype_bytes"],
        bytes=sum(state["dtype_bytes"].values()),
        scratch_bytes=state["scratch_bytes"],
        data_reused=True,
        limitation="Prepared repeated frozen-weight sample; original dtype/message/order/bytes; no model apply; Python enqueue gaps included",
    )


def install_worker() -> None:
    from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater

    from relax.distributed.weight_sync.provenance import runtime_identity
    from relax.distributed.weight_sync.store import atomic_write
    from task_delta_weight_sync.recovery_probe import scheduler_state
    from task_delta_weight_sync.sharded_gpu_audit import AUDIT_LOADER

    if getattr(WeightUpdater, "_relax_prepared_transport", False):
        raise RuntimeError("Prepared transport wrapper is already installed")
    original = WeightUpdater.update_weights_from_tensor
    prepared_state = None
    prepared_owner = None

    def wrapped(self, named_tensors, load_format=None):
        nonlocal prepared_state, prepared_owner
        if load_format != AUDIT_LOADER:
            return original(self, named_tensors, load_format=load_format)
        if len(named_tensors) != 1:
            return original(self, named_tensors, load_format=load_format)
        encoded, payload = named_tensors[0]
        if not isinstance(encoded, str) or len(encoded) > 65536 or payload.numel() != 0:
            raise ValueError("Prepared transport requires a bounded empty-tensor control")
        control = json.loads(encoded)
        command = control.get("command")
        if command not in {"prepare_transfer", "run_transfer", "close_transfer"}:
            return original(self, named_tensors, load_format=load_format)
        directory = Path(control["directory"]).resolve()
        if (
            not directory.is_relative_to(Path(os.environ["TMPDIR"]).resolve().parent)
            or control.get("paused") is not True
            or not scheduler_state()["paused"]
            or any(
                not isinstance(control.get(key), str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,180}", control[key]) is None
                for key in ("member", "phase", "update_id")
            )
        ):
            raise ValueError("Transport experiment requires a paused owned run")
        group = self._model_update_group[control["group_name"]]
        if command == "prepare_transfer":
            if prepared_state is not None:
                raise RuntimeError("Prepared scratch is already allocated")
            # SGLang's WeightUpdater is a frozen slots dataclass. Keep scratch
            # in this worker-local wrapper rather than mutating its instance.
            prepared_state = prepare(control["plan"], self.device)
            prepared_owner = self
        elif command == "run_transfer":
            if prepared_state is None or prepared_owner is not self:
                raise RuntimeError("Prepared scratch belongs to another updater or is not allocated")
            if type(control.get("round")) is not int or not 0 <= control["round"] < 4:
                raise ValueError("Invalid prepared transfer round")
            report = replay(prepared_state, group)
            report.update(identity=runtime_identity(), command=command, round=control["round"])
            atomic_write(
                directory / control["member"] / f"prepared-{control['round']}.json",
                json.dumps(report, indent=2).encode(),
            )
        else:
            if prepared_state is None or prepared_owner is not self:
                raise RuntimeError("Prepared scratch belongs to another updater or is not allocated")
            prepared_state = None
            prepared_owner = None
            WeightUpdater.update_weights_from_tensor = original
            del WeightUpdater._relax_prepared_transport
        return True, "Prepared transport control completed"

    WeightUpdater.update_weights_from_tensor = wrapped
    WeightUpdater._relax_prepared_transport = True


def run(self: Any, directory: Path) -> None:
    """Called only on native source rank, under the independent fail-stop
    guard."""
    import ray
    import torch

    from relax.distributed.weight_sync.provenance import runtime_identity
    from relax.distributed.weight_sync.store import atomic_write
    from task_delta_weight_sync.recovery_probe import pause_schedulers

    engines, native, plan = self._ab_engines, self._ab_native, self._ab_wire_plan
    _, totals = plan_capacity(plan)
    if sum(totals.values()) != 70214365792:
        raise ValueError("Prepared replay differs from the actual 35B native tensor stream")
    pause_schedulers(engines)

    def dispatch(command, round_no=None):
        control = dict(
            command=command,
            phase=command,
            update_id="prepared-transport",
            paused=True,
            directory=str(directory),
            node_budget_bytes=384 << 30,
            group_name=native._group_name,
        )
        if command == "prepare_transfer":
            control["plan"] = plan
        if round_no is not None:
            control["round"] = round_no
        if len(json.dumps(dict(control, member="f" * 32 + ":tp:0", tp_rank=0))) > 65536:
            raise ValueError("Native wire plan does not fit a bounded audit control")
        return [engine.run_memory_audit.remote(control, 600) for engine in engines]

    ray.get(dispatch("install_transfer"), timeout=610)
    sample = next(native.model[0].parameters()).detach()
    source = prepare(plan, torch.cuda.current_device(), sample)
    ray.get(dispatch("prepare_transfer"), timeout=610)
    reports = []
    for round_no in range(4):  # Warm-up separate from three observations.
        started = monotonic()
        refs = dispatch("run_transfer", round_no)
        report = replay(source, native._model_update_groups)
        ray.get(refs, timeout=610)
        report.update(
            control_all_receivers_wall_s=monotonic() - started,
            identity=runtime_identity(),
            round=round_no,
            warmup=round_no == 0,
        )
        receivers = [
            json.loads((directory / f"{engine._actor_id.hex()}:tp:{tp}" / f"prepared-{round_no}.json").read_text())
            for engine in engines
            for tp in range(2)
        ]
        if len({row["group_rank"] for row in [report, *receivers]}) != 9 or any(
            row["bytes"] != 70214365792 or row["group_world_size"] != 9 for row in [report, *receivers]
        ):
            raise RuntimeError("Incomplete nine-rank prepared transfer")
        report.update(
            receivers=receivers, max_rank_cuda_event_s=max(row["cuda_event_s"] for row in [report, *receivers])
        )
        reports.append(report)
    ray.get(dispatch("close_transfer"), timeout=610)
    del source
    ray.get([engine.continue_generation.remote(timeout=30) for engine in engines], timeout=40)
    ray.get([engine.disk_weight_generation_probe.remote(30) for engine in engines], timeout=40)
    atomic_write(
        directory / "prepared-nccl-verified.json",
        json.dumps(dict(status="VERIFIED", plan=plan, rounds=reports), indent=2).encode(),
    )
