# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Separate diagnostic run: record executed packs and optimizer boundary
times."""

import copy
import json
import os
import time
from pathlib import Path
from types import MethodType
from typing import Any

from scripts.tools.flops_benchmark.capture import _fence


class _Audit:
    def __init__(self, args: Any, optimizer: Any) -> None:
        import torch.distributed as dist
        from megatron.core import mpu

        from relax.backends.megatron.data import DataIterator
        from relax.utils.data.flops_balancing import get_dp_packing_geometry

        self.rank = dist.get_rank(dist.group.WORLD)
        self.coordinates = {
            "rank": self.rank,
            "dp_rank": mpu.get_data_parallel_rank(),
            "tp_rank": mpu.get_tensor_model_parallel_rank(),
            "pp_rank": mpu.get_pipeline_model_parallel_rank(),
            "cp_rank": mpu.get_context_parallel_rank(),
        }
        geometry_args = copy.copy(args)
        geometry_args.balance_by_flops = True
        self.geometry = get_dp_packing_geometry(geometry_args, "train")
        cp = mpu.get_context_parallel_world_size()
        pad = mpu.get_tensor_model_parallel_world_size() * args.data_pad_size_multiplier
        directory = Path(os.environ["RELAX_FLOPS_AUDIT_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"rank-{self.rank:05d}.jsonl"
        if self.path.exists():
            raise FileExistsError(self.path)
        self.rollout_id, self.step_id = -1, -1
        self.start = 0.0
        self.iterators: dict[int, int] = {}
        original_next, original_step = DataIterator.get_next, optimizer.step

        def next_wrapper(iterator: Any, keys: Any) -> dict:
            offset = iterator.offset
            result = original_next(iterator, keys)
            identifier = self.iterators.setdefault(id(iterator), len(self.iterators))
            indices = iterator.micro_batch_indices[offset]
            lengths = [iterator.rollout_data["total_lengths"][i] for i in indices]
            alignment = self.geometry["sequence_alignment"]
            tokens = sum((length + alignment - 1) // alignment * alignment for length in lengths)
            oversized = tokens > self.geometry["capacity"]
            if oversized and len(indices) != 1:
                raise ValueError("Executed microbatch exceeds token capacity")
            self.write(
                {
                    "kind": "microbatch",
                    "iterator": identifier,
                    "offset": offset,
                    "indices": indices,
                    "lengths": lengths,
                    "aligned_sample_tokens": tokens,
                    "physical_tokens": (tokens // cp + pad - 1) // pad * pad * cp,
                    "capacity": self.geometry["capacity"],
                    "oversized_singleton": oversized,
                    "dummy": offset in iterator.dummy_micro_batch_offsets,
                }
            )
            return result

        def step_wrapper(instance: Any) -> Any:
            _fence()
            before = time.perf_counter()
            result = original_step()
            _fence()
            self.write(
                {
                    "kind": "timing",
                    "forward_backward_seconds": before - self.start,
                    "optimizer_seconds": time.perf_counter() - before,
                    "update_successful": bool(result[0]),
                }
            )
            return result

        DataIterator.get_next = next_wrapper
        optimizer.step = MethodType(step_wrapper, optimizer)

    def write(self, value: dict) -> None:
        with self.path.open("a") as stream:
            stream.write(
                json.dumps(
                    {**self.coordinates, "rollout_id": self.rollout_id, "step_id": self.step_id, **value},
                    allow_nan=False,
                )
                + "\n"
            )


def before_train_step(
    args: Any, rollout_id: int, step_id: int, model: Any, optimizer: Any, opt_param_scheduler: Any
) -> None:
    audit = getattr(optimizer, "_relax_flops_audit", None)
    if audit is None:
        audit = _Audit(args, optimizer)
        optimizer._relax_flops_audit = audit
    audit.rollout_id, audit.step_id = rollout_id, step_id
    audit.iterators = {}
    _fence()
    audit.start = time.perf_counter()
