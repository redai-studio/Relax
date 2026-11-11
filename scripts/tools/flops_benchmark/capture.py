# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in single-step export from the optimizer's actual FP32 shards.

Use the existing custom_megatron_before_train_step_hook_path argument. This
module is never installed in timed performance runs.
"""

import hashlib
import json
import os
import random
from pathlib import Path
from types import MethodType
from typing import Any, Iterator

from scripts.tools.flops_benchmark.common import digest, file_digest, write_json


def _leaves(optimizer: Any, path: tuple[int, ...] = ()) -> Iterator[tuple[tuple[int, ...], Any]]:
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        yield path, optimizer
    else:
        for index, child in enumerate(children):
            yield from _leaves(child, (*path, index))


def _shards(optimizer: Any, names: dict[int, tuple]) -> Iterator[tuple[dict, Any, Any, dict]]:
    import torch
    import torch.distributed as dist
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    for child_path, child in _leaves(optimizer):
        if getattr(child, "is_stub_optimizer", False):
            continue
        if not isinstance(child, DistributedOptimizer) or child.ddp_config.use_megatron_fsdp:
            raise TypeError("FLOPs numerical export requires Megatron DistributedOptimizer")
        inner = child.optimizer
        owners = {
            id(parameter): (sub.state[parameter], group)
            for sub in getattr(inner, "sub_optimizers", [inner])
            for group in sub.param_groups
            for parameter in group["params"]
        }
        precision_aware = child.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        for parameter, (group_index, param_index) in child.model_param_group_index_map.items():
            chunk, name, shape, size = names[id(parameter)]
            span = child._get_model_param_range_map(parameter)["param"]
            shard = inner.param_groups[group_index]["params"][param_index]
            gradient = getattr(shard, "decoupled_grad", None) if precision_aware else shard.grad
            working_parameters = getattr(inner, "param_to_inner_param", None)
            master = (
                working_parameters[shard]
                if working_parameters is not None
                else child._get_main_param_and_optimizer_states(parameter)["param"]
            )
            if master is None or master.dtype != torch.float32:
                raise ValueError(f"Missing true FP32 master for {name}; run the baseline warmup first")
            if not (0 <= span.start < span.end <= size) or master.numel() != span.end - span.start:
                raise ValueError(f"Invalid optimizer shard range for {name}")
            working = working_parameters[shard] if working_parameters is not None else shard
            state, actual_group = owners[id(working)]
            if state is not inner.state[shard]:
                raise ValueError("Optimizer outer state differs from the actual working parameter state")
            for moment in ("exp_avg", "exp_avg_sq"):
                if state[moment].numel() != master.numel():
                    raise ValueError(f"Optimizer moment shape differs for {name}")
            state = {
                **state,
                "step": state.get("step", actual_group.get("step")),
                "hyperparameters": {key: actual_group[key] for key in ("lr", "betas", "eps", "weight_decay")},
            }
            metadata = {
                "optimizer": list(child_path),
                "chunk": chunk,
                "name": name,
                "shape": list(shape),
                "numel": size,
                "start": span.start,
                "end": span.end,
                "dp_group_ranks": dist.get_process_group_ranks(child.data_parallel_group),
            }
            yield metadata, gradient, master, state


def _fence() -> None:
    import torch

    # CPU masters can be pinned buffers written by asynchronous D2H copies.
    if torch.cuda.is_initialized():
        torch.cuda.synchronize(torch.cuda.current_device())


def _expected_parameters(optimizer: Any, names: dict[int, tuple]) -> list[dict]:
    import torch.distributed as dist

    expected = []
    all_ids: set[int] = set()
    for child_path, child in _leaves(optimizer):
        if getattr(child, "is_stub_optimizer", False):
            continue
        child_parameters = {id(parameter): parameter for buffer in child.buffers for parameter in buffer.params}
        if all_ids.intersection(child_parameters):
            raise ValueError("A trainable parameter belongs to multiple optimizer children")
        all_ids.update(child_parameters)
        for parameter in child_parameters.values():
            chunk, name, shape, size = names[id(parameter)]
            expected.append(
                {
                    "optimizer": list(child_path),
                    "chunk": chunk,
                    "name": name,
                    "shape": list(shape),
                    "numel": size,
                    "dp_group_ranks": dist.get_process_group_ranks(child.data_parallel_group),
                }
            )
    if all_ids != set(names):
        raise ValueError("Trainable model parameters and optimizer buffers differ")
    return expected


def _snapshot(tensor: Any) -> Any:
    import torch

    result = tensor.detach().to(device="cpu", dtype=torch.float32, copy=True).reshape(-1)
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite tensor in FLOPs numerical capture")
    return result


def _state_fingerprint(value: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        cpu = value.detach().to("cpu").contiguous()
        if cpu.is_floating_point() and not torch.isfinite(cpu).all():
            raise ValueError("Non-finite optimizer/RNG state")
        return {
            "dtype": str(cpu.dtype),
            "shape": list(cpu.shape),
            "sha256": hashlib.sha256(memoryview(cpu.reshape(-1).view(torch.uint8).numpy())).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        return _state_fingerprint(torch.from_numpy(value))
    if isinstance(value, torch.Generator):
        return _state_fingerprint(value.get_state())
    if isinstance(value, dict):
        return {key: _state_fingerprint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_state_fingerprint(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported checkpoint state type: {type(value).__name__}")


def _optimizer_fingerprint(state: dict, expected_step: int) -> dict:
    if state["step"] is None or float(state["step"]) != expected_step:
        raise ValueError(f"Adam step did not restore/advance to {expected_step}")
    return _state_fingerprint({key: state[key] for key in ("step", "exp_avg", "exp_avg_sq", "hyperparameters")})


class _Capture:
    def __init__(self, args: Any, model: Any, optimizer: Any, scheduler: Any, directory: Path) -> None:
        import numpy as np
        import torch
        import torch.distributed as dist
        from megatron.core import mpu
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        from relax.backends.megatron.data import DataIterator

        for flag in (
            "no_load_optim",
            "no_load_rng",
            "finetune",
            "reset_optimizer_states",
            "override_opt_param_scheduler",
        ):
            if getattr(args, flag, False):
                raise ValueError(f"Numerical capture requires full checkpoint restore: {flag}")
        if not args.use_checkpoint_opt_param_scheduler:
            raise ValueError("Numerical capture requires the restored scheduler")
        self.active = False
        self.calls = 0
        self.rollout_id = -1
        self.step_id = -1
        self.names = {
            id(parameter): (chunk, name, tuple(parameter.shape), parameter.numel())
            for chunk, module in enumerate(model)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        world = dist.group.WORLD
        self.rank = dist.get_rank(world)
        self.directory = directory / f"rank-{self.rank:05d}"
        self.directory.mkdir(parents=True, exist_ok=False)
        manifest_path = Path(os.environ["RELAX_FLOPS_MANIFEST"])
        manifest = json.loads(manifest_path.read_text())
        selected = int(os.environ["RELAX_FLOPS_CAPTURE_ROLLOUT"])
        _fence()
        scheduler_state = scheduler.state_dict()
        if scheduler_state["num_steps"] != selected * args.global_batch_size:
            raise ValueError("Scheduler did not restore the warmup sample count")
        data_state = json.loads((Path(args.save) / "sft_data_state" / f"{selected + 1:07d}.json").read_text())
        expected_batch = manifest["batches"][selected]
        if data_state["source_ids_sha256"] != expected_batch["source_ids_sha256"]:
            raise ValueError("The actual SFT optimizer batch differs from the frozen workload")
        self.metadata = {
            "version": 1,
            "rank": self.rank,
            "world_size": dist.get_world_size(world),
            "manifest_sha256": file_digest(manifest_path),
            "tolerances_sha256": file_digest(Path(__file__).with_name("tolerances.json")),
            "model": manifest["model"],
            "model_revision": manifest["model_revision"],
            "tp_rank": mpu.get_tensor_model_parallel_rank(),
            "pp_rank": mpu.get_pipeline_model_parallel_rank(),
            "cp_rank": mpu.get_context_parallel_rank(),
            "ep_rank": mpu.get_expert_model_parallel_rank(),
            "dp_rank": mpu.get_data_parallel_rank(),
            "dp_size": mpu.get_data_parallel_world_size(),
            "model_chunks": len(model),
            "expected_samples": expected_batch["samples"],
            "seed": args.seed,
            "loss_type": args.loss_type,
            "per_token": args.calculate_per_token_loss,
            "global_batch_size": args.global_batch_size,
            "load": args.load,
            "batch_source_ids_sha256": data_state["source_ids_sha256"],
            "initial_state_id": os.environ["RELAX_FLOPS_INITIAL_STATE_ID"],
            "scheduler": _state_fingerprint(scheduler_state),
            "rng": _state_fingerprint(
                {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state(torch.cuda.current_device()),
                    "megatron_cuda": get_cuda_rng_tracker().get_states(),
                }
            ),
            "expected_parameters": _expected_parameters(optimizer, self.names),
        }
        self.before: dict[str, dict] = {}
        self.consumed: dict[int, list[dict]] = {}
        original_next = DataIterator.get_next

        def next_wrapper(iterator: Any, keys: Any) -> dict:
            result = original_next(iterator, keys)
            if self.active:
                rows = self.consumed.setdefault(id(iterator), [])
                if not result.get("__is_dummy__", False):
                    for tokens, mask in zip(result["tokens"], result["loss_masks"], strict=True):
                        tokens = torch.as_tensor(tokens).to(device="cpu", dtype=torch.int64).contiguous()
                        mask = torch.as_tensor(mask).to(device="cpu", dtype=torch.int64).contiguous()
                        rows.append(
                            {
                                "length": tokens.numel(),
                                "effective_tokens": int(mask[1:].sum()),
                                "tokens_sha256": hashlib.sha256(memoryview(tokens.numpy())).hexdigest(),
                                "mask_sha256": hashlib.sha256(memoryview(mask.numpy())).hexdigest(),
                            }
                        )
            return result

        DataIterator.get_next = next_wrapper
        original_prepare, original_step = optimizer.prepare_grads, optimizer.step

        def prepare_wrapper(instance: Any) -> Any:
            found_inf = original_prepare()
            if self.active:
                self.calls += 1
                if found_inf:
                    raise ValueError("Non-finite gradients; optimizer update would be skipped")
                self.save_before(instance)
            return found_inf

        def step_wrapper(instance: Any) -> Any:
            self.calls = 0
            result = original_step()
            if self.active:
                if self.calls != 1 or not result[0]:
                    raise ValueError("Capture requires exactly one successful optimizer update")
                self.save_after(instance)
            return result

        optimizer.prepare_grads = MethodType(prepare_wrapper, optimizer)
        optimizer.step = MethodType(step_wrapper, optimizer)

    def save_before(self, optimizer: Any) -> None:
        import torch

        _fence()
        self.before = {}
        for metadata, gradient, master, state in _shards(optimizer, self.names):
            key = digest(metadata)
            if key in self.before or gradient is None or gradient.numel() != master.numel():
                raise ValueError(f"Missing or duplicate reduced gradient for {metadata['name']}")
            self.before[key] = metadata
            torch.save(
                {
                    "gradient": _snapshot(gradient),
                    "master": _snapshot(master),
                    "optimizer_state": _optimizer_fingerprint(state, self.rollout_id),
                },
                self.directory / f"{key}.before.pt",
            )

    def save_after(self, optimizer: Any) -> None:
        import torch

        _fence()
        entries = []
        for metadata, _, master, state in _shards(optimizer, self.names):
            if float(state["step"]) != self.rollout_id + 1:
                raise ValueError("Adam state did not advance exactly one step")
            key = digest(metadata)
            if self.before.pop(key, None) != metadata:
                raise ValueError("Optimizer shard layout changed during the update")
            path = self.directory / f"{key}.before.pt"
            before = torch.load(path, map_location="cpu", weights_only=True)
            after = _snapshot(master)
            delta = after - before["master"]
            if not torch.isfinite(delta).all():
                raise ValueError("Non-finite FP32 parameter update")
            output = self.directory / f"{key}.pt"
            torch.save({"gradient": before["gradient"], "delta": delta}, output)
            initial_hash = hashlib.sha256(memoryview(before["master"].numpy())).hexdigest()
            entries.append(
                {
                    **metadata,
                    "file": output.name,
                    "sha256": file_digest(output),
                    "initial_master_sha256": initial_hash,
                    "initial_optimizer_state": before["optimizer_state"],
                }
            )
            path.unlink()
        if self.before:
            raise ValueError("Optimizer shards were lost during the update")
        write_json(
            self.directory / "manifest.json",
            {
                **self.metadata,
                "rollout_id": self.rollout_id,
                "step_id": self.step_id,
                "prepare_calls": self.calls,
                "update_successful": True,
                "shards": entries,
                "consumed_samples": list(self.consumed.values()),
            },
        )
        self.active = False


def before_train_step(
    args: Any, rollout_id: int, step_id: int, model: Any, optimizer: Any, opt_param_scheduler: Any
) -> None:
    """Capture only the explicitly selected optimizer step; absent exports fail
    comparison."""
    selected = int(os.environ["RELAX_FLOPS_CAPTURE_ROLLOUT"])
    if rollout_id != selected or step_id != 0:
        return
    if getattr(optimizer, "_relax_flops_capture", None) is not None:
        raise RuntimeError("The selected FLOPs capture step was executed more than once")
    capture = _Capture(args, model, optimizer, opt_param_scheduler, Path(os.environ["RELAX_FLOPS_CAPTURE_DIR"]))
    optimizer._relax_flops_capture = capture
    capture.active, capture.rollout_id, capture.step_id = True, rollout_id, step_id
