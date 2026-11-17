# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in, disk-backed capture of a complete optimizer-owned FP32 step.

This diagnostic intentionally synchronizes and writes large files. Never enable
it for timing. The adapter follows the pinned Megatron prepare_grads -> clipping
-> step contract; it does not run prepare_grads or optimizer.step a second time.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def capture_rollout_inputs(
    rollout_data: dict[str, Any],
    directory: Path,
    *,
    rank: int,
    topology: dict[str, int],
    normalization: str,
    global_batch_size: int,
    device: torch.device,
    group: Any,
) -> None:
    """Record the effective input multiset, including masks, before reordering.

    This is enabled only for the separate numerical run. Device-to-host copies
    here are deliberate diagnostics and must never enter performance runs.
    """
    error = None
    try:
        rows = []
        for tokens, mask, total, response in zip(
            rollout_data["tokens"],
            rollout_data["loss_masks"],
            rollout_data["total_lengths"],
            rollout_data["response_lengths"],
            strict=True,
        ):
            digest = hashlib.sha256()
            digest.update(json.dumps([int(total), int(response)]).encode())
            for value, dtype in ((tokens, torch.int64), (mask, torch.float32)):
                tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=dtype).contiguous()
                digest.update(tensor.numel().to_bytes(8, "little"))
                digest.update(tensor.numpy().tobytes())
            rows.append(digest.hexdigest())
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"inputs-rank-{rank:05d}.json"
        if path.exists():
            raise ValueError("Input capture already exists")
        path.write_text(
            json.dumps(
                dict(
                    samples=rows, topology=topology, normalization=normalization, global_batch_size=global_batch_size
                ),
                indent=2,
            )
            + "\n"
        )
    except Exception as exc:
        error = exc
    flag = torch.tensor(int(error is not None), device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    if bool(flag):
        raise RuntimeError("Input capture failed on a training rank") from error


class OptimizerStepCapture:
    def __init__(self, optimizer: Any, directory: Path, *, group: Any, device: torch.device) -> None:
        self.optimizer = optimizer
        self.directory = directory
        self.group = group
        self.device = device
        self.parameters: list[torch.Tensor] = []
        self.gradient_parameters: list[torch.Tensor] = []
        self.manifest: dict[str, Any] = {"format": 1, "gradient_stage": "unscaled_reduced_before_clipping"}
        self._prepared = False
        self._original_prepare = optimizer.prepare_grads
        self._had_instance_method = "prepare_grads" in vars(optimizer)

    def _agree_error(self, error: Exception | None, phase: str) -> None:
        flag = torch.tensor(int(error is not None), device=self.device)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.group)
        if bool(flag):
            raise RuntimeError(f"Optimizer capture failed during {phase} on a training rank") from error

    def __enter__(self) -> "OptimizerStepCapture":
        error = None
        try:
            leaves = getattr(self.optimizer, "chained_optimizers", [self.optimizer])
            for leaf in leaves:
                config = leaf.config
                if getattr(config, "chunked_optimizer_state_offload", False):
                    raise ValueError("Capture does not support chunked optimizer-state offload")
                sources = list(leaf.get_parameters())
                inner = getattr(leaf, "optimizer", None)
                hybrid = type(inner).__module__ == "megatron.core.optimizer.cpu_offloading.hybrid_optimizer"
                if getattr(config, "use_megatron_fsdp", False):
                    raise ValueError("Capture does not support Megatron FSDP")
                if hybrid and getattr(inner, "param_update_in_fp32", False):
                    self.parameters.extend(inner.param_to_inner_param[source] for source in sources)
                elif getattr(config, "use_precision_aware_optimizer", False) or getattr(
                    config, "optimizer_cpu_offload", False
                ):
                    raise ValueError("This optimizer does not expose supported FP32 update tensors")
                else:
                    self.parameters.extend(sources)
                self.gradient_parameters.extend(sources)
            if any(parameter.dtype != torch.float32 for parameter in self.parameters):
                raise ValueError("Optimizer parameters must be actual FP32 master shards")
            self.directory.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            error = exc
        self._agree_error(error, "initialization")

        def prepare_and_capture() -> bool:
            found_inf = self._original_prepare()
            error = None
            try:
                if self._prepared:
                    raise RuntimeError("prepare_grads called more than once for the captured update")
                if found_inf:
                    raise RuntimeError("The captured update overflowed")
                entries = []
                for index, (parameter, source) in enumerate(
                    zip(self.parameters, self.gradient_parameters, strict=True)
                ):
                    gradient = getattr(source, "decoupled_grad", None)
                    if gradient is None:
                        gradient = source.grad
                    if gradient is None:
                        raise ValueError(f"Missing reduced gradient for optimizer parameter {index}")
                    initial = parameter.detach().to(device="cpu", copy=True)
                    grad = gradient.detach().to(device="cpu", dtype=torch.float32, copy=True)
                    if not bool(torch.isfinite(initial).all()) or not bool(torch.isfinite(grad).all()):
                        raise ValueError("Non-finite initial parameter or gradient")
                    filename = f"parameter-{index:06d}.pt"
                    torch.save({"initial": initial, "gradient": grad}, self.directory / filename)
                    entries.append({"file": filename, "shape": list(parameter.shape), "elements": parameter.numel()})
                self.manifest["parameters"] = entries
                self._prepared = True
            except Exception as exc:
                error = exc
            self._agree_error(error, "reduced gradient export")
            return found_inf

        self.optimizer.prepare_grads = prepare_and_capture
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._had_instance_method:
            self.optimizer.prepare_grads = self._original_prepare
        else:
            del self.optimizer.prepare_grads
        if exc_type is not None:
            return
        error = None
        try:
            if not self._prepared:
                raise RuntimeError("This optimizer did not use the expected prepare_grads contract")
            for entry, parameter in zip(self.manifest["parameters"], self.parameters, strict=True):
                path = self.directory / entry["file"]
                payload = torch.load(path, map_location="cpu", weights_only=True)
                payload["update"] = parameter.detach().to(device="cpu", copy=True) - payload["initial"]
                if not bool(torch.isfinite(payload["update"]).all()):
                    raise ValueError("Non-finite FP32 parameter update")
                torch.save(payload, path)
        except Exception as err:
            error = err
        self._agree_error(error, "FP32 update export")

    def finish(self, loss: dict[str, float], successful: bool) -> None:
        error = None
        try:
            if not successful:
                raise RuntimeError("The captured optimizer update was skipped")
            self.manifest.update(
                loss=loss,
                successful=successful,
                elements=sum(entry["elements"] for entry in self.manifest["parameters"]),
            )
            (self.directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")
        except Exception as exc:
            error = exc
        self._agree_error(error, "manifest completion")
