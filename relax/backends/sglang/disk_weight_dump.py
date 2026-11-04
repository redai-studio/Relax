# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Read-only SGLang custom loader that dumps live persistent model state."""

import json
import os
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import save_file

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
DUMP_LOADER = "relax.backends.sglang.disk_weight_dump.dump_weights"


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def dump_model(model: torch.nn.Module, directory: Path, tp_rank: int) -> dict:
    """Save every state_dict entry, preserving dtype and documenting
    aliases/exclusions."""
    if not directory.is_absolute():
        raise ValueError("Dump directory must be absolute")
    rank_dir = directory / f"rank-{tp_rank}"
    rank_dir.mkdir(parents=True, exist_ok=False)
    state = model.state_dict()
    parameters = dict(model.named_parameters(remove_duplicate=False))
    persistent_buffers = dict(model.named_buffers(remove_duplicate=False))
    excluded = []
    for module_name, module in model.named_modules():
        for name in module._non_persistent_buffers_set:
            tensor = module._buffers.get(name)
            if tensor is not None:
                excluded.append(
                    {
                        "name": f"{module_name}.{name}" if module_name else name,
                        "module": module_name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    }
                )
    excluded_names = {entry["name"] for entry in excluded}
    required = set(parameters) | (set(persistent_buffers) - excluded_names)
    if not required <= set(state):
        raise ValueError(f"state_dict omits persistent inference tensors: {sorted(required - set(state))}")
    aliases: dict[int, list[str]] = {}
    for name, parameter in parameters.items():
        aliases.setdefault(id(parameter), []).append(name)
    alias_groups = sorted((sorted(names) for names in aliases.values() if len(names) > 1), key=lambda names: names[0])
    if any(tensor.is_cuda for tensor in state.values()):
        torch.cuda.synchronize()
    weight_map: dict[str, str] = {}
    entries = []
    total_bytes = 0
    for index, (name, tensor) in enumerate(state.items()):
        filename = f"tensor-{index:08d}.safetensors"
        destination = rank_dir / filename
        # save_file writes a single CPU tensor, so peak host materialization is one tensor.
        cpu_tensor = tensor.detach().to(device="cpu", copy=True).contiguous()
        save_file({name: cpu_tensor}, destination)
        byte_count = tensor.numel() * tensor.element_size()
        total_bytes += byte_count
        weight_map[name] = filename
        entries.append(
            {
                "name": name,
                "kind": "parameter" if name in parameters else "buffer",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "elements": tensor.numel(),
                "bytes": byte_count,
            }
        )
    coverage = {
        "tp_rank": tp_rank,
        "tensor_count": len(entries),
        "parameter_count": sum(name in parameters for name in state),
        "persistent_buffer_count": sum(name in persistent_buffers and name not in parameters for name in state),
        "elements": sum(entry["elements"] for entry in entries),
        "bytes": total_bytes,
        "tensors": entries,
        "parameter_aliases": alias_groups,
        "excluded_nonpersistent_buffers": sorted(excluded, key=lambda entry: entry["name"]),
    }
    _atomic_json(rank_dir / "model.safetensors.index.json", {"weight_map": weight_map})
    _atomic_json(rank_dir / "coverage.json", coverage)
    from relax.distributed.weight_sync.provenance import runtime_identity

    _atomic_json(rank_dir / "runtime.json", runtime_identity())
    logger.info("Dumped rank=%s tensors=%s bytes=%s to %s", tp_rank, len(entries), total_bytes, rank_dir)
    return coverage


def dump_weights(model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    """SGLang custom_weight_loader hook; the only input tensor is a zero-length
    control."""
    controls = list(named_tensors)
    if len(controls) != 1 or controls[0][1].numel() != 0:
        raise ValueError("Dump loader requires one zero-length control tensor")
    directory = Path(controls[0][0])
    from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank

    dump_model(model, directory, get_tensor_model_parallel_rank())
