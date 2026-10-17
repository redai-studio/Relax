# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Export a committed Relax checkpoint in the reference DiffSynth LoRA
layout."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from relax.backends.fsdp.checkpoint import _load_dcp_model_state


def export_adapter(checkpoint_dir: Path, output: Path) -> None:
    """Run offline, outside a distributed process group; preserve reference
    keys."""
    if torch.distributed.is_initialized():
        raise RuntimeError("Export in a separate CPU process after the training checkpoint is committed")
    if not (checkpoint_dir / "COMMITTED").is_file():
        raise ValueError("Refusing to export an uncommitted checkpoint")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    contract = json.loads((checkpoint_dir / "adapter_contract.json").read_text())
    rank = int(contract["lora"]["rank"])
    if float(contract["lora"]["alpha"]) != rank:
        raise ValueError("This export requires alpha == rank (unit LoRA scaling)")
    state = _load_dcp_model_state(str(checkpoint_dir / "fsdp"))
    if not state:
        raise ValueError("Checkpoint has no adapter parameters")
    pairs: dict[str, set[str]] = {}
    for name, tensor in state.items():
        kind = next((part for part in ("lora_A", "lora_B") if name.endswith(f".{part}.default.weight")), None)
        if kind is None or not name.startswith("pipe.dit.") or tensor.ndim != 2:
            raise ValueError(f"Unexpected SenseNova adapter tensor: {name}")
        if tensor.shape[0 if kind == "lora_A" else 1] != rank:
            raise ValueError(f"Adapter rank mismatch: {name}")
        stem = name.rsplit(f".{kind}.", 1)[0]
        pairs.setdefault(stem, set()).add(kind)
    if any(kinds != {"lora_A", "lora_B"} for kinds in pairs.values()):
        raise ValueError("Incomplete LoRA A/B pair")
    from safetensors.torch import save_file

    output.parent.mkdir(parents=True, exist_ok=True)
    # DiffSynth loads against pipe.dit: keys must be relative to that module.
    # Preserve the BF16 forward view of FP32 optimizer masters.
    save_file(
        {name.removeprefix("pipe.dit."): tensor.to(torch.bfloat16).contiguous() for name, tensor in state.items()},
        str(output),
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(checkpoint_dir),
                "contract": contract,
                "dtype": "bfloat16",
                "tensors": len(state),
                "parameters": sum(t.numel() for t in state.values()),
            },
            indent=2,
        )
    )


def export_full(checkpoint: Path, base: Path, output: Path) -> Path:
    """Export only model tensors, retaining BF16 forward weights and
    tokenizer."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not (checkpoint / "COMMITTED").is_file():
        raise ValueError("Full checkpoint must be committed")
    contract = json.loads((checkpoint / "adapter_contract.json").read_text())
    if contract.get("trainable_mode") != "full":
        raise ValueError("Expected a full-parameter checkpoint")
    if output.exists():
        raise FileExistsError(output)
    expected = {}
    for path in base.glob("model*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            expected.update({key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()})
    if not expected:
        raise ValueError("Base model has no tensors")
    state = _load_dcp_model_state(str(checkpoint / "fsdp"))
    if any(not key.startswith("pipe.dit.") for key in state):
        raise ValueError("Unexpected full checkpoint key")
    if {key.removeprefix("pipe.dit.") for key in state} != expected.keys():
        raise ValueError("Full checkpoint does not match base model keyset")
    for key, tensor in state.items():
        if tuple(tensor.shape) != expected[key.removeprefix("pipe.dit.")]:
            raise ValueError(f"Full checkpoint shape mismatch: {key}")
    output.mkdir(parents=True)
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_index = 0
    weight_map = {}
    total_bytes = 0

    def flush() -> None:
        nonlocal shard_index, shard_bytes
        if not shard:
            return
        shard_index += 1
        name = f"model-{shard_index:05d}.safetensors"
        save_file(shard, str(output / name))
        weight_map.update({key: name for key in shard})
        shard.clear()
        shard_bytes = 0

    for key in sorted(state):
        tensor = state.pop(key).to(torch.bfloat16).contiguous()
        size = tensor.numel() * tensor.element_size()
        if shard_bytes + size > 4_000_000_000:
            flush()
        shard[key.removeprefix("pipe.dit.")] = tensor
        shard_bytes += size
        total_bytes += size
    flush()
    for path in base.iterdir():
        if path.is_file() and path.suffix in (".json", ".txt") and "safetensors.index" not in path.name:
            shutil.copy2(path, output / path.name)
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2)
    )
    (output / "export.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), "contract": contract, "tensors": len(weight_map)}, indent=2)
    )
    return output
