# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Record byte digests of the rank-local full SFT training state."""

import hashlib
import json
from functools import wraps
from pathlib import Path
from typing import Any


_VISION_COMPONENTS = ("vision_tower", "multi_modal_projector", "patch_merge_mlp")
_COMPONENTS = {"language", "optimizer", *_VISION_COMPONENTS}


def _update_digest(digest: Any, name: str, value: Any) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        header = (name, str(value.dtype), tuple(value.shape))
        digest.update(json.dumps(header).encode() + b"\n")
        data = value.detach().reshape(-1).contiguous().view(torch.uint8).cpu()
        digest.update(memoryview(data.numpy()))
    else:
        digest.update(json.dumps((name, value), sort_keys=True).encode() + b"\n")


def _capture_state(model: Any, optimizer: Any) -> dict[str, str]:
    digests = {}
    for chunk, module in enumerate(model):
        for name, value in sorted(module.named_parameters()):
            component = next(
                (component for component in _VISION_COMPONENTS if component in name.split(".")),
                "language",
            )
            if component not in digests:
                digests[component] = hashlib.sha256()
            _update_digest(digests[component], f"{chunk}.{name}", value)

    optimizer_digest = hashlib.sha256()
    for index, child in enumerate(getattr(optimizer, "chained_optimizers", (optimizer,))):
        inner = child.optimizer
        for group, param_group in enumerate(inner.param_groups):
            if "step" in param_group:
                _update_digest(optimizer_digest, f"{index}.{group}.step", param_group["step"])
            for param_idx, param in enumerate(param_group["params"]):
                key = f"{index}.{group}.{param_idx}"
                _update_digest(optimizer_digest, key, param)
                for name, value in sorted(inner.state.get(param, {}).items()):
                    _update_digest(optimizer_digest, f"{key}.{name}", value)
    return {
        **{name: digest.hexdigest() for name, digest in digests.items()},
        "optimizer": optimizer_digest.hexdigest(),
    }


def record_full_sft_step(
    args: Any, rollout_id: int, step_id: int, model: Any, optimizer: Any, _scheduler: Any
) -> None:
    """Capture each successful update without writing a full checkpoint every
    step."""
    if step_id != 0 or args.lora_rank:
        raise ValueError("MiniMax full SFT parity expects one full-parameter update per rollout.")
    state = getattr(optimizer, "_minimax_parity_step", None)
    if state is None:
        state = {}
        optimizer._minimax_parity_step = state
        original_step = optimizer.step
        directory = Path(args.save) / "parity-state"
        directory.mkdir(parents=True, exist_ok=True)

        @wraps(original_step)
        def step(*step_args, **step_kwargs):
            result = original_step(*step_args, **step_kwargs)
            if result[0]:
                report = _capture_state(model, optimizer)
                path = directory / f"step-{state['rollout']:07d}-rank-{args.rank:04d}.json"
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(report, sort_keys=True) + "\n")
                temp.replace(path)
            return result

        optimizer.step = step
    state["rollout"] = rollout_id


def read_rank_digests(checkpoint: Path, index: int) -> dict[str, dict[str, str]]:
    files = sorted((checkpoint / "parity-state").glob(f"step-{index:07d}-rank-*.json"))
    ranks = [int(path.stem.rsplit("-", 1)[1]) for path in files]
    if ranks != list(range(16)):
        raise ValueError(f"{checkpoint}: expected 16 rank digests for step {index + 1}, found {ranks}.")
    result = {path.stem: json.loads(path.read_text()) for path in files}
    if any(report.keys() != _COMPONENTS for report in result.values()):
        raise ValueError(f"{checkpoint}: incomplete model or optimizer digest at step {index + 1}.")
    return result
