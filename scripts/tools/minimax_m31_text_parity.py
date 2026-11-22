# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared TP cross entropy for the independent MiniMax text SFT recipes."""

import argparse
import ast
import math
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import torch


def _aligned_loss(
    logits: "torch.Tensor", batch: dict[str, Any], *, predictor_mask: bool
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import vocab_parallel_cross_entropy

    if logits.ndim != 3 or logits.size(0) != 1 or len(batch["total_lengths"]) != 1:
        raise ValueError("MiniMax text parity requires one THD sample per microbatch.")
    if parallel_state.get_context_parallel_world_size() != 1:
        raise ValueError("MiniMax text parity requires CP=1.")

    (tokens,) = batch["unconcat_tokens"]
    (mask,) = batch["loss_masks"]
    (total,) = batch["total_lengths"]
    (response,) = batch["response_lengths"]
    if mask.numel() != response or total != tokens.numel():
        raise ValueError("Invalid token or mask length in the MiniMax text parity batch.")

    # Relax stores a full-length mask in predictor coordinates. Miles stores
    # the response suffix in target coordinates.
    start = 0 if predictor_mask else total - response - 1
    if start < 0 or (predictor_mask and response != total):
        raise ValueError("Invalid response span in the MiniMax text parity batch.")
    relative = mask.reshape(-1).nonzero().flatten()
    positions = relative + start
    selected = logits[0].index_select(0, positions)
    pre_shifted_labels = batch.get("unconcat_pre_shifted_labels") if predictor_mask else None
    targets = (
        pre_shifted_labels[0].index_select(0, positions)
        if pre_shifted_labels is not None
        else tokens.index_select(0, positions + 1)
    )

    if targets.numel():
        group = parallel_state.get_tensor_model_parallel_group()
        nll = torch.cat(
            [
                vocab_parallel_cross_entropy(chunk.float().clone(), target, tp_group=group)
                for chunk, target in zip(selected.split(256), targets.split(256), strict=True)
            ]
        )
    else:
        nll = selected.float().sum(dim=-1)
    # Restore the native response coordinates for Relax's dataset metrics.
    response_nll = nll.new_zeros(response).index_copy(0, relative, nll)
    return nll.sum(), response_nll


def relax_sft_loss(args: Any, batch: dict, logits: "torch.Tensor", _reduce: Any) -> tuple:
    from relax.backends.megatron.loss import _sft_loss_metrics

    loss, response_nll = _aligned_loss(logits, batch, predictor_mask=True)
    return loss, _sft_loss_metrics(args, batch, loss, response_nll)


def miles_sft_loss(_args: Any, batch: dict, logits: "torch.Tensor", _reduce: Any) -> tuple:
    loss, _ = _aligned_loss(logits, batch, predictor_mask=False)
    return loss, {"loss": loss.detach().clone()}


def log_framework(args: Any, framework: str) -> None:
    if getattr(args, "rank", 0) == 0:
        print(f"MiniMax native parity: framework={framework}", flush=True)


def configure_miles_worker(args: Any) -> None:
    from miles.backends.training_utils.loss.hub import losses

    losses.sft_loss_function = miles_sft_loss
    log_framework(args, "miles")


def _read_metrics(path: Path, steps: int, framework: str) -> dict[int, tuple[float, float]]:
    log = path.read_text()
    frameworks = set(re.findall(r"MiniMax native parity: framework=(\w+)", log))
    if frameworks != {framework}:
        raise ValueError(f"{path}: expected {framework} framework, found {sorted(frameworks)}.")
    metrics = {}
    for match in re.finditer(r"step (\d+): (\{[^\n]+\})", log):
        if "'train/loss'" not in match[2] or "'train/grad_norm'" not in match[2]:
            continue
        values = ast.literal_eval(match[2])
        step = int(match[1])
        pair = (float(values["train/loss"]), float(values["train/grad_norm"]))
        if not all(math.isfinite(value) for value in pair) or (step in metrics and metrics[step] != pair):
            raise ValueError(f"{path}: invalid or conflicting metrics for step {step + 1}.")
        metrics[step] = pair
    if steps < 1 or sorted(metrics) != list(range(steps)):
        raise ValueError(f"{path}: expected {steps} steps, found {sorted(metrics)}.")
    return metrics


def compare_logs(relax: Path, miles: Path, steps: int) -> None:
    if relax.resolve() == miles.resolve():
        raise ValueError("Provide separate Relax and Miles logs.")
    left = _read_metrics(relax, steps, "relax")
    right = _read_metrics(miles, steps, "miles")
    different = [step + 1 for step in left if struct.pack("!dd", *left[step]) != struct.pack("!dd", *right[step])]
    if different:
        raise SystemExit(f"Loss or gradient norm differs at steps {different}.")
    from relax.utils.logging_utils import get_logger

    get_logger(__name__).info("Loss and gradient norm match bitwise for %d steps.", steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs=2, required=True, type=Path, metavar=("RELAX_LOG", "MILES_LOG"))
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    compare_logs(*args.logs, args.steps)
