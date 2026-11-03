# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in, bounded CPU samples of latent-master and effective MXFP4 updates.

The counter counts successful local optimizer updates after this recorder is
enabled, not global training steps. Comparisons span successive recorded
samples; initialization/resume starts a new comparison segment. Nothing is
copied from a GPU, and the optimizer's master tensors are never modified.
"""

from __future__ import annotations

import json
import os
import re
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable


if TYPE_CHECKING:
    import torch


@dataclass
class _Sample:
    master: torch.Tensor
    effective: torch.Tensor
    scales: torch.Tensor
    codes: torch.Tensor
    local_step: int
    finite: bool


@dataclass
class _State:
    path: Path
    interval: int
    identity: dict[str, Any]
    local_step: int = 0
    segment: int = 0
    previous: dict[int, _Sample] = field(default_factory=dict)


def _scale_and_codes(sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the reference's E8M0 clamp, all-zero and midpoint rules
    exactly."""
    import torch

    finite_block = torch.isfinite(sample).all(dim=1, keepdim=True)
    values = torch.where(finite_block, sample, 0.0)
    amax = values.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
    exponent = torch.ceil(torch.log2(scale.clamp(min=2.0**-127, max=2.0**127))).to(torch.int32)
    scale_bits = torch.where(exponent == -127, 0x00400000, (exponent + 127) << 23)
    scales = scale_bits.view(torch.float32)
    normalized = values / scales
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
    codes = torch.bucketize(normalized.abs(), boundaries)
    codes = codes | ((normalized < 0).to(torch.int64) * 8)
    return scales, codes


def _new_state(directory: str, optimizer: Any) -> _State:
    from relax.utils.env import Envs

    interval = Envs.RELAX_DSV4_FP4_STATS_INTERVAL
    if interval <= 0:
        raise ValueError("RELAX_DSV4_FP4_STATS_INTERVAL must be a positive integer")
    identity = {
        "rank_env": os.environ.get("RANK"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "optimizer_id": hex(id(optimizer)),
    }
    filename = "master_updates.rank-{}.host-{}.pid-{}.optimizer-{}.jsonl".format(
        re.sub(r"[^A-Za-z0-9_.-]", "_", identity["rank_env"] or "unknown"),
        re.sub(r"[^A-Za-z0-9_.-]", "_", identity["hostname"]),
        identity["pid"],
        identity["optimizer_id"],
    )
    path = Path(directory) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return _State(path=path, interval=interval, identity=identity)


def _matrix_sample(
    optimizer: Any,
    resolve_master: Callable[[Any, Any], torch.Tensor],
    binding_index: int,
    state: _State,
) -> tuple[dict[str, Any], _Sample]:
    import torch

    from relax.models.deepseek_v4.quantization import mxfp4_qdq_reference

    parameter, module = optimizer._relax_dsv4_fp4_bindings[binding_index]
    master = resolve_master(optimizer, parameter)
    if master.device.type != "cpu" or master.dtype != torch.float32:
        raise TypeError("DeepSeek-V4 MXFP4 QAT update metrics require an existing CPU FP32 optimizer master")
    if master.ndim != 2 or master.shape[1] % 32 or not master.is_contiguous() or master.numel() == 0:
        raise ValueError(
            "DeepSeek-V4 MXFP4 QAT update metrics require a nonempty contiguous 2-D master with K divisible by 32"
        )
    block_count = master.numel() // 32
    sample_count = min(block_count, 256)
    if sample_count == 1:
        indices = torch.zeros(1, dtype=torch.int64)
    else:
        indices = torch.arange(sample_count, dtype=torch.int64) * (block_count - 1) // (sample_count - 1)
    # Read complete K32 blocks from CPU storage; no parameter/GPU value access.
    sample = master.detach().view(-1, 32).index_select(0, indices).clone()
    finite_mask = torch.isfinite(sample)
    effective = mxfp4_qdq_reference(sample)
    finite_master = bool(finite_mask.all())
    finite_effective = bool(torch.isfinite(effective).all())
    finite = finite_master and finite_effective
    scales, codes = _scale_and_codes(sample)
    previous = state.previous.get(binding_index)
    comparable = previous is not None and previous.finite and finite and previous.master.shape == sample.shape
    module_state = getattr(module, "_relax_mxfp4_qat", None)
    record: dict[str, Any] = {
        "binding_index": binding_index,
        "module_name": getattr(module_state, "module_name", type(module).__name__),
        "matrix_shape": list(master.shape),
        "total_k32_blocks": block_count,
        "sampled_k32_blocks": sample_count,
        "sampled_values": sample.numel(),
        "sample_block_indices": indices.tolist(),
        "status": (
            "ok" if finite else "nonfinite_master_sample" if not finite_master else "nonfinite_effective_fp4_sample"
        ),
        "nonfinite_values": int((~finite_mask).sum()),
        "nonfinite_blocks": int((~finite_mask.all(dim=1)).sum()),
        "nonfinite_effective_fp4_values": int((~torch.isfinite(effective)).sum()),
        "comparison": "previous_record" if comparable else "baseline",
        "previous_local_successful_updates": previous.local_step if comparable else None,
        "updates_since_previous_record": state.local_step - previous.local_step if comparable else None,
        "master_update_rms": None,
        "master_update_max_abs": None,
        "effective_fp4_changed_fraction": None,
        "e8m0_scale_changed_fraction": None,
        "fp4_code_changed_fraction": None,
        "quantization_error_rms": None,
    }
    if finite:
        # Float64 reduction avoids overflow when diagnostic samples have large
        # finite FP32 values. Only these small CPU copies are converted.
        error = effective.double() - sample.double()
        record["quantization_error_rms"] = float(error.square().mean().sqrt())
    if comparable:
        delta = sample.double() - previous.master.double()
        record.update(
            master_update_rms=float(delta.square().mean().sqrt()),
            master_update_max_abs=float(delta.abs().amax()),
            effective_fp4_changed_fraction=float((effective != previous.effective).double().mean()),
            e8m0_scale_changed_fraction=float((scales != previous.scales).double().mean()),
            fp4_code_changed_fraction=float((codes != previous.codes).double().mean()),
        )
    return record, _Sample(sample, effective, scales, codes, state.local_step, finite)


def record_master_samples(
    optimizer: Any,
    resolve_master: Callable[[Any, Any], torch.Tensor],
    *,
    reason: str,
) -> None:
    """Record at most 3 x 256 complete K32 blocks; disabled without STATS_DIR.

    Call once after each successful ``step_with_ready_grads`` and with the
    actual hook name after initialization/resume. Every other reason clears the
    comparison baseline. A nonfinite sample is explicitly recorded and logged;
    this observer does not repair the sample or change training state.
    """
    from relax.utils.env import Envs

    directory = Envs.RELAX_DSV4_FP4_STATS_DIR
    if not directory:
        return
    bindings = getattr(optimizer, "_relax_dsv4_fp4_bindings", ())
    if not bindings:
        return
    state = getattr(optimizer, "_relax_dsv4_fp4_stats_state", None)
    if state is None:
        state = _new_state(directory, optimizer)
        optimizer._relax_dsv4_fp4_stats_state = state
    if reason == "step_with_ready_grads":
        state.local_step += 1
        if state.previous and state.local_step % state.interval:
            return
    else:
        state.previous.clear()
        state.segment += 1

    import torch

    selected = sorted({0, len(bindings) // 2, len(bindings) - 1})
    records = []
    with torch.no_grad():
        for binding_index in selected:
            record, sample = _matrix_sample(optimizer, resolve_master, binding_index, state)
            records.append(record)
            state.previous[binding_index] = sample
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **state.identity,
        "reason": reason,
        "local_successful_updates": state.local_step,
        "comparison_segment": state.segment,
        "record_interval": state.interval,
        "counter_scope": "local optimizer successful updates since recorder initialization; not global training step",
        "total_bound_matrices": len(bindings),
        "status": "ok" if all(record["status"] == "ok" for record in records) else "nonfinite_sample",
        "matrices": records,
    }
    with state.path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    if payload["status"] != "ok":
        from relax.utils.logging_utils import get_logger

        get_logger(__name__).error(
            "[DSV4-MXFP4] Nonfinite CPU master/effective FP4 sample after %s, local successful updates=%d; details: %s",
            reason,
            state.local_step,
            state.path,
        )
