# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Advantage-whitening CP-invariance regression on CPU with Gloo collectives.

`compute_advantages_and_returns` whitens advantages over a process group via
`distributed_masked_whiten`. Under static context parallelism each rank holds
only its zig-zag slice of every sequence, so the collective must span DP with
CP. Under dynamic context parallelism responses are already merged to full
length on every static CP rank, so the collective must span only the
CP-excluding DP group to avoid counting each token `cp_size` times in Bessel's
correction.

Real Gloo worker processes (`<= 4` ranks per case, 30s timeout, ~25s total) are
required because an in-process mock cannot detect an `all_reduce` hang when one
rank owns zero response tokens (`all_masks.numel() == 0` on `(dp=1, cp=1)` at
`(dp_size=2, cp_size=2)`) or exercise split `dp_only` vs `dp_cp` collectives.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import time
from argparse import Namespace
from datetime import timedelta
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

SEQS = [(64, 48), (256, 16), (100, 90), (320, 20)]
REWARDS = [1.5, 2.0, -0.5, -1.0]

# Orthogonal failure modes:
# - (1, 2, False): pure static CP group membership
# - (2, 2, False): DP x CP reduction + empty local CP batch on (dp=1, cp=1)
# - (1, 4, False): 4-way zig-zag with non-divisible total_len=100
# - (2, 2, True): dynamic CP replicated responses + DP-only Bessel count
WHITEN_CASES = [
    (1, 2, False),
    (2, 2, False),
    (1, 4, False),
    (2, 2, True),
]


def _ensure_megatron_importable() -> None:
    try:
        import megatron.core  # noqa: F401
    except ModuleNotFoundError:
        megatron = ModuleType("megatron")
        megatron.__path__ = []
        core = ModuleType("megatron.core")
        core.__path__ = []
        mpu = ModuleType("megatron.core.mpu")
        core.mpu = mpu
        megatron.core = core
        sys.modules["megatron"] = megatron
        sys.modules["megatron.core"] = core
        sys.modules["megatron.core.mpu"] = mpu


def _sample_mask(response_len: int) -> torch.Tensor:
    return (torch.arange(response_len) % 3 != 0).to(dtype=torch.float32)


def _sample_raw_rewards(sample_idx: int, response_len: int) -> torch.Tensor:
    """Vary raw reward per token so every response position is checked."""
    return REWARDS[sample_idx] + 0.02 * torch.arange(response_len, dtype=torch.float32)


def _independent_whitened_reference() -> dict[str, list[float]]:
    """Closed-form float64 masked whitening with unbiased sample variance."""
    masked_vals: list[float] = []
    full_raw: list[list[float]] = []
    for sample_idx, (_, resp_len) in enumerate(SEQS):
        raw = [REWARDS[sample_idx] + 0.02 * pos for pos in range(resp_len)]
        full_raw.append(raw)
        for pos, val in enumerate(raw):
            if pos % 3 != 0:
                masked_vals.append(val)

    total_valid = len(masked_vals)
    mean = sum(masked_vals) / total_valid
    unbiased_var = sum((val - mean) ** 2 for val in masked_vals) / (total_valid - 1)
    inv_std = 1.0 / math.sqrt(unbiased_var + 1e-8)
    return {str(i): [(val - mean) * inv_std for val in raw] for i, raw in enumerate(full_raw)}


def _cp_response_slices(total_len: int, response_len: int, cp_rank: int, cp_size: int) -> list[tuple[int, int]]:
    """Response-coordinate `[lo, hi)` slices owned by `cp_rank`'s two zig-zag
    chunks."""
    if cp_size == 1:
        return [(0, response_len)]
    prompt_len = total_len - response_len
    chunk_size = (total_len + 2 * cp_size - 1) // (2 * cp_size)
    slices: list[tuple[int, int]] = []
    for chunk_id in (cp_rank, 2 * cp_size - 1 - cp_rank):
        lo = max(chunk_id * chunk_size, prompt_len - 1)
        hi = min((chunk_id + 1) * chunk_size, total_len - 1)
        if hi > lo:
            slices.append((lo - (prompt_len - 1), hi - (prompt_len - 1)))
    return slices


def _slice_response_tensor(
    full_tensor: torch.Tensor, total_len: int, response_len: int, cp_rank: int, cp_size: int
) -> torch.Tensor:
    slices = _cp_response_slices(total_len, response_len, cp_rank, cp_size)
    if not slices:
        return torch.empty(0, dtype=full_tensor.dtype)
    return torch.cat([full_tensor[lo:hi] for lo, hi in slices], dim=0)


def _make_args(*, dynamic_cp: bool) -> Namespace:
    return Namespace(
        advantage_estimator="grpo",
        normalize_advantages=True,
        kl_coef=0.0,
        kl_loss_type="low_var_kl",
        use_kl_loss=False,
        use_rollout_logprobs=False,
        use_opd=False,
        qkv_format="thd",
        is_vl_model=False,
        uses_unsplit_forward=False,
        dynamic_context_parallel=dynamic_cp,
        gamma=1.0,
        lambd=1.0,
    )


def _whiten_worker(
    rank: int,
    world_size: int,
    cp_size: int,
    dp_size: int,
    dynamic_cp: bool,
    init_file: str,
    result_dir: str,
    timeout_s: int,
) -> None:
    import torch.distributed as dist

    torch.set_num_threads(1)
    _ensure_megatron_importable()

    cp_rank = rank % cp_size
    dp_rank = rank // cp_size
    effective_cp = 1 if dynamic_cp else cp_size

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_s),
    )
    try:
        dp_cp_group = dist.new_group(ranks=list(range(world_size)))
        dp_only_groups = [
            dist.new_group(ranks=[r for r in range(world_size) if r % cp_size == c]) for c in range(cp_size)
        ]

        fake_mpu = SimpleNamespace(
            is_pipeline_last_stage=lambda: True,
            get_context_parallel_world_size=lambda: cp_size,
            get_context_parallel_rank=lambda: cp_rank,
            get_tensor_model_parallel_world_size=lambda: 1,
            get_data_parallel_group=lambda with_context_parallel=False: (
                dp_cp_group if with_context_parallel else dp_only_groups[cp_rank]
            ),
        )

        from relax.backends.megatron import cp_utils

        loss_module = importlib.import_module("relax.backends.megatron.loss")
        loss_module.mpu = fake_mpu
        cp_utils.mpu = fake_mpu

        my_samples = [i for i in range(len(SEQS)) if i % dp_size == dp_rank]
        rewards = []
        log_probs = []
        for i in my_samples:
            total_len, resp_len = SEQS[i]
            raw_reward = _sample_raw_rewards(i, resp_len)
            sliced_reward = _slice_response_tensor(raw_reward, total_len, resp_len, cp_rank, effective_cp)
            rewards.append(sliced_reward)
            log_probs.append(torch.zeros_like(sliced_reward))

        rollout_data = {
            "log_probs": log_probs,
            "loss_masks": [_sample_mask(SEQS[i][1]) for i in my_samples],
            "rewards": rewards,
            "response_lengths": [SEQS[i][1] for i in my_samples],
            "total_lengths": [SEQS[i][0] for i in my_samples],
        }

        loss_module.compute_advantages_and_returns(_make_args(dynamic_cp=dynamic_cp), rollout_data)

        sample_tokens: dict[str, dict[str, float]] = {}
        for i, adv in zip(my_samples, rollout_data["advantages"], strict=True):
            total_len, resp_len = SEQS[i]
            slices = _cp_response_slices(total_len, resp_len, cp_rank, effective_cp)
            positions = [pos for lo, hi in slices for pos in range(lo, hi)]
            assert len(positions) == adv.numel()
            if positions:
                sample_tokens[str(i)] = {
                    str(pos): float(val) for pos, val in zip(positions, adv.tolist(), strict=True)
                }

        out = {
            "empty_local_batch": sum(adv.numel() for adv in rollout_data["advantages"]) == 0,
            "samples": sample_tokens,
        }
        with open(os.path.join(result_dir, f"rank{rank}.json"), "w") as f:
            json.dump(out, f)
    finally:
        dist.destroy_process_group()


def _run_case(
    dp_size: int,
    cp_size: int,
    dynamic_cp: bool,
    base_dir,
    timeout_s: int = 30,
) -> tuple[dict[str, list[float]], bool]:
    import torch.multiprocessing as mp

    world_size = dp_size * cp_size
    case_dir = base_dir / f"dp{dp_size}_cp{cp_size}_dcp{int(dynamic_cp)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    init_file = case_dir / "gloo_init"

    ctx = mp.spawn(
        _whiten_worker,
        args=(world_size, cp_size, dp_size, dynamic_cp, str(init_file), str(case_dir), timeout_s),
        nprocs=world_size,
        join=False,
    )
    deadline = time.time() + timeout_s + 5
    while not ctx.join(timeout=1):
        if time.time() > deadline:
            for proc in ctx.processes:
                if proc.is_alive():
                    proc.terminate()
            pytest.fail(
                f"dp={dp_size} cp={cp_size} dynamic_cp={dynamic_cp} workers timed out after {timeout_s}s; "
                "a rank-dependent guard skipped the whitening all_reduce"
            )

    per_sample_tokens: dict[str, dict[int, float]] = {str(i): {} for i in range(len(SEQS))}
    saw_empty_local_batch = False
    for rank in range(world_size):
        with open(case_dir / f"rank{rank}.json") as f:
            payload = json.load(f)
        saw_empty_local_batch = saw_empty_local_batch or bool(payload["empty_local_batch"])
        for sample_idx, pos_map in payload["samples"].items():
            target = per_sample_tokens[sample_idx]
            for pos_str, value in pos_map.items():
                pos = int(pos_str)
                if pos in target:
                    assert target[pos] == pytest.approx(value, abs=1e-5), (
                        f"dp={dp_size} cp={cp_size} dynamic_cp={dynamic_cp}: "
                        f"ranks disagree on sample {sample_idx} token {pos}: {target[pos]} vs {value}"
                    )
                else:
                    target[pos] = value

    merged: dict[str, list[float]] = {}
    for i, (_, resp_len) in enumerate(SEQS):
        token_map = per_sample_tokens[str(i)]
        assert sorted(token_map) == list(range(resp_len)), (
            f"dp={dp_size} cp={cp_size} dynamic_cp={dynamic_cp}: "
            f"reconstructed positions for sample {i} do not cover 0..{resp_len - 1}"
        )
        merged[str(i)] = [token_map[pos] for pos in range(resp_len)]
    return merged, saw_empty_local_batch


@pytest.fixture(scope="module")
def single_rank_baseline(tmp_path_factory) -> dict[str, list[float]]:
    baseline_dir = tmp_path_factory.mktemp("whiten_baseline")
    baseline, _ = _run_case(1, 1, False, baseline_dir)
    reference = _independent_whitened_reference()
    assert sorted(baseline) == sorted(reference)
    for sample_idx, expected in reference.items():
        assert baseline[sample_idx] == pytest.approx(expected, abs=1e-5)
    return baseline


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed with Gloo is required",
)
@pytest.mark.parametrize(("dp_size", "cp_size", "dynamic_cp"), WHITEN_CASES)
def test_whitened_advantages_are_cp_invariant(
    dp_size: int,
    cp_size: int,
    dynamic_cp: bool,
    tmp_path,
    single_rank_baseline: dict[str, list[float]],
) -> None:
    got, saw_empty_local_batch = _run_case(dp_size, cp_size, dynamic_cp, tmp_path)

    if (dp_size, cp_size, dynamic_cp) == (2, 2, False):
        assert saw_empty_local_batch, "expected (dp=2, cp=2) to exercise an empty local CP mask shard"

    assert sorted(got) == sorted(single_rank_baseline)
    for sample_idx, expected in single_rank_baseline.items():
        assert got[sample_idx] == pytest.approx(expected, abs=1e-5), (
            f"dp={dp_size} cp={cp_size} dynamic_cp={dynamic_cp}: "
            f"sample {sample_idx} whitened tokens mismatch single-rank baseline"
        )


def _all_empty_worker(rank: int, world_size: int, cp_size: int, init_file: str, result_dir: str) -> None:
    import torch.distributed as dist

    torch.set_num_threads(1)
    _ensure_megatron_importable()
    cp_rank = rank % cp_size

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        dp_cp_group = dist.new_group(ranks=list(range(world_size)))
        dp_only_groups = [
            dist.new_group(ranks=[r for r in range(world_size) if r % cp_size == c]) for c in range(cp_size)
        ]
        fake_mpu = SimpleNamespace(
            is_pipeline_last_stage=lambda: True,
            get_context_parallel_world_size=lambda: cp_size,
            get_context_parallel_rank=lambda: cp_rank,
            get_tensor_model_parallel_world_size=lambda: 1,
            get_data_parallel_group=lambda with_context_parallel=False: (
                dp_cp_group if with_context_parallel else dp_only_groups[cp_rank]
            ),
        )

        from relax.backends.megatron import cp_utils

        loss_module = importlib.import_module("relax.backends.megatron.loss")
        loss_module.mpu = fake_mpu
        cp_utils.mpu = fake_mpu

        out: dict[str, dict[str, list[int]]] = {}
        for dynamic_cp in (False, True):
            rollout_data = {
                "log_probs": [torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32)],
                "loss_masks": [torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32)],
                "rewards": [1.5, -0.5],
                "response_lengths": [0, 0],
                "total_lengths": [16, 20],
            }
            loss_module.compute_advantages_and_returns(_make_args(dynamic_cp=dynamic_cp), rollout_data)
            out[str(dynamic_cp)] = {
                "adv_lens": [int(t.numel()) for t in rollout_data["advantages"]],
                "ret_lens": [int(t.numel()) for t in rollout_data["returns"]],
            }
        with open(os.path.join(result_dir, f"empty_rank{rank}.json"), "w") as f:
            json.dump(out, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed with Gloo is required",
)
def test_whiten_group_all_empty_responses_is_noop(tmp_path) -> None:
    import torch.multiprocessing as mp

    world_size = 4
    init_file = tmp_path / "empty_gloo_init"
    mp.spawn(
        _all_empty_worker,
        args=(world_size, 2, str(init_file), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    expected = {
        "False": {"adv_lens": [0, 0], "ret_lens": [0, 0]},
        "True": {"adv_lens": [0, 0], "ret_lens": [0, 0]},
    }
    for rank in range(world_size):
        with open(tmp_path / f"empty_rank{rank}.json") as f:
            payload = json.load(f)
        assert payload == expected


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed with Gloo is required",
)
def test_whiten_nonempty_zero_mask_still_raises(monkeypatch, tmp_path) -> None:
    import torch.distributed as dist

    _ensure_megatron_importable()
    from relax.backends.megatron import cp_utils

    loss_module = importlib.import_module("relax.backends.megatron.loss")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path / 'zero-mask-gloo-init'}",
        rank=0,
        world_size=1,
    )
    try:
        fake_mpu = SimpleNamespace(
            is_pipeline_last_stage=lambda: True,
            get_context_parallel_world_size=lambda: 1,
            get_data_parallel_group=lambda with_context_parallel=False: dist.group.WORLD,
        )
        monkeypatch.setattr(loss_module, "mpu", fake_mpu)
        monkeypatch.setattr(cp_utils, "mpu", fake_mpu)

        rollout_data = {
            "log_probs": [torch.zeros(2, dtype=torch.float32)],
            "loss_masks": [torch.zeros(2, dtype=torch.float32)],
            "rewards": [1.0],
            "response_lengths": [2],
            "total_lengths": [6],
        }
        with pytest.raises(ValueError, match="global mask sum"):
            loss_module.compute_advantages_and_returns(_make_args(dynamic_cp=False), rollout_data)
    finally:
        dist.destroy_process_group()
