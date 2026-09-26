from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from packaging.version import parse
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


GLOO_GROUP = None


def init_gloo_group(distributed_timeout_minutes: int = 30):
    """Initialize Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        GLOO_GROUP = dist.new_group(backend="gloo", timeout=timedelta(minutes=distributed_timeout_minutes))
    return GLOO_GROUP


def get_gloo_group():
    """Get the Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        raise RuntimeError("Gloo group has not been initialized. Call _init_gloo_group() first.")
    return GLOO_GROUP


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: str | Backend = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str = None,
    pg_options: Any | None = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version.
    # Use packaging.version for a numeric comparison: a plain string compare is wrong
    # for torch >= 2.10 (e.g. "2.11.0" < "2.6" lexicographically), which would pick the
    # pre-2.6 name "pg_options" and raise TypeError on newer PyTorch.
    pg_options_param_name = "backend_options" if parse(torch.__version__) >= parse("2.6") else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def distributed_masked_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    process_group: dist.ProcessGroup | None = None,
    shift_mean: bool = True,
    epsilon: float = 1e-8,
):
    """Performs whitening on a tensor using global statistics from all
    participating GPUs.

    It calculates the global mean and variance across all ranks in the default
    process group (the WORLD) and uses these global statistics to normalize the
    local data on each rank.

    Args:
        values (torch.Tensor): The local tensor of values to whiten.
        mask (torch.Tensor): The local mask corresponding to the values.
        process_group: The process group for all_reduce.
                      If None, uses the default world group.
        shift_mean (bool): If True, the output is zero-mean. Defaults to True.
        epsilon (float): A small value for numerical stability.

    Returns:
        torch.Tensor: The locally whitened tensor using global statistics.
    """
    # Calculate local intermediate statistics
    local_sum = (values * mask).sum()
    local_sum_sq = ((values**2) * mask).sum()
    local_mask_sum = mask.sum()

    stats_tensor = torch.tensor(
        [local_sum, local_sum_sq, local_mask_sum],
        device=values.device,
        dtype=torch.float32,
    )

    # Aggregate via all_reduce within the DP group
    dist.all_reduce(stats_tensor, group=process_group)

    # Calculate global stats from aggregated results
    global_sum, global_sum_sq, global_mask_sum = stats_tensor

    if global_mask_sum.item() == 0:
        raise ValueError("The global mask sum across all participating GPUs is zero.")

    global_mean = global_sum / global_mask_sum
    global_mean_sq = global_sum_sq / global_mask_sum
    global_var = global_mean_sq - global_mean**2

    # Bessel's correction for unbiased estimate
    if global_mask_sum.item() >= 2:
        bessel_correction = global_mask_sum / (global_mask_sum - 1)
        global_var = global_var * bessel_correction

    # Whiten local data using global stats
    whitened_values = (values - global_mean) * torch.rsqrt(global_var + epsilon)

    if not shift_mean:
        whitened_values += global_mean

    return whitened_values


def distributed_masked_normalize(
    values: torch.Tensor,
    mask: torch.Tensor,
    process_group: dist.ProcessGroup | None = None,
    variance_floor: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize valid values with global population moments.

    This is the REINFORCE++ normalization contract: statistics are computed
    over every mask-nonzero element across the supplied process group, the
    population variance uses ``ddof=0``, and the output is explicitly zero
    outside the mask.

    Returns:
        A tuple of ``(normalized, mean, variance, count)``. The three moment
        tensors contain global values and are identical on every rank.
    """
    if values.shape != mask.shape:
        raise ValueError(f"values and mask must have the same shape, got {values.shape} and {mask.shape}.")
    if variance_floor <= 0:
        raise ValueError(f"variance_floor must be positive, got {variance_floor}.")

    working_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    working_values = values.to(dtype=working_dtype)
    valid_mask = mask.to(device=values.device) != 0
    working_mask = valid_mask.to(dtype=working_dtype)
    masked_values = torch.where(valid_mask, working_values, torch.zeros_like(working_values))

    mean_stats = torch.stack((masked_values.sum(), working_mask.sum()))
    dist.all_reduce(mean_stats, group=process_group)
    global_sum, global_count = mean_stats.unbind()

    # Keep the empty-global-batch invariant on device. Calling ``.item()`` here
    # would synchronize every CUDA training batch with the host.
    torch._assert_async(
        global_count > 0,
        "The global mask sum across all participating ranks is zero.",
    )
    safe_global_count = global_count.clamp_min(1)

    global_mean = global_sum / safe_global_count
    centered = torch.where(valid_mask, working_values - global_mean, torch.zeros_like(working_values))
    centered_sum_sq = centered.square().sum()
    dist.all_reduce(centered_sum_sq, group=process_group)
    global_variance = (centered_sum_sq / safe_global_count).clamp_min(0.0)
    inverse_std = torch.rsqrt(global_variance.clamp_min(variance_floor))
    normalized = torch.where(
        valid_mask,
        (working_values - global_mean) * inverse_std,
        torch.zeros_like(working_values),
    )

    return normalized, global_mean, global_variance, global_count
