# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import gc
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist

from relax.utils import device as device_utils
from relax.utils.device import device_module
from relax.utils.log_style import format_role_tag
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


# Per-process role tag ("actor", "critic", "genrm", ...) prepended to
# print_memory output so colocate PPO logs are readable at a glance.
# Set once by TrainRayActor.init(); each Ray actor is its own process,
# so there's no cross-role contamination.
_ROLE: str | None = None


def set_role(role: str | None) -> None:
    global _ROLE
    _ROLE = role


def clear_memory(clear_host_memory: bool = False):
    device_module.synchronize()
    gc.collect()
    device_module.empty_cache()
    if clear_host_memory:
        if device_utils.is_npu_available:
            torch.npu.host_empty_cache()
        else:
            torch._C._host_emptyCache()


def available_memory():
    dev = device_module.current_device()
    free, total = device_module.mem_get_info(dev)
    return {
        "device": str(dev),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(device_module.memory_allocated(dev)),
        "reserved_GB": _byte_to_gb(device_module.memory_reserved(dev)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    role_tag = f"{format_role_tag(_ROLE)} " if _ROLE else ""
    logger.info(
        f"{role_tag}[Rank {dist.get_rank()}] Memory-Usage {msg}"
        f"{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info


@contextmanager
def log_step_peak_memory(rollout_id: int, step: int, ranks: dict[str, int], role: str = "actor") -> Iterator[None]:
    """Log this process's allocator peaks over one training iteration.

    Resetting peaks neither releases cached memory nor synchronizes the device.
    These counters exclude allocations outside the PyTorch allocator, including
    some NCCL memory, and do not measure whole-device physical memory peaks.
    Values use the same 1024**3-byte GB convention as available_memory().
    """
    reset = getattr(device_module, "reset_peak_memory_stats", None)
    max_allocated = getattr(device_module, "max_memory_allocated", None)
    max_reserved = getattr(device_module, "max_memory_reserved", None)
    if not all(callable(fn) for fn in (reset, max_allocated, max_reserved)):
        logger.warning("Step-Memory-Peak unavailable: backend does not expose allocator peak counters")
        yield
        return

    try:
        device = device_module.current_device()
        reset(device)
    except (AttributeError, NotImplementedError, RuntimeError) as exc:
        logger.warning("Step-Memory-Peak unavailable: %s", exc)
        yield
        return

    status = "failed"
    try:
        yield
        status = "complete"
    finally:
        try:
            allocated = max_allocated(device)
            reserved = max_reserved(device)
        except (AttributeError, NotImplementedError, RuntimeError) as exc:
            logger.warning("Step-Memory-Peak unavailable after step: %s", exc)
        else:
            memory = {
                "device": str(device),
                "peak_allocated_GB": _byte_to_gb(allocated),
                "peak_reserved_GB": _byte_to_gb(reserved),
            }
            # Ray ignores words containing digits when deduplicating worker logs.
            # Keep a separate all-letter identity so every rank and step survives.
            key = f"{role}-rank-{ranks['global_rank']}-rollout-{rollout_id}-step-{step}".translate(
                str.maketrans("0123456789", "abcdefghij")
            )
            logger.info(
                "Step-Memory-Peak key=%s role=%s phase=train rollout_id=%d step=%d status=%s ranks=%s memory=%s",
                key,
                role,
                rollout_id,
                step,
                status,
                ranks,
                memory,
            )
