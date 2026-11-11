# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Process-local compatibility for the pinned TQ CPU thread context."""

from contextlib import contextmanager
from typing import Any, Iterator


def install_tq_topology_cache() -> None:
    """Cache topology for TQ RPCs without changing thread targets or restore.

    TQ imports the context into both client and storage modules. Install before
    worker threads start; replacing common alone would leave those aliases
    live.
    """
    import os

    import psutil
    import torch
    import transfer_queue.client as client
    import transfer_queue.storage.simple_storage as storage
    from transfer_queue.utils.common import DEFAULT_TORCH_NUM_THREADS

    from relax.utils.logging_utils import get_logger

    modules = (client, storage)
    pending = [
        m for m in modules if not getattr(m.limit_pytorch_auto_parallel_threads, "_relax_cached_topology", False)
    ]
    if not pending:
        return
    physical_cores = next(
        (
            getattr(m.limit_pytorch_auto_parallel_threads, "_relax_physical_cores", None)
            for m in modules
            if getattr(m.limit_pytorch_auto_parallel_threads, "_relax_cached_topology", False)
        ),
        None,
    )
    if physical_cores is None:
        physical_cores = psutil.cpu_count(logical=False)
    if not isinstance(physical_cores, int) or physical_cores <= 0:
        return

    def wrap(original: Any) -> Any:
        @contextmanager
        def cached_limit(target_num_threads: int | None = None, info: str = "") -> Iterator[None]:
            if not isinstance(target_num_threads, int) or not 0 < target_num_threads <= physical_cores:
                with original(target_num_threads=target_num_threads, info=info):
                    yield
                return
            try:
                torch.set_num_threads(target_num_threads)
                yield
            finally:
                torch.set_num_threads(DEFAULT_TORCH_NUM_THREADS)

        cached_limit._relax_cached_topology = True
        cached_limit._relax_physical_cores = physical_cores
        return cached_limit

    for module in pending:
        module.limit_pytorch_auto_parallel_threads = wrap(module.limit_pytorch_auto_parallel_threads)
        get_logger(__name__).info(
            "Reference TQ CPU topology cached: pid=%s module=%s physical=%s current_threads=%s restore_threads=%s",
            os.getpid(),
            module.__name__,
            physical_cores,
            torch.get_num_threads(),
            DEFAULT_TORCH_NUM_THREADS,
        )
