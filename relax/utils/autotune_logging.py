# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in timing of actual autotune benchmarks, without changing their
results."""

from __future__ import annotations

import inspect
import json
import os
import socket
import time
from functools import wraps
from pathlib import Path
from typing import Any


def install_autotune_logging(process_group: Any) -> None:
    """Record cache-miss benchmark calls on each training rank.

    Cache hits never enter the wrapped benchmark methods. Candidate timings
    returned to the tuners are unchanged. JSONL intervals use wall-clock time;
    profiler ranges identify overlap with the local capture window directly.
    """
    output_dir = os.environ.get("RELAX_AUTOTUNE_LOG_DIR")
    if not output_dir:
        return

    import torch
    import torch.distributed as dist
    import triton
    from torch._inductor.runtime.runtime_utils import cache_dir
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torch._inductor.select_algorithm import AlgorithmSelectorCache
    from triton.runtime.autotuner import Autotuner

    from relax.utils.logging_utils import get_logger

    if getattr(Autotuner, "_relax_timing_installed", False):
        return
    rank = dist.get_rank(group=process_group)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    host, pid = socket.gethostname(), os.getpid()
    output = directory / f"rank{rank}_{host}_pid{pid}.jsonl"
    logger = get_logger(__name__)
    hooks: list[dict[str, str]] = []

    def wrap(cls: type, method: str, kind: str) -> None:
        if not inspect.isfunction(cls.__dict__.get(method)):
            raise RuntimeError(f"Autotune timing requires an instance method: {cls.__name__}.{method}")
        original = getattr(cls, method)

        @wraps(original)
        def observed(self: Any, *args: Any, **kwargs: Any) -> Any:
            function = getattr(self, "base_fn", getattr(self, "fn", None))
            name = getattr(function, "__name__", type(function).__name__)
            if kind == "inductor_algorithm_search" and args and isinstance(args[0], str):
                name = args[0]
            start = time.time_ns()
            counter_start = time.perf_counter_ns()
            success = False
            try:
                with torch.profiler.record_function(f"[AUTOTUNE]{kind}|{name}"):
                    result = original(self, *args, **kwargs)
                success = True
                return result
            finally:
                end = time.time_ns()
                counter_end = time.perf_counter_ns()
                event = {
                    "kind": kind,
                    "kernel": name,
                    "rank": rank,
                    "host": host,
                    "pid": pid,
                    "start_ns": start,
                    "end_ns": end,
                    "duration_s": (counter_end - counter_start) / 1e9,
                    "success": success,
                    "level": "search" if kind.endswith("search") else "candidate",
                    "triton_cache_dir": str(triton.knobs.cache.dir),
                }
                try:
                    with output.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event, sort_keys=True) + "\n")
                except OSError:
                    # Preserve the original benchmark result or exception.
                    # Any logging failure invalidates the capture audit.
                    logger.warning("Autotune event logging failed: %s", output, exc_info=True)

        setattr(cls, method, observed)
        hooks.append({"class": f"{cls.__module__}.{cls.__qualname__}", "method": method, "kind": kind})

    wrap(Autotuner, "_bench", "triton_candidate")
    wrap(CachingAutotuner, "benchmark_all_configs", "inductor_search")
    wrap(CachingAutotuner, "bench", "inductor_candidate")
    wrap(AlgorithmSelectorCache, "autotune", "inductor_algorithm_search")
    Autotuner._relax_timing_installed = True
    manifest = {
        "rank": rank,
        "host": host,
        "pid": pid,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "hooks": hooks,
        "triton_cache_dir": str(triton.knobs.cache.dir),
        "inductor_cache_dir": cache_dir(),
        "cache_env": {
            key: os.environ[key]
            for key in (
                "FLA_CACHE_RESULTS",
                "TRITON_CACHE_AUTOTUNING",
                "TORCHINDUCTOR_FX_GRAPH_CACHE",
                "TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE",
            )
            if key in os.environ
        },
    }
    output.with_suffix(".installed.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Autotune benchmark timing installed: %s", json.dumps(manifest, sort_keys=True))
