# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import os
import time
import traceback
from pathlib import Path

import torch

from relax.utils import device as device_utils
from relax.utils.http_utils import router_worker_base_urls
from relax.utils.logging_utils import get_logger
from relax.utils.memory_utils import print_memory


logger = get_logger(__name__)


def _get_rank_tag() -> str:
    """Build a rank tag string like ``rank0_dp0_tp0_pp0`` from Megatron mpu."""
    global_rank = torch.distributed.get_rank()
    from megatron.core import mpu

    dp = mpu.get_data_parallel_rank(with_context_parallel=True)
    tp = mpu.get_tensor_model_parallel_rank()
    pp = mpu.get_pipeline_model_parallel_rank()

    return f"rank{global_rank}_dp{dp}_tp{tp}_pp{pp}"


class TrainProfiler:
    def __init__(self, args):
        self.args = args
        self._torch_profiler_overall = None
        self._memory_profiler_overall = None

        if args.use_pytorch_profiler and ("train_overall" in args.profile_target):
            self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")
            logger.info(f"PyTorch profiler for overall training is enabled, dump dir: {_get_train_trace_dir(args)}")

        if args.record_memory_history and ("train_overall" in args.profile_target):
            self._memory_profiler_overall = _BaseMemoryProfiler.create(args)
            self._memory_profiler_overall.start()
            logger.info(f"Memory profiler for overall training is enabled, dump dir: {args.memory_snapshot_dir}")

    def on_init_end(self):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.start()

    def step(self, rollout_id: int):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.step()

        if (
            self._memory_profiler_overall is not None
            and ((s := self.args.memory_snapshot_num_steps) is not None)
            and (rollout_id == s - 1)
        ):
            self._memory_profiler_overall.stop()

    def iterate_train_actor(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_actor")

    def iterate_train_log_probs(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_log_probs")


def _profile_simple_loop(iterator, args, name):
    if not (args.use_pytorch_profiler and (name in args.profile_target)):
        yield from iterator
        return

    torch_profiler = _create_torch_profiler(args, name=name)
    torch_profiler.start()
    for item in iterator:
        yield item
        torch_profiler.step()


def _get_trace_base_dir(args):
    """Return the base directory for all profiler outputs.

    Uses ``./traces/<tb_experiment_name>`` as the default location. Falls back
    to ``./traces/<timestamp>`` when ``--tb-experiment-name`` is not set.
    """
    task_name = getattr(args, "tb_experiment_name", None)
    if task_name is None:
        from datetime import datetime

        task_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("traces", task_name)


def _get_train_trace_dir(args):
    """Return the output directory for training profiler traces.

    Uses ``./traces/<tb_experiment_name>/train_trace`` as the default location.
    """
    return os.path.join(_get_trace_base_dir(args), "train_trace")


def _create_torch_profiler(args, name):
    trace_dir = _get_train_trace_dir(args)
    worker_name = f"{name}_{_get_rank_tag()}"
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            # TODO the train_actor and train_log_probs ones may need to have different args to control step
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start + 1,  # end is inclusive
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            trace_dir,
            worker_name=worker_name,
            use_gzip=True,
        ),
        record_shapes=True,
        with_stack=args.profile_with_stack,
        profile_memory=args.profile_with_memory,
        with_flops=args.profile_with_flops,
    )


class _BaseMemoryProfiler:
    @staticmethod
    def create(args):
        c = {
            "torch": _TorchMemoryProfiler,
            "memray": _MemrayMemoryProfiler,
        }[args.memory_recorder]
        return c(args)

    def __init__(self, args):
        snapshot_dir = getattr(args, "memory_snapshot_dir", None)
        if snapshot_dir is None:
            snapshot_dir = os.path.join(_get_trace_base_dir(args), "memory_snapshot")
        os.makedirs(snapshot_dir, exist_ok=True)
        rank_tag = _get_rank_tag()
        self._path_dump = (
            Path(snapshot_dir) / f"memory_snapshot_time{time.time()}_{rank_tag}_{args.memory_snapshot_path}"
        )

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class _TorchMemoryProfiler(_BaseMemoryProfiler):
    def start(self):
        logger.info("Attach OOM dump memory history.")

        # Memory snapshot APIs are currently CUDA-specific.
        # On non-CUDA backends, log a warning and skip.
        device_mod = device_utils.get_torch_device_module()
        if not hasattr(device_mod, "memory"):
            logger.warning(
                f"Memory snapshot profiling is not supported on {device_utils.get_device_name()} backend, skipping."
            )
            return

        device_mod.memory._record_memory_history(
            max_entries=1000000,
            # record stack information for the trace events
            # trace_alloc_record_context=True,
            stacks="all",
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            logger.info(
                f"Observe OOM, will dump snapshot to {self._path_dump}. ({device=} {alloc=} {device_alloc=} {device_free=}; stacktrace is as follows)"
            )
            traceback.print_stack()
            device_mod.memory._dump_snapshot(self._path_dump)
            print_memory("when oom")

        if hasattr(torch._C, "_cuda_attach_out_of_memory_observer"):
            torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    def stop(self):
        logger.info(f"Dump memory snapshot to: {self._path_dump}")
        device_mod = device_utils.get_torch_device_module()
        if not hasattr(device_mod, "memory"):
            logger.warning(
                f"Memory snapshot profiling is not supported on {device_utils.get_device_name()} backend, skipping."
            )
            return
        device_mod.memory._dump_snapshot(self._path_dump)
        device_mod.memory._record_memory_history(enabled=None)


class _MemrayMemoryProfiler(_BaseMemoryProfiler):
    def __init__(self, args):
        super().__init__(args)
        assert args.memory_snapshot_num_steps is not None, "In memray, must provide --memory-snapshot-num-steps"

    def start(self):
        logger.info("Memray tracker started.")
        import memray

        self._tracker = memray.Tracker(
            file_name=self._path_dump,
            native_traces=True,
        )
        self._tracker.__enter__()

    def stop(self):
        logger.info(f"Memray tracker stopped and dump snapshot to: {self._path_dump}")
        self._tracker.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# SGLang profiling orchestration
#
# These helpers coordinate profiling across all SGLang engines by discovering
# worker URLs from the router and issuing HTTP start/stop requests.
# ---------------------------------------------------------------------------


def _get_sglang_trace_dir(args) -> str:
    """Return the base output directory for SGLang profiler traces.

    Uses the user-specified ``--sglang-profile-output-dir`` if set, otherwise
    falls back to ``./traces/<tb_experiment_name>/sglang_trace``.
    """
    base_dir = getattr(args, "sglang_profile_output_dir", None)
    if base_dir is None:
        base_dir = os.path.join(_get_trace_base_dir(args), "sglang_trace")
    return base_dir


def _should_profile_sglang(args, rollout_id: int) -> bool:
    """Determine whether SGLang profiling should be active for the given
    rollout step.

    Resolution order:
    1. ``--sglang-profile`` must be enabled (master switch).
    2. ``--sglang-profile-steps`` (explicit list) takes precedence if set.
    3. ``--sglang-profile-step-start`` / ``--sglang-profile-step-end`` (range)
       is checked next.  Both bounds are *inclusive* and use absolute rollout IDs.
    4. If neither is set, every step is profiled.
    """
    if not getattr(args, "sglang_profile", False):
        return False

    profile_steps = getattr(args, "sglang_profile_steps", None)
    if profile_steps is not None:
        return rollout_id in profile_steps

    step_start = getattr(args, "sglang_profile_step_start", None)
    step_end = getattr(args, "sglang_profile_step_end", None)
    if step_start is not None or step_end is not None:
        lo = step_start if step_start is not None else 0
        hi = step_end if step_end is not None else float("inf")
        return lo <= rollout_id <= hi

    # No filter specified — profile every step.
    return True


async def _get_sglang_worker_urls(args) -> list[str]:
    """Discover SGLang worker URLs from the router."""
    import sglang_router
    from packaging.version import parse

    from relax.utils.http_utils import get

    if parse(sglang_router.__version__) <= parse("0.2.1") or getattr(args, "use_slime_router", False):
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        return router_worker_base_urls(response["urls"])
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        return router_worker_base_urls(worker["url"] for worker in response["workers"])


async def start_sglang_profile(args, rollout_id: int) -> None:
    """Start torch profiling on all SGLang engines if ``--sglang-profile`` is
    enabled.

    Profile traces are organized as::

        traces/<tb_experiment_name>/sglang_trace/rollout_<rollout_id>/<trace files>

    When ``--sglang-profile-output-dir`` is explicitly set, that path is used
    as the base instead.
    """
    if not _should_profile_sglang(args, rollout_id):
        return

    from relax.utils.http_utils import post

    # Build per-step output directory:  <base>/rollout_<id>
    base_dir = _get_sglang_trace_dir(args)
    step_dir = os.path.join(base_dir, f"rollout_{rollout_id}")
    os.makedirs(step_dir, exist_ok=True)

    num_steps = getattr(args, "sglang_profile_num_steps", None)
    if num_steps is not None and num_steps < 0:
        num_steps = None

    urls = await _get_sglang_worker_urls(args)
    base_payload = {
        "output_dir": step_dir,
        "num_steps": num_steps,
        "activities": getattr(args, "sglang_profile_activities", None),
        "profile_by_stage": getattr(args, "sglang_profile_by_stage", False),
        "with_stack": getattr(args, "sglang_profile_with_stack", False),
        "record_shapes": getattr(args, "sglang_profile_record_shapes", False),
    }

    logger.info(
        f"Starting SGLang profiling on {len(urls)} engines for rollout step {rollout_id}, "
        f"output_dir={step_dir}, num_steps={num_steps}"
    )
    tasks = []
    for i, url in enumerate(urls):
        payload = {**base_payload, "profile_prefix": f"engine{i}"}
        tasks.append(post(f"{url}/start_profile", payload))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for url, result in zip(urls, results, strict=False):
        if isinstance(result, BaseException):
            logger.warning(f"Failed to start profile on {url}: {result}")
        else:
            logger.info(f"Started profiling on {url}")


async def stop_sglang_profile(args, rollout_id: int) -> None:
    """Stop torch profiling on all SGLang engines if ``--sglang-profile`` is
    enabled."""
    if not _should_profile_sglang(args, rollout_id):
        return

    # If num_steps was set, SGLang auto-stops — skip explicit stop.
    if getattr(args, "sglang_profile_num_steps", -1) > 0:
        return

    from relax.utils.http_utils import post

    urls = await _get_sglang_worker_urls(args)
    logger.info(f"Stopping SGLang profiling on {len(urls)} engines for rollout step {rollout_id}")
    tasks = [post(f"{url}/stop_profile", {}) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for url, result in zip(urls, results, strict=False):
        if isinstance(result, BaseException):
            logger.warning(f"Failed to stop profile on {url}: {result}")
        else:
            logger.info(f"Stopped profiling on {url}")
