# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Multi-hardware backend abstraction layer.
#
# Inspired by verl (https://github.com/verl-project/verl) device.py
# and slime (https://github.com/THUDM/slime) plugin architecture.
#
# This module provides a unified device abstraction that allows Relax to run
# on multiple hardware backends (NVIDIA CUDA, Ascend NPU, AMD ROCm, Kunlunxin XPU,
# PPU, etc.) with minimal code changes throughout the framework.

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Callable, Optional

import torch

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Accelerator type enum
# ---------------------------------------------------------------------------
class AcceleratorType(str, Enum):
    """Supported hardware accelerator types."""

    CUDA = "cuda"  # NVIDIA GPU
    NPU = "npu"  # Ascend NPU (Huawei)
    XPU = "xpu"  # Intel / Kunlunxin XPU
    PPU = "ppu"  # PPU (Enflame / custom)
    KLX = "klx"  # Kunlunxin (P800; masquerades as CUDA: torch.cuda + nccl)
    ROCM = "rocm"  # AMD ROCm (uses 'cuda' device in PyTorch but HIP backend)
    CPU = "cpu"  # CPU fallback


# ---------------------------------------------------------------------------
# Probes — low-level availability checks
# ---------------------------------------------------------------------------
def _is_torch_device_module_available(name: str) -> bool:
    """Check if the ``torch.<name>`` plugin backend exists and is available.

    Covers ``torch.npu`` / ``torch.xpu`` / ``torch.ppu`` style plugin
    backends: they are absent from stock torch builds, so probe defensively
    (attribute access may trigger the plugin import).
    """
    try:
        device_module = getattr(torch, name, None)
        if device_module is None:
            return False
        return device_module.is_available()
    except (ImportError, AttributeError):
        return False


def _is_klx() -> bool:
    """Check if a Kunlunxin (KLX) accelerator is available.

    The Kunlunxin runtime masquerades as CUDA (``torch.cuda`` / ``nccl``), so
    it can't be detected via ``torch.xpu``.  Probe the ``/dev/xpuctrl`` device
    node, which is present on Kunlunxin XPU hosts.
    """
    return os.path.exists("/dev/xpuctrl")


def _is_rocm() -> bool:
    """Check if the current CUDA build is actually AMD ROCm/HIP."""
    return getattr(torch.version, "hip", None) is not None


is_cuda_available: bool = torch.cuda.is_available()
is_npu_available: bool = _is_torch_device_module_available("npu")
is_xpu_available: bool = _is_torch_device_module_available("xpu")
is_ppu_available: bool = _is_torch_device_module_available("ppu")
is_rocm: bool = _is_rocm()


# ---------------------------------------------------------------------------
# Backend specification — single source of truth
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackendSpec:
    """Per-accelerator backend specification.

    Adding a new accelerator requires only a new ``AcceleratorType`` member
    plus one row in :data:`_BACKEND`; everything else is derived from it.
    """

    dist_backend: str  # default collective communication backend
    visible_devices_env: str  # env var controlling visible devices
    ray_resource: str  # Ray resource name
    torch_namespace: str  # torch.<ns> module backing this accelerator
    # Subfamily test for the shared "cuda" namespace; None for plain CUDA.
    cuda_family_probe: Optional[Callable[[], bool]] = None
    allow_non_blocking_copy: bool = True  # async host<->device copies OK?
    allow_pinned_host_memory: bool = True  # pinned host tensors OK?


_BACKEND = {
    AcceleratorType.CUDA: BackendSpec("nccl", "CUDA_VISIBLE_DEVICES", "GPU", "cuda"),
    AcceleratorType.ROCM: BackendSpec("nccl", "CUDA_VISIBLE_DEVICES", "GPU", "cuda", cuda_family_probe=_is_rocm),
    AcceleratorType.NPU: BackendSpec("hccl", "ASCEND_RT_VISIBLE_DEVICES", "NPU", "npu"),
    AcceleratorType.XPU: BackendSpec("xccl", "XPU_VISIBLE_DEVICES", "XPU", "xpu"),
    AcceleratorType.PPU: BackendSpec("eccl", "PPU_VISIBLE_DEVICES", "PPU", "ppu"),
    AcceleratorType.KLX: BackendSpec(
        "nccl",
        "CUDA_VISIBLE_DEVICES",
        "GPU",
        "cuda",
        cuda_family_probe=_is_klx,
        allow_non_blocking_copy=False,
        allow_pinned_host_memory=False,
    ),
    AcceleratorType.CPU: BackendSpec("gloo", "", "CPU", "cpu"),
}


def _spec(accel: AcceleratorType) -> BackendSpec:
    """Return the :class:`BackendSpec` for ``accel``, falling back to
    CUDA's."""
    return _BACKEND.get(accel, _BACKEND[AcceleratorType.CUDA])


_RAY_RESOURCE_TO_ACCEL = {
    spec.ray_resource: accel for accel, spec in reversed(_BACKEND.items()) if spec.ray_resource != "CPU"
}

_RAY_PROBE_ORDER = (
    *(
        spec.ray_resource
        for accel, spec in _BACKEND.items()
        if spec.torch_namespace != "cuda" and accel is not AcceleratorType.CPU
    ),
    _BACKEND[AcceleratorType.CUDA].ray_resource,
)


# ---------------------------------------------------------------------------
# Accelerator detection & resolution
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _detect_accelerator() -> AcceleratorType:
    """Detect the available hardware accelerator.

    Detection order follows specificity: NPU > XPU > PPU > CUDA/ROCm/KLX > CPU.
    Environment variable ``RELAX_DEVICE_TYPE`` can override auto-detection.
    """
    override = Envs.RELAX_DEVICE_TYPE.lower().strip()
    if override:
        for accel in AcceleratorType:
            if override == accel.value:
                logger.info(f"Device type overridden by RELAX_DEVICE_TYPE={override}")
                return accel
        logger.warning(f"Unknown RELAX_DEVICE_TYPE='{override}', falling back to auto-detection")

    # Independent plugin namespaces in _BACKEND order; CPU is the fallback.
    for accel, spec in _BACKEND.items():
        if spec.torch_namespace == "cuda" or accel is AcceleratorType.CPU:
            continue
        if _is_torch_device_module_available(spec.torch_namespace):
            return accel

    # CUDA family: refine by family probe; plain CUDA is the fallback.
    if _is_torch_device_module_available("cuda"):
        for accel, spec in _BACKEND.items():
            if spec.cuda_family_probe is not None and spec.cuda_family_probe():
                return accel
        return AcceleratorType.CUDA

    return AcceleratorType.CPU


def _detect_accelerator_from_ray_cluster() -> Optional[AcceleratorType]:
    """Infer the accelerator type from Ray cluster resources.

    Used as a fallback when the local process has no accelerator (e.g. the Ray
    head node).  Queries ``ray.cluster_resources()`` and maps the first
    non-zero accelerator resource back to an :class:`AcceleratorType`.

    Returns ``None`` if Ray is not initialised or the cluster has no
    accelerator resources.
    """
    try:
        import ray

        if not ray.is_initialized():
            return None
        resources = ray.cluster_resources()
        for ray_key in _RAY_PROBE_ORDER:
            if resources.get(ray_key, 0) > 0:
                accel = _RAY_RESOURCE_TO_ACCEL[ray_key]
                logger.info(
                    f"Local process has no accelerator; detected '{ray_key}' "
                    f"from Ray cluster resources — using {accel.value}"
                )
                return accel
        return None
    except Exception as e:
        logger.debug(f"Ray cluster accelerator detection failed: {e}")
        return None


def _current_accelerator() -> AcceleratorType:
    """Accelerator for actor-configuration values (dist backend, Ray resource
    name, visible-devices env var), unlike the local-only
    :func:`_detect_accelerator`.

    Resolution: local accelerator → Ray cluster resources if Ray is
    initialised → CUDA default (the typical pre-``ray.init`` driver case).
    Deliberately not cached.
    """
    accel = _detect_accelerator()
    if accel != AcceleratorType.CPU:
        return accel

    try:
        import ray

        ray_initialized = ray.is_initialized()
    except Exception:
        ray_initialized = False

    if ray_initialized:
        cluster_accel = _detect_accelerator_from_ray_cluster()
        if cluster_accel is not None:
            return cluster_accel
        return AcceleratorType.CPU

    return AcceleratorType.CUDA


# ---------------------------------------------------------------------------
# Public API — device info
# ---------------------------------------------------------------------------
def get_accelerator_type() -> AcceleratorType:
    """Return the detected :class:`AcceleratorType`."""
    return _detect_accelerator()


def ray_get_device_ids():
    import ray

    if get_accelerator_type() == AcceleratorType.NPU:
        return ray.get_runtime_context().get_accelerator_ids()["NPU"]
    return ray.get_gpu_ids()


def get_device_name() -> str:
    """Return the PyTorch device type string (``'cuda'``, ``'npu'``, ...)."""
    return _spec(_detect_accelerator()).torch_namespace


def get_torch_device_module():
    """Return the ``torch.<device>`` module for the active accelerator (e.g.
    ``torch.cuda``, ``torch.npu``)."""
    ns = _spec(_detect_accelerator()).torch_namespace
    device_module = getattr(torch, ns, None)
    if device_module is None:  # e.g. RELAX_DEVICE_TYPE=npu on a host without torch_npu
        logger.warning(f"torch.{ns} not found, falling back to torch.cuda")
        return torch.cuda
    return device_module


# ---------------------------------------------------------------------------
# Public API — distributed backend
# ---------------------------------------------------------------------------


def get_dist_backend() -> str:
    """Return the default distributed communication backend name.

    Returns ``'nccl'`` for NVIDIA/AMD, ``'hccl'`` for Ascend NPU, etc.

    Uses :func:`_current_accelerator` so callers on a CPU-only Ray driver/head
    (e.g. argparse defaults) get the cluster's backend rather than ``'gloo'``.
    """
    return _spec(_current_accelerator()).dist_backend


# ---------------------------------------------------------------------------
# Public API — environment variables
# ---------------------------------------------------------------------------


def get_visible_devices_env_var() -> str:
    """Return the environment variable name for controlling visible devices.

    E.g. ``'CUDA_VISIBLE_DEVICES'`` for NVIDIA, ``'ASCEND_RT_VISIBLE_DEVICES'``
    for Ascend NPU.

    Uses :func:`_current_accelerator` so a CPU-only Ray driver/head still gets
    the right env var name to read (e.g. when forwarding it to actors).
    """
    return _spec(_current_accelerator()).visible_devices_env


def get_visible_devices() -> Optional[str]:
    """Return the value of the visible-devices environment variable, or
    None."""
    env_var = get_visible_devices_env_var()
    if not env_var:
        return None
    return os.environ.get(env_var)


# ---------------------------------------------------------------------------
# Public API — Ray resource name
# ---------------------------------------------------------------------------


def get_ray_accelerator_name() -> str:
    """Return the Ray resource name for the current accelerator.

    E.g. ``'GPU'`` for NVIDIA/AMD, ``'NPU'`` for Ascend.

    When the local process has no accelerator (e.g. a CPU-only Ray head node),
    falls back to querying Ray cluster resources via
    :func:`_current_accelerator` so placement groups are created with the
    correct resource type.
    """
    return _spec(_current_accelerator()).ray_resource


# ---------------------------------------------------------------------------
# Public API — device operations (thin wrappers)
# ---------------------------------------------------------------------------
class DeviceModule:
    """Single-object facade over the ``torch.<device>`` module returned by
    :func:`get_torch_device_module`.

    Undeclared attributes (``current_device``, ``set_device``, ``Event``,
    ...) forward to the active device module on every access; operations
    with extra dispatch logic are declared explicitly below and take
    precedence over the forwarding.
    """

    def __getattr__(self, name: str):
        return getattr(get_torch_device_module(), name)

    def synchronize(self, device=None) -> None:
        """Synchronize the current (or specified) device."""
        accel = _detect_accelerator()
        if accel == AcceleratorType.CPU:
            return  # no-op for CPU
        mod = get_torch_device_module()
        if device is not None:
            mod.synchronize(device)
        else:
            mod.synchronize()

    def empty_cache(self) -> None:
        """Release all unoccupied cached memory."""
        accel = _detect_accelerator()
        if accel == AcceleratorType.CPU:
            return
        mod = get_torch_device_module()
        mod.empty_cache()

    def memory_allocated(self, device=None) -> int:
        """Return the current GPU memory occupied by tensors in bytes."""
        mod = get_torch_device_module()
        if device is not None:
            return mod.memory_allocated(device)
        return mod.memory_allocated()

    def memory_reserved(self, device=None) -> int:
        """Return the current GPU memory managed by the caching allocator in
        bytes."""
        mod = get_torch_device_module()
        if device is not None:
            return mod.memory_reserved(device)
        return mod.memory_reserved()

    def mem_get_info(self, device=None):
        """Return ``(free, total)`` memory in bytes for the given device."""
        mod = get_torch_device_module()
        if device is not None:
            return mod.mem_get_info(device)
        return mod.mem_get_info()

    def get_device_properties(self, device=None):
        """Return device properties for the given device."""
        mod = get_torch_device_module()
        if device is not None:
            return mod.get_device_properties(device)
        return mod.get_device_properties(mod.current_device())

    def current_stream(self, device=None):
        """Return the currently selected stream for the given device."""
        mod = get_torch_device_module()
        if device is not None:
            return mod.current_stream(device)
        return mod.current_stream()

    def Stream(self, device=None, **kwargs):
        """Create a new stream on the given device."""
        mod = get_torch_device_module()
        if device is not None:
            return mod.Stream(device=device, **kwargs)
        return mod.Stream(**kwargs)

    def stream_context(self, stream):
        """Return a context manager that sets the given stream as the current
        stream.

        Equivalent to ``torch.cuda.stream(s)`` but dispatches to the correct
        device backend (e.g. ``torch.npu.stream(s)`` on Ascend NPU).
        """
        mod = get_torch_device_module()
        return mod.stream(stream)

    def is_initialized(self) -> bool:
        """Return True if the device backend has been initialized.

        Equivalent to ``torch.cuda.is_initialized()`` but dispatches to the
        correct device backend.
        """
        mod = get_torch_device_module()
        if hasattr(mod, "is_initialized"):
            return mod.is_initialized()
        return is_available()


device_module = DeviceModule()


# ---------------------------------------------------------------------------
# Public API — device string helpers
# ---------------------------------------------------------------------------
def make_device_string(index: Optional[int] = None) -> str:
    """Build a device string like ``'cuda:0'`` / ``'npu:2'``; ``index=None``
    resolves to the current device."""
    name = get_device_name()
    if name == "cpu":
        return "cpu"
    if index is None:
        index = device_module.current_device()
    return f"{name}:{index}"


def make_current_torch_device() -> torch.device:
    """Return a ``torch.device`` for the current accelerator and device
    index."""
    return torch.device(make_device_string())


# ---------------------------------------------------------------------------
# Public API — NUMA affinity
# ---------------------------------------------------------------------------
def set_numa_affinity(local_rank: int) -> None:
    """Set NUMA affinity for the given local rank.

    On NVIDIA GPUs, uses pynvml. On other backends, this is a no-op with a
    warning.
    """
    accel = _detect_accelerator()
    if accel in (AcceleratorType.CUDA,):
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            pynvml.nvmlDeviceSetCpuAffinity(handle)
            logger.info(f"Set NUMA affinity for GPU {local_rank}")
            pynvml.nvmlShutdown()
        except ImportError:
            logger.info("pynvml not available, skipping NUMA affinity setup")
        except Exception as e:
            logger.info(f"Failed to set NUMA affinity: {e}")
    elif accel == AcceleratorType.ROCM:
        logger.info("ROCm/HIP environment detected, skipping NUMA affinity setup")
    elif accel == AcceleratorType.NPU:
        logger.info("Ascend NPU environment, skipping NUMA affinity setup (not yet supported)")
    else:
        logger.info(f"NUMA affinity not supported for {accel.value}, skipping")


# ---------------------------------------------------------------------------
# Public API — expandable segments (CUDA-specific, no-op on others)
# ---------------------------------------------------------------------------
def set_expandable_segments(enable: bool) -> None:
    """Configure CUDA memory allocator expandable segments.

    Only effective on NVIDIA CUDA. No-op on other backends.
    """
    if _detect_accelerator() == AcceleratorType.CUDA:
        try:
            torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
        except Exception as e:
            logger.warning(f"Failed to set expandable_segments: {e}")


# ---------------------------------------------------------------------------
# Public API — availability check
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """Return True if any accelerator device is available (not CPU-only)."""
    return _detect_accelerator() != AcceleratorType.CPU


def is_klx() -> bool:
    """Return True if running on a Kunlunxin (KLX) accelerator."""
    return _is_klx()


def use_non_blocking_copy() -> bool:
    """Whether host<->device copies may be asynchronous
    (``non_blocking=True``)."""
    return _spec(_current_accelerator()).allow_non_blocking_copy


def use_pinned_host_memory() -> bool:
    """Whether host-side backup tensors may use pinned memory."""
    return _spec(_current_accelerator()).allow_pinned_host_memory


# ---------------------------------------------------------------------------
# Public API — backend-specific hooks
# ---------------------------------------------------------------------------
def maybe_backend_process_on_model_switch() -> None:
    """Run backend-specific bookkeeping before switching the active model tag.

    Keeps hardware-specific logic out of the framework code: callers in the
    training path invoke this unconditionally and the per-backend behavior is
    decided here.
    """
    if _is_klx():
        from hydrax.Hydra import TensorState

        TensorState.reinitialize_all()


def maybe_backend_barrier_on_weight_chunk(group) -> None:
    """Run a backend-required barrier after sending a weight chunk.

    Keeps hardware-specific synchronization out of the framework code: callers
    in the chunked weight-send loop invoke this unconditionally.
    """
    if _is_klx():
        import torch.distributed as dist

        dist.barrier(group=group)
