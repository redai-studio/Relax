import socket
import time
import uuid
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from relax.backends.sglang._scale_weight_sync_precheck import launch_probe, transport_fingerprint
from relax.utils import device as device_utils
from relax.utils import scale_utils
from relax.utils.distributed_utils import get_gloo_group, init_process_group
from relax.utils.env import Envs
from relax.utils.http_utils import _wrap_ipv6
from relax.utils.logging_utils import get_logger

from ..weight_conversion import convert_to_hf
from .common import all_gather_param, named_params_and_buffers


logger = get_logger(__name__)


class UpdateWeightFromDistributed:
    """Update distributed engines via NCCL.

    Each PP rank: group "slime-pp_{pp_rank}", only DP=TP=0 broadcasts. Non-
    expert (TP) and expert (EP) params separate.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """Initialize.

        Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """Create NCCL "slime-pp_{pp_rank}" if PP source (DP=TP=0).

        Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}"

        if self._is_pp_src_rank:
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args, self._group_name, self._model_update_groups, self.rollout_engines
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args,
                self._group_name,
                rollout_engines,
                engine_gpu_counts=engine_gpu_counts,
            )

    def disconnect_rollout_engines(self) -> None:
        if not getattr(self, "_is_pp_src_rank", False) or self._model_update_groups is None:
            return
        disconnect_rollout_engines_from_distributed(
            self.args, self._group_name, self._model_update_groups, self.rollout_engines
        )
        self._model_update_groups = None

    @torch.no_grad()
    def update_weights(self) -> None:
        """Pause → flush → non-expert (TP) → expert (EP) → continue.

        Progress on PP source.
        """
        self.weight_version += 1

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            # int4/fp4 pre_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        converted_named_tensors = []
        # non expert params
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            buffer_size = self._update_weight_from_distributed(
                name, param, converted_named_tensors, buffer_size, pbar=pbar
            )

        if converted_named_tensors:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        named_tensors = []
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." not in name:
                continue
            buffer_size = self._update_expert_weight_from_distributed(
                name, param, named_tensors, buffer_size, pbar=pbar
            )

        if named_tensors:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            # int4/fp4 post_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _update_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int | None:
        """
        Non-expert: gather TP → rm pad → HF → buffer (flush if full). All gather, PP source buffers.
        Returns updated bytes on source, None on non-source.
        """
        param = all_gather_param(self.args, name, param)
        if not self._is_pp_src_rank:
            return

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
            buffer_size = 0
        converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        buffer_size += param_size
        return buffer_size

    def _update_expert_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int:
        """
        Expert: gather TP → rm pad → buffer. EP gather + HF deferred. Threshold × EP size.
        """
        param = all_gather_param(self.args, name, param)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_expert_bucket_weights_from_distributed(
        self, named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Gather EP → HF → broadcast.

        Clears buffer.
        """
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=device_utils.make_current_torch_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_pp_src_rank:
            return

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        self._update_bucket_weights_from_distributed(converted_hf_tensors, pbar)

    def _update_bucket_weights_from_distributed(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Lock → broadcast → clear → unlock → pbar++.

        Lock prevents NCCL deadlock.
        """
        # lock the rollout engines to prevent dead lock on broadcast.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        refs = update_weights_from_distributed(
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_engines,
            converted_named_tensors,
        )

        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())
        pbar.update(1)


def precheck_colocate_receiver(
    engines: list, local_gpu_count: int, timeout: float
) -> tuple[bool, bool, tuple[ObjectRef, ...]]:
    """Verify sender-to-candidate NCCL.

    Return any receiver probes still needing an exit ACK.
    """
    if timeout <= 5 or not engines:
        return False, True, ()
    deadline = time.monotonic() + timeout
    refs = []
    process = None
    success = False
    try:
        if device_utils.get_dist_backend() != "nccl":
            return False, True, ()
        device = torch.cuda.current_device()
        if torch.cuda.mem_get_info(device)[0] < Envs.RELAX_SCALE_WEIGHT_SYNC_PRECHECK_MIN_FREE_BYTES:
            logger.warning("[Colocate FULL] Insufficient GPU memory on the NCCL probe sender")
            return False, True, ()
        address = _wrap_ipv6(ray._private.services.get_node_ip_address())
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = str(sock.getsockname()[1])
        token = f"colocate-precheck-{uuid.uuid4().hex}"
        world_size = 1 + len(engines) * local_gpu_count
        probe_timeout = deadline - time.monotonic() - 4  # reserve time for child exit ACKs
        if probe_timeout <= 0:
            return False, True, ()
        expires_at = time.time() + probe_timeout
        process = launch_probe(address, port, 0, world_size, device, probe_timeout, token)
        for i, engine in enumerate(engines):
            refs.append(
                engine.run_scale_weight_sync_precheck.remote(
                    master_address=address,
                    ports=",".join([port] * local_gpu_count),
                    group_rank=1 + i * local_gpu_count,
                    run_token=token,
                    tp_size=local_gpu_count,
                    timeout_secs=probe_timeout,
                    world_size=world_size,
                    expires_at=expires_at,
                )
            )
        success = process.wait(timeout=max(0, deadline - time.monotonic() - 4)) == 0
    except Exception:
        logger.exception("[Colocate FULL] NCCL precheck failed")
    finally:
        sender_stopped = process is None
        if process is not None:
            try:
                scale_utils._terminate_probe_process(process, grace_secs=0.5)
                sender_stopped = process.poll() is not None
            except OSError:
                logger.exception("[Colocate FULL] Could not stop the sender probe")
    if refs:
        try:
            replies = ray.get(refs, timeout=max(0, deadline - time.monotonic()))
            if all(reply.get("stopped", True) for reply in replies):
                sender = {"fingerprint": transport_fingerprint()}
                compatible = all(
                    reply.get("success")
                    and isinstance(reply.get("fingerprint"), dict)
                    and scale_utils._scale_weight_sync_precheck_fingerprints_match(sender, reply)[0]
                    for reply in replies
                )
                return success and sender_stopped and compatible, sender_stopped, ()
        except Exception:
            logger.exception("[Colocate FULL] Receiver probes did not confirm exit before the deadline")
    return False, sender_stopped, tuple(refs)


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
    max_retries: int = 3,
    *,
    timeout_secs: float | None = None,
    remember_group: Callable[[dist.ProcessGroup], None] | None = None,
) -> dist.ProcessGroup:
    """Create NCCL group: training rank 0 + all engine GPUs. Blocks until
    joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)

    master_address = ray._private.services.get_node_ip_address()
    world_size = sum(engine_gpu_counts) + 1  # +1 for training rank 0

    cumulative = [0]
    for c in engine_gpu_counts:
        cumulative.append(cumulative[-1] + c)

    last_error = None
    dist_backend = device_utils.get_dist_backend()
    for attempt in range(1, max_retries + 1):
        started_at = time.monotonic()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        timeout_args = {"timeout": timeout_secs} if timeout_secs is not None else {}
        refs = [
            engine.init_weights_update_group.remote(
                master_address,
                master_port,
                cumulative[i] + 1,
                world_size,
                group_name,
                backend=dist_backend,
                **timeout_args,
            )
            for i, engine in enumerate(rollout_engines)
        ]
        try:
            model_update_groups = init_process_group(
                backend=dist_backend,
                init_method=f"tcp://{_wrap_ipv6(master_address)}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=group_name,
                timeout=(
                    timedelta(minutes=args.distributed_timeout_minutes)
                    if timeout_secs is None
                    else timedelta(seconds=timeout_secs)
                ),
            )
            if remember_group is not None:
                remember_group(model_update_groups)
            if timeout_secs is None:
                ray.get(refs)
            else:
                acknowledgments = ray.get(refs, timeout=max(0, timeout_secs - (time.monotonic() - started_at)))
                if any(not isinstance(ack, dict) or ack.get("success") is not True for ack in acknowledgments):
                    raise RuntimeError("Rollout receiver did not acknowledge creation of the NCCL group")
            return model_update_groups
        except Exception as e:
            last_error = e
            logger.warning(
                f"Failed to connect rollout engines (attempt {attempt}/{max_retries}, port={master_port}): {e}",
                exc_info=(attempt == max_retries),
            )
            try:
                ray.get(refs, timeout=5)
            except Exception:
                pass
            if attempt < max_retries:
                time.sleep(5.0 * attempt)

    raise RuntimeError(f"Failed to connect rollout engines after {max_retries} attempts") from last_error


def disconnect_rollout_engines_from_distributed(
    args, group_name, model_update_groups, rollout_engines, *, strict: bool = False
):
    """Destroy NCCL on training and engines."""
    kwargs = {"strict": True} if strict else {}
    refs = [engine.destroy_weights_update_group.remote(group_name, **kwargs) for engine in rollout_engines]
    dist.destroy_process_group(model_update_groups)
    results = ray.get(refs, timeout=35) if strict else ray.get(refs)
    if strict and any(not isinstance(result, dict) or result.get("success") is not True for result in results):
        raise RuntimeError(f"Engine did not acknowledge destruction of weight-update group {group_name}")
    # Wait for NCCL socket ports to be released by the OS
    time.sleep(2.0)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    *,
    timeout_secs: float | None = None,
) -> list[ObjectRef]:
    """Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines)."""
    timeout_args = {"timeout": timeout_secs} if timeout_secs is not None else {}
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
            **timeout_args,
        )
        for engine in rollout_engines
    ]

    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs


def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
):
    """Trigger post-process for int4/fp4 quantization on all rollout
    engines."""
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
            )
            for engine in rollout_engines
        ]
    )
