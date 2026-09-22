# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from time import monotonic
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from relax.backends.megatron.misc_utils import strip_param_name_prefix
from relax.utils import device as device_utils
from relax.utils import megatron_bridge_utils
from relax.utils.device import make_current_torch_device
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.logging_utils import get_logger
from relax.utils.megatron_peft_utils import (
    LORA_ADAPTER_NAME,
    is_lora_adapter_mode,
    is_lora_adapter_param,
    is_lora_enabled,
    is_lora_merge_mode,
)

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .hf_weight_iterator_base import HfWeightIteratorBase
from .lora_adapter_sync import LoraAdapterSync
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)


logger = get_logger(__name__)


class UpdateWeightFromTensor:
    """Update rollout engines from tensor dict:

    load(dict→GPU) → broadcast PP/EP(GPU NCCL) → gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: GPU→CPU serialize → gather_object(Gloo CPU, collects from rollout_num_gpus_per_engine ranks) → Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
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
        """Compute param buckets.

        IPC Gloo groups are created later in ``connect_rollout_engines`` once
        ``engine_gpu_counts`` is known.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0

        self.lora_enabled = is_lora_enabled(args)
        self.lora_merge_mode = is_lora_merge_mode(args) if self.lora_enabled else False
        self.lora_adapter_mode = is_lora_adapter_mode(args) if self.lora_enabled else False

        # Adapter-mode incremental-sync state (base-once + adapter-delta protocol) lives in the
        # shared LoraAdapterSync helper; the backend keeps only its in-memory transport below.
        self._lora_sync = LoraAdapterSync(args, model) if self.lora_adapter_mode else None

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._model_update_groups = None
        self.distributed_rollout_engines: list[ActorHandle] = []

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """Split colocated/distributed engines.

        Global source rank (DP=TP=PP=0) creates NCCL for distributed. Map ranks
        to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Route via CUDA IPC only for engines on the same Ray node as the actor;
        # cross-node IPC fails with cudaErrorMapBufferObjectFailed.
        engine_node_ids = [
            info.get("node_id", "")
            for info in ray.get([engine.get_pid_and_node_id.remote() for engine in rollout_engines])
        ]
        local_actor_node_id = ray.get_runtime_context().get_node_id()
        gathered_actor_node_ids: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_actor_node_ids, local_actor_node_id, group=get_gloo_group())
        actor_node_id_set = {nid for nid in gathered_actor_node_ids if nid}

        colocate_engine_nums = 0
        for engine_node_id in engine_node_ids:
            if not engine_node_id or engine_node_id not in actor_node_id_set:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "slime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Create IPC Gloo gather groups (only on first call; partitioning is
        # fixed across reconnects).
        if self._ipc_gather_group is None:
            for i in range(colocate_engine_nums):
                group_ranks = list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if dist.get_rank() in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = colocate_gpu_offsets[i]

        # Map training ranks to colocated engine actors.
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self) -> None:
        """version++, flush caches, process buckets with pipelining.

        Pipelining: overlap chunk N's IPC transfer with chunk N+1's HF
        conversion + serialization + Gloo gather.  At most two chunks'
        GPU tensors are alive simultaneously (bounded by
        ``update_weight_buffer_size``).

        In LoRA merge mode the adapters are folded into the base weights during
        HF export so the rollout engine serves one merged model; otherwise all
        weights are sent as-is.
        """
        # LoRA adapter mode has its own base-once + adapter-delta sync protocol.
        if self.lora_enabled and self.lora_adapter_mode:
            self._update_weights_adapter_mode()
            return

        self.weight_version += 1

        # Pause/flush must cover both IPC and distributed-broadcast engines,
        # otherwise NCCL-path engines see torn reads and stale radix-KV cache.
        all_engines = list(self.rollout_engines) + list(self.distributed_rollout_engines)

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in all_engines])
            ray.get([engine.flush_cache.remote() for engine in all_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=all_engines,
                )
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        if self.lora_enabled and self.lora_merge_mode:
            renamed = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
            adapter_keys = [k for k in renamed if is_lora_adapter_param(k)]
            n_expert = sum(1 for k in adapter_keys if ".experts." in k)
            logger.info(
                "[lora-merge] colocate sync v=%d: %d/%d backup tensors are LoRA adapters "
                "(%d non-expert + %d expert) available for splicing",
                self.weight_version,
                len(adapter_keys),
                len(renamed),
                len(adapter_keys) - n_expert,
                n_expert,
            )
            n_adapter_backup = len(adapter_keys)
            if n_adapter_backup == 0:
                logger.error(
                    "[lora-merge] NO adapter tensors in backup dict — merge would degrade to "
                    "base-only. Check adapter naming / weights_getter output."
                )
            export_ctx = megatron_bridge_utils.splice_adapter_weights(renamed)
        else:
            export_ctx = nullcontext()

        # Pipeline: when chunk N's IPC refs are in-flight on the engine,
        # chunk N+1's HF conversion + serialize + gather can proceed in
        # parallel.  We defer ``ray.get`` to the *next* iteration so the
        # two stages overlap.
        prev_refs: list[ObjectRef] = []
        prev_long_lived_tensors = None
        with export_ctx:
            for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
                refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
                # Wait for the *previous* chunk's IPC to finish before
                # releasing its GPU tensors.
                if prev_refs:
                    ray.get(prev_refs)
                del prev_long_lived_tensors
                prev_refs = refs
                prev_long_lived_tensors = long_lived_tensors
                # Backend-specific per-chunk synchronization is handled in device
                # utils so this path stays hardware-agnostic.
                device_utils.maybe_backend_barrier_on_weight_chunk(group=get_gloo_group())
            # Drain the last chunk.
            if prev_refs:
                ray.get(prev_refs)
            del prev_long_lived_tensors

        # All ranks must finish sending before rank 0 triggers Marlin repack,
        # otherwise engines in slower gather groups may still be processing
        # weight chunks when their parameters get reshaped by post_process.
        dist.barrier(group=get_gloo_group())

        # int4/fp4 post_process
        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=all_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in all_engines])
        dist.barrier(group=get_gloo_group())

    def _update_weights_adapter_mode(self) -> None:
        """LoRA adapter mode: sync base once, then push only the adapter each
        step.

        Base is frozen in LoRA training (only adapter params get gradients) and
        SGLang restores it across colocate sleep/wake via
        enable_weights_cpu_backup, so it is synced exactly once at init; every
        subsequent step pushes only the adapter.
        """
        # Adapter push currently targets only colocated IPC engines (see _push_lora_adapter).
        # Distributed (NCCL) engines are not yet supported — fail loud rather than silently
        # leave their adapters stale.
        if self.use_distribute:
            raise NotImplementedError(
                "--lora-adapter-mode does not yet support distributed (non-colocated) rollout "
                "engines; the adapter is pushed only to colocated IPC engines. Use colocate mode "
                "or --lora-merge-mode for distributed rollout."
            )

        if not self._lora_sync.base_sync_done:
            self._sync_base_and_lora()
            self._lora_sync.base_sync_done = True
        else:
            self._sync_lora_delta_only()

    def _sync_base_and_lora(self) -> None:
        """First-time sync: push base-only weights, then register the LoRA
        adapter."""
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        # Weights include base + LoRA adapter params (adapters are ordinary parameters).
        megatron_local_weights = self.weights_getter()

        # Snapshot the current adapter state for delta computation in subsequent steps. Keys
        # use the CPU-backup format (vp_stages.N.xxx) consistent with what _sync_lora_delta_only
        # reads each step. Select adapter params via the shared predicate so this agrees with the
        # checkpoint path (both Megatron-Bridge adapter.linear_* and PEFT lora_* names).
        self._lora_sync.prev_state = {
            k: v.clone() for k, v in megatron_local_weights.items() if is_lora_adapter_param(k)
        }
        # A non-empty backup is what makes every subsequent delta sync work; an empty one means
        # the adapter is never refreshed on the engine (silent stale policy) — fail loud.
        if not self._lora_sync.prev_state:
            raise RuntimeError(
                "Adapter mode: no LoRA adapter parameters found in weights_getter() output; the live "
                "adapter would never be pushed to the rollout engine. Check adapter naming / "
                "weights_getter output."
            )

        logger.info("Adapter mode: first sync — pushing base weights, then registering LoRA adapter")

        # 1) Send BASE weights only. In adapter mode the HF iterator pulls adapter params OUT of
        #    the conversion buckets (collect_adapters=True) without merging, so only base weights
        #    flow through SGLang's base-model load_weights (which has no notion of lora_A/lora_B).
        prev_refs: list[ObjectRef] = []
        prev_long_lived_tensors = None
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            if prev_refs:
                ray.get(prev_refs)
            del prev_long_lived_tensors
            prev_refs = refs
            prev_long_lived_tensors = long_lived_tensors
        if prev_refs:
            ray.get(prev_refs)
        del prev_long_lived_tensors

        dist.barrier(group=get_gloo_group())

        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        # 2) Register the LoRA adapter on the engines via SGLang's LoRA API. first_sync gates the
        #    unload-before-load; kept as `not self._lora_sync.adapter_loaded` (rather than hardcoded
        #    True) so a re-entry after engine restart still unloads stale state first.
        push_error: Exception | None = None
        try:
            self._push_lora_adapter(megatron_local_weights, first_sync=not self._lora_sync.adapter_loaded)
        except Exception as e:  # noqa: BLE001 - re-raised by the resume helper, on every rank
            logger.exception("LoRA adapter push failed; resuming generation before aborting")
            push_error = e
        else:
            self._lora_sync.adapter_loaded = True

        dist.barrier(group=get_gloo_group())
        self._resume_generation_after_adapter_push(push_error)

    def _sync_lora_delta_only(self) -> None:
        """Subsequent syncs: refresh ONLY the LoRA adapter on the rollout
        engines.

        Base weights stay put on the engines (that is the point of adapter
        mode); the adapter is re-registered via SGLang's disk-based
        load_lora_adapter under a fixed name. The sync is skipped entirely when
        no adapter parameter changed beyond the delta threshold.
        """
        self.weight_version += 1

        all_params = self.weights_getter()

        # The skip decision MUST be collective: each rank owns different adapter params, so a
        # per-rank early return would desync the world-collective gather inside _push_lora_adapter
        # and hang. Skip only when NO rank saw a change beyond threshold.
        unchanged, new_state = self._lora_sync.should_skip(all_params)
        if unchanged:
            logger.debug("LoRA adapter weights unchanged (below threshold) on all ranks, skipping sync")
            self._lora_sync.prev_state = new_state
            return

        logger.info("Adapter mode: refreshing LoRA adapter")

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        push_error: Exception | None = None
        try:
            self._push_lora_adapter(all_params, first_sync=not self._lora_sync.adapter_loaded)
        except Exception as e:  # noqa: BLE001 - re-raised by the resume helper, on every rank
            logger.exception("LoRA adapter push failed; resuming generation before aborting")
            push_error = e
        else:
            self._lora_sync.adapter_loaded = True

        dist.barrier(group=get_gloo_group())
        self._resume_generation_after_adapter_push(push_error)
        self._lora_sync.prev_state = new_state

    def _resume_generation_after_adapter_push(self, push_error: Exception | None) -> None:
        """Resume rollout generation, then re-raise an adapter-push failure on
        every rank.

        Only the gather-src rank can fail inside ``_push_lora_adapter``, and generation is
        paused at that point. Raising there directly would leave the engines paused forever
        and every other rank blocked in the barrier below waiting for a rank that already
        unwound, so the verdict is shared first (MAX all-reduce), generation is resumed, and
        only then does everyone raise together.

        The failure is never swallowed: in adapter mode the engine's base weights are frozen,
        so a dropped adapter means the rollout policy silently stops tracking the trained one.
        """
        failed = torch.tensor([1 if push_error is not None else 0], dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
        if failed.item():
            raise RuntimeError(
                "LoRA adapter push to the rollout engines failed; aborting the weight update "
                "instead of generating with a stale or missing adapter. Generation has been "
                "resumed so the engines are not left paused."
            ) from push_error

    def _push_lora_adapter(self, all_params: Mapping[str, torch.Tensor], *, first_sync: bool) -> None:
        """Export the trained LoRA adapter and (re)register it in-memory on
        every colocated rollout engine.

        The adapter is pushed to SGLang via ``load_lora_adapter_from_tensors``
        (SGLang >= 0.5.12) — no disk IO. It is exported on every rank, all-gathered across PP so
        each engine's gather-src rank holds the full adapter, then pushed once per engine.

        Collective contract: every world rank runs the export and the PP all-gather in lockstep
        (placeholder ranks contribute an empty shard); only each engine's gather-src rank issues
        the load.
        """
        t0 = monotonic()
        local_adapter = self._lora_sync.export_local_adapter(all_params)
        t_export = monotonic() - t0

        # Every engine's gather-src rank must hold the FULL adapter to push it, so all-gather across
        # PP (the only axis the adapter varies on — TP is bridge-gathered, DP is replicated). Cheap:
        # the adapter is small and this is a no-op for PP=1.
        t1 = monotonic()
        full_adapter = self._lora_sync.gather_full_adapter(local_adapter, all_gather=True)
        t_gather = monotonic() - t1

        # Only each engine's gather-src rank issues the load (one push per engine).
        if self._ipc_engine is None or dist.get_rank() != self._ipc_gather_src:
            return

        config_dict = self._lora_sync.config_dict()

        # SGLang broadcasts this single blob to every TP worker (each slices its own shard), so the
        # tensors must be host-shared, not CUDA-IPC handles. The default file_descriptor sharing
        # strategy sends an fd that cannot cross the Ray -> HTTP hops to the server process; switch
        # to file_system (self-describing /dev/shm filenames in the pickle) around the serialize.
        from torch.multiprocessing import get_sharing_strategy, set_sharing_strategy

        tensors = {name: t.contiguous() for name, t in full_adapter.items()}
        prev_strategy = get_sharing_strategy()
        set_sharing_strategy("file_system")
        try:
            serialized = MultiprocessingSerializer.serialize(tensors, output_str=True)
            t3 = monotonic()
            if not first_sync:
                # Tolerate "already gone". SGLang's unload raises when the name is not in its
                # registry, and a previous push that died between this unload and the load
                # below leaves exactly that state (the registry entry is dropped, nothing
                # replaces it). Failing here would make every retry die before reloading, so
                # one transient error would wedge the engine permanently. A genuine unload
                # failure still surfaces: the load right after reports "already loaded".
                try:
                    ray.get(self._ipc_engine.unload_lora_adapter.remote(LORA_ADAPTER_NAME))
                except Exception as e:  # noqa: BLE001 - see above; the load is the real gate
                    logger.warning(
                        "Unloading the previous LoRA adapter '%s' failed (%s); continuing to the "
                        "load, which will report 'already loaded' if a stale copy is still there.",
                        LORA_ADAPTER_NAME,
                        e,
                    )
            # Keep `tensors` alive across the synchronous load: file_system storages live only while
            # the producer holds them, and the server maps them during this call.
            ray.get(
                self._ipc_engine.load_lora_adapter_from_tensors.remote(
                    lora_name=LORA_ADAPTER_NAME,
                    serialized_tensors=serialized,
                    config_dict=config_dict,
                    load_format=None,
                    pinned=False,
                )
            )
            logger.info(
                "[lora-adapter] tensor push: export=%.2fs gather=%.2fs load=%.2fs (%d tensors, rank=%d)",
                t_export,
                t_gather,
                monotonic() - t3,
                len(tensors),
                dist.get_rank(),
            )
        finally:
            set_sharing_strategy(prev_strategy)

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        long_lived_tensors = None
        if self._ipc_engine is not None:
            refs_colocated, long_lived_tensors = _send_to_colocated_engine(
                hf_named_tensors,
                ipc_engine=self._ipc_engine,
                ipc_gather_src=self._ipc_gather_src,
                ipc_gather_group=self._ipc_gather_group,
                weight_version=self.weight_version,
            )
            all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    long_live_tensors = []

    # Colocated IPC requires accelerator tensors (uses device IPC handles via
    # shared memory). The bridge usually returns device tensors, but for K2.x
    # multi-modal wrappers some text-backbone tensors leak through on cpu —
    # coerce here so FlattenedTensorBucket's torch.cat doesn't see mixed
    # devices. Synchronous copy: this runs on the weight-update path (not the
    # rollout/train hot path) and FlattenedTensorBucket may flatten on a
    # different stream — correctness over a few µs.
    cur_device = make_current_torch_device()
    hf_named_tensors = [
        (name, tensor.to(cur_device) if tensor.device != cur_device else tensor) for name, tensor in hf_named_tensors
    ]

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        # An empty local chunk contributes no bucket; FlattenedTensorBucket
        # rejects an empty named_tensors list, so skip building one.
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors} if hf_named_tensors else {}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = flattened_tensor_bucket.get_metadata()
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": metadata,
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if dist.get_rank() == ipc_gather_src:
        # Ranks can contribute uneven bucket counts: under PP/EP/MoE layouts a TP
        # rank may legitimately hold no HF tensors for a chunk. Pad short ranks
        # with an empty bucket so every engine call gets one entry per rank. When
        # all ranks are empty num_buckets is 0 and no update is issued.
        num_buckets = max(len(tensors) for tensors in serialized_named_tensors)
        empty_serialized_tensor = None
        for i in range(num_buckets):
            serialized_tensors_for_bucket = []
            for tensors in serialized_named_tensors:
                if i < len(tensors):
                    serialized_tensors_for_bucket.append(tensors[i])
                    continue
                if empty_serialized_tensor is None:
                    empty_tensor_data = _empty_flattened_tensor_data(cur_device)
                    long_live_tensors.append(empty_tensor_data)
                    empty_serialized_tensor = MultiprocessingSerializer.serialize(empty_tensor_data, output_str=True)
                serialized_tensors_for_bucket.append(empty_serialized_tensor)
            kwargs = {
                "serialized_named_tensors": serialized_tensors_for_bucket,
                "load_format": "flattened_bucket",
                "weight_version": str(weight_version),
            }
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors


def _empty_flattened_tensor_data(device):
    return {
        "flattened_tensor": torch.empty(0, dtype=torch.uint8, device=device),
        "metadata": [],
    }
