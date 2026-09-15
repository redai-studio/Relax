# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import logging
import os
import random
import socket
import time
from argparse import Namespace
from functools import partial
from typing import Any, List

import ray
import requests
import torch
import torch.distributed as dist
import transfer_queue as tq
from megatron.core import mpu


try:
    # NPU patch
    from mindspeed.megatron_adaptor import repatch
except ImportError:
    repatch = None

from tensordict import TensorDict
from transformers import AutoConfig, AutoTokenizer

from relax.distributed.checkpoint_service.client.engine import create_client
from relax.distributed.ray.train_actor import TrainRayActor
from relax.engine.sft.eval.runner import run_sft_eval
from relax.engine.sft.predict.runner import run_sft_predict
from relax.engine.sft.runtime import (
    is_sft_mode,
    sft_partition_id,
    sft_task_name,
    should_run_sft_eval,
    should_run_sft_predict,
    should_skip_mtp_only_weight_management,
)
from relax.utils import device as device_utils
from relax.utils import tracking_utils
from relax.utils.async_utils import run
from relax.utils.data.stream_dataloader import (
    MicroBatchListIterator,
    StreamingTQIterator,
    create_stream_dataloader,
    get_data_from_transfer_queue,
    post_process_rollout_data,
)
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.env import Envs
from relax.utils.megatron_peft_utils import (
    is_lora_adapter_mode,
    is_lora_adapter_param,
    is_lora_enabled,
    is_lora_merge_mode,
    summarize_lora_modules,
)
from relax.utils.memory_utils import clear_memory, print_memory
from relax.utils.metrics.metric_utils import compute_rollout_step
from relax.utils.opd.opd_utils import (
    append_managed_opd_teacher_offload_handle,
    append_managed_opd_teacher_onload_handle,
    consume_opd_train_data,
    has_managed_opd_teacher_manager,
)
from relax.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from relax.utils.replay import capture_hooks
from relax.utils.rotate_ckpt import rotate_ckpt
from relax.utils.s3_model_loader import prepare_model_maybe_update_args
from relax.utils.timer import Timer, inverse_timer, timer, with_defer
from relax.utils.tracking_utils import init_tracking
from relax.utils.training import train_dump_utils
from relax.utils.training.data_fields import build_data_fields
from relax.utils.training.routing_replay import RoutingReplay
from relax.utils.types import RolloutBatch
from relax.utils.utils import (
    _extract_audio_seqlens,
    _extract_images_seqlens,
    get_debug_data,
    get_serve_url,
    merge_dict_list,
    process_args,
)

from ...utils.profile_utils import TrainProfiler
from ...utils.training.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .collective_utils import _agree_drained
from .cp_utils import all_gather_with_cp, maybe_padded_total_lengths, slice_with_cp
from .data import (
    ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY,
    ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY,
    DataIterator,
    build_rollout_minibatch_plan,
    concat_rollout_batches,
    get_data_iterator,
    log_perf_data,
    log_perf_data_fwd,
    log_rollout_data,
)
from .initialize import init, is_megatron_main_rank
from .loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from .model import forward_only, initialize_model_and_optimizer, save, train
from .weight_update.common import named_params_and_buffers
from .weight_update.train_offload import MegatronTrainStateOffloader
from .weight_update.update_weight_from_distributed import UpdateWeightFromDistributed
from .weight_update.update_weight_from_tensor import UpdateWeightFromTensor


logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


ROLLOUT_MINI_BATCH_METAS_KEY = "rollout_mini_batch_metas"


def _split_by_rollout_mini_counts(values: Any, counts: list[int]) -> list[Any]:
    if not counts or any((not isinstance(c, int)) or c <= 0 for c in counts):
        raise ValueError(f"rollout mini counts must be positive integers, got {counts}")
    total = sum(counts)
    if len(values) != total:
        raise ValueError(f"sum(rollout mini counts) must equal values length, got sum={total}, values={len(values)}")
    chunks = []
    start = 0
    for c in counts:
        chunks.append(values[start : start + c])
        start += c
    return chunks


def _rollout_mini_row_counts(
    sample_indices: list[int],
    identities_per_mini: int,
    num_rollout_minis: int,
) -> list[int]:
    identity_positions = {
        sample_index: position for position, sample_index in enumerate(dict.fromkeys(sample_indices))
    }
    expected_identity_count = identities_per_mini * num_rollout_minis
    if len(identity_positions) != expected_identity_count:
        raise RuntimeError(
            "debug rollout logical sample count does not match rollout mini plan: "
            f"expected={expected_identity_count}, got={len(identity_positions)}."
        )
    counts = [0 for _ in range(num_rollout_minis)]
    for sample_index in sample_indices:
        counts[identity_positions[sample_index] // identities_per_mini] += 1
    return counts


def _slice_rollout_batch(rollout_data: dict, start: int, end: int) -> dict:
    num_samples = len(rollout_data.get("total_lengths", []))
    if start < 0 or end < start or end > num_samples:
        raise ValueError(f"Invalid slice range: start={start}, end={end}, size={num_samples}")

    def _slice(value):
        if isinstance(value, (str, bytes)):
            return value
        if isinstance(value, (list, tuple)) and len(value) == num_samples:
            return value[start:end]
        ndim = getattr(value, "ndim", None)
        size = getattr(value, "size", None)
        if ndim and ndim > 0 and callable(size):
            try:
                if int(size(0)) == num_samples:
                    return value[start:end]
            except (TypeError, ValueError, IndexError):
                pass
        return value

    return {k: _slice(v) for k, v in rollout_data.items()}


class MegatronTrainRayActor(TrainRayActor):
    @property
    def _per_step_rollout(self) -> bool:
        """RL: rollout consumes weights every train step. SFT: only on
        periodic predict steps; Megatron stays awake between."""
        return not is_sft_mode(self.args)

    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        with timer("init_actor"):
            return self._init(args, role, with_ref, with_opd_teacher)

    @with_defer(lambda: Timer().start("train_wait"))
    def _init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        monkey_patch_torch_dist(args)
        from relax.utils.checkpoint_write_patch import patch_checkpoint_write

        patch_checkpoint_write()
        if role == "reference" or role == "actor_fwd":
            process_args(args, role)
        super().init(args, role, with_ref, with_opd_teacher)

        # Materialize an S3 checkpoint on this node before Megatron init(args)
        # reads model metadata or weights.
        # Leaf actor with private args, safe to remap in place.
        prepare_model_maybe_update_args(args)

        self.genrm_manager = None

        init(args)
        if repatch is not None:
            repatch(args)
        tq.init(args.tq_config)
        self.data_system_client = tq.get_client()
        if is_megatron_main_rank():
            init_tracking(args, primary=False)

        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        # Single source of truth for VL-model routing across data prep, forward,
        # and CP helpers. Detected by the presence of processor/preprocessor
        # config in the HF checkpoint dir — only multimodal models ship one.
        # This way VL models with text-only batches (no --multimodal-keys) still
        # take the bridge VL+CP+thd path.
        args.is_vl_model = any(
            os.path.exists(os.path.join(args.hf_checkpoint, name))
            for name in ("processor_config.json", "preprocessor_config.json")
        )

        from relax.utils.training.flops_counter import FlopsCounter

        self.flops_counter = FlopsCounter(self.hf_config)

        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
        }
        dist.barrier(group=get_gloo_group())

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters

        self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
            args, role
        )

        if is_lora_enabled(args) and dist.get_rank() == 0:
            self._log_lora_checkpoint_state()

        # MoE LoRA expert fold: forward-only reference models (actor_fwd / reference) receive each
        # grouped-expert weight ALREADY folded (base + delta) into its base slot via DCS (see
        # DeviceDirectBackend._update_expert_weight_from_distributed). This holds in BOTH rollout
        # paths: merge mode always folds, and adapter mode folds on the actor_fwd pass (the rollout
        # gets the pure adapter via SGLang instead). Their own grouped-expert LoRA adapters are never
        # re-synced, so they must contribute nothing — otherwise the effective weight would be
        # base + 2*delta. Zero them here, after the checkpoint load (a resume otherwise restores
        # trained adapters with linear_out != 0). Non-expert adapters are kept: those ARE re-synced
        # raw each step and carry the delta for the non-expert path. Colocate/hybrid keep the
        # reference in-process (role stays "actor"), so this never fires there.
        if (
            role in ("actor_fwd", "reference")
            and is_lora_enabled(args)
            and (is_lora_merge_mode(args) or is_lora_adapter_mode(args))
            and args.num_experts
        ):
            self._zero_expert_lora_adapters()

        # Train-state offload for colocate sleep/wake. Picks torch_memory_saver
        # (VMM pause) or manual selective CPU offload based on TMS availability;
        # both implementations live in the offloader. No-op when offload_train is off.
        self._train_state_offloader = MegatronTrainStateOffloader(self.model, self.optimizer, args)

        start_rollout_id = loaded_rollout_id + 1

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return start_rollout_id

        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size
        # Hybrid mode uses the TensorBackuper path: actor handles ref/actor_fwd
        # internally via _switch_model and pushes weights to rollout via
        # UpdateWeightFromTensor instead of DCS.
        skip_weight_management = should_skip_mtp_only_weight_management(
            self.args,
            with_ref=with_ref,
            with_opd_teacher=with_opd_teacher,
        )
        use_tensor_backuper = not skip_weight_management and (not self.args.fully_async or self.args.hybrid)
        if use_tensor_backuper:
            self.weights_backuper = TensorBackuper.create(
                source_getter=lambda: named_params_and_buffers(
                    self.args,
                    self.model,
                    convert_to_global_name=args.megatron_to_hf_mode == "raw",
                    translate_gpu_to_cpu=not self.args.enable_weights_backuper,
                ),
                single_tag=None if args.enable_weights_backuper else "actor",
            )
            self._active_model_tag: str | None = "actor"
            self.weights_backuper.backup("actor")

            if with_ref:
                self.load_other_checkpoint("ref", args.ref_load)

            # Load teacher model for Megatron-based on-policy distillation
            if with_opd_teacher:
                self.load_other_checkpoint("teacher", args.opd_teacher_load)

            if self.args.keep_old_actor:
                # Load old_actor checkpoint
                self.load_other_checkpoint("old_actor", args.load)
                # Create rollout_actor as a copy of current actor
                if args.update_weights_interval == 1:
                    self.weights_backuper.backup("rollout_actor")

            update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
            # Push-side repack is decided by the HF config: an FP8 release auto-routes
            # through quantize_params_fp8, a compressed-tensors release through
            # quantize_params_compressed_tensors, an unquantized BF16 dir is passed
            # through verbatim. The OPEN_TRAINING_INT4_FAKE_QAT_FLAG env var ONLY
            # controls the training-side forward STE in the megatron patch — it is
            # independent of push routing (matches slime/backends/megatron_utils/actor.py).
            # K2.6 INT4 release ships an ignore list that omits vision_tower /
            # mm_projector — without augment_compressed_tensors_ignore the bridge
            # would try to INT4-pack those BF16 tensors and SGLang would reject
            # them with "weight_packed not found in params_dict".
            from relax.utils.quant_cast import augment_compressed_tensors_ignore

            push_quant_config = augment_compressed_tensors_ignore(
                getattr(self.hf_config, "quantization_config", None),
                args.hf_checkpoint,
            )
            self.weight_updater = update_weight_cls(
                self.args,
                self.model,
                weights_getter=lambda: self.weights_backuper.get("actor"),
                model_name=type(self.hf_config).__name__.lower()
                if self.args.model_name is None
                else self.args.model_name,
                quantization_config=push_quant_config,
            )
        elif not skip_weight_management:
            is_pp_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
            )
            if is_pp_src_rank:
                master_address = ray._private.services.get_node_ip_address()
                with socket.socket() as sock:
                    sock.bind(("", 0))
                    master_port = sock.getsockname()[1]
            else:
                master_address = None
                master_port = None
            metadata = {
                "tp_size": mpu.get_tensor_model_parallel_world_size(),
                "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
                "pp_size": mpu.get_pipeline_model_parallel_world_size(),
                "ep_size": mpu.get_expert_model_parallel_world_size(),
                "cp_size": mpu.get_context_parallel_world_size(),
                "pp_rank": mpu.get_pipeline_model_parallel_rank(),
                "is_pp_src_rank": is_pp_src_rank,
                "master_address": master_address,
                "master_port": master_port,
            }
            from relax.utils.quant_cast import augment_compressed_tensors_ignore

            push_quant_config = augment_compressed_tensors_ignore(
                getattr(self.hf_config, "quantization_config", None),
                args.hf_checkpoint,
            )
            self.checkpoint_engine_client = run(
                create_client(
                    args=self.args,
                    coordinator_url=self.args.coordinator_url,
                    role=role,
                    rank=dist.get_rank(),
                    model=self.model,
                    model_name=type(self.hf_config).__name__.lower()
                    if self.args.model_name is None
                    else self.args.model_name,
                    quantization_config=push_quant_config,
                    backend_type=self.args.checkpoint_engine_backend,
                    metadata=metadata,
                    lock=self.lock,
                )
            )
        else:
            logger.info("MTP-only SFT: skipping weight snapshots, rollout updater, and DCS client")
        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
            self._switch_model("actor")
            if self._per_step_rollout:
                self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from relax.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()
        self.data_iterator = None

        if role == "actor":
            capture_hooks.maybe_enable_for_actor()

        if dist.get_rank() == 0:
            logger.info(
                "[per_rank_fetch] enabled=%s (effective when rollout_routed_experts not in data_fields)",
                self.args.per_rank_fetch,
            )

        return start_rollout_id

    @torch.no_grad()
    def _log_lora_checkpoint_state(self) -> None:
        """Log (rank 0) the LoRA adapter state right after checkpoint load.

        Answers "which LoRA was loaded": counts adapter params by architectural
        role, and reports RESUMED (adapters non-zero -> loaded from a trained
        checkpoint) vs FRESH (all-zero -> newly initialized). Detection keys on
        ``linear_out``/``lora_B`` which init to zero; the ``.any()`` GPU sync
        runs once at init on rank 0 and short-circuits once a non-zero is
        found.
        """
        adapter_names: list[str] = []
        resumed = False
        for chunk in self.model:
            for name, param in chunk.named_parameters():
                if not is_lora_adapter_param(name):
                    continue
                adapter_names.append(name)
                if not resumed and ("linear_out" in name or ".lora_B." in name):
                    resumed = bool(param.detach().abs().any().item())
        logger.info(
            "LoRA checkpoint state (role=%s): %d adapter params, wrapped modules by role (rank0-local)=%s | %s",
            self.role,
            len(adapter_names),
            summarize_lora_modules(adapter_names),
            "RESUMED from checkpoint (adapters non-zero)" if resumed else "FRESH init (adapters zero)",
        )

    @torch.no_grad()
    def _zero_expert_lora_adapters(self) -> None:
        """Zero every grouped-expert LoRA adapter param on this (reference)
        model.

        Used for MoE on the actor_fwd / reference roles in BOTH rollout paths
        (merge mode always; adapter mode on the actor_fwd fold pass): their
        expert weights arrive pre-folded (base + delta) in the base slot and
        their expert adapters are never re-synced, so the adapters must stay at
        zero to avoid double-counting the delta. Covers both
        ``.adapter.linear_in``/``.adapter.linear_out`` (Megatron-Bridge) and
        the shared/per-expert packing (2D or 3D) transparently — a zero tensor
        of either shape yields a zero delta. Non-expert adapters are left
        untouched.
        """
        zeroed = 0
        for chunk in self.model:
            for name, param in chunk.named_parameters():
                if ".experts." in name and is_lora_adapter_param(name):
                    param.data.zero_()
                    zeroed += 1
        logger.info("[LoRA] zeroed %d grouped-expert adapter params on role=%s", zeroed, self.role)

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        # In disaggregate PPO (use_critic + not colocate), the actor's NCCL
        # weight-sync groups to rollout engines do not survive a sleep, so we
        # must explicitly tear them down before destroy_process_groups() and
        # reconnect on wake_up()/update_weights().
        if (
            self.role == "actor"
            and self.args.use_critic
            and not self.args.colocate
            and hasattr(self, "weight_updater")
            and hasattr(self.weight_updater, "disconnect_rollout_engines")
        ):
            self.weight_updater.disconnect_rollout_engines()
        destroy_process_groups()

        self._train_state_offloader.offload()

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        self._train_state_offloader.reload()

        clear_memory()
        reload_process_groups(timeout_minutes=self.args.distributed_timeout_minutes)
        print_memory("after wake_up model")

    def _switch_model(self, target_tag: str) -> None:
        # Backend-specific bookkeeping is handled in device utils so this
        # framework path stays hardware-agnostic.
        device_utils.maybe_backend_process_on_model_switch()
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from relax.utils.training.routing_replay import RoutingReplay

        for iterator in data_iterator:
            iterator.reset()

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        def pad_func(experts, pad):
            _, num_layers, topk = experts.shape
            pad = (
                torch.arange(
                    pad * num_layers * topk,
                    device=experts.device,
                    dtype=experts.dtype,
                ).reshape((pad, num_layers, topk))
                % self.args.num_experts
            )
            return torch.cat([experts, pad], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            rollout_routed_experts = batch["rollout_routed_experts"]
            tokens = batch["tokens"]
            # Reshape from flattened 2D (seq_i, num_layers*topk) back to 3D (seq_i, num_layers, topk)
            # because dict_to_tensordict flattened the last two dims for NestedTensor jagged layout.
            rollout_routed_experts = [
                r.reshape(r.shape[0], self.args.num_layers, self.args.moe_router_topk) for r in rollout_routed_experts
            ]
            assert len(rollout_routed_experts) == len(tokens)
            for a, b in zip(rollout_routed_experts, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # Pad each sample so its routed-experts stream lines up with how the forward's MoE
            # router sees tokens, then CP-slice identically.
            #
            # is_vl_model (a VL model such as Kimi-K2.6 — detected from processor/preprocessor
            # config, True even on a text-only batch) with CP>1 + thd takes the bridge unsplit path:
            # get_batch/model.py feed the bridge UNSPLIT tokens and it splits along CP internally
            # after padding each sample to tp*cp*2 (see maybe_padded_total_lengths +
            # get_logits_and_tokens_offset_with_cp, which commit 2edaade aligned the loss/metric side
            # to). Routing replay was NOT updated then, so it still used slice_with_cp's 2*cp align +
            # a global tp*dpm pad -> the recorded routing landed on different per-sample chunk
            # boundaries than the bridge, misaligning routed_experts vs tokens at CP>1. Pad each
            # sample to the SAME tp*cp*2 length so slice_with_cp's chunk_size = padded_len // (2*cp)
            # matches the bridge; the per-CP-rank stream is then tp-divisible, so no global pad (the
            # bridge adds none either).
            #
            # Otherwise (non-VL) the forward uses get_batch's per-sample slice_with_cp (2*cp align)
            # + global tp*data_pad_size_multiplier pad, so mirror that here.
            padded_total_lengths = maybe_padded_total_lengths(
                [t.shape[0] for t in tokens], self.args.qkv_format, getattr(self.args, "is_vl_model", False)
            )
            if padded_total_lengths is not None:
                rollout_routed_experts = [
                    pad_func(r, padded_total_lengths[i] - r.shape[0]) for i, r in enumerate(rollout_routed_experts)
                ]
            else:
                # Pad each sample by 1 (the last, non-loss token) so it matches the token length.
                rollout_routed_experts = [pad_func(r, 1) for r in rollout_routed_experts]
            rollout_routed_experts = [slice_with_cp(r, pad_func) for r in rollout_routed_experts]
            rollout_routed_experts = torch.cat(rollout_routed_experts, dim=0)
            if padded_total_lengths is None:
                pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
                pad = (pad_size - rollout_routed_experts.size(0) % pad_size) % pad_size
                if pad != 0:
                    rollout_routed_experts = pad_func(rollout_routed_experts, pad)

            if self.args.sequence_parallel:
                seqlen = rollout_routed_experts.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                rollout_routed_experts = rollout_routed_experts[start:end]

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays), (
                f"{routing_replay_offset} vs {len(RoutingReplay.all_routing_replays)}"
            )

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
        collect_topk: bool = False,
    ) -> dict[str, list[torch.Tensor]]:
        with timer(f"{store_prefix}log_probs"):
            log_prob_func = get_log_probs_and_entropy
            if collect_topk:
                log_prob_func = partial(log_prob_func, with_topk=True, topk_k=self.args.opd_log_prob_top_k)
            return forward_only(
                log_prob_func,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def _run_step_evaluation(self, rollout_id: int, *, end_update_weight: bool = False) -> None:
        is_sft = is_sft_mode(self.args)
        has_rollout = getattr(self, "rollout_manager", None) is not None

        if not is_sft and dist.get_rank() != 0:
            return

        if is_sft:
            should_run_eval = should_run_sft_eval(self.args, rollout_id)
            should_run_predict = has_rollout and should_run_sft_predict(self.args, rollout_id)
            try:
                if should_run_eval:
                    if dist.get_rank() == 0:
                        run(
                            self.data_system_client.async_clear_partition(
                                partition_id=sft_partition_id(self.args, rollout_id)
                            )
                        )
                    dist.barrier(group=get_gloo_group())
                    run_sft_eval(self, rollout_id)

                if should_run_predict:
                    run_sft_predict(self, rollout_id)
            except Exception as e:
                logger.warning(f"SFT eval/predict at rollout_id {rollout_id} failed: {e}")
                raise
            return

        # RL path: trigger rollout-based evaluation if configured.
        # Telemetry marks for RL eval live on the Rollout side
        # (see Rollout._run_eval_with_mark), so we don't emit them here.
        if not has_rollout:
            return
        rollout_serve_url = get_serve_url("rollout")
        try:
            # NOTE: /evaluate runs rollout-side inference and can legitimately
            # take much longer than a short RPC, so it is intentionally left
            # without an HTTP timeout -- a fixed timeout would falsely abort a
            # long eval. This block's failure is non-fatal (logged, training
            # continues).
            response = requests.get(f"{rollout_serve_url}/evaluate", params={"train_step": rollout_id})
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"Error during actor post-train evaluation for rollout_id {rollout_id}: {e}")
        finally:
            # Cleanup MUST run even if /evaluate failed, otherwise rollout + health
            # monitoring stay paused forever. Short call -> keep the timeout;
            # swallow its own errors independently.
            if end_update_weight:
                try:
                    response = requests.get(
                        f"{rollout_serve_url}/end_update_weight", timeout=self.args.rollout_http_timeout
                    )
                    response.raise_for_status()
                except Exception as e:
                    logger.warning(f"Error ending update weight for rollout_id {rollout_id}: {e}")

    def _request_rollout_evaluation(self, rollout_id: int, *, end_update_weight: bool = False) -> None:
        """Backward-compatible name kept for existing internal call sites."""
        self._run_step_evaluation(rollout_id, end_update_weight=end_update_weight)

    def train(self, rollout_id: int) -> None:
        if self.args.offload_rollout and dist.get_rank() == 0:
            pre_train_offload_handles = []
            if self.genrm_manager is not None:
                pre_train_offload_handles.append(self.genrm_manager.offload.remote())
            append_managed_opd_teacher_offload_handle(pre_train_offload_handles, self)
            if pre_train_offload_handles:
                ray.get(pre_train_offload_handles)

        # Gate all ranks behind rank-0's GenRM/OPD-teacher offload: otherwise
        # other ranks wake_up() and reclaim GPU memory while colocated GenRM or
        # teacher still holds its static pool, causing cuMemCreate OOM (mirrors
        # the update_weights barrier).
        if self.args.offload_rollout and (self.genrm_manager is not None or has_managed_opd_teacher_manager(self)):
            dist.barrier(group=get_gloo_group())

        if self.args.offload_train and self._per_step_rollout:
            self.wake_up()

        if self.args.debug_train_only:
            logger.info(f"start to get rollout_id: {rollout_id} data from transfer queue for debug with mcore.")
            dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
            if is_sft_mode(self.args):
                batch_size = self.args.global_batch_size // dp_size
                rollout_mini_local_sample_counts = None
            else:
                plan = build_rollout_minibatch_plan(self.args, dp_size)
                batch_size = plan.mini_local_sample_request * plan.num_rollout_minis
                rollout_mini_local_sample_counts = None
            rollout_data = get_debug_data(self.args, rollout_id, batch_size, dp_rank=mpu.get_data_parallel_rank())
            post_process_rollout_data(self.args, rollout_data)
            if not is_sft_mode(self.args):
                rollout_mini_local_sample_counts = _rollout_mini_row_counts(
                    rollout_data["sample_indices"],
                    plan.mini_local_sample_request,
                    plan.num_rollout_minis,
                )
                rollout_data[ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY] = rollout_mini_local_sample_counts

            if self.role == "critic":
                return self.train_critic(rollout_id, rollout_data)
            else:
                return self.train_actor(rollout_id, rollout_data)
        else:
            logger.info(f"start to get rollout_id: {rollout_id} data from transfer queue for train with mcore.")
            if is_sft_mode(self.args):
                batch_size = self.args.global_batch_size // mpu.get_data_parallel_world_size(
                    with_context_parallel=False
                )
                num_rollout_minis = 1
            else:
                dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
                if self.args.partial_rollout and self.args.use_dynamic_global_batch_size:
                    dynamic_size = ray.get(self.rollout_manager.get_dynamic_global_batch_size.remote())
                    batch_size = dynamic_size // dp_size
                    num_rollout_minis = 1
                else:
                    plan = build_rollout_minibatch_plan(self.args, dp_size)
                    batch_size = plan.mini_local_sample_request
                    num_rollout_minis = plan.num_rollout_minis
            batch_index = 0
            # PPO shares the RL train path across actor and critic backends. If both
            # used the same TQ task_name, the first consumer (critic) would mark the
            # whole partition consumed and the actor would exit its fetch loop with
            # zero mini batches. Use a role-specific task_name so consumption is
            # tracked independently per consumer.
            base_task_name = sft_task_name(self.args, component="backend")
            if not is_sft_mode(self.args) and self.role == "critic":
                task_name = f"{base_task_name}_critic"
            else:
                task_name = base_task_name
            rollout_mini_batches: list[RolloutBatch] = []
            rollout_mini_batch_metas: list = []
            rollout_mini_local_sample_counts: list[int] = []
            empty_poll_sleep_s = Envs.RELAX_EMPTY_POLL_SLEEP_MS / 1000.0
            fetch_iter = 0
            while batch_index < num_rollout_minis and not self.all_consumed(task_name, rollout_id):
                consumer = "critic" if self.role == "critic" else "actor"
                data_fields = build_data_fields(self.args, consumer=consumer)
                with timer("train_get_data"):
                    rollout_data, batch_meta = self._get_data_from_transfer_queue(
                        task_name, rollout_id, data_fields, batch_size, batch_index
                    )
                if rollout_data is None:
                    if fetch_iter % 100 == 0:
                        logger.info(
                            f"[fetch loop] rollout_id={rollout_id} batch_index={batch_index}/"
                            f"{num_rollout_minis} iter={fetch_iter} task={task_name} "
                            f"fields={data_fields} — empty meta, retrying."
                        )
                    fetch_iter += 1
                    if empty_poll_sleep_s > 0:
                        time.sleep(empty_poll_sleep_s)
                    continue
                batch_index += 1
                if is_sft_mode(self.args):
                    if self.role == "critic":
                        return self.train_critic(rollout_id, rollout_data)
                    else:
                        return self.train_actor(rollout_id, rollout_data)
                rollout_mini_batches.append(rollout_data)
                rollout_mini_batch_metas.append(batch_meta)
                rollout_mini_local_sample_counts.append(len(rollout_data["total_lengths"]))

            if not is_sft_mode(self.args):
                if len(rollout_mini_batches) != num_rollout_minis:
                    raise RuntimeError(
                        f"Expected {num_rollout_minis} rollout mini batches for rollout_id={rollout_id}, "
                        f"got {len(rollout_mini_batches)}."
                    )
                rollout_data = concat_rollout_batches(rollout_mini_batches)
                rollout_data[ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY] = rollout_mini_local_sample_counts
                rollout_data[ROLLOUT_MINI_BATCH_METAS_KEY] = rollout_mini_batch_metas
                if self.args.partial_rollout and self.args.use_dynamic_global_batch_size:
                    rollout_data[ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY] = [dynamic_size]
                if self.role == "critic":
                    return self.train_critic(rollout_id, rollout_data)
                else:
                    return self.train_actor(rollout_id, rollout_data)

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )
        # Ship values to TransferQueue before we sleep — CP all-gather needs
        # live NCCL groups and .cpu() needs live GPU memory.
        self._put_critic_values_to_transfer_queue(rollout_data)

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
        )
        tracking_utils.flush_metrics(self.args, compute_rollout_step(self.args, rollout_id))
        # Save critic ckpt while still awake — the outer critic loop used to
        # call save_model after train_critic returned, which meant save_model
        # ran with tms.paused and hit `CUDA error: invalid argument` on the
        # first cudaMalloc (e.g. all_gather_object's byte tensor).
        is_train_done = (rollout_id + 1) == self.args.num_rollout
        if self.args.save is not None and (
            self.args.rotate_ckpt
            or (
                self.args.save_interval is not None
                and ((rollout_id + 1) % self.args.save_interval == 0 or is_train_done)
            )
        ):
            self.save_model(rollout_id, force_sync=is_train_done)
        # Mirror train_actor's post-train sleep so critic frees GPU before the
        # next rollout's SGLang resume tries to alloc KV pages.
        if self._per_step_rollout and self.args.offload_train:
            self.sleep()

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # PPO colocate: ``values`` and ``loss_masks`` reach us via TransferQueue
        # and land on CPU (critic ``.cpu()`` s ``values`` before PUT). Inline
        # GAE + normalize_advantages need GPU tensors — dispatch here so the
        # rest of the pipeline can assume same-device inputs.
        if self.args.advantage_estimator == "ppo":
            cur_device = torch.cuda.current_device()
            for key in ("values", "loss_masks"):
                tensors = rollout_data.get(key)
                if tensors is None:
                    continue
                rollout_data[key] = [
                    t.to(cur_device) if isinstance(t, torch.Tensor) and t.device.type != "cuda" else t for t in tensors
                ]

        # Create data iterator for actor forward + routing replay + train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        # Create a separate iterator with a larger token budget for ref/teacher log-probs
        if self.args.use_dynamic_batch_size and self.args.log_probs_max_tokens_per_gpu != self.args.max_tokens_per_gpu:
            data_iterator_logprobs, num_microbatches_logprobs = get_data_iterator(
                self.args,
                self.model,
                rollout_data,
                max_tokens_per_gpu=self.args.log_probs_max_tokens_per_gpu,
            )
        else:
            data_iterator_logprobs, num_microbatches_logprobs = data_iterator, num_microbatches

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            # All RL algorithms need ref/teacher/actor inline forwards to produce old_log_probs.
            should_compute_old_log_probs = self.args.compute_advantages_and_returns
            # PPO fully_async has a standalone Advantages service that produces
            # advantages/returns via TransferQueue; every other path (including
            # PPO colocate) computes GAE inline from critic's ``values``.
            should_compute_gae_in_actor = self.args.compute_advantages_and_returns and not (
                self.args.advantage_estimator == "ppo" and self.args.fully_async and not self.args.hybrid
            )

            if should_compute_old_log_probs:
                if "ref" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator_logprobs,
                            num_microbatches_logprobs,
                            store_prefix="ref_",
                        )
                    )

                # Forward teacher model to get teacher_log_probs for Megatron-based OPD
                if "teacher" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("teacher")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator_logprobs,
                            num_microbatches_logprobs,
                            store_prefix="teacher_",
                            collect_topk=self.args.use_opd and self.args.opd_log_prob_top_k > 0,
                        )
                    )

                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                            collect_topk=self.args.use_opd and self.args.opd_log_prob_top_k > 0,
                        )
                    )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                # Restore model tag before train (keep_old_actor may have switched to old_actor).
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

            if should_compute_gae_in_actor:
                if self.args.use_critic and "values" not in rollout_data:
                    raise RuntimeError("Actor training with critic requires 'values' in rollout data.")
                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                capture_hooks.begin_rollout_for(self.args, rollout_id)
                try:
                    compute_advantages_and_returns(self.args, rollout_data)
                    capture_hooks.capture_rollout_advantage(rollout_data=rollout_data, args=self.args)
                finally:
                    capture_hooks.end_rollout_for()

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(
            self.args, rollout_id=rollout_id, rollout_data=rollout_data, tokenizer=self.tokenizer
        )

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        if hasattr(self, "weights_backuper"):
            self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and hasattr(self, "weights_backuper")
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        total_lengths = rollout_data["total_lengths"]
        all_total_lengths = [None] * mpu.get_data_parallel_world_size(with_context_parallel=False)
        dist.all_gather_object(
            all_total_lengths, total_lengths, group=mpu.get_data_parallel_group(with_context_parallel=False)
        )
        all_total_lengths = sum(all_total_lengths, [])  # flatten
        Timer().seq_lens = all_total_lengths
        # Count supervised tokens (loss_mask==1) per sample. For SFT this is the
        # assistant-only tokens; for RL it's the response-only mask sum. Avoids
        # the SFT `response_length == total_length` convention (see data.py:296).
        response_token_counts = [
            int(m.sum().item()) if isinstance(m, torch.Tensor) else int(sum(m)) for m in rollout_data["loss_masks"]
        ]
        all_response_token_counts = [None] * mpu.get_data_parallel_world_size(with_context_parallel=False)
        dist.all_gather_object(
            all_response_token_counts,
            response_token_counts,
            group=mpu.get_data_parallel_group(with_context_parallel=False),
        )
        all_response_token_counts = sum(all_response_token_counts, [])  # flatten
        Timer().response_lens = all_response_token_counts
        mm_inputs = rollout_data.get("multimodal_train_inputs")
        if mm_inputs is not None:
            images_seqlens = _extract_images_seqlens(mm_inputs)
            all_images_seqlens = [None] * mpu.get_data_parallel_world_size(with_context_parallel=False)
            dist.all_gather_object(
                all_images_seqlens, images_seqlens, group=mpu.get_data_parallel_group(with_context_parallel=False)
            )
            Timer().images_seqlens = sum(all_images_seqlens, [])
            audio_seqlens = _extract_audio_seqlens(mm_inputs)
            all_audio_seqlens = [None] * mpu.get_data_parallel_world_size(with_context_parallel=False)
            dist.all_gather_object(
                all_audio_seqlens, audio_seqlens, group=mpu.get_data_parallel_group(with_context_parallel=False)
            )
            Timer().audio_seqlens = sum(all_audio_seqlens, [])
        log_perf_data(rollout_id, self.args, flops_counter=self.flops_counter)

        is_train_done = (rollout_id + 1) == self.args.num_rollout
        if self.args.save is not None and (
            self.args.rotate_ckpt
            or self.args.save_interval is not None
            and ((rollout_id + 1) % self.args.save_interval == 0 or is_train_done)
        ):
            self.save_model(rollout_id, force_sync=is_train_done)
        has_rollout = getattr(self, "rollout_manager", None) is not None
        if self._per_step_rollout:
            if self.args.offload_train:
                self.sleep()
            if has_rollout:
                self.update_weights()
        tracking_utils.flush_metrics(self.args, compute_rollout_step(self.args, rollout_id))
        # RL-only generative eval (uses SGLang via rollout_manager.eval). SFT
        # uses local eval/predict runner below.
        dist.barrier(group=get_gloo_group())
        self._run_step_evaluation(rollout_id)

        # On the final training step the rollout component has already exited
        # its main loop, so nothing else awaits the eval handler. Block here
        # until eval finishes; otherwise the controller's atexit shutdown
        # races with eval and tears down the SGLang engines mid-flight.
        if is_train_done:
            self._wait_for_previous_eval()

    def compute_ref_log_prob(self, rollout_id: int) -> None:
        if self.args.use_routing_replay:
            os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"

        logger.info(f"start to get rollout_id: {rollout_id} data from transfer queue for compute_ref_log_prob.")
        data_fields = ["tokens", "total_lengths", "response_lengths", "loss_masks", "rollout_log_probs"]
        if self.args.multimodal_keys is not None:
            data_fields.append("multimodal_train_inputs")

        if self._use_streaming_fwd():
            self._streaming_fwd_putback(rollout_id, "ref_log_probs", "ref_", data_fields)
        else:
            batch_size = (
                self.args.global_batch_size
                // mpu.get_data_parallel_world_size(with_context_parallel=False)
                // self.args.num_iters_per_train_update
            )
            batch_index = 0
            while not self.all_consumed("ref_log_probs", rollout_id):
                data, batch_meta = self._get_data_from_transfer_queue(
                    "ref_log_probs", rollout_id, data_fields, batch_size, batch_index
                )
                if data is None:
                    continue
                batch_index += 1
                logger.info(
                    f"Successfully got rollout_id: {rollout_id} data from transfer queue for compute_ref_log_prob"
                )
                data_iterator, num_microbatches = get_data_iterator(
                    self.args, self.model, data, max_tokens_per_gpu=self.args.log_probs_max_tokens_per_gpu
                )
                output_dict = self.compute_log_prob(
                    data_iterator,
                    num_microbatches,
                    store_prefix="ref_",
                )
                self._put_data_to_transfer_queue(output_dict, batch_meta, data)
        self.prof.step(rollout_id=rollout_id)

        self.recv_weight_fully_async(rollout_id)
        log_perf_data_fwd(self.args, rollout_id)

    def compute_actor_log_prob(self, rollout_id: int) -> None:
        if self.args.use_routing_replay:
            if self.args.use_rollout_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
            else:
                os.environ["ROUTING_REPLAY_STAGE"] = "record"
        logger.info(f"start to get rollout_id: {rollout_id} data from transfer queue for compute_actor_log_prob.")
        data_fields = ["tokens", "total_lengths", "response_lengths", "loss_masks", "rollout_log_probs"]
        if self.args.multimodal_keys is not None:
            data_fields.append("multimodal_train_inputs")

        if self._use_streaming_fwd():
            self._streaming_fwd_putback(
                rollout_id,
                "actor_log_probs",
                "",
                data_fields,
                clear_routing_replay_forward=self.args.use_rollout_routing_replay,
            )
        else:
            batch_size = (
                self.args.global_batch_size
                // mpu.get_data_parallel_world_size(with_context_parallel=False)
                // self.args.num_iters_per_train_update
            )
            batch_index = 0
            while not self.all_consumed("actor_log_probs", rollout_id):
                data, batch_meta = self._get_data_from_transfer_queue(
                    "actor_log_probs", rollout_id, data_fields, batch_size, batch_index
                )
                if data is None:
                    continue
                batch_index += 1
                logger.info(
                    f"Successfully got rollout_id: {rollout_id} data from transfer queue for compute_actor_log_prob"
                )
                data_iterator, num_microbatches = get_data_iterator(
                    self.args, self.model, data, max_tokens_per_gpu=self.args.log_probs_max_tokens_per_gpu
                )

                output_dict = self.compute_log_prob(
                    data_iterator,
                    num_microbatches,
                    store_prefix="",
                )
                self._put_data_to_transfer_queue(output_dict, batch_meta, data)
                if self.args.use_rollout_routing_replay:
                    RoutingReplay.clear_all_forward()
        self.prof.step(rollout_id=rollout_id)

        self.recv_weight_fully_async(rollout_id)
        log_perf_data_fwd(self.args, rollout_id)

    def _hybrid_forward_subbatch(self, sub_batch: RolloutBatch) -> None:
        """Run the ref/teacher/actor forward passes for a single hybrid sub-
        batch in place.

        Shared by the streaming and debug_train_only paths so both compute
        identical log-probs before advantages are merged.
        """
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, sub_batch)
        # Separate iterator with larger token budget for ref/teacher log-probs (fallthrough mode).
        if self.args.use_dynamic_batch_size and self.args.log_probs_max_tokens_per_gpu != self.args.max_tokens_per_gpu:
            data_iterator_logprobs, num_microbatches_logprobs = get_data_iterator(
                self.args,
                self.model,
                sub_batch,
                max_tokens_per_gpu=self.args.log_probs_max_tokens_per_gpu,
            )
        else:
            data_iterator_logprobs, num_microbatches_logprobs = data_iterator, num_microbatches

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, sub_batch)

        if self.args.compute_advantages_and_returns:
            # Ref forward
            if "ref" in self.weights_backuper.backup_tags:
                if self.args.use_routing_replay:
                    os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                self._switch_model("ref")
                sub_batch.update(
                    self.compute_log_prob(data_iterator_logprobs, num_microbatches_logprobs, store_prefix="ref_")
                )

            # Teacher forward for Megatron-based OPD
            if "teacher" in self.weights_backuper.backup_tags:
                if self.args.use_routing_replay:
                    os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                self._switch_model("teacher")
                sub_batch.update(
                    self.compute_log_prob(data_iterator_logprobs, num_microbatches_logprobs, store_prefix="teacher_")
                )

            # Actor forward
            self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
            if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                if self.args.use_routing_replay:
                    if self.args.use_rollout_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                    else:
                        os.environ["ROUTING_REPLAY_STAGE"] = "record"
                sub_batch.update(self.compute_log_prob(data_iterator, num_microbatches, store_prefix=""))
                if self.args.use_rollout_routing_replay:
                    RoutingReplay.clear_all_forward()

    @staticmethod
    def _split_rollout_batch(rollout_data: RolloutBatch, sample_counts: list[int]) -> List[RolloutBatch]:
        """Split a merged rollout batch along explicit sample boundaries.

        Keys whose value is not a per-sample list are copied into every chunk
        unchanged. Used by the debug_train_only path to feed the collected-
        sub-batch forward loop (one global batch per chunk).
        """
        num_samples = len(rollout_data["tokens"])
        chunks: List[RolloutBatch] = []
        start = 0
        for sample_count in sample_counts:
            end = start + sample_count
            chunk: RolloutBatch = {}
            for key, value in rollout_data.items():
                if isinstance(value, list) and len(value) == num_samples:
                    chunk[key] = value[start:end]
                else:
                    chunk[key] = value
            chunks.append(chunk)
            start = end
        return chunks

    def _use_streaming_fwd(self) -> bool:
        """Whether ref / actor_fwd forward should stream via token-budget
        fetches.

        Mirrors ``train_async``'s auto-selection of the streaming path: only
        the fully-async + dynamic-batch combination benefits, and we currently
        only support PP=1 (VPP implies PP>1) because the forward+putback loop
        has no pipelined meta tracking across stages.
        """
        return (
            getattr(self.args, "fully_async", False)
            and getattr(self.args, "use_dynamic_batch_size", False)
            and mpu.get_pipeline_model_parallel_world_size() == 1
            and (mpu.get_virtual_pipeline_model_parallel_world_size() or 1) == 1
        )

    def _streaming_fwd_putback(
        self,
        rollout_id: int,
        task_name: str,
        store_prefix: str,
        data_fields: list,
        clear_routing_replay_forward: bool = False,
    ) -> None:
        """Stream token-budget micro-batches, run forward-only, put log_probs
        back.

        Forward has no backward / gradient all-reduce, so each DP rank drains
        its own per-DP bucket independently (no cross-DP ``num_microbatches``
        MAX alignment, unlike ``get_data_iterator``).  Each streamed chunk is
        already bounded by ``token_budget`` (~one micro-batch), so it is
        forwarded as a single micro-batch.  ``forward_only`` reorders outputs
        by ``micro_batch_indices`` when ``use_dynamic_batch_size`` is set, so
        we build the iterator with an identity index list.
        """
        dp_rank = mpu.get_data_parallel_rank()
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
        cp_size = mpu.get_context_parallel_world_size()
        token_budget = self.args.max_tokens_per_gpu * cp_size

        streaming_iter = StreamingTQIterator(
            args=self.args,
            tq_client=self.data_system_client,
            data_fields=data_fields,
            rollout_id=rollout_id,
            token_budget=token_budget,
            loss_scale=1.0,  # forward-only: __loss_scale__ is unused
            # streaming=True drained predicate; PP=1 here (see _use_streaming_fwd) so
            # all_consumed's PP reduction is harmless.
            all_consumed_fn=lambda: self.all_consumed(
                task_name, rollout_id, partition_id=f"train_{rollout_id}", streaming=True
            ),
            dp_rank=dp_rank,
            dp_size=dp_size,
            task_name=task_name,
        )
        for data, batch_meta in streaming_iter:
            n = len(data["total_lengths"])
            data_iterator = [DataIterator(data, micro_batch_indices=[list(range(n))])]
            num_microbatches = [1]
            output_dict = self.compute_log_prob(
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )
            self._put_data_to_transfer_queue(output_dict, batch_meta, data)
            if clear_routing_replay_forward:
                RoutingReplay.clear_all_forward()

    def train_hybrid(self, rollout_id) -> None:
        """Hybrid mode: actor internally handles ref/actor_fwd/advantages
        computation via _switch_model, then trains, while using the async data
        pipeline (transfer queue with max-staleness).

        This combines colocate's weight-sharing (TensorBackuper + _switch_model) with
        fully-async's streaming data pipeline. Actor and rollout run on separate GPUs,
        but actor/ref/actor_fwd share the same GPUs via role switching.

        Data is processed in sub-batches (controlled by num_iters_per_train_update) to
        reduce peak GPU memory during forward passes, matching fully-async behavior.
        Advantages are computed after all sub-batches are collected to ensure correct
        global normalization across the full batch and DP group.
        """
        logger.info(f"start to get rollout_id: {rollout_id} data from transfer queue for train_hybrid.")
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
        plan = build_rollout_minibatch_plan(self.args, dp_size)
        batch_size = plan.mini_local_sample_request

        # ── Phase 1: Collect sub-batches and compute ref/actor forward in small chunks ──
        collected_batches: list[RolloutBatch] = []
        rollout_mini_local_sample_counts: list[int] = []

        def mark_rollout_mini_boundary(sub_batch: RolloutBatch) -> None:
            sub_batch[ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY] = [len(sub_batch["total_lengths"])]

        if self.args.debug_train_only:
            # Bypass the transfer queue and load the offline debug rollout dump
            # directly (mirrors `train`'s debug_train_only path). The dump holds
            # the full rollout (rollout_batch_size * n_samples_per_prompt) for
            # this step, so load the whole per-rank slice and split it into
            # rollout mini windows.
            logger.info(f"start to get rollout_id: {rollout_id} data from debug rollout data for train_hybrid.")
            full_batch_size = plan.mini_local_sample_request * plan.num_rollout_minis
            debug_data = get_debug_data(self.args, rollout_id, full_batch_size, dp_rank=mpu.get_data_parallel_rank())
            post_process_rollout_data(self.args, debug_data)
            debug_mini_row_counts = _rollout_mini_row_counts(
                debug_data["sample_indices"],
                plan.mini_local_sample_request,
                plan.num_rollout_minis,
            )
            for sub_batch in self._split_rollout_batch(debug_data, debug_mini_row_counts):
                mark_rollout_mini_boundary(sub_batch)
                self._hybrid_forward_subbatch(sub_batch)
                collected_batches.append(sub_batch)
                rollout_mini_local_sample_counts.append(len(sub_batch["total_lengths"]))
        else:
            batch_index = 0
            # Surface stuck-loop conditions: when the partition can never reach the
            # requested batch_size (e.g. rollout dropped samples without refilling),
            # `get_meta` keeps returning size=0 while `all_consumed` stays False,
            # producing a silent infinite spin. Warn periodically so the failure mode
            # is visible in logs instead of presenting as a totally silent hang.
            loop_start = time.monotonic()
            last_progress = loop_start
            last_warn = loop_start
            while batch_index < plan.num_rollout_minis and not self.all_consumed("train", rollout_id, streaming=True):
                data_fields = [
                    "tokens",
                    "total_lengths",
                    "response_lengths",
                    "loss_masks",
                    "rollout_log_probs",
                    "rewards",
                    "sample_indices",
                    "sample_index_mask_sums",
                    "raw_reward",
                    "group_index",
                ]
                data_fields += ["rollout_routed_experts"] if self.args.use_rollout_routing_replay else []
                if self.args.multimodal_keys is not None:
                    data_fields.append("multimodal_train_inputs")
                if self.args.use_opd and self.args.opd_type == "sglang":
                    data_fields.append("teacher_log_probs")
                with timer("train_get_data"):
                    sub_batch, batch_meta = self._get_data_from_transfer_queue(
                        "train", rollout_id, data_fields, batch_size, batch_index
                    )
                if sub_batch is None:
                    now = time.monotonic()
                    stalled = now - last_progress
                    if now - last_warn >= 60.0 and stalled >= 60.0:
                        logger.warning(
                            f"train_hybrid({rollout_id}) batch_index={batch_index} stalled for {stalled:.0f}s: "
                            f"partition train_{rollout_id} has no data of size={batch_size} available but "
                            f"all_consumed=False. Likely the rollout under-filled this partition."
                        )
                        last_warn = now
                    # Throttle the spin so the controller is not hammered with metadata
                    # polls while we wait for upstream data.
                    time.sleep(0.1)
                    continue
                last_progress = time.monotonic()
                last_warn = last_progress
                batch_index += 1

                # Forward passes on this sub-batch (small memory footprint)
                mark_rollout_mini_boundary(sub_batch)
                self._hybrid_forward_subbatch(sub_batch)
                collected_batches.append(sub_batch)
                rollout_mini_local_sample_counts.append(len(sub_batch["total_lengths"]))

        if len(collected_batches) != plan.num_rollout_minis:
            raise RuntimeError(
                f"Expected {plan.num_rollout_minis} rollout mini batches for train_hybrid({rollout_id}), "
                f"got {len(collected_batches)}."
            )

        if self._active_model_tag != "actor":
            self._switch_model("actor")

        # ── Phase 2: Merge sub-batches and compute advantages with correct global normalization ──
        # Merge all sub-batch dicts: each value is a list, so we concatenate them.
        rollout_data: RolloutBatch = {}
        for sb in collected_batches:
            for key, value in sb.items():
                if key not in rollout_data:
                    rollout_data[key] = []
                if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
                    rollout_data[key].extend(value)
                else:
                    rollout_data[key].append(value)
        rollout_data[ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY] = rollout_mini_local_sample_counts

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            # ── Phase 3: Train on the full merged batch ──
            data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(
            self.args, rollout_id=rollout_id, rollout_data=rollout_data, tokenizer=self.tokenizer
        )

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        # Update CPU actor weight backup
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        total_lengths = rollout_data["total_lengths"]
        all_total_lengths = [None] * mpu.get_data_parallel_world_size(with_context_parallel=False)
        dist.all_gather_object(
            all_total_lengths, total_lengths, group=mpu.get_data_parallel_group(with_context_parallel=False)
        )
        all_total_lengths = sum(all_total_lengths, [])  # flatten
        Timer().seq_lens = all_total_lengths
        mm_inputs = rollout_data.get("multimodal_train_inputs")
        if mm_inputs is not None:
            images_seqlens = _extract_images_seqlens(mm_inputs)
            all_images_seqlens = [None] * mpu.get_data_parallel_world_size(with_context_parallel=False)
            dist.all_gather_object(
                all_images_seqlens, images_seqlens, group=mpu.get_data_parallel_group(with_context_parallel=False)
            )
            Timer().images_seqlens = sum(all_images_seqlens, [])
            audio_seqlens = _extract_audio_seqlens(mm_inputs)
            all_audio_seqlens = [None] * mpu.get_data_parallel_world_size(with_context_parallel=False)
            dist.all_gather_object(
                all_audio_seqlens, audio_seqlens, group=mpu.get_data_parallel_group(with_context_parallel=False)
            )
            Timer().audio_seqlens = sum(all_audio_seqlens, [])
        log_perf_data(rollout_id, self.args, flops_counter=self.flops_counter)

        is_train_done = (rollout_id + 1) == self.args.num_rollout
        if self.args.save is not None and (
            self.args.rotate_ckpt
            or self.args.save_interval is not None
            and ((rollout_id + 1) % self.args.save_interval == 0 or is_train_done)
        ):
            self.save_model(rollout_id, force_sync=is_train_done)

        if self.args.debug_train_only:
            # In debug_train_only mode no rollout/eval services exist, so skip the
            # weight-sync + eval coordination below (mirrors `train`'s debug path
            # which never touches those services). Metrics are still flushed.
            tracking_utils.flush_metrics(self.args, compute_rollout_step(self.args, rollout_id))
            return

        # Mirror train_async's pause/resume coordination so the rollout service
        # has a chance to finish its in-flight step (and refill any partition
        # gaps it owes) before we swap weights. Without this gate the rollout
        # can race ahead while train_hybrid is still mid-consume on a
        # partially-filled partition, deadlocking against the staleness bound.
        # Returned flags are for update_weights_fully_async only — hybrid uses
        # the sync update_weights path so we just discard them.
        self._wait_for_previous_eval()
        self._check_services_health()

        # Sync weights to rollout via UpdateWeightFromTensor (colocate mode)
        self.update_weights()
        tracking_utils.flush_metrics(self.args, compute_rollout_step(self.args, rollout_id))
        dist.barrier(group=get_gloo_group())
        self._run_step_evaluation(rollout_id, end_update_weight=True)

        # On the final training step the rollout component has already exited
        # its main loop, so the eval just triggered above will not be awaited
        # anywhere. Block until it finishes; otherwise the controller's atexit
        # shutdown races with eval and tears down the SGLang engines mid-flight.
        if is_train_done:
            self._wait_for_previous_eval()

    def train_async(self, rollout_id) -> None:
        if self.args.use_routing_replay:
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"

        logger.info(f"start to get rollout_id: {rollout_id} data from transfer queue for train step.")
        data_fields = [
            "tokens",
            "total_lengths",
            "response_lengths",
            "loss_masks",
            "advantages",
            "returns",
            "rollout_log_probs",
            "rewards",
            "sample_indices",
            "sample_index_mask_sums",
            "raw_reward",
            "group_index",
        ]
        # In true on-policy mode, actor_fwd is absent and old_log_probs is
        # recomputed inline from the train forward (see policy_loss_function).
        if not getattr(self.args, "true_on_policy_mode", False):
            data_fields.append("log_probs")
        if self.args.kl_coef != 0 or self.args.use_kl_loss:
            data_fields.append("ref_log_probs")
        if self.args.multimodal_keys is not None:
            data_fields.append("multimodal_train_inputs")
        if self.args.use_opd:
            consume_opd_train_data(data_fields, self.args)
        if getattr(self.args, "use_dynamic_batch_size", False):
            # Fully-async + dynamic-batch path: drain this DP's bucket from TQ
            # using token-budget fetches, align num_microbatches across DPs
            # via all-reduce(MAX) + dummy mb padding, then run a single
            # train_one_step.  See docs/draft/dynamic_batch_size_fully_async.md.
            self.data_iterator, self.num_microbatches = self._drain_dynamic_batch_rollout(rollout_id, data_fields)
        elif self.data_iterator is None:
            self.data_iterator, self.num_microbatches = create_stream_dataloader(
                self.args,
                rollout_id=rollout_id,
                task_name="actor_train",
                data_fields=data_fields,
                dp_rank=mpu.get_data_parallel_rank(),
            )
        else:
            for data_iterator in self.data_iterator:
                data_iterator.step(f"train_{rollout_id}")
        with inverse_timer("train_wait"), timer("train"):
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    self.data_iterator,
                    self.num_microbatches,
                )
            self.prof.step(rollout_id=rollout_id)
            if len(self.data_iterator) > 1 and all(
                isinstance(iterator, StreamingTQIterator) for iterator in self.data_iterator
            ):
                data_for_log = []
                for data_iterator in self.data_iterator:
                    data_for_log.extend(data_iterator.get_buffer())
            else:
                data_for_log = self.data_iterator[0].get_buffer()
            rollout_data = merge_dict_list(data_for_log)
            log_rollout_data(rollout_id, self.args, rollout_data)
            train_dump_utils.save_debug_train_data(
                self.args, rollout_id=rollout_id, rollout_data=rollout_data, tokenizer=self.tokenizer
            )

            # Wait for prior eval before pausing rollout for weight sync.
            self._wait_for_previous_eval()

            rollout_only, actor_fwd_only = self._check_services_health()
            self.update_weights_fully_async(rollout_id, rollout_only=rollout_only, actor_fwd_only=actor_fwd_only)
            dist.barrier(group=get_gloo_group())
            self._run_step_evaluation(rollout_id, end_update_weight=True)
            # On the final training step the rollout component has already
            # exited its main loop, so the eval just triggered above will not
            # be awaited anywhere. Block until it finishes; otherwise the
            # controller's atexit shutdown races with eval and tears down the
            # SGLang engines mid-flight.
            if (rollout_id + 1) == self.args.num_rollout:
                self._wait_for_previous_eval()

            if self.args.use_routing_replay:
                RoutingReplay.clear_all()

        total_lengths = rollout_data["total_lengths"]
        all_total_lengths = [None] * mpu.get_data_parallel_world_size(with_context_parallel=False)
        dist.all_gather_object(
            all_total_lengths, total_lengths, group=mpu.get_data_parallel_group(with_context_parallel=False)
        )
        all_total_lengths = sum(all_total_lengths, [])  # flatten
        Timer().seq_lens = all_total_lengths
        response_token_counts = [
            int(m.sum().item()) if isinstance(m, torch.Tensor) else int(sum(m)) for m in rollout_data["loss_masks"]
        ]
        all_response_token_counts = [None] * mpu.get_data_parallel_world_size(with_context_parallel=False)
        dist.all_gather_object(
            all_response_token_counts,
            response_token_counts,
            group=mpu.get_data_parallel_group(with_context_parallel=False),
        )
        all_response_token_counts = sum(all_response_token_counts, [])  # flatten
        Timer().response_lens = all_response_token_counts
        mm_inputs = rollout_data.get("multimodal_train_inputs")
        if mm_inputs is not None:
            images_seqlens = _extract_images_seqlens(mm_inputs)
            all_images_seqlens = [None] * mpu.get_data_parallel_world_size(with_context_parallel=False)
            dist.all_gather_object(
                all_images_seqlens, images_seqlens, group=mpu.get_data_parallel_group(with_context_parallel=False)
            )
            Timer().images_seqlens = sum(all_images_seqlens, [])
            audio_seqlens = _extract_audio_seqlens(mm_inputs)
            all_audio_seqlens = [None] * mpu.get_data_parallel_world_size(with_context_parallel=False)
            dist.all_gather_object(
                all_audio_seqlens, audio_seqlens, group=mpu.get_data_parallel_group(with_context_parallel=False)
            )
            Timer().audio_seqlens = sum(all_audio_seqlens, [])
        log_perf_data(rollout_id, self.args, flops_counter=self.flops_counter)
        tracking_utils.flush_metrics(self.args, compute_rollout_step(self.args, rollout_id))

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving; resume the
        # paused model (process groups + tms) so save can issue collectives and
        # touch GPU tensors.
        if self.args.offload_train and self._per_step_rollout:
            reload_process_groups()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        if dist.get_rank() == 0:
            rotate_ckpt(self.args, global_step=rollout_id)

        dist.barrier(group=get_gloo_group())

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from relax.backends.megatron.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model, force_sync=force_sync)

        if self.args.offload_train and self._per_step_rollout:
            destroy_process_groups()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.offload_train:
            # CRITICAL: Barrier before onload_weights to ensure ALL ranks have
            # completed sleep() (and released GPU memory via tms.pause()) before
            # SGLang's resume_memory_occupation tries to reclaim GPU memory.
            # Without this, rank 0 may trigger SGLang resume while other ranks
            # still hold GPU memory, causing cuMemCreate OOM in SGLang schedulers.
            dist.barrier(group=get_gloo_group())

        if self.args.offload_rollout and dist.get_rank() == 0:
            # Onload rollout weights. genRM (no NCCL weight sync — the reward
            # model is static) is deferred to after the weight all-gather: in
            # colocate mode its static pool would collide with the all-gather's
            # temp buffers and OOM. See onload_kv below (post_sync_handles).
            onload_handles = [self.rollout_manager.onload_weights.remote()]
            append_managed_opd_teacher_onload_handle(onload_handles, self)
            ray.get(onload_handles)

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_rollout_engines.remote())
            dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )

        # Disaggregate PPO tears down the actor↔rollout NCCL groups on sleep(),
        # so we must wake_up() fully (not just reload_process_groups) and force
        # a reconnect here, then sleep() at the end.
        reconnect_rollout_engines = self.args.offload_train and self.args.use_critic and not self.args.colocate

        if reconnect_rollout_engines:
            self.wake_up()
        elif self.args.offload_train:
            reload_process_groups(timeout_minutes=self.args.distributed_timeout_minutes)

        if num_new_engines > 0 or reconnect_rollout_engines:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        with self._train_state_offloader.disable_during_update():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights", clear_before_print=not device_utils.is_npu_available)

            if self.args.ci_test and len(rollout_engines) > 0:
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")
        if reconnect_rollout_engines:
            self.sleep()
        elif self.args.offload_train:
            destroy_process_groups()

        # RL warms KV here for the next per-step generate. SFT's /predict
        # calls onload_kv itself. genRM (deferred from before the weight
        # all-gather) is onloaded here too, in parallel.
        # When --defer-reward-to-post-process is set the userland
        # custom_reward_post_process function owns GenRM lifecycle, so skip
        # onloading GenRM here (it must stay offloaded for rollout to have
        # all GPUs during generate in shared-bundles mode).
        if self.args.offload_rollout and dist.get_rank() == 0:
            post_sync_handles = []
            if self._per_step_rollout:
                post_sync_handles.append(self.rollout_manager.onload_kv.remote())
            if self.genrm_manager is not None and not getattr(self.args, "defer_reward_to_post_process", False):
                post_sync_handles.append(self.genrm_manager.onload.remote())
            if post_sync_handles:
                ray.get(post_sync_handles)

    @timer("wait update_weights_fully_async")
    def _check_services_health(self) -> tuple[bool, bool]:
        """Check rollout and actor_fwd service health before weight update.

        Only rank 0 sends HTTP requests to check service availability, then
        the results are broadcast via allreduce to ensure all ranks have a
        consistent view.

        Returns:
            (rollout_only, actor_fwd_only): Flags indicating which services
            are unavailable and should be skipped during weight update.
        """
        # Default: both services healthy → update both
        rollout_only = False
        actor_fwd_only = False

        # When true_on_policy_mode is enabled, actor_fwd is intentionally absent
        # (its log_probs are recomputed inline by the train forward). Force
        # rollout-only weight update and skip the actor_fwd HTTP probe.
        actor_fwd_absent = getattr(self.args, "true_on_policy_mode", False)

        if dist.get_rank() == 0:
            # Check rollout service
            try:
                rollout_serve_url = get_serve_url("rollout")
                while True:
                    response = requests.get(
                        f"{rollout_serve_url}/can_do_update_weight_for_async",
                        timeout=self.args.rollout_http_timeout,
                    )
                    response.raise_for_status()
                    res = response.json()
                    if res:
                        response = requests.get(f"{rollout_serve_url}/recover_rollout_engines")
                        response.raise_for_status()
                        break
                    else:
                        time.sleep(1)
            except Exception as e:
                logger.warning(
                    f"Error checking rollout service: {e}, maybe caused by rollout server failure. "
                    "Will continue without rollout update for this step."
                )
                actor_fwd_only = True

            # Check actor_fwd service. Skip the probe when actor_fwd is
            # intentionally absent — either true_on_policy_mode (log_probs
            # recomputed inline by train forward) or hybrid mode (actor
            # handles forward via _switch_model, no separate service).
            if actor_fwd_absent or getattr(self.args, "hybrid", False):
                rollout_only = True
            else:
                try:
                    actor_fwd_serve_url = get_serve_url("actor_fwd")
                    response = requests.get(f"{actor_fwd_serve_url}/get_step", timeout=self.args.rollout_http_timeout)
                    response.raise_for_status()
                except Exception as e:
                    logger.warning(
                        f"Error checking actor_fwd service: {e}, maybe caused by actor_fwd server failure. "
                        "Will continue without actor_fwd update for this step."
                    )
                    rollout_only = True

        # Broadcast results from rank 0 to all ranks via allreduce
        # Encode booleans as integers: 1 = skip, 0 = healthy
        flags = torch.tensor(
            [int(rollout_only), int(actor_fwd_only)],
            dtype=torch.int32,
            device="cpu",
        )
        dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=get_gloo_group())
        rollout_only = bool(flags[0].item())
        actor_fwd_only = bool(flags[1].item())

        return rollout_only, actor_fwd_only

    def _wait_for_previous_eval(self, max_wait_seconds: int = 1800) -> None:
        """Block until the rollout service's previous evaluation has finished.

        Only rank 0 polls the rollout HTTP endpoint; all ranks synchronise via
        a barrier afterwards so they proceed together.
        Args:
            max_wait_seconds: Maximum time (in seconds) to wait for eval
                completion before giving up and proceeding. Defaults to 300.
        """
        if dist.get_rank() == 0:
            try:
                rollout_serve_url = get_serve_url("rollout")
                start_time = time.monotonic()
                while True:
                    response = requests.get(f"{rollout_serve_url}/is_eval_done", timeout=10)
                    response.raise_for_status()
                    if response.json().get("done", True):
                        break
                    elapsed = time.monotonic() - start_time
                    if elapsed >= max_wait_seconds:
                        logger.warning(
                            f"Timed out waiting for previous evaluation after {elapsed:.1f}s "
                            f"(max_wait_seconds={max_wait_seconds}), proceeding with weight update."
                        )
                        break
                    logger.info("Waiting for previous evaluation to finish before updating weights...")
                    time.sleep(10)
            except Exception as e:
                logger.warning(f"Error checking eval status: {e}, proceeding with weight update.")
        dist.barrier(group=get_gloo_group())

    @timer
    def update_weights_fully_async(self, rollout_id, rollout_only=False, actor_fwd_only=False) -> None:
        """Fully async version of update_weights, which will be called by actor
        train engine after it train update.

        This sends weights to both rollout and actor_fwd nodes using pipelined
        design.
        """
        if rollout_only and actor_fwd_only:
            logger.warning("Both rollout_only and actor_fwd_only are True, skipping async weight update.")
            return

        dist.barrier(group=get_gloo_group())
        print_memory("before update_weights")

        weight_sync_lock = getattr(self, "_weight_sync_lock", None)
        if weight_sync_lock is not None and dist.get_rank() == 0:
            acquired = False
            while not acquired:
                acquired = ray.get(weight_sync_lock.acquire.remote())
                if not acquired:
                    time.sleep(1)

        try:
            if not rollout_only:
                run(self.checkpoint_engine_client.init_process_groups_for_actor_fwd_ref(rollout_id))
            run(self.checkpoint_engine_client.update_weights_for_rollout(rollout_only, actor_fwd_only))
        finally:
            if weight_sync_lock is not None and dist.get_rank() == 0:
                ray.get(weight_sync_lock.release.remote())

    @timer
    def recv_weight_fully_async(self, rollout_id) -> None:
        """Receive weights from actor (called by actor_fwd side)."""
        dist.barrier(group=get_gloo_group())
        print_memory("before update_weights")
        run(self.checkpoint_engine_client.init_process_groups_for_actor_fwd_ref(rollout_id))
        run(self.checkpoint_engine_client.recv_weight_fully_async())
        print_memory("after update_weights", clear_before_print=not device_utils.is_npu_available)

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        old_ckpt_step = None
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step
        elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if old_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def all_consumed(self, task_name, rollout_id, partition_id: str | None = None, streaming: bool = False):
        # Only (TP=0, PP=0, CP=0) queries the transfer queue; otherwise different cp_ranks
        # may observe different consumption status due to concurrent fetches and diverge,
        # leaving some ranks idle while others enter the next collective and hang.
        #
        # streaming=True uses the producer-driven drained predicate (check_stream_drained)
        # instead of the tensor-wide .all() check, which is unreliable without a preset
        # partition size. This is the lockstep-across-PP drain path, so including the PP
        # group below is safe (unlike the 1F1B schedule — see all_consumed_streaming).
        if partition_id is None:
            partition_id = sft_partition_id(self.args, rollout_id)
        if (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
            and mpu.get_context_parallel_rank() == 0
        ):
            if streaming:
                status = run(self.data_system_client.async_check_stream_drained(task_name, partition_id))
            else:
                status = run(self.data_system_client.async_check_consumption_status(task_name, partition_id))
        else:
            status = True

        return _agree_drained(status, include_pipeline=True)

    def all_consumed_streaming(self, task_name, rollout_id, window_id=None, window_quota=None):
        """End-of-stream check for the streaming PP schedule — NO pipeline-
        group collective.

        ``all_consumed`` reduces the consumption flag across the PP group.
        That is FATAL for the streaming iterator: each PP stage pulls from its
        own ``StreamingTQIterator`` at different points in the 1F1B schedule
        (warmup / steady / cooldown), so stage A may call this check (because its
        sampler returned empty) while stage B is busy in fwd/bwd compute and not
        calling it.  A PP-group collective then blocks stage A forever waiting for
        the others to join → the hang observed at long sequences (PP>1, any DP).

        The sampler result cache guarantees every PP stage sees the SAME data
        sequence per ``batch_index``, so each stage independently reaches
        end-of-stream at the same micro-batch count.  We therefore query the
        controller WITHOUT any PP collective.  Within a PP stage the TP ranks
        are in lockstep (only tp_rank==0 fetches; the get_data TP broadcast keeps
        them aligned on the "data is None" branch), so we reduce the flag over
        the tensor-and-context-parallel group ONLY to give tp_rank>0 / cp_rank>0
        the same answer — those ranks are in lockstep within a stage.

        When ``window_id`` is given (multi-mini train path) the check is PER
        WINDOW: the sampler reports drained once it has globally dispatched
        ``window_quota`` (``mini_global_samples``) for this rollout-mini window,
        or the whole partition is drained (under-produced final window).
        Otherwise it falls back to the partition-level stream-drained check.
        """
        if mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
            # Streaming end-of-stream predicate (no preset global batch): the producer
            # marks production_completed and we require every actually-inserted sample
            # to be consumed, rather than a tensor-wide .all() over (possibly dynamic)
            # rows. See TransferQueue check_stream_drained / check_window_drained.
            if window_id is not None:
                status = run(
                    self.data_system_client.async_check_window_drained(
                        task_name, f"train_{rollout_id}", window_id, window_quota
                    )
                )
            else:
                status = run(self.data_system_client.async_check_stream_drained(task_name, f"train_{rollout_id}"))
        else:
            status = True

        # Intra-PP-stage group only (TP x CP in a single collective). Crucially NO
        # pipeline-group reduction: PP stages are NOT in lockstep during 1F1B, so a
        # PP collective here would deadlock.
        return _agree_drained(status, include_pipeline=False)

    def _drain_dynamic_batch_rollout(self, rollout_id, data_fields):
        """Build data iterators for the fully-async + dynamic-batch train path.

        Streaming mode (default when ``use_dynamic_batch_size`` + ``fully_async``):
            Returns one ``StreamingTQIterator`` per rollout-mini window.  Each
            iterator pulls token-budget micro-batches from the same TQ partition
            until its local sample target is reached, and the training loop runs
            one optimizer step per iterator.  All PP stages independently build
            their own iterators; the sampler result cache ensures they all see
            the same data sequence and raise ``StopIteration`` at the same
            micro-batch count.

        Legacy mode (VPP or explicit opt-out):
            Pre-drains the full bucket, cross-DP MAX-aligns ``num_microbatches``,
            pads shorter DPs with dummy mbs, and returns a ``MicroBatchListIterator``.
            This is the only supported mode when VPP is active (interleaved schedule
            requires a fixed ``num_microbatches``).
        """
        dp_rank = mpu.get_data_parallel_rank()
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
        cp_size = mpu.get_context_parallel_world_size()
        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1

        task_name = "actor_train"
        token_budget = self.args.max_tokens_per_gpu * cp_size
        plan = build_rollout_minibatch_plan(self.args, dp_size)
        # loss_scale: each sample in one optimizer step is weighted equally.
        # dp_world_size_with_cp compensates for the later DP allreduce average.
        loss_scale = 1.0 / plan.mini_global_samples * mpu.get_data_parallel_world_size(with_context_parallel=True)

        # Use streaming iterator when VPP is inactive (streaming schedule handles PP>1).
        use_streaming = vpp_size == 1

        if use_streaming:
            batch_index_stride = 1_000_000
            data_iterator = [
                StreamingTQIterator(
                    args=self.args,
                    tq_client=self.data_system_client,
                    data_fields=data_fields,
                    rollout_id=rollout_id,
                    token_budget=token_budget,
                    loss_scale=loss_scale,
                    # PP-collective-free PER-WINDOW end-of-stream check: the
                    # streaming PP schedule pulls per-stage at different schedule
                    # points, so a PP-group broadcast inside __next__ would
                    # deadlock.  window_id + window_quota make it query per-window
                    # drained (sampler dispatched mini_global_samples for this
                    # window, or partition drained).  Default-arg binds mini_idx
                    # to avoid the late-binding closure bug.  See
                    # all_consumed_streaming for the full rationale.
                    all_consumed_fn=(
                        lambda mi=mini_idx: self.all_consumed_streaming(
                            task_name, rollout_id, window_id=mi, window_quota=plan.mini_global_samples
                        )
                    ),
                    # Train path: sampler-driven per-round dummy alignment. On an
                    # empty fetch the iterator reads the DUMMY-round flag from the
                    # fetched meta and emits a zero-grad dummy so all DPs run equal
                    # micro-batch counts (MoE EP all-to-all matches by call order).
                    per_round_dummy=True,
                    dp_rank=dp_rank,
                    dp_size=dp_size,
                    task_name=task_name,
                    # Global per-window quota (mini_global_samples): the sampler
                    # stops dispatching this window once the quota is met across
                    # all DPs.  Per-DP counts stay uneven (token-budget driven);
                    # the dummy-pad aligns cross-DP mb counts.  No per-DP hard
                    # count target on the consumer — termination is the per-window
                    # drained signal above.
                    window_quota=plan.mini_global_samples,
                    rollout_mini_index=mini_idx,
                    start_batch_index=mini_idx * batch_index_stride,
                )
                for mini_idx in range(plan.num_rollout_minis)
            ]
            num_microbatches = [1 for _ in range(plan.num_rollout_minis)]  # streaming schedule ignores this value
            logger.info(
                "[dynamic-batch] rollout=%s split into %d rollout-mini windows "
                "(mini_global_samples=%d mini_local_samples=%d)",
                rollout_id,
                plan.num_rollout_minis,
                plan.mini_global_samples,
                plan.mini_local_sample_request,
            )
            return data_iterator, num_microbatches

        # ── Legacy drain path (VPP only) ────────────────────────────────────
        if plan.num_rollout_minis != 1:
            raise NotImplementedError(
                "fully_async + dynamic-batch rollout-mini updates are only supported without virtual pipeline "
                f"parallelism. Got vpp_size={vpp_size}, num_rollout_minis={plan.num_rollout_minis}."
            )
        dp_group = mpu.get_data_parallel_group()
        partition_id = f"train_{rollout_id}"
        sampling_config = {
            "dp_rank": dp_rank,
            "dp_size": dp_size,
            "task_name": task_name,
        }

        mbs: List[tuple] = []
        batch_index = 0
        empty_streak = 0
        while not self.all_consumed(task_name, rollout_id, partition_id=partition_id, streaming=True):
            rollout_data, batch_meta = get_data_from_transfer_queue(
                self.args,
                self.data_system_client,
                data_fields,
                batch_size=None,
                partition_id=partition_id,
                task_name=task_name,
                sampling_config=sampling_config,
                batch_index=batch_index,
                broadcast_pp=False,
                token_budget=token_budget,
                allow_underfill=True,
            )
            if rollout_data is None:
                empty_streak += 1
                time.sleep(min(0.2 * empty_streak, 2.0))
                continue
            empty_streak = 0
            mbs.append((rollout_data, batch_meta))
            batch_index += 1

        k_local = len(mbs)

        device = device_utils.make_current_torch_device()
        k_tensor = torch.tensor([k_local], dtype=torch.int, device=device)
        dist.all_reduce(k_tensor, op=dist.ReduceOp.MAX, group=dp_group)
        k_global = int(k_tensor.item())
        logger.debug(
            f"[dynamic-batch] rollout={rollout_id} dp_rank={dp_rank} "
            f"k_local={k_local} k_global={k_global} "
            f"sample_counts={[len(m[0].get('total_lengths', [])) for m in mbs]} "
            f"token_totals={[sum(m[0].get('total_lengths', [0])) for m in mbs]}"
        )

        from megatron.core.utils import get_model_config

        config = get_model_config(self.model[0])
        vp_group_size = config.microbatch_group_size_per_vp_stage
        k_global = max(
            ((k_global + vp_group_size - 1) // vp_group_size) * vp_group_size,
            vp_group_size,
        )

        if k_local == 0 and k_global > 0:
            raise RuntimeError(
                f"DP rank {dp_rank} drained 0 micro-batches but K_global={k_global} "
                f"for rollout {rollout_id}. Check sampler balance / rollout production."
            )

        if k_local < k_global:
            shortest_idx = min(
                range(k_local),
                key=lambda i: sum(mbs[i][0].get("total_lengths", [0])),
            )
            template_batch, template_meta = mbs[shortest_idx]
            for _ in range(k_global - k_local):
                mbs.append((template_batch, template_meta))

        vpp_loss_scale = (
            k_global / plan.mini_global_samples * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
        data_iterator = [
            MicroBatchListIterator(mbs, dummy_after=k_local, loss_scale=vpp_loss_scale) for _ in range(vpp_size)
        ]
        num_microbatches = [k_global]
        return data_iterator, num_microbatches

    def _get_data_from_transfer_queue(
        self, task_name, rollout_id, data_fields, batch_size, batch_index, partition_id: str | None = None
    ):
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will recieve the data.
        if partition_id is None:
            partition_id = sft_partition_id(self.args, rollout_id)
        sampling_config = {
            # CP partners share one logical DP slot: they must present the same
            # dp_rank to the TQ sampler so its (partition_id, task_name, dp_rank,
            # batch_index) cache deduplicates their fetches. Otherwise CP=2 turns
            # 4 logical shards into 8 competing consumers and one of them gets
            # starved at the producer→consumer boundary (e.g. after sft-predict).
            "dp_rank": mpu.get_data_parallel_rank(with_context_parallel=False),
            "task_name": task_name,
        }
        # Skip the PP broadcast when PP world size is 1: it's a self-broadcast
        # but still pays the full pickle cost on every TP rank-0 fetcher
        # (multimodal pixel_values can be hundreds of MB → seconds per call).
        broadcast_pp = mpu.get_pipeline_model_parallel_world_size() > 1
        # Per-rank fetch (opt-in via --per-rank-fetch) lets every TP/PP
        # rank pull its own copy from TQ in parallel instead of paying one
        # rank-0 pickle + one TP/PP broadcast.  Cross-rank consistency relies
        # on every model-parallel rank presenting the same logical
        # ``dp_rank``/``batch_index`` to the TQ sampler, so sampler replay
        # returns byte-identical sample ids regardless of PP/TP world size. The only
        # remaining incompatibility is ``rollout_routed_experts`` — it relies on
        # the NestedTensor jagged bcast path that this mode bypasses.
        per_rank_fetch = self.args.per_rank_fetch and "rollout_routed_experts" not in data_fields
        rollout_data, batch_meta = get_data_from_transfer_queue(
            self.args,
            self.data_system_client,
            data_fields,
            batch_size,
            partition_id,
            task_name,
            sampling_config,
            batch_index,
            broadcast_pp=broadcast_pp,
            per_rank_fetch=per_rank_fetch,
        )

        return rollout_data, batch_meta

    def _gather_cp_output_for_transfer_queue(self, output_dict, rollout_data):
        token_fields = {
            "log_probs",
            "ref_log_probs",
            "rollout_log_probs",
            "teacher_log_probs",
            "values",
            "advantages",
            "returns",
            "opd_reverse_kl",
        }
        fields_to_gather = [
            key for key, value in output_dict.items() if key in token_fields and isinstance(value, List)
        ]
        if mpu.get_context_parallel_world_size() == 1 or not fields_to_gather:
            return output_dict
        if rollout_data is None:
            raise ValueError("rollout_data is required to gather CP-sharded outputs before putting to TransferQueue")

        total_lengths = [int(length) for length in rollout_data["total_lengths"]]
        response_lengths = [int(length) for length in rollout_data["response_lengths"]]
        max_seq_lens = rollout_data.get("max_seq_lens")
        if max_seq_lens is None:
            max_seq_lens = [None] * len(total_lengths)
        padded_total_lengths = maybe_padded_total_lengths(
            total_lengths,
            self.args.qkv_format,
            getattr(self.args, "is_vl_model", False)
            or "multimodal_train_inputs" in rollout_data
            or getattr(self.args, "uses_unsplit_forward", False),
        )
        if padded_total_lengths is None:
            padded_total_lengths = [None] * len(total_lengths)

        gathered_output = dict(output_dict)
        for key in fields_to_gather:
            values = output_dict[key]
            if len(values) != len(total_lengths):
                raise ValueError(
                    f"Cannot gather field '{key}' with {len(values)} values for {len(total_lengths)} samples"
                )
            gathered_output[key] = [
                all_gather_with_cp(
                    value,
                    total_length,
                    response_length,
                    padded_total_length=padded_total_length,
                    qkv_format=self.args.qkv_format,
                    max_seq_len=max_seq_len,
                )
                for value, total_length, response_length, max_seq_len, padded_total_length in zip(
                    values,
                    total_lengths,
                    response_lengths,
                    max_seq_lens,
                    padded_total_lengths,
                    strict=False,
                )
            ]

        return gathered_output

    def _put_data_to_transfer_queue(self, output_dict=None, batch_meta=None, rollout_data=None):
        if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
            output_dict = self._gather_cp_output_for_transfer_queue(output_dict, rollout_data)
            if mpu.get_context_parallel_rank() != 0:
                return
            output_dict = {
                key: value.cpu()
                if not isinstance(value, List)
                else torch.nested.as_nested_tensor([item.cpu() for item in value], layout=torch.jagged)
                for key, value in output_dict.items()
            }
            output_dict = TensorDict(output_dict, batch_size=[len(batch_meta.samples)])
            run(self.data_system_client.async_put(data=output_dict, metadata=batch_meta))

    def _put_critic_values_to_transfer_queue(self, rollout_data: RolloutBatch) -> None:
        if getattr(self.args, "advantage_estimator", None) != "ppo":
            return

        values = rollout_data.get("values")
        batch_metas = rollout_data.get(ROLLOUT_MINI_BATCH_METAS_KEY)
        counts = rollout_data.get(ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY)
        if values is None or batch_metas is None or counts is None:
            return
        if len(batch_metas) != len(counts):
            raise ValueError(
                f"{ROLLOUT_MINI_BATCH_METAS_KEY} length must match {ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY}, "
                f"got metas={len(batch_metas)}, counts={len(counts)}"
            )

        value_chunks = _split_by_rollout_mini_counts(values, counts)
        start = 0
        for value_chunk, batch_meta, count in zip(value_chunks, batch_metas, counts, strict=False):
            end = start + count
            mini_rollout_data = _slice_rollout_batch(rollout_data, start, end)
            self._put_data_to_transfer_queue({"values": value_chunk}, batch_meta, mini_rollout_data)
            start = end
