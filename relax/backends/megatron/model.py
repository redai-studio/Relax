# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import dataclasses
import gc
import math
import os
import string
import uuid
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config, unwrap_model
from megatron.training.global_vars import get_args
from megatron.training.training import get_model

from relax.backends.megatron.checkpoint import _save_lora_to_checkpoint
from relax.engine.sft.runtime import should_bypass_main_output_layer
from relax.utils import tracking_utils
from relax.utils.data.stream_dataloader import StreamingTQIterator
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.megatron_bridge_utils import patch_megatron_model
from relax.utils.megatron_peft_utils import is_lora_enabled
from relax.utils.memory_utils import clear_memory
from relax.utils.opd.opd_utils import consume_opd_train_data
from relax.utils.replay import capture_hooks
from relax.utils.timer import timer
from relax.utils.training.ppo_utils import (
    install_critic_value_head_runtime_check,
    maybe_verify_critic_value_head_movement,
    release_critic_lm_heads,
    release_sequence_classification_lm_heads,
    validate_critic_value_head_registration,
    validate_sequence_classification_head_registration,
)

from .checkpoint import load_checkpoint, save_checkpoint
from .data import ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY, DataIterator, get_batch
from .loss import loss_function
from .model_provider import (
    get_model_provider_func,
    validate_mtp_only_trainable_params,
    wrap_model_provider_with_freeze,
)


logger = get_logger(__name__)


def _find_lm_output_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    """Walk DDP / bridge-VL wrappers to the lm_head; None on non-last PP
    stages.

    ``unwrap_model`` strips Megatron's known wrapper classes (DDP, FP16, ...)
    in one shot. The bounded ``.module``/``.language_model`` walk that follows
    handles VL bridges and non-Megatron DDP shapes used by tests.
    """
    module = unwrap_model(model)
    for _ in range(4):  # bounded; bridge depth is at most 2
        ol = getattr(module, "output_layer", None)
        # Megatron sets `output_layer = nn.Identity()` on non-last PP stages
        # (placeholder); we must return None there so `_bypass_output_layer`
        # is a no-op and the loss never gets called on these ranks.
        if ol is not None and not isinstance(ol, torch.nn.Identity):
            return ol
        # `.module`: any residual DDP / FP16 / FP32 wrapper not stripped by
        # `unwrap_model` (e.g. test fakes, non-Megatron DDP shapes).
        # `.language_model`: Megatron-Bridge multimodal convention — every
        # known VL/Omni bridge (Qwen3-VL, Qwen3.5-VL, Qwen2.5-VL, Gemma3-VL,
        # Nemotron-VL, Qwen3-Omni) wraps the inner GPTModel under
        # `self.language_model`. If a future bridge breaks this convention,
        # this walk returns None → bypass becomes no-op → SFT chunked path
        # silently falls back to legacy (safe).
        module = getattr(module, "module", None) or getattr(module, "language_model", None)
        if module is None:
            return None
    return None


@contextmanager
def _bypass_output_layer(
    model: torch.nn.Module,
    *,
    mtp_output_layer_calls: int = 0,
    gather_passthrough: bool = True,
) -> Iterator[Callable | None]:
    """Defer the main output_layer so model() returns hidden_states.

    With ``--sequence-parallel`` the decoder emits ``[S/TP, B, H]`` and the
    original lm_head would AG before the matmul; we do that AG here so
    downstream SFT slicing sees the full sequence. The yielded callable runs
    the *original* lm_head forward with ``sequence_parallel=False`` (input
    already gathered) so it emits ``[chunk, 1, V/TP]`` per call.

    MTP invokes the same output layer once per prediction depth before the
    main head. ``mtp_output_layer_calls`` lets those calls use the real head;
    only the following main-head call becomes a passthrough. This preserves
    MTP loss computation while still deferring the main SFT logits.

    ``gather_passthrough=False`` keeps sequence-parallel hidden states local;
    MTP-only needs only a zero-valued autograd anchor and does not consume the
    gathered sequence. No-op on PP stages with no output layer.
    """
    assert mtp_output_layer_calls >= 0, f"{mtp_output_layer_calls=}"
    output_layer = _find_lm_output_layer(model)
    if output_layer is None:
        yield None
        return

    original_forward = output_layer.forward
    sp_enabled = bool(getattr(output_layer, "sequence_parallel", False))
    tp_group = getattr(output_layer, "tp_group", None) or mpu.get_tensor_model_parallel_group()
    remaining_mtp_calls = mtp_output_layer_calls
    deferred_weight = None
    main_head_deferred = False

    if sp_enabled and gather_passthrough:
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    def _passthrough(input_, weight=None, runtime_gather_output=None, **kwargs):
        nonlocal deferred_weight, main_head_deferred, remaining_mtp_calls
        if remaining_mtp_calls > 0:
            remaining_mtp_calls -= 1
            return original_forward(
                input_,
                weight=weight,
                runtime_gather_output=runtime_gather_output,
                **kwargs,
            )
        if main_head_deferred:
            raise RuntimeError("output_layer was called more than once after all MTP head calls")
        main_head_deferred = True
        deferred_weight = weight
        if sp_enabled and gather_passthrough:
            input_ = gather_from_sequence_parallel_region(input_, tensor_parallel_output_grad=False, group=tp_group)
        return input_, None

    def _chunked_call(input_, weight=None, runtime_gather_output=None):
        # ColumnParallelLinear's cuBLAS matmul requires input.dtype == weight.dtype.
        # The VL bridge upcasts hidden_states to fp32 before output_layer; downcast
        # here so matmul stays bf16/bf16. The caller upcasts logits back to fp32.
        call_weight = weight if weight is not None else deferred_weight
        w = call_weight if call_weight is not None else output_layer.weight
        if w is None:
            raise RuntimeError("Unable to resolve lm_head weight for chunked SFT loss")
        if input_.dtype != w.dtype:
            input_ = input_.to(w.dtype)
        prev_sp = output_layer.sequence_parallel
        output_layer.sequence_parallel = False
        try:
            return original_forward(input_, weight=call_weight, runtime_gather_output=runtime_gather_output)
        finally:
            output_layer.sequence_parallel = prev_sp

    output_layer.forward = _passthrough
    try:
        yield _chunked_call
        if not main_head_deferred:
            observed_mtp_calls = mtp_output_layer_calls - remaining_mtp_calls
            raise RuntimeError(
                "The main output_layer was not reached after the configured MTP head calls: "
                f"{mtp_output_layer_calls=}, {observed_mtp_calls=}"
            )
    finally:
        try:
            del output_layer.forward
        except AttributeError:
            output_layer.forward = original_forward


def _attach_mtp_forward_kwargs(args: Namespace, batch: dict, forward_kwargs: dict) -> None:
    """Attach Megatron MTP kwargs for training forwards."""
    if not getattr(args, "enable_mtp_training", False):
        return

    # VL+THD+CP unsplit path: bridge's preprocess_packed_seqs repacks
    # hidden_states with per-sample align=tp*cp*2, which does not match the
    # legacy `batch["tokens"]` / `batch["full_loss_masks"]` layout (per-sample
    # align=2*cp_size + global pad). data.py builds these bridge-aligned
    # tensors when the unsplit path is taken with MTP enabled; use them so
    # the rolled labels/mask line up with the MTP chunked hidden_states.
    if batch.get("unsplit_mtp_labels") is not None:
        forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["unsplit_mtp_labels"]}
        if forward_kwargs.get("loss_mask") is None:
            forward_kwargs["loss_mask"] = batch["unsplit_mtp_loss_mask"]
        return

    # Use the packed text-model labels. Qwen3/VL bridge forwards may receive
    # unsplit input_ids, then convert them to this layout internally.
    forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}
    if forward_kwargs.get("loss_mask") is None:
        forward_kwargs["loss_mask"] = batch["full_loss_masks"]


def _main_loss_has_tokens(batch: dict) -> bool:
    """Return whether the current CI batch still has any main-loss tokens."""
    loss_mask = batch.get("full_loss_masks")
    if loss_mask is None:
        return True

    num_tokens = loss_mask.detach().sum()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(num_tokens, group=mpu.get_data_parallel_group(with_context_parallel=True))
    return bool(num_tokens.item() > 0)


def get_optimizer_param_scheduler(args: Namespace, optimizer: MegatronOptimizer) -> OptimizerParamScheduler:
    """Create and configure the optimizer learning-rate/weight-decay scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        optimizer (MegatronOptimizer): Megatron optimizer bound to the model.

    Returns:
        OptimizerParamScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # Iteration-based training.
    args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def _build_optimizer_config_kwargs(args: Namespace) -> dict[str, object]:
    """Build optimizer kwargs from normalized runtime arguments."""
    kwargs = {}
    for field in dataclasses.fields(OptimizerConfig):
        if hasattr(args, field.name):
            kwargs[field.name] = getattr(args, field.name)
    if args.fp16:
        kwargs["bf16"] = False
        kwargs["fp16"] = True
        kwargs["params_dtype"] = torch.float16
        logger.info(f"FP16 mode enabled. Optimizer config: {kwargs}")
    return kwargs


def setup_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
    """Build model(s), wrap with DDP, and construct optimizer and scheduler.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        role (str): Logical role of the model (e.g., "actor", "critic").
        no_wd_decay_cond (Callable[..., bool] | None): Predicate to exclude
            parameters from weight decay.
        scale_lr_cond (Callable[..., bool] | None): Predicate to scale LR for
            selected parameter groups.
        lr_mult (float): Global learning-rate multiplier for the optimizer.

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
            - List of model chunks wrapped by ``DDP``.
            - The constructed ``MegatronOptimizer`` instance.
            - The learning-rate/weight-decay scheduler tied to the optimizer.
    """
    assert not args.moe_use_upcycling
    assert args.load is not None or args.pretrained_checkpoint is not None

    model = get_model(
        wrap_model_provider_with_freeze(get_model_provider_func(args, role), args),
        ModelType.encoder_or_decoder,
        wrap_with_ddp=role in ["actor", "critic"],
    )
    validate_mtp_only_trainable_params(args, model)

    # Some model providers (e.g., Qwen3VLGPTModel) rebuild the decoder in __init__,
    # which causes duplicate RoutingReplay registrations. Rebuild the list from
    # the actual model modules to remove stale (orphaned) entries.
    if Envs.ENABLE_ROUTING_REPLAY:
        from relax.utils.training.routing_replay import RoutingReplay

        active_replays = []
        for model_chunk in model:
            for module in model_chunk.modules():
                if hasattr(module, "routing_replay") and module.routing_replay is not None:
                    active_replays.append(module.routing_replay)
        if active_replays:
            RoutingReplay.all_routing_replays = active_replays

    if getattr(args, "dynamic_context_parallel", False) or getattr(args, "context_parallel_size", 1) > 1:
        # Global, idempotent GatedDeltaNet patch — apply whenever CP is active
        # (dynamic CP, or static context_parallel_size > 1), incl. weight-only
        # roles that still run forward.
        _patch_gdn_for_dynamic_cp()
        model_config = get_model_config(model[0])
        if getattr(model_config, "experimental_attention_variant", None) == "gated_delta_net" and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            logger.info(
                f"[GDN CP] role={role} linear_cp_mode={model_config.linear_cp_mode} "
                f"TP={model_config.tensor_model_parallel_size} max_CP={model_config.context_parallel_size} "
                f"key_heads={model_config.linear_num_key_heads} value_heads={model_config.linear_num_value_heads}"
            )

    if args.only_load_weight:
        return model, None, None
    # Optimizer
    kwargs = _build_optimizer_config_kwargs(args)
    config = OptimizerConfig(**kwargs)
    config.timers = None

    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=model,
        use_gloo_process_groups=args.use_gloo_process_groups,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    return model, optimizer, opt_param_scheduler


def _resolve_gdn_cp(self, packed_seq_params):
    """Resolve (cp_size, cp_group, cp_rank) for a GDN forward.

    Prefers the per-micro-batch dynamic CP group carried on
    ``packed_seq_params`` (set in ``data.py``); falls back to the module's
    static CP group.
    """
    if packed_seq_params is not None and getattr(packed_seq_params, "local_cp_size", None) is not None:
        cp_group = packed_seq_params.cp_group
        cp_size = packed_seq_params.local_cp_size
    else:
        cp_group = self.pg_collection.cp
        cp_size = cp_group.size()
    cp_rank = cp_group.rank() if cp_size > 1 else 0
    return cp_size, cp_group, cp_rank


def _assert_gdn_full_recompute() -> None:
    """Fail loud if GDN CP>1 runs without full activation recompute.

    The all-gather path below runs the recurrent scan on the *full* sequence
    duplicated on every CP rank, so the GDN activation scales with the full
    context length. Only ``--recompute-granularity full`` (whole-layer
    checkpointing) keeps that a per-layer transient; ``selective`` does not
    cover GDN (its module list has no gdn/mamba entry) and silently OOMs.

    Only relevant to training forwards that build a graph (and thus retain
    activations): skipped when grad is disabled (weight-only / inference roles
    that run the GDN forward under ``no_grad``).
    """
    if not torch.is_grad_enabled():
        return
    if getattr(_assert_gdn_full_recompute, "_checked", False):
        return
    args = get_args()
    if getattr(args, "recompute_granularity", None) != "full":
        raise ValueError(
            "GatedDeltaNet context-parallel (cp>1) requires whole-layer activation recompute: "
            "pass `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`. "
            f"Got recompute_granularity={getattr(args, 'recompute_granularity', None)!r}. "
            "`selective` recompute does not cover GDN and will OOM (its full-sequence duplicated scan "
            "activation stays resident)."
        )
    _assert_gdn_full_recompute._checked = True


def _gdn_cp_gather_full(qkvzba, cu_seqlens_cpu, cp_size, cp_group):
    """Thin wrapper over :func:`cp_utils.gdn_cp_gather_full` (kept for the
    local call site); the collective + reassembly live in cp_utils so they are
    importable/testable without the heavy training deps."""
    from .cp_utils import gdn_cp_gather_full

    return gdn_cp_gather_full(qkvzba, cu_seqlens_cpu, cp_size, cp_group)


def _patch_gdn_for_dynamic_cp() -> None:
    """Patch GDN forward for dynamic CP and Relax's all-gather mode.

    CP=1 and MCore-native headwise/chunkwise modes call the patched MCore
    forward directly. Only static ``linear_cp_mode='all_gather'`` with CP>1
    executes Relax's existing fallback. No shared module/config state is
    modified.
    """
    try:
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    except ImportError:
        return

    if getattr(GatedDeltaNet, "_dcp_patched", False):
        return

    _orig_forward = GatedDeltaNet.forward

    def _dcp_gdn_forward(
        self, hidden_states, attention_mask, inference_context=None, packed_seq_params=None, *args, **kwargs
    ):
        import torch._dynamo
        from megatron.core.ssm.gated_delta_net import causal_conv1d

        from .cp_utils import gdn_cp_slice

        cp_size, cp_group, cp_rank = _resolve_gdn_cp(self, packed_seq_params)
        if cp_size == 1 or self.config.linear_cp_mode != "all_gather":
            return _orig_forward(
                self, hidden_states, attention_mask, inference_context, packed_seq_params, *args, **kwargs
            )

        is_thd = packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd"
        assert is_thd, (
            "GDN linear_cp_mode='all_gather' with cp_size>1 only supports packed (thd) sequences; "
            "use linear_cp_mode='headwise' or 'chunkwise' for SBHD/static-batch inputs."
        )
        assert inference_context is None, "GDN all-gather CP path does not support inference."
        # Packed (thd) + deterministic is unsupported: a single conv/scan over the
        # concatenated samples would bleed state across cu_seqlens boundaries, and
        # torch_chunk_gated_delta_rule rejects cu_seqlens outright. Mirror upstream
        # GatedDeltaNet.forward, which asserts the same for packed sequences.
        assert not self.config.deterministic_mode, (
            "GDN context-parallel all-gather path does not support deterministic mode with "
            "packed (thd) sequences (cross-sample conv/scan contamination); matches upstream "
            "GatedDeltaNet."
        )
        _assert_gdn_full_recompute()

        cu_seqlens = packed_seq_params.cu_seqlens_q
        # Precompute the host-side boundary list once per micro-batch (cached on the
        # shared packed_seq_params object) so the gather/slice below don't force a
        # per-GDN-layer .tolist() device sync — repeated under full recompute.
        cu_seqlens_cpu = getattr(packed_seq_params, "_gdn_cu_seqlens_cpu", None)
        if cu_seqlens_cpu is None:
            cu_seqlens_cpu = cu_seqlens.tolist()
            packed_seq_params._gdn_cu_seqlens_cpu = cu_seqlens_cpu
        _, batch, _ = hidden_states.shape

        # Input projection on the CP-sharded (and SP-sharded) sequence.
        qkvzba, _ = self.in_proj(hidden_states)

        # Gather the full sequence across CP (duplicated scan, reduce-scatter grad),
        # sequential order.
        qkvzba = _gdn_cp_gather_full(qkvzba, cu_seqlens_cpu, cp_size, cp_group)

        seq_len = qkvzba.shape[0]
        qkvzba = qkvzba.transpose(0, 1)  # s b x -> b s x

        # Split into q/k/v, gate(z), beta, alpha using TP-local (full, no /cp) sizes.
        qk = self.qk_dim_local_tp
        v = self.v_dim_local_tp
        nvh = self.num_value_heads // self.tp_size
        qkv, gate, beta, alpha = torch.split(qkvzba, [qk * 2 + v, v, nvh, nvh], dim=-1)
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)

        # Convolution with the full (TP-local, un-CP-sliced) conv weights. Always the
        # FLA causal path: it resets state at each cu_seqlens boundary, so packed
        # samples don't leak across each other. (Deterministic mode is rejected above.)
        seq_len = qkv.shape[1]
        conv1d_weight = self.conv1d.weight
        conv1d_bias = self.conv1d.bias if self.conv_bias else None
        assert self.activation in ["silu", "swish"]
        qkv, _ = causal_conv1d(
            x=qkv,
            weight=conv1d_weight.squeeze(1),
            bias=conv1d_bias,
            activation=self.activation,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )

        # Reuse the module's own prep (split/l2norm/GQA-expand) with CP disabled so
        # its internal `// cp_size_headwise` becomes a no-op. Wrap in the dynamo-disable
        # guard added by docker/patch/megatron/20260506-85bced0ae.patch (Qwen3.6 GDN
        # torch.compile failure); calling _prepare_qkv_for_gated_delta_rule directly
        # would re-trigger that compile failure.
        with torch._dynamo.config.patch(disable=True):
            query, key, value, gate, beta, alpha = self._prepare_qkv_for_gated_delta_rule(
                qkv, gate, beta, alpha, batch, seq_len, cp_size_headwise=1
            )

        # g/beta from the full (un-CP-sliced) A_log / dt_bias.
        g, beta = self._compute_g_and_beta(self.A_log, self.dt_bias, alpha, beta)

        core_attn_out, _ = self.gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens,
        )

        norm_out = self._apply_gated_norm(core_attn_out, gate)
        norm_out = norm_out.reshape(batch, seq_len, -1).transpose(0, 1).contiguous()  # b s x -> s b x

        # Re-slice this rank's zig-zag shard, then output projection (CP-sharded seq).
        norm_out = gdn_cp_slice(norm_out, cu_seqlens_cpu, cp_size, cp_rank)
        out, out_bias = self.out_proj(norm_out)
        return out, out_bias

    GatedDeltaNet.forward = _dcp_gdn_forward
    GatedDeltaNet._dcp_patched = True


def enable_forward_pre_hook(model_chunks: Sequence[DDP]) -> None:
    """Enable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to enable hooks on.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks: Sequence[DDP], param_sync: bool = True) -> None:
    """Disable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to disable hooks on.
        param_sync (bool): Whether to synchronize parameters when disabling.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    """Unwrap common DDP/precision wrappers to inspect model-local flags."""
    seen: set[int] = set()
    while hasattr(module, "module") and id(module) not in seen:
        seen.add(id(module))
        module = module.module
    return module


def has_conditional_branch_sync(model_chunks: Sequence[DDP]) -> bool:
    """Return whether any model chunk installed a conditional branch sync
    adapter."""
    return any(
        getattr(_unwrap_module(model_chunk), "_relax_conditional_branch_sync_installed", False)
        for model_chunk in model_chunks
    )


def force_param_sync(model_chunks: Sequence[DDP]) -> None:
    """Synchronize distributed-optimizer parameters without changing hook
    state."""
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.start_param_sync(force_sync=True)


@torch.no_grad()
def forward_only(
    f: Callable[..., dict[str, list[torch.Tensor]]],
    args: Namespace,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    store_prefix: str = "",
    per_sample_output: bool = True,
) -> dict[str, list[torch.Tensor]]:
    """Run forward passes only and collect non-loss outputs (e.g., logprobs).

    The model is put into evaluation mode, a forward-only pipeline pass is
    executed, and relevant outputs are aggregated and returned.

    Args:
        f (Callable[..., dict[str, list[torch.Tensor]]]): Post-forward callback used to
            compute and package outputs to collect. This should accept a logits
            tensor as its first positional argument and additional keyword-only
            arguments; see ``get_log_probs_and_entropy``/``get_values`` in
            ``megatron_utils.loss`` for examples. It will be partially applied
            so that the callable returned from the internal forward step only
            requires the logits tensor.
        args (Namespace): Runtime arguments.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding batches for inference.
        num_microbatches (Sequence[int]): Number of microbatches per rollout step.
        store_prefix (str): Prefix to prepend to stored output keys.
        per_sample_output (bool): Whether the callback returns one tensor per
            sample. Dynamic CP reconstructs and reorders only per-sample
            outputs; per-microbatch aggregates remain CP-local and are reduced
            by their caller.

    Returns:
        dict[str, list[torch.Tensor]]: Aggregated outputs keyed by ``store_prefix + key``.
    """

    # reset data iterator
    for iterator in data_iterator:
        iterator.reset()

    config = get_model_config(model[0])

    def forward_step(
        data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
        """Forward step used by Megatron's pipeline engine.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
            Output tensor(s) and a callable that computes and packages results
            to be collected by the engine.
        """

        assert not return_schedule_plan, "forward_only step should never return schedule plan"

        # Get the batch.
        is_vl_model = getattr(args, "is_vl_model", False)
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "loss_masks",
                "multimodal_train_inputs",
                "total_lengths",
                "response_lengths",
                "max_seq_lens",
                "classification_labels",
                "sample_weights",
            ],
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
            is_vl_model,
        )
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = batch["packed_seq_params"]
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]

        # VL model with text-only batch: is_vl_model=True but no
        # multimodal_train_inputs in batch — keep mm_kwargs empty so bridge
        # takes the image_grid_thw=None branch.
        mm_kwargs = batch.get("multimodal_train_inputs") or {}
        has_mm_inputs = batch.get("multimodal_train_inputs", None) is not None
        needs_unsplit = is_vl_model or has_mm_inputs or getattr(args, "uses_unsplit_forward", False)

        # Bridge Qwen3VLModel.forward (VL or text-only Qwen3.6) does CP+SP
        # splitting internally, so pass unsplit tokens.
        if needs_unsplit and "unsplit_tokens" in batch:
            forward_input_ids = batch["unsplit_tokens"]
            forward_packed_seq_params = None
        else:
            forward_input_ids = tokens
            forward_packed_seq_params = packed_seq_params

        # thd bridge+CP: bridge needs per-sample attention_mask + matching thd
        # packed_seq_params (align_size = tp*cp*2).  loss_mask is None because
        # labels=None means GPTModel won't run internal loss; Relax's loss is
        # computed externally from full_loss_masks.
        if needs_unsplit and "vlm_packed_seq_params" in batch:
            forward_attention_mask = batch["unsplit_attention_mask"]
            forward_packed_seq_params = batch["vlm_packed_seq_params"]
            forward_loss_mask = None
        else:
            forward_attention_mask = None
            forward_loss_mask = batch["full_loss_masks"]

        # Dynamic CP: the VL bridge (Qwen3VLModel.forward) reads pg_collection.cp
        # directly, so point it at this mb's dynamic CP sub-group for the forward and
        # restore afterwards. Swap every mb (incl. cp==1 -> size-1 group): new's base
        # pg is the static max_cp group, so cp==1 mbs must be switched down too, else
        # the bridge would CP-split cp==1 data by max_cp. (TE / GDN read cp from
        # packed_seq_params.)
        _orig_cp_group = None
        dynamic_cp_size = batch.get("dynamic_cp_size")
        if dynamic_cp_size is not None and needs_unsplit:
            inner = model
            while hasattr(inner, "module"):
                inner = inner.module
            _orig_cp_group = inner.pg_collection.cp
            inner.pg_collection.cp = mpu.get_dynamic_data_context_parallel_groups(group_size=dynamic_cp_size)

        forward_kwargs = {
            "input_ids": forward_input_ids,
            "position_ids": None,
            "attention_mask": forward_attention_mask,
            "labels": None,
            "packed_seq_params": forward_packed_seq_params,
            "loss_mask": forward_loss_mask,
            **mm_kwargs,
        }
        output_tensor = model(**forward_kwargs)

        if _orig_cp_group is not None:
            inner.pg_collection.cp = _orig_cp_group

        f_partial = partial(
            f,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=args.use_rollout_entropy,
            max_seq_lens=batch.get("max_seq_lens", None),
            padded_total_lengths=batch.get("padded_total_lengths", None),
            loss_masks=batch.get("loss_masks", None),
            dynamic_cp_size=batch.get("dynamic_cp_size", None),
            dynamic_cp_rank=batch.get("dynamic_cp_rank", None),
            classification_labels=batch.get("classification_labels", None),
            sample_weights=batch.get("sample_weights", None),
        )

        if getattr(args, "dynamic_context_parallel", False) and per_sample_output:
            # Carry per-mb dynamic-CP metadata on the result so
            # dynamic_cp_merge_output can reconstruct the full mb (CP
            # all-gather + cross-sub-group gather + reorder) before the write-back.
            dcp_meta = {
                "dynamic_cp_size": batch["dynamic_cp_size"],
                "dynamic_cp_rank": batch["dynamic_cp_rank"],
                "total_lengths": total_lengths,
                "response_lengths": response_lengths,
                "padded_total_lengths": batch.get("padded_total_lengths"),
                "partition_order": batch.get("_dcp_partition_order"),
            }

            def _inject_meta(logits, *, _orig=f_partial, _meta=dcp_meta):
                empty, res = _orig(logits)
                res["_dcp_meta"] = _meta
                return empty, res

            return output_tensor, _inject_meta

        return output_tensor, f_partial

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.custom_megatron_before_log_prob_hook_path:
        from relax.utils.misc import load_function

        custom_before_log_prob_hook = load_function(args.custom_megatron_before_log_prob_hook_path)
        custom_before_log_prob_hook(args, model, store_prefix)

    forward_backward_func = get_forward_backward_func()
    # Don't care about timing during evaluation
    config.timers = None
    forward_data_store = []
    num_steps_per_rollout = len(num_microbatches)
    for step_id in range(num_steps_per_rollout):
        forward_data_store += forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
        )

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    rollout_data = {}
    # Store the results on the last stage
    if mpu.is_pipeline_last_stage():
        if getattr(args, "dynamic_context_parallel", False) and per_sample_output:
            # Reconstruct each mb's full sample set (CP all-gather + cross-sub-group
            # gather + reorder) so the per-sample outputs line up with the full
            # micro_batch_indices used by the write-back below.
            from .cp_utils import dynamic_cp_merge_output

            forward_data_store = dynamic_cp_merge_output(
                forward_data_store,
                static_cp_size=mpu.get_context_parallel_world_size(),
                static_cp_rank=mpu.get_context_parallel_rank(),
            )

        keys = forward_data_store[0].keys()
        for key in keys:
            values = []
            for value in forward_data_store:
                assert isinstance(value[key], list)
                values += value[key]

            if args.use_dynamic_batch_size and per_sample_output:
                # TODO: This is ugly... Find a better way to make the data have the same order.
                # TODO: move this out of the loop.
                iterator = data_iterator[0]
                micro_batch_indices = getattr(iterator, "micro_batch_indices", None)
                if micro_batch_indices is not None and len(forward_data_store) == len(micro_batch_indices):
                    dummy_offsets = getattr(iterator, "dummy_micro_batch_offsets", set())
                    origin_values = [None] * len(iterator.rollout_data["total_lengths"])
                    can_restore_order = True
                    for offset, value in enumerate(forward_data_store):
                        if offset in dummy_offsets:
                            continue
                        micro_values = value[key]
                        indices = micro_batch_indices[offset]
                        if len(micro_values) != len(indices):
                            can_restore_order = False
                            break
                        for micro_value, origin_index in zip(micro_values, indices, strict=False):
                            origin_values[origin_index] = micro_value
                    if can_restore_order:
                        if any(value is None for value in origin_values):
                            raise RuntimeError("Dynamic forward output did not cover every local row.")
                        values = origin_values
                else:
                    origin_indices = sum(data_iterator[0].micro_batch_indices, [])
                    # Per-sample callbacks (log_probs/values) emit one tensor per
                    # sample, so values aligns with origin_indices and we can
                    # restore the pre-balance order. Per-microbatch callbacks
                    # (e.g. compute_sft_eval_step) emit one aggregate per
                    # microbatch — len(values) == num_microbatches, not
                    # num_samples — and have no per-sample order to restore.
                    if len(values) == len(origin_indices):
                        origin_values = [None] * len(values)
                        for value, origin_index in zip(values, origin_indices, strict=False):
                            origin_values[origin_index] = value
                        values = origin_values
            rollout_data[f"{store_prefix}{key}"] = values
    return rollout_data


def train_one_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_microbatches: int,
    step_global_batch_size: int,
) -> tuple[dict[str, float], float]:
    """Execute a single pipeline-parallel training step.

    Runs forward/backward over ``num_microbatches``, applies optimizer step and
    one scheduler step when gradients are valid.

    Args:
        args (Namespace): Runtime arguments.
        rollout_id (int): Rollout identifier.
        step_id (int): Step index within the current rollout.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        num_microbatches (int): Number of microbatches to process.
        step_global_batch_size (int): Number of distinct sample indices in this optimizer step.

    Returns:
        tuple[dict[str, float], float]: Reduced loss dictionary (last stage only)
        and gradient norm for logging.
    """
    args = get_args()

    # Trajectory-replay capture: open a per-step accumulator (no-op unless
    # capture is enabled and this step is selected).
    capture_hooks.begin_step_for(args, rollout_id, step_id)

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        from relax.utils.misc import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    main_loss_has_tokens = False

    def forward_step(
        data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[
        torch.Tensor,
        Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]],
    ]:
        """Forward step used by Megatron's pipeline engine during training.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]]]:
            Output tensor(s) and the loss function, which returns
            (loss, num_elems, {"keys": list[str], "values": torch.Tensor}).
        """

        nonlocal main_loss_has_tokens
        is_vl_model = getattr(args, "is_vl_model", False)
        mtp_only = getattr(args, "mtp_only_training", False)
        # Get the batch.
        with timer(f"get_data_batch_{uuid.uuid4().hex[:8]}", keep=False):
            _opd_keys: list[str] = []
            if args.use_opd:
                consume_opd_train_data(_opd_keys, args)
            batch = get_batch(
                data_iterator,
                [
                    "tokens",
                    "multimodal_train_inputs",
                    "packed_seq_params",
                    "total_lengths",
                    "response_lengths",
                    "loss_masks",
                    "classification_labels",
                    "log_probs",
                    "ref_log_probs",
                    "values",
                    "advantages",
                    "returns",
                    "rollout_log_probs",
                    "max_seq_lens",
                    "sample_index_mask_sums",
                    *_opd_keys,
                ],
                args.data_pad_size_multiplier,
                args.qkv_format,
                args.allgather_cp,
                is_vl_model,
            )
        if args.ci_test and args.enable_mtp_training:
            main_loss_has_tokens = main_loss_has_tokens or _main_loss_has_tokens(batch)

        if Envs.ENABLE_ROUTING_REPLAY:
            old_stage = os.environ["ROUTING_REPLAY_STAGE"]
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

        # set in the SFT branch below; left as None for return_schedule_plan or
        # the non-SFT path so the original loss_function is used.
        lm_head_forward = None
        if return_schedule_plan:
            assert not args.enable_mtp_training, "MTP training should not be enabled when using combined 1f1b"
            # build_schedule_plan path doesn't go through model() so the
            # _bypass_output_layer wrapping can't apply. The combined-1f1b ×
            # chunked-logits incompatibility is enforced as a hard assert in
            # arguments.py.slime_validate_args, so bypass mode is guaranteed
            # False here — no runtime fallback or advisory needed.
            output_tensor = model.build_schedule_plan(
                input_ids=batch["tokens"],
                position_ids=None,
                attention_mask=None,
                labels=None,
                packed_seq_params=batch["packed_seq_params"],
                loss_mask=batch["full_loss_masks"],
            )
        else:
            has_mm_inputs = batch.get("multimodal_train_inputs", None) is not None
            needs_unsplit = is_vl_model or has_mm_inputs or getattr(args, "uses_unsplit_forward", False)
            use_unsplit = needs_unsplit and "unsplit_tokens" in batch

            forward_kwargs = {
                "input_ids": batch["unsplit_tokens"] if use_unsplit else batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": None if use_unsplit else batch["packed_seq_params"],
                "loss_mask": batch["full_loss_masks"],
            }

            # thd VL+CP: bridge needs per-sample attention_mask + matching thd
            # packed_seq_params (align_size = tp*cp*2).  loss_mask is None
            # because labels=None means GPTModel won't run internal loss;
            # Relax's loss is computed externally from full_loss_masks.
            if needs_unsplit and "vlm_packed_seq_params" in batch:
                forward_kwargs["attention_mask"] = batch["unsplit_attention_mask"]
                forward_kwargs["packed_seq_params"] = batch["vlm_packed_seq_params"]
                forward_kwargs["loss_mask"] = None

            _attach_mtp_forward_kwargs(args, batch, forward_kwargs)

            # VL model with text-only batch has is_vl_model=True but no
            # multimodal_train_inputs in batch — no kwargs to splice in.
            mm_inputs = batch.get("multimodal_train_inputs")
            if is_vl_model and mm_inputs:
                forward_kwargs.update(mm_inputs)

            # Dynamic CP: point pg_collection.cp at this mb's dynamic CP sub-group for
            # the VL bridge forward. Set every mb (incl. size 1) to avoid a stale group
            # leaking from a previous mb; restored once after forward_backward (below).
            # GDN / TE read cp from packed_seq_params, so 1F1B interleaving stays correct.
            dynamic_cp_size = batch.get("dynamic_cp_size")
            if dynamic_cp_size is not None and needs_unsplit:
                inner = model
                while hasattr(inner, "module"):
                    inner = inner.module
                inner.pg_collection.cp = mpu.get_dynamic_data_context_parallel_groups(group_size=dynamic_cp_size)

            # SFT chunking defers the main lm_head into the loss. MTP-only
            # bypasses that main head entirely while preserving the preceding
            # MTP head calls that build the auxiliary-loss autograd graph.
            if should_bypass_main_output_layer(args):
                mtp_output_layer_calls = (
                    int(getattr(args, "mtp_num_layers", 0) or 0) if getattr(args, "enable_mtp_training", False) else 0
                )
                with _bypass_output_layer(
                    model,
                    mtp_output_layer_calls=mtp_output_layer_calls,
                    gather_passthrough=not mtp_only,
                ) as lm_head_forward:
                    output_tensor = model(**forward_kwargs)
            else:
                output_tensor = model(**forward_kwargs)

        if Envs.ENABLE_ROUTING_REPLAY:
            os.environ["ROUTING_REPLAY_STAGE"] = old_stage

        # Always dispatch via loss_function. MTP-only consumes the bypassed
        # hidden states directly; chunked SFT uses lm_head_forward to compute
        # the regular language loss in bounded chunks.
        return output_tensor, partial(loss_function, args, batch, num_microbatches, lm_head_forward=lm_head_forward)

    # Dynamic CP: forward_step overwrites pg_collection.cp per micro-batch (VL bridge);
    # save the original static CP group here and restore after forward+backward.
    _dcp_orig_cp_group = None
    if getattr(args, "dynamic_context_parallel", False):
        inner = model[0]
        while hasattr(inner, "module"):
            inner = inner.module
        _dcp_orig_cp_group = inner.pg_collection.cp

    # Forward pass.
    use_streaming = (
        getattr(args, "use_dynamic_batch_size", False)
        and getattr(args, "fully_async", False)
        and mpu.get_virtual_pipeline_model_parallel_world_size() is None
        and isinstance(data_iterator[0], StreamingTQIterator)
    )
    if use_streaming:
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        if pp_size <= 1:
            from relax.backends.megatron.streaming_schedules import (
                streaming_forward_backward_no_pipelining,
            )

            forward_backward_func = streaming_forward_backward_no_pipelining
        else:
            from relax.backends.megatron.streaming_schedules import (
                streaming_forward_backward_pipelining_without_interleaving,
            )

            forward_backward_func = streaming_forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    if _dcp_orig_cp_group is not None:
        inner.pg_collection.cp = _dcp_orig_cp_group

    # CI check: verify only MTP parameters have non-zero gradients when truncation happens
    # This check must happen before optimizer.step() as gradients may be modified during step
    if args.ci_test and args.enable_mtp_training:
        from relax.backends.megatron.ci_utils import check_mtp_only_grad

        check_mtp_only_grad(model, step_id, require_non_mtp_zero=not main_loss_has_tokens)

    critic_value_head_snapshot = None
    if args.ci_test and rollout_id == 0 and step_id == 0 and getattr(model[0], "role", "actor") == "critic":
        from relax.backends.megatron.ci_utils import capture_critic_value_head_update

        critic_value_head_snapshot = capture_critic_value_head_update(model)

    # Update parameters. Single optimizer.step() call handles prepare_grads, unscale,
    # clip, and inner step in one shot — avoids the double prepare_grads/unscale and
    # double grad_scaler.update that the previous external prepare_grads() flow caused.
    # In fp16 with dynamic loss scaling, step() returns (False, None, None) on overflow.
    valid_step = True
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

    if not getattr(args, "check_for_nan_in_loss_and_grad", True):
        # fp16 with dynamic loss scaling auto-disables this flag (see Megatron arguments.py).
        # Detect overflow via the documented (False, None, None) return signature.
        found_inf_flag = not update_successful and grad_norm is None and num_zeros_in_grad is None
        if found_inf_flag:
            valid_step = False
            current_scale = optimizer.get_loss_scale().item()
            logger.warning(
                "Inf found in gradients (step_id=%d, loss_scale=%s), skipping parameter "
                "update (dynamic loss scaling will reduce scale)",
                step_id,
                current_scale,
            )
        else:
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    if valid_step:
        # Update learning rate.
        assert update_successful
        opt_param_scheduler.step(increment=step_global_batch_size)
    else:
        grad_norm = float("nan")

    if critic_value_head_snapshot is not None:
        from relax.backends.megatron.ci_utils import assert_critic_value_head_updated

        assert_critic_value_head_updated(
            critic_value_head_snapshot,
            update_successful=bool(update_successful),
            learning_rates=[param_group.get("lr", 0.0) for param_group in optimizer.param_groups],
        )

    maybe_verify_critic_value_head_movement(model, optimizer, bool(update_successful))

    # release grad
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        keys = losses_reduced[0]["keys"]
        values = None
        for x in losses_reduced:
            if values is None:
                values = x["values"]
            else:
                values += x["values"]
        assert len(keys) + 1 == values.numel()
        torch.distributed.all_reduce(values, group=mpu.get_data_parallel_group(with_context_parallel=True))

        loss_reduced = {}
        values = values.tolist()
        is_sequence_classification = getattr(args, "task_type", "causal_lm") == "seq_cls"
        num_samples_or_tokens = (
            values[0] if args.calculate_per_token_loss or is_sequence_classification else step_global_batch_size
        )
        if num_samples_or_tokens == 0:
            # Degenerate / zero-signal batch: every micro-batch across all DP ranks contributed
            # zero loss-normalising units. This happens when the whole batch carries no learning
            # signal — e.g. every GRPO group has identical reward (zero advantage), all tokens are
            # masked out by TIS rejection sampling, or the batch is entirely dummy-padded. The
            # optimizer step above already ran as a (near) no-op on the zero gradients, so this only
            # affects the *reported* metrics: return zeros instead of dividing by zero and killing
            # the run. This is a stopgap robustness guard; the proper upstream fix (drop zero-variance
            # groups + oversample so such batches never form) is tracked in
            # docs/draft/degenerate_batch_zerodivision.md.
            logger.warning(
                "train_one_step: num_samples_or_tokens == 0 (degenerate/zero-signal batch); "
                "reporting zero loss for this step instead of dividing by zero. keys=%s",
                keys,
            )
            for key in keys:
                loss_reduced[key] = 0.0
            capture_hooks.end_step_for()
            return loss_reduced, grad_norm
        for key, value in zip(keys, values[1:], strict=False):
            # Per-token and sequence-classification metrics use the all-reduced
            # effective count. RL sample-mean metrics use the step's logical GBS.
            loss_reduced[key] = value / num_samples_or_tokens
        capture_hooks.end_step_for()
        return loss_reduced, grad_norm
    capture_hooks.end_step_for()
    return {}, grad_norm


def should_disable_forward_pre_hook(args: Namespace) -> bool:
    """Block forward pre-hook for certain configurations."""
    return args.use_distributed_optimizer and args.overlap_param_gather


def train(
    rollout_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
) -> None:
    """Run training over a rollout consisting of multiple steps.

    The model is switched to train mode, training hooks are configured, and
    ``train_one_step`` is invoked for each step in the rollout.

    Args:
        rollout_id (int): Rollout identifier.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        num_microbatches (Sequence[int]): Microbatches per step in the rollout.
    """
    args = get_args()
    is_data_iterator = isinstance(data_iterator[0], DataIterator)
    if is_data_iterator:
        for iterator in data_iterator:
            iterator.reset()
    else:
        data_iter = []
        for iterator in data_iterator:
            data_iter.append(iter(iterator))  # type: ignore
        data_iterator = data_iter
    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Setup some training config params.
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    # train() is invoked once per rollout in Relax (vs. once per run upstream),
    # so guard the sync-func setup to be idempotent — re-assigning would trip
    # Megatron's "no_sync_func must be None" assert on rollout 1+.
    if isinstance(model[0], DDP) and args.overlap_grad_reduce and config.no_sync_func is None:
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather and config.param_sync_func is None:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    pre_hook_enabled = False
    param_sync_func = None
    keep_forward_pre_hook_disabled = False
    if args.reset_optimizer_states:
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            logger.info("Reset optimizer states")
        for chained_optimizer in optimizer.chained_optimizers:
            for group in chained_optimizer.optimizer.param_groups:
                if "step" in group:
                    group["step"] = 0
            for state in chained_optimizer.optimizer.state.values():
                if "step" in state:
                    if isinstance(state["step"], torch.Tensor):
                        state["step"].zero_()
                    else:
                        state["step"] = 0
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False
        keep_forward_pre_hook_disabled = has_conditional_branch_sync(model)
        if keep_forward_pre_hook_disabled:
            logger.info(
                "Keeping forward pre-hook disabled for all training sub-steps because "
                "conditional multimodal branch sync is installed."
            )

    num_steps_per_rollout = len(num_microbatches)
    use_step_iterators = (
        not is_data_iterator and len(data_iterator) > 1 and isinstance(data_iterator[0], StreamingTQIterator)
    )
    if use_step_iterators and len(data_iterator) != num_steps_per_rollout:
        raise ValueError(
            f"streaming data_iterator length ({len(data_iterator)}) must match "
            f"num_steps_per_rollout ({num_steps_per_rollout})"
        )
    first_iterator = data_iterator[0]
    if is_data_iterator:
        global_batch_sizes = first_iterator.rollout_data.get(ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY, None)
    elif use_step_iterators:
        global_batch_sizes = [
            iterator.window_quota if iterator.window_quota is not None else args.global_batch_size
            for iterator in data_iterator
        ]
    else:
        global_batch_sizes = None
    if global_batch_sizes is None:
        global_batch_sizes = [args.global_batch_size for _ in range(num_steps_per_rollout)]
    global_batch_sizes = [int(size) for size in global_batch_sizes]
    if len(global_batch_sizes) != num_steps_per_rollout:
        raise RuntimeError(
            f"global_batch_sizes length {len(global_batch_sizes)} does not match "
            f"num_microbatches length {num_steps_per_rollout}."
        )

    # Run training iterations till done.
    for step_id in range(num_steps_per_rollout):
        step_data_iterator = [data_iterator[step_id]] if use_step_iterators else data_iterator
        # Run training step.
        with timer(f"train_micro_batch_{step_id}", keep=False):
            loss_dict, grad_norm = train_one_step(
                args,
                rollout_id,
                step_id,
                step_data_iterator,
                model,
                optimizer,
                opt_param_scheduler,
                num_microbatches[step_id],
                global_batch_sizes[step_id],
            )
        if keep_forward_pre_hook_disabled:
            force_param_sync(model)

        if step_id == 0:
            # Enable forward pre-hook after training step has successfully run. All subsequent
            # forward passes will use the forward pre-hook / `param_sync_func` in
            # `forward_backward_func`.
            if should_disable_forward_pre_hook(args) and not keep_forward_pre_hook_disabled:
                enable_forward_pre_hook(model)
                config.param_sync_func = param_sync_func
                pre_hook_enabled = True

        if args.enable_mtp_training:
            from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

            mtp_loss_scale = 1 / num_microbatches[step_id]
            tracker = MTPLossLoggingHelper.tracker
            if "values" in tracker:
                values = tracker["values"]
                if tracker.get("reduce_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))
                if tracker.get("avg_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG)
                # here we assume only one mtp layer
                mtp_losses = (tracker["values"] * mtp_loss_scale).item()
                MTPLossLoggingHelper.clean_loss_in_tracker()

                # CI check: verify MTP loss is within expected bounds
                if args.ci_test:
                    from relax.backends.megatron.ci_utils import check_mtp_loss

                    check_mtp_loss(mtp_losses)

        # per train step log.
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            role = getattr(model[0], "role", "actor")
            role_tag = "" if role == "actor" else f"{role}-"
            log_dict = {
                f"train/{role_tag}{key}": val.mean().item() if isinstance(val, torch.Tensor) else val
                for key, val in loss_dict.items()
            }

            log_dict[f"train/{role_tag}grad_norm"] = (
                grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            if args.enable_mtp_training:
                log_dict[f"train/{role_tag}mtp_loss"] = mtp_losses
            log_dict[f"train/{role_tag}global_batch_size"] = global_batch_sizes[step_id]

            for param_group_id, param_group in enumerate(optimizer.param_groups):
                log_dict[f"train/{role_tag}lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)

            log_dict["train/step"] = accumulated_step_id
            num_per_epoch = getattr(args, "num_rollout_per_epoch", None)
            if num_per_epoch:
                log_dict[f"train/{role_tag}cur_epoch"] = (accumulated_step_id + 1) / (
                    num_per_epoch * num_steps_per_rollout
                )
            tracking_utils.log(args, log_dict, step_key="train/step")
            tracking_utils.flush_metrics(args, accumulated_step_id)

            if args.ci_test and not args.ci_disable_kl_checker:
                if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
                    # TODO: figure out why KL is not exactly zero when using PPO loss with KL clipping, and whether this is expected behavior or a bug.
                    assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
                if accumulated_step_id == 0 and "train/kl_loss" in log_dict:
                    assert log_dict["train/kl_loss"] == 0.0, f"{log_dict=}"

            logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")

            if args.ci_save_grad_norm is not None:
                ci_save_grad_norm_path = args.ci_save_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                torch.save(grad_norm, ci_save_grad_norm_path)
            elif args.ci_load_grad_norm is not None:
                ci_load_grad_norm_path = args.ci_load_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                expected_grad_norm = torch.load(ci_load_grad_norm_path)
                assert math.isclose(
                    grad_norm,
                    expected_grad_norm,
                    rel_tol=0.01,
                    abs_tol=0.01,
                ), f"grad norm mismatch: {grad_norm} != {expected_grad_norm}"

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        # NOTE(wuhuan): Sync the latest distributed-optimizer parameters before exporting weights
        # to rollout engines. this is important for --overlap-grad-reduce --overlap-param-gather
        disable_forward_pre_hook(model, param_sync=True)
        enable_forward_pre_hook(model)
    elif keep_forward_pre_hook_disabled:
        # Forward pre-hooks stayed disabled for the whole rollout to preserve a stable
        # branch/parameter-gather order. Per-step sync above has made weights current.
        enable_forward_pre_hook(model)
        config.param_sync_func = param_sync_func


def save(
    iteration: int, model: Sequence[DDP], optimizer: MegatronOptimizer, opt_param_scheduler: OptimizerParamScheduler
) -> None:
    """Persist a training checkpoint safely with forward hooks disabled.

    Args:
        iteration (int): Current global iteration number.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
    """
    args = get_args()
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model)
    save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far=0,
        checkpointing_context=None,
        train_data_iterator=None,
        preprocess_common_state_dict_fn=None,
    )
    if is_lora_enabled(args):
        checkpoint_dir = Path(args.save) / f"iter_{iteration:07d}"
        _save_lora_to_checkpoint(model, str(checkpoint_dir), args)
    if should_disable_forward_pre_hook(args):
        enable_forward_pre_hook(model)


def _install_streaming_fp8_writer(bridge, strategy, block_size):
    """Monkey-patch the bridge source's save_generator with a
    StreamingFP8Writer.

    Returns (writer, restore_fn). Caller MUST invoke restore_fn in a finally
    block so subsequent BF16 exports on the same bridge instance are not
    hijacked.
    """
    from relax.utils.quant_cast.fp8_checkpoint import StreamingFP8Writer

    state = getattr(bridge.hf_pretrained, "state", None)
    source = getattr(state, "source", None) if state is not None else None
    if source is None or not hasattr(source, "key_to_filename_map"):
        raise ValueError("Online FP8 export requires --hf-checkpoint to point to a safetensors-backed HF directory.")

    writer = StreamingFP8Writer(
        source.key_to_filename_map,
        strategy,
        block_size,
        device="cuda",
    )
    original_save_generator = source.save_generator
    source.save_generator = writer.save_generator

    def restore() -> None:
        source.save_generator = original_save_generator

    return writer, restore


def _apply_fp8_quantization_config(config_path, strategy, block_size, modules_to_not_convert):
    """Merge FP8 `quantization_config` into an already-written HF
    config.json."""
    import json

    from relax.utils.quant_cast.fp8 import build_quantization_config

    if not os.path.isfile(config_path):
        return
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = build_quantization_config(strategy, block_size, modules_to_not_convert)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _invoke_save_hf_post_hook(args, hf_path, rollout_id, *, is_lora, force_sync=False):
    """Invoke the user-supplied post-save hook; log-and-swallow exceptions.

    Called on WORLD rank 0 only, after every HF shard + LoRA adapter + FP8
    config patch have hit disk. Sync-return-fast contract: hooks that need to
    do heavy I/O (uploads, RPCs) must enqueue to their own background pool and
    return immediately.

    When ``force_sync=True`` (final save), also invokes the hook module's
    optional module-level ``flush()`` so the container does not exit while
    background work is still in flight. ``flush()`` is best-effort: absent
    attr is a no-op, and exceptions are swallowed.
    """
    hook_path = getattr(args, "save_hf_post_hook_path", None)
    if not hook_path:
        return
    try:
        from relax.utils.misc import load_function

        hook = load_function(hook_path)
        hook(
            args,
            str(hf_path),
            rollout_id,
            dtype=getattr(args, "save_hf_dtype", "bf16"),
            is_lora=is_lora,
        )
    except Exception:
        logger.exception(f"save-hf post-hook {hook_path!r} raised; training continues")

    if force_sync:
        try:
            import importlib

            module_path = hook_path.rpartition(".")[0]
            hook_module = importlib.import_module(module_path)
            flush = getattr(hook_module, "flush", None)
            if callable(flush):
                logger.info(f"save-hf post-hook: draining {module_path}.flush() on final save")
                flush()
        except Exception:
            logger.exception(f"save-hf post-hook flush for {hook_path!r} raised; training continues")


def _resolve_save_hf_path(save_hf: str, rollout_id: int) -> Path:
    """Resolve an HF checkpoint path while preserving legacy templates."""
    has_rollout_id = any(field_name == "rollout_id" for _, field_name, _, _ in string.Formatter().parse(save_hf))
    path = Path(save_hf.format(rollout_id=rollout_id))
    if not has_rollout_id:
        path /= f"iter_{rollout_id:07d}"
    return path


def save_hf_model(args, rollout_id: int, model: Sequence[DDP], *, force_sync: bool = False) -> None:
    """Save Megatron model in HuggingFace format.

    Args:
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        rollout_id (int): Rollout ID for path formatting.
    """
    should_log = not torch.distributed.is_initialized() or (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )

    try:
        from megatron.bridge import AutoBridge

        path = _resolve_save_hf_path(args.save_hf, rollout_id)

        if should_log:
            logger.info(f"Saving model in HuggingFace format to {path}")

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)

        path.mkdir(parents=True, exist_ok=True)

        # strict=True fails whenever the reference declares weights this model structurally
        # never emits: Bridge refuses every shard holding such a key, losing the real
        # tensors that shared it (measured on gemma-4-26B text-mode SFT: 58 of 657
        # language tensors written). Relax for exactly those cases -- an MTP base trained
        # without MTP, or a VL base trained text-only, where only the VL providers declare
        # vision_config. Keep strict=True everywhere else: it is the only export-time
        # guard against a mapping bug silently truncating the checkpoint.
        from relax.utils.hf_export import (
            reconcile_hf_export_index,
            reference_expects_mtp,
            reference_expects_vision,
        )

        model_has_mtp = bool(getattr(args, "mtp_num_layers", 0))
        allow_missing_mtp_keys = reference_expects_mtp(args.hf_checkpoint) and not model_has_mtp
        # Short-circuits: a reference without vision weights can never be missing them.
        allow_missing_vision_keys = reference_expects_vision(args.hf_checkpoint) and not hasattr(
            get_model_config(model[0]), "vision_config"
        )

        strict = not (allow_missing_mtp_keys or allow_missing_vision_keys)

        save_fp8 = getattr(args, "save_hf_dtype", "bf16") == "fp8"
        fp8_writer = None
        restore_save_generator = None
        if save_fp8:
            fp8_writer, restore_save_generator = _install_streaming_fp8_writer(
                bridge,
                args.save_hf_fp8_quant_mode,
                args.save_hf_fp8_block_size,
            )
            # StreamingFP8Writer strict-checks against the source index and cannot express
            # an absent group. Redundant above, kept so the constraint survives edits.
            strict = strict and not (allow_missing_mtp_keys or allow_missing_vision_keys)

        try:
            with patch_megatron_model(model):
                bridge.save_hf_pretrained(
                    model,
                    path=path,
                    strict=strict,
                )
        finally:
            if restore_save_generator is not None:
                restore_save_generator()

        # A non-strict save can leave "ghost" index entries: keys Bridge listed but wrote
        # to no shard (seen for mtp.*, not for vision -- but this is a no-op when there is
        # nothing to fix, so gate on both relaxations). Rebuilds the index from what was
        # written and supplements MTP from the base so the checkpoint stays deployable;
        # vision is left out, a text-only export stays text-only. Bridge writes on WORLD
        # rank 0, so reconcile there. FP8 has its own streaming index and is skipped.
        is_export_writer = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank(group=torch.distributed.group.WORLD) == 0
        )
        if (allow_missing_mtp_keys or allow_missing_vision_keys) and is_export_writer and not save_fp8:
            reconcile_hf_export_index(
                str(path), reference_hf_dir=args.hf_checkpoint, supplement_mtp=allow_missing_mtp_keys
            )

        if save_fp8 and is_export_writer and fp8_writer is not None:
            _apply_fp8_quantization_config(
                str(path / "config.json"),
                args.save_hf_fp8_quant_mode,
                args.save_hf_fp8_block_size,
                fp8_writer.result.modules_to_not_convert,
            )

        if is_lora_enabled(args):
            _save_lora_to_checkpoint(model, str(path), args, bridge=bridge)
        if should_log:
            logger.info(f"Successfully saved HuggingFace model to {path}")

        if is_export_writer:
            _invoke_save_hf_post_hook(args, path, rollout_id, is_lora=is_lora_enabled(args), force_sync=force_sync)
    except Exception as e:
        if should_log:
            logger.error(f"Failed to save HuggingFace format: {e}")


def initialize_model_and_optimizer(
    args: Namespace, role: str = "actor"
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
    """Initialize model(s), optimizer, scheduler, and load from checkpoint.

    Args:
        args (Namespace): Runtime arguments.
        role (str): Logical role of the model (e.g., "actor", "critic").

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
            DDP-wrapped model chunks, optimizer, scheduler, and iteration index.
    """

    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from relax.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        logger.info("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    model[0].role = role
    value_head_param_ids = ()
    classification_head_param_ids = ()
    if role == "critic":
        value_head_param_ids = validate_critic_value_head_registration(model, optimizer)
    if role == "actor" and getattr(args, "task_type", "causal_lm") == "seq_cls":
        classification_head_param_ids = validate_sequence_classification_head_registration(model, optimizer, args)
    clear_memory()
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    if role == "critic":
        release_critic_lm_heads(model)
        loaded_value_head_param_ids = validate_critic_value_head_registration(model, optimizer)
        assert loaded_value_head_param_ids == value_head_param_ids, (
            "critic value head parameter identities changed during checkpoint loading"
        )
        install_critic_value_head_runtime_check(model)
    if role == "actor" and getattr(args, "task_type", "causal_lm") == "seq_cls":
        release_sequence_classification_lm_heads(model)
        loaded_classification_head_param_ids = validate_sequence_classification_head_registration(
            model, optimizer, args
        )
        assert loaded_classification_head_param_ids == classification_head_param_ids, (
            "sequence classification head parameter identities changed during checkpoint loading"
        )
    clear_memory()

    return model, optimizer, opt_param_scheduler, iteration
