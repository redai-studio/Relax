# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""FSDPTurbo actor for colocate synchronous GRPO and SFT.

GRPO synchronizes HF weights to colocated SGLang engines; SFT consumes
TransferQueue batches and eval chunks. Both regimes use HF-native checkpoints.
"""

from __future__ import annotations

import os
import re
import time
import types
from argparse import Namespace
from collections import defaultdict
from typing import Any, Callable

import ray
import torch
import torch.distributed as dist
import transfer_queue as tq

from relax.algorithms.advantages import compute_advantages_and_returns
from relax.backends.fsdp_turbo.checkpoint import load_checkpoint, save_checkpoint
from relax.backends.fsdp_turbo.model_utils import (
    build_microbatches,
    compute_rollout_log_probs,
    grpo_microbatch_loss,
    load_deepseek_v4_model,
    load_hf_model,
    sft_microbatch_loss,
    sft_microbatch_nll_stats,
    sft_step_supervised_tokens,
)
from relax.backends.fsdp_turbo.turbo_config_utils import load_turbo_config, validate_turbo_config
from relax.backends.fsdp_turbo.weight_update import FSDPTurboUpdateWeight
from relax.distributed.ray.train_actor import TrainRayActor
from relax.engine.sft.runtime import (
    is_sft_mode,
    sft_partition_id,
    sft_partition_ids,
    sft_task_name,
    should_run_sft_eval,
)
from relax.utils.async_utils import run
from relax.utils.data.stream_dataloader import (
    get_data_from_transfer_queue,
    post_process_rollout_data,
)
from relax.utils.device import make_current_torch_device
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.timer import Timer
from relax.utils.training.data_fields import build_data_fields
from relax.utils.utils import get_debug_data


logger = get_logger(__name__)


def _summarize_chrome_trace(path: str, top_k: int = 50, max_bytes: int = 512 * 1024 * 1024) -> str:
    """Summarize an exported chrome trace in process.

    Aggregate trace JSON directly because this torch_npu stack exposes no
    ``key_averages``/``events`` API.
    """
    import json
    from collections import Counter

    if os.path.getsize(path) > max_bytes:
        raise RuntimeError(f"trace is {os.path.getsize(path)} bytes (> {max_bytes}); analyze it offline instead")
    with open(path, encoding="utf-8") as f:
        trace = json.load(f)
    cat_tot = Counter()
    agg = defaultdict(lambda: [0, 0, 0])
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "X" or not e.get("dur"):
            continue
        cat = e.get("cat", "?")
        cat_tot[cat] += e["dur"]
        rec = agg[(cat, e.get("name", "?"))]
        rec[0] += e["dur"]
        rec[1] += 1
        rec[2] = max(rec[2], e["dur"])
    lines = ["-- total time by category (s) --"]
    lines.extend(f"  {tot / 1e6:12.2f}  {cat}" for cat, tot in cat_tot.most_common(8))
    lines.append(f"-- top {top_k} events by total time --")
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][0])[:top_k]
    lines.extend(
        f"  {tot / 1e6:10.2f}s {cnt:7d}x avg {tot / cnt / 1e3:9.3f}ms max {mx / 1e3:9.2f}ms [{cat}] {name[:90]}"
        for (cat, name), (tot, cnt, mx) in ranked
    )
    return "\n".join(lines)


def _is_qwen4_exp(hf_config) -> bool:
    """Qwen3.8-Flash-Next (qwen4_exp) detection — plain or composite config."""
    for candidate in (hf_config, getattr(hf_config, "text_config", None)):
        if getattr(candidate, "model_type", "") == "qwen4_exp":
            return True
    return False


def _is_deepseek_v4(hf_config) -> bool:
    """DeepSeek-V4 (deepseek_v4) detection — plain or composite config."""
    for candidate in (hf_config, getattr(hf_config, "text_config", None)):
        if getattr(candidate, "model_type", "") == "deepseek_v4":
            return True
    return False


def _apply_param_freeze(model: torch.nn.Module, args: Namespace) -> None:
    """Apply parameter freeze regexes to HF names before FSDP2 sharding.

    FSDP2 shards parameters individually, but ``requires_grad`` must be settled
    before gradient machinery is built.
    """
    only_train = getattr(args, "only_train_params_name_list", None)
    freeze = getattr(args, "freeze_params_name_list", None)
    if not only_train and not freeze:
        return

    if only_train:
        for name, param in model.named_parameters():
            param.requires_grad = False
            for pattern in only_train:
                if re.search(pattern, name):
                    param.requires_grad = True
                    break
    if freeze:
        for name, param in model.named_parameters():
            for pattern in freeze:
                if re.search(pattern, name):
                    param.requires_grad = False
                    break

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    frozen = [name for name, param in model.named_parameters() if not param.requires_grad]
    logger.info(
        f"[fsdp_turbo] param freeze applied (only-train patterns={only_train or []}, "
        f"freeze patterns={freeze or []}): {len(frozen)} frozen / {len(trainable)} trainable "
        f"params; frozen sample: {frozen[:5]}"
    )


def _pin_fp_state_to_compute_dtype(model: torch.nn.Module, turbo_config, master_dtype: torch.dtype) -> None:
    """Pin floating-point buffers to the FSDP2 compute dtype.

    FSDP2 mixed precision re-casts managed parameters only; buffers retain the
    master dtype. Managed parameters must keep a uniform original dtype, so
    they are not pinned here.
    """
    from fsdp_turbo.utils.dtype import get_dtype

    fsdp_plan = getattr(getattr(turbo_config, "distributed", None), "fsdp_plan", None)
    param_dtype_str = getattr(fsdp_plan, "param_dtype", None) if fsdp_plan is not None else None
    compute_dtype = get_dtype(param_dtype_str) if param_dtype_str else None
    if compute_dtype is None or compute_dtype == master_dtype:
        return

    pinned_buffers = []
    for name, buffer in model.named_buffers():
        if buffer is not None and buffer.is_floating_point() and buffer.dtype != compute_dtype:
            buffer.data = buffer.data.to(compute_dtype)
            pinned_buffers.append(name)

    logger.info(
        f"[fsdp_turbo] master dtype {master_dtype} != compute dtype {compute_dtype}: pinned "
        f"{len(pinned_buffers)} floating-point buffers (sample: {pinned_buffers[:5]}) to {compute_dtype}"
    )


def _pin_ngram_tables_to_bf16(model: torch.nn.Module) -> None:
    """Re-pin CPU-offloaded PLE n-gram tables to bf16 after materialization.

    Shared-memory loading can give non-source ranks the fp32 placeholder dtype;
    these frozen FSDP-ignored tables must not promote activations.
    """
    from fsdp_turbo.models.qwen.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    fixed = []
    for name, module in model.named_modules():
        if not isinstance(module, Qwen4ExpTextNGramEmbedding):
            continue
        weight = module.ngram_embedding.weight
        if weight is not None and weight.is_floating_point() and weight.dtype != torch.bfloat16:
            module.ngram_embedding.weight.data = weight.data.to(torch.bfloat16)
            fixed.append(f"{name} ({weight.dtype} -> torch.bfloat16)")
    if fixed:
        logger.warning(f"[fsdp_turbo] n-gram table dtype re-pinned after materialize: {fixed}")


def _build_qwen4_exp_turbo(args, turbo_config, torch_dtype):
    """Build and wrap Qwen3.8-Flash-Next with fsdp_turbo's consolidated
    modeling.

    Nested parent/child ``apply_modules`` require post-order sharding so
    children are sharded before parents.
    """
    from fsdp_turbo.fsdp_turbo import FSDPTurbo
    from fsdp_turbo.models.qwen.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration
    from fsdp_turbo.utils.model import convert_model_dtype, set_num_hidden_layers
    from transformers import AutoConfig

    model_path = args.hf_checkpoint
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    for cfg in (
        model_config,
        getattr(model_config, "text_config", None),
        getattr(model_config, "vision_config", None),
    ):
        if cfg is not None:
            cfg._attn_implementation = "sdpa"

    # Keep stock SDPA numerics in the vision tower under a dedicated registry name.
    if model_config.vision_config is not None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward as _orig_sdpa_attention
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _ATTN_REGISTRY

        def _vision_attention_local(*args, **kwargs):
            return _orig_sdpa_attention(*args, **kwargs)

        _ATTN_REGISTRY.register("sdpa_vision", _vision_attention_local)
        model_config.vision_config._attn_implementation = "sdpa_vision"

    set_num_hidden_layers(model_config, getattr(turbo_config.model, "num_hidden_layers", None))

    text_cfg = getattr(model_config, "text_config", model_config)
    vision_cfg = getattr(model_config, "vision_config", None)
    text_cfg.gdn_implementation = "triton"
    text_cfg.causal_conv1d_implementation = "triton"
    text_cfg.qsa_implementation = "triton"
    text_cfg.skip_gdn_recompute = True
    text_cfg.skip_flash_attn_recompute = True
    if vision_cfg is not None:
        vision_cfg.skip_flash_attn_recompute = True
    text_cfg.use_grouped_expert_matmul = True
    text_cfg._experts_implementation = "eager"
    model_config._experts_implementation = "eager"
    # Start with an empty PLE table. VP materializes and shards it before wrapping;
    # otherwise materialize frozen CPU weights after wrapping.
    vp_enabled = getattr(turbo_config.distributed, "vp_plan", None) is not None
    text_cfg.ple_embedding_on_cpu = True
    text_cfg._ngram_hf_dir = model_path
    # Allow functional and smoke tests to shrink the PLE table deterministically.
    ngram_vocab_size_base = os.environ.get("NGRAM_VOCAB_SIZE_BASE", "").strip()
    if ngram_vocab_size_base:
        text_cfg.ngram_vocab_size_base = int(ngram_vocab_size_base)
        if text_cfg.ngram_vocab_size_base <= 0:
            raise ValueError(f"Invalid NGRAM_VOCAB_SIZE_BASE='{ngram_vocab_size_base}'.")
    text_cfg.enable_chunk_loss = getattr(turbo_config.memory, "chunk_batch", False)
    text_cfg.enable_dynamic_chunk_loss = False
    text_cfg.enable_ep_balance = False
    text_cfg.max_dup_experts_num = 2
    text_cfg.mtp_num_layers = int(getattr(turbo_config.model, "mtp_num_layers", 0) or 0)

    model = Qwen4ExpForConditionalGeneration.from_pretrained(
        model_path,
        config=model_config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
    )
    convert_model_dtype(model, turbo_config.model.torch_dtype)

    from fsdp_turbo.models.qwen.modeling_qwen4_exp import restore_ngram_integer_buffers

    restore_ngram_integer_buffers(model)

    _pin_fp_state_to_compute_dtype(model, turbo_config, torch_dtype)
    _apply_param_freeze(model, args)

    # Clear the default ParallelState created by from_pretrained so FSDPTurbo uses the real plan.
    from fsdp_turbo.distributed.parallel_state import ParallelState, Singleton

    Singleton._instances.pop(ParallelState, None)

    from fsdp_turbo.distributed.fully_shard_parallel import fully_shard_parallel as _fsp

    _orig_get_fsdp_modules = _fsp.get_fsdp_modules

    def _get_fsdp_modules_post_order(model, fsdp_plan, ignored_modules):
        fsdp_modules = {}
        for name, module in _fsp._post_order_traverse(model):
            for pattern, plan in fsdp_plan.apply_modules.items():
                if _fsp.module_name_match(pattern, name) and name not in ignored_modules:
                    fsdp_modules.setdefault(module, {}).update(plan)
        if not fsdp_modules:
            raise RuntimeError(f"[FSDP2] No module named {list(fsdp_plan.apply_modules.keys())}.")
        return fsdp_modules

    _fsp.get_fsdp_modules = _get_fsdp_modules_post_order
    try:
        if vp_enabled:
            # Materialize the trainable table before VP shards its local slice onto the accelerator.
            from fsdp_turbo.models.qwen.modeling_qwen4_exp import materialize_ngram_embedding_on_cpu

            materialize_ngram_embedding_on_cpu(model, dtype=torch.bfloat16, hf_dir=model_path, freeze=False)
        turbo = FSDPTurbo(config=turbo_config, model=model)
    finally:
        _fsp.get_fsdp_modules = _orig_get_fsdp_modules

    # Without VP, materialize frozen tables after model weights leave CPU memory.
    if not vp_enabled:
        from fsdp_turbo.models.qwen.modeling_qwen4_exp import materialize_ngram_embedding_on_cpu

        materialize_ngram_embedding_on_cpu(turbo, dtype=torch.bfloat16, hf_dir=model_path, freeze=True)
        _pin_ngram_tables_to_bf16(turbo)

    logger.info("[qwen4_exp] model built and wrapped by FSDPTurbo")
    return turbo


_FETCH_SPLIT_MAX_RETRIES = 50


def _concat_rollout_batches(batches: list[dict]) -> dict:
    """Merge per-fetch dicts, concatenating per-sample lists."""
    merged: dict[str, Any] = {}
    for key, first_value in batches[0].items():
        values = [batch[key] for batch in batches]
        if isinstance(first_value, list):
            merged[key] = [item for value in values for item in value]
        else:
            merged[key] = values[-1]
    return merged


def _install_fused_moe_dispatcher(model: torch.nn.Module, turbo_config, hf_config) -> int:
    """Bind the lean fused NPU MoE compute path for ``expert_parallel_size=1``.

    Avoid unnecessary ep>1 dispatch collectives and host syncs while keeping
    expert parameters FSDP-sharded under their original HF names.
    """
    distributed = turbo_config.distributed
    ep_plan = getattr(distributed, "ep_plan", None)
    if ep_plan is None or int(getattr(distributed, "expert_parallel_size", 1) or 1) != 1:
        return 0
    if not ep_plan.apply_modules or ep_plan.dispatcher not in ("fused", "eager"):
        return 0

    try:
        import torch_npu  # noqa: F401 — the fused kernels are NPU-only
        from fsdp_turbo.ops.moe import permute, unpermute
        from fsdp_turbo.ops.npu.grouped_matmul import GroupedMatmul
        from fsdp_turbo.utils.str_match import module_name_match
        from torch.distributed.tensor import DTensor
    except ImportError as exc:
        raise RuntimeError(
            f"distributed.ep_plan.dispatcher={ep_plan.dispatcher!r} needs the fsdp_turbo fused MoE "
            f"kernels (NPU), which are unavailable: {exc}. Remove distributed.ep_plan from the "
            "turbo config to fall back to the transformers reference MoE path (very slow on NPU)."
        ) from exc

    num_experts = 0
    for cfg in (hf_config, getattr(hf_config, "text_config", None)):
        num_experts = int(getattr(cfg, "num_experts", 0) or 0)
        if num_experts:
            break

    def _ep1_experts_forward(self, hidden_states, top_k_index, top_k_weights):
        gate_up = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj

        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])

        # Derive on-device grouped-GEMM row counts from the expert weight stack.
        num_experts = gate_up.shape[0]
        counts = torch.histc(top_k_index, bins=num_experts, min=0, max=num_experts).long()

        permuted, sort_indices = permute(x, top_k_index)
        fc1 = GroupedMatmul.apply(permuted, gate_up, None, counts, 1)
        gate, up = fc1.chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate) * up
        y = GroupedMatmul.apply(act, down, None, counts, 1)
        out = unpermute(y, sort_indices, top_k_weights)
        return out.view(*shape)

    installed = 0
    matched_total = 0
    for name, module in model.named_modules():
        if not any(module_name_match(pattern, name) for pattern in ep_plan.apply_modules):
            continue
        total = int(getattr(module, "num_experts", 0) or 0)
        if not total:
            try:
                total = len(module)
            except TypeError:
                total = 0
        total = total or num_experts
        if not total:
            raise RuntimeError(
                f"Cannot determine the expert count for module {name!r}; set num_experts on the "
                "module/hf config or remove distributed.ep_plan from the turbo config."
            )
        if not hasattr(module, "num_experts"):
            module.num_experts = total
        module.forward = types.MethodType(_ep1_experts_forward, module)
        installed += 1
        matched_total = total

    if installed:
        logger.info(
            f"[fsdp_turbo] lean ep=1 fused MoE bound on {installed} experts modules "
            f"(experts/module={matched_total}): npu token permute -> packed npu_grouped_matmul "
            f"-> silu*up -> npu_grouped_matmul -> npu token unpermute; no A2A dispatch machinery"
        )
    else:
        logger.warning(
            f"[fsdp_turbo] ep_plan.apply_modules matched no modules "
            f"(patterns={ep_plan.apply_modules}); MoE stays on the transformers reference path."
        )
    return installed


class _MegatronParityWarmupCosineScheduler:
    """Megatron-compatible linear-warmup/cosine scheduler.

    Write ``param_group["lr"]`` directly to match Megatron rounding; ``step()``
    runs after ``optimizer.step()``.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, lr_fn: Callable[[int], float]) -> None:
        self.optimizer = optimizer
        self._lr_fn = lr_fn
        self.last_epoch = 0
        self._apply_lr()

    def _apply_lr(self) -> None:
        lr = self._lr_fn(self.last_epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self) -> None:
        self.last_epoch += 1
        self._apply_lr()


class FSDPTurboTrainRayActor(TrainRayActor):
    """Colocate GRPO trainer over an FSDPTurbo-wrapped HF model."""

    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int:
        super().init(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher)
        if role != "actor":
            raise ValueError(f"fsdp_turbo backend only supports the 'actor' role, got {role!r}")
        if with_ref or with_opd_teacher:
            raise ValueError("fsdp_turbo backend does not support reference/teacher models (keep --kl-coef 0).")

        # SFT has no rollout service to install a manager.
        self.rollout_manager = None

        # Heavy optional dependency: imported inside the actor process only.
        from fsdp_turbo.fsdp_turbo import FSDPTurbo
        from transformers import AutoConfig, AutoTokenizer

        turbo_config = load_turbo_config(args.turbo_config)
        self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        # CP is SFT-only because GRPO logprob/advantage handling is not CP-aware;
        # qwen4_exp shards fused sequences natively in its consolidated modeling.
        validate_turbo_config(turbo_config, allow_cp=is_sft_mode(args), native_cp=_is_qwen4_exp(self.hf_config))

        tq.init(args.tq_config)
        self.data_system_client = tq.get_client()

        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

        # Preserve a torch.dtype from the YAML loader; string-mapping it would incorrectly hit the bf16 fallback.
        raw_dtype = turbo_config.model.torch_dtype
        if isinstance(raw_dtype, torch.dtype):
            torch_dtype = raw_dtype
        else:
            dtype_map = {
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
                "fp16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(str(raw_dtype).lower(), torch.bfloat16)
        logger.info(f"[fsdp_turbo] model.torch_dtype={raw_dtype!r} -> master dtype {torch_dtype}")
        if _is_qwen4_exp(self.hf_config):
            self.turbo = _build_qwen4_exp_turbo(args, turbo_config, torch_dtype)
        elif _is_deepseek_v4(self.hf_config):
            model = load_deepseek_v4_model(args.hf_checkpoint, torch_dtype)
            _apply_param_freeze(model, args)
            self.turbo = FSDPTurbo(config=turbo_config, model=model)
        else:
            model = load_hf_model(args.hf_checkpoint, torch_dtype)
            _apply_param_freeze(model, args)
            self.turbo = FSDPTurbo(config=turbo_config, model=model)
        # MTP (Qwen3.8/qwen4_exp multi-token prediction): scaling applied to
        # the MTP loss stashed by the patched forward; None disables it.
        self._mtp_loss_scaling = (
            float(getattr(turbo_config.model, "mtp_loss_scaling_factor", 0.0) or 0.0)
            if int(getattr(turbo_config.model, "mtp_num_layers", 0) or 0) > 0
            else None
        )
        # Enable HF activation checkpointing for training; no_grad eval and logprob forwards are unaffected.
        if hasattr(self.turbo.model, "gradient_checkpointing_enable"):
            self.turbo.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.turbo.train()
        if not _is_qwen4_exp(self.hf_config) and not _is_deepseek_v4(self.hf_config):
            # Install after FSDP wrapping and before the first forward.
            _install_fused_moe_dispatcher(self.turbo.model, turbo_config, self.hf_config)

        from fsdp_turbo.distributed.parallel_state import get_parallel_state
        from fsdp_turbo.quantization.cache import hook_optimizer_step
        from fsdp_turbo.training.factories import OptimizerFactory, SchedulerFactory

        optimizer_config = turbo_config.optimizer
        self.optimizer = OptimizerFactory.create(
            optimizer_config.optimizer_type,
            self.turbo.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            betas=(optimizer_config.adam_beta1, optimizer_config.adam_beta2),
            eps=optimizer_config.adam_epsilon,
        )
        hook_optimizer_step(self.turbo, self.optimizer)
        total_steps = (
            int(turbo_config.run.max_steps) if int(turbo_config.run.max_steps) > 0 else max(int(args.num_rollout), 1)
        )
        self.lr_scheduler = SchedulerFactory.create(
            optimizer_config.lr_scheduler_type,
            self.optimizer,
            T_max=total_steps,
            eta_min=optimizer_config.min_lr,
        )
        # The factory has no warmup, so match Megatron: linear warmup, then cosine over
        # total_steps - warmup_span. Keep warmup_span fractional and step after optimizer.step().
        warmup_span = float(getattr(optimizer_config, "warmup_steps", 0) or 0)
        if warmup_span <= 0:
            warmup_ratio = float(getattr(optimizer_config, "warmup_ratio", 0.0) or 0.0)
            if warmup_ratio > 0:
                warmup_span = warmup_ratio * total_steps
        if warmup_span > 0:
            if optimizer_config.lr_scheduler_type != "CosineAnnealingLR":
                raise ValueError(
                    "[fsdp_turbo] optimizer.warmup_ratio/warmup_steps is set but warmup is only "
                    f"implemented for CosineAnnealingLR, got '{optimizer_config.lr_scheduler_type}'"
                )
            import math

            base_lr = float(optimizer_config.lr)
            min_lr = float(optimizer_config.min_lr)
            decay_span = max(float(total_steps) - warmup_span, 1.0)

            def _warmup_cosine_lr(step_idx: int) -> float:
                if step_idx < warmup_span:
                    # megatron warmup: init_lr + (max_lr - init_lr) * step / warmup_steps.
                    return base_lr * step_idx / warmup_span
                progress = step_idx - warmup_span
                if progress >= decay_span:
                    # Clamp beyond the decay span instead of letting the cosine rise again.
                    return min_lr
                # megatron cosine: min_lr + (max_lr - min_lr) * (1 + cos(pi * progress / decay)) / 2.
                return min_lr + (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress / decay_span)) / 2.0

            self.lr_scheduler = _MegatronParityWarmupCosineScheduler(self.optimizer, _warmup_cosine_lr)
            logger.info(
                f"[fsdp_turbo] lr warmup enabled: {warmup_span:g} step(s) ramping from 0, then cosine decay "
                f"to {optimizer_config.min_lr} over {decay_span:g} steps (megatron WarmupDecayLR parity)"
            )

        self.parallel_state = get_parallel_state()
        self._dp_size = int(self.parallel_state.get_group_size("dp"))
        self._dp_rank = int(self.parallel_state.get_rank("dp"))
        self._fsdp_size = int(self.parallel_state.get_group_size("fsdp"))
        self._tp_size = int(self.parallel_state.get_group_size("tp"))
        self._build_same_data_group()

        self.weight_updater = FSDPTurboUpdateWeight(args, self.turbo)

        from relax.utils.training.flops_counter import FlopsCounter

        self.flops_counter = FlopsCounter(self.hf_config)

        # Initialize the metrics adapter on the primary rank.
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            from relax.utils.tracking_utils import init_tracking

            init_tracking(args, primary=False)

        self.train_parallel_config = {
            "train_backend": "fsdp_turbo",
            "dp_size": self._dp_size,
            "tp_size": self._tp_size,
            "pp_size": 1,
        }

        resumed_rollout_id = 0
        if args.load is not None:
            resumed_rollout_id = load_checkpoint(args, self.turbo, self.optimizer, self.hf_config, self.tokenizer)

        logger.info(
            f"[fsdp_turbo] actor initialized: dp={self._dp_size} fsdp={self._fsdp_size} tp={self._tp_size} "
            f"role={self.role} resume_rollout_id={resumed_rollout_id}"
        )
        # Start inter-step train_wait; _train_actor brackets full-step train and compute-only actor_train.
        Timer().start("train_wait")
        return resumed_rollout_id

    def _build_same_data_group(self) -> None:
        """Build a Gloo group for ranks sharing one logical DP slot.

        The row-major (dp, fsdp, tp) mesh makes each slot contiguous; fetch and
        drain decisions must be unanimous within a slot without coupling
        independent slots.
        """
        block = self._fsdp_size * self._tp_size
        if block <= 1:
            self._same_data_group = get_gloo_group()
            return
        group_ranks = list(range(self._dp_rank * block, (self._dp_rank + 1) * block))
        self._same_data_group = dist.new_group(ranks=group_ranks, backend="gloo")

    # ------------------------------------------------------------------
    # Data plane
    # ------------------------------------------------------------------

    def all_consumed(self, task_name: str, rollout_id: int, partition_id: str | None = None) -> bool:
        """Unanimous drain check over the ranks sharing this DP slot."""
        if partition_id is None:
            partition_id = sft_partition_id(self.args, rollout_id)
        status = run(self.data_system_client.async_check_consumption_status(task_name, partition_id))
        flag = torch.tensor([1 if status else 0], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self._same_data_group)
        return bool(flag.item())

    def _fetch_batch_with_agreement(
        self,
        task_name: str,
        partition_id: str,
        data_fields: list[str],
        batch_size: int,
        batch_index: int,
    ) -> dict | None:
        """Fetch one TQ batch after DP-slot ranks agree on empty versus data.

        Ranks use identical ``dp_rank``/``batch_index`` sampling keys, so a re-
        fetch replays the same samples.
        """
        sampling_config = {"dp_rank": self._dp_rank, "task_name": task_name}
        for _attempt in range(_FETCH_SPLIT_MAX_RETRIES):
            rollout_data, _batch_meta = get_data_from_transfer_queue(
                self.args,
                self.data_system_client,
                data_fields,
                batch_size,
                partition_id,
                task_name,
                sampling_config,
                batch_index,
                broadcast_pp=False,
                per_rank_fetch=True,
                synchronize_per_rank_fetch=False,
            )
            tally = torch.tensor([1, 0] if rollout_data is not None else [0, 1], dtype=torch.int32)
            dist.all_reduce(tally, op=dist.ReduceOp.SUM, group=self._same_data_group)
            got_count, missing_count = tally.tolist()
            if got_count and missing_count:
                time.sleep(0.25)
                continue
            return rollout_data
        raise RuntimeError(
            f"[fsdp_turbo] TQ fetch for partition={partition_id} batch_index={batch_index} did not converge "
            f"across the DP replica after {_FETCH_SPLIT_MAX_RETRIES} attempts."
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, rollout_id: int) -> None:
        if self.args.debug_train_only:
            logger.info(f"[fsdp_turbo] debug train only: rollout_id={rollout_id}")
            rollout_data = get_debug_data(self.args, rollout_id, self.args.global_batch_size, dp_rank=self._dp_rank)
            post_process_rollout_data(self.args, rollout_data)
            self._train_actor(rollout_id, rollout_data)
            return

        task_name = sft_task_name(self.args, component="backend")
        partition_id = sft_partition_id(self.args, rollout_id)
        data_fields = build_data_fields(self.args, consumer="actor")
        batch_size = max(self.args.global_batch_size // self._dp_size, 1)

        logger.info(f"[fsdp_turbo] fetching rollout_id={rollout_id} data (task={task_name}, fields={data_fields})")
        fetched: list[dict] = []
        batch_index = 0
        empty_poll_sleep_s = Envs.RELAX_EMPTY_POLL_SLEEP_MS / 1000.0
        while not self.all_consumed(task_name, rollout_id, partition_id):
            rollout_data = self._fetch_batch_with_agreement(
                task_name, partition_id, data_fields, batch_size, batch_index
            )
            if rollout_data is None:
                if empty_poll_sleep_s > 0:
                    time.sleep(empty_poll_sleep_s)
                continue
            fetched.append(rollout_data)
            batch_index += 1

        if not fetched:
            logger.warning(f"[fsdp_turbo] no training data for rollout_id={rollout_id}; skipping step")
            return
        rollout_data = _concat_rollout_batches(fetched) if len(fetched) > 1 else fetched[0]
        self._train_actor(rollout_id, rollout_data)

    def _train_actor(self, rollout_id: int, rollout_data: dict) -> None:
        # Close stale half-open intervals before starting full-step train so a prior exception cannot
        # leave timer state that masks the next error.
        if "train" in Timer().start_time:
            Timer().end("train")
        if "train_wait" in Timer().start_time:
            Timer().end("train_wait")
        Timer().start("train")

        # actor_train covers only fwd/bwd/optimizer, excluding checkpoint, weight sync, and eval;
        # close it before the perf logger snapshots and resets timers.
        if "actor_train" in Timer().start_time:
            Timer().end("actor_train")
        Timer().start("actor_train")

        device = make_current_torch_device()
        max_tokens = getattr(self.args, "max_tokens_per_gpu", None)
        microbatches = build_microbatches(rollout_data["total_lengths"], max_tokens)

        # Token counts for the perf logger (perf/actor_train_tokens, tok_per_s).
        Timer().seq_lens = [int(length) for length in rollout_data["total_lengths"]]

        metrics: dict[str, float] = defaultdict(float)
        skip_optimizer_step = False
        if is_sft_mode(self.args):
            # Masked next-token NLL; loss_masks mark supervised tokens
            # (response_lengths == total_lengths on the SFT data plane).
            num_local_samples = max(len(rollout_data["total_lengths"]), 1)
            # Optionally match Megatron's token-mean NLL instead of the sample-mean default.
            per_token_loss = bool(getattr(self.args, "calculate_per_token_loss", False))
            step_token_count = sft_step_supervised_tokens(rollout_data) if per_token_loss else None
            if per_token_loss:
                # Match Megatron token-equal weighting under FSDP DP averaging by normalizing with
                # global supervised tokens / data size, not local tokens. All-reduce over the data mesh
                # counts CP/TP replicas once; every rank participates, and zero-token skipping is global.
                data_size = self._dp_size * self._fsdp_size
                if data_size > 1:
                    token_t = torch.tensor(float(step_token_count), dtype=torch.float32, device=device)
                    dist.all_reduce(
                        token_t,
                        op=dist.ReduceOp.SUM,
                        group=self.parallel_state.get_group("data"),
                    )
                    step_token_count = float(token_t.item()) / data_size
                if (step_token_count or 0) <= 0:
                    # A global zero-token step skips optimizer and LR updates;
                    # sample-mean mode retains its zero-grad step behavior.
                    logger.warning(
                        f"[fsdp_turbo] rollout={rollout_id}: zero supervised tokens in the whole "
                        "step; skipping fwd/bwd and optimizer + LR updates"
                    )
                    skip_optimizer_step = True
                    metrics["loss"] = 0.0
            for microbatch in microbatches:
                if skip_optimizer_step:
                    break
                loss, micro_metrics = sft_microbatch_loss(
                    self.turbo,
                    rollout_data,
                    microbatch,
                    device,
                    num_local_samples,
                    self._mtp_loss_scaling,
                    per_token_loss=per_token_loss,
                    step_token_count=step_token_count,
                )
                loss.backward()
                for key, value in micro_metrics.items():
                    if key in ("nll_sum", "token_count"):
                        # Preserve raw sums for the exact step token-mean below.
                        metrics[key] += value
                    else:
                        metrics[key] += value / len(microbatches)
            if per_token_loss and metrics.get("token_count", 0.0) > 0.0:
                metrics["loss"] = metrics.pop("nll_sum") / metrics.pop("token_count")
        else:
            # Recompute old-policy log-probs unless rollout log-probs are trusted.
            if not self.args.use_rollout_logprobs:
                logger.info(
                    f"[fsdp_turbo] rollout={rollout_id}: recomputing old log-probs ({len(microbatches)} microbatches)"
                )
                rollout_data["log_probs"] = compute_rollout_log_probs(self.turbo, rollout_data, microbatches, device)
                logger.info(f"[fsdp_turbo] rollout={rollout_id}: old log-probs done")

            # KL is zero because this backend requires kl_coef == 0.
            kl = [
                torch.zeros(int(length), dtype=torch.float32, device=device)
                for length in rollout_data["response_lengths"]
            ]
            advantages, returns = compute_advantages_and_returns(
                self.args,
                rewards=rollout_data["rewards"],
                kl=kl,
                loss_masks=rollout_data["loss_masks"],
                response_lengths=rollout_data["response_lengths"],
                total_lengths=rollout_data["total_lengths"],
            )
            rollout_data["advantages"] = advantages
            rollout_data["returns"] = returns

            # RELAX_PROFILE_MB=N profiles the first N GRPO microbatches on the primary rank.
            profile_mb_count = int(os.environ.get("RELAX_PROFILE_MB", "0") or 0)
            profile_is_primary = (not dist.is_initialized()) or dist.get_rank() == 0
            for mb_idx, microbatch in enumerate(microbatches):
                use_prof = profile_mb_count and mb_idx < profile_mb_count and profile_is_primary
                if use_prof:
                    from torch_npu import profiler

                    # Level0 avoids the default Level1's per-aten-op events and oversized traces.
                    experimental_config = profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level0)
                    prof = profiler.profile(
                        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                        experimental_config=experimental_config,
                    )
                    prof.__enter__()
                loss, micro_metrics = grpo_microbatch_loss(self.turbo, self.args, rollout_data, microbatch, device)
                loss.backward()
                if use_prof:
                    prof.__exit__(None, None, None)
                    trace_path = f"/tmp/relax_npu_prof_rollout{rollout_id}_mb{mb_idx}.json"
                    prof.export_chrome_trace(trace_path)
                    # Log a kernel summary without requiring offline trace analysis.
                    try:
                        summary = _summarize_chrome_trace(trace_path)
                        logger.info(
                            f"[fsdp_turbo] NPU kernel top summary (rollout={rollout_id} mb={mb_idx}):\n{summary}"
                        )
                    except Exception as e:  # noqa: BLE001 - summary is best-effort
                        logger.warning(f"[fsdp_turbo] kernel summary unavailable: {e}")
                    # Compress the trace in place.
                    try:
                        import gzip
                        import shutil

                        with open(trace_path, "rb") as f_in:
                            with gzip.open(trace_path + ".gz", "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        os.remove(trace_path)
                        trace_path += ".gz"
                    except Exception as e:  # noqa: BLE001 - compression is best-effort
                        logger.warning(f"[fsdp_turbo] trace gzip failed, kept raw json: {e}")
                    logger.info(f"[fsdp_turbo] NPU op-level profile exported to {trace_path}")
                for key, value in micro_metrics.items():
                    metrics[key] += value / len(microbatches)

        if skip_optimizer_step:
            # Clear stale gradients and leave optimizer and LR state untouched.
            self.optimizer.zero_grad()
            metrics["grad_norm"] = 0.0
        else:
            from fsdp_turbo.training.clip_grads import clip_grad_norm

            grad_norm = clip_grad_norm(
                self.turbo,
                max_norm=self.turbo.config.optimizer.clip_grad or None,
                norm_type=self.turbo.config.optimizer.clip_grad_norm_type,
                group=None,
            )
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            metrics["grad_norm"] = float(grad_norm) if grad_norm is not None else 0.0
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])

        # Exclude checkpoint, weight sync, and eval from actor_train.
        Timer().end("actor_train")

        # Log once from global rank 0; FSDP-replicated data needs no reduction.
        is_primary = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_primary:
            from relax.utils import tracking_utils

            log_dict = {f"train/{key}": value for key, value in metrics.items()}
            log_dict["train/global_batch_size"] = float(len(rollout_data["total_lengths"]))
            for param_group_id, param_group in enumerate(self.optimizer.param_groups):
                log_dict[f"train/lr-pg_{param_group_id}"] = float(param_group["lr"])
            log_dict["train/step"] = rollout_id
            num_per_epoch = getattr(self.args, "num_rollout_per_epoch", None)
            if num_per_epoch:
                log_dict["train/cur_epoch"] = (rollout_id + 1) / num_per_epoch
            tracking_utils.log(self.args, log_dict, step_key="train/step")
            tracking_utils.flush_metrics(self.args, rollout_id)
            logger.info(f"step {rollout_id}: {log_dict}")

        is_train_done = (rollout_id + 1) == self.args.num_rollout
        if (
            self.args.save is not None
            and self.args.save_interval is not None
            and ((rollout_id + 1) % self.args.save_interval == 0 or is_train_done)
        ):
            self.save_model(rollout_id, force_sync=is_train_done)

        if self.rollout_manager is not None and not self.args.debug_train_only:
            self.update_weights()

        if is_sft_mode(self.args) and should_run_sft_eval(self.args, rollout_id):
            self._run_sft_eval(rollout_id)

        # Close full-step train before the perf logger resets timers, then reopen train_wait.
        Timer().end("train")
        Timer().start("train_wait")

        self._log_rollout_and_perf(rollout_id, rollout_data, is_primary)

    def _log_rollout_and_perf(self, rollout_id: int, rollout_data: dict, is_primary: bool) -> None:
        """Log rollout statistics and performance timing.

        FSDP-replicated sample lengths need no reduction. Every rank runs the
        perf logger to reset timers; only the primary rank emits logs.
        """
        from relax.utils import tracking_utils
        from relax.utils.metrics.metric_utils import compute_rollout_step
        from relax.utils.training.train_metric_utils import log_perf_data_raw

        total_lengths = [int(length) for length in rollout_data.get("total_lengths") or []]
        if total_lengths and is_primary:
            rollout_log = {
                "total_lengths/max": float(max(total_lengths)),
                "total_lengths/min": float(min(total_lengths)),
                "total_lengths/mean": sum(total_lengths) / len(total_lengths),
            }
            prefixed = {f"rollout/{key}": value for key, value in rollout_log.items()}
            prefixed["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            tracking_utils.log(self.args, prefixed, step_key="rollout/step")
            logger.info(f"rollout {rollout_id}: {rollout_log}")

        log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=is_primary,
            flops_counter=self.flops_counter,
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
        )

    # ------------------------------------------------------------------
    # SFT eval (PPL over the producer's eval chunks)
    # ------------------------------------------------------------------

    def _run_sft_eval(self, rollout_id: int) -> None:
        """Consume SFT eval chunks and log PPL.

        All-reduce sufficient statistics over Gloo, then divide out duplicate
        fetches within each DP slot so supervised tokens are counted once.
        """
        import re

        from relax.engine.sft.eval.ppl import compute_ppl_metrics

        args = self.args
        task_name = "sft_eval"
        data_fields = ["tokens", "loss_masks", "total_lengths", "response_lengths"]
        if getattr(args, "multimodal_keys", None) is not None:
            data_fields.append("multimodal_train_inputs")
        batch_size = max(args.global_batch_size // self._dp_size, 1)
        device = make_current_torch_device()

        # The producer waits for partition removal, not consumption; clear training partitions here
        # because the Actor component would otherwise clear them only after train() returns.
        if dist.get_rank() == 0:
            for partition_id in sft_partition_ids(args, rollout_id):
                run(self.data_system_client.async_clear_partition(partition_id=partition_id))
        dist.barrier(group=get_gloo_group())

        pattern = re.compile(rf"^sft_eval_{rollout_id}_n(\d+)_\d+$")
        n_chunks = 0
        while n_chunks == 0:
            partitions = run(self.data_system_client.async_get_partition_list())
            if partitions:
                for partition in partitions:
                    match = pattern.match(partition)
                    if match:
                        n_chunks = int(match.group(1))
                        break
            if n_chunks == 0:
                time.sleep(1)

        max_tokens = getattr(args, "max_tokens_per_gpu", None)
        local_neg_log_prob = 0.0
        local_num_tokens = 0
        self.turbo.eval()
        try:
            for chunk_idx in range(n_chunks):
                partition_id = f"sft_eval_{rollout_id}_n{n_chunks}_{chunk_idx}"
                # Wait for the chunk so a missing partition cannot end the loop early.
                while True:
                    partitions = run(self.data_system_client.async_get_partition_list())
                    if partitions and partition_id in partitions:
                        break
                    time.sleep(1)
                batch_index = 0
                while not self.all_consumed(task_name, rollout_id, partition_id=partition_id):
                    eval_data = self._fetch_batch_with_agreement(
                        task_name, partition_id, data_fields, batch_size, batch_index
                    )
                    if eval_data is None:
                        time.sleep(0.25)
                        continue
                    batch_index += 1
                    for microbatch in build_microbatches(eval_data["total_lengths"], max_tokens):
                        nll, count = sft_microbatch_nll_stats(self.turbo, eval_data, microbatch, device)
                        local_neg_log_prob += nll
                        local_num_tokens += count
                # TQ does not auto-clean consumed partitions; free this chunk for the producer.
                dist.barrier(group=get_gloo_group())
                if dist.get_rank() == 0:
                    run(self.data_system_client.async_clear_partition(partition_id=partition_id))
        finally:
            self.turbo.train()

        # gloo all_reduce requires CPU tensors (mirrors all_consumed above).
        stats = torch.tensor([local_neg_log_prob, float(local_num_tokens)], dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=get_gloo_group())
        # Divide out identical fetches from fsdp*tp ranks in each DP slot.
        stats /= max(self._fsdp_size * self._tp_size, 1)
        metrics = compute_ppl_metrics(float(stats[0].item()), int(stats[1].item()))
        logger.info(f"[fsdp_turbo] SFT eval @ rollout_id={rollout_id}: {metrics}")

    # ------------------------------------------------------------------
    # Weight sync / checkpointing / lifecycle
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return
        if self.args.offload_rollout and dist.get_rank() == 0:
            # Onload rollout weights before copying buckets because offload drops their NPU pages.
            ray.get(self.rollout_manager.onload_weights.remote())
        dist.barrier(group=get_gloo_group())
        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())
        self.weight_updater.update_weights()
        # Warm KV and graph regions for the next generate round.
        if self.args.offload_rollout and dist.get_rank() == 0:
            ray.get(self.rollout_manager.onload_kv.remote())

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only or self.role != "actor":
            return
        save_checkpoint(self.args, rollout_id, self.turbo, self.optimizer, self.hf_config, self.tokenizer)

    def sleep(self, tags=None) -> None:
        logger.warning(
            "[fsdp_turbo] sleep() is a no-op: sharded train weights stay resident. "
            "Budget --sglang-mem-fraction-static so engine + train sides fit together."
        )

    def wake_up(self, tags=None) -> None:
        # Resident sharded weights need no reload.
        logger.warning("[fsdp_turbo] wake_up() is a no-op: sharded train weights stay resident.")

    def _get_parallel_config(self) -> dict:
        return self.train_parallel_config

    # ------------------------------------------------------------------
    # Unsupported service entry points (colocate GRPO only)
    # ------------------------------------------------------------------

    def train_async(self, rollout_id: int) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support fully-async training")

    def train_hybrid(self, rollout_id: int) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support hybrid training")

    def train_critic(self, rollout_id: int, rollout_data: dict | None = None) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support critic training")

    def compute_ref_log_prob(self, rollout_id: int) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support reference log-probs (keep --kl-coef 0)")

    def compute_actor_log_prob(self, rollout_id: int) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support standalone actor log-probs")

    def update_weights_fully_async(
        self, rollout_id: int, rollout_only: bool = False, actor_fwd_only: bool = False
    ) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support fully-async weight updates")

    def recv_weight_fully_async(self, rollout_id: int) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support fully-async weight updates")
